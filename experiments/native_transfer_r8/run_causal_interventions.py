"""Activation-level R8 interventions in frozen recipient models."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file

from .native_host import (
    SPECS,
    CanonicalLatentBridge,
    FrozenNeuralHost,
    NativeHostError,
    canonical_json_bytes,
    module_sha256,
    sha256_file,
)
from .recipient_worker import (
    RecipientWorkerError,
    _disable_network,
    _json,
    _jsonl,
    _load_package,
)


class CausalInterventionError(RuntimeError):
    """Raised when a native activation intervention is not actually applied."""


def _blocks(host: FrozenNeuralHost) -> list[torch.nn.Module]:
    model = host.model
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "encoder", None), "block", None),
    )
    for value in candidates:
        if value is not None and len(value):
            return list(value)
    raise CausalInterventionError(f"cannot identify neural blocks for {host.spec.key}")


def _hidden(output: Any, host: FrozenNeuralHost, layer: int) -> torch.Tensor:
    states = output.encoder_hidden_states if host.spec.encoder_decoder else output.hidden_states
    if states is None or len(states) <= layer + 1:
        raise CausalInterventionError("host did not expose requested hidden state")
    return states[layer + 1].detach()


def _replacement_hook(replacement: torch.Tensor) -> Callable[..., Any]:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            if not output or not isinstance(output[0], torch.Tensor):
                raise CausalInterventionError("block output schema changed")
            return (replacement.to(output[0]), *output[1:])
        if not isinstance(output, torch.Tensor):
            raise CausalInterventionError("block output is not a tensor")
        return replacement.to(output)

    return hook


def _forward_patched(
    host: FrozenNeuralHost,
    prompts: list[str],
    prefix: torch.Tensor,
    *,
    block: torch.nn.Module,
    replacement: torch.Tensor,
) -> torch.Tensor:
    handle = block.register_forward_hook(_replacement_hook(replacement))
    try:
        logits, _ = host.logits(prompts, prefix=prefix)
    finally:
        handle.remove()
    return logits


def _base_aligned_replacement(
    real_state: torch.Tensor,
    base_state: torch.Tensor,
) -> torch.Tensor:
    if real_state.shape[0] != base_state.shape[0] or real_state.shape[2] != base_state.shape[2]:
        raise CausalInterventionError("BASE and package state shapes cannot be aligned")
    prefix_length = real_state.shape[1] - base_state.shape[1]
    if prefix_length <= 0:
        raise CausalInterventionError("package interface did not add neural state")
    replacement = torch.zeros_like(real_state)
    replacement[:, prefix_length:, :] = base_state
    return replacement


def _append(
    rows: list[dict[str, Any]],
    *,
    host: FrozenNeuralHost,
    source_rows: list[dict[str, Any]],
    capability_id: str,
    package_sha256: str,
    condition: str,
    layer: int | None,
    logits: torch.Tensor,
) -> None:
    predictions = logits.argmax(dim=-1)
    probabilities = host.canonical_probabilities(logits)
    for source, prediction, probability in zip(source_rows, predictions, probabilities):
        rows.append(
            {
                "host": host.spec.key,
                "architecture_family": host.spec.architecture_family,
                "capability_id": capability_id,
                "row_id": source["row_id"],
                "prompt_sha256": source["prompt_sha256"],
                "condition": condition,
                "layer": layer,
                "prediction_token_id": int(prediction.item()),
                "canonical_output_probabilities": [
                    float(value) for value in probability.cpu().tolist()
                ],
                "package_sha256": package_sha256,
            }
        )


def run(
    config_path: Path,
    campaign_root: Path,
    source_dir: Path,
    output: Path,
    *,
    host_key: str,
) -> dict[str, Any]:
    if output.exists():
        raise CausalInterventionError(f"immutable causal output exists: {output}")
    config = _json(config_path)
    bridge_dir = campaign_root / "pre_reveal/bridges" / host_key
    bridge_receipt = _json(bridge_dir / "receipt.json")
    bridge_path = bridge_dir / bridge_receipt["bridge"]["path"]
    host = FrozenNeuralHost(SPECS[host_key], device=config["training"]["device"])
    bridge = CanonicalLatentBridge(host).to(host.device)
    bridge.load_state_dict(load_file(str(bridge_path), device=str(host.device)), strict=True)
    bridge.freeze()
    if module_sha256(bridge) != bridge_receipt["bridge"]["state_sha256"]:
        raise CausalInterventionError("causal bridge differs from frozen receipt")
    blocks = _blocks(host)
    layers = sorted({0, len(blocks) // 2, len(blocks) - 1})
    _disable_network()
    all_rows = []
    batch_size = int(config["training"]["batch_size"])
    causal_count = int(config["splits"]["causal_rows_per_capability"])
    started = time.perf_counter()
    package_dirs = sorted((source_dir / "packages").iterdir(), key=lambda path: path.name)
    for package_dir in package_dirs:
        capability_id = package_dir.name
        _, latent = _load_package(package_dir / "after.abipkg")
        package_sha = sha256_file(package_dir / "after.abipkg")
        real_prefix = bridge(latent.to(host.device)).squeeze(0)
        null_prefix = torch.zeros_like(real_prefix)
        source_rows = _jsonl(source_dir / "worker_inputs" / f"{capability_id}.jsonl")[:causal_count]
        for start in range(0, len(source_rows), batch_size):
            batch = source_rows[start : start + batch_size]
            prompts = [str(row["prompt"]) for row in batch]
            with torch.inference_mode():
                base_logits, base_output = host.logits(
                    prompts, prefix=None, output_hidden_states=True
                )
                real_logits, real_output = host.logits(
                    prompts, prefix=real_prefix, output_hidden_states=True
                )
            _append(
                all_rows,
                host=host,
                source_rows=batch,
                capability_id=capability_id,
                package_sha256=package_sha,
                condition="BASE",
                layer=None,
                logits=base_logits,
            )
            _append(
                all_rows,
                host=host,
                source_rows=batch,
                capability_id=capability_id,
                package_sha256=package_sha,
                condition="AFTER",
                layer=None,
                logits=real_logits,
            )
            _append(
                all_rows,
                host=host,
                source_rows=batch,
                capability_id=capability_id,
                package_sha256=package_sha,
                condition="PACKAGE_PATH_ABLATION",
                layer=None,
                logits=base_logits,
            )
            for layer in layers:
                base = _hidden(base_output, host, layer)
                real = _hidden(real_output, host, layer)
                clean = _base_aligned_replacement(real, base)
                with torch.inference_mode():
                    patched = _forward_patched(
                        host,
                        prompts,
                        real_prefix,
                        block=blocks[layer],
                        replacement=clean,
                    )
                    rescued = _forward_patched(
                        host,
                        prompts,
                        null_prefix,
                        block=blocks[layer],
                        replacement=real,
                    )
                    destroyed = _forward_patched(
                        host,
                        prompts,
                        real_prefix,
                        block=blocks[layer],
                        replacement=torch.zeros_like(real),
                    )
                _append(
                    all_rows,
                    host=host,
                    source_rows=batch,
                    capability_id=capability_id,
                    package_sha256=package_sha,
                    condition="CLEAN_STATE_PATCH",
                    layer=layer,
                    logits=patched,
                )
                _append(
                    all_rows,
                    host=host,
                    source_rows=batch,
                    capability_id=capability_id,
                    package_sha256=package_sha,
                    condition="CAPABILITY_STATE_RESCUE",
                    layer=layer,
                    logits=rescued,
                )
                _append(
                    all_rows,
                    host=host,
                    source_rows=batch,
                    capability_id=capability_id,
                    package_sha256=package_sha,
                    condition="DOWNSTREAM_NEURAL_DESTRUCTION",
                    layer=layer,
                    logits=destroyed,
                )
    bridge.verify_frozen()
    host.verify_frozen()
    output.mkdir(parents=True)
    raw_path = output / "observations.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in all_rows))
    manifest = {
        "format": "abi-native-transfer-r8-causal-interventions/1",
        "status": "RAW_CAUSAL_INTERVENTIONS_COMPLETE",
        "host": host_key,
        "architecture_family": host.spec.architecture_family,
        "config_sha256": sha256_file(config_path),
        "bridge_sha256_before": sha256_file(bridge_path),
        "bridge_sha256_after": sha256_file(bridge_path),
        "host_model_state_sha256_before": host.model_state_sha256,
        "host_model_state_sha256_after": host.model_state_sha256,
        "target_token_ids": host.target_token_ids,
        "layers": layers,
        "conditions": [
            "BASE",
            "AFTER",
            "PACKAGE_PATH_ABLATION",
            "CLEAN_STATE_PATCH",
            "CAPABILITY_STATE_RESCUE",
            "DOWNSTREAM_NEURAL_DESTRUCTION",
        ],
        "rows": len(all_rows),
        "observations_sha256": sha256_file(raw_path),
        "recipient_parameters_trainable": 0,
        "recipient_optimizer_steps": 0,
        "bridge_optimizer_steps_after_reveal": 0,
        "teacher_loaded": False,
        "test_labels_available": False,
        "wall_seconds": time.perf_counter() - started,
    }
    manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    (output / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    del host, bridge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", required=True, choices=("qwen2", "pythia", "t5"))
    args = parser.parse_args()
    try:
        value = run(
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
            Path(args.source_dir).resolve(),
            Path(args.output).resolve(),
            host_key=args.host,
        )
    except (CausalInterventionError, RecipientWorkerError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
