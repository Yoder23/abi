"""Label-free R8 worker for independently packaged capability composition."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

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


class CompositionWorkerError(RuntimeError):
    """Raised when a composition execution input is invalid."""


def _execute(
    host: FrozenNeuralHost,
    rows: list[dict[str, Any]],
    prefix: torch.Tensor | None,
    *,
    host_key: str,
    condition: str,
    package_hashes: list[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        with torch.inference_mode():
            logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=prefix)
            predicted = logits.argmax(dim=-1)
            probabilities = host.canonical_probabilities(logits)
        for row, token, probability in zip(batch, predicted, probabilities):
            output.append(
                {
                    "host": host_key,
                    "condition": condition,
                    "row_id": row["row_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "prediction_token_id": int(token.item()),
                    "canonical_output_probabilities": [
                        float(value) for value in probability.cpu().tolist()
                    ],
                    "package_sha256s": package_hashes,
                }
            )
    return output


def run(
    config_path: Path,
    campaign_root: Path,
    source_dir: Path,
    inputs_dir: Path,
    output: Path,
    *,
    host_key: str,
) -> dict[str, Any]:
    if output.exists():
        raise CompositionWorkerError(f"immutable composition output exists: {output}")
    config = _json(config_path)
    pair = _json(inputs_dir / "pair.json")
    first_id, second_id = (str(value) for value in pair["capability_ids"])
    first_rows = _jsonl(inputs_dir / "first.jsonl")
    second_rows = _jsonl(inputs_dir / "second.jsonl")
    cross_rows = _jsonl(inputs_dir / "cross.jsonl")
    first_path = source_dir / "packages" / first_id / "after.abipkg"
    second_path = source_dir / "packages" / second_id / "after.abipkg"
    _, first_latent = _load_package(first_path)
    _, second_latent = _load_package(second_path)
    first_sha, second_sha = sha256_file(first_path), sha256_file(second_path)
    bridge_dir = campaign_root / "pre_reveal/bridges" / host_key
    bridge_receipt = _json(bridge_dir / "receipt.json")
    bridge_path = bridge_dir / bridge_receipt["bridge"]["path"]
    host = FrozenNeuralHost(SPECS[host_key], device=config["training"]["device"])
    bridge = CanonicalLatentBridge(host).to(host.device)
    bridge.load_state_dict(load_file(str(bridge_path), device=str(host.device)), strict=True)
    bridge.freeze()
    if module_sha256(bridge) != bridge_receipt["bridge"]["state_sha256"]:
        raise CompositionWorkerError("composition bridge differs from pre-reveal freeze")
    first_prefix = bridge(first_latent.to(host.device)).squeeze(0)
    second_prefix = bridge(second_latent.to(host.device)).squeeze(0)
    combined_prefix = torch.cat((first_prefix, second_prefix), dim=0)
    _disable_network()
    batch_size = int(config["training"]["batch_size"])
    started = time.perf_counter()
    observations = []
    executions = (
        ("BASE_FIRST", first_rows, None, []),
        ("FIRST_ONLY", first_rows, first_prefix, [first_sha]),
        ("COMBINED_ON_FIRST", first_rows, combined_prefix, [first_sha, second_sha]),
        ("REMOVED_FIRST", first_rows, None, []),
        ("BASE_SECOND", second_rows, None, []),
        ("SECOND_ONLY", second_rows, second_prefix, [second_sha]),
        ("COMBINED_ON_SECOND", second_rows, combined_prefix, [first_sha, second_sha]),
        ("BASE_CROSS", cross_rows, None, []),
        ("FIRST_ONLY_CROSS", cross_rows, first_prefix, [first_sha]),
        ("SECOND_ONLY_CROSS", cross_rows, second_prefix, [second_sha]),
        ("COMBINED_CROSS", cross_rows, combined_prefix, [first_sha, second_sha]),
    )
    for condition, rows, prefix, hashes in executions:
        observations.extend(
            _execute(
                host,
                rows,
                prefix,
                host_key=host_key,
                condition=condition,
                package_hashes=hashes,
                batch_size=batch_size,
            )
        )
    bridge.verify_frozen()
    host.verify_frozen()
    output.mkdir(parents=True)
    raw_path = output / "observations.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in observations))
    manifest = {
        "format": "abi-native-transfer-r8-composition-worker/1",
        "host": host_key,
        "architecture_family": host.spec.architecture_family,
        "config_sha256": sha256_file(config_path),
        "pair_sha256": sha256_file(inputs_dir / "pair.json"),
        "worker_input_hashes": {
            name: sha256_file(inputs_dir / f"{name}.jsonl")
            for name in ("first", "second", "cross")
        },
        "packages": {"first": first_sha, "second": second_sha},
        "target_token_ids": host.target_token_ids,
        "bridge_sha256_before": sha256_file(bridge_path),
        "bridge_sha256_after": sha256_file(bridge_path),
        "host_model_state_sha256_before": host.model_state_sha256,
        "host_model_state_sha256_after": host.model_state_sha256,
        "conditions": [row[0] for row in executions],
        "observations_sha256": sha256_file(raw_path),
        "rows": len(observations),
        "wall_seconds": time.perf_counter() - started,
        "physical_filesystem_isolation": False,
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
    parser.add_argument("--inputs-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", required=True, choices=("qwen2", "pythia", "t5"))
    args = parser.parse_args()
    try:
        value = run(
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
            Path(args.source_dir).resolve(),
            Path(args.inputs_dir).resolve(),
            Path(args.output).resolve(),
            host_key=args.host,
        )
    except (CompositionWorkerError, RecipientWorkerError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
