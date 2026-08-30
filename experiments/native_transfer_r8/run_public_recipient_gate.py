"""Run raw public development controls for a frozen R8 recipient bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from .capability_generator import canonical_json_bytes, generate_rows, public_capabilities
from .native_host import SPECS, FrozenNeuralHost, build_bridge, module_sha256, sha256_file
from .recipient_worker import _random_latent

CONDITIONS = ("BASE", "BEFORE", "AFTER", "ZERO", "RANDOM", "WRONG")


class PublicGateError(RuntimeError):
    """Raised when frozen public-gate inputs are missing or stale."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicGateError(f"expected JSON object: {path}")
    return value


def _evidence(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise PublicGateError(f"stale evidence hash: {label}")


def _clear(bridge: torch.nn.Module) -> None:
    clear = getattr(bridge, "clear_context", None)
    if clear is not None:
        clear()


def run(config_path: Path, campaign_root: Path, output: Path, *, host_key: str) -> dict[str, Any]:
    if output.exists():
        raise PublicGateError(f"immutable public-gate output exists: {output}")
    if (campaign_root / "heldout_reveal.json").exists() or list(
        campaign_root.rglob("*.abipkg")
    ):
        raise PublicGateError("public gate requires held-out reveal and packages absent")
    config = _json(config_path)
    extraction_dir = campaign_root / "pre_reveal/meta_extraction"
    extraction = _json(extraction_dir / "receipt.json")
    _evidence(extraction, "meta-extraction")
    latent_path = extraction_dir / extraction["latents"]["path"]
    if sha256_file(latent_path) != extraction["latents"]["sha256"]:
        raise PublicGateError("meta latent changed")
    tensors = load_file(str(latent_path), device="cpu")
    before = tensors["before"].float()
    after = tensors["development_after"].float()
    count = int(config["splits"]["development_capabilities"])
    capabilities = public_capabilities(
        int(config["splits"]["development_seed"]),
        split="development",
        count=count,
    )
    rows = [
        generate_rows(
            capability,
            split="bridge_development",
            rows=256,
            depths=config["capability_family"]["evaluation_depths"],
            seed=int(config["training"]["seed"]) + 4001 * index,
        )
        for index, capability in enumerate(capabilities)
    ]
    bridge_dir = campaign_root / "pre_reveal/bridges" / host_key
    bridge_receipt = _json(bridge_dir / "receipt.json")
    _evidence(bridge_receipt, "bridge")
    bridge_path = bridge_dir / bridge_receipt["bridge"]["path"]
    if sha256_file(bridge_path) != bridge_receipt["bridge"]["sha256"]:
        raise PublicGateError("frozen bridge changed")
    host = FrozenNeuralHost(SPECS[host_key], device=config["training"]["device"])
    bridge = build_bridge(host, config).to(host.device)
    bridge.load_state_dict(load_file(str(bridge_path), device=str(host.device)), strict=True)
    bridge.freeze()
    if module_sha256(bridge) != bridge_receipt["bridge"]["state_sha256"]:
        raise PublicGateError("loaded bridge state changed")
    raw = []
    batch_size = int(config["training"]["batch_size"])
    for capability_index, (capability, capability_rows) in enumerate(
        zip(capabilities, rows)
    ):
        wrong = after[(capability_index + 1) % count]
        choices: dict[str, torch.Tensor | None] = {
            "BASE": None,
            "BEFORE": before,
            "AFTER": after[capability_index],
            "ZERO": torch.zeros_like(before),
            "RANDOM": _random_latent(after[capability_index], capability.capability_id),
            "WRONG": wrong,
        }
        for condition in CONDITIONS:
            latent = choices[condition]
            if latent is None:
                _clear(bridge)
                prefix = None
            else:
                prefix = bridge(latent.to(host.device)).squeeze(0)
            for start in range(0, len(capability_rows), batch_size):
                batch = capability_rows[start : start + batch_size]
                with torch.inference_mode():
                    logits, _ = host.logits(
                        [str(row["prompt"]) for row in batch], prefix=prefix
                    )
                    predictions = logits.argmax(dim=-1)
                    probabilities = host.canonical_probabilities(logits)
                for row, prediction, probability in zip(
                    batch, predictions, probabilities
                ):
                    raw.append(
                        {
                            "capability_id": capability.capability_id,
                            "row_id": row["row_id"],
                            "prompt_sha256": row["prompt_sha256"],
                            "condition": condition,
                            "prediction_token_id": int(prediction.item()),
                            "canonical_output_probabilities": [
                                float(value) for value in probability.cpu().tolist()
                            ],
                        }
                    )
    bridge.verify_frozen()
    host.verify_frozen()
    output.mkdir(parents=True)
    raw_path = output / "observations.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in raw))
    receipt = {
        "format": "abi-native-transfer-r8-public-recipient-gate/1",
        "config_sha256": sha256_file(config_path),
        "host_key": host_key,
        "architecture_family": host.spec.architecture_family,
        "model_revision": host.spec.revision,
        "bridge_receipt_sha256": sha256_file(bridge_dir / "receipt.json"),
        "bridge_sha256_before": sha256_file(bridge_path),
        "bridge_sha256_after": sha256_file(bridge_path),
        "host_model_state_sha256_before": host.model_state_sha256,
        "host_model_state_sha256_after": host.model_state_sha256,
        "target_token_ids": host.target_token_ids,
        "conditions": list(CONDITIONS),
        "capabilities": count,
        "rows_per_capability_condition": 256,
        "rows": len(raw),
        "observations": {"path": raw_path.name, "sha256": sha256_file(raw_path)},
        "heldout_reveal_present": False,
        "heldout_packages_present": 0,
        "recipient_optimizer_steps": 0,
        "bridge_optimizer_steps_after_freeze": 0,
    }
    receipt["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    (output / "receipt.json").write_bytes(
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", required=True, choices=("qwen2", "pythia", "t5"))
    args = parser.parse_args()
    try:
        value = run(
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
            Path(args.output).resolve(),
            host_key=args.host,
        )
    except (OSError, ValueError, PublicGateError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
