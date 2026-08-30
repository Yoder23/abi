"""Run preregistered unrelated digit tasks with and without each R8 package."""

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
from .recipient_worker import RecipientWorkerError, _disable_network, _json, _load_package


class NonInterferenceError(RuntimeError):
    """Raised when unrelated-task evidence cannot be executed immutably."""


def unrelated_tasks() -> list[dict[str, Any]]:
    rows = []
    for value in range(8):
        prompts = (
            (f"Return the digit {value} exactly. Answer =", value, "copy"),
            (f"The digit after {value} modulo 8 is", (value + 1) % 8, "successor"),
            (f"The digit before {value} modulo 8 is", (value - 1) % 8, "predecessor"),
        )
        for prompt, answer, family in prompts:
            task_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            rows.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "answer": answer,
                    "family": family,
                }
            )
    for left in range(8):
        for right in range(5):
            prompt = f"Compute {left} plus {right} modulo 8. Answer ="
            rows.append(
                {
                    "task_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "answer": (left + right) % 8,
                    "family": "addition",
                }
            )
    if len(rows) != 64 or len({row["task_id"] for row in rows}) != 64:
        raise NonInterferenceError("unrelated task inventory changed")
    return rows


def _append(
    output: list[dict[str, Any]],
    *,
    host: FrozenNeuralHost,
    capability_id: str,
    package_sha256: str,
    condition: str,
    tasks: list[dict[str, Any]],
    logits: torch.Tensor,
) -> None:
    predicted = logits.argmax(dim=-1)
    probabilities = host.canonical_probabilities(logits)
    for task, token, probability in zip(tasks, predicted, probabilities):
        output.append(
            {
                "host": host.spec.key,
                "capability_id": capability_id,
                "task_id": task["task_id"],
                "prompt_sha256": task["prompt_sha256"],
                "task_family": task["family"],
                "condition": condition,
                "prediction_token_id": int(token.item()),
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
        raise NonInterferenceError(f"immutable non-interference output exists: {output}")
    config = _json(config_path)
    bridge_dir = campaign_root / "pre_reveal/bridges" / host_key
    bridge_receipt = _json(bridge_dir / "receipt.json")
    bridge_path = bridge_dir / bridge_receipt["bridge"]["path"]
    host = FrozenNeuralHost(SPECS[host_key], device=config["training"]["device"])
    bridge = CanonicalLatentBridge(host).to(host.device)
    bridge.load_state_dict(load_file(str(bridge_path), device=str(host.device)), strict=True)
    bridge.freeze()
    if module_sha256(bridge) != bridge_receipt["bridge"]["state_sha256"]:
        raise NonInterferenceError("bridge state differs from pre-reveal freeze")
    tasks = unrelated_tasks()
    public_tasks = [
        {key: row[key] for key in ("task_id", "prompt", "prompt_sha256", "family")}
        for row in tasks
    ]
    _disable_network()
    rows_out = []
    batch_size = int(config["training"]["batch_size"])
    started = time.perf_counter()
    for package_dir in sorted((source_dir / "packages").iterdir(), key=lambda path: path.name):
        capability_id = package_dir.name
        _, latent = _load_package(package_dir / "after.abipkg")
        package_sha = sha256_file(package_dir / "after.abipkg")
        prefix = bridge(latent.to(host.device)).squeeze(0)
        for start in range(0, len(public_tasks), batch_size):
            batch = public_tasks[start : start + batch_size]
            prompts = [str(row["prompt"]) for row in batch]
            with torch.inference_mode():
                base_logits, _ = host.logits(prompts, prefix=None)
                after_logits, _ = host.logits(prompts, prefix=prefix)
            _append(
                rows_out,
                host=host,
                capability_id=capability_id,
                package_sha256=package_sha,
                condition="BASE",
                tasks=batch,
                logits=base_logits,
            )
            _append(
                rows_out,
                host=host,
                capability_id=capability_id,
                package_sha256=package_sha,
                condition="AFTER",
                tasks=batch,
                logits=after_logits,
            )
    bridge.verify_frozen()
    host.verify_frozen()
    output.mkdir(parents=True)
    raw_path = output / "observations.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows_out))
    manifest = {
        "format": "abi-native-transfer-r8-noninterference/1",
        "host": host_key,
        "architecture_family": host.spec.architecture_family,
        "config_sha256": sha256_file(config_path),
        "bridge_sha256_before": sha256_file(bridge_path),
        "bridge_sha256_after": sha256_file(bridge_path),
        "host_model_state_sha256_before": host.model_state_sha256,
        "host_model_state_sha256_after": host.model_state_sha256,
        "target_token_ids": host.target_token_ids,
        "task_inventory_sha256": hashlib.sha256(canonical_json_bytes(tasks)).hexdigest(),
        "task_rows": len(tasks),
        "observations_sha256": sha256_file(raw_path),
        "rows": len(rows_out),
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
    except (NonInterferenceError, RecipientWorkerError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
