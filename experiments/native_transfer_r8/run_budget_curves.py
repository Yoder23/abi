"""Run preregistered nested canonical-latent information budgets."""

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


class BudgetCurveError(RuntimeError):
    """Raised when a registered nested budget cannot be executed."""


def _budget_latent(before: torch.Tensor, after: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction <= 0 or fraction > 1:
        raise BudgetCurveError("information budget fraction is outside (0, 1]")
    rows = before.reshape(-1, 8).clone()
    learned = after.reshape(-1, 8)
    retained = round(fraction * rows.shape[0])
    rows[:retained] = learned[:retained]
    return rows.reshape_as(before)


def run(
    config_path: Path,
    campaign_root: Path,
    source_dir: Path,
    output: Path,
    *,
    host_key: str,
) -> dict[str, Any]:
    if output.exists():
        raise BudgetCurveError(f"immutable budget output exists: {output}")
    config = _json(config_path)
    budgets = [float(value) for value in config["information_budgets"]]
    if budgets != sorted(set(budgets)) or budgets[-1] != 1.0:
        raise BudgetCurveError("nested information budgets changed")
    bridge_dir = campaign_root / "pre_reveal/bridges" / host_key
    bridge_receipt = _json(bridge_dir / "receipt.json")
    bridge_path = bridge_dir / bridge_receipt["bridge"]["path"]
    host = FrozenNeuralHost(SPECS[host_key], device=config["training"]["device"])
    bridge = CanonicalLatentBridge(host).to(host.device)
    bridge.load_state_dict(load_file(str(bridge_path), device=str(host.device)), strict=True)
    bridge.freeze()
    if module_sha256(bridge) != bridge_receipt["bridge"]["state_sha256"]:
        raise BudgetCurveError("budget bridge differs from pre-reveal freeze")
    _disable_network()
    observations = []
    batch_size = int(config["training"]["batch_size"])
    started = time.perf_counter()
    package_hashes = {}
    for package_dir in sorted((source_dir / "packages").iterdir(), key=lambda path: path.name):
        capability_id = package_dir.name
        _, before = _load_package(package_dir / "before.abipkg")
        _, after = _load_package(package_dir / "after.abipkg")
        after_sha = sha256_file(package_dir / "after.abipkg")
        before_sha = sha256_file(package_dir / "before.abipkg")
        package_hashes[capability_id] = {"before": before_sha, "after": after_sha}
        rows = _jsonl(source_dir / "worker_inputs" / f"{capability_id}.jsonl")
        for fraction in budgets:
            latent = _budget_latent(before, after, fraction)
            prefix = bridge(latent.to(host.device)).squeeze(0)
            condition = f"BUDGET_{int(round(100 * fraction)):03d}"
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                with torch.inference_mode():
                    logits, _ = host.logits(
                        [str(row["prompt"]) for row in batch], prefix=prefix
                    )
                    predicted = logits.argmax(dim=-1)
                    probabilities = host.canonical_probabilities(logits)
                for row, token, probability in zip(batch, predicted, probabilities):
                    observations.append(
                        {
                            "host": host_key,
                            "capability_id": capability_id,
                            "row_id": row["row_id"],
                            "condition": condition,
                            "retained_fraction": fraction,
                            "retained_delta_float32_bytes": round(fraction * 24) * 8 * 4,
                            "prediction_token_id": int(token.item()),
                            "canonical_output_probabilities": [
                                float(value) for value in probability.cpu().tolist()
                            ],
                            "before_package_sha256": before_sha,
                            "after_package_sha256": after_sha,
                        }
                    )
    bridge.verify_frozen()
    host.verify_frozen()
    output.mkdir(parents=True)
    raw_path = output / "observations.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in observations))
    manifest = {
        "format": "abi-native-transfer-r8-information-budget-curves/1",
        "host": host_key,
        "architecture_family": host.spec.architecture_family,
        "config_sha256": sha256_file(config_path),
        "budgets": budgets,
        "package_hashes": package_hashes,
        "target_token_ids": host.target_token_ids,
        "bridge_sha256_before": sha256_file(bridge_path),
        "bridge_sha256_after": sha256_file(bridge_path),
        "host_model_state_sha256_before": host.model_state_sha256,
        "host_model_state_sha256_after": host.model_state_sha256,
        "observations_sha256": sha256_file(raw_path),
        "rows": len(observations),
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
    except (BudgetCurveError, RecipientWorkerError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
