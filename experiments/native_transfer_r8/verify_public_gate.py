"""Recompute the R8 public prerequisite verdict from raw recipient rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from .capability_generator import canonical_json_bytes, generate_rows, public_capabilities
from .native_host import sha256_file
from .run_public_recipient_gate import CONDITIONS


class PublicVerificationError(RuntimeError):
    """Raised when public falsification evidence cannot be recomputed."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicVerificationError(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise PublicVerificationError(f"expected JSON object: {path}")
    return value


def _evidence(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise PublicVerificationError(f"stale evidence hash: {label}")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_bytes().splitlines()
    if not lines:
        raise PublicVerificationError(f"missing raw rows: {path}")
    rows = [json.loads(line) for line in lines]
    if any(not isinstance(row, dict) for row in rows):
        raise PublicVerificationError("raw row schema changed")
    if path.read_bytes() != b"".join(canonical_json_bytes(row) for row in rows):
        raise PublicVerificationError("raw rows are not canonical")
    return rows


def _bootstrap(values: list[int], *, seed: int, replicates: int) -> dict[str, float]:
    generator = random.Random(seed)
    estimates = sorted(
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    )
    return {
        "point": sum(values) / len(values),
        "lower_95": estimates[math.floor(0.025 * replicates)],
        "upper_95": estimates[min(replicates - 1, math.ceil(0.975 * replicates) - 1)],
    }


def verify(config_path: Path, campaign_root: Path, gate_dir: Path) -> dict[str, Any]:
    config = _json(config_path)
    if (campaign_root / "heldout_reveal.json").exists() or list(
        campaign_root.rglob("*.abipkg")
    ):
        raise PublicVerificationError("held-out material contaminated the public gate")
    source_dir = campaign_root / "pre_reveal/source_public"
    source = _json(source_dir / "receipt.json")
    extraction_dir = campaign_root / "pre_reveal/meta_extraction"
    extraction = _json(extraction_dir / "receipt.json")
    bridge_dir = campaign_root / "pre_reveal/bridges/pythia"
    bridge = _json(bridge_dir / "receipt.json")
    run = _json(gate_dir / "receipt.json")
    for label, value in (
        ("source", source),
        ("extraction", extraction),
        ("bridge", bridge),
        ("public-run", run),
    ):
        _evidence(value, label)
        if value.get("config_sha256") != sha256_file(config_path):
            raise PublicVerificationError(f"config binding changed: {label}")
    source_artifact = source["states"]
    if sha256_file(source_dir / source_artifact["path"]) != source_artifact["sha256"]:
        raise PublicVerificationError("source state artifact changed")
    latent_path = extraction_dir / extraction["latents"]["path"]
    if sha256_file(latent_path) != extraction["latents"]["sha256"]:
        raise PublicVerificationError("extracted latent changed")
    bridge_path = bridge_dir / bridge["bridge"]["path"]
    if (
        sha256_file(bridge_path) != bridge["bridge"]["sha256"]
        or run.get("bridge_sha256_before") != sha256_file(bridge_path)
        or run.get("bridge_sha256_after") != sha256_file(bridge_path)
        or run.get("bridge_receipt_sha256") != sha256_file(bridge_dir / "receipt.json")
    ):
        raise PublicVerificationError("bridge identity changed")
    if (
        run.get("recipient_optimizer_steps") != 0
        or run.get("bridge_optimizer_steps_after_freeze") != 0
        or run.get("heldout_reveal_present") is not False
        or run.get("heldout_packages_present") != 0
        or tuple(run.get("conditions", ())) != CONDITIONS
    ):
        raise PublicVerificationError("public freeze boundary changed")

    split = config["splits"]
    meta = public_capabilities(
        int(split["meta_seed"]), split="meta_train", count=int(split["meta_train_capabilities"])
    )
    development = public_capabilities(
        int(split["development_seed"]),
        split="development",
        count=int(split["development_capabilities"]),
    )
    tensors = load_file(str(latent_path), device="cpu")
    expected_meta = torch.tensor(
        [
            [[(source_state + capability.offsets[op]) % 8 for source_state in range(8)] for op in range(3)]
            for capability in meta
        ]
    )
    expected_development = torch.tensor(
        [
            [[(source_state + capability.offsets[op]) % 8 for source_state in range(8)] for op in range(3)]
            for capability in development
        ]
    )
    extraction_meta_accuracy = float(
        (tensors["meta_after"].argmax(dim=-1) == expected_meta).float().mean()
    )
    extraction_development_accuracy = float(
        (tensors["development_after"].argmax(dim=-1) == expected_development)
        .float()
        .mean()
    )

    raw_path = gate_dir / run["observations"]["path"]
    if sha256_file(raw_path) != run["observations"]["sha256"]:
        raise PublicVerificationError("raw observation hash changed")
    rows = _jsonl(raw_path)
    expected_count = len(development) * 256 * len(CONDITIONS)
    if len(rows) != expected_count or run.get("rows") != expected_count:
        raise PublicVerificationError("raw observation depth changed")
    token_ids = [int(value) for value in run["target_token_ids"]]
    if len(token_ids) != 8 or len(set(token_ids)) != 8:
        raise PublicVerificationError("target token mapping changed")
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {}
    labels: dict[tuple[str, str], int] = {}
    for index, capability in enumerate(development):
        expected_rows = generate_rows(
            capability,
            split="bridge_development",
            rows=256,
            depths=config["capability_family"]["evaluation_depths"],
            seed=int(config["training"]["seed"]) + 4001 * index,
        )
        for row in expected_rows:
            labels[(capability.capability_id, str(row["row_id"]))] = int(row["answer"])
    for row in rows:
        key = (
            str(row.get("capability_id")),
            str(row.get("condition")),
            str(row.get("row_id")),
        )
        if key in keyed or key[1] not in CONDITIONS:
            raise PublicVerificationError("duplicate or unknown raw condition row")
        label_key = (key[0], key[2])
        if label_key not in labels:
            raise PublicVerificationError("raw row does not match regenerated evaluator")
        probabilities = row.get("canonical_output_probabilities")
        if (
            not isinstance(probabilities, list)
            or len(probabilities) != 8
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in probabilities)
            or abs(sum(float(value) for value in probabilities) - 1.0) > 1e-4
        ):
            raise PublicVerificationError("raw probability row changed")
        keyed[key] = row
    metrics: dict[str, dict[str, float]] = {}
    paired = []
    for capability in development:
        capability_metrics = {}
        for condition in CONDITIONS:
            condition_rows = [
                row
                for (capability_id, row_condition, _), row in keyed.items()
                if capability_id == capability.capability_id and row_condition == condition
            ]
            if len(condition_rows) != 256:
                raise PublicVerificationError("condition row set is incomplete")
            correct = [
                int(
                    int(row["prediction_token_id"])
                    == token_ids[labels[(capability.capability_id, str(row["row_id"]))]]
                )
                for row in condition_rows
            ]
            capability_metrics[condition] = sum(correct) / len(correct)
        metrics[capability.capability_id] = capability_metrics
        after_by_id = {
            row_id: row
            for (capability_id, condition, row_id), row in keyed.items()
            if capability_id == capability.capability_id and condition == "AFTER"
        }
        base_by_id = {
            row_id: row
            for (capability_id, condition, row_id), row in keyed.items()
            if capability_id == capability.capability_id and condition == "BASE"
        }
        for row_id in sorted(after_by_id):
            target = token_ids[labels[(capability.capability_id, row_id)]]
            paired.append(
                int(int(after_by_id[row_id]["prediction_token_id"]) == target)
                - int(int(base_by_id[row_id]["prediction_token_id"]) == target)
            )
    bootstrap = _bootstrap(
        paired,
        seed=int(config["training"]["seed"]) + 99001,
        replicates=int(config["gates"]["bootstrap_replicates"]),
    )
    gains = [value["AFTER"] - value["BASE"] for value in metrics.values()]
    native_public_gate = all(
        gain >= float(config["gates"]["recipient_gain_minimum"]) for gain in gains
    ) and bootstrap["lower_95"] > 0
    source_assessments = source.get("assessments", [])
    source_summary_bound = (
        len(source_assessments)
        == int(split["meta_train_capabilities"]) + int(split["development_capabilities"])
        and all(row.get("after", {}).get("rows") == 256 for row in source_assessments)
    )
    source_raw_recomputable = False
    result = {
        "format": "abi-native-transfer-r8-public-falsification/1",
        "status": "FAIL_CLOSED_LEVEL_0_PUBLIC_PREREQUISITE_FAILED",
        "exact_question_answer": "NO",
        "verdict_level": 0,
        "scope": "R8 v10 registered architecture; public prerequisite only; held-out reveal was never opened",
        "metrics": metrics,
        "after_minus_base_bootstrap": bootstrap,
        "minimum_public_capability_gain": min(gains),
        "maximum_public_capability_gain": max(gains),
        "extraction_meta_atomic_accuracy": extraction_meta_accuracy,
        "extraction_development_atomic_accuracy": extraction_development_accuracy,
        "gates": {
            "source_summary_hash_bound": source_summary_bound,
            "source_raw_rows_recomputable": source_raw_recomputable,
            "canonical_extraction_exact": extraction_meta_accuracy == 1.0
            and extraction_development_accuracy == 1.0,
            "pythia_public_native_transfer": native_public_gate,
            "heldout_remained_unrevealed": True,
            "three_recipient_families": False,
            "neural_causality": False,
            "physical_isolation": False,
            "matched_baselines": False,
        },
        "trusted_scientific_booleans_consumed": 0,
        "decision": "Stop before held-out reveal and do not spend Qwen/T5 compute: the smallest recipient failed every public development capability.",
        "claim_boundary": "R7 is unchanged. R8 does not establish native neural capability transfer, recipient independence, teacher extraction, or superiority to LoRA/distillation.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        print(json.dumps({"status": "FAIL_CLOSED", "error": f"immutable output exists: {output}"}, indent=2))
        return 2
    try:
        value = verify(
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
            Path(args.gate_dir).resolve(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    except (OSError, ValueError, PublicVerificationError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
