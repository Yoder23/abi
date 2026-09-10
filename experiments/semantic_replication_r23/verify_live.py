"""Strict recomputation of R23 fresh live evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)
from experiments.generative_transfer_r21.protocol import SEEDS

from .acquire import _jsonl, validate_source
from .binding import load_config, load_seed_reveal
from .live_binding import load_live_config
from .protocol import hidden_rows, semantic_score


def verify(config_path: Path, result_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    result_dir = result_dir.resolve()
    live = load_live_config(root, config_path)
    r23_path = (root / str(live["r23_config"]["path"])).resolve()
    reveal_path = (root / str(live["seed_reveal"]["path"])).resolve()
    source_run = (root / str(live["source_receipt"]["path"])).parent.resolve()
    config = load_config(root, r23_path)
    seed = load_seed_reveal(config, reveal_path)
    expected = hidden_rows(seed)
    expected_by_id = {row["record_id"]: row for row in expected}
    source_rows = validate_source(root, r23_path, reveal_path, source_run)
    source_by_id = {row["record_id"]: row for row in source_rows}
    engine = json_object(root / str(live["candidate_engine"]["path"]))
    stored_rows = _jsonl(root / str(live["candidate_observations"]["path"]))
    stored = {
        (row["method"], int(row["seed"]), row["record_id"]): row
        for row in stored_rows
    }
    receipt = json_object(result_dir / "receipt.json")
    live_path = result_dir / "live_observations.jsonl"
    cpu_path = result_dir / "cpu_observations.jsonl"
    lifecycle_path = result_dir / "lifecycle_observations.jsonl"
    if (
        receipt.get("format") != "abi-r23-fresh-live-verification/1"
        or receipt.get("status") != "PASS_FRESH_LIVE_SEMANTIC_REPLICATION"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("evidence_sha256") != selfless_evidence_hash(receipt)
        or receipt.get("candidate_engine_sha256")
        != sha256_file(root / str(live["candidate_engine"]["path"]))
        or receipt.get("candidate_observations_sha256")
        != sha256_file(root / str(live["candidate_observations"]["path"]))
        or receipt.get("source_software_loaded") is not False
        or receipt.get("source_parameters_in_packages") != 0
        or receipt.get("receiver_training_steps") != 0
        or not isinstance(receipt.get("execution_pid"), int)
    ):
        raise R14Error("R23 live receipt failed")
    for name, path, count in (
        ("live_observations", live_path, 1080),
        ("cpu_observations", cpu_path, 54),
        ("lifecycle_observations", lifecycle_path, 24),
    ):
        binding = receipt.get(name, {})
        if (
            binding.get("rows") != count
            or not path.is_file()
            or binding.get("sha256") != sha256_file(path)
        ):
            raise R14Error(f"R23 live {name} binding failed")
    live_rows = _jsonl(live_path)
    expected_keys = {
        (method, seed_value, row["record_id"])
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
        for seed_value in SEEDS
        for row in expected
    }
    keys = {
        (row["method"], int(row["seed"]), row["record_id"]) for row in live_rows
    }
    if keys != expected_keys:
        raise R14Error("R23 fresh live matrix changed")
    for row in live_rows:
        key = (row["method"], int(row["seed"]), row["record_id"])
        reference = expected_by_id[row["record_id"]]
        teacher = source_by_id[row["record_id"]]["teacher_output"]
        score = semantic_score(reference, row["output"], teacher)
        if row["output"] != stored[key]["output"]:
            raise R14Error("R23 live output differs from frozen candidate")
        if any(row.get(name) != value for name, value in score.items()):
            raise R14Error("R23 live score changed")
        if row.get("output_sha256") != hashlib.sha256(row["output"].encode()).hexdigest():
            raise R14Error("R23 live output hash changed")
    aggregates = base_run._aggregate(live_rows)
    if (
        aggregates != engine["aggregates"]
        or receipt.get("recomputed_aggregates") != aggregates
        or receipt.get("recomputed_gates") != engine["gates"]
        or not all(receipt["recomputed_gates"].values())
    ):
        raise R14Error("R23 live aggregate or gate changed")
    cpu_rows = _jsonl(cpu_path)
    expected_cpu = {
        (method, seed_value, task)
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
        for seed_value in SEEDS
        for task in ("prose", "summary", "email", "bullets", "clarification", "abstention")
    }
    if {(row["method"], row["seed"], row["task"]) for row in cpu_rows} != expected_cpu:
        raise R14Error("R23 CPU matrix changed")
    for row in cpu_rows:
        candidate = stored[(row["method"], row["seed"], row["record_id"])]["output"]
        if row["output_sha256"] != hashlib.sha256(candidate.encode()).hexdigest():
            raise R14Error("R23 CPU output changed")
    lifecycle = _jsonl(lifecycle_path)
    package_ids = {
        package["cake_id"]
        for seed_systems in engine["systems"].values()
        for system in seed_systems.values()
        for package in system["packages"]
    }
    if {row["cake_id"] for row in lifecycle} != package_ids or not all(
        all(
            row[key]
            for key in (
                "removed",
                "absent_rejected",
                "restored_archive_exact",
                "restored_output_exact",
                "targeted_corruption_rejected",
            )
        )
        for row in lifecycle
    ):
        raise R14Error("R23 lifecycle evidence failed")
    if (
        receipt.get("packages_signature_verified") != 24
        or receipt.get("package_removals") != 24
        or receipt.get("package_restorations") != 24
        or receipt.get("targeted_corruptions_rejected") != 24
    ):
        raise R14Error("R23 lifecycle receipt counts changed")
    result = {
        "format": "abi-r23-live-strict-verification/1",
        "status": "PASS_STRICTLY_VERIFIED_SEMANTIC_REPLICATION",
        "live_config_sha256": sha256_file(config_path),
        "live_receipt_sha256": sha256_file(result_dir / "receipt.json"),
        "live_rows_recomputed": len(live_rows),
        "cpu_rows_recomputed": len(cpu_rows),
        "lifecycle_rows_recomputed": len(lifecycle),
        "packages_recomputed": len(package_ids),
        "stored_scientific_booleans_trusted": False,
        "scientific_claim": "R23_BOUNDED_FRESH_SEMANTIC_SUPPLIED_CONTENT_TRANSFER",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.config, args.result)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
