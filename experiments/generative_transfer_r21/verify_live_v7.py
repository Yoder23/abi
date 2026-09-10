"""Strictly recompute the hostile-repaired R21 fresh live evidence."""

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

from .hash_assurance_binding import selfless_evidence_hash
from .hostile_repair_binding import load_hostile_repair_config
from .live_binding import load_live_config
from .live_verify_v6 import _configs, _jsonl, _stored_checks
from .protocol import WordLabeler, score_output


def verify(config_path: Path, result_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    repair = load_hostile_repair_config(root, config_path)
    live_config_path = root / str(repair["live_verification_config"]["path"])
    live = load_live_config(root, live_config_path)
    receipt = json_object(result_dir / "receipt.json")
    binding = json_object(result_dir / "hostile_repair_binding.json")
    live_path = result_dir / "live_observations.jsonl"
    cpu_path = result_dir / "cpu_observations.jsonl"
    lifecycle_path = result_dir / "lifecycle_observations.jsonl"
    if (
        binding.get("format") != "abi-r21-hostile-repair-run/1"
        or binding.get("hostile_repair_config_sha256") != sha256_file(config_path)
        or binding.get("live_receipt_sha256") != sha256_file(result_dir / "receipt.json")
        or binding.get("evidence_sha256") != selfless_evidence_hash(binding)
        or binding.get("targeted_tensor_mutations") != 24
        or binding.get("candidate_changes") != 0
        or binding.get("gate_changes") != 0
    ):
        raise R14Error("R21 hostile-repair binding failed")
    if (
        receipt.get("format") != "abi-r21-fresh-live-verification/1"
        or receipt.get("status") != "PASS_FRESH_LIVE_PUBLIC_PREREQUISITE"
        or receipt.get("config_sha256") != sha256_file(live_config_path)
        or receipt.get("evidence_sha256") != selfless_evidence_hash(receipt)
        or receipt.get("source_software_loaded") is not False
        or receipt.get("packages_signature_verified") != 24
        or receipt.get("package_removals") != 24
        or receipt.get("package_restorations") != 24
        or receipt.get("corrupt_packages_rejected") != 24
    ):
        raise R14Error("R21 live receipt failed")
    for name, path, expected_rows in (
        ("live_observations", live_path, 1080),
        ("cpu_observations", cpu_path, 54),
        ("lifecycle_observations", lifecycle_path, 24),
    ):
        recorded = receipt.get(name, {})
        if recorded.get("rows") != expected_rows or recorded.get("sha256") != sha256_file(path):
            raise R14Error(f"R21 {name} binding failed")
    base, _ = _configs(root, live)
    engine = json_object(root / str(live["candidate_engine"]["path"]))
    stored_rows = _jsonl(root / str(live["candidate_observations"]["path"]))
    labeler = WordLabeler(json_object(root / str(live["candidate_labeler"]["path"])))
    expected, teacher_by_id, aggregates = _stored_checks(
        root, live, base, engine, stored_rows, labeler
    )
    stored = {(row["method"], int(row["seed"]), row["record_id"]): row for row in stored_rows}
    live_rows = _jsonl(live_path)
    if len(live_rows) != 1080:
        raise R14Error("R21 live row count changed")
    for row in live_rows:
        key = (row["method"], int(row["seed"]), row["record_id"])
        reference = next(item for item in expected if item["record_id"] == row["record_id"])
        if row["output"] != stored[key]["output"]:
            raise R14Error("R21 strict live output changed")
        score = score_output(reference, row["output"], teacher_by_id[row["record_id"]])
        if any(row.get(name) != value for name, value in score.items()):
            raise R14Error("R21 strict live score changed")
        if row.get("output_sha256") != hashlib.sha256(row["output"].encode()).hexdigest():
            raise R14Error("R21 strict live output hash changed")
    from . import run as base_run

    if base_run._aggregate(live_rows) != aggregates:
        raise R14Error("R21 strict live aggregate changed")
    cpu_rows = _jsonl(cpu_path)
    expected_cpu = {
        (method, seed, task)
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
        for seed in (21021, 21022, 21023)
        for task in ("prose", "summary", "email", "bullets", "clarification", "abstention")
    }
    if {(row["method"], row["seed"], row["task"]) for row in cpu_rows} != expected_cpu:
        raise R14Error("R21 strict CPU matrix changed")
    for row in cpu_rows:
        candidate = stored[(row["method"], row["seed"], row["record_id"])]["output"]
        if row["output_sha256"] != hashlib.sha256(candidate.encode()).hexdigest():
            raise R14Error("R21 strict CPU output changed")
    lifecycle = _jsonl(lifecycle_path)
    package_ids = {
        package["cake_id"]
        for seed in engine["systems"].values()
        for method in seed.values()
        for package in method["packages"]
    }
    if {row["cake_id"] for row in lifecycle} != package_ids or not all(
        all(
            row[key]
            for key in (
                "removed",
                "absent_rejected",
                "restored_archive_exact",
                "restored_output_exact",
                "corrupt_rejected",
            )
        )
        for row in lifecycle
    ):
        raise R14Error("R21 strict lifecycle recomputation failed")
    if (
        receipt.get("recomputed_gates") != engine.get("gates")
        or not all(receipt["recomputed_gates"].values())
        or receipt.get("recomputed_aggregates") != aggregates
        or receipt.get("recomputed_comparisons") != engine.get("comparisons")
    ):
        raise R14Error("R21 strict receipt claims changed")
    verification = {
        "format": "abi-r21-hostile-repair-strict-verification/1",
        "status": "PASS_STRICTLY_VERIFIED_PUBLIC_PREREQUISITE",
        "hostile_repair_config_sha256": sha256_file(config_path),
        "live_receipt_sha256": sha256_file(result_dir / "receipt.json"),
        "live_rows_recomputed": len(live_rows),
        "cpu_rows_recomputed": len(cpu_rows),
        "lifecycle_rows_recomputed": len(lifecycle),
        "packages_recomputed": len(package_ids),
        "stored_scientific_booleans_trusted": False,
        "scientific_claim": "R21_BOUNDED_PUBLIC_LABEL_SEPARATED_GENERATIVE_TRANSFER",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "full_abi_moonshot": "OPEN",
        "next_action": "preregister fresh hidden replication",
    }
    verification["evidence_sha256"] = selfless_evidence_hash(verification)
    return verification


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
