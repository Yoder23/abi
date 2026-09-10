"""Seal the bounded R15B result from recomputed strict/live/hostile evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)

from .accounting import account
from .hostile_audit import audit
from .verify import _verify_evidence, verify
from .verify_live import _scientific_extraction


def _verify_live_receipt(run_dir: Path, live_dir: Path) -> dict[str, Any]:
    original = json_object(run_dir / "receipt.json")
    live = json_object(live_dir / "receipt.json")
    _verify_evidence(live, "R15B live receipt")
    if live.get("format") != "abi-r15b-live-verification/2":
        raise R14Error("R15B live receipt does not include repaired tensor verification")
    snapshot = live.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise R14Error("R15B live source snapshot inventory is missing")
    _verify_evidence(snapshot, "R15B live source snapshot inventory")
    if (
        snapshot.get("model_id") != original["source"]["model_id"]
        or snapshot.get("revision") != original["source"]["revision"]
        or snapshot.get("file_count") != len(snapshot.get("files", []))
        or snapshot.get("total_bytes")
        != sum(int(item["bytes"]) for item in snapshot.get("files", []))
        or not snapshot.get("files")
    ):
        raise R14Error("R15B live source snapshot inventory changed")
    source_path = live_dir / "source_observations.jsonl"
    original_source = run_dir / original["source"]["observations"]["path"]
    if (
        sha256_file(source_path) != live["source_rows_sha256"]
        or source_path.read_bytes() != original_source.read_bytes()
        or live["source_rows_replayed_byte_exact"] != original["source"]["observations"]["rows"]
    ):
        raise R14Error("R15B stored live source replay changed")
    if len(live["extractions"]) != len(original["isolated_extractions"]):
        raise R14Error("R15B stored live extraction inventory changed")
    if (
        live.get("source_bundle_tensors_replayed_exact")
        != len(original["source"]["capability_receipts"])
        or len(live.get("source_bundles", []))
        != len(original["source"]["capability_receipts"])
        or any(item.get("tensors_byte_exact") is not True for item in live["source_bundles"])
    ):
        raise R14Error("R15B stored live source tensor replay changed")
    for stored, source_item in zip(
        live["source_bundles"], original["source"]["capability_receipts"]
    ):
        if (
            stored.get("capability_id") != source_item["capability_id"]
            or stored.get("sha256") != source_item["bundle"]["sha256"]
            or stored.get("path") != source_item["bundle"]["path"]
        ):
            raise R14Error("R15B stored live source bundle identity changed")
    for index, item in enumerate(live["extractions"]):
        live_result = json_object(live_dir / item["path"] / "result.json")
        original_result = json_object(
            run_dir / original["isolated_extractions"][index]["path"] / "result.json"
        )
        if (
            sha256_file(live_dir / item["path"] / "result.json") != item["result_sha256"]
            or evidence_hash(_scientific_extraction(live_result))
            != item["scientific_evidence_sha256"]
            or _scientific_extraction(live_result) != _scientific_extraction(original_result)
        ):
            raise R14Error("R15B stored live extraction changed")
    original_workers = {item["host"]: item for item in original["recipient_workers"]}
    for item in live["recipients"]:
        worker = json_object(run_dir / original_workers[item["host"]]["path"])
        original_rows = run_dir / worker["observations"]["path"]
        live_rows = live_dir / "recipients" / item["host"] / "observations.jsonl"
        if (
            sha256_file(live_rows) != item["rows_sha256"]
            or live_rows.read_bytes() != original_rows.read_bytes()
            or item["rows"] != worker["observations"]["rows"]
        ):
            raise R14Error("R15B stored live recipient changed")
    if (
        live.get("verdict") != "PASS"
        or live["isolated_extractions_replayed"] != len(live["extractions"])
        or live["recipient_rows_replayed_byte_exact"]
        != sum(int(item["rows"]) for item in live["recipients"])
    ):
        raise R14Error("R15B stored live replay totals changed")
    return live


def seal(
    config: Path,
    reveal: Path,
    run_dir: Path,
    strict_path: Path,
    live_dir: Path,
    hostile_path: Path,
    accounting_path: Path,
) -> dict[str, Any]:
    strict = verify(config, reveal, run_dir)
    stored_strict = json_object(strict_path)
    _verify_evidence(stored_strict, "R15B stored strict receipt")
    if {key: value for key, value in stored_strict.items() if key != "evidence_sha256"} != strict:
        raise R14Error("R15B stored strict verification changed")
    live = _verify_live_receipt(run_dir, live_dir)
    stored_hostile = json_object(hostile_path)
    _verify_evidence(stored_hostile, "R15B stored hostile receipt")
    recomputed_hostile = audit(config, reveal, run_dir)
    if stored_hostile != recomputed_hostile or any(
        item.get("rejected") is not True for item in stored_hostile["cases"]
    ):
        raise R14Error("R15B hostile audit changed")
    stored_accounting = json_object(accounting_path)
    _verify_evidence(stored_accounting, "R15B accounting receipt")
    if stored_accounting != account(config, reveal, run_dir):
        raise R14Error("R15B information accounting changed")
    certificate = {
        "format": "abi-r15b-bounded-certificate/2",
        "repair_of": "results/preexisting_representation_r15b/heldout_v1_certificate.json",
        "verdict": "PASS",
        "claim": "BOUNDED_PREEXISTING_REPRESENTATION_CAPABILITY_RECOVERY",
        "claim_ceiling": "NOT_ENGLISH_DOMAIN_OR_TEACHER_BEHAVIOR_CLONING",
        "full_abi_moonshot": "OPEN",
        "artifact_publication": "LOCAL_SEAL_PENDING_PUBLIC_RECONSTRUCTION",
        "chronology": {
            "implementation_freeze_commit": "0d4a75c771cbc46ce2680c81a57ce5ce193acdd0",
            "preregistration_commit": "8a51dae0512d983798bd4b4c3afb0157ce9a6456",
            "reveal_commit": "dafefc97b785d4ed8d3344a5904389560790e2b5",
        },
        "results": {
            "source_model": "Qwen/Qwen2-7B-Instruct",
            "source_training_steps": 0,
            "source_anchors_exact": 24,
            "source_anchors_total": 24,
            "capabilities_exact": strict["capabilities_exact"],
            "package_rows_exact": strict["package_evaluation_rows"],
            "package_rows_total": strict["package_evaluation_rows"],
            "recipient_hosts": strict["recipient_hosts"],
            "live_source_rows": live["source_rows_replayed_byte_exact"],
            "live_source_bundles": live["source_bundle_tensors_replayed_exact"],
            "source_snapshot_evidence_sha256": live["source_snapshot"]["evidence_sha256"],
            "live_recipient_rows": live["recipient_rows_replayed_byte_exact"],
            "hostile_cases_rejected": stored_hostile["cases_passed"],
            "hostile_cases_total": stored_hostile["cases_total"],
        },
        "information_accounting": stored_accounting,
        "evidence": {
            "run_receipt_sha256": sha256_file(run_dir / "receipt.json"),
            "strict_receipt_sha256": sha256_file(strict_path),
            "live_receipt_sha256": sha256_file(live_dir / "receipt.json"),
            "hostile_receipt_sha256": sha256_file(hostile_path),
            "accounting_receipt_sha256": sha256_file(accounting_path),
        },
        "unproven": [
            "deep teacher-behavior cloning",
            "arbitrary pretrained knowledge extraction",
            "English fluency transfer",
            "autonomous open-world domain discovery or labeling",
            "minimal English substrate",
            "production LayerCake ingestion",
            "superiority to LoRA or distillation",
            "independent different-hardware reproduction",
        ],
    }
    certificate["evidence_sha256"] = evidence_hash(certificate)
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--live-dir", type=Path, required=True)
    parser.add_argument("--hostile", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    certificate = seal(
        args.config,
        args.reveal,
        args.run_dir,
        args.strict,
        args.live_dir,
        args.hostile,
        args.accounting,
    )
    write_json_once(args.output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
