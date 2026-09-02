"""Seal the recomputed R14 outcome without promoting a failed claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core import R14Error, evidence_hash, json_object, write_json_once
from .diagnose import diagnose
from .hostile_audit import audit
from .verify_live import verify as verify_live


def _audit_science(value: dict[str, Any]) -> list[tuple[str, str]]:
    return [(str(item["case"]), str(item["outcome"])) for item in value["cases"]]


def seal(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    replay_dir: Path,
    hostile_path: Path,
) -> dict[str, Any]:
    diagnosis = diagnose(config_path, reveal_path, run_dir)
    live = verify_live(config_path, reveal_path, run_dir, replay_dir)
    stored_hostile = json_object(hostile_path)
    payload = dict(stored_hostile)
    stored_hash = payload.pop("evidence_sha256", None)
    if stored_hash != evidence_hash(payload):
        raise R14Error("hostile audit evidence hash changed")
    recomputed_hostile = audit(config_path, reveal_path, run_dir, replay_dir)
    if (
        stored_hostile.get("verdict") != "PASS"
        or stored_hostile.get("cases_passed") != stored_hostile.get("cases_total")
        or _audit_science(stored_hostile) != _audit_science(recomputed_hostile)
        or recomputed_hostile.get("verdict") != "PASS"
    ):
        raise R14Error("hostile audit did not recompute")
    if diagnosis["verdict"] != "FAIL" or diagnosis["capabilities_passing"] != 1:
        raise R14Error("R14 diagnosis is not the registered one-of-three negative")
    receipt = json_object(run_dir / "receipt.json")
    result = {
        "format": "abi-r14-bounded-negative-certificate/1",
        "verdict": "R14_NOT_CERTIFIED",
        "outcome": "BOUNDED_PARTIAL_NON_EXHAUSTIVE_RECOVERY_1_OF_3",
        "claim_target": diagnosis["claim"],
        "capabilities_passing": diagnosis["capabilities_passing"],
        "capabilities_total": diagnosis["capabilities_total"],
        "extractor_queries_per_capability": diagnosis["extractor_queries_per_capability"],
        "atomic_extractor_queries": diagnosis["atomic_extractor_queries"],
        "evaluation_rows_per_capability": int(receipt["data"]["evaluation_rows_per_capability"]),
        "order_counterfactual_rows_per_capability": int(
            receipt["data"]["counterfactual_rows_per_capability"]
        ),
        "evaluation_behavior_space_per_capability": diagnosis[
            "evaluation_behavior_space_per_capability"
        ],
        "recipient_execution": diagnosis["recipient_execution"],
        "failures": diagnosis["failures"],
        "diagnosis_evidence_sha256": diagnosis["evidence_sha256"],
        "live_replay_evidence_sha256": live["evidence_sha256"],
        "hostile_audit_evidence_sha256": stored_hostile["evidence_sha256"],
        "fresh_source_processes": live["distinct_processes"],
        "byte_exact_source_replay_files": live["byte_exact_observation_files"],
        "hostile_controls_passed": stored_hostile["cases_passed"],
        "hostile_controls_total": stored_hostile["cases_total"],
        "stored_status_booleans_consumed": 0,
        "claim_ceiling": "NOT_LOSSLESS_TEACHER_CLONING_NOT_PREEXISTING_KNOWLEDGE",
        "next_frontend_question": "CONTROLLED_FOREIGN_NEURAL_STATE_EXTRACTION",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--hostile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    result = seal(
        Path(args.config).resolve(),
        Path(args.reveal).resolve(),
        Path(args.run_dir).resolve(),
        Path(args.replay_dir).resolve(),
        Path(args.hostile).resolve(),
    )
    write_json_once(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
