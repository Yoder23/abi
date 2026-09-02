"""Issue the bounded R13 certificate only after live replay and hostile audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import R13Error, evidence_hash, json_object, write_json_once
from .verify import _evidence
from .verify_live import verify_final


def seal(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    live_replay_dir: Path,
    hostile_audit_path: Path,
) -> dict:
    final = verify_final(config_path, reveal_path, run_dir, live_replay_dir)
    hostile = json_object(hostile_audit_path)
    _evidence(hostile, "hostile audit")
    if (
        hostile.get("format") != "abi-r13-hostile-audit/1"
        or hostile.get("verdict") != "PASS"
        or hostile.get("original_evidence_modified") is not False
        or hostile.get("expected_outcomes") != 9
        or hostile.get("matched_outcomes") != 9
        or hostile.get("baseline_final_evidence_sha256") != final["evidence_sha256"]
        or any(item.get("expected") != item.get("actual") for item in hostile.get("cases", []))
    ):
        raise R13Error("hostile audit gate failed")
    result = {
        "format": "abi-r13-bounded-capability-extraction-certificate/1",
        "verdict": "PASS",
        "claim": "BOUNDED_ENUMERABLE_CAPABILITY_EXTRACTION",
        "claim_ceiling": "NOT_BEHAVIORAL_TRANSPLANT_NOT_GENERAL_KNOWLEDGE",
        "capabilities": final["capabilities"],
        "strict_verification_evidence_sha256": final[
            "strict_verification_evidence_sha256"
        ],
        "live_replay_evidence_sha256": final["live_replay_evidence_sha256"],
        "hostile_audit_evidence_sha256": hostile["evidence_sha256"],
        "run_evidence_sha256": final["run_evidence_sha256"],
        "all_observation_files_byte_exact": final["all_observation_files_byte_exact"],
        "hostile_outcomes": hostile["matched_outcomes"],
        "stored_status_booleans_consumed": 0,
        "controlling_public_release_changed": False,
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--live-replay-dir", required=True)
    parser.add_argument("--hostile-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = seal(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.run_dir).resolve(),
            Path(args.live_replay_dir).resolve(),
            Path(args.hostile_audit).resolve(),
        )
        write_json_once(Path(args.output).resolve(), result)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
