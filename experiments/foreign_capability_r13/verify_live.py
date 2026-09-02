"""Final R13 verifier requiring static recomputation and exact live replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import R13Error, evidence_hash, json_object, write_json_once
from .verify import _evidence, verify


def verify_final(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    live_replay_dir: Path,
) -> dict:
    strict = verify(config_path, reveal_path, run_dir)
    live = json_object(live_replay_dir / "receipt.json")
    _evidence(live, "live replay")
    if (
        live.get("format") != "abi-r13-live-replay/1"
        or live.get("claim") != strict["claim"]
        or live.get("strict_verification_evidence_sha256") != strict["evidence_sha256"]
        or live.get("run_evidence_sha256") != strict["run_evidence_sha256"]
        or live.get("all_observation_files_byte_exact") is not True
        or live.get("distinct_worker_processes") != 7
        or len(live.get("source_replays", [])) != 4
        or len(live.get("recipient_replays", [])) != 3
        or not all(item.get("byte_exact") is True for item in live["source_replays"])
        or not all(item.get("byte_exact") is True for item in live["recipient_replays"])
    ):
        raise R13Error("final live-replay gate failed")
    result = {
        "format": "abi-r13-final-certificate/1",
        "verdict": "PASS",
        "claim": "BOUNDED_ENUMERABLE_CAPABILITY_EXTRACTION",
        "strict_verification_evidence_sha256": strict["evidence_sha256"],
        "live_replay_evidence_sha256": live["evidence_sha256"],
        "run_evidence_sha256": strict["run_evidence_sha256"],
        "capabilities": strict["capabilities"],
        "source_process_replays": len(live["source_replays"]),
        "recipient_process_replays": len(live["recipient_replays"]),
        "all_observation_files_byte_exact": True,
        "stored_status_booleans_consumed": 0,
        "claim_ceiling": "NOT_BEHAVIORAL_TRANSPLANT_NOT_GENERAL_KNOWLEDGE",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--live-replay-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = verify_final(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.run_dir).resolve(),
            Path(args.live_replay_dir).resolve(),
        )
        write_json_once(Path(args.output).resolve(), result)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
