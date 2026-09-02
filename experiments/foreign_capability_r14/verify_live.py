"""Fail-closed verification of R14 fresh-process source replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .core import R14Error, evidence_hash, json_object, sha256_file, write_json_once


def verify(config_path: Path, reveal_path: Path, run_dir: Path, replay_dir: Path) -> dict[str, Any]:
    config = json_object(config_path)
    run_receipt = json_object(run_dir / "receipt.json")
    replay = json_object(replay_dir / "receipt.json")
    payload = dict(replay)
    stored = payload.pop("evidence_sha256", None)
    if stored != evidence_hash(payload):
        raise R14Error("live replay evidence hash changed")
    original_path = run_dir / run_receipt["source_observations"]["path"]
    if (
        sha256_file(original_path) != run_receipt["source_observations"]["sha256"]
        or replay["original_source_observations_sha256"] != sha256_file(original_path)
        or replay["config_sha256"] != sha256_file(config_path)
        or replay["reveal_sha256"] != sha256_file(reveal_path)
    ):
        raise R14Error("live replay custody changed")
    original_rows = [
        json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines()
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in original_rows:
        grouped[str(row["capability_id"])].append(row)
    workers = replay["workers"]
    if len(workers) != int(config["data"]["heldout_capabilities"]) or len(
        {int(worker["pid"]) for worker in workers}
    ) != len(workers):
        raise R14Error("live replay worker inventory changed")
    for index, worker in enumerate(workers):
        worker_payload = {
            key: value
            for key, value in worker.items()
            if key not in {"byte_exact", "original_subset_sha256"}
        }
        worker_evidence = worker_payload.pop("evidence_sha256", None)
        if worker_evidence != evidence_hash(worker_payload):
            raise R14Error("source replay worker evidence changed")
        path = replay_dir / "sources" / str(index) / worker["observations"]["path"]
        expected = b"".join(
            canonical_json_bytes(row) for row in grouped[str(worker["capability_id"])]
        )
        if (
            not path.is_file()
            or sha256_file(path) != worker["observations"]["sha256"]
            or path.read_bytes() != expected
            or worker["original_subset_sha256"] != hashlib.sha256(expected).hexdigest()
            or worker["byte_exact"] is not True
            or worker["optimizer_steps"] != 0
            or worker["config_sha256"] != sha256_file(config_path)
            or worker["reveal_sha256"] != sha256_file(reveal_path)
        ):
            raise R14Error("source replay is not byte-exact")
    result = {
        "format": "abi-r14-source-live-verification/1",
        "verdict": "PASS",
        "workers": len(workers),
        "byte_exact_observation_files": len(workers),
        "distinct_processes": len({int(worker["pid"]) for worker in workers}),
        "optimizer_steps": 0,
        "stored_status_booleans_consumed": 0,
        "replay_evidence_sha256": replay["evidence_sha256"],
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        result = verify(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.run_dir).resolve(),
            Path(args.replay_dir).resolve(),
        )
        write_json_once(output, result)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
