"""Fresh-process source and recipient replay for the R13-B evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .core import R13Error, evidence_hash, json_object, write_json_once
from .verify import _evidence, _jsonl, verify


def _run(command: list[str], root: Path, log_dir: Path, label: str) -> None:
    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (log_dir / f"{label}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise R13Error(f"live replay worker failed: {label}")


def run(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R13Error(f"immutable live replay exists: {output}")
    strict = verify(config_path, reveal_path, run_dir)
    config = json_object(config_path)
    receipt = json_object(run_dir / "receipt.json")
    _evidence(receipt, "run receipt")
    root = config_path.parents[3]
    output.mkdir(parents=True)
    original_source = _jsonl(
        run_dir / receipt["source_observations"]["path"],
        receipt["source_observations"]["sha256"],
        receipt["source_observations"]["rows"],
    )
    source_replays = []
    for index, source in enumerate(receipt["source_acquisitions"]):
        capability_id = str(source["capability_id"])
        worker_dir = output / "sources" / capability_id
        _run(
            [
                sys.executable,
                "-m",
                "experiments.foreign_capability_r13.source_replay_worker",
                "--config",
                str(config_path),
                "--reveal",
                str(reveal_path),
                "--run-dir",
                str(run_dir),
                "--capability-index",
                str(index),
                "--output",
                str(worker_dir),
            ],
            root,
            output / "logs",
            f"source-{index}",
        )
        worker = json_object(worker_dir / "receipt.json")
        _evidence(worker, f"source replay {index}")
        expected = b"".join(
            canonical_json_bytes(row)
            for row in original_source
            if str(row["capability_id"]) == capability_id
        )
        actual_path = worker_dir / worker["observations"]["path"]
        actual = actual_path.read_bytes()
        source_replays.append(
            {
                "capability_id": capability_id,
                "pid": worker["pid"],
                "source_optimizer_steps": worker["source_optimizer_steps"],
                "adapter_sha256": worker["adapter_sha256"],
                "rows": worker["observations"]["rows"],
                "expected_sha256": hashlib.sha256(expected).hexdigest(),
                "actual_sha256": hashlib.sha256(actual).hexdigest(),
                "byte_exact": actual == expected,
            }
        )
    recipient_replays = []
    manifest_path = run_dir / "package_manifest.json"
    for host_key in config["recipient_hosts"]:
        worker_dir = output / "recipients" / str(host_key)
        _run(
            [
                sys.executable,
                "-m",
                "experiments.foreign_capability_r13.recipient_worker",
                "--config",
                str(config_path),
                "--reveal",
                str(reveal_path),
                "--manifest",
                str(manifest_path),
                "--host",
                str(host_key),
                "--output",
                str(worker_dir),
            ],
            root,
            output / "logs",
            f"recipient-{host_key}",
        )
        worker = json_object(worker_dir / "receipt.json")
        _evidence(worker, f"recipient replay {host_key}")
        original_worker = next(
            item for item in receipt["recipient_workers"] if item["host"] == host_key
        )
        original_path = run_dir / "recipients" / str(host_key) / original_worker[
            "observations"
        ]["path"]
        replay_path = worker_dir / worker["observations"]["path"]
        original = original_path.read_bytes()
        replayed = replay_path.read_bytes()
        recipient_replays.append(
            {
                "host": host_key,
                "pid": worker["pid"],
                "source_adapter_loaded": worker["source_adapter_loaded"],
                "recipient_optimizer_steps": worker["host_receipt"][
                    "recipient_optimizer_steps"
                ],
                "rows": worker["observations"]["rows"],
                "expected_sha256": hashlib.sha256(original).hexdigest(),
                "actual_sha256": hashlib.sha256(replayed).hexdigest(),
                "byte_exact": replayed == original,
            }
        )
    all_replays = [*source_replays, *recipient_replays]
    pids = [int(item["pid"]) for item in all_replays]
    if (
        not all(item["byte_exact"] for item in all_replays)
        or len(pids) != len(set(pids))
        or any(item["source_optimizer_steps"] != 0 for item in source_replays)
        or any(item["source_adapter_loaded"] is not False for item in recipient_replays)
        or any(item["recipient_optimizer_steps"] != 0 for item in recipient_replays)
    ):
        raise R13Error("live replay gate failed")
    result = {
        "format": "abi-r13-live-replay/1",
        "verdict": "PASS",
        "claim": "BOUNDED_ENUMERABLE_CAPABILITY_EXTRACTION",
        "strict_verification_evidence_sha256": strict["evidence_sha256"],
        "run_evidence_sha256": receipt["evidence_sha256"],
        "source_replays": source_replays,
        "recipient_replays": recipient_replays,
        "distinct_worker_processes": len(set(pids)),
        "all_observation_files_byte_exact": True,
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.run_dir).resolve(),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
