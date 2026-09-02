"""Orchestrate fresh-process replay of all frozen R14 source adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .core import R14Error, evidence_hash, json_object, sha256_file, write_json_once


def run(config_path: Path, reveal_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable live replay output exists: {output}")
    config = json_object(config_path)
    original_receipt = json_object(run_dir / "receipt.json")
    original_path = run_dir / original_receipt["source_observations"]["path"]
    if sha256_file(original_path) != original_receipt["source_observations"]["sha256"]:
        raise R14Error("original source observations changed")
    original_rows = [
        json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines()
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in original_rows:
        grouped[str(row["capability_id"])].append(row)
    output.mkdir(parents=True)
    workers = []
    for index in range(int(config["data"]["heldout_capabilities"])):
        worker_output = output / "sources" / str(index)
        command = [
            sys.executable,
            "-m",
            "experiments.foreign_capability_r14.source_replay_worker",
            "--config",
            str(config_path),
            "--reveal",
            str(reveal_path),
            "--run-dir",
            str(run_dir),
            "--capability-index",
            str(index),
            "--output",
            str(worker_output),
        ]
        environment = dict(os.environ)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        completed = subprocess.run(
            command,
            cwd=config_path.parents[3],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"source-{index}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (logs / f"source-{index}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise R14Error(f"source replay failed: {index}")
        worker = json_object(worker_output / "receipt.json")
        replay_path = worker_output / worker["observations"]["path"]
        expected = b"".join(canonical_json_bytes(row) for row in grouped[worker["capability_id"]])
        if replay_path.read_bytes() != expected:
            raise R14Error(f"source replay differs: {worker['capability_id']}")
        workers.append(
            {
                **worker,
                "original_subset_sha256": hashlib.sha256(expected).hexdigest(),
                "byte_exact": True,
            }
        )
    if len({int(worker["pid"]) for worker in workers}) != len(workers):
        raise R14Error("source replay workers did not use distinct processes")
    result = {
        "format": "abi-r14-source-live-replay/1",
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "original_source_observations_sha256": sha256_file(original_path),
        "workers": workers,
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
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
