"""Run one command under a verified Windows logical-processor affinity.

The child inherits this process's affinity.  A hash-bound sidecar records the
requested, applied, and child-observed masks without changing system-wide
scheduling, power plans, or core-parking policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil


class CpuAffinityRunError(RuntimeError):
    """Raised when affinity cannot be applied or evidence would be replaced."""


def _canonical_sha(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("evidence_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_processors(value: str) -> list[int]:
    try:
        processors = sorted({int(item) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "logical processors must be comma-separated integers"
        ) from exc
    if not processors or processors[0] < 0:
        raise argparse.ArgumentTypeError(
            "at least one non-negative logical processor is required"
        )
    return processors


def run_with_affinity(
    *,
    processors: list[int],
    command: list[str],
    cwd: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise CpuAffinityRunError(
            f"affinity evidence is immutable: {output}"
        )
    parent = psutil.Process()
    available_before = parent.cpu_affinity()
    if not set(processors).issubset(available_before):
        raise CpuAffinityRunError(
            "requested processors are outside the available affinity"
        )
    parent.cpu_affinity(processors)
    applied = parent.cpu_affinity()
    if applied != processors:
        raise CpuAffinityRunError(
            f"requested affinity {processors}, observed {applied}"
        )
    started = time.time_ns()
    child = subprocess.Popen(command, cwd=cwd)
    child_process = psutil.Process(child.pid)
    child_observed = child_process.cpu_affinity()
    return_code = child.wait()
    ended = time.time_ns()
    document: dict[str, Any] = {
        "format": "abi-cpu-affinity-command/1",
        "status": "PASS" if return_code in (0, 1, 2) else "FAIL",
        "claim_boundary": (
            "PASS means the child inherited the requested affinity and "
            "terminated normally. It does not mean the benchmark gates passed."
        ),
        "available_logical_processors_before": available_before,
        "requested_logical_processors": processors,
        "applied_parent_logical_processors": applied,
        "observed_child_logical_processors": child_observed,
        "child_inherited_exact_affinity": child_observed == processors,
        "command": command,
        "cwd": str(cwd),
        "child_return_code": int(return_code),
        "started_unix_time_ns": started,
        "ended_unix_time_ns": ended,
        "wall_seconds": (ended - started) / 1_000_000_000,
        "system_wide_configuration_changed": False,
        "final_test_accessed": False,
    }
    document["status"] = (
        "PASS"
        if document["child_inherited_exact_affinity"]
        and return_code in (0, 1, 2)
        else "FAIL"
    )
    document["evidence_sha256"] = _canonical_sha(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logical-processors", type=_parse_processors, required=True
    )
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    document = run_with_affinity(
        processors=args.logical_processors,
        command=command,
        cwd=args.cwd.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
