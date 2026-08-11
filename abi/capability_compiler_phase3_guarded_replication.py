"""Run exactly two fresh-process host-initialization replications of V494."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-guarded-host-replication/1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EXACTLY_TWO_FRESH_PROCESS_HOST_INITIALIZATIONS"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("guarded replication governance changed")
    initializations = protocol.get("host_initializations", [])
    if len(initializations) != 2 or len(set(initializations)) != 2:
        raise Phase3Error("exactly two unique host initializations required")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"guarded replication binding changed: {relative}")
    return protocol, sha256_file(path)


def evaluate_replications(protocol: dict[str, Any], results: list[dict[str, Any]], output_hashes: list[str]) -> dict[str, bool]:
    initial_sha = str(protocol["candidate"]["checkpoint_sha256"])
    reference_sha = str(protocol["reference"]["outputs_sha256"])
    return {
        "exactly_two_fresh_replications": len(results) == 2,
        "both_complete_gate_passes": len(results) == 2 and all(row.get("passed") is True and all(row.get("gates", {}).values()) for row in results),
        "same_unchanged_checkpoint": len(results) == 2 and all(row.get("checkpoint_sha256") == initial_sha for row in results),
        "same_locked_observation_depth": len(results) == 2 and all(row.get("observations") == 1400 for row in results),
        "deterministic_output_identity_to_reference": len(output_hashes) == 2 and all(value == reference_sha for value in output_hashes),
        "teacher_absent_at_inference": len(results) == 2 and all(row.get("teacher_present_at_inference") is False for row in results),
        "final_test_not_accessed": len(results) == 2 and all(row.get("final_test_accessed") is False for row in results),
    }


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable replication output exists: {output}")
    results: list[dict[str, Any]] = []
    output_hashes: list[str] = []
    receipts: list[dict[str, Any]] = []
    screen_protocol = str(protocol["screen_protocol"])
    for initialization in protocol["host_initializations"]:
        target = output / str(initialization)
        command = [
            sys.executable,
            "-m",
            "abi.capability_compiler_phase3_guarded_screen",
            "--protocol",
            screen_protocol,
            "--output-dir",
            target.relative_to(root).as_posix(),
        ]
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        receipts.append({
            "host_initialization": initialization,
            "command": command[1:],
            "returncode": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        })
        if completed.returncode != 0:
            raise Phase3Error(f"host initialization failed: {initialization}")
        result_path = target / "result.json"
        outputs_path = target / "development_outputs.jsonl"
        result = _json(result_path)
        results.append(result)
        output_hashes.append(sha256_file(outputs_path))
    gates = evaluate_replications(protocol, results, output_hashes)
    passed = all(gates.values())
    summary = {
        "format": FORMAT,
        "status": "PASS_THREE_HOST_INITIALIZATIONS_TOTAL_RUNTIME_OPEN" if passed else "FAIL_HOST_REPLICATIONS_CLOSED",
        "protocol_sha256": protocol_sha,
        "new_host_initializations": len(results),
        "total_host_initializations_including_v494": 1 + len(results),
        "checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"],
        "replications": [
            {
                "host_initialization": initialization,
                "result_sha256": sha256_file(output / str(initialization) / "result.json"),
                "outputs_sha256": output_hash,
                "functional_passes_v1": result["functional_passes_v1"],
                "repetition_collapses_v2": result["repetition_collapses_v2"],
                "teacher_relative_lower_95": result["teacher_comparison_v1"]["lower_95"],
                "evaluation_wall_seconds": result["evaluation_wall_seconds"],
            }
            for initialization, result, output_hash in zip(protocol["host_initializations"], results, output_hashes)
        ],
        "fresh_process_receipts": receipts,
        "gates": gates,
        "passed": passed,
        "teacher_present_at_inference": False,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
    }
    summary["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(summary)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable(output / "replication_result.json", json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_GUARDED_REPLICATION_PROTOCOL_V499.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_guarded_replication/replication_v500")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
