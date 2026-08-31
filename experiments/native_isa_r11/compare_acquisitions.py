"""Compare two independently acquired R11 runs without hiding byte drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .core import R11Error, sha256_file
from .run import _json, _write_json_once
from .verify import verify


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise R11Error("acquisition comparison row is not an object")
            yield value


def _compare_rows(first: Path, second: Path) -> dict[str, Any]:
    identity = ("host", "capability_id", "condition", "row_id", "prompt_sha256")
    behavior = (
        "prediction_token_id",
        "canonical_prediction",
        "canonical_output_utf8_hex",
    )
    count = 0
    behavior_mismatches = 0
    neural_state_mismatches = 0
    first_rows = _rows(first)
    second_rows = _rows(second)
    while True:
        try:
            left = next(first_rows)
        except StopIteration:
            left = None
        try:
            right = next(second_rows)
        except StopIteration:
            right = None
        if left is None or right is None:
            if left is not right:
                raise R11Error("acquisition row counts differ")
            break
        if any(left[field] != right[field] for field in identity):
            raise R11Error(f"acquisition row identity differs at index {count}")
        behavior_mismatches += int(any(left[field] != right[field] for field in behavior))
        neural_state_mismatches += int(left["neural_state"] != right["neural_state"])
        count += 1
    return {
        "rows": count,
        "behavior_mismatches": behavior_mismatches,
        "neural_state_mismatches": neural_state_mismatches,
        "first_sha256": sha256_file(first),
        "second_sha256": sha256_file(second),
    }


def compare(config: Path, reveal: Path, first: Path, second: Path, output: Path) -> dict[str, Any]:
    first_verification = verify(config, reveal, first)
    second_verification = verify(config, reveal, second)
    first_receipt = _json(first / "receipt.json")
    second_receipt = _json(second / "receipt.json")
    first_packages = [item["sha256"] for item in first_receipt["packages"]["after"]]
    second_packages = [item["sha256"] for item in second_receipt["packages"]["after"]]
    teacher = _compare_rows(
        first / "teacher_observations.jsonl", second / "teacher_observations.jsonl"
    )
    recipients = _compare_rows(
        first / "recipient_observations.jsonl",
        second / "recipient_observations.jsonl",
    )
    report = {
        "format": "abi-native-neural-isa-r11-acquisition-comparison/1",
        "first_verification_evidence_sha256": first_verification["evidence_sha256"],
        "second_verification_evidence_sha256": second_verification["evidence_sha256"],
        "teacher_rows": teacher,
        "recipient_rows": recipients,
        "behavior_mismatches": teacher["behavior_mismatches"] + recipients["behavior_mismatches"],
        "package_pairs": len(first_packages),
        "identical_package_pairs": sum(
            int(left == right) for left, right in zip(first_packages, second_packages)
        ),
        "first_package_sha256": first_packages,
        "second_package_sha256": second_packages,
        "functional_replication": teacher["behavior_mismatches"] == 0
        and recipients["behavior_mismatches"] == 0,
        "bitwise_acquisition_replication": first_packages == second_packages,
    }
    report["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    if not report["functional_replication"]:
        raise R11Error("independent acquisition changed registered behavior")
    if report["bitwise_acquisition_replication"]:
        raise R11Error("expected acquisition-byte diagnostic did not reproduce drift")
    _write_json_once(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = compare(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.first).resolve(),
            Path(args.second).resolve(),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
