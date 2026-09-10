"""Fail-closed hostile mutation audit for R18 evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, evidence_hash, write_json_once

from .verify import verify


def _flip(path: Path) -> None:
    value = bytearray(path.read_bytes())
    if not value:
        raise RuntimeError("cannot mutate empty R18 evidence")
    value[len(value) // 2] ^= 1
    path.write_bytes(value)


def _remove(relative: str) -> Callable[[Path], None]:
    def mutate(run: Path) -> None:
        (run / relative).unlink()

    return mutate


def _remove_primary_package(run: Path) -> None:
    next((run / "extraction/packages").glob("*.abipkg")).unlink()


def _corrupt_primary_package(run: Path) -> None:
    _flip(next((run / "extraction/packages").glob("*.abipkg")))


def _remove_control_package(run: Path) -> None:
    next((run / "control_extraction/packages").glob("*.abipkg")).unlink()


def _tamper_receipt(run: Path) -> None:
    path = run / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source"]["present_at_package_execution"] = True
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _forge_metrics(run: Path) -> None:
    path = run / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["metrics"]["package_functional_exact"] = 999
    value["evidence_sha256"] = evidence_hash(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _forge_evaluation(run: Path) -> None:
    matrix = run / "evaluation.jsonl"
    rows = [json.loads(line) for line in matrix.read_text(encoding="utf-8").splitlines()]
    rows[0]["package_functional_exact"] = False
    matrix.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt_path = run / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    import hashlib

    receipt["artifacts"]["evaluation"]["sha256"] = hashlib.sha256(matrix.read_bytes()).hexdigest()
    receipt["evidence_sha256"] = evidence_hash(
        {key: item for key, item in receipt.items() if key != "evidence_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _forge_isolation_claim(run: Path) -> None:
    path = run / "extraction/result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["old_root_present"] = True
    value["evidence_sha256"] = evidence_hash(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


CASES: dict[str, Callable[[Path], None]] = {
    "missing_receipt": _remove("receipt.json"),
    "tampered_receipt": _tamper_receipt,
    "hash_consistent_forged_metrics": _forge_metrics,
    "missing_source_bundle": _remove("source_bundle.json"),
    "corrupt_source_bundle": lambda run: _flip(run / "source_bundle.json"),
    "missing_primary_package": _remove_primary_package,
    "corrupt_primary_package": _corrupt_primary_package,
    "missing_control_package": _remove_control_package,
    "missing_evaluation": _remove("evaluation.jsonl"),
    "hash_consistent_forged_evaluation": _forge_evaluation,
    "missing_mountinfo": _remove("extraction/mountinfo.txt"),
    "hash_consistent_bad_isolation_claim": _forge_isolation_claim,
}


def audit(source_run: Path, run_dir: Path) -> dict[str, Any]:
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="abi-r18-hostile-") as raw:
        root = Path(raw)
        for name, mutation in CASES.items():
            candidate = root / name
            shutil.copytree(run_dir, candidate)
            mutation(candidate)
            rejected = False
            try:
                verify(source_run, candidate)
            except (R14Error, OSError, ValueError, KeyError, json.JSONDecodeError):
                rejected = True
            outcomes.append({"case": name, "rejected": rejected})
    if not all(item["rejected"] for item in outcomes):
        raise R14Error("R18 hostile audit accepted a mutation")
    result = {
        "format": "abi-r18-hostile-audit/1",
        "verdict": "PASS",
        "cases": outcomes,
        "cases_passed": len(outcomes),
        "cases_total": len(outcomes),
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_run, args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
