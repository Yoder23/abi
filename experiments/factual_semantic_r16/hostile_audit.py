"""Fail-closed hostile mutation audit for R16 evidence."""

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
        raise RuntimeError("cannot mutate empty R16 evidence")
    value[len(value) // 2] ^= 1
    path.write_bytes(value)


def _missing_source(run: Path) -> None:
    (run / "source_observations.jsonl").unlink()


def _corrupt_source(run: Path) -> None:
    _flip(run / "source_observations.jsonl")


def _missing_bundle(run: Path) -> None:
    (run / "source_bundle.json").unlink()


def _missing_package(run: Path) -> None:
    next((run / "extraction/packages").glob("*.abipkg")).unlink()


def _corrupt_package(run: Path) -> None:
    _flip(next((run / "extraction/packages").glob("*.abipkg")))


def _missing_evaluation(run: Path) -> None:
    (run / "evaluation.jsonl").unlink()


def _tamper_receipt(run: Path) -> None:
    path = run / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source"]["present_at_package_execution"] = True
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _forged_metrics(run: Path) -> None:
    path = run / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["metrics"]["selected_facts"] = 999
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    value["evidence_sha256"] = evidence_hash(scientific)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


CASES: dict[str, Callable[[Path], None]] = {
    "missing_source": _missing_source,
    "corrupt_source": _corrupt_source,
    "missing_bundle": _missing_bundle,
    "missing_package": _missing_package,
    "corrupt_package": _corrupt_package,
    "missing_evaluation": _missing_evaluation,
    "tampered_receipt": _tamper_receipt,
    "hash_consistent_forged_metrics": _forged_metrics,
}


def audit(config: Path, reveal: Path, run_dir: Path) -> dict[str, Any]:
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="abi-r16-hostile-") as raw:
        root = Path(raw)
        for name, mutation in CASES.items():
            candidate = root / name
            shutil.copytree(run_dir, candidate)
            mutation(candidate)
            rejected = False
            try:
                verify(config, reveal, candidate)
            except (R14Error, OSError, ValueError, KeyError, json.JSONDecodeError):
                rejected = True
            outcomes.append({"case": name, "rejected": rejected})
    if not all(item["rejected"] for item in outcomes):
        raise R14Error("R16 hostile audit accepted a mutation")
    result = {
        "format": "abi-r16-hostile-audit/1",
        "verdict": "PASS",
        "cases": outcomes,
        "cases_passed": len(outcomes),
        "cases_total": len(outcomes),
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config, args.reveal, args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
