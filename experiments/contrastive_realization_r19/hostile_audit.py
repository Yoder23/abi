"""Hostile mutation audit for strict R19 development verification."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, evidence_hash, write_json_once

from .package import R19PackageError
from .verify_development import verify


def _flip(path: Path) -> None:
    value = bytearray(path.read_bytes())
    value[len(value) // 2] ^= 1
    path.write_bytes(value)


def _remove(relative: str) -> Callable[[Path], None]:
    def mutate(run: Path) -> None:
        (run / relative).unlink()

    return mutate


def _remove_package(run: Path) -> None:
    next((run / "r17_public_v2/extraction/packages").glob("*.abipkg")).unlink()


def _corrupt_package(run: Path) -> None:
    _flip(next((run / "r18_failed_hidden/extraction/packages").glob("*.abipkg")))


def _forge_metrics(run: Path) -> None:
    path = run / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["datasets"]["r18_failed_hidden"]["metrics"]["package_functional_exact"] = 999
    value["evidence_sha256"] = evidence_hash(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _forge_rejection(run: Path) -> None:
    path = run / "r17_public_v2/control_rejection.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["packages_emitted"] = 1
    value["evidence_sha256"] = evidence_hash(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = run / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    from experiments.foreign_capability_r14.core import sha256_file

    receipt["datasets"]["r17_public_v2"]["control"]["evidence"]["sha256"] = sha256_file(path)
    receipt["evidence_sha256"] = evidence_hash(
        {key: item for key, item in receipt.items() if key != "evidence_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _invent_control_package(run: Path) -> None:
    target = run / "r18_failed_hidden/control_extraction"
    target.mkdir()
    (target / "forged.abipkg").write_text("{}", encoding="utf-8")


CASES: dict[str, Callable[[Path], None]] = {
    "missing_receipt": _remove("receipt.json"),
    "tampered_receipt": lambda run: _flip(run / "receipt.json"),
    "hash_consistent_forged_metrics": _forge_metrics,
    "missing_source_bundle": _remove("r17_public_v2/source_bundle.json"),
    "corrupt_primary_package": _corrupt_package,
    "missing_primary_package": _remove_package,
    "missing_control_rejection": _remove("r18_failed_hidden/control_rejection.json"),
    "hash_consistent_forged_control_rejection": _forge_rejection,
    "invented_control_package": _invent_control_package,
    "missing_evaluation": _remove("r17_public_v2/evaluation.jsonl"),
    "corrupt_manifest": lambda run: _flip(run / "r18_failed_hidden/extraction/manifest.json"),
    "missing_mountinfo": _remove("r17_public_v2/extraction/mountinfo.txt"),
}


def audit(
    r17_source: Path,
    r18_config: Path,
    r18_reveal: Path,
    r18_source: Path,
    run_dir: Path,
    work_root: Path,
) -> dict[str, Any]:
    baseline = verify(r17_source, r18_config, r18_reveal, r18_source, run_dir)
    outcomes = []
    if work_root.exists():
        raise R14Error(f"immutable R19 hostile work directory exists: {work_root}")
    work_root.mkdir(parents=True)
    for name, mutation in CASES.items():
        candidate = work_root / f"{len(outcomes):02d}"
        shutil.copytree(run_dir, candidate)
        mutation(candidate)
        rejected = False
        try:
            verify(
                r17_source,
                r18_config,
                r18_reveal,
                r18_source,
                candidate,
                _preverified_sources=True,
            )
        except (
            R14Error,
            R19PackageError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ):
            rejected = True
        outcomes.append({"case": name, "rejected": rejected})
        print(f"{name}: {'REJECTED' if rejected else 'ACCEPTED'}", flush=True)
    if not all(item["rejected"] for item in outcomes):
        raise R14Error("R19 hostile audit accepted a mutation")
    result = {
        "format": "abi-r19-development-hostile-audit/1",
        "verdict": "PASS",
        "cases": outcomes,
        "cases_passed": len(outcomes),
        "cases_total": len(outcomes),
        "preverified_source_strict_evidence_sha256": baseline["evidence_sha256"],
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r17-source", type=Path, required=True)
    parser.add_argument("--r18-config", type=Path, required=True)
    parser.add_argument("--r18-reveal", type=Path, required=True)
    parser.add_argument("--r18-source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.r17_source,
        args.r18_config,
        args.r18_reveal,
        args.r18_source,
        args.run_dir,
        args.work_root,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
