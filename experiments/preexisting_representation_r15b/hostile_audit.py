"""Fail-closed hostile mutation audit for R15B held-out evidence."""

from __future__ import annotations

import argparse
import hashlib
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
        raise RuntimeError(f"cannot mutate empty file: {path}")
    value[len(value) // 2] ^= 1
    path.write_bytes(value)


def _delete_bundle(run_dir: Path) -> None:
    next((run_dir / "source").glob("*.safetensors")).unlink()


def _corrupt_bundle(run_dir: Path) -> None:
    _flip(next((run_dir / "source").glob("*.safetensors")))


def _delete_package(run_dir: Path) -> None:
    next((run_dir / "packages").glob("*.abipkg")).unlink()


def _corrupt_package(run_dir: Path) -> None:
    _flip(next((run_dir / "packages").glob("*.abipkg")))


def _delete_label(run_dir: Path) -> None:
    next((run_dir / "labels").glob("*.json")).unlink()


def _corrupt_source_rows(run_dir: Path) -> None:
    _flip(run_dir / "source_observations.jsonl")


def _delete_isolated_result(run_dir: Path) -> None:
    next((run_dir / "extractions").glob("*/result.json")).unlink()


def _corrupt_recipient_rows(run_dir: Path) -> None:
    next((run_dir / "recipients").glob("*/observations.jsonl")).unlink()


def _tamper_receipt(run_dir: Path) -> None:
    path = run_dir / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["teacher_present_at_recipient_execution"] = True
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


CASES: dict[str, Callable[[Path], None]] = {
    "missing_bundle": _delete_bundle,
    "corrupt_bundle": _corrupt_bundle,
    "missing_package": _delete_package,
    "corrupt_package": _corrupt_package,
    "missing_label": _delete_label,
    "corrupt_source_rows": _corrupt_source_rows,
    "missing_isolated_result": _delete_isolated_result,
    "missing_recipient_rows": _corrupt_recipient_rows,
    "tampered_receipt": _tamper_receipt,
}


def audit(config: Path, reveal: Path, run_dir: Path) -> dict[str, Any]:
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="abi-r15b-hostile-") as raw:
        root = Path(raw)
        for name, mutate in CASES.items():
            candidate = root / name
            shutil.copytree(run_dir, candidate)
            mutate(candidate)
            rejected = False
            error = None
            try:
                verify(config, reveal, candidate)
            except (R14Error, OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
                rejected = True
                error = type(exc).__name__
            outcomes.append({"case": name, "rejected": rejected, "error": error})
        wrong_reveal = root / "wrong_reveal.json"
        wrong_secret = bytes(reversed(range(32)))
        wrong_reveal.write_text(
            json.dumps(
                {
                    "format": "abi-r15b-heldout-reveal/1",
                    "commitment": hashlib.sha256(wrong_secret).hexdigest(),
                    "secret_hex": wrong_secret.hex(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        rejected = False
        error = None
        try:
            verify(config, wrong_reveal, run_dir)
        except (R14Error, OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            rejected = True
            error = type(exc).__name__
        outcomes.append({"case": "wrong_reveal", "rejected": rejected, "error": error})
    if not all(item["rejected"] for item in outcomes):
        raise R14Error("R15B hostile audit did not fail closed")
    result = {
        "format": "abi-r15b-hostile-audit/1",
        "verdict": "PASS",
        "cases": outcomes,
        "cases_passed": len(outcomes),
        "cases_total": len(CASES) + 1,
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
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
