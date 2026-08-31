"""Run fail-closed hostile controls against a completed R11 evidence tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .core import R11Error
from .verify import R11VerificationError, verify


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise R11Error(f"immutable hostile report exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _flip(path: Path) -> None:
    value = bytearray(path.read_bytes())
    value[len(value) // 2] ^= 1
    path.write_bytes(value)


def _missing_receipt(run_dir: Path) -> None:
    (run_dir / "receipt.json").unlink()


def _missing_package(run_dir: Path) -> None:
    next((run_dir / "packages").glob("*.abipkg")).unlink()


def _corrupt_package(run_dir: Path) -> None:
    _flip(next((run_dir / "packages").glob("*.abipkg")))


def _missing_teacher_rows(run_dir: Path) -> None:
    (run_dir / "teacher_observations.jsonl").unlink()


def _corrupt_recipient_rows(run_dir: Path) -> None:
    _flip(run_dir / "recipient_observations.jsonl")


def _stale_receipt_claim(run_dir: Path) -> None:
    path = run_dir / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["recipient_execution"]["teacher_loaded_in_recipient_process"] = True
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


RUN_CASES: dict[str, Callable[[Path], None]] = {
    "missing_receipt": _missing_receipt,
    "missing_package": _missing_package,
    "corrupt_package": _corrupt_package,
    "missing_teacher_rows": _missing_teacher_rows,
    "corrupt_recipient_rows": _corrupt_recipient_rows,
    "stale_receipt_claim": _stale_receipt_claim,
}


def run_hostile(config_path: Path, reveal_path: Path, run_dir: Path, output: Path) -> dict:
    baseline = verify(config_path, reveal_path, run_dir)
    if baseline["verdict"] != "PASS":
        raise R11Error("baseline evidence does not verify before hostile controls")
    results = []
    with tempfile.TemporaryDirectory(prefix="abi-r11-hostile-") as temporary:
        scratch = Path(temporary)
        for name, mutate in RUN_CASES.items():
            case_dir = scratch / name
            shutil.copytree(run_dir, case_dir)
            mutate(case_dir)
            try:
                verify(config_path, reveal_path, case_dir)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                results.append(
                    {
                        "case": name,
                        "rejected": True,
                        "exception": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            else:
                raise R11Error(f"hostile case did not fail closed: {name}")

        bad_reveal = scratch / "bad_reveal.json"
        bad_reveal.write_text(
            json.dumps({"commitment": "0" * 64, "secret_hex": "0" * 64}, sort_keys=True),
            encoding="utf-8",
        )
        try:
            verify(config_path, bad_reveal, run_dir)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            results.append(
                {
                    "case": "wrong_heldout_reveal",
                    "rejected": True,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }
            )
        else:
            raise R11Error("wrong held-out reveal did not fail closed")

    report = {
        "format": "abi-native-neural-isa-r11-hostile-controls/1",
        "baseline_evidence_sha256": baseline["evidence_sha256"],
        "cases": results,
        "case_count": len(results),
        "rejected_count": sum(int(item["rejected"]) for item in results),
        "verdict": "PASS" if all(item["rejected"] for item in results) else "FAIL",
    }
    report["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    if report["verdict"] != "PASS":
        raise R11VerificationError("one or more hostile controls did not reject")
    _write_once(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run_hostile(
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
