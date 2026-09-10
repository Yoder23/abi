"""Expanded fail-closed hostile audit for the additive R16 v2 verifier."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, evidence_hash, write_json_once

from .hostile_audit import CASES as BASE_CASES
from .verify_strict_v2 import verify_v2


def _flip(path: Path) -> None:
    value = bytearray(path.read_bytes())
    if not value:
        raise RuntimeError("cannot mutate empty R16 evidence")
    value[len(value) // 2] ^= 1
    path.write_bytes(value)


def _missing_residual(run: Path, _live: Path) -> None:
    (run / "source_prompt_end_residuals.safetensors").unlink()


def _corrupt_rotated_bundle(run: Path, _live: Path) -> None:
    _flip(run / "rotated_score_bundle.json")


def _missing_extraction_result(run: Path, _live: Path) -> None:
    (run / "extraction/result.json").unlink()


def _corrupt_control_package(run: Path, _live: Path) -> None:
    _flip(next((run / "control_extraction/packages").glob("*.abipkg")))


def _forged_source_boundary(run: Path, _live: Path) -> None:
    path = run / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source"]["present_at_package_execution"] = True
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    value["evidence_sha256"] = evidence_hash(scientific)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _forged_live_hash(_run: Path, live: Path) -> None:
    value = json.loads(live.read_text(encoding="utf-8"))
    value["files_replayed_byte_exact"][0]["sha256"] = "0" * 64
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    value["evidence_sha256"] = evidence_hash(scientific)
    live.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


EXTRA_CASES: dict[str, Callable[[Path, Path], None]] = {
    "missing_residual": _missing_residual,
    "corrupt_rotated_bundle": _corrupt_rotated_bundle,
    "missing_extraction_result": _missing_extraction_result,
    "corrupt_control_package": _corrupt_control_package,
    "hash_consistent_source_boundary": _forged_source_boundary,
    "hash_consistent_live_hash": _forged_live_hash,
}


def audit_v2(
    config: Path, reveal: Path, run_dir: Path, live_path: Path
) -> dict[str, Any]:
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="abi-r16-hostile-v2-") as raw:
        root = Path(raw)
        for name, mutation in BASE_CASES.items():
            candidate = root / f"base-{name}"
            shutil.copytree(run_dir, candidate)
            live = root / f"base-{name}-live.json"
            shutil.copy2(live_path, live)
            mutation(candidate)
            try:
                verify_v2(config, reveal, candidate, live)
                rejected = False
            except (R14Error, OSError, ValueError, KeyError, json.JSONDecodeError):
                rejected = True
            outcomes.append({"case": name, "rejected": rejected})
        for name, mutation in EXTRA_CASES.items():
            candidate = root / name
            shutil.copytree(run_dir, candidate)
            live = root / f"{name}-live.json"
            shutil.copy2(live_path, live)
            mutation(candidate, live)
            try:
                verify_v2(config, reveal, candidate, live)
                rejected = False
            except (R14Error, OSError, ValueError, KeyError, json.JSONDecodeError):
                rejected = True
            outcomes.append({"case": name, "rejected": rejected})
        wrong_reveal = root / "wrong-reveal.json"
        wrong_reveal.write_text(
            json.dumps(
                {
                    "format": "abi-r16-heldout-reveal/1",
                    "commitment": "0" * 64,
                    "secret_hex": "00" * 32,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            verify_v2(config, wrong_reveal, run_dir, live_path)
            rejected = False
        except (R14Error, OSError, ValueError, KeyError, json.JSONDecodeError):
            rejected = True
        outcomes.append({"case": "wrong_reveal", "rejected": rejected})
    if not all(item["rejected"] for item in outcomes):
        raise R14Error("R16 v2 hostile audit accepted a mutation")
    result = {
        "format": "abi-r16-hostile-audit/2",
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
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_v2(args.config, args.reveal, args.run_dir, args.live)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
