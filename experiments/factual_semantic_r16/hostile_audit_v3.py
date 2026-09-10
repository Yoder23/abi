"""Hostile checks for the R16 v3 live-file and public-evidence repairs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    write_json_once,
)

from .hostile_audit_v2 import audit_v2
from .verify_strict_v3 import verify_v3


def _flip(path: Path) -> None:
    value = bytearray(path.read_bytes())
    if not value:
        raise RuntimeError("cannot mutate empty R16 evidence")
    value[len(value) // 2] ^= 1
    path.write_bytes(value)


def _missing_live_file(live_dir: Path, _public: Path) -> None:
    (live_dir / "source_bundle.json").unlink()


def _corrupt_live_package(live_dir: Path, _public: Path) -> None:
    _flip(next((live_dir / "extraction/packages").glob("*.abipkg")))


def _corrupt_live_extraction(live_dir: Path, _public: Path) -> None:
    _flip(live_dir / "extraction/result.json")


def _missing_public_artifact(_live_dir: Path, public: Path) -> None:
    (public / "observations.jsonl").unlink()


def _corrupt_public_residual(_live_dir: Path, public: Path) -> None:
    _flip(public / "prompt_end_residuals.safetensors")


def _forge_public_metrics(_live_dir: Path, public: Path) -> None:
    path = public / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["metrics"]["open_exact"] = 47
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    value["evidence_sha256"] = evidence_hash(scientific)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


CASES: dict[str, Callable[[Path, Path], None]] = {
    "missing_actual_live_file": _missing_live_file,
    "corrupt_actual_live_package": _corrupt_live_package,
    "corrupt_actual_live_extraction_result": _corrupt_live_extraction,
    "missing_public_requalification_artifact": _missing_public_artifact,
    "corrupt_public_requalification_residual": _corrupt_public_residual,
    "hash_consistent_forged_public_metrics": _forge_public_metrics,
}


def audit_v3(
    config: Path,
    reveal: Path,
    run_dir: Path,
    live_path: Path,
    public_requalification: Path,
) -> dict[str, Any]:
    inherited = audit_v2(config, reveal, run_dir, live_path)
    base = verify_v3(config, reveal, run_dir, live_path, public_requalification)
    repository = config.resolve().parents[3]
    source_inventory = json_object(
        repository
        / "results/preexisting_representation_r15b/heldout_v1_live_v2/receipt.json"
    )["source_snapshot"]
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="abi-r16-hostile-v3-") as raw:
        root = Path(raw)
        for name, mutation in CASES.items():
            live_dir = root / f"{name}-live"
            public = root / f"{name}-public"
            shutil.copytree(live_path.parent, live_dir)
            shutil.copytree(public_requalification, public)
            mutation(live_dir, public)
            try:
                verify_v3(
                    config,
                    reveal,
                    run_dir,
                    live_dir / live_path.name,
                    public,
                    source_inventory=source_inventory,
                )
                rejected = False
            except (R14Error, OSError, ValueError, KeyError, json.JSONDecodeError):
                rejected = True
            outcomes.append({"case": name, "rejected": rejected})
    if not all(item["rejected"] for item in outcomes):
        raise R14Error("R16 v3 hostile audit accepted a mutation")
    result = {
        "format": "abi-r16-hostile-audit/3",
        "verdict": "PASS",
        "inherited_v2_cases_passed": inherited["cases_passed"],
        "v3_cases": outcomes,
        "v3_cases_passed": len(outcomes),
        "cases_passed": inherited["cases_passed"] + len(outcomes),
        "cases_total": inherited["cases_total"] + len(outcomes),
        "base_strict_evidence_sha256": base["evidence_sha256"],
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--public-requalification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_v3(
        args.config,
        args.reveal,
        args.run_dir,
        args.live,
        args.public_requalification,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
