"""Hostile mutation audit for the frozen R19 held-out verifier."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from experiments.foreign_capability_r14.core import R14Error, evidence_hash, write_json_once

from .package import R19PackageError
from .verify_heldout import verify

Target = Literal["config", "reveal", "source", "run"]


@dataclass(frozen=True)
class Case:
    target: Target
    mutate: Callable[[Path], None]


def _flip(path: Path) -> None:
    value = bytearray(path.read_bytes())
    if not value:
        raise RuntimeError("cannot mutate empty R19 held-out evidence")
    value[len(value) // 2] ^= 1
    path.write_bytes(value)


def _remove(relative: str) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        (root / relative).unlink()

    return mutate


def _forge_metrics(run: Path) -> None:
    path = run / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["dataset"]["metrics"]["package_functional_exact"] = 999
    value["evidence_sha256"] = evidence_hash(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _forge_accounting(run: Path) -> None:
    path = run / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["information_accounting"]["final_package_bytes"] += 1
    value["evidence_sha256"] = evidence_hash(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _forge_control_rejection(run: Path) -> None:
    rejection_path = run / "hidden/control_rejection.json"
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    rejection["packages_emitted"] = 1
    rejection["evidence_sha256"] = evidence_hash(
        {key: item for key, item in rejection.items() if key != "evidence_sha256"}
    )
    rejection_path.write_text(
        json.dumps(rejection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_path = run / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    from experiments.foreign_capability_r14.core import sha256_file

    receipt["dataset"]["control"]["evidence"]["sha256"] = sha256_file(rejection_path)
    receipt["evidence_sha256"] = evidence_hash(
        {key: item for key, item in receipt.items() if key != "evidence_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remove_package(run: Path) -> None:
    next((run / "hidden/extraction/packages").glob("*.abipkg")).unlink()


def _invent_control_package(run: Path) -> None:
    target = run / "hidden/control_extraction"
    target.mkdir()
    (target / "forged.abipkg").write_text("{}", encoding="utf-8")


def _forge_source_metrics(source: Path) -> None:
    path = source / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["metrics"]["evaluation_functional_exact"] = 999
    value["evidence_sha256"] = evidence_hash(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _alter_config(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source"]["model_id"] = "forged/source"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


CASES: dict[str, Case] = {
    "missing_config": Case("config", lambda path: path.unlink()),
    "altered_config": Case("config", _alter_config),
    "missing_reveal": Case("reveal", lambda path: path.unlink()),
    "corrupt_reveal": Case("reveal", _flip),
    "missing_source_receipt": Case("source", _remove("receipt.json")),
    "missing_source_rows": Case("source", _remove("source_observations.jsonl")),
    "corrupt_source_rows": Case("source", lambda root: _flip(root / "source_observations.jsonl")),
    "missing_source_strict": Case("source", _remove("strict_verification.json")),
    "hash_consistent_forged_source_metrics": Case("source", _forge_source_metrics),
    "missing_receipt": Case("run", _remove("receipt.json")),
    "tampered_receipt": Case("run", lambda root: _flip(root / "receipt.json")),
    "hash_consistent_forged_metrics": Case("run", _forge_metrics),
    "hash_consistent_forged_accounting": Case("run", _forge_accounting),
    "missing_source_bundle": Case("run", _remove("hidden/source_bundle.json")),
    "missing_primary_package": Case("run", _remove_package),
    "corrupt_primary_package": Case(
        "run", lambda root: _flip(next((root / "hidden/extraction/packages").glob("*.abipkg")))
    ),
    "missing_control_rejection": Case("run", _remove("hidden/control_rejection.json")),
    "hash_consistent_forged_control_rejection": Case("run", _forge_control_rejection),
    "invented_control_package": Case("run", _invent_control_package),
    "missing_evaluation": Case("run", _remove("hidden/evaluation.jsonl")),
    "corrupt_manifest": Case("run", lambda root: _flip(root / "hidden/extraction/manifest.json")),
    "missing_mountinfo": Case("run", _remove("hidden/extraction/mountinfo.txt")),
}


def audit(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    run_dir: Path,
    work_root: Path,
) -> dict[str, Any]:
    baseline = verify(config_path, reveal_path, source_run, run_dir)
    if baseline.get("verdict") != "PASS":
        raise R14Error("R19 held-out hostile audit requires a passing baseline")
    if work_root.exists():
        raise R14Error(f"immutable R19 held-out hostile work directory exists: {work_root}")
    work_root.mkdir(parents=True)
    outcomes = []
    for index, (name, case) in enumerate(CASES.items()):
        candidate = work_root / f"{index:02d}"
        candidate.mkdir()
        config = candidate / "config.json"
        reveal = candidate / "reveal.json"
        source = candidate / "source"
        run = candidate / "run"
        shutil.copy2(config_path, config)
        shutil.copy2(reveal_path, reveal)
        shutil.copytree(source_run, source)
        shutil.copytree(run_dir, run)
        target: Path = {"config": config, "reveal": reveal, "source": source, "run": run}[
            case.target
        ]
        case.mutate(target)
        rejected = False
        try:
            verify(
                config,
                reveal,
                source,
                run,
                _preverified_source=case.target == "run",
            )
        except (
            R14Error,
            R19PackageError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            rejected = True
        outcomes.append({"case": name, "rejected": rejected})
        print(f"{name}: {'REJECTED' if rejected else 'ACCEPTED'}", flush=True)
    if not all(item["rejected"] for item in outcomes):
        raise R14Error("R19 held-out hostile audit accepted a mutation")
    result = {
        "format": "abi-r19-heldout-hostile-audit/1",
        "verdict": "PASS",
        "cases": outcomes,
        "cases_passed": len(outcomes),
        "cases_total": len(outcomes),
        "baseline_strict_evidence_sha256": baseline["evidence_sha256"],
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.config,
        args.reveal,
        args.source_run,
        args.run_dir,
        args.work_root,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
