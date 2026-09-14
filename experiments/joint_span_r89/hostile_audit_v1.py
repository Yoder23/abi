"""Exercise the strict R88/R89 verifier against physical evidence mutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from experiments.foreign_capability_r14.core import evidence_hash, write_json_once


class AuditError(RuntimeError):
    pass


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _required(root: Path) -> list[Path]:
    fixed = [
        "results/joint_span_r88/candidate_v1/bridge.safetensors",
        "results/joint_span_r88/candidate_v1/metadata.json",
        "results/joint_span_r88/candidate_binding_v1.json",
        "results/joint_span_r88/source_v1/result.json",
        "results/joint_span_r88/source_v1/prior_corrected_scores.jsonl",
        "results/joint_span_r88/screen_v1/result.json",
        "results/joint_span_r88/screen_v1/evaluation.jsonl",
        "results/joint_span_r89/holdout_binding_v1.json",
        "results/joint_span_r89/source_v1/result.json",
        "results/joint_span_r89/source_v1/prior_corrected_scores.jsonl",
        "results/joint_span_r89/screen_v1/result.json",
        "results/joint_span_r89/screen_v1/evaluation.jsonl",
        "results/role_invariant_choice_r76/reasoning-role-invariant-search-v2.abix",
        "catalogs/joint_span_validation_r88_v1.json",
        "catalogs/joint_span_validation_r89_v1.json",
        "experiments/joint_span_r88/screen_host_v1.py",
        "experiments/joint_span_r88/PROTOCOL.md",
        "experiments/joint_span_r88/core_v1.py",
        "experiments/joint_span_r88/train_candidate_v1.py",
        "experiments/joint_span_r89/screen_frozen_v1.py",
        "experiments/joint_span_r89/PROTOCOL.md",
    ]
    parent = root / "results/role_invariant_choice_r76/candidate_v1"
    paths = [root / value for value in fixed]
    paths.extend(path for path in parent.iterdir() if path.is_file())
    if any(not path.is_file() for path in paths):
        raise AuditError("hostile audit prerequisite is absent")
    return sorted(set(paths))


def _materialize(root: Path, destination: Path, required: list[Path]) -> None:
    for source in required:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)


def _replace(path: Path, data: bytes) -> None:
    path.unlink()
    path.write_bytes(data)


def _delete(relative: str) -> Callable[[Path], None]:
    return lambda root: (root / relative).unlink()


def _append(relative: str) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        path = root / relative
        _replace(path, path.read_bytes() + b"\nHOSTILE_MUTATION")
    return mutate


def _flip_result(root: Path) -> None:
    path = root / "results/joint_span_r89/screen_v1/result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["gates"]["candidate_quality"] = False
    _replace(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _drop_raw_row(root: Path) -> None:
    path = root / "results/joint_span_r88/screen_v1/evaluation.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    _replace(path, b"".join(lines[:-1]))


CASES: dict[str, Callable[[Path], None]] = {
    "missing_r88_raw": _delete("results/joint_span_r88/screen_v1/evaluation.jsonl"),
    "dropped_r88_row": _drop_raw_row,
    "forged_r89_gate": _flip_result,
    "mutated_candidate_checkpoint": _append("results/joint_span_r88/candidate_v1/bridge.safetensors"),
    "mutated_parent_checkpoint": _append("results/role_invariant_choice_r76/candidate_v1/model.safetensors"),
    "mutated_imported_artifact": _append("results/role_invariant_choice_r76/reasoning-role-invariant-search-v2.abix"),
    "mutated_candidate_binding": _append("results/joint_span_r88/candidate_binding_v1.json"),
    "mutated_holdout_screen": _append("experiments/joint_span_r89/screen_frozen_v1.py"),
    "mutated_r89_catalog": _append("catalogs/joint_span_validation_r89_v1.json"),
    "missing_tokenizer": _delete("results/role_invariant_choice_r76/candidate_v1/tokenizer.json"),
}


def _invoke(module_root: Path, evidence_root: Path) -> dict:
    output = evidence_root / "verification.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.joint_span_r89.verify_r88_r89_v1",
            "--root",
            str(evidence_root),
            "--output",
            str(output),
        ],
        cwd=module_root,
        capture_output=True,
        timeout=180,
    )
    return {
        "returncode": completed.returncode,
        "stdout_sha256": _digest(completed.stdout),
        "stderr_sha256": _digest(completed.stderr),
        "receipt_created": output.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists():
        parser.error(f"immutable hostile receipt exists: {args.output}")
    required = _required(root)
    rows = []
    with tempfile.TemporaryDirectory(prefix="abi-r89-hostile-") as temporary:
        temporary_root = Path(temporary)
        baseline_root = temporary_root / "baseline"
        _materialize(root, baseline_root, required)
        baseline = _invoke(root, baseline_root)
        if baseline["returncode"] != 0 or baseline["receipt_created"] is not True:
            raise AuditError("strict verifier rejected the unmodified baseline")
        for name, mutation in CASES.items():
            case_root = temporary_root / name
            _materialize(root, case_root, required)
            mutation(case_root)
            observed = _invoke(root, case_root)
            rejected = observed["returncode"] != 0 and observed["receipt_created"] is False
            rows.append({"case": name, "rejected": rejected, **observed})
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    if not all(row["rejected"] for row in rows):
        raise AuditError("one or more hostile mutations were accepted")
    receipt = {
        "format": "abi-r88-r89-hostile-audit/1",
        "verdict": "PASS_R88_R89_HOSTILE_AUDIT",
        "baseline": baseline,
        "hostile_cases": len(rows),
        "hostile_cases_rejected": sum(row["rejected"] for row in rows),
        "cases": rows,
        "copy_method": "same-volume-hardlinks-with-unlink-before-mutation",
        "original_evidence_changed": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": "Fail-closed audit of the bounded R88/R89 verifier only.",
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
