"""Run the one authorized B100 parent-exposure normalization."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

from . import capability_compiler_phase4_abi_lineage as lineage
from .capability_compiler_phase3 import Phase3Error


STEP_TARGETS = {
    "ABI_CAPABILITY_COMPILER_PHASE3_QUALIFIED_TRANSITION_CONTROL_PROTOCOL_V440.json": 8750,
    "ABI_CAPABILITY_COMPILER_PHASE3_COPY_BALANCED_TRANSITION_PROTOCOL_V458.json": 2188,
    "ABI_CAPABILITY_COMPILER_PHASE3_TOKEN_SUBSTRATE_CONFORMANCE_PROTOCOL_V462.json": 2188,
}


def _patched_run(root: Path, protocol_path: Path, *, output: Path | None = None) -> dict[str, Any]:
    original = lineage._json
    governance = original(protocol_path)
    predecessor = root / governance["predecessor_protocol"]
    lineage.preflight(root, predecessor)
    predecessor_governance = original(predecessor)
    for field in ("budget_manifest", "seeds", "budgets", "teacher_artifacts", "base_protocols"):
        if governance[field] != predecessor_governance[field]:
            raise Phase3Error(f"normalized B100 changed predecessor field: {field}")
    changed: dict[str, tuple[int, int]] = {}

    def normalized(path: Path) -> dict[str, Any]:
        value = original(path)
        target = STEP_TARGETS.get(path.name)
        if target is None:
            return value
        result = copy.deepcopy(value)
        before = int(result["training"]["steps"])
        result["training"]["steps"] = target
        changed[path.name] = (before, target)
        return result

    lineage._json = normalized
    try:
        for key in ("v443", "v459", "v463"):
            normalized(root / governance["base_protocols"][key])
        result = lineage.preflight(root, protocol_path) if output is None else lineage.train_lineage(root, protocol_path, "B100", 104729, output)
    finally:
        lineage._json = original
    expected = {
        "ABI_CAPABILITY_COMPILER_PHASE3_QUALIFIED_TRANSITION_CONTROL_PROTOCOL_V440.json": (7000, 8750),
        "ABI_CAPABILITY_COMPILER_PHASE3_COPY_BALANCED_TRANSITION_PROTOCOL_V458.json": (1750, 2188),
        "ABI_CAPABILITY_COMPILER_PHASE3_TOKEN_SUBSTRATE_CONFORMANCE_PROTOCOL_V462.json": (1750, 2188),
    }
    if changed != expected:
        raise Phase3Error(f"normalized parent-step scope changed: {changed}")
    result["parent_step_changes"] = {name: {"before": values[0], "after": values[1]} for name, values in changed.items()}
    result["intervention_scope"] = "V443_V459_V463_STEPS_ONLY"
    return result


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    result = _patched_run(root, protocol_path)
    if result.get("status") != "PASS_PHASE4_ABI_LINEAGE_PREFLIGHT" or [row["id"] for row in result["budgets"]] != ["B100"]:
        raise Phase3Error("normalized B100 preflight changed")
    result["status"] = "PASS_B100_EXPOSURE_NORMALIZED_PREFLIGHT"
    return result


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    result = _patched_run(root, protocol_path, output=output)
    # The lineage result is already immutable. Write a separate intervention receipt.
    receipt = {
        "format": "abi-capability-compiler-phase4-b100-exposure-normalized-receipt/1",
        "status": result["status"],
        "protocol_sha256": result["protocol_sha256"],
        "lineage_result": "result.json",
        "parent_step_changes": result["parent_step_changes"],
        "intervention_scope": result["intervention_scope"],
        "training_performed": True,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }
    (output / "exposure_normalized_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("train")
    run.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = root / args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else train(root, protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
