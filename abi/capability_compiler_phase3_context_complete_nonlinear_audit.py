"""Run the fixed nonlinear audit with only the measured context cap repaired."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_fixed_budget_nonlinear_coefficient_audit as nonlinear
from .capability_compiler_phase3 import Phase3Error


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    maximum = int(protocol.get("context_complete_maximum_actions", 0))
    if maximum != 512 or protocol.get("context_change_only") is not True:
        raise Phase3Error("context-complete control boundary changed")
    original = dual._calibration_examples

    def complete(examples, *, seed, train_per_capability, validation_per_capability, maximum_tokens):
        if int(maximum_tokens) != 128:
            raise Phase3Error("historical calibration cap changed")
        return original(
            examples, seed=seed, train_per_capability=train_per_capability,
            validation_per_capability=validation_per_capability, maximum_tokens=maximum,
        )

    nonlinear.dual._calibration_examples = complete
    try:
        result = nonlinear.execute(root, protocol_path, output)
    finally:
        nonlinear.dual._calibration_examples = original
    result["context_complete_maximum_actions"] = maximum
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CONTEXT_COMPLETE_NONLINEAR_AUDIT_PROTOCOL_V369.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/context_complete_nonlinear_v370")
    args = parser.parse_args(); root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
