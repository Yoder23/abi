"""Emit the R10 negative verdict while preserving its passing component gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .runtime import sha256_file
from .verify_repaired_v3 import (
    ORIGINAL_COUNT_EXPRESSION,
    ORIGINAL_TOLERANCE_EXPRESSION,
    ORIGINAL_VERIFIER_SHA256,
    REPAIRED_COUNT_EXPRESSION,
    REPAIRED_TOLERANCE_EXPRESSION,
    R10VerifierRepairError,
)

ORIGINAL_FAIL_TAIL = """    if not passed:
        _fail("one or more recomputed R10 gates failed")
    return result"""
NEGATIVE_REPORT_TAIL = "    return result"


def _reporting_verifier(root: Path) -> Any:
    path = root / "experiments/copy_paste_r10/verify.py"
    if sha256_file(path) != ORIGINAL_VERIFIER_SHA256:
        raise R10VerifierRepairError("frozen v1 verifier changed")
    source = path.read_text(encoding="utf-8")
    replacements = (
        (ORIGINAL_COUNT_EXPRESSION, REPAIRED_COUNT_EXPRESSION),
        (ORIGINAL_TOLERANCE_EXPRESSION, REPAIRED_TOLERANCE_EXPRESSION),
        (ORIGINAL_FAIL_TAIL, NEGATIVE_REPORT_TAIL),
    )
    for original, repaired in replacements:
        if source.count(original) != 1:
            raise R10VerifierRepairError("negative-report verifier boundary changed")
        source = source.replace(original, repaired)
    namespace = {
        "__file__": str(path),
        "__name__": "experiments.copy_paste_r10._negative_report_impl",
        "__package__": "experiments.copy_paste_r10",
    }
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["verify"]


def report(config_path: Path, run_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    result = _reporting_verifier(root)(config_path, run_dir)
    if result.get("verdict") != "FAIL":
        raise R10VerifierRepairError("R10 unexpectedly ceased to be a negative result")
    gates = result["recomputed_gates"]
    component_pass = bool(
        gates["recipient_matrix"]
        and gates["temporal_separation"]
        and gates["same_package_reuse"]
        and not gates["source_learning"]
    )
    if not component_pass:
        raise R10VerifierRepairError("registered R10 component interpretation changed")
    source_after = {
        capability_id: values["AFTER"]["accuracy"]
        for capability_id, values in result["source_metrics"].items()
    }
    result.pop("evidence_sha256", None)
    result.update(
        {
            "format": "abi-copy-paste-r10-negative-report/1",
            "verdict": "FAIL_SOURCE_NATIVE_GENERALIZATION",
            "runtime_copy_paste_component": "PASS_BOUNDED_SYNTHETIC",
            "source_native_after_accuracy_range": [
                min(source_after.values()),
                max(source_after.values()),
            ],
            "interpretation": (
                "The identical source-extracted packages execute their canonical transition "
                "semantics exactly on all frozen recipients, but the source model's native "
                "decoder does not meet the registered out-of-sample composed-prompt gate."
            ),
            "claim_boundary": (
                "Supports only runtime-owned copy/paste of this synthetic canonical IR. "
                "Does not support lossless source behavior, native neural transplantation, "
                "LayerCake integration, English/domain extraction, or teacher parity."
            ),
        }
    )
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = report(Path(args.config).resolve(), Path(args.run_dir).resolve())
        output = Path(args.output).resolve()
        if output.exists():
            raise R10VerifierRepairError(f"immutable negative report exists: {output}")
        output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
