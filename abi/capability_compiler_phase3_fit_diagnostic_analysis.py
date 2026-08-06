"""Derive the V26 failure-owner decision from the immutable diagnostic."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_direct_core import _json
from .capability_compiler_phase3_fit_diagnostic_verifier import embedded_evidence_sha256


RESULT_SHA256 = "10103f6eceeb1fed2f627ba92ec37bbc25255d4aabf394736458d58384457871"
EVIDENCE_SHA256 = "fd8508074c3539c50869624ecf2a3a4a444b354ddf37f0929c2064050d7df15b"


def rejection_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["capability"]) for row in rows)
    return {capability: counts[capability] for capability in CAPABILITIES}


def system_summary(system: Mapping[str, Any]) -> dict[str, Any]:
    train = system["training_teacher_forced"]
    development = system["development_teacher_forced"]
    autonomous = system["training_autonomous_sample"]
    return {
        "checkpoint_sha256": system["checkpoint_sha256"],
        "training": {
            "representable": system["training_representable"],
            "rejected": len(system["training_rejected"]),
            "action_accuracy": train["action_accuracy"],
            "exact_sequence_rate": train["exact_sequence_rate"],
            "mean_action_nll": train["mean_action_nll"],
        },
        "development": {
            "representable": system["development_representable"],
            "rejected": len(system["development_rejected"]),
            "representable_rate": system["development_representable"] / 1400,
            "action_accuracy_on_representable": development["action_accuracy"],
            "exact_sequence_rate_on_representable": development["exact_sequence_rate"],
            "mean_action_nll_on_representable": development["mean_action_nll"],
            "rejected_by_capability": rejection_counts(system["development_rejected"]),
        },
        "autonomous_training_sample": {
            "observations": autonomous["observations"],
            "exact_response_bytes": autonomous["exact_response_bytes"],
            "exact_response_rate": autonomous["exact_response_bytes"] / autonomous["observations"],
            "mean_common_prefix_fraction": autonomous["mean_common_prefix_fraction"],
            "repetition_collapses": autonomous["repetition_collapses"],
            "generation_errors": autonomous["generation_errors"],
        },
        "classification": system["classification"],
    }


def analyze(root: Path, result_path: Path) -> dict[str, Any]:
    if sha256_file(result_path) != RESULT_SHA256:
        raise Phase3Error("V26 result file identity changed")
    source = _json(result_path)
    if embedded_evidence_sha256(source) != EVIDENCE_SHA256 or source.get("evidence_sha256") != EVIDENCE_SHA256:
        raise Phase3Error("V26 embedded evidence identity changed")
    systems = {name: system_summary(source["systems"][name]) for name in ("V23", "V24")}
    if not all(value["classification"]["train_fit_or_capacity_limit"] for value in systems.values()):
        raise Phase3Error("V26 primary failure owner changed")
    if source["ownership"].get("layercake_host_regression") is not False:
        raise Phase3Error("V26 LayerCake boundary changed")
    decision: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-fit-diagnostic-decision/1",
        "status": "COMPLETE_FAILED_ARCHITECTURES_CLOSED",
        "source": {
            "path": result_path.relative_to(root).as_posix(),
            "file_sha256": RESULT_SHA256,
            "evidence_sha256": EVIDENCE_SHA256,
        },
        "systems": systems,
        "primary_failure_owner": "ABI_MODEL_FIT_CAPACITY_OPTIMIZATION_OR_TARGET_REPRESENTATION",
        "secondary_failure_owner": "ABI_HELD_OUT_REPRESENTABILITY_AND_GENERALIZATION",
        "layercake_boundary": {
            "regression_identified": False,
            "current_v2_host_changed": False,
            "quality_or_performance_inherited": False,
            "note": "V26 replays sealed v1 private checkpoint graphs. The separately certified v2 host is construct-only and was not a quality variable."
        },
        "decision": {
            "v23_closed": True,
            "v24_closed": True,
            "phase3_certified": False,
            "phase4_through_8": "LOCKED",
            "training_authorized_by_this_decision": False,
            "next_bounded_work": "Preregister one no-training Unicode-atomic open-vocabulary representation bake-off. Require 100% lossless target representability on the 7,000 training and 1,400 development teacher targets before any new fit run.",
            "prohibited": [
                "nearby fixed-vocabulary or pointer-weight variants",
                "data expansion before representability and training fit are repaired",
                "LayerCake host modification without a separately measured host defect",
                "final-test access",
                "ABI superiority claims"
            ]
        },
        "claim_boundary": "This decision attributes two failed ABI checkpoint designs. It is not a Phase 3 certificate and does not compare ABI with LoRA or distillation."
    }
    decision["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(decision)).hexdigest()
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "verify"))
    parser.add_argument("--result", default="results/abi_capability_compiler_phase3_fit_diagnostic/fit_generalization_v26.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_fit_diagnostic/fit_decision_v26.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    expected = analyze(root, (root / args.result).resolve())
    output = (root / args.output).resolve()
    if args.command == "write":
        if output.exists():
            raise Phase3Error(f"V26 decision is immutable: {output}")
        _write_immutable(output, json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    elif _json(output) != expected:
        raise Phase3Error("stored V26 decision differs from recomputation")
    print(json.dumps({"status": "PASS", "evidence_sha256": expected["evidence_sha256"], "phase3_certified": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
