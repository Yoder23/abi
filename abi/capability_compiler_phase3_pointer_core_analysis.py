"""Recompute V24 gates and its preregistered paired V24-minus-V23 diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, repetition_collapse, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_analysis import stratified_bootstrap, wilson


DECISION_FORMAT = "abi-capability-compiler-phase3-pointer-core-decision/1"
PROTOCOL_FORMAT = "abi-capability-compiler-phase3-pointer-core-screen/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise Phase3Error(f"expected JSONL objects: {path}")
    return rows


def _verify_rows(root: Path, catalog: Path, outputs: Path, receipt: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if sha256_file(outputs) != receipt.get("outputs_sha256"):
        raise Phase3Error("V24 output binding failed")
    rows = _jsonl(outputs)
    probes = development_probes(catalog)
    expected = {str(probe["probe_id"]): probe for probe in probes}
    ids = [str(row.get("probe_id")) for row in rows]
    if len(rows) != 1400 or len(set(ids)) != 1400 or set(ids) != set(expected):
        raise Phase3Error("V24 prompt identity or depth changed")
    for row in rows:
        probe = expected[str(row["probe_id"])]
        if row.get("capability") != probe["canonical_capability"]:
            raise Phase3Error("V24 capability identity changed")
        if row.get("functional_pass") is not evaluate_functional(str(row.get("output", "")), probe["evaluator"]):
            raise Phase3Error("V24 functional score differs from raw-output recomputation")
        if row.get("repetition_collapse") is not repetition_collapse(str(row.get("output", ""))):
            raise Phase3Error("V24 collapse score differs from raw-output recomputation")
        if row.get("generation_error") is not None and (not isinstance(row["generation_error"], str) or row.get("output") != ""):
            raise Phase3Error("V24 generation-error evidence changed")
    return rows, {str(row["probe_id"]): row for row in rows}


def build_decision(*, root: Path, protocol_path: Path, candidate_dir: Path, evaluation_dir: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    metadata_path = candidate_dir / "metadata.json"
    receipt_path = evaluation_dir / "receipt.json"
    outputs_path = evaluation_dir / "development_outputs.jsonl"
    metadata = _json(metadata_path)
    receipt = _json(receipt_path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status") != "PREREGISTERED_CONDITIONAL_ABSOLUTE_SCREEN"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("promotion_eligible") is not False
        or metadata.get("protocol_sha256") != protocol_sha
        or receipt.get("protocol_sha256") != protocol_sha
        or metadata.get("final_test_accessed") is not False
        or receipt.get("final_test_accessed") is not False
        or metadata.get("promotion_eligible") is not False
        or receipt.get("promotion_eligible") is not False
        or metadata.get("teacher_present_at_inference") is not False
        or metadata.get("source_blocks_retained") != 0
        or metadata.get("imported_information", {}).get("source_parameters_copied") != 0
        or metadata.get("imported_information", {}).get("stored_logits") != 0
        or metadata.get("imported_information", {}).get("stored_activations") != 0
    ):
        raise Phase3Error("V24 governance or ownership identity failed")
    for name, document in (("model.safetensors", metadata["checkpoint"]), ("tokenizer.json", metadata["tokenizer"]), ("model_config.json", metadata["model_config"])):
        path = candidate_dir / name
        if not path.is_file() or sha256_file(path) != document["sha256"]:
            raise Phase3Error(f"V24 candidate binding failed: {name}")
    if (candidate_dir / "model.safetensors").stat().st_size != metadata["checkpoint"]["bytes"]:
        raise Phase3Error("V24 checkpoint byte count changed")
    rows, raw = _verify_rows(root, (root / protocol["development_catalog"]).resolve(), outputs_path, receipt)

    v23_metadata_path = (root / protocol["v23_reference"]["metadata"]).resolve()
    v23_outputs_path = (root / protocol["v23_reference"]["outputs"]).resolve()
    v23_metadata = _json(v23_metadata_path)
    v23_rows = _jsonl(v23_outputs_path)
    v23_raw = {str(row["probe_id"]): row for row in v23_rows}
    if (
        metadata["training"]["record_sequence_sha256"] != v23_metadata["training"]["record_sequence_sha256"]
        or metadata["seed"] != v23_metadata["seed"]
        or metadata["model_config"]["sha256"] != v23_metadata["model_config"]["sha256"]
        or metadata["tokenizer"]["sha256"] != v23_metadata["tokenizer"]["sha256"]
        or metadata["model_config"]["trainable_parameters"] != v23_metadata["model_config"]["trainable_parameters"]
        or metadata["imported_information"] != v23_metadata["imported_information"]
        or set(raw) != set(v23_raw)
    ):
        raise Phase3Error("V24-to-V23 matched representation comparison identity failed")

    per_capability: dict[str, Any] = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passes = sum(row["functional_pass"] is True for row in values)
        collapses = sum(row["repetition_collapse"] is True for row in values)
        if len(values) != 100:
            raise Phase3Error(f"V24 capability depth changed: {capability}")
        per_capability[capability] = {"passes": passes, "observations": 100, "collapses": collapses, "wilson": wilson(passes, 100)}
    functional = sum(row["functional_pass"] is True for row in rows)
    collapses = sum(row["repetition_collapse"] is True for row in rows)
    errors = sum(row.get("generation_error") is not None for row in rows)
    per_receipt = {name: {key: value[key] for key in ("passes", "collapses", "observations")} for name, value in per_capability.items()}
    for key, value in {"observations": 1400, "functional_passes": functional, "repetition_collapses": collapses, "generation_errors": errors, "per_capability": per_receipt}.items():
        if receipt.get(key) != value:
            raise Phase3Error(f"V24 receipt differs from raw evidence: {key}")

    thresholds = protocol["absolute_screen"]
    ordinary = all(value["wilson"]["point"] >= thresholds["per_capability_functional_point_estimate_minimum"] and value["wilson"]["lower_95"] >= thresholds["per_capability_functional_wilson_lower_minimum"] for value in per_capability.values())
    critical = all(per_capability[name]["wilson"]["point"] >= thresholds["prompt_grounding_instruction_adherence_and_abstention_point_minimum"] and per_capability[name]["wilson"]["lower_95"] >= thresholds["prompt_grounding_instruction_adherence_and_abstention_wilson_lower_minimum"] for name in ("prompt_grounding", "instruction_following", "abstention"))
    gates = {
        "per_capability_functional": ordinary,
        "critical_capabilities": critical,
        "zero_repetition_collapses": collapses <= thresholds["repetition_collapse_count_maximum"],
        "zero_generation_errors": errors <= thresholds["generation_error_count_maximum"],
    }
    if all(gates.values()):
        raise Phase3Error("V24 unexpectedly reaches teacher-relative evaluation")
    comparison = stratified_bootstrap(raw, v23_raw, replicates=int(protocol["diagnostic_comparison_if_screen_fails"]["replicates"]), seed=int(protocol["diagnostic_comparison_if_screen_fails"]["seed"]))
    v23_functional = sum(row["functional_pass"] is True for row in v23_rows)
    v23_collapses = sum(row["repetition_collapse"] is True for row in v23_rows)
    error_types = Counter(row["generation_error"].split(":", 1)[0] for row in rows if row.get("generation_error") is not None)
    result: dict[str, Any] = {
        "format": DECISION_FORMAT,
        "status": "FAIL_ABSOLUTE_QUALITY_POINTER_REPRESENTATION_CLOSED_HOST_UTF8_GAP_EXPOSED",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED",
        "phase3_certified": False,
        "phase4_through_8": "LOCKED",
        "candidate": {
            "system": "P0",
            "seed": metadata["seed"],
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "checkpoint_bytes": metadata["checkpoint"]["bytes"],
            "trainable_parameters": metadata["model_config"]["trainable_parameters"],
            "functional_passes": functional,
            "observations": 1400,
            "functional_wilson": wilson(functional, 1400),
            "repetition_collapses": collapses,
            "generation_errors": errors,
            "generation_error_types": dict(sorted(error_types.items())),
            "best_case_if_every_error_became_a_pass": {"passes": functional + errors, "observations": 1400, "point": (functional + errors) / 1400},
            "per_capability": per_capability,
            "training": metadata["training"],
            "representation": metadata["representation"],
            "imported_information": metadata["imported_information"],
            "teacher_present_at_inference": False,
            "source_blocks_retained": 0,
        },
        "matched_v23_diagnostic": {
            "v24_functional_passes": functional,
            "v23_functional_passes": v23_functional,
            "pass_delta": functional - v23_functional,
            "paired_stratified_bootstrap": comparison,
            "v24_repetition_collapses": collapses,
            "v23_repetition_collapses": v23_collapses,
            "collapse_delta": collapses - v23_collapses,
            "v24_generation_errors": errors,
            "v23_generation_errors": sum(row.get("generation_error") is not None for row in v23_rows),
            "identical_seed": True,
            "identical_record_sequence": True,
            "identical_architecture_and_parameters": True,
            "identical_fixed_vocabulary": True,
            "identical_imported_information": True,
            "only_intended_change": "pointer-supervised target representation",
            "promotion_effect": "NONE",
        },
        "gates": {
            **gates,
            "absolute_screen_pass": all(gates.values()),
            "teacher_relative_noninferiority": "NOT_REACHED_PREREGISTERED_EARLY_STOP",
            "matched_causal_controls": "NOT_REACHED_PREREGISTERED_EARLY_STOP",
            "remaining_paired_seeds": "PROHIBITED_BY_DECISION_RULE",
            "final_test_accessed": False,
        },
        "ownership": {
            "abi_acquisition_or_representation_failure": True,
            "layercake_host_regression": False,
            "layercake_host_construct_status": "INDEPENDENT_PASS_CONSTRUCT_ONLY",
            "layercake_utf8_validity_gap_exposed": True,
            "utf8_gap_observation": "The byte-lexeme action surface can autonomously assemble invalid UTF-8 sequences from independently addressable multibyte fragments; 31 outputs were discarded by strict UTF-8 decoding.",
            "utf8_gap_not_sufficient_to_explain_failure": True,
            "reason": "Even granting a functional pass to all 31 decoding errors yields only 632/1,400, far below every-capability gates; ABI planning and collapse failures remain decisive."
        },
        "decision": {
            "v24_promoted": False,
            "v24_closed": True,
            "controls_run": False,
            "remaining_seeds_run": False,
            "reason": "Pointer supervision increased aggregate passes by 97 but failed every discriminating absolute family, increased collapses by 62, introduced 31 generation errors, and reduced prompt grounding from 50/100 to 39/100.",
            "next_step": "Do not train another pointer-core variant. Preserve V24, audit the exposed Unicode-validity contract separately in LayerCake, and require a new measured acquisition bottleneck plus fresh governance before any ABI successor."
        },
        "evidence": {
            "metadata_sha256": sha256_file(metadata_path),
            "checkpoint_sha256": sha256_file(candidate_dir / "model.safetensors"),
            "model_config_sha256": sha256_file(candidate_dir / "model_config.json"),
            "tokenizer_sha256": sha256_file(candidate_dir / "tokenizer.json"),
            "receipt_sha256": sha256_file(receipt_path),
            "outputs_sha256": sha256_file(outputs_path),
            "v23_metadata_sha256": sha256_file(v23_metadata_path),
            "v23_outputs_sha256": sha256_file(v23_outputs_path),
        },
        "negative_evidence_preserved": True,
        "claim_boundary": "V24 is negative ABI representation evidence plus a bounded LayerCake UTF-8 contract finding. It does not establish Phase 3, teacher-relative quality, host performance, or ABI superiority over LoRA or distillation."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_decision(*, root: Path, protocol_path: Path, candidate_dir: Path, evaluation_dir: Path, decision_path: Path) -> dict[str, Any]:
    stored = _json(decision_path)
    expected = build_decision(root=root, protocol_path=protocol_path, candidate_dir=candidate_dir, evaluation_dir=evaluation_dir)
    if stored != expected:
        raise Phase3Error("V24 decision differs from raw-evidence recomputation")
    return {"status": "PASS", "evidence_sha256": expected["evidence_sha256"], "phase3_certified": False}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze", "verify"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_PROTOCOL_V24.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_pointer_core/development_v24/P0-seed240017")
    parser.add_argument("--evaluation-dir", default="results/abi_capability_compiler_phase3_pointer_core/evaluation_v24/P0-seed240017")
    parser.add_argument("--decision", default="results/abi_capability_compiler_phase3_pointer_core/pointer_core_decision_v24.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    values = {"root": root, "protocol_path": (root / args.protocol).resolve(), "candidate_dir": (root / args.candidate_dir).resolve(), "evaluation_dir": (root / args.evaluation_dir).resolve()}
    decision_path = (root / args.decision).resolve()
    if args.command == "analyze":
        result = build_decision(**values)
        _write_immutable(decision_path, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    else:
        result = verify_decision(**values, decision_path=decision_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
