"""Recompute and verify the immutable V23 direct-core absolute-screen decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable


DECISION_FORMAT = "abi-capability-compiler-phase3-direct-core-decision/1"
PROTOCOL_FORMAT = "abi-capability-compiler-phase3-direct-core-screen/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase3Error(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise Phase3Error(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def wilson(successes: int, observations: int, z: float = 1.959963984540054) -> dict[str, float]:
    if observations <= 0 or successes < 0 or successes > observations:
        raise Phase3Error("invalid Wilson inputs")
    point = successes / observations
    denominator = 1.0 + z * z / observations
    center = (point + z * z / (2.0 * observations)) / denominator
    half = z * math.sqrt(point * (1.0 - point) / observations + z * z / (4.0 * observations**2)) / denominator
    return {"point": point, "lower_95": center - half, "upper_95": center + half}


def _require_identity(
    *,
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    evaluation_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    protocol = _json(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    metadata_path = candidate_dir / "metadata.json"
    receipt_path = evaluation_dir / "receipt.json"
    outputs_path = evaluation_dir / "development_outputs.jsonl"
    metadata = _json(metadata_path)
    receipt = _json(receipt_path)
    rows = _jsonl(outputs_path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status") != "PREREGISTERED_CONDITIONAL_ABSOLUTE_SCREEN"
        or protocol.get("controls_deferred_until_absolute_pass") is not True
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
        raise Phase3Error("V23 governance or ownership identity failed")
    for name, document in (("model.safetensors", metadata["checkpoint"]), ("tokenizer.json", metadata["tokenizer"]), ("model_config.json", metadata["model_config"])):
        path = candidate_dir / name
        if not path.is_file() or sha256_file(path) != document["sha256"]:
            raise Phase3Error(f"V23 candidate binding failed: {name}")
    if (candidate_dir / "model.safetensors").stat().st_size != metadata["checkpoint"]["bytes"]:
        raise Phase3Error("V23 checkpoint byte count changed")
    if sha256_file(outputs_path) != receipt.get("outputs_sha256"):
        raise Phase3Error("V23 output binding failed")

    probes = development_probes((root / protocol["development_catalog"]).resolve())
    expected = {str(probe["probe_id"]): probe for probe in probes}
    actual_ids = [str(row.get("probe_id")) for row in rows]
    if len(rows) != 1400 or len(set(actual_ids)) != 1400 or set(actual_ids) != set(expected):
        raise Phase3Error("V23 prompt identity or depth changed")
    for row in rows:
        probe = expected[str(row["probe_id"])]
        if row.get("capability") != probe["canonical_capability"]:
            raise Phase3Error("V23 capability identity changed")
        recomputed_pass = evaluate_functional(str(row.get("output", "")), probe["evaluator"])
        recomputed_collapse = repetition_collapse(str(row.get("output", "")))
        if row.get("functional_pass") is not recomputed_pass or row.get("repetition_collapse") is not recomputed_collapse:
            raise Phase3Error("V23 stored scorer result differs from raw-output recomputation")
        error = row.get("generation_error")
        if error is not None and not isinstance(error, str):
            raise Phase3Error("V23 generation error field changed")
    return protocol, metadata, receipt, rows, protocol_sha


def build_decision(
    *,
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    evaluation_dir: Path,
) -> dict[str, Any]:
    protocol, metadata, receipt, rows, protocol_sha = _require_identity(
        root=root,
        protocol_path=protocol_path,
        candidate_dir=candidate_dir,
        evaluation_dir=evaluation_dir,
    )
    per_capability: dict[str, Any] = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passes = sum(row["functional_pass"] is True for row in values)
        collapses = sum(row["repetition_collapse"] is True for row in values)
        if len(values) != 100:
            raise Phase3Error(f"V23 capability depth changed: {capability}")
        per_capability[capability] = {
            "passes": passes,
            "observations": len(values),
            "collapses": collapses,
            "wilson": wilson(passes, len(values)),
        }
    functional_passes = sum(row["functional_pass"] is True for row in rows)
    repetition_collapses = sum(row["repetition_collapse"] is True for row in rows)
    generation_errors = sum(row.get("generation_error") is not None for row in rows)
    recomputed_receipt = {
        "observations": len(rows),
        "functional_passes": functional_passes,
        "repetition_collapses": repetition_collapses,
        "generation_errors": generation_errors,
        "per_capability": {
            name: {key: value[key] for key in ("passes", "collapses", "observations")}
            for name, value in per_capability.items()
        },
    }
    for key, value in recomputed_receipt.items():
        if receipt.get(key) != value:
            raise Phase3Error(f"V23 receipt differs from raw evidence: {key}")

    thresholds = protocol["absolute_screen"]
    ordinary = all(
        value["wilson"]["point"] >= float(thresholds["per_capability_functional_point_estimate_minimum"])
        and value["wilson"]["lower_95"] >= float(thresholds["per_capability_functional_wilson_lower_minimum"])
        for value in per_capability.values()
    )
    critical_names = ("prompt_grounding", "instruction_following", "abstention")
    critical = all(
        per_capability[name]["wilson"]["point"] >= float(thresholds["prompt_grounding_instruction_adherence_and_abstention_point_minimum"])
        and per_capability[name]["wilson"]["lower_95"] >= float(thresholds["prompt_grounding_instruction_adherence_and_abstention_wilson_lower_minimum"])
        for name in critical_names
    )
    absolute_gates = {
        "per_capability_functional": ordinary,
        "critical_capabilities": critical,
        "zero_repetition_collapses": repetition_collapses <= int(thresholds["repetition_collapse_count_maximum"]),
        "zero_generation_errors": generation_errors <= int(thresholds["generation_error_count_maximum"]),
    }
    absolute_pass = all(absolute_gates.values())
    if absolute_pass:
        raise Phase3Error("V23 unexpectedly reaches deferred teacher/control evaluation; use a successor decision protocol")
    result: dict[str, Any] = {
        "format": DECISION_FORMAT,
        "status": "FAIL_ABSOLUTE_QUALITY_SCREEN_ARCHITECTURE_CLOSED",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED",
        "phase3_certified": False,
        "phase4_through_8": "LOCKED",
        "candidate": {
            "system": "A0",
            "seed": metadata["seed"],
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "checkpoint_bytes": metadata["checkpoint"]["bytes"],
            "trainable_parameters": metadata["model_config"]["trainable_parameters"],
            "functional_passes": functional_passes,
            "observations": len(rows),
            "functional_wilson": wilson(functional_passes, len(rows)),
            "repetition_collapses": repetition_collapses,
            "generation_errors": generation_errors,
            "per_capability": per_capability,
            "training": metadata["training"],
            "imported_information": metadata["imported_information"],
            "teacher_present_at_inference": metadata["teacher_present_at_inference"],
            "source_blocks_retained": metadata["source_blocks_retained"],
        },
        "gates": {
            **absolute_gates,
            "absolute_screen_pass": absolute_pass,
            "teacher_relative_noninferiority": "NOT_REACHED_PREREGISTERED_EARLY_STOP",
            "matched_causal_controls": "NOT_REACHED_PREREGISTERED_EARLY_STOP",
            "remaining_paired_seeds": "PROHIBITED_BY_DECISION_RULE",
            "final_test_accessed": False,
        },
        "ownership": {
            "abi_acquisition_or_representation_failure": True,
            "layercake_host_regression": False,
            "layercake_host_construct_status": "INDEPENDENT_PASS_CONSTRUCT_ONLY",
            "layercake_quality_or_speed_inherited": False,
            "reason": "The external LayerCake host construct was separately verified before this run. V23 generated autonomously through the LayerCake-native plan, but its ABI-trained fixed-action representation failed the locked functional and collapse gates.",
        },
        "decision": {
            "architecture_promoted": False,
            "controls_run": False,
            "remaining_seeds_run": False,
            "exact_v23_architecture_closed": True,
            "reason": "V23 scored 504/1,400 with 77 repetition collapses; 13 of 14 capabilities missed the ordinary point-and-Wilson gate, and all three critical capabilities missed their stricter gates.",
            "measured_bottleneck": "The fixed-action target representation does not explicitly supervise the host plan's prompt-pointer actions. Raw outputs are often fluent templates but substitute, duplicate, or lose prompt entities, so prompt identity and grounding fail during autonomous realization.",
            "next_authorized_question": "A separately preregistered pointer-supervised target-representation experiment may test whether deterministic copying of identity-bearing prompt lexemes fixes the measured grounding bottleneck. It is a representation change, not a V23 hyperparameter sweep.",
        },
        "evidence": {
            "metadata_sha256": sha256_file(candidate_dir / "metadata.json"),
            "model_config_sha256": sha256_file(candidate_dir / "model_config.json"),
            "tokenizer_sha256": sha256_file(candidate_dir / "tokenizer.json"),
            "checkpoint_sha256": sha256_file(candidate_dir / "model.safetensors"),
            "receipt_sha256": sha256_file(evaluation_dir / "receipt.json"),
            "outputs_sha256": sha256_file(evaluation_dir / "development_outputs.jsonl"),
        },
        "negative_evidence_preserved": True,
        "claim_boundary": "This closes one development-only ABI acquisition architecture. It does not establish teacher-relative quality, Phase 3, LayerCake quality or speed, or ABI superiority over LoRA or distillation.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_decision(
    *,
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    evaluation_dir: Path,
    decision_path: Path,
) -> dict[str, Any]:
    stored = _json(decision_path)
    expected = build_decision(root=root, protocol_path=protocol_path, candidate_dir=candidate_dir, evaluation_dir=evaluation_dir)
    if stored != expected:
        raise Phase3Error("V23 decision differs from raw-evidence recomputation")
    return {"status": "PASS", "evidence_sha256": expected["evidence_sha256"], "phase3_certified": False}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze", "verify"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_CORE_PROTOCOL_V23.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_direct_core/development_v23/A0-seed240017")
    parser.add_argument("--evaluation-dir", default="results/abi_capability_compiler_phase3_direct_core/evaluation_v23/A0-seed240017")
    parser.add_argument("--decision", default="results/abi_capability_compiler_phase3_direct_core/direct_core_decision_v23_corrected_v1.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    values = {
        "root": root,
        "protocol_path": (root / args.protocol).resolve(),
        "candidate_dir": (root / args.candidate_dir).resolve(),
        "evaluation_dir": (root / args.evaluation_dir).resolve(),
    }
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
