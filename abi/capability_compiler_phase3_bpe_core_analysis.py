"""Recompute the V38 BPE-core screen from immutable raw evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_bpe_core import load_protocol
from .capability_compiler_phase3_direct_core import _json


DECISION_FORMAT = "abi-capability-compiler-phase3-bpe-core-decision/1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase3Error(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise Phase3Error(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def wilson(successes: int, observations: int, z: float = 1.959963984540054) -> dict[str, float]:
    if observations <= 0 or not 0 <= successes <= observations:
        raise Phase3Error("invalid Wilson inputs")
    point = successes / observations
    denominator = 1.0 + z * z / observations
    center = (point + z * z / (2.0 * observations)) / denominator
    half = z * math.sqrt(
        point * (1.0 - point) / observations + z * z / (4.0 * observations**2)
    ) / denominator
    return {"point": point, "lower_95": center - half, "upper_95": center + half}


def paired_stratified_bootstrap(
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    strata = {
        capability: [
            int(row["candidate_pass"]) - int(row["teacher_pass"])
            for row in rows
            if row["capability"] == capability
        ]
        for capability in CAPABILITIES
    }
    if any(len(values) != 100 for values in strata.values()):
        raise Phase3Error("paired bootstrap depth changed")
    observed = sum(sum(values) for values in strata.values()) / len(rows)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        total = 0
        for values in strata.values():
            total += sum(values[rng.randrange(len(values))] for _ in values)
        draws.append(total / len(rows))
    draws.sort()
    lower_index = int(0.025 * replicates)
    upper_index = min(replicates - 1, int(0.975 * replicates))
    return {
        "candidate_minus_teacher": observed,
        "lower_95": draws[lower_index],
        "upper_95": draws[upper_index],
        "replicates": replicates,
        "seed": seed,
        "method": "capability_stratified_paired_percentile_bootstrap",
    }


def _identity(
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    fit_dir: Path,
    evaluation_dir: Path,
):
    protocol, protocol_sha = load_protocol(root, protocol_path)
    metadata = _json(candidate_dir / "metadata.json")
    fit_receipt = _json(fit_dir / "receipt.json")
    evaluation_receipt = _json(evaluation_dir / "receipt.json")
    fit_rows = _jsonl(fit_dir / "training_fit_rows.jsonl")
    output_rows = _jsonl(evaluation_dir / "development_outputs.jsonl")
    if (
        metadata.get("protocol_sha256") != protocol_sha
        or fit_receipt.get("protocol_sha256") != protocol_sha
        or evaluation_receipt.get("protocol_sha256") != protocol_sha
        or metadata.get("teacher_present_at_inference") is not False
        or metadata.get("source_blocks_retained") != 0
        or metadata.get("promotion_eligible") is not False
        or metadata.get("final_test_accessed") is not False
        or fit_receipt.get("final_test_accessed") is not False
        or evaluation_receipt.get("final_test_accessed") is not False
        or evaluation_receipt.get("promotion_eligible") is not False
        or metadata.get("representation", {}).get("pointer_supervision") is not False
        or metadata.get("imported_information", {}).get("stored_logits") != 0
        or metadata.get("imported_information", {}).get("stored_activations") != 0
        or metadata.get("imported_information", {}).get("source_parameters_copied") != 0
    ):
        raise Phase3Error("V38 governance or ownership identity failed")
    for name, document in (
        ("model.safetensors", metadata["checkpoint"]),
        ("tokenizer.json", metadata["tokenizer"]),
        ("model_config.json", metadata["model_config"]),
    ):
        path = candidate_dir / name
        if not path.is_file() or sha256_file(path) != document["sha256"]:
            raise Phase3Error(f"V38 candidate binding failed: {name}")
    if sha256_file(fit_dir / "training_fit_rows.jsonl") != fit_receipt.get("rows_sha256"):
        raise Phase3Error("V38 fit-row binding failed")
    if sha256_file(evaluation_dir / "development_outputs.jsonl") != evaluation_receipt.get("outputs_sha256"):
        raise Phase3Error("V38 output binding failed")
    return protocol, protocol_sha, metadata, fit_receipt, evaluation_receipt, fit_rows, output_rows


def build_decision(
    *,
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    fit_dir: Path,
    evaluation_dir: Path,
) -> dict[str, Any]:
    (
        protocol,
        protocol_sha,
        metadata,
        fit_receipt,
        evaluation_receipt,
        fit_rows,
        output_rows,
    ) = _identity(root, protocol_path, candidate_dir, fit_dir, evaluation_dir)

    if len(fit_rows) != 7000 or len({row.get("record_id") for row in fit_rows}) != 7000:
        raise Phase3Error("V38 training-fit depth changed")
    fit_actions = sum(int(row["actions"]) for row in fit_rows)
    fit_correct = sum(int(row["correct_actions"]) for row in fit_rows)
    fit_exact = sum(row.get("exact_sequence") is True for row in fit_rows)
    fit_nll = sum(float(row["action_nll_sum"]) for row in fit_rows)
    fit_recomputed = {
        "records": len(fit_rows),
        "actions": fit_actions,
        "correct_actions": fit_correct,
        "action_accuracy": fit_correct / fit_actions,
        "exact_sequences": fit_exact,
        "exact_sequence_rate": fit_exact / len(fit_rows),
        "mean_action_nll": fit_nll / fit_actions,
    }
    for key, value in fit_recomputed.items():
        stored = fit_receipt.get(key)
        if isinstance(value, float):
            if not math.isclose(float(stored), value, rel_tol=0.0, abs_tol=1e-12):
                raise Phase3Error(f"V38 fit aggregate changed: {key}")
        elif stored != value:
            raise Phase3Error(f"V38 fit aggregate changed: {key}")

    probes = development_probes((root / protocol["development_catalog"]).resolve())
    expected = {str(probe["probe_id"]): probe for probe in probes}
    if (
        len(output_rows) != 1400
        or len({row.get("probe_id") for row in output_rows}) != 1400
        or {str(row.get("probe_id")) for row in output_rows} != set(expected)
    ):
        raise Phase3Error("V38 development depth changed")
    for row in output_rows:
        probe = expected[str(row["probe_id"])]
        recomputed_pass = evaluate_functional(str(row.get("output", "")), probe["evaluator"])
        recomputed_collapse = repetition_collapse(str(row.get("output", "")))
        if (
            row.get("capability") != probe["canonical_capability"]
            or row.get("functional_pass") is not recomputed_pass
            or row.get("repetition_collapse") is not recomputed_collapse
        ):
            raise Phase3Error("V38 raw-output scorer identity changed")

    per_capability: dict[str, Any] = {}
    for capability in CAPABILITIES:
        values = [row for row in output_rows if row["capability"] == capability]
        passes = sum(row["functional_pass"] is True for row in values)
        collapses = sum(row["repetition_collapse"] is True for row in values)
        if len(values) != 100:
            raise Phase3Error(f"V38 capability depth changed: {capability}")
        per_capability[capability] = {
            "passes": passes,
            "observations": len(values),
            "collapses": collapses,
            "wilson": wilson(passes, len(values)),
        }
    functional_passes = sum(row["functional_pass"] is True for row in output_rows)
    repetition_collapses = sum(row["repetition_collapse"] is True for row in output_rows)
    generation_errors = sum(row.get("generation_error") is not None for row in output_rows)
    receipt_recomputed = {
        "observations": len(output_rows),
        "functional_passes": functional_passes,
        "repetition_collapses": repetition_collapses,
        "generation_errors": generation_errors,
        "per_capability": {
            name: {key: value[key] for key in ("passes", "collapses", "observations")}
            for name, value in per_capability.items()
        },
    }
    for key, value in receipt_recomputed.items():
        if evaluation_receipt.get(key) != value:
            raise Phase3Error(f"V38 evaluation aggregate changed: {key}")

    teacher = {
        str(row["probe_id"]): row
        for row in _jsonl((root / protocol["teacher_reference"]).resolve())
    }
    paired_rows: list[dict[str, Any]] = []
    for row in output_rows:
        probe_id = str(row["probe_id"])
        if probe_id not in teacher:
            raise Phase3Error(f"V38 teacher output missing: {probe_id}")
        probe = expected[probe_id]
        paired_rows.append(
            {
                "probe_id": probe_id,
                "capability": row["capability"],
                "candidate_pass": bool(row["functional_pass"]),
                "teacher_pass": evaluate_functional(str(teacher[probe_id].get("output", "")), probe["evaluator"]),
            }
        )
    comparison = paired_stratified_bootstrap(
        paired_rows,
        replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]),
        seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]),
    )

    thresholds = protocol["absolute_screen"]
    ordinary = all(
        value["wilson"]["point"] >= float(thresholds["per_capability_functional_point_estimate_minimum"])
        and value["wilson"]["lower_95"] >= float(thresholds["per_capability_functional_wilson_lower_minimum"])
        for value in per_capability.values()
    )
    critical = all(
        per_capability[name]["wilson"]["point"]
        >= float(thresholds["prompt_grounding_instruction_adherence_and_abstention_point_minimum"])
        and per_capability[name]["wilson"]["lower_95"]
        >= float(thresholds["prompt_grounding_instruction_adherence_and_abstention_wilson_lower_minimum"])
        for name in ("prompt_grounding", "instruction_following", "abstention")
    )
    gates = {
        "per_capability_functional": ordinary,
        "critical_capabilities": critical,
        "zero_repetition_collapses": repetition_collapses <= int(thresholds["repetition_collapse_count_maximum"]),
        "zero_generation_errors": generation_errors <= int(thresholds["generation_error_count_maximum"]),
    }
    gates["absolute_screen_pass"] = all(gates.values())
    gates["teacher_relative_noninferiority"] = (
        comparison["lower_95"]
        >= float(protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"])
    )
    initial_pass = gates["absolute_screen_pass"] and gates["teacher_relative_noninferiority"]
    fit_thresholds = protocol["fit_diagnostic"]
    fit_classification = {
        "training_action_accuracy_at_least_0_99": fit_recomputed["action_accuracy"] >= float(fit_thresholds["action_accuracy_reference"]),
        "training_exact_sequence_rate_at_least_0_90": fit_recomputed["exact_sequence_rate"] >= float(fit_thresholds["exact_sequence_rate_reference"]),
        "promotion_effect": "DIAGNOSTIC_ONLY",
    }
    result: dict[str, Any] = {
        "format": DECISION_FORMAT,
        "status": "PASS_INITIAL_SCREEN_CONTROLS_REQUIRED" if initial_pass else "FAIL_INITIAL_SCREEN_BPE_FIXED_ACTION_CANDIDATE_CLOSED",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED",
        "phase3_certified": False,
        "phase4_through_8": "LOCKED",
        "candidate": {
            "system": "B0",
            "seed": metadata["seed"],
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "checkpoint_bytes": metadata["checkpoint"]["bytes"],
            "trainable_parameters": metadata["model_config"]["trainable_parameters"],
            "functional_passes": functional_passes,
            "observations": len(output_rows),
            "functional_wilson": wilson(functional_passes, len(output_rows)),
            "repetition_collapses": repetition_collapses,
            "generation_errors": generation_errors,
            "per_capability": per_capability,
            "training_fit": fit_recomputed,
            "fit_classification": fit_classification,
            "training": metadata["training"],
            "imported_information": metadata["imported_information"],
            "teacher_present_at_inference": metadata["teacher_present_at_inference"],
            "source_blocks_retained": metadata["source_blocks_retained"],
        },
        "teacher_relative": comparison,
        "gates": {
            **gates,
            "matched_causal_controls": "REQUIRED_NEXT" if initial_pass else "NOT_REACHED_PREREGISTERED_EARLY_STOP",
            "remaining_paired_seeds": "REQUIRED_AFTER_CONTROLS" if initial_pass else "PROHIBITED_BY_DECISION_RULE",
            "final_test_accessed": False,
        },
        "ownership": {
            "abi_acquisition_or_representation_failure": not initial_pass,
            "layercake_host_regression": False,
            "layercake_host_construct_status": "INDEPENDENT_PASS_CONSTRUCT_ONLY",
            "layercake_quality_or_speed_inherited": False,
        },
        "decision": {
            "candidate_promoted": False,
            "initial_screen_pass": initial_pass,
            "controls_run": False,
            "remaining_seeds_run": False,
            "next_step": "Preregister the complete matched causal control matrix and paired seeds." if initial_pass else "Preserve this negative candidate and require a new measured bottleneck before any successor training.",
        },
        "evidence": {
            "metadata_sha256": sha256_file(candidate_dir / "metadata.json"),
            "model_config_sha256": sha256_file(candidate_dir / "model_config.json"),
            "tokenizer_sha256": sha256_file(candidate_dir / "tokenizer.json"),
            "checkpoint_sha256": sha256_file(candidate_dir / "model.safetensors"),
            "fit_receipt_sha256": sha256_file(fit_dir / "receipt.json"),
            "fit_rows_sha256": sha256_file(fit_dir / "training_fit_rows.jsonl"),
            "evaluation_receipt_sha256": sha256_file(evaluation_dir / "receipt.json"),
            "outputs_sha256": sha256_file(evaluation_dir / "development_outputs.jsonl"),
        },
        "negative_evidence_preserved": True,
        "claim_boundary": "Development-only ABI acquisition screen. It cannot certify Phase 3, inherit LayerCake quality or performance, or establish ABI superiority over LoRA or distillation.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze", "verify"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_BPE_CORE_PROTOCOL_V38.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_bpe_core/development_v38/B0-seed240017")
    parser.add_argument("--fit-dir", default="results/abi_capability_compiler_phase3_bpe_core/fit_v38/B0-seed240017")
    parser.add_argument("--evaluation-dir", default="results/abi_capability_compiler_phase3_bpe_core/evaluation_v38/B0-seed240017")
    parser.add_argument("--decision", default="results/abi_capability_compiler_phase3_bpe_core/bpe_core_decision_v38.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    values = {
        "root": root,
        "protocol_path": (root / args.protocol).resolve(),
        "candidate_dir": (root / args.candidate_dir).resolve(),
        "fit_dir": (root / args.fit_dir).resolve(),
        "evaluation_dir": (root / args.evaluation_dir).resolve(),
    }
    decision_path = (root / args.decision).resolve()
    result = build_decision(**values)
    if args.command == "analyze":
        _write_immutable(
            decision_path,
            json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
    elif _json(decision_path) != result:
        raise Phase3Error("stored V38 decision differs from raw-evidence recomputation")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
