import hashlib
import json

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_b50_matched_quality import (
    CRITICAL_CAPABILITIES,
    FORMAT,
    SEEDS,
    SYSTEMS,
    _candidate_cost,
    _source_artifact_costs,
    _verified_baseline_pack,
    absolute_gates,
    evaluation,
    paired_quality,
)


THRESHOLDS = {
    "per_capability_functional_point_estimate_minimum": 0.9,
    "per_capability_functional_wilson_lower_minimum": 0.85,
    "critical_point_minimum": 0.95,
    "critical_wilson_lower_minimum": 0.9,
    "repetition_collapse_v2_count_maximum": 0,
}


def _rows(passes=100, collapse=False):
    rows = []
    for capability in (
        "abstention",
        "clarification",
        "coherence",
        "conversation",
        "email_drafting_from_notes",
        "fact_free_reasoning",
        "fluent_realization",
        "format_control",
        "grammar",
        "instruction_following",
        "prompt_grounding",
        "rewriting",
        "supplied_text_summarization",
        "tone_control",
    ):
        for index in range(100):
            passed = index < passes
            rows.append(
                {
                    "probe_id": f"{capability}-{index:03d}",
                    "capability": capability,
                    "functional_pass_v1": passed,
                    "functional_pass_v2": passed,
                    "repetition_collapse_v2": collapse and index == 0,
                }
            )
    return rows


def test_matched_quality_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-matched-quality/1"
    assert SYSTEMS == ("L0", "L1", "D0", "D1", "D2")
    assert SEEDS == (104729, 130363, 155921)
    assert CRITICAL_CAPABILITIES == (
        "prompt_grounding",
        "instruction_following",
        "abstention",
    )


def test_absolute_gates_require_wilson_depth_and_zero_collapse():
    passing = evaluation(_rows(100))
    assert all(absolute_gates(passing, THRESHOLDS).values())
    low = evaluation(_rows(90))
    assert not absolute_gates(low, THRESHOLDS)["critical_capabilities"]
    collapsed = evaluation(_rows(100, collapse=True))
    assert not absolute_gates(collapsed, THRESHOLDS)["zero_repetition_collapse"]


def test_paired_quality_direction_and_reproducibility():
    candidate = _rows(100)
    comparator = _rows(90)
    first = paired_quality(candidate, comparator, replicates=200, seed=77)
    second = paired_quality(candidate, comparator, replicates=200, seed=77)
    assert first == second
    assert first["lower_95"] > 0


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _with_evidence_digest(payload):
    result = dict(payload)
    result["evidence_sha256"] = hashlib.sha256(
        (
            json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return result


def test_baseline_pack_requires_independent_verification(tmp_path):
    pack = _with_evidence_digest(
        {
            "status": "PASS_EXACT_B50_BASELINE_SEQUENCE_PACK_READY",
            "budget": {"selection_sha256": "selection"},
            "imported_information": {"unique_source_attempts": 4953},
        }
    )
    _write_json(tmp_path / "pack.json", pack)
    verification = _with_evidence_digest(
        {
            "status": "PASS_INDEPENDENT_EXACT_B50_BASELINE_PACK_VERIFICATION",
            "result_under_test_sha256": hashlib.sha256(
                (tmp_path / "pack.json").read_bytes()
            ).hexdigest(),
            "selection_sha256": "selection",
            "imported_information": pack["imported_information"],
            "gates": {"records_reconstructed": True},
            "attacks": {"mutation_rejected": True},
            "training_performed": False,
            "model_inference_performed": False,
            "teacher_model_loaded": False,
            "final_test_accessed": False,
        }
    )
    _write_json(tmp_path / "verify.json", verification)
    protocol = {
        "baseline_pack_result": "pack.json",
        "baseline_pack_verification": "verify.json",
    }
    observed_pack, observed_verification = _verified_baseline_pack(
        tmp_path, protocol
    )
    assert observed_pack == pack
    assert observed_verification == verification
    verification["attacks"]["mutation_rejected"] = False
    verification = _with_evidence_digest(
        {key: value for key, value in verification.items() if key != "evidence_sha256"}
    )
    _write_json(tmp_path / "verify.json", verification)
    with pytest.raises(Phase3Error, match="independently verified baseline pack"):
        _verified_baseline_pack(tmp_path, protocol)


def _source_cost_fixture(tmp_path):
    paths = {
        "phase1_certificate": "phase1.json",
        "targeted_extraction_result": "targeted.json",
        "targeted_source_summary": "targeted-summary.json",
        "abstention_source_summary": "abstention.json",
        "targeted_combined_ir_result": "combined.json",
        "host_supervision_result": "host.json",
        "host_supervision_verification": "host-verify.json",
        "top64_result": "top64.json",
    }
    counts = {f"capability-{index}": 500 for index in range(13)}
    counts["abstention"] = 381
    payloads = {
        "phase1.json": {
            "status": "PASS",
            "source": {"source_inference_seconds": 10.0},
        },
        "targeted.json": {
            "status": "FAIL_SOURCE_EVIDENCE_INADEQUATE",
            "eligible_by_capability": counts,
            "summary": {"sha256": "BOUND_AFTER_WRITE"},
            "accounting": {
                "source_inference_seconds": 20.0,
                "source_load_seconds": 2.0,
                "teacher_input_tokens": 200,
                "authoritative_teacher_tokens": 100,
            },
        },
        "targeted-summary.json": {
            "status": "FAIL_SOURCE_EVIDENCE_INADEQUATE",
            "accounting": {
                "wall_seconds_this_process": 22.0,
                "raw_generation_prompt_bytes": 2000,
                "raw_teacher_output_bytes": 1000,
            },
        },
        "abstention.json": {
            "status": "PASS_SOURCE_EVIDENCE_READY_FOR_NORMALIZATION",
            "selection": {"selected_counts": {"abstention": 700}},
            "accounting": {
                "source_inference_seconds_this_process": 3.0,
                "source_load_seconds": 1.0,
                "wall_seconds_this_process": 4.0,
                "teacher_input_tokens": 30,
                "authoritative_teacher_tokens": 15,
                "raw_generation_prompt_bytes": 300,
                "raw_teacher_output_bytes": 150,
            },
        },
        "combined.json": {
            "status": "PASS_BALANCED_IR_CONSTRUCTION",
            "source_policy": {
                "v135_non_abstention_capabilities": 13,
                "v119_abstention_capabilities": 1,
            },
        },
        "host.json": {
            "controls": {"teacher_model_loaded": False},
            "artifact": {"sha256": "host-artifact"},
        },
        "host-verify.json": {
            "status": "PASS_INDEPENDENT_HOSTILE_ARTIFACT_VERIFICATION",
            "artifact": {"sha256": "host-artifact"},
            "teacher_model_loaded": False,
        },
        "top64.json": {
            "status": "PASS_EXACT_B50_TOP64_CACHE_READY",
            "source_inference_seconds": 5.0,
            "source_load_seconds": 1.0,
            "wall_seconds": 7.0,
            "stored_logit_values": 64,
            "stored_logit_value_bytes": 128,
            "stored_logit_index_bytes": 256,
            "peak_cuda_allocated_bytes": 1024,
            "peak_process_rss_bytes": 2048,
        },
    }
    payloads["phase1.json"]["source"].update(
        {
            "source_load_seconds": 1.0,
            "wall_seconds": 12.0,
            "teacher_input_tokens_all_attempts": 100,
            "authoritative_teacher_tokens_all_attempts": 50,
        }
    )
    for name, payload in payloads.items():
        _write_json(tmp_path / name, payload)
    payloads["targeted.json"]["summary"]["sha256"] = hashlib.sha256(
        (tmp_path / "targeted-summary.json").read_bytes()
    ).hexdigest()
    _write_json(tmp_path / "targeted.json", payloads["targeted.json"])
    return {"source_artifact_evidence": paths}, payloads


def test_source_costs_preserve_failed_v135_and_add_v119_without_double_count(tmp_path):
    protocol, _ = _source_cost_fixture(tmp_path)
    result = _source_artifact_costs(tmp_path, protocol)
    common = result["common_sequence_artifacts"]
    assert common["total_source_inference_seconds"] == 33.0
    assert common["v135_status_preserved"] == "FAIL_SOURCE_EVIDENCE_INADEQUATE"
    assert common["host_supervision_additional_teacher_inference_seconds"] == 0.0
    assert result["richer_top64_control_addition"]["source_inference_seconds"] == 5.0


def test_source_costs_fail_if_v135_failure_is_hidden(tmp_path):
    protocol, payloads = _source_cost_fixture(tmp_path)
    payloads["targeted.json"]["status"] = "PASS_TARGETED_EXTRACTION_READY"
    _write_json(tmp_path / "targeted.json", payloads["targeted.json"])
    with pytest.raises(Phase3Error, match="source-artifact accounting changed"):
        _source_artifact_costs(tmp_path, protocol)


def test_candidate_cost_reports_interrupted_attempt_as_lower_bound(tmp_path):
    successful = tmp_path / "successful"
    successful.mkdir()
    stage = successful / "v1"
    stage.mkdir()
    _write_json(stage / "metadata.json", {"training": {"wall_seconds": 2.0, "steps": 3}})
    stage_sha = hashlib.sha256((stage / "metadata.json").read_bytes()).hexdigest()
    lineage = {
        "seed": 104729,
        "budget": {
            "id": "B50",
            "authoritative_teacher_output_tokens": 152266,
            "unique_source_attempts": 4953,
        },
        "stage_metadata_sha256": {"v1": stage_sha},
        "wall_seconds": 5.0,
    }
    _write_json(successful / "result.json", lineage)
    interrupted = tmp_path / "interrupted"
    interrupted_stage = interrupted / "v0"
    interrupted_stage.mkdir(parents=True)
    _write_json(
        interrupted_stage / "metadata.json",
        {"training": {"wall_seconds": 4.0, "steps": 7}},
    )
    interrupted_sha = hashlib.sha256(
        (interrupted_stage / "metadata.json").read_bytes()
    ).hexdigest()
    result = _candidate_cost(
        tmp_path,
        successful / "result.json",
        expected_seed=104729,
        final_parameters=10,
        interrupted_attempt={
            "root": "interrupted",
            "stage_metadata_sha256": {"v0": interrupted_sha},
            "terminal_stage_cost_status": "NOT_RECOVERABLE_NO_METADATA",
        },
    )
    assert result["training_seconds"] == 2.0
    assert result["recorded_interrupted_training_seconds"] == 4.0
    assert result["total_consumed_wall_seconds_lower_bound"] == 9.0
    assert result["interrupted_cost_completeness"] == "LOWER_BOUND_ONLY"
