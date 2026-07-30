import pytest

from abi.layercake_acquisition import (
    AcquisitionAccountingError,
    assert_deployed_layercake_is_teacher_free,
    build_imported_information_ledger,
    build_labeled_extraction_record,
    select_minimum_passing_budget,
    validate_labeled_extraction_record,
)
from abi.layercake_full_core_acquisition import _DeterministicRowSampler


ABI_SHA = "d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761"


def _record(*, prompt="Rewrite this.", output="Please revise this.", scope="english_core"):
    return build_labeled_extraction_record(
        destination_scope=scope,
        capability="rewriting" if scope == "english_core" else "python_generation",
        domain="domain_independent" if scope == "english_core" else "python",
        provenance="unit-test-fixture",
        split="search",
        source_model="open/source",
        source_model_revision="abc123",
        prompt=prompt,
        output=output,
        teacher_tokens=5,
        teacher_token_counter="source-runtime",
    )


def _ledger(records):
    return build_imported_information_ledger(
        records,
        logits_stored_count=0,
        logits_stored_bytes=0,
        hidden_activations_stored_count=0,
        hidden_activations_stored_bytes=0,
        frozen_source_parameters_copied=0,
        frozen_source_parameter_bytes_copied=0,
        final_imported_substrate_parameters=100,
        final_imported_substrate_parameter_bytes=200,
        bridge_parameters_trained=20,
        bridge_parameter_bytes=40,
        artifact_disk_footprint_bytes=1000,
        peak_process_resident_memory_bytes=2000,
        cpu_core_hours=1.5,
        source_model_inference_hours=0.25,
        one_time_source_extraction_seconds=900,
        per_host_acquisition_and_certification_seconds=1800,
        final_deployed_footprint_bytes=800,
        final_cpu_inference_seconds=0.05,
        active_parameter_seconds=1234,
        external_hardware_used=False,
        external_hardware_description="",
    )


def test_balanced_sampler_equalizes_capabilities_without_adding_records():
    rows = [
        {"record_id": f"a-{index}", "capability": "a"}
        for index in range(10)
    ] + [{"record_id": "b-0", "capability": "b"}]
    sampler = _DeterministicRowSampler(
        rows,
        seed=17,
        strategy="balanced_capabilities",
    )
    selected = sampler.batch(20)
    counts = {
        capability: sum(
            row["capability"] == capability for row in selected
        )
        for capability in ("a", "b")
    }
    assert counts == {"a": 10, "b": 10}
    assert {row["record_id"] for row in selected}.issubset(
        {row["record_id"] for row in rows}
    )


def test_record_is_content_addressed_and_enforces_destination_boundary():
    record = _record()
    validate_labeled_extraction_record(record)
    changed = dict(record, output="tampered")
    with pytest.raises(AcquisitionAccountingError, match="stale or invalid"):
        validate_labeled_extraction_record(changed)
    with pytest.raises(AcquisitionAccountingError, match="domain_independent"):
        build_labeled_extraction_record(
            destination_scope="english_core",
            capability="rewriting",
            domain="python",
            provenance="test",
            split="search",
            source_model="open/source",
            source_model_revision="abc123",
            prompt="x",
            output="y",
            teacher_tokens=1,
            teacher_token_counter="source-runtime",
        )


def test_ledger_counts_text_weights_and_every_nontext_transfer_channel():
    record = _record()
    ledger = _ledger([record])
    expected_bytes = (
        record["prompt_utf8_bytes"]
        + record["output_utf8_bytes"]
        + 200
        + 40
    )
    assert ledger["teacher_tokens"] == 5
    assert ledger["logits_stored_bytes"] == 0
    assert ledger["hidden_activations_stored_bytes"] == 0
    assert ledger["total_accounted_transfer_bytes"] == expected_bytes
    assert ledger["total_imported_payload_bits"] == expected_bytes * 8
    assert ledger["external_hardware_used"] is False


def test_ledger_rejects_duplicate_records_and_unexplained_external_hardware():
    record = _record()
    with pytest.raises(AcquisitionAccountingError, match="duplicate record_id"):
        _ledger([record, record])
    kwargs = dict(
        logits_stored_count=0,
        logits_stored_bytes=0,
        hidden_activations_stored_count=0,
        hidden_activations_stored_bytes=0,
        frozen_source_parameters_copied=0,
        frozen_source_parameter_bytes_copied=0,
        final_imported_substrate_parameters=1,
        final_imported_substrate_parameter_bytes=4,
        bridge_parameters_trained=1,
        bridge_parameter_bytes=4,
        artifact_disk_footprint_bytes=8,
        peak_process_resident_memory_bytes=16,
        cpu_core_hours=0,
        source_model_inference_hours=0,
        one_time_source_extraction_seconds=0,
        per_host_acquisition_and_certification_seconds=0,
        final_deployed_footprint_bytes=8,
        final_cpu_inference_seconds=0,
        active_parameter_seconds=0,
        external_hardware_used=True,
        external_hardware_description="",
    )
    with pytest.raises(AcquisitionAccountingError, match="non-empty"):
        build_imported_information_ledger([record], **kwargs)


def test_budget_selector_uses_nested_validation_records_and_reports_lower_failure():
    observations = [
        {
            "budget_id": "b100",
            "split": "validation",
            "teacher_tokens": 100,
            "total_imported_payload_bits": 8000,
            "record_ids": ["a"],
            "common_gates": {"quality": False, "inference": True},
        },
        {
            "budget_id": "b200",
            "split": "validation",
            "teacher_tokens": 200,
            "total_imported_payload_bits": 15000,
            "record_ids": ["a", "b"],
            "common_gates": {"quality": True, "inference": True},
        },
        {
            "budget_id": "b400",
            "split": "validation",
            "teacher_tokens": 400,
            "total_imported_payload_bits": 30000,
            "record_ids": ["a", "b", "c"],
            "common_gates": {"quality": True, "inference": True},
        },
    ]
    decision = select_minimum_passing_budget(observations)
    assert decision["selected_budget_id"] == "b200"
    assert decision["largest_lower_failing_budget_id"] == "b100"
    assert decision["absolute_minimum_claimed"] is False


def test_budget_selector_rejects_test_selection_and_non_nested_budgets():
    observations = [
        {
            "budget_id": "a",
            "split": "validation",
            "teacher_tokens": 10,
            "total_imported_payload_bits": 80,
            "record_ids": ["a"],
            "common_gates": {"quality": False},
        },
        {
            "budget_id": "b",
            "split": "final_test",
            "teacher_tokens": 20,
            "total_imported_payload_bits": 160,
            "record_ids": ["b"],
            "common_gates": {"quality": True},
        },
    ]
    with pytest.raises(AcquisitionAccountingError, match="validation"):
        select_minimum_passing_budget(observations)


def test_deployment_manifest_must_exclude_teacher_and_keep_layercake_abi():
    manifest = {
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "canonical_semantic_abi_sha256": ABI_SHA,
        "components": [{"type": "layercake_core"}, {"type": "abi_bridge"}],
    }
    assert_deployed_layercake_is_teacher_free(
        manifest, expected_canonical_abi_sha256=ABI_SHA
    )
    manifest["components"].append({"type": "source_transformer_block"})
    with pytest.raises(AcquisitionAccountingError, match="forbidden"):
        assert_deployed_layercake_is_teacher_free(
            manifest, expected_canonical_abi_sha256=ABI_SHA
        )
