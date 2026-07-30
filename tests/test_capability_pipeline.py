import io
import json
import zipfile

import pytest

from abi.capability_pipeline import (
    CapabilityPipelineError,
    TRAINING_ARTIFACT_ROLE,
    build_capability_inventory,
    build_extraction_bundle,
    build_inventory_survey_plan,
    build_nested_teacher_budgets,
    build_probe_result,
    build_semantic_retention_certificate,
    build_source_model_manifest,
    build_user_selection_plan,
    records_for_selection,
    verify_extraction_bundle,
)
from abi.layercake_acquisition import (
    ENGLISH_CORE_CAPABILITIES,
    build_labeled_extraction_record,
)


REVISION = "a" * 40


def _source(model_id="Qwen/test", revision=REVISION, weight="1" * 64):
    return build_source_model_manifest(
        model_id=model_id,
        revision=revision,
        revision_is_immutable=True,
        architecture="Qwen2ForCausalLM",
        parameter_count=500_000_000,
        tokenizer_id=model_id,
        tokenizer_revision=revision,
        license_id="Apache-2.0",
        weight_files=[
            {
                "relative_path": "model.safetensors",
                "sha256": weight,
                "bytes": 1_000_000,
            }
        ],
    )


def _record(
    source,
    *,
    capability,
    domain="domain_independent",
    scope="english_core",
    split="validation",
    suffix="0",
):
    return build_labeled_extraction_record(
        destination_scope=scope,
        capability=capability,
        domain=domain,
        provenance=f"probe-{suffix}",
        split=split,
        source_model=source["model_id"],
        source_model_revision=source["revision"],
        prompt=f"prompt {capability} {suffix}",
        output=f"output {capability} {suffix}",
        teacher_tokens=5,
        teacher_token_counter="generated_token_ids",
    )


def _result(source, record, *, suffix="0", passed=True):
    return build_probe_result(
        record=record,
        source_manifest_sha256=source["source_manifest_sha256"],
        probe_id=f"probe-{record['capability']}-{suffix}",
        evaluator={"kind": "contains_all", "values": ["output"]},
        passed=passed,
        score=1.0 if passed else 0.0,
        seed=int(suffix) if suffix.isdigit() else 0,
    )


def _english_evidence(source, *, repeats=1):
    records = []
    results = []
    for capability in sorted(ENGLISH_CORE_CAPABILITIES):
        for index in range(repeats):
            record = _record(source, capability=capability, suffix=str(index))
            records.append(record)
            results.append(_result(source, record, suffix=str(index)))
    return records, results


def test_source_manifest_is_content_addressed_and_rejects_unsafe_weights():
    source = _source()
    assert source["promotion_eligible"] is True
    assert source["weight_file_count"] == 1
    with pytest.raises(CapabilityPipelineError, match="safe relative"):
        build_source_model_manifest(
            model_id="bad",
            revision=REVISION,
            revision_is_immutable=True,
            architecture="bad",
            parameter_count=1,
            tokenizer_id="bad",
            tokenizer_revision=REVISION,
            license_id="unknown",
            weight_files=[
                {"relative_path": "../escape.bin", "sha256": "0" * 64, "bytes": 1}
            ],
        )


def test_inventory_is_probe_bounded_and_unverified_selection_fails_closed():
    source = _source()
    records, results = _english_evidence(source)
    inventory = build_capability_inventory(
        source_manifest=source,
        records=records,
        probe_results=results,
    )
    assert inventory["catalog_scope"] == "probe_defined_not_exhaustive"
    assert inventory["exhaustive_domain_discovery_claimed"] is False
    assert inventory["available_entry_count"] == 0
    with pytest.raises(CapabilityPipelineError, match="no qualified source"):
        build_user_selection_plan(
            [inventory], include_english_core=True, domains=[]
        )
    development = build_user_selection_plan(
        [inventory],
        include_english_core=True,
        domains=[],
        allow_unverified_development_selection=True,
    )
    assert development["promotion_eligible"] is False

    survey = build_inventory_survey_plan(inventory)
    assert survey["selection_purpose"] == "source_capability_survey"
    assert survey["promotion_eligible"] is False
    assert len(survey["selected_items"]) == len(ENGLISH_CORE_CAPABILITIES)


def test_final_test_cannot_qualify_inventory_or_enter_training_artifact(tmp_path):
    source = _source()
    validation_record = _record(source, capability="rewriting", split="validation")
    final_record = _record(
        source, capability="rewriting", split="final_test", suffix="final"
    )
    validation_result = _result(source, validation_record)
    final_result = _result(source, final_record, suffix="final")
    inventory = build_capability_inventory(
        source_manifest=source,
        records=[validation_record, final_record],
        probe_results=[validation_result, final_result],
        minimum_distinct_probes=2,
        minimum_wilson_lower_bound=0,
    )
    entry = inventory["entries"][0]
    assert entry["probe_count"] == 2
    assert entry["qualification_probe_count"] == 1
    assert entry["available"] is False
    with pytest.raises(CapabilityPipelineError, match="qualification_splits"):
        build_capability_inventory(
            source_manifest=source,
            records=[final_record],
            probe_results=[final_result],
            minimum_distinct_probes=1,
            minimum_wilson_lower_bound=0,
            qualification_splits=("final_test",),
        )


def test_user_can_select_english_and_domain_from_different_sources():
    english_source = _source(model_id="Qwen/english", weight="1" * 64)
    domain_source = _source(model_id="DeepSeek/code", weight="2" * 64)
    english_records, english_results = _english_evidence(english_source)
    python_record = _record(
        domain_source,
        capability="python_generation",
        domain="python",
        scope="domain_cake",
    )
    python_result = _result(domain_source, python_record)
    english_inventory = build_capability_inventory(
        source_manifest=english_source,
        records=english_records,
        probe_results=english_results,
        minimum_distinct_probes=1,
        minimum_wilson_lower_bound=0,
    )
    python_inventory = build_capability_inventory(
        source_manifest=domain_source,
        records=[python_record],
        probe_results=[python_result],
        minimum_distinct_probes=1,
        minimum_wilson_lower_bound=0,
    )
    plan = build_user_selection_plan(
        [english_inventory, python_inventory],
        include_english_core=True,
        domains=["python"],
    )
    assert plan["promotion_eligible"] is True
    assert set(plan["selected_source_manifest_sha256"]) == {
        english_source["source_manifest_sha256"],
        domain_source["source_manifest_sha256"],
    }
    chosen = records_for_selection(
        [*english_records, python_record], plan, split="validation"
    )
    assert len(chosen) == len(ENGLISH_CORE_CAPABILITIES) + 1


def test_nested_budgets_are_deterministic_stratified_and_nested():
    source = _source()
    records, _ = _english_evidence(source, repeats=3)
    budgets = build_nested_teacher_budgets(
        records,
        requested_teacher_token_budgets=[15, 30, 60],
        split="validation",
        ordering_seed="locked-v1",
    )
    assert [row["teacher_tokens"] for row in budgets] == [15, 30, 60]
    assert set(budgets[0]["record_ids"]).issubset(budgets[1]["record_ids"])
    assert set(budgets[1]["record_ids"]).issubset(budgets[2]["record_ids"])
    repeat = build_nested_teacher_budgets(
        records,
        requested_teacher_token_budgets=[15, 30, 60],
        split="validation",
        ordering_seed="locked-v1",
    )
    assert budgets == repeat


def test_bundle_is_deterministic_schema_closed_and_tamper_evident(tmp_path):
    source = _source()
    records, results = _english_evidence(source)
    inventory = build_capability_inventory(
        source_manifest=source,
        records=records,
        probe_results=results,
        minimum_distinct_probes=1,
        minimum_wilson_lower_bound=0,
    )
    plan = build_user_selection_plan(
        [inventory], include_english_core=True, domains=[]
    )
    budgets = build_nested_teacher_budgets(
        records,
        requested_teacher_token_budgets=[10, 30],
        split="validation",
        ordering_seed="locked-v1",
    )
    first = tmp_path / "first.abix"
    second = tmp_path / "second.abix"
    kwargs = dict(
        source_manifests=[source],
        records=records,
        probe_results=results,
        inventories=[inventory],
        selection=plan,
        budgets=budgets,
        ledger={"schema_version": "test-ledger/1", "teacher_tokens": 70},
    )
    first_result = build_extraction_bundle(first, **kwargs)
    second_result = build_extraction_bundle(second, **kwargs)
    assert first.read_bytes() == second.read_bytes()
    assert first_result["archive_sha256"] == second_result["archive_sha256"]
    verified = verify_extraction_bundle(first)
    assert verified["verified"] is True
    assert verified["record_count"] == len(records)

    with zipfile.ZipFile(io.BytesIO(first.read_bytes()), "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["records.jsonl"] += b"{}\n"
    tampered = tmp_path / "tampered.abix"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(CapabilityPipelineError, match="member"):
        verify_extraction_bundle(tampered)

    final_record = _record(
        source,
        capability="python_generation",
        domain="python",
        scope="domain_cake",
        split="final_test",
        suffix="final",
    )
    final_result = _result(source, final_record, suffix="final")
    final_inventory = build_capability_inventory(
        source_manifest=source,
        records=[final_record],
        probe_results=[final_result],
        minimum_distinct_probes=1,
        minimum_wilson_lower_bound=0,
    )
    final_plan = build_user_selection_plan(
        [final_inventory],
        include_english_core=False,
        domains=["python"],
        allow_unverified_development_selection=True,
    )
    with pytest.raises(CapabilityPipelineError, match="search records only"):
        build_extraction_bundle(
            tmp_path / "leaky.abix",
            source_manifests=[source],
            records=[final_record],
            probe_results=[final_result],
            inventories=[final_inventory],
            selection=final_plan,
            budgets=[],
            ledger={"schema_version": "test-ledger/1"},
            artifact_role=TRAINING_ARTIFACT_ROLE,
        )

    validation_record = _record(
        source,
        capability="python_generation",
        domain="python",
        scope="domain_cake",
        split="validation",
        suffix="validation",
    )
    validation_result = _result(
        source, validation_record, suffix="validation"
    )
    validation_inventory = build_capability_inventory(
        source_manifest=source,
        records=[validation_record],
        probe_results=[validation_result],
        minimum_distinct_probes=1,
        minimum_wilson_lower_bound=0,
    )
    validation_plan = build_user_selection_plan(
        [validation_inventory],
        include_english_core=False,
        domains=["python"],
    )
    with pytest.raises(CapabilityPipelineError, match="search records only"):
        build_extraction_bundle(
            tmp_path / "validation-leak.abix",
            source_manifests=[source],
            records=[validation_record],
            probe_results=[validation_result],
            inventories=[validation_inventory],
            selection=validation_plan,
            budgets=[],
            ledger={"schema_version": "test-ledger/1"},
            artifact_role=TRAINING_ARTIFACT_ROLE,
        )


def test_retention_certificate_separates_exact_bytes_from_bounded_semantics():
    evaluations = []
    for prompt_id in ("p1", "p2"):
        for seed in (1, 2):
            evaluations.append(
                {
                    "destination": "english_core",
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "source_passed": True,
                    "layercake_passed": True,
                    "critical": prompt_id == "p1",
                }
            )
    certificate = build_semantic_retention_certificate(
        extraction_archive_sha256="a" * 64,
        deployed_artifact_sha256_before="b" * 64,
        deployed_artifact_sha256_after="b" * 64,
        evaluations=evaluations,
        teacher_present_at_inference=False,
        source_transformer_blocks_retained=0,
        minimum_distinct_prompts_per_destination=2,
        minimum_seeds=2,
    )
    assert certificate["status"] == "PASS"
    assert certificate["payload_byte_identity"] is True
    assert certificate["semantic_claim_scope"] == (
        "zero_measured_regressions_on_locked_probe_suite"
    )
    assert certificate["global_semantic_identity_claimed"] is False

    evaluations[0]["layercake_passed"] = False
    failed = build_semantic_retention_certificate(
        extraction_archive_sha256="a" * 64,
        deployed_artifact_sha256_before="b" * 64,
        deployed_artifact_sha256_after="b" * 64,
        evaluations=evaluations,
        teacher_present_at_inference=False,
        source_transformer_blocks_retained=0,
        minimum_distinct_prompts_per_destination=2,
        minimum_seeds=2,
    )
    assert failed["status"] == "FAIL"
    assert failed["zero_measured_source_to_layercake_regressions"] is False
