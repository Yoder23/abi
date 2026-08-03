import json
from pathlib import Path

import pytest

from abi.capability_pipeline import (
    CapabilityPipelineError,
    read_extraction_bundle,
    verify_extraction_bundle,
)
from abi.capability_segregation import SEGREGATED_RECORD_SCHEMA
from abi.certification_catalog import (
    PROBES_PER_SPLIT,
    SPLITS,
    build_certification_catalog,
)
from abi.hf_extraction import probe_label_evidence_sha256
from abi.layercake_acquisition import (
    ENGLISH_CORE_CAPABILITIES,
    build_labeled_extraction_record,
)
from abi.moonshot import (
    _automatic_budgets,
    _defer_empty_search_error_for_supplements,
    _passing_search_supplements,
    _records_for_exact_inventory_selection,
    _survey_evidence_budgets,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "LAYERCAKE_ACQUISITION_PROTOCOL.json"
CATALOG = ROOT / "catalogs" / "development_capability_probes_v1.json"
SEGREGATED_CATALOG = (
    ROOT / "catalogs" / "english_and_first_domains_certification_v6.json"
)
SURVEY = ROOT / "results" / "abi_moonshot" / "qwen2-1.5b-development-survey.abix"
COMPOSED = (
    ROOT / "results" / "abi_moonshot" / "qwen2-1.5b-english-python-math.abix"
)


def test_empty_search_is_deferred_only_for_explicit_development_supplements():
    assert _defer_empty_search_error_for_supplements(
        selected_records=[],
        include_passing_search_supplements=True,
        development=True,
    )
    assert not _defer_empty_search_error_for_supplements(
        selected_records=[],
        include_passing_search_supplements=True,
        development=False,
    )
    assert not _defer_empty_search_error_for_supplements(
        selected_records=[],
        include_passing_search_supplements=False,
        development=True,
    )
    assert not _defer_empty_search_error_for_supplements(
        selected_records=[{"record_id": "already-selected"}],
        include_passing_search_supplements=True,
        development=True,
    )


def test_passing_search_supplements_fail_closed_on_scope_split_and_result():
    selection = {
        "selected_items": [
            {
                "destination_scope": "english_core",
                "domain": "domain_independent",
                "capability": "rewriting",
                "source_model": "teacher/model",
                "source_model_revision": "a" * 40,
            }
        ]
    }
    common = {
        "destination_scope": "english_core",
        "domain": "domain_independent",
        "capability": "rewriting",
        "source_model": "teacher/model",
        "source_model_revision": "a" * 40,
    }
    records = [
        {**common, "record_id": "passing-search", "split": "search"},
        {**common, "record_id": "failed-search", "split": "search"},
        {**common, "record_id": "passing-validation", "split": "validation"},
        {
            **common,
            "record_id": "wrong-source",
            "split": "search",
            "source_model_revision": "b" * 40,
        },
    ]
    results = [
        {"record_id": "passing-search", "passed": True},
        {"record_id": "failed-search", "passed": False},
        {"record_id": "passing-validation", "passed": True},
        {"record_id": "wrong-source", "passed": True},
    ]
    assert _passing_search_supplements(
        records=records,
        probe_results=results,
        selection=selection,
    ) == [records[0]]


def test_exact_inventory_selection_allows_only_declared_training_omissions():
    inventory_hash = "i" * 64
    present_result_hash = "p" * 64
    missing_result_hash = "m" * 64
    source_hash = "s" * 64
    record = {
        "record_id": "record-present",
        "destination_scope": "english_core",
        "domain": "domain_independent",
        "capability": "rewriting",
        "source_model": "teacher/model",
        "source_model_revision": "a" * 40,
        "provenance": "catalog-v1:rewrite-1",
    }
    result = {
        "record_id": record["record_id"],
        "probe_result_sha256": present_result_hash,
    }
    inventory = {
        "inventory_sha256": inventory_hash,
        "entries": [
            {
                "destination_scope": "english_core",
                "domain": "domain_independent",
                "capability": "rewriting",
                "probe_result_sha256": [
                    present_result_hash,
                    missing_result_hash,
                ],
            }
        ],
    }
    source = {
        "model_id": "teacher/model",
        "revision": "a" * 40,
        "source_manifest_sha256": source_hash,
    }
    selection = {
        "selected_items": [
            {
                "destination_scope": "english_core",
                "domain": "domain_independent",
                "capability": "rewriting",
                "source_manifest_sha256": source_hash,
                "inventory_sha256": inventory_hash,
            }
        ]
    }
    kwargs = {
        "records": [record],
        "probe_results": [result],
        "inventories": [inventory],
        "sources": [source],
        "selection": selection,
    }

    with pytest.raises(
        CapabilityPipelineError,
        match="inventory evidence is absent",
    ):
        _records_for_exact_inventory_selection(**kwargs)
    assert _records_for_exact_inventory_selection(
        **kwargs,
        training_material_inventory_hashes={inventory_hash},
    ) == [record]

    with pytest.raises(
        CapabilityPipelineError,
        match="no materialized evidence",
    ):
        _records_for_exact_inventory_selection(
            **{**kwargs, "probe_results": []},
            training_material_inventory_hashes={inventory_hash},
        )


def test_protocol_is_honestly_open_and_locks_scientific_depth():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["format"] == "abi-layercake-acquisition-protocol/2"
    assert protocol["status"] == "OPEN_HOST_IMPORT_PROVEN_ZERO_REGRESSION_NOT_MET"
    assert protocol["moonshot_passed"] is False
    assert (
        protocol["discovery"]["minimum_distinct_prompts_per_capability_for_promotion"]
        == 100
    )
    assert protocol["discovery"]["exhaustive_domain_discovery_claim_allowed"] is False
    assert protocol["minimum_information_search"]["absolute_global_minimum_claim_allowed"] is False
    assert protocol["retention_and_losslessness"]["global_semantic_identity_claim_allowed"] is False
    assert protocol["layercake_promotion"]["teacher_present_at_inference"] is False
    assert protocol["layercake_promotion"]["source_transformer_blocks_retained"] == 0


def test_development_catalog_covers_locked_english_scope_but_is_not_promotional():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert catalog["status"] == "DEVELOPMENT_ONLY_NOT_PROMOTION_DEPTH"
    english = {
        probe["capability"]
        for probe in catalog["probes"]
        if probe["destination_scope"] == "english_core"
    }
    assert english == set(ENGLISH_CORE_CAPABILITIES)
    assert all(probe["split"] == "validation" for probe in catalog["probes"])


def test_certification_catalog_has_locked_disjoint_depth_for_every_capability():
    catalog = build_certification_catalog()
    probes = catalog["probes"]
    assert catalog["generation"]["final_test_used_for_selection"] is False
    assert len(probes) == 18 * len(SPLITS) * PROBES_PER_SPLIT
    assert len({probe["probe_id"] for probe in probes}) == len(probes)

    english = {
        probe["capability"]
        for probe in probes
        if probe["destination_scope"] == "english_core"
    }
    assert english == set(ENGLISH_CORE_CAPABILITIES)

    grouped = {}
    for probe in probes:
        key = (
            probe["destination_scope"],
            probe["domain"],
            probe["capability"],
        )
        grouped.setdefault(key, []).append(probe)
    assert len(grouped) == 18
    for rows in grouped.values():
        assert len(rows) == len(SPLITS) * PROBES_PER_SPLIT
        assert {
            split: sum(row["split"] == split for row in rows)
            for split in SPLITS
        } == {split: PROBES_PER_SPLIT for split in SPLITS}

    corrected = build_certification_catalog("v2")
    assert corrected["generation"]["supersedes"] == catalog["catalog_id"]
    assert len(corrected["probes"]) == len(probes)
    assert {
        probe["prompt"] for probe in corrected["probes"]
    }.isdisjoint({probe["prompt"] for probe in probes})
    python_rows = [
        probe
        for probe in corrected["probes"]
        if probe["capability"] == "python_generation"
    ]
    assert all(
        row["evaluator"]["kind"] == "python_function_expression"
        and row["max_new_tokens"] == 128
        for row in python_rows
    )
    final_revision = build_certification_catalog("v3")
    assert final_revision["generation"]["supersedes"] == corrected["catalog_id"]
    assert {
        probe["prompt"] for probe in final_revision["probes"]
    }.isdisjoint({probe["prompt"] for probe in corrected["probes"]})
    email_rows = [
        probe
        for probe in final_revision["probes"]
        if probe["capability"] == "email_drafting"
    ]
    assert all(
        row["evaluator"]["kind"] == "all_of"
        and row["max_new_tokens"] == 192
        for row in email_rows
    )
    unambiguous = build_certification_catalog("v4")
    assert unambiguous["generation"]["supersedes"] == final_revision["catalog_id"]
    v4_email_rows = [
        probe
        for probe in unambiguous["probes"]
        if probe["capability"] == "email_drafting"
    ]
    assert all(
        "document code DOC-" in row["prompt"]
        and "under 80 words" in row["prompt"]
        and row["max_new_tokens"] == 160
        for row in v4_email_rows
    )

    segregated = build_certification_catalog("v6")
    assert json.loads(
        SEGREGATED_CATALOG.read_text(encoding="utf-8")
    ) == segregated
    assert segregated["generation"]["supersedes"].endswith("-v5")
    assert all(
        probe["record_schema"] == SEGREGATED_RECORD_SCHEMA
        and probe["label_evidence_sha256"]
        == probe_label_evidence_sha256(probe)
        for probe in segregated["probes"]
    )
    english = [
        probe
        for probe in segregated["probes"]
        if probe["destination_scope"] == "english_core"
    ]
    assert all(
        probe["domain_labels"] == []
        and probe["domain_claims"] == []
        and probe["output_introduces_unsupplied_facts"] is False
        for probe in english
    )
    reasoning = [
        probe
        for probe in english
        if probe["capability"] == "domain_independent_reasoning"
    ]
    assert all(
        probe["content_basis"] == "abstract_or_nonce_content"
        and "Compute " not in probe["prompt"]
        for probe in reasoning
    )
    domain = [
        probe
        for probe in segregated["probes"]
        if probe["destination_scope"] == "domain_cake"
    ]
    assert all(
        probe["domain_labels"] == [probe["domain"]]
        and len(probe["domain_claims"]) == 1
        and probe["output_introduces_unsupplied_facts"] is True
        for probe in domain
    )


def test_real_qwen_artifacts_match_protocol_and_selected_domains_are_absent():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    survey = verify_extraction_bundle(SURVEY)
    composed = verify_extraction_bundle(COMPOSED)
    assert survey["archive_sha256"] == protocol["current_evidence"][
        "development_survey_archive_sha256"
    ]
    assert composed["archive_sha256"] == protocol["current_evidence"][
        "development_composition_archive_sha256"
    ]
    assert survey["promotion_eligible_selection"] is False
    assert composed["promotion_eligible_selection"] is False
    assert composed["training_eligible"] is False

    payload = read_extraction_bundle(COMPOSED)
    destinations = {
        (record["destination_scope"], record["domain"])
        for record in payload["records"]
    }
    assert destinations == {
        ("english_core", "domain_independent"),
        ("domain_cake", "python"),
        ("domain_cake", "mathematics"),
    }
    assert payload["ledger"]["teacher_tokens"] == 333
    assert payload["ledger"]["teacher_generated_output_bytes"] == 1385
    assert payload["ledger"]["external_hardware_used"] is False
    assert payload["ledger"]["final_imported_substrate_parameters"] == 0
    assert payload["ledger"]["bridge_parameters_trained"] == 0


def test_automatic_budgets_start_with_complete_capability_coverage():
    source = {
        "model_id": "source",
        "revision": "revision",
    }
    records = []
    for capability_index, capability in enumerate(("grammar", "coherence", "rewriting")):
        for example_index in range(4):
            records.append(
                build_labeled_extraction_record(
                    destination_scope="english_core",
                    domain="domain_independent",
                    capability=capability,
                    provenance=f"{capability}-{example_index}",
                    source_model=source["model_id"],
                    source_model_revision=source["revision"],
                    split="search",
                    prompt=f"prompt {capability} {example_index}",
                    output=f"output {capability} {example_index}",
                    teacher_tokens=5 + example_index,
                    teacher_token_counter="test-tokenizer",
                )
            )
    budgets = _automatic_budgets(records, ordering_seed="coverage-v1")
    first_ids = set(budgets[0]["record_ids"])
    covered = {
        record["capability"]
        for record in records
        if record["record_id"] in first_ids
    }
    assert covered == {"grammar", "coherence", "rewriting"}


def test_final_only_survey_has_no_budget_and_mixed_budgets_exclude_final():
    source = {
        "model_id": "source",
        "revision": "revision",
    }
    records = [
        build_labeled_extraction_record(
            destination_scope="english_core",
            domain="domain_independent",
            capability="grammar",
            provenance=f"grammar-{split}",
            source_model=source["model_id"],
            source_model_revision=source["revision"],
            split=split,
            prompt=f"prompt grammar {split}",
            output=f"output grammar {split}",
            teacher_tokens=5,
            teacher_token_counter="test-tokenizer",
        )
        for split in ("search", "validation", "final_test")
    ]
    final_record = records[-1]
    assert _survey_evidence_budgets(
        [final_record], ordering_seed="final-evidence-v1"
    ) == []

    mixed = _survey_evidence_budgets(
        records, ordering_seed="mixed-evidence-v1"
    )
    assert mixed
    assert all(
        final_record["record_id"] not in budget["record_ids"]
        and budget["split"] in {"search", "validation"}
        for budget in mixed
    )
