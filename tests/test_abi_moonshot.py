import json
from pathlib import Path

from abi.capability_pipeline import read_extraction_bundle, verify_extraction_bundle
from abi.certification_catalog import (
    PROBES_PER_SPLIT,
    SPLITS,
    build_certification_catalog,
)
from abi.layercake_acquisition import (
    ENGLISH_CORE_CAPABILITIES,
    build_labeled_extraction_record,
)
from abi.moonshot import _automatic_budgets


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "LAYERCAKE_ACQUISITION_PROTOCOL.json"
CATALOG = ROOT / "catalogs" / "development_capability_probes_v1.json"
SURVEY = ROOT / "results" / "abi_moonshot" / "qwen2-1.5b-development-survey.abix"
COMPOSED = (
    ROOT / "results" / "abi_moonshot" / "qwen2-1.5b-english-python-math.abix"
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
