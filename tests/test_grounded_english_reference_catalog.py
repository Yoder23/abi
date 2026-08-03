from __future__ import annotations

import json

from abi.grounded_english_reference_catalog import (
    CAPABILITIES,
    SEARCH_PER_CAPABILITY,
    VALIDATION_PER_CAPABILITY,
    build_grounded_preflight_catalog,
    build_grounded_english_reference_catalog,
)
from abi.hf_extraction import probe_label_evidence_sha256
from abi.hf_extraction import prompt_contract_sha256
from abi.layercake_acquisition import ENGLISH_CORE_CAPABILITIES
from abi.teacher_artifact_adequacy_audit import _evaluator_is_content_specific


def test_grounded_catalog_is_full_depth_disjoint_and_prompt_specific():
    catalog = build_grounded_english_reference_catalog()
    probes = catalog["probes"]
    assert set(CAPABILITIES) == set(ENGLISH_CORE_CAPABILITIES)
    assert len(probes) == len(CAPABILITIES) * (
        SEARCH_PER_CAPABILITY + VALIDATION_PER_CAPABILITY
    )
    assert {probe["split"] for probe in probes} == {"search", "validation"}
    assert len({probe["probe_id"] for probe in probes}) == len(probes)
    assert len({probe["prompt"] for probe in probes}) == len(probes)
    signatures = {
        json.dumps(probe["evaluator"], sort_keys=True, separators=(",", ":"))
        for probe in probes
    }
    assert len(signatures) == len(probes)
    assert all(_evaluator_is_content_specific(probe["evaluator"]) for probe in probes)
    for capability in CAPABILITIES:
        assert sum(
            probe["capability"] == capability and probe["split"] == "search"
            for probe in probes
        ) == SEARCH_PER_CAPABILITY
        assert sum(
            probe["capability"] == capability and probe["split"] == "validation"
            for probe in probes
        ) == VALIDATION_PER_CAPABILITY


def test_grounded_catalog_is_domain_free_and_labels_bind_exact_prompts():
    catalog = build_grounded_english_reference_catalog()
    forbidden = (
        "atomic number",
        "chemical element",
        "periodic table",
        "independence day",
        "national holiday",
        "united states independence",
        "arithmetic",
        "calculate",
        "equation",
        "python",
    )
    for probe in catalog["probes"]:
        assert probe["destination_scope"] == "english_core"
        assert probe["domain"] == "domain_independent"
        assert probe["domain_labels"] == []
        assert probe["domain_claims"] == []
        assert probe["output_introduces_unsupplied_facts"] is False
        assert all(marker not in probe["prompt"].casefold() for marker in forbidden)
        assert probe["label_evidence_sha256"] == probe_label_evidence_sha256(probe)


def test_preflight_is_exactly_one_search_probe_per_capability():
    preflight = build_grounded_preflight_catalog()
    assert preflight["generation"]["preflight_only"] is True
    assert len(preflight["probes"]) == len(CAPABILITIES)
    assert {probe["capability"] for probe in preflight["probes"]} == set(CAPABILITIES)
    assert {probe["split"] for probe in preflight["probes"]} == {"search"}


def test_v2_changes_only_preflight_measured_capabilities_and_stays_unique():
    v1 = build_grounded_english_reference_catalog("v1")
    v2 = build_grounded_english_reference_catalog("v2")
    changed = {
        left["capability"]
        for left, right in zip(v1["probes"], v2["probes"], strict=True)
        if left["prompt"] != right["prompt"]
        or left["evaluator"] != right["evaluator"]
    }
    assert changed == {
        "abstention",
        "clarification",
        "coherence",
        "conversation",
        "format_control",
        "prompt_grounding",
        "summarization",
        "tone_control",
    }
    assert v2["generation"]["supersedes"] == v1["catalog_id"]
    assert len({probe["prompt"] for probe in v2["probes"]}) == len(v2["probes"])
    assert len(
        {
            json.dumps(probe["evaluator"], sort_keys=True, separators=(",", ":"))
            for probe in v2["probes"]
        }
    ) == len(v2["probes"])


def test_v3_binds_every_evaluator_and_changes_only_two_v2_false_negatives():
    v2 = build_grounded_english_reference_catalog("v2")
    v3 = build_grounded_english_reference_catalog("v3")
    behavior_changed = {
        left["capability"]
        for left, right in zip(v2["probes"], v3["probes"], strict=True)
        if left["prompt"] != right["prompt"]
        or {
            key: value
            for key, value in left["evaluator"].items()
            if key != "prompt_contract_sha256"
        }
        != {
            key: value
            for key, value in right["evaluator"].items()
            if key != "prompt_contract_sha256"
        }
    }
    assert behavior_changed == {"clarification", "conversation"}
    assert all(
        probe["evaluator"]["prompt_contract_sha256"]
        == prompt_contract_sha256(probe["prompt"])
        for probe in v3["probes"]
    )
    preflight = build_grounded_preflight_catalog("v3")
    assert preflight["catalog_id"].endswith("v3-gpu-preflight")
