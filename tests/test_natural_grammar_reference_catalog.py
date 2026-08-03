from __future__ import annotations

import json

from abi.hf_extraction import (
    evaluate_output,
    probe_label_evidence_sha256,
    prompt_contract_sha256,
)
from abi.capability_segregation import LINGUISTIC_FORM
from abi.natural_grammar_reference_catalog import (
    SEARCH_PER_STRUCTURE,
    STRUCTURES,
    VALIDATION_PER_STRUCTURE,
    build_natural_grammar_preflight_catalog,
    build_natural_grammar_reference_catalog,
)
from abi.teacher_artifact_adequacy_audit import _evaluator_is_content_specific


def test_natural_grammar_catalog_is_deep_disjoint_and_exact():
    catalog = build_natural_grammar_reference_catalog()
    probes = catalog["probes"]
    assert len(probes) == len(STRUCTURES) * (
        SEARCH_PER_STRUCTURE + VALIDATION_PER_STRUCTURE
    )
    assert len({probe["probe_id"] for probe in probes}) == len(probes)
    assert len({probe["prompt"] for probe in probes}) == len(probes)
    assert len(
        {
            json.dumps(probe["evaluator"], sort_keys=True, separators=(",", ":"))
            for probe in probes
        }
    ) == len(probes)
    assert {probe["split"] for probe in probes} == {"search", "validation"}
    assert not any(probe["split"] == "final_test" for probe in probes)
    for structure, _ in STRUCTURES:
        prefix = f"natural-grammar-{structure}-"
        assert sum(
            str(probe["probe_id"]).startswith(prefix)
            and probe["split"] == "search"
            for probe in probes
        ) == SEARCH_PER_STRUCTURE
        assert sum(
            str(probe["probe_id"]).startswith(prefix)
            and probe["split"] == "validation"
            for probe in probes
        ) == VALIDATION_PER_STRUCTURE


def test_natural_grammar_records_are_domain_free_and_hash_bound():
    catalog = build_natural_grammar_reference_catalog()
    forbidden = (
        "atomic number",
        "chemical element",
        "independence day",
        "python",
        "equation",
        "calculate",
    )
    for probe in catalog["probes"]:
        assert probe["destination_scope"] == "english_core"
        assert probe["capability"] == "grammar"
        assert probe["domain"] == "domain_independent"
        assert probe["knowledge_class"] == LINGUISTIC_FORM
        assert probe["content_basis"] == "domain_free_instruction"
        assert probe["domain_labels"] == []
        assert probe["domain_claims"] == []
        assert probe["output_introduces_unsupplied_facts"] is False
        assert all(term not in probe["prompt"].casefold() for term in forbidden)
        assert probe["label_evidence_sha256"] == probe_label_evidence_sha256(probe)
        evaluator = probe["evaluator"]
        assert evaluator["kind"] == "exact"
        assert evaluator["case_sensitive"] is True
        assert evaluator["prompt_contract_sha256"] == prompt_contract_sha256(
            probe["prompt"]
        )
        assert _evaluator_is_content_specific(evaluator)
        assert evaluate_output(evaluator["value"], evaluator) == (True, 1.0)
        wrong = probe["prompt"].split("\nSentence: ", 1)[1]
        assert evaluate_output(wrong, evaluator) == (False, 0.0)


def test_natural_grammar_preflight_has_one_row_per_structure():
    preflight = build_natural_grammar_preflight_catalog()
    assert preflight["generation"]["preflight_only"] is True
    assert len(preflight["probes"]) == len(STRUCTURES)
    assert {probe["split"] for probe in preflight["probes"]} == {"search"}
    for structure, _ in STRUCTURES:
        assert sum(
            str(probe["probe_id"]).startswith(
                f"natural-grammar-{structure}-search-"
            )
            for probe in preflight["probes"]
        ) == 1
