from __future__ import annotations

import json

from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.natural_abstention_catalog import build_catalog


def test_abstention_catalog_has_disjoint_search_and_validation() -> None:
    catalog = build_catalog()
    assert len(catalog["probes"]) == 200
    assert catalog["generation"]["split_counts"] == {
        "search": 100,
        "validation": 100,
    }
    prompts = [probe["prompt"] for probe in catalog["probes"]]
    assert len(prompts) == len(set(prompts))
    search = {
        probe["prompt"] for probe in catalog["probes"] if probe["split"] == "search"
    }
    validation = {
        probe["prompt"]
        for probe in catalog["probes"]
        if probe["split"] == "validation"
    }
    assert search.isdisjoint(validation)
    assert all(
        probe["destination_scope"] == "english_core"
        and probe["domain"] == "domain_independent"
        and probe["domain_labels"] == []
        and probe["domain_claims"] == []
        for probe in catalog["probes"]
    )


def test_abstention_evaluator_accepts_observed_valid_phrasings() -> None:
    evaluator = build_catalog()["probes"][0]["evaluator"]
    valid = (
        "The note does not provide any information about the factory location.",
        "The purchase date is not mentioned in the fictional note.",
        "The page count cannot be determined from the given information.",
        "The answer is unavailable from the supplied text.",
    )
    assert all(evaluate_output(output, evaluator)[0] for output in valid)
    assert not evaluate_output("The exact price is twelve credits.", evaluator)[0]


def test_abstention_catalog_loads_under_extraction_schema(tmp_path) -> None:
    path = tmp_path / "abstention.json"
    path.write_text(json.dumps(build_catalog()), encoding="utf-8")
    assert (
        load_probe_catalog(path)["catalog_id"]
        == "abi-natural-abstention-search-validation-v2"
    )
