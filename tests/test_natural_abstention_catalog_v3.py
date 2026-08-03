from __future__ import annotations

import json

from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.natural_abstention_catalog_v3 import build_catalog


def test_v3_catalog_is_fresh_validation_only_and_segregated() -> None:
    catalog = build_catalog()
    assert len(catalog["probes"]) == 100
    prompts = [probe["prompt"] for probe in catalog["probes"]]
    assert len(prompts) == len(set(prompts))
    assert all(
        probe["split"] == "validation"
        and probe["destination_scope"] == "english_core"
        and probe["domain"] == "domain_independent"
        and probe["domain_labels"] == []
        and probe["domain_claims"] == []
        for probe in catalog["probes"]
    )


def test_v3_evaluator_covers_v2_development_false_negative_classes() -> None:
    evaluator = build_catalog()["probes"][0]["evaluator"]
    valid = (
        "The note does not state the page count.",
        "The passage does not supply the requested information.",
        "The exact price is not available.",
        "The question cannot be answered from this passage alone.",
        "The evidence is insufficient to determine the manufacturer.",
        "There is no evidence-based answer for the model number.",
        "The fictional note does not include any information about the owner.",
    )
    assert all(evaluate_output(output, evaluator)[0] for output in valid)
    assert not evaluate_output("The exact price is twelve credits.", evaluator)[0]


def test_v3_catalog_loads_under_extraction_schema(tmp_path) -> None:
    path = tmp_path / "abstention-v3.json"
    path.write_text(json.dumps(build_catalog()), encoding="utf-8")
    assert (
        load_probe_catalog(path)["catalog_id"]
        == "abi-natural-abstention-validation-v3"
    )
