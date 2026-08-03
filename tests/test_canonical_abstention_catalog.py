from __future__ import annotations

import json

from abi.canonical_abstention_catalog import CANONICAL_RESPONSE, build_catalog
from abi.hf_extraction import evaluate_output, load_probe_catalog


def test_canonical_abstention_is_fresh_validation_only() -> None:
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
        and probe["evaluator"] == {
            "kind": "exact",
            "value": CANONICAL_RESPONSE,
        }
        for probe in catalog["probes"]
    )


def test_canonical_abstention_evaluator_is_exact() -> None:
    evaluator = build_catalog()["probes"][0]["evaluator"]
    assert evaluate_output(CANONICAL_RESPONSE, evaluator)[0]
    assert evaluate_output(CANONICAL_RESPONSE.lower(), evaluator)[0]
    assert not evaluate_output(
        f"{CANONICAL_RESPONSE}. The note omits it.", evaluator
    )[0]
    assert not evaluate_output("The answer is twelve credits.", evaluator)[0]


def test_canonical_abstention_catalog_loads(tmp_path) -> None:
    path = tmp_path / "canonical-abstention.json"
    path.write_text(json.dumps(build_catalog()), encoding="utf-8")
    assert (
        load_probe_catalog(path)["catalog_id"]
        == "abi-canonical-abstention-validation-v1"
    )
