from __future__ import annotations

import json

from abi.hf_extraction import load_probe_catalog
from abi.natural_gap_catalog import build_catalog


def test_gap_catalog_is_complete_disjoint_and_segregated(tmp_path) -> None:
    catalog = build_catalog()
    assert len(catalog["probes"]) == 300
    assert catalog["generation"]["capability_counts"] == {
        "clarification": 100,
        "abstention": 100,
        "domain_independent_reasoning": 100,
    }
    prompts = [probe["prompt"] for probe in catalog["probes"]]
    assert len(prompts) == len(set(prompts))
    assert all(
        probe["split"] == "search"
        and probe["destination_scope"] == "english_core"
        and probe["domain"] == "domain_independent"
        and probe["domain_labels"] == []
        and probe["domain_claims"] == []
        for probe in catalog["probes"]
    )
    path = tmp_path / "gap.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert (
        load_probe_catalog(path)["catalog_id"]
        == "abi-natural-instruction-gap-search-v1"
    )
