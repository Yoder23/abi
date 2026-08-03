from __future__ import annotations

from collections import Counter, defaultdict

from abi.capability_compiler_phase1_catalog import (
    ADVERSARIAL_FAMILIES,
    ADVERSARIAL_PER_FAMILY,
    CAPABILITY_ALIASES,
    DOMAINS,
    FINAL_PER_CAPABILITY,
    ISOLATION_PER_DOMAIN,
    SEARCH_PER_CAPABILITY,
    VALIDATION_PER_CAPABILITY,
    build_catalog,
)
from abi.hf_extraction import load_probe_catalog


def test_phase1_catalog_has_locked_depth_and_boundaries(tmp_path):
    catalog = build_catalog()
    counts = Counter((row["canonical_capability"], row["split"]) for row in catalog["probes"])
    for capability in CAPABILITY_ALIASES.values():
        assert counts[(capability, "search")] == SEARCH_PER_CAPABILITY
        assert counts[(capability, "validation")] == VALIDATION_PER_CAPABILITY
        assert counts[(capability, "final_test")] == FINAL_PER_CAPABILITY
    assert Counter(row["domain"] for row in catalog["domain_isolation_probes"]) == {
        domain: ISOLATION_PER_DOMAIN for domain in DOMAINS
    }
    assert Counter(row["family"] for row in catalog["adversarial_probes"]) == {
        family: ADVERSARIAL_PER_FAMILY for family in ADVERSARIAL_FAMILIES
    }
    assert all(not row["training_eligible"] for row in catalog["domain_isolation_probes"])
    assert all(not row["training_eligible"] for row in catalog["adversarial_probes"])

    families = defaultdict(set)
    for row in catalog["probes"]:
        families[row["split"]].add(row["phase1_template_family"])
        assert row["destination_scope"] == "english_core"
        assert row["domain"] == "domain_independent"
        assert row["domain_labels"] == []
        assert row["domain_claims"] == []
        assert row["output_introduces_unsupplied_facts"] is False
    assert not families["search"] & families["validation"]
    assert not families["search"] & families["final_test"]
    assert not families["validation"] & families["final_test"]

    path = tmp_path / "catalog.json"
    import json

    path.write_text(json.dumps(catalog), encoding="utf-8")
    loaded = load_probe_catalog(path)
    assert len(loaded["probes"]) == len(catalog["probes"])


def test_phase1_catalog_is_deterministic_and_prompt_disjoint():
    first = build_catalog()
    second = build_catalog()
    assert first == second
    prompts = [row["prompt"] for row in first["probes"]]
    assert len(prompts) == len(set(prompts))
