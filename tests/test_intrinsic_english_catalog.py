from __future__ import annotations

from collections import Counter, defaultdict

from abi.intrinsic_english_catalog import (
    CAPABILITIES,
    FAMILIES_PER_CAPABILITY,
    SEARCH_PER_CAPABILITY,
    VALIDATION_PER_CAPABILITY,
    build_intrinsic_english_catalog,
    build_intrinsic_preflight_catalog,
)


def test_intrinsic_catalog_is_balanced_unique_and_domain_empty():
    catalog = build_intrinsic_english_catalog()
    rows = catalog["probes"]
    assert len(rows) == len(CAPABILITIES) * (
        SEARCH_PER_CAPABILITY + VALIDATION_PER_CAPABILITY
    )
    assert len({row["prompt"] for row in rows}) == len(rows)
    counts = Counter((row["capability"], row["split"]) for row in rows)
    families = defaultdict(set)
    for row in rows:
        assert row["destination_scope"] == "english_core"
        assert row["domain_labels"] == []
        assert row["domain_claims"] == []
        assert row["label_method"] == "preregistered_catalog"
        assert row["output_introduces_unsupplied_facts"] is False
        assert row["evaluator"]["prompt_contract_sha256"]
        families[row["capability"]].add(row["surface_family"])
    for capability in CAPABILITIES:
        assert counts[(capability, "search")] == SEARCH_PER_CAPABILITY
        assert counts[(capability, "validation")] == VALIDATION_PER_CAPABILITY
        assert families[capability] == set(range(FAMILIES_PER_CAPABILITY))


def test_intrinsic_preflight_selects_every_surface_family_per_capability():
    catalog = build_intrinsic_preflight_catalog()
    rows = catalog["probes"]
    counts = Counter(row["capability"] for row in rows)
    assert counts == Counter(
        {capability: FAMILIES_PER_CAPABILITY for capability in CAPABILITIES}
    )
    for capability in CAPABILITIES:
        assert {
            row["surface_family"]
            for row in rows
            if row["capability"] == capability
        } == set(range(FAMILIES_PER_CAPABILITY))
    assert all(row["split"] == "search" for row in rows)
