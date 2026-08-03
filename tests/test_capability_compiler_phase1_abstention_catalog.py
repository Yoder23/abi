from abi.capability_compiler_phase1_abstention_catalog import (
    EXTRA_ABSTENTION_CONSTRUCTIONS,
    SUPPLEMENT_RECORDS,
    build_catalog,
)


def test_abstention_supplement_is_fresh_bounded_and_deterministic():
    first = build_catalog()
    second = build_catalog()
    assert first == second
    assert len(first["probes"]) == SUPPLEMENT_RECORDS
    assert len({row["prompt"] for row in first["probes"]}) == SUPPLEMENT_RECORDS
    assert all(row["split"] == "search" for row in first["probes"])
    assert all(row["canonical_capability"] == "abstention" for row in first["probes"])
    assert all(row["destination_scope"] == "english_core" for row in first["probes"])
    assert all(row["domain_labels"] == [] and row["domain_claims"] == [] for row in first["probes"])
    assert all(
        set(EXTRA_ABSTENTION_CONSTRUCTIONS).issubset(row["evaluator"]["values"])
        for row in first["probes"]
    )
