from abi.capability_compiler_phase3_broad_abstention import build_catalog
from abi.natural_abstention_catalog import ABSTENTION_MARKERS


def _story(name: str) -> str:
    return (
        f"{name} found a green lantern beside the narrow garden path. "
        f"Then {name} carried it toward a quiet wooden porch. "
        "A friend opened the gate and smiled warmly. "
        "Together they placed the lantern safely on a table."
    )


def test_fresh_catalog_is_search_only_segregated_and_uses_established_evaluator() -> None:
    catalog = build_catalog(
        [_story("Lina"), _story("Mara")], corpus_manifest={"fixture": True}, seed=9
    )
    assert len(catalog["probes"]) == 2
    for row in catalog["probes"]:
        assert row["split"] == "search"
        assert row["capability"] == "abstention"
        assert row["domain_labels"] == row["domain_claims"] == []
        contains_any = row["evaluator"]["rules"][-1]
        assert contains_any["values"] == list(ABSTENTION_MARKERS)
