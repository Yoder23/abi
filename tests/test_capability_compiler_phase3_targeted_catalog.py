from abi.capability_compiler_phase3_targeted_catalog import (
    SEARCH_PER_CAPABILITY,
    SOURCE_OFFSET,
    build_catalog,
)


def test_targeted_catalog_is_balanced_unique_search_only() -> None:
    catalog = build_catalog()
    probes = catalog["probes"]
    assert len(probes) == 14 * SEARCH_PER_CAPABILITY
    assert len({row["prompt"] for row in probes}) == len(probes)
    assert {row["split"] for row in probes} == {"search"}
    assert min(row["source_index"] for row in probes) == SOURCE_OFFSET
    assert all(row["domain"] == "domain_independent" for row in probes)
    counts = {}
    for row in probes:
        counts[row["canonical_capability"]] = counts.get(row["canonical_capability"], 0) + 1
    assert set(counts.values()) == {SEARCH_PER_CAPABILITY}


def test_targeted_catalog_fails_closed_on_prior_prompt_overlap() -> None:
    catalog = build_catalog()
    prompt = catalog["probes"][0]["prompt"]
    import hashlib

    try:
        build_catalog(excluded_prompt_hashes={hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    except RuntimeError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("prior prompt overlap was accepted")
