from __future__ import annotations

import pytest

from abi.broad_english_catalog import BroadEnglishCatalogError
from abi.english_scale_catalog import (
    SCALE_CAPABILITIES,
    _unique_context_sample,
    build_catalog,
)
import json

from abi.hf_extraction import load_probe_catalog


def _story(prefix: str, index: int) -> str:
    return (
        f"{prefix} character {index} opened a quiet red door. "
        f"The character found a folded note marked {prefix}{index}. "
        "A friendly neighbor waited beside the garden path. "
        "They read the note together and smiled. "
        "Afterward, they placed it safely on a wooden shelf. "
        "The evening ended with a gentle conversation."
    )


def _stories(prefix: str, count: int) -> list[str]:
    return [_story(prefix, index) for index in range(count)]


def test_scale_catalog_uses_one_distinct_domain_clean_probe_per_context(
    tmp_path,
):
    catalog = build_catalog(
        search_stories=_stories("search", 12),
        validation_stories=_stories("validation", 8),
        final_stories=_stories("final", 8),
        corpus_manifest={"schema_version": "test-corpus/1"},
    )

    assert len(catalog["probes"]) == 28
    assert catalog["generation"]["one_probe_per_distinct_context"] is True
    search = [
        probe for probe in catalog["probes"] if probe["split"] == "search"
    ]
    assert {probe["capability"] for probe in search} == set(
        SCALE_CAPABILITIES
    )
    assert all(probe["destination_scope"] == "english_core" for probe in search)
    assert all(probe["domain"] == "domain_independent" for probe in search)
    assert all(probe["domain_labels"] == [] for probe in search)
    assert all(probe["domain_claims"] == [] for probe in search)
    assert len({probe["raw_context_sha256"] for probe in search}) == len(search)
    path = tmp_path / "scale.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert load_probe_catalog(path)["catalog_id"] == catalog["catalog_id"]


def test_scale_catalog_is_deterministic_and_balanced():
    arguments = {
        "search_stories": _stories("search", 12),
        "validation_stories": _stories("validation", 8),
        "final_stories": _stories("final", 8),
        "corpus_manifest": {"schema_version": "test-corpus/1"},
        "seed": 19824,
    }
    first = build_catalog(**arguments)
    second = build_catalog(**arguments)
    assert first == second
    assert first["generation"]["capability_counts"]["search"] == {
        capability: 3 for capability in SCALE_CAPABILITIES
    }


def test_scale_catalog_rejects_cross_split_overlap():
    repeated = _story("repeated", 1)
    with pytest.raises(BroadEnglishCatalogError, match="overlap"):
        build_catalog(
            search_stories=[repeated],
            validation_stories=[repeated],
            final_stories=[_story("final", 1)],
            corpus_manifest={"schema_version": "test-corpus/1"},
        )


def test_scale_sampler_deduplicates_and_excludes_exact_contexts():
    first = _story("first", 1)
    second = _story("second", 2)
    selected = _unique_context_sample(
        [first, first, second],
        count=2,
        seed=7,
        exclusion_markers=(),
    )
    excluded = {
        probe["raw_context_sha256"]
        for probe in build_catalog(
            search_stories=[selected[0]],
            validation_stories=[_story("validation", 3)],
            final_stories=[_story("final", 4)],
            corpus_manifest={"schema_version": "test-corpus/1"},
        )["probes"]
        if probe["split"] == "search"
    }
    remaining = _unique_context_sample(
        [selected[0], selected[1]],
        count=1,
        seed=8,
        exclusion_markers=(),
        excluded_context_sha256=frozenset(
            bytes.fromhex(value) for value in excluded
        ),
    )
    assert remaining == [selected[1]]


def test_scale_sampler_skips_rows_that_cannot_form_bounded_context():
    oversized = (
        ("word " * 200)
        + ". This is a second sentence. This is a third sentence. "
        "This is a fourth sentence."
    )
    valid = _story("valid", 5)
    assert _unique_context_sample(
        [oversized, valid],
        count=1,
        seed=9,
        exclusion_markers=(),
        maximum_context_characters=240,
    ) == [valid]
