from __future__ import annotations

import json

import pytest

from abi.broad_english_catalog import (
    BroadEnglishCatalogError,
    DEFAULT_CORE_EXCLUSION_MARKERS,
    build_catalog,
)
from abi.hf_extraction import load_probe_catalog


STORIES = (
    "Mira found a red kite beside the old tree. The wind lifted it gently. "
    "She held the string and smiled. Her friend waved from the path. They "
    "walked home together after the wind became quiet.",
    "Jon opened a small box in the hall. A folded note rested inside. He read "
    "the note twice. Then he carried the box to his sister. They talked about "
    "the surprise and laughed.",
    "Asha heard a soft sound near the gate. She looked behind a wooden bench. "
    "A tiny puppy was waiting there. Asha brought it a bowl of water. The "
    "puppy wagged its tail.",
)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "abi-raw-language-corpus-manifest/1",
        "dataset_id": "fixture",
        "dataset_revision": "fixture-revision",
    }


def test_broad_catalog_is_disjoint_and_segregated(tmp_path) -> None:
    catalog = build_catalog(
        search_stories=[STORIES[0]],
        validation_stories=[STORIES[1]],
        final_stories=[STORIES[2]],
        corpus_manifest=_manifest(),
    )
    assert len(catalog["probes"]) == 42
    assert catalog["generation"]["closed_book_fact_prompts"] == 0
    assert {
        probe["split"] for probe in catalog["probes"]
    } == {"search", "validation", "final_test"}
    assert all(
        probe["destination_scope"] == "english_core"
        and probe["domain"] == "domain_independent"
        and probe["domain_labels"] == []
        and probe["domain_claims"] == []
        and probe["output_introduces_unsupplied_facts"] is False
        for probe in catalog["probes"]
    )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert load_probe_catalog(path)["catalog_id"] == catalog["catalog_id"]


def test_broad_catalog_rejects_overlap() -> None:
    with pytest.raises(BroadEnglishCatalogError, match="overlap"):
        build_catalog(
            search_stories=[STORIES[0]],
            validation_stories=[STORIES[0]],
            final_stories=[STORIES[2]],
            corpus_manifest=_manifest(),
        )


def test_broad_catalog_prompts_exclude_locked_domain_markers() -> None:
    catalog = build_catalog(
        search_stories=[STORIES[0]],
        validation_stories=[STORIES[1]],
        final_stories=[STORIES[2]],
        corpus_manifest=_manifest(),
    )
    for probe in catalog["probes"]:
        folded = probe["prompt"].casefold()
        assert all(
            marker not in folded
            for marker in DEFAULT_CORE_EXCLUSION_MARKERS
        )
