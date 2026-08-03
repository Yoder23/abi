from __future__ import annotations

import json

import pytest

from abi.hf_extraction import load_probe_catalog
from abi.natural_instruction_catalog import (
    CAPABILITY_PATTERNS,
    NaturalInstructionCatalogError,
    build_catalog,
)


def _rows() -> list[dict[str, str]]:
    samples = {
        "grammar": (
            "Correct the following sentence for grammar.",
            "the two dogs runs through the quiet park every morning",
        ),
        "email_drafting": (
            "Draft a polite email from the supplied notes.",
            "meeting moved to friday and the room is still undecided",
        ),
        "summarization": (
            "Summarize the following passage in one sentence.",
            "the group tried a small test and then reviewed what happened",
        ),
        "rewriting": (
            "Rewrite the following text so it is clearer.",
            "the review was hard to follow because every idea was repeated",
        ),
        "tone_control": (
            "Rewrite this in a friendly tone.",
            "send the file before the meeting begins",
        ),
        "conversation": (
            "Write a supportive reply to this conversation.",
            "i am nervous about speaking during the meeting",
        ),
        "format_control": (
            "Return the supplied items as a bullet list.",
            "a red kite a blue cup and a green scarf",
        ),
        "coherence": (
            "Rearrange these events into a coherent paragraph.",
            "the rain stopped the door opened everyone walked outside",
        ),
        "cake_output_realization": (
            "Construct a sentence using the given words.",
            "quiet garden silver key wooden gate",
        ),
        "prompt_grounding": (
            "Based on the supplied text identify the main event.",
            "mira found a folded note and carried it to her sister",
        ),
        "instruction_following": (
            "Write exactly one short sentence from the supplied text.",
            "the lamp was beside the chair in the empty room",
        ),
    }
    return [
        {"instruction": instruction, "input": supplied, "output": "ignored"}
        for instruction, supplied in samples.values()
    ]


def test_catalog_uses_only_prompts_and_is_schema_valid(tmp_path) -> None:
    catalog = build_catalog(
        rows=_rows(),
        source_file_manifest={
            "file_name": "fixture.json",
            "sha256": "0" * 64,
            "bytes": 1,
        },
        minimum_per_capability=1,
    )
    assert len(catalog["probes"]) == len(CAPABILITY_PATTERNS)
    assert catalog["generation"]["original_reference_answers_imported"] == 0
    assert all(
        probe["split"] == "search"
        and probe["domain_labels"] == []
        and probe["domain_claims"] == []
        and "ignored" not in probe["prompt"]
        for probe in catalog["probes"]
    )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert (
        load_probe_catalog(path)["catalog_id"]
        == "abi-natural-domain-filtered-instruction-search-v1"
    )


def test_catalog_rejects_specialist_and_closed_book_rows() -> None:
    rows = _rows()[:-1]
    rows.extend(
        [
            {
                "instruction": "Write Python code for this function.",
                "input": "sort a list of values",
                "output": "ignored",
            },
            {
                "instruction": "What country has this capital city?",
                "input": "the supplied city name",
                "output": "ignored",
            },
        ]
    )
    with pytest.raises(
        NaturalInstructionCatalogError,
        match="instruction_following has only 0 safe rows",
    ):
        build_catalog(
            rows=rows,
            source_file_manifest={
                "file_name": "fixture.json",
                "sha256": "0" * 64,
                "bytes": 1,
            },
            minimum_per_capability=1,
        )
