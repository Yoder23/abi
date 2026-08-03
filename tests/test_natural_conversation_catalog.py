from __future__ import annotations

import hashlib

import pytest

from abi.natural_conversation_catalog import (
    NaturalConversationCatalogError,
    _oasst_candidates,
    _safe_prompt_surface,
    _ultrachat_candidates,
    build_catalog,
)


SHA = "a" * 64


def _ultra(text: str, answer: str = "REFERENCE MUST NOT CROSS") -> dict:
    return {
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]
    }


def _oasst(text: str, **overrides: object) -> dict:
    row = {
        "role": "prompter",
        "parent_id": None,
        "lang": "en",
        "synthetic": False,
        "deleted": False,
        "review_result": True,
        "text": text,
        "detoxify": {
            "toxicity": 0.0,
            "severe_toxicity": 0.0,
            "obscene": 0.0,
            "identity_attack": 0.0,
            "insult": 0.0,
            "threat": 0.0,
            "sexual_explicit": 0.0,
        },
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    "text",
    [
        "Explain the history of a country.",
        "Write Python code for sorting.",
        "What is the capital city?",
        "Draft an email dated 2026-01-01.",
        "Translate this sentence into French.",
        "How many planets are there?",
        "Cite research about grammar.",
        "Write a recipe using these ingredients.",
    ],
)
def test_unsafe_or_specialist_prompts_are_rejected(text: str) -> None:
    assert _safe_prompt_surface(text) is None


def test_ultrachat_reads_only_first_user_surface() -> None:
    rows = [
        _ultra(
            "Write a friendly note apologizing for arriving late.",
            answer="Python 2026 REFERENCE MUST NOT CROSS",
        )
    ]
    candidates = list(_ultrachat_candidates(rows, shard_sha256=SHA))
    assert len(candidates) == 1
    assert "REFERENCE" not in candidates[0]["source_prompt"]
    assert candidates[0]["source"] == "ultrachat_200k"


def test_ultrachat_rejects_non_user_first_message() -> None:
    rows = [{"messages": [{"role": "assistant", "content": "Write a note."}]}]
    assert list(_ultrachat_candidates(rows, shard_sha256=SHA)) == []


def test_oasst_requires_reviewed_human_english_root() -> None:
    text = "Write a friendly note apologizing for arriving late."
    assert len(list(_oasst_candidates([_oasst(text)], shard_sha256=SHA))) == 1
    for override in (
        {"parent_id": "parent"},
        {"lang": "de"},
        {"synthetic": True},
        {"deleted": True},
        {"review_result": False},
    ):
        assert (
            list(
                _oasst_candidates(
                    [_oasst(text, **override)],
                    shard_sha256=SHA,
                )
            )
            == []
        )


def test_oasst_toxicity_is_fail_closed() -> None:
    row = _oasst("Write a friendly note apologizing for arriving late.")
    row["detoxify"]["toxicity"] = 0.051
    assert list(_oasst_candidates([row], shard_sha256=SHA)) == []


def test_catalog_is_deterministic_deduplicated_and_search_only() -> None:
    ultra = list(
        _ultrachat_candidates(
            [
                _ultra("Write a friendly note apologizing for arriving late."),
                _ultra("Write a friendly note apologizing for arriving late."),
                _ultra("Rewrite this greeting so it sounds more welcoming."),
            ],
            shard_sha256=SHA,
        )
    )
    catalog_a = build_catalog(
        ultrachat_candidates=ultra,
        oasst_candidates=[],
        source_corpora=[],
        maximum_per_capability=2,
        minimum_per_capability=1,
        seed=7,
    )
    catalog_b = build_catalog(
        ultrachat_candidates=ultra,
        oasst_candidates=[],
        source_corpora=[],
        maximum_per_capability=2,
        minimum_per_capability=1,
        seed=7,
    )
    assert catalog_a == catalog_b
    assert all(probe["split"] == "search" for probe in catalog_a["probes"])
    hashes = [probe["natural_prompt_sha256"] for probe in catalog_a["probes"]]
    assert len(hashes) == len(set(hashes))
    assert catalog_a["generation"]["corpus_reference_answers_imported"] == 0
    assert catalog_a["generation"]["corpus_assistant_messages_imported"] == 0


def test_catalog_prefers_reviewed_human_surface() -> None:
    text_a = "Write a gentle apology that sounds sincere and concise."
    text_b = "Write a warm invitation that sounds sincere and concise."
    ultra = list(_ultrachat_candidates([_ultra(text_a)], shard_sha256=SHA))
    oasst = list(_oasst_candidates([_oasst(text_b)], shard_sha256=SHA))
    catalog = build_catalog(
        ultrachat_candidates=ultra,
        oasst_candidates=oasst,
        source_corpora=[],
        maximum_per_capability=1,
        minimum_per_capability=1,
        seed=7,
    )
    assert len(catalog["probes"]) == 1
    assert catalog["probes"][0]["source_prompt_corpus"] == "oasst1"


def test_catalog_requires_a_capability_at_minimum() -> None:
    ultra = list(
        _ultrachat_candidates(
            [_ultra("Write a friendly note apologizing for arriving late.")],
            shard_sha256=SHA,
        )
    )
    with pytest.raises(NaturalConversationCatalogError):
        build_catalog(
            ultrachat_candidates=ultra,
            oasst_candidates=[],
            source_corpora=[],
            maximum_per_capability=2,
            minimum_per_capability=2,
            seed=7,
        )


def test_label_evidence_binds_source_provenance() -> None:
    ultra = list(
        _ultrachat_candidates(
            [_ultra("Write a friendly note apologizing for arriving late.")],
            shard_sha256=SHA,
        )
    )
    catalog = build_catalog(
        ultrachat_candidates=ultra,
        oasst_candidates=[],
        source_corpora=[],
        maximum_per_capability=1,
        minimum_per_capability=1,
        seed=7,
    )
    probe = catalog["probes"][0]
    assert probe["source_shard_sha256"] == SHA
    assert probe["natural_prompt_sha256"] == hashlib.sha256(
        "Write a friendly note apologizing for arriving late.".encode("utf-8")
    ).hexdigest()
