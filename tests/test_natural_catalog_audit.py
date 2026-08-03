from __future__ import annotations

import json
from pathlib import Path

from abi.natural_catalog_audit import audit_catalog
from abi.natural_conversation_catalog import (
    _ultrachat_candidates,
    build_catalog,
)


SHA = "a" * 64


def test_audit_accepts_bound_search_catalog(tmp_path: Path) -> None:
    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a friendly note apologizing for arriving late.",
                },
                {
                    "role": "assistant",
                    "content": "This corpus answer must never be imported.",
                },
            ]
        }
    ]
    candidates = list(_ultrachat_candidates(rows, shard_sha256=SHA))
    catalog = build_catalog(
        ultrachat_candidates=candidates,
        oasst_candidates=[],
        source_corpora=[],
        maximum_per_capability=1,
        minimum_per_capability=1,
        seed=7,
    )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    result = audit_catalog(path)
    # Production requires 100 rows/capability, so the synthetic one-row test
    # reaches the exact expected fail-closed condition and no other condition.
    assert result["status"] == "FAIL"
    assert result["failures"] == {
        "capability_below_preregistered_minimum": 1
    }
    observations = result["observations"]
    assert observations["corpus_assistant_messages_imported"] == 0
    assert observations["corpus_reference_answers_imported"] == 0
    assert observations["unique_natural_prompt_hashes"] == 1
