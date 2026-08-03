from __future__ import annotations

import json

import pytest

from abi.capability_compiler_phase1_extract import (
    Phase1ExtractionError,
    _attempt_row,
    load_journal,
    selected_attempts,
)


def _probe(probe_id="p1"):
    return {
        "probe_id": probe_id,
        "canonical_capability": "grammar",
        "evaluator": {"kind": "exact", "value": "Good."},
    }


def _sample(*, output="Good.", finish="eos_token"):
    return {
        "rendered_prompt": "<user>fix</user><assistant>",
        "input_tokens": 7,
        "output": output,
        "teacher_tokens": 2,
        "teacher_token_counter": "authoritative_generated_token_ids",
        "authoritative_generated_token_ids": [123, 32000],
        "finish_reason": finish,
        "generation_max_new_tokens": 20,
    }


def test_attempt_requires_eos_and_evaluator_pass():
    passed = _attempt_row(
        protocol_sha256="a" * 64,
        catalog_sha256="b" * 64,
        probe=_probe(),
        attempt_index=0,
        kind="initial",
        generation_prompt="fix",
        sample=_sample(),
    )
    length = _attempt_row(
        protocol_sha256="a" * 64,
        catalog_sha256="b" * 64,
        probe=_probe("p2"),
        attempt_index=0,
        kind="initial",
        generation_prompt="fix",
        sample=_sample(finish="length"),
    )
    wrong = _attempt_row(
        protocol_sha256="a" * 64,
        catalog_sha256="b" * 64,
        probe=_probe("p3"),
        attempt_index=0,
        kind="initial",
        generation_prompt="fix",
        sample=_sample(output="Bad."),
    )
    assert passed["functional_pass"] is True
    assert length["functional_pass"] is False
    assert wrong["functional_pass"] is False


def test_journal_roundtrip_and_tampering(tmp_path):
    row = _attempt_row(
        protocol_sha256="a" * 64,
        catalog_sha256="b" * 64,
        probe=_probe(),
        attempt_index=0,
        kind="initial",
        generation_prompt="fix",
        sample=_sample(),
    )
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_journal(path, protocol_sha256="a" * 64, catalog_sha256="b" * 64)
    selected, failed = selected_attempts([_probe()], loaded)
    assert list(selected) == ["p1"]
    assert failed == []

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["output"] = "changed"
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(Phase1ExtractionError, match="stale attempt hash"):
        load_journal(path, protocol_sha256="a" * 64, catalog_sha256="b" * 64)


def test_missing_authoritative_ids_fail_closed():
    sample = _sample()
    sample["authoritative_generated_token_ids"] = [123]
    with pytest.raises(Phase1ExtractionError, match="authoritative"):
        _attempt_row(
            protocol_sha256="a" * 64,
            catalog_sha256="b" * 64,
            probe=_probe(),
            attempt_index=0,
            kind="initial",
            generation_prompt="fix",
            sample=sample,
        )
