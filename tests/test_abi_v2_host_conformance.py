from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from abi_v2.canonical import (
    ABI_VERSION,
    CanonicalABIError,
    canonical_context,
    canonical_output_intent,
    verify_reference,
)
from abi_v2.host_certification import (
    HostCertificationError,
    _adapter_document,
    _CapabilityOpenAudit,
    _neutral_texts,
    _reject_capability_paths,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_frozen_reference_vectors_are_exact() -> None:
    payload = json.loads(
        (ROOT / "abi_v2/reference_vectors/core_vectors.json").read_text(encoding="utf-8")
    )
    states = [verify_reference(record) for record in payload["records"]]
    assert len(states) == 15
    assert all(state["abi_version"] == ABI_VERSION for state in states)
    for state in states:
        norm = math.sqrt(sum(value * value for value in state["normalized_semantic_vector_fp32"]))
        assert math.isclose(norm, 1.0, abs_tol=1e-7)


def test_canonical_state_is_deterministic_and_output_is_byte_exact() -> None:
    record = {
        "prompt": "Preserve café, 東, and 37.",
        "instruction_type": "format",
        "constraints": ["exact", "preserve_numbers"],
        "relation": "none",
        "topic": "neutral",
        "uncertainty": "certain",
        "output_intent": "exact_anchor",
        "anchors": [{"text": "café, 東, and 37", "role": "literal"}],
    }
    first = canonical_context(record)
    second = canonical_context(record)
    assert first == second
    output = canonical_output_intent("café, 東, and 37", capability_id="test-only")
    assert bytes.fromhex(output["authoritative_utf8_hex"]).decode("utf-8") == "café, 東, and 37"


@pytest.mark.parametrize(
    "mutation",
    [
        {"constraints": ["exact", "exact"]},
        {"instruction_type": "not-a-real-instruction"},
        {"anchors": [{"text": "value", "role": "not-a-real-role"}]},
    ],
)
def test_malformed_canonical_inputs_are_rejected(mutation: dict[str, object]) -> None:
    record: dict[str, object] = {
        "prompt": "Neutral.",
        "instruction_type": "answer",
        "constraints": [],
        "relation": "none",
        "topic": "neutral",
        "uncertainty": "certain",
        "output_intent": "fluent_text",
    }
    record.update(mutation)
    with pytest.raises(CanonicalABIError):
        canonical_context(record)


@pytest.mark.parametrize("suffix", [".abi", ".cake", ".abix", ".abicir"])
def test_capability_paths_are_rejected_before_certification(suffix: str) -> None:
    with pytest.raises(HostCertificationError, match="forbidden"):
        _reject_capability_paths([f"unrevealed{suffix}"])


def test_capability_open_audit_denies_payload_suffixes() -> None:
    audit = _CapabilityOpenAudit()
    with pytest.raises(PermissionError, match="unavailable"):
        audit("open", ("unrevealed.cake", "rb", 0))
    assert audit.blocked_attempts == 1


def test_frozen_adapter_is_capability_blind() -> None:
    host = {
        "host_id": "synthetic-host",
        "architecture": "SyntheticHost",
        "model": "neutral/synthetic",
        "revision": "frozen-revision",
        "checkpoint_sha256": "0" * 64,
    }
    adapter = _adapter_document(
        host_key="qwen2",
        host=host,
        spec_sha256="1" * 64,
        suite_sha256="2" * 64,
        implementation_sha256="3" * 64,
        tokenizer_mode="SyntheticTokenizer",
    )
    serialized = json.dumps(adapter, sort_keys=True).casefold()
    assert adapter["frozen"] is True
    assert adapter["trainable_parameters"] == 0
    assert adapter["optimizer_steps"] == 0
    assert adapter["capability_paths_accepted"] is False
    assert not any(domain in serialized for domain in ("python", "chemistry", "civics"))


def test_generated_certification_data_is_domain_neutral_and_unicode_valid() -> None:
    texts = _neutral_texts(128)
    assert len(texts) == len(set(texts)) == 128
    for text in texts:
        assert text.encode("utf-8").decode("utf-8") == text
        assert not any(domain in text.casefold() for domain in ("python", "chemistry", "civics"))
