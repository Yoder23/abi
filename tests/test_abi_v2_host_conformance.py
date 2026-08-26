from __future__ import annotations

import json
import math
import zipfile
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
    _adapter_document,
    _neutral_texts,
)
from abi_v2.isolated_certification import (
    _inspect_regular_file,
    build_capsule,
    verify_capsule,
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


def test_certification_capsule_physically_excludes_capabilities_and_success_ids(
    tmp_path: Path,
) -> None:
    capsule = tmp_path / "capsule"
    manifest = build_capsule(ROOT, host_key="layercake", destination=capsule)
    receipt = verify_capsule(capsule)
    paths = {row["path"] for row in manifest["files"]}
    assert receipt["capability_archives_present"] == 0
    assert receipt["source_success_ledgers_present"] == 0
    assert not any(Path(path).suffix in {".abi", ".cake", ".abix", ".abicir"} for path in paths)
    assert not any("source_success" in path.casefold() for path in paths)


def test_content_scanner_detects_renamed_capability_archive(tmp_path: Path) -> None:
    disguised = tmp_path / "neutral-runtime-cache.bin"
    with zipfile.ZipFile(disguised, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"cake_id": "test", "cake_type": "domain", "abi_hash": "0" * 64}),
        )
        archive.writestr("tensors.safetensors", b"not-a-real-tensor")
        archive.writestr("signature.json", b"{}")
    row = _inspect_regular_file(disguised)
    assert row["capability_archive_signatures"] == ["layercake-capability-package"]


def test_content_scanner_detects_success_id_in_neutral_file(tmp_path: Path) -> None:
    disguised = tmp_path / "neutral-runtime-cache.bin"
    disguised.write_bytes(b"prefix phase1-validation-grounding-0042-v3 suffix")
    row = _inspect_regular_file(disguised)
    assert row["campaign_identifier_matches"] == 1


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
