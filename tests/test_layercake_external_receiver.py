import json
from pathlib import Path

import pytest
import torch

from abi.failure_attribution import AttributionError
from abi.layercake_external_receiver import (
    PAYLOAD_FORMAT,
    NATIVE_PAYLOAD_ROLE,
    _generate,
    _manifest_sha,
    _mapping_or_empty,
    _select_token,
    _sha256_file,
    _state_contract,
    _verified_package,
    _parser,
)


def _control():
    return {
        "architecture_id": "exact-architecture",
        "architecture_hash": "a" * 64,
        "canonical_semantic_abi_file": {"sha256": "b" * 64},
        "native_runtime_artifact": {
            "tokenizer": {"sha256": "c" * 64},
        },
    }


def _payload(tmp_path: Path) -> Path:
    package = tmp_path / "payload"
    package.mkdir()
    payload = package / "payload.safetensors"
    payload.write_bytes(b"exact payload")
    manifest = {
        "format": PAYLOAD_FORMAT,
        "status": "SEALED_NATIVE_POSITIVE_CONTROL",
        "target": {
            "architecture_id": "exact-architecture",
            "architecture_hash": "a" * 64,
            "canonical_semantic_abi_sha256": "b" * 64,
        },
        "payload": {
            "path": payload.name,
            "bytes": payload.stat().st_size,
            "sha256": _sha256_file(payload),
        },
    }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    (package / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return package


def test_state_contract_binds_names_shapes_dtypes_and_bytes():
    first = _state_contract({"weight": torch.tensor([[1.0, 2.0]])})
    second = _state_contract({"weight": torch.tensor([[1.0, 3.0]])})
    assert first["tensor_count"] == 1
    assert first["parameter_count"] == 2
    assert first["state_sha256"] != second["state_sha256"]


def test_optional_null_accounting_section_normalizes_to_empty_mapping():
    assert _mapping_or_empty(None) == {}
    assert _mapping_or_empty({"source_parameters_copied": 0}) == {
        "source_parameters_copied": 0
    }


def test_locked_native_decoding_blocks_repeated_fourgram():
    logits = torch.tensor([[0.0, 8.0, 9.0]])
    generated = [1, 2, 1, 2, 1, 2, 1]
    assert _select_token(logits, generated, no_repeat_ngram_size=4) == 1


def test_locked_natural_screen_decoding_stops_before_eos():
    class Tokenizer:
        eos_token_id = 2

        @staticmethod
        def encode(value):
            assert value == "Prompt\n"
            return [1]

        @staticmethod
        def decode(tokens, **_kwargs):
            return "should be empty" if tokens else ""

    class Model:
        @staticmethod
        def prefill(input_ids):
            assert input_ids.tolist() == [[1]]
            return {
                "next_logits": torch.tensor([[0.0, 1.0, 9.0]]),
                "task_routes": torch.tensor([3]),
                "past_key_values": tuple(
                    (torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))
                    for _ in range(3)
                ),
            }

        @staticmethod
        def decode_step(*_args, **_kwargs):
            raise AssertionError("EOS must not be appended or decoded")

    output, tokens, route, cache_lengths = _generate(
        Model(), Tokenizer(), "Prompt", max_new_tokens=8
    )
    assert output == ""
    assert tokens == []
    assert route == 3
    assert cache_lengths == [1, 1, 1]


def test_payload_package_rejects_mutation(tmp_path):
    package = _payload(tmp_path)
    _, payload = _verified_package(
        package, expected_format=PAYLOAD_FORMAT, control=_control()
    )
    payload.write_bytes(b"mutated payload")
    with pytest.raises(AttributionError, match="payload bytes changed"):
        _verified_package(
            package, expected_format=PAYLOAD_FORMAT, control=_control()
        )


def test_payload_package_rejects_path_escape(tmp_path):
    package = _payload(tmp_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload"]["path"] = "../outside.safetensors"
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AttributionError, match="escapes"):
        _verified_package(
            package, expected_format=PAYLOAD_FORMAT, control=_control()
        )


def test_native_quality_scope_command_requires_matched_control_inputs():
    args = _parser().parse_args(
        [
            "evaluate-native-quality-scope",
            "--contract",
            "contract.json",
            "--layercake-root",
            "layercake",
            "--receiver",
            "receiver",
            "--payload",
            "payload",
            "--catalog",
            "catalog.json",
            "--native-same-path-evidence",
            "same-path.json",
            "--output",
            "evidence.json",
        ]
    )
    assert args.command == "evaluate-native-quality-scope"
    assert args.native_same_path_evidence == "same-path.json"
    assert NATIVE_PAYLOAD_ROLE == "known_good_layercake_native_payload"
