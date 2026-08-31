from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.copy_paste_r10.runtime import (
    CanonicalTransitionVM,
    CopyPasteRuntimeError,
    apply_host_codec,
    build_package,
    discover_canonical_token_map,
    load_package,
    write_package_once,
)
from experiments.copy_paste_r10.slot import CanonicalCapabilitySlot
from experiments.native_transfer_r8.capability_generator import (
    OpaqueCapability,
    generate_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def _transition(offsets: tuple[int, int, int]) -> torch.Tensor:
    value = torch.zeros(3, 8, 8)
    for operation, offset in enumerate(offsets):
        for start in range(8):
            value[operation, start, (start + offset) % 8] = 1.0
    return value


def test_proof_ledger_keeps_runtime_and_native_claims_separate() -> None:
    ledger = json.loads((ROOT / "docs/abi_proof_ledger.json").read_text(encoding="utf-8"))
    claims = {item["id"]: item for item in ledger["claims"]}
    assert claims["ABI-C4"]["state"] == "OPEN_R10_TARGET"
    assert claims["ABI-C5"]["state"] == "FAILED_TESTED_R8_R9_MECHANISMS"
    assert claims["ABI-C8"]["state"] == "NOT_PASSED"
    assert ledger["promotion_boundary"]["r10_may_promote"] == ["ABI-C4"]


def test_vm_executes_registered_normal_and_hostile_surfaces() -> None:
    capability = OpaqueCapability(capability_id="test", offsets=(2, 5, 3), seed_commitment="a" * 64)
    rows = generate_rows(
        capability,
        split="r10_copy_paste_evaluation",
        rows=64,
        depths=[4, 5, 6, 7],
        seed=19,
    )
    outputs = CanonicalTransitionVM().execute(
        _transition(capability.offsets), [str(row["prompt"]) for row in rows]
    )
    assert outputs.argmax(dim=-1).tolist() == [row["answer"] for row in rows]
    assert any(str(row["prompt"]).startswith("Counterfactual") for row in rows)
    assert any(str(row["prompt"]).startswith("Incorrect proposal") for row in rows)


def test_vm_fails_closed_outside_registered_grammar() -> None:
    vm = CanonicalTransitionVM()
    with pytest.raises(CopyPasteRuntimeError):
        vm.execute(_transition((1, 2, 3)), ["What is 2 + 2?"])
    with pytest.raises(CopyPasteRuntimeError):
        vm.execute(
            _transition((1, 2, 3)),
            ["Opaque program: start 0 ; apply vok ; result = 7"],
        )


def test_package_is_content_addressed_and_rejects_tampering(tmp_path: Path) -> None:
    latent = _transition((2, 5, 3))
    provenance = {
        "source_receipt_sha256": "a" * 64,
        "extraction_receipt_sha256": "b" * 64,
    }
    item = write_package_once(tmp_path, latent, provenance)
    package, restored = load_package(tmp_path / item["path"])
    assert torch.equal(latent, restored)
    assert set(package) == {
        "format",
        "family",
        "interpreter_abi",
        "latent_dtype",
        "latent_hex",
        "latent_sha256",
        "latent_shape",
        "provenance",
    }
    path = tmp_path / item["path"]
    path.write_bytes(path.read_bytes().replace(b'"family"', b'"prompt"', 1))
    with pytest.raises(CopyPasteRuntimeError):
        load_package(path)


def test_package_schema_rejects_unregistered_provenance() -> None:
    with pytest.raises(CopyPasteRuntimeError):
        build_package(_transition((1, 2, 3)), {"source_receipt_sha256": "a" * 64})


def test_host_codec_realizes_vm_distribution_without_mutating_base() -> None:
    base = torch.tensor([[9.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]])
    untouched = base.clone()
    distribution = torch.nn.functional.one_hot(torch.tensor([6]), num_classes=8).float()
    target_ids = list(range(1, 9))
    result = apply_host_codec(base, distribution, target_ids, margin=20.0)
    assert torch.equal(base, untouched)
    assert int(result.argmax(dim=-1)) == target_ids[6]
    assert result.shape == base.shape


def test_canonical_token_discovery_accepts_only_exact_isolated_decode() -> None:
    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            value = int(text.strip())
            return [99, 100 + value] if value == 0 else [100 + value]

        def decode(self, ids: list[int], **_: object) -> str:
            return "" if ids == [99] else str(ids[0] - 100)

    ids, texts = discover_canonical_token_map(Tokenizer(), encoder_decoder=True)
    assert ids == list(range(100, 108))
    assert texts == list("01234567")


def test_slot_paste_remove_restore_uses_identical_package_bytes(tmp_path: Path) -> None:
    capability = OpaqueCapability(capability_id="test", offsets=(2, 5, 3), seed_commitment="a" * 64)
    row = generate_rows(capability, split="slot_test", rows=1, depths=[5], seed=101)[0]
    item = write_package_once(
        tmp_path,
        _transition(capability.offsets),
        {
            "source_receipt_sha256": "a" * 64,
            "extraction_receipt_sha256": "b" * 64,
        },
    )
    path = tmp_path / item["path"]
    slot = CanonicalCapabilitySlot()
    first = slot.paste(path)
    assert int(slot.execute([row["prompt"]]).argmax(dim=-1)) == row["answer"]
    slot.remove()
    with pytest.raises(CopyPasteRuntimeError):
        slot.execute([row["prompt"]])
    restored = slot.paste(path)
    assert restored["package_sha256"] == first["package_sha256"]
    assert int(slot.execute([row["prompt"]]).argmax(dim=-1)) == row["answer"]
