from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator

from experiments.native_transfer_r8.capability_generator import (
    committed_heldout_capabilities,
    generate_rows,
    public_capabilities,
    worker_rows,
)
from experiments.native_transfer_r8.extract_capability import (
    ExtractionError,
    load_package,
    write_package_once,
)
from experiments.native_transfer_r8.freeze_campaign import FreezeError, freeze
from experiments.native_transfer_r8.native_host import CanonicalLatentBridge
from experiments.native_transfer_r8.recipient_worker import _random_latent
from experiments.native_transfer_r8.run_baselines import DenseLinearBridge, LoRALinear
from experiments.native_transfer_r8.verify import R8VerificationError, verify

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/native_transfer_r8/configs/preregistered_v2.json"
SECRET = "6ee11dd33028a7a701c32e58203ec96bfc0f38eb3808b8bf8e68b8cafdde11ae"


def test_preregistration_locks_primary_scientific_depth() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["status"] == "PREREGISTERED_BEFORE_HELDOUT_REVEAL"
    assert value["splits"]["meta_train_capabilities"] >= 24
    assert value["splits"]["development_capabilities"] >= 4
    assert value["splits"]["heldout_capabilities"] >= 8
    assert value["splits"]["evaluation_rows_per_capability"] >= 512
    assert value["gates"]["recipient_families_minimum"] == 3
    assert len(value["models"]["recipients"]) == 3
    assert value["supersedes"] == "preregistered_v1.json"
    assert value["amendment"]["scientific_thresholds_changed"] is False
    schema = json.loads(
        (ROOT / "experiments/native_transfer_r8/preregistration.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(value)


def test_source_training_rows_are_unique_and_within_registered_universe() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    capability = public_capabilities(123, split="meta_train", count=1)[0]
    count = config["splits"]["source_train_rows_per_capability"]
    rows = generate_rows(
        capability,
        split="source_train",
        rows=count,
        depths=config["capability_family"]["source_train_depths"],
        seed=9,
    )
    assert len(rows) == count == len({row["row_id"] for row in rows})
    assert sum(row["flavor"] == "atomic_coverage" for row in rows) == 24
    meta = public_capabilities(123, split="meta_train", count=24)
    assert len({value.offsets for value in meta}) == len(meta)


def test_committed_heldout_capabilities_are_deterministic_and_private() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    values = committed_heldout_capabilities(
        SECRET,
        expected_commitment=config["splits"]["heldout_secret_commitment_sha256"],
        count=8,
    )
    repeated = committed_heldout_capabilities(
        SECRET,
        expected_commitment=config["splits"]["heldout_secret_commitment_sha256"],
        count=8,
    )
    assert values == repeated
    assert len({value.capability_id for value in values}) == 8
    with pytest.raises(Exception):
        committed_heldout_capabilities(
            "00" * 32,
            expected_commitment=config["splits"]["heldout_secret_commitment_sha256"],
            count=8,
        )


def test_worker_rows_remove_every_private_answer_field() -> None:
    capability = public_capabilities(123, split="meta_train", count=1)[0]
    rows = generate_rows(
        capability, split="heldout_evaluation", rows=64, depths=(4, 5, 6, 7), seed=9
    )
    public = worker_rows(rows)
    forbidden = {"answer", "offsets", "seed", "program", "start", "label"}
    assert all(not forbidden.intersection(row) for row in public)
    assert all(row["prompt_sha256"] for row in public)


def test_package_is_one_host_neutral_immutable_tensor(tmp_path: Path) -> None:
    latent = torch.softmax(torch.randn(3, 8, 8), dim=-1)
    package = tmp_path / "capability.abipkg"
    receipt = write_package_once(
        package,
        latent,
        capability_id="r8-heldout-test",
        reveal_commitment_sha256="a" * 64,
        source_before_sha256="b" * 64,
        source_after_sha256="c" * 64,
    )
    value, restored = load_package(package)
    assert torch.equal(latent, restored)
    assert receipt["forbidden_term_matches"] == []
    assert value["host_specific_payloads"] == 0
    assert value["executable_payloads"] == 0
    mutated = package.read_text(encoding="utf-8").replace(
        '"host_specific_payloads": 0', '"host_specific_payloads": 1'
    )
    package.write_text(mutated, encoding="utf-8")
    with pytest.raises(ExtractionError):
        load_package(package)


class _Tokenizer:
    def encode(self, _text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [0]


class _DummyHost:
    hidden_width = 16
    device = torch.device("cpu")
    target_token_ids = list(range(8))
    tokenizer = _Tokenizer()
    embedding = torch.nn.Embedding(8, 16)

    class _Spec:
        key = "dummy"

    spec = _Spec()


def test_bridge_is_prompt_blind_and_zero_control_is_finite() -> None:
    bridge = CanonicalLatentBridge(_DummyHost())
    assert not hasattr(bridge, "prompt")
    real = torch.softmax(torch.randn(2, 3, 8, 8), dim=-1)
    assert bridge(real).shape == (2, 24, 16)
    assert torch.isfinite(bridge(torch.zeros(3, 8, 8))).all()
    bridge.freeze()
    assert not any(parameter.requires_grad for parameter in bridge.parameters())


def test_recipient_worker_does_not_import_generator_or_scorer() -> None:
    source = (ROOT / "experiments/native_transfer_r8/recipient_worker.py").read_text(
        encoding="utf-8"
    )
    assert "capability_generator" not in source
    assert "evaluator_private" not in source
    assert "answer" not in source.split("def _jsonl", 1)[0]


def test_freeze_fails_closed_when_required_components_are_missing(tmp_path: Path) -> None:
    with pytest.raises(FreezeError):
        freeze(ROOT, CONFIG, tmp_path / "revision")


def test_baseline_adapters_are_neural_and_leave_lora_base_unchanged() -> None:
    dense = DenseLinearBridge(192, 24, 16)
    assert dense(torch.randn(2, 3, 8, 8)).shape == (2, 24, 16)
    base = torch.nn.Linear(16, 12)
    before = base.weight.detach().clone()
    lora = LoRALinear(base, rank=4)
    output = lora(torch.randn(3, 16)).sum()
    output.backward()
    assert torch.equal(base.weight, before)
    assert base.weight.grad is None
    assert lora.a.grad is not None and lora.b.grad is not None


def test_verifier_fails_closed_without_raw_campaign_evidence(tmp_path: Path) -> None:
    with pytest.raises(R8VerificationError):
        verify(ROOT, CONFIG, tmp_path / "missing-campaign")


def test_random_package_control_preserves_empirical_value_distribution() -> None:
    latent = torch.softmax(torch.arange(192, dtype=torch.float32).reshape(3, 8, 8), dim=-1)
    randomized = _random_latent(latent, "heldout-test")
    assert torch.equal(torch.sort(latent.flatten()).values, torch.sort(randomized.flatten()).values)
    assert torch.allclose(randomized.sum(dim=-1), torch.ones(3, 8))
    assert not torch.equal(latent, randomized)
