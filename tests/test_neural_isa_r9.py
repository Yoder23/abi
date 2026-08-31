from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
)
from experiments.native_transfer_r8.native_host import sha256_file
from experiments.neural_isa_r9.backend import PackageConditionedGRUBackend
from experiments.neural_isa_r9.verify_specific_diagnostic import (
    R9VerificationError,
    _bootstrap,
    _evidence,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/neural_isa_r9/configs/preregistered_v1.json"
CONFIG_V2 = ROOT / "experiments/neural_isa_r9/configs/preregistered_v2.json"


def test_r9_preregistration_is_bound_to_sealed_r8_inputs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "PREREGISTERED_BEFORE_GATE_A_EXECUTION"
    assert config["gate_a"]["capability_specific_weights_allowed"] is True
    assert config["gate_a"]["universal_decoder_claim_allowed"] is False
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("extraction_receipt", "extraction_receipt_sha256"),
        ("canonical_latents", "canonical_latents_sha256"),
    ):
        path = ROOT / config["r8_reference"][path_key]
        assert path.is_file()
        assert sha256_file(path) == config["r8_reference"][hash_key]


def test_backend_is_neural_and_shape_checked() -> None:
    backend = PackageConditionedGRUBackend(16, hidden_width=8)
    states = torch.randn(3, 7, 16)
    lengths = torch.tensor([7, 5, 3])
    package = torch.softmax(torch.randn(3, 8, 8), dim=-1)
    output = backend(states, lengths, package)
    assert output.shape == (3, 8)
    with pytest.raises(ValueError, match="canonical package shape"):
        backend(states, lengths, torch.randn(2, 8, 8))


def test_v2_repair_is_stricter_and_implementation_bound() -> None:
    config = json.loads(CONFIG_V2.read_text(encoding="utf-8"))
    assert config["supersedes"] == "preregistered_v1.json"
    assert config["gates"]["training_accuracy_minimum"] == 0.98
    assert config["gate_a"]["backend"]["recipient_state_layers"] == [
        "embedding",
        "final",
    ]
    for relative, expected in config["implementation"].items():
        assert sha256_file(ROOT / relative) == expected


def test_verifier_rejects_stale_receipts() -> None:
    value = {"format": "test", "rows": 3}
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    _evidence(value, "valid")
    value["rows"] = 4
    with pytest.raises(R9VerificationError, match="stale evidence hash"):
        _evidence(value, "tampered")


def test_bootstrap_is_deterministic_and_deep() -> None:
    first = _bootstrap([1] * 90 + [0] * 10, seed=41, replicates=1000)
    second = _bootstrap([1] * 90 + [0] * 10, seed=41, replicates=1000)
    assert first == second
    assert first["point"] == 0.9
    with pytest.raises(R9VerificationError, match="bootstrap depth"):
        _bootstrap([1], seed=41, replicates=999)
