from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from abi_v2.verify_host_independence import (
    HostIndependenceVerificationError,
    _verify_bindings,
    verify,
)

ROOT = Path(__file__).resolve().parents[1]


def test_host_independence_release_verifies() -> None:
    certificate = verify(ROOT)
    assert certificate["technical_moonshot"] == "ABI TECHNICAL MOONSHOT: PROVEN"
    assert certificate["winning_family"] == "FAMILY_A_OBSERVABLE_CANONICAL_STATE"


def test_evidence_binding_mutation_fails_closed() -> None:
    certificate = json.loads(
        (ROOT / "results/abi_host_independence/release_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    mutated = copy.deepcopy(certificate)
    mutated["evidence_bindings"]["matrix_summary"]["sha256"] = "0" * 64
    with pytest.raises(HostIndependenceVerificationError, match="evidence changed"):
        _verify_bindings(ROOT, mutated)


def test_unsafe_evidence_binding_path_fails_closed() -> None:
    certificate = json.loads(
        (ROOT / "results/abi_host_independence/release_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    mutated = copy.deepcopy(certificate)
    mutated["evidence_bindings"]["matrix_summary"]["path"] = "../outside.json"
    with pytest.raises(HostIndependenceVerificationError, match="unsafe binding path"):
        _verify_bindings(ROOT, mutated)
