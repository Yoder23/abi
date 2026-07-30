from __future__ import annotations

import json
from pathlib import Path

import pytest

from abi.domain_package_certification import (
    PROTOCOL_FORMAT,
    _load_protocol,
)
from abi.layercake_domains import DomainConformanceError


def _protocol() -> dict:
    return {
        "format": PROTOCOL_FORMAT,
        "status": "PREREGISTERED_BEFORE_PACKAGE_RUNTIME_CERTIFICATION",
        "immutable_layercake_target": {
            "repository_commit": (
                "04cf2927a16fba686cd640e18a78708e5658bbda"
            ),
            "direct_decoder_abi_version": "lc-direct-neural-decoder/1",
            "direct_decoder_abi_sha256": (
                "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
            ),
            "sealed_repository_may_be_modified": False,
        },
        "required_seeds": [9824, 9825, 9826],
        "receiver_count": 3,
        "domains": [
            {"domain": "chemistry"},
            {"domain": "civics"},
            {"domain": "python"},
        ],
    }


def test_package_protocol_requires_exact_three_seed_receiver_contract(
    tmp_path: Path,
):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(_protocol()), encoding="utf-8")
    assert _load_protocol(path)["required_seeds"] == [9824, 9825, 9826]


def test_package_protocol_rejects_duplicate_domains(tmp_path: Path):
    value = _protocol()
    value["domains"][2]["domain"] = "civics"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DomainConformanceError, match="domains"):
        _load_protocol(path)
