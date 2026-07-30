import json

import pytest

from abi.layercake_product_host import ProductHostError
from abi.layercake_product_host_certification import (
    PROTOCOL_FORMAT,
    _claimed_hash,
)


def test_claimed_hash_rejects_changed_evidence():
    value = {"status": "PASS"}
    from abi.layercake_product_host_certification import _canonical_sha

    value["evidence_sha256"] = _canonical_sha(value)
    changed = dict(value)
    changed["status"] = "FAIL"
    with pytest.raises(ProductHostError, match="claim hash"):
        _claimed_hash(changed)


def test_combined_protocol_is_json_and_preregistered():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    value = json.loads(
        (root / "COMBINED_LAYERCAKE_HOST_CERTIFICATION_PROTOCOL.json").read_text(
            encoding="utf-8"
        )
    )
    assert value["format"] == PROTOCOL_FORMAT
    assert (
        value["status"]
        == "PREREGISTERED_BEFORE_COMBINED_HOST_CERTIFICATION"
    )
    assert value["final_test_accessed"] is False
