import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_certificate_readiness import (
    EXPECTED_CONTROLS,
    validate_protocol,
)


def test_expected_controls_are_complete_and_ordered():
    assert EXPECTED_CONTROLS == (
        "parent",
        "A1_label_free",
        "A2_shuffled",
        "A3_bridge_only",
        "A4_monolithic",
    )


def test_protocol_rejects_cross_lineage_splicing(tmp_path: Path):
    protocol = {
        "format": "abi-capability-compiler-phase3-certificate-readiness/1",
        "status": "PREREGISTERED_READ_ONLY_CERTIFICATE_READINESS_AUDIT",
        "final_test_access": "PROHIBITED",
        "historical_evidence_mutation": "PROHIBITED",
        "required_final_controls": list(EXPECTED_CONTROLS),
        "cross_lineage_evidence_splicing": "ALLOWED",
        "bindings": {},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance changed"):
        validate_protocol(tmp_path, path)
