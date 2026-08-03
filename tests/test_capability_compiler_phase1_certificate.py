import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase1_certificate import (
    Phase1CertificateError,
    verify_certificate,
    verify_certificate_data,
)


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json"


def test_phase1_certificate_verifies():
    result = verify_certificate(CERTIFICATE)
    assert result["status"] == "PASS"
    assert result["phase2"] == "OPEN_NOT_STARTED"
    assert result["training_performed"] is False


def test_phase1_certificate_status_tampering_fails():
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    certificate["status"] = "PASSISH"
    with pytest.raises(Phase1CertificateError, match="not certified"):
        verify_certificate_data(certificate, root=ROOT)
