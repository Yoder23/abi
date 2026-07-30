from pathlib import Path

import pytest

from abi.layercake_product_host import ProductHostError, verify_domain_package


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "results"
    / "abi_moonshot"
    / "packages"
    / "abi-python-token-plan-seed9824.cake"
)
PUBLIC = PACKAGE.with_suffix(".pub")


def test_lightweight_package_verifier_accepts_certified_package():
    result = verify_domain_package(PACKAGE, PUBLIC)
    assert result["cake_id"] == "abi-python-token-plan"
    assert result["domain"] == "python"
    assert result["signed"] is True
    assert result["teacher_present"] is False


def test_lightweight_package_verifier_rejects_tampering(tmp_path):
    tampered = tmp_path / "tampered.cake"
    raw = bytearray(PACKAGE.read_bytes())
    raw[len(raw) // 2] ^= 1
    tampered.write_bytes(raw)
    with pytest.raises(ProductHostError):
        verify_domain_package(tampered, PUBLIC)


def test_lightweight_package_verifier_rejects_wrong_key():
    wrong = (
        ROOT
        / "results"
        / "abi_moonshot"
        / "packages"
        / "abi-civics-token-plan-seed9824.pub"
    )
    with pytest.raises(ProductHostError, match="publisher identity"):
        verify_domain_package(PACKAGE, wrong)
