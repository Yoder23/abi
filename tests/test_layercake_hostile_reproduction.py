import pytest

from abi.layercake_hostile_reproduction import _rejection
from abi.layercake_product_host import ProductHostError


def test_rejection_records_expected_failure():
    result = _rejection(
        "bad-value", lambda: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert result["status"] == "PASS_REJECTED"
    assert result["error_type"] == "ValueError"


def test_rejection_fails_when_attack_is_accepted():
    with pytest.raises(ProductHostError, match="accepted"):
        _rejection("accepted", lambda: None)
