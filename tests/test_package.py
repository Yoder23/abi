"""Tests for the ABI package imports and basic interface."""
import importlib


def test_abi_package_importable():
    abi = importlib.import_module("abi")
    assert abi is not None


def test_abi_has_expected_submodules():
    import abi  # noqa: F401
    import abi.models  # noqa: F401
    import abi.training  # noqa: F401
    import abi.evaluation  # noqa: F401
