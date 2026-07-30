from __future__ import annotations

import pytest

from abi.layercake_core_loader import ABIEnglishCoreConfig


def test_versioned_core_config_accepts_only_locked_depths() -> None:
    assert ABIEnglishCoreConfig(layers=3).layers == 3
    assert ABIEnglishCoreConfig(
        layers=6,
        architecture_version=(
            "layercake-shallow-sparse-english/2-six-block-task-cakes"
        ),
    ).layers == 6
    with pytest.raises(ValueError, match="three or six"):
        ABIEnglishCoreConfig(layers=4)


def test_versioned_core_config_preserves_sparse_topology() -> None:
    with pytest.raises(ValueError, match="instruction-cake topology"):
        ABIEnglishCoreConfig(task_cakes=9)
    with pytest.raises(ValueError, match="width or head"):
        ABIEnglishCoreConfig(width=512)
