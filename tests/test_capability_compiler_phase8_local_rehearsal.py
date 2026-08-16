from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase8_local_rehearsal import _resolve_entry


def test_resolves_abi_entry_inside_clean_root(tmp_path):
    abi = tmp_path / "abi_release"
    layercake = tmp_path / "layercake_release"
    repository, target = _resolve_entry(abi, layercake, "results/a.json")
    assert repository == "abi"
    assert target == (abi / "results/a.json").resolve()


def test_resolves_declared_sibling_layercake_entry(tmp_path):
    abi = tmp_path / "abi_release"
    layercake = tmp_path / "layercake_release"
    repository, target = _resolve_entry(
        abi, layercake, "../layercake_release/layercake/cake/package.py"
    )
    assert repository == "layercake"
    assert target == (layercake / "layercake/cake/package.py").resolve()


@pytest.mark.parametrize(
    "relative",
    [
        "/absolute",
        "../escape",
        "a/../../escape",
        "../layercake_release/../escape",
        "../layercake_release",
    ],
)
def test_release_inventory_path_escape_fails_closed(tmp_path, relative):
    with pytest.raises(Phase3Error):
        _resolve_entry(
            tmp_path / "abi_release", tmp_path / "layercake_release", relative
        )
