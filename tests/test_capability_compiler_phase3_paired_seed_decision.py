import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_paired_seed_decision import _rows


def test_rows_rejects_wrong_depth(tmp_path):
    path = tmp_path / "outputs.jsonl"
    path.write_text('{"probe_id":"only"}\n', encoding="utf-8")
    with pytest.raises(Phase3Error, match="depth changed"):
        _rows(path)
