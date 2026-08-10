from pathlib import Path

from abi.capability_compiler_phase3_structural_core import _json


def test_structural_core_json_requires_object(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text("{}", encoding="utf-8")
    assert _json(path) == {}
