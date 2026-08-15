import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_clarification_route_verify import _json


def test_json_rejects_non_object(tmp_path):
    path = tmp_path / "value.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(Phase3Error, match="expected JSON object"):
        _json(path)


def test_json_accepts_object(tmp_path):
    path = tmp_path / "value.json"
    path.write_text('{"status":"ok"}', encoding="utf-8")
    assert _json(path) == {"status": "ok"}
