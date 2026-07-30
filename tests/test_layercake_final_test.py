import json
from pathlib import Path

from abi.layercake_final_test import PROTOCOL_FORMATS, _load_protocol


def test_final_protocol_is_complete_and_preregistered():
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "ABI_MOONSHOT_FINAL_TEST_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["format"] in PROTOCOL_FORMATS
    assert protocol["final_test_open_authorized"] is True
    assert protocol["source_output_contract"]["admissible_for_training"] is False
    assert protocol["candidate_gates"]["observations_total"] == 1700
    loaded_root, loaded = _load_protocol(protocol_path)
    assert loaded_root == root
    assert loaded == protocol
