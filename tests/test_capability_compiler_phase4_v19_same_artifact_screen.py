import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_v19_same_artifact_screen import compare_ordinary, load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_V19_SAME_ARTIFACT_PROTOCOL_V714.json"


def test_exact_row_comparison_is_symmetric_across_all_bound_fields():
    row = {
        "probe_id": "p",
        "capability": "coherence",
        "output": "x",
        "original_output": "x",
        "output_token_ids": [1],
        "automatic_capability_route": "coherence",
        "control_residual_route": 0,
        "task_route": 2,
        "guard_terminated": False,
    }
    assert all(compare_ordinary(row, dict(row)).values())
    changed = dict(row)
    changed["output"] = "y"
    assert compare_ordinary(changed, row)["output"] is False


def test_changed_package_binding_fails_closed(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["bindings"][protocol["package"]] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="binding changed"):
        load_protocol(ROOT, changed)
