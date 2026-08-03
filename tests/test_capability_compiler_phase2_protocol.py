import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase2_common import Phase2Error
from abi.capability_compiler_phase2_verify import verify_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE2_PROTOCOL_V1.json"


def _mutated(tmp_path: Path, mutate) -> Path:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_real_phase2_preregistration_passes():
    result = verify_protocol(ROOT)
    assert result["status"] == "PASS"
    assert result["candidate_training_performed"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("candidate_training_performed_before_preregistration", True),
        lambda value: value["splits"].__setitem__("final_access", "ALLOWED"),
        lambda value: value["statistics"].__setitem__("bootstrap_resamples", 10),
        lambda value: value["implementation_bindings"].__setitem__(
            "abi/capability_compiler_phase2_common.py", "0" * 64
        ),
        lambda value: value["implementation_bindings"].__setitem__(
            "../outside.py", "0" * 64
        ),
    ],
)
def test_phase2_preregistration_mutations_fail_closed(tmp_path, mutate):
    with pytest.raises(Phase2Error):
        verify_protocol(ROOT, _mutated(tmp_path, mutate))
