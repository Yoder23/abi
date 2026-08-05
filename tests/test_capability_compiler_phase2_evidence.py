import pytest

from abi.capability_compiler_phase2_common import Phase2Error
from abi.capability_compiler_phase2_evidence import validate_runtime_result


def _runtime(mode: str):
    observation = {
        "output_tokens": 1,
        "time_to_first_output_seconds": 0.1,
        "total_seconds": 0.2,
    }
    if mode == "cold":
        observation.update(
            first_output_from_cold_start_seconds=1.1,
            total_from_cold_start_seconds=1.2,
        )
    count = 1 if mode == "cold" else 20
    return {
        "format": "abi-capability-compiler-phase2-runtime/1",
        "status": "PASS",
        "system": "D0",
        "mode": mode,
        "model_load_seconds": 1.0,
        "observation_count": count,
        "observations": [dict(observation) for _ in range(count)],
        "p95_supported": False,
        "p99_supported": False,
        "final_prompts_accessed": False,
    }


@pytest.mark.parametrize("mode", ["cold", "warm"])
def test_runtime_contract_accepts_only_registered_depth(mode):
    value = _runtime(mode)
    validate_runtime_result(value, system="D0", mode=mode)
    value["observation_count"] += 1
    with pytest.raises(Phase2Error):
        validate_runtime_result(value, system="D0", mode=mode)


def test_runtime_contract_rejects_fake_cold_start():
    value = _runtime("cold")
    value["observations"][0]["first_output_from_cold_start_seconds"] = 0.5
    with pytest.raises(Phase2Error):
        validate_runtime_result(value, system="D0", mode="cold")


def test_runtime_contract_rejects_unsupported_tail_claims():
    value = _runtime("warm")
    value["p95_supported"] = True
    with pytest.raises(Phase2Error):
        validate_runtime_result(value, system="D0", mode="warm")
