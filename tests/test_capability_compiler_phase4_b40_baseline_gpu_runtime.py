from abi.capability_compiler_phase4_b40_baseline_gpu_runtime import (
    FORMAT,
    RESULT_FORMAT,
    SYSTEMS,
)
from abi.capability_compiler_phase3 import Phase3Error
from pathlib import Path
import json
import pytest


def test_b40_lora_runtime_contract_is_budget_specific():
    assert FORMAT == "abi-capability-compiler-phase4-b40-baseline-gpu-runtime/1"
    assert RESULT_FORMAT == "abi-capability-compiler-phase4-b40-baseline-gpu-runtime-result/1"
    assert SYSTEMS == ("L0", "L1")


def test_runtime_protocol_authorizes_only_strongest_l1(tmp_path, monkeypatch):
    from abi import capability_compiler_phase4_b40_baseline_gpu_runtime as runtime

    protocol = {
        "format": FORMAT,
        "status": "PREREGISTERED_SAME_CHECKPOINT_B40_LORA_CUDA_RUNTIME",
        "device": "cuda",
        "training_authorized": False,
        "teacher_query_generation_authorized": False,
        "source_base_loading_for_lora_authorized": True,
        "final_test_access": "PROHIBITED",
        "authorized_systems": ["L0"],
        "runtime": {
            "distinct_prompts": 100,
            "repeated_observations": 20,
            "p95_minimum_observations": 100,
            "p99_minimum_observations": 1000,
        },
        "systems": {"L0": {}, "L1": {}},
        "bindings": {},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance changed"):
        runtime.load_protocol(tmp_path, path)
