from abi.capability_compiler_phase4_b40_baseline_gpu_runtime import (
    FORMAT,
    RESULT_FORMAT,
    SYSTEMS,
)


def test_b40_lora_runtime_contract_is_budget_specific():
    assert FORMAT == "abi-capability-compiler-phase4-b40-baseline-gpu-runtime/1"
    assert RESULT_FORMAT == "abi-capability-compiler-phase4-b40-baseline-gpu-runtime-result/1"
    assert SYSTEMS == ("L0", "L1")
