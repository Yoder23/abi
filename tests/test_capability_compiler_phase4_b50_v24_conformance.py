from abi.capability_compiler_phase4_b50_v24_conformance import (
    FORMAT,
    RESULT_FORMAT,
)


def test_v24_conformance_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-v24-conformance/1"
    assert RESULT_FORMAT == (
        "abi-capability-compiler-phase4-b50-v24-conformance-result/1"
    )
