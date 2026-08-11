from abi.capability_compiler_phase4_b100_exposure_normalized import STEP_TARGETS


def test_only_three_parent_stage_step_targets_change():
    assert STEP_TARGETS == {
        "ABI_CAPABILITY_COMPILER_PHASE3_QUALIFIED_TRANSITION_CONTROL_PROTOCOL_V440.json": 8750,
        "ABI_CAPABILITY_COMPILER_PHASE3_COPY_BALANCED_TRANSITION_PROTOCOL_V458.json": 2188,
        "ABI_CAPABILITY_COMPILER_PHASE3_TOKEN_SUBSTRATE_CONFORMANCE_PROTOCOL_V462.json": 2188,
    }
