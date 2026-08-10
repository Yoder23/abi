from abi.capability_compiler_phase3_direct_linear_feasibility import accounting
def test_direct_linear_accounting():
    a=accounting(); assert a["deployed_parameters"]==277_220_352; assert a["fp16_payload_bytes"]==554_440_704; assert a["active_incremental_macs_at_maximum_context"]==184_857_600
