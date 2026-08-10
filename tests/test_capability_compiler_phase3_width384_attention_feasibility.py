from abi.capability_compiler_phase3_width384_attention_feasibility import accounting
def test_width384_envelope():
 a=accounting(); assert a["deployed_parameters"]==451_814_400; assert a["source_to_target_active_mac_ratio"]>10
