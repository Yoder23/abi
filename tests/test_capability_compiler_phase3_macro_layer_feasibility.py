from abi.capability_compiler_phase3_macro_layer_feasibility import active_layer_macs


def test_macro_active_macs_are_lower_than_v15():
    current = 32 * active_layer_macs(3072, 192, 384, 768)
    macro = 16 * active_layer_macs(3072, 384, 704, 1280)
    assert current == 320864256
    assert macro == 303824896
    assert macro < current
