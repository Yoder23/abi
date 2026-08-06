from abi.capability_compiler_phase3_expanded_oracle import EXPANDED_TRAINABLE_PARAMETERS, OUTPUT_RANK, SEQUENCE_RANK


def test_expanded_bridge_is_material_but_bounded():
    assert SEQUENCE_RANK == 256
    assert OUTPUT_RANK == 128
    assert 1_057_798 < EXPANDED_TRAINABLE_PARAMETERS < 3_000_000
