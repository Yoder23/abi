from abi.capability_compiler_phase3_causal_field_feasibility import _causal_decoder_parameters


def test_causal_decoder_parameter_accounting_is_tied_and_bounded():
    value = _causal_decoder_parameters(
        vocabulary=32015,
        width=256,
        layers=6,
        feedforward=1024,
        maximum_sequence=512,
    )
    assert value == 13097999
    assert 12_000_000 <= value <= 14_407_080


def test_probability_field_payload_fits_locked_ceiling():
    actions = 432_371
    payload = actions * (32 * (2 + 2) + 2) + (14_000 + 1) * 8
    assert payload == 56_320_238
    assert payload < 128 * 1024 * 1024
