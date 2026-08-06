from abi.capability_compiler_phase3_representation_bakeoff import encode_row, spell_selected


def test_spell_selection_is_deterministic():
    assert spell_selected(b"alpha", 10) == spell_selected(b"alpha", 10)


def test_hybrid_fallback_is_lossless_and_unicode_atomic():
    def split(value):
        return [str(value).encode("utf-8")]
    row = {"record_id": "x", "capability": "grammar", "prompt": "prompt", "output": "café"}
    chars = {char.encode("utf-8") for char in "café"}
    _, actions = encode_row(row, split, {b"prompt"}, chars, mode="hybrid", exposure_modulus=None)
    assert b"".join(value for _, value in actions) == "café".encode("utf-8")
    assert all(value.decode("utf-8") for _, value in actions)
