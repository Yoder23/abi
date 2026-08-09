from abi.capability_compiler_phase3_span_copy_feasibility import _encode


class Tokenizer:
    vocab_size = 10
    lexeme_to_id = {b"route": 4, b"\n": 5, b"alpha": 6, b"beta": 7}

    def split(self, value):
        value = value.encode() if isinstance(value, str) else value
        return {b"\nalpha": [b"\n", b"alpha"], b"alpha": [b"alpha"], b"beta": [b"beta"]}[value]

    def decode_actions(self, actions, source):
        inverse = {value: key for key, value in self.lexeme_to_id.items()}
        return b"".join(source[action - self.vocab_size] if action >= self.vocab_size else inverse[action] for action in actions if action != 2)


def test_encode_uses_exact_unique_lexeme_pointer() -> None:
    value = _encode(Tokenizer(), b"route", "alpha", "alpha")
    assert value["roundtrip"] is True
    assert value["pointer_actions"] == 1
    assert value["source_actions"] == 3
