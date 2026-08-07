from abi.capability_compiler_phase3_bpe_pointer_resilience import _pointer_targets


class TinyTokenizer:
    lexeme_to_id = {b"ID": 4, b" ": 5, b"fixed": 6}


def test_pointer_targets_copy_unique_identity_pieces_only() -> None:
    source = [b"route", b" ", b"ID", b" ", b"fixed", b" ", b"ID2", b" ", b"ID2"]
    output = [b"ID", b" ", b"fixed", b" ", b"ID2"]
    tokenizer = TinyTokenizer()
    tokenizer.lexeme_to_id[b"ID2"] = 7
    assert _pointer_targets(source, output, 10, tokenizer) == [12, 5, 14, 5, 7, 2]
