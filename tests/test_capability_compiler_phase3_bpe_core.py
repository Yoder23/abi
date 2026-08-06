from abi.capability_compiler_phase3_bpe_core import FORMAT, _examples


class TinyTokenizer:
    vocab_size = 6
    lexeme_to_id = {b"a": 4, b"b": 5}

    @staticmethod
    def split(value):
        return [character.encode("utf-8") for character in value]

    @staticmethod
    def decode_actions(actions, source_lexemes):
        table = {4: b"a", 5: b"b"}
        return b"".join(table[action] for action in actions if action != 2)


def test_format_is_locked():
    assert FORMAT == "abi-capability-compiler-phase3-bpe-core-screen/1"


def test_examples_are_fixed_action_lossless():
    rows = [
        {
            "ir_record_id": f"r{index}",
            "capability": "grammar",
            "normalized_acquisition_prompt": "ab",
            "normalized_output": "ba",
            "authoritative_teacher_tokens": 2,
        }
        for index in range(7000)
    ]
    examples = _examples(
        rows,
        TinyTokenizer(),
        2,
        {"maximum_source_lexemes": 2, "maximum_target_actions": 3},
    )
    assert examples[0]["source_ids"] == [4, 5]
    assert examples[0]["target_actions"] == [5, 4, 2]
    assert all(action < TinyTokenizer.vocab_size for action in examples[0]["target_actions"])
