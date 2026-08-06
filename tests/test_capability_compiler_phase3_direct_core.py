from pathlib import Path
import sys

from abi.capability_compiler_phase3_direct_core import _examples, _fixed_tokenizer


def test_fixed_lexeme_inventory_and_examples_are_lossless():
    sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "layercake_release").resolve()))
    from layercake.portable_token_plan import EOS_ID, GENERIC_TOKENIZER_FORMAT, LosslessLexemePointerTokenizer

    rows = [
        {
            "ir_record_id": "r1",
            "capability": "grammar",
            "normalized_acquisition_prompt": "Fix quiet bridge.",
            "normalized_output": "Fix the quiet bridge.",
            "authoritative_teacher_tokens": 6,
        }
    ]
    tokenizer = _fixed_tokenizer(rows, LosslessLexemePointerTokenizer, GENERIC_TOKENIZER_FORMAT)
    inventory = [{**rows[0], "ir_record_id": f"r{index}"} for index in range(7000)]
    examples = _examples(inventory, tokenizer, EOS_ID, {"maximum_source_lexemes": 64, "maximum_target_actions": 64})
    assert examples[0]["target_actions"][-1] == EOS_ID
    assert tokenizer.decode_actions(examples[0]["target_actions"], tokenizer.split(rows[0]["normalized_acquisition_prompt"])) == b"Fix the quiet bridge."
