from pathlib import Path
import sys

from abi.capability_compiler_phase3_pointer_core import _copy_lexemes, _fixed_tokenizer, _pointer_examples


def _api():
    sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "layercake_release").resolve()))
    from layercake.portable_token_plan import EOS_ID, GENERIC_TOKENIZER_FORMAT, LosslessLexemePointerTokenizer

    return EOS_ID, GENERIC_TOKENIZER_FORMAT, LosslessLexemePointerTokenizer


def test_copy_lexemes_are_unique_source_words_or_numbers_present_in_target():
    _, _, tokenizer_type = _api()
    source = tokenizer_type.split("Ask Mira to meet at 08.15 in the garden, garden.")
    target = tokenizer_type.split("Mira will meet at 08.15 in the garden.")
    assert _copy_lexemes(source, target) == [b"Mira", b"meet", b"at", b"08.15", b"in", b"the"]


def test_pointer_examples_losslessly_reconstruct_and_use_dynamic_actions():
    eos_id, format_version, tokenizer_type = _api()
    row = {
        "ir_record_id": "r1",
        "capability": "prompt_grounding",
        "normalized_acquisition_prompt": "Tell Mira to meet at 08.15.",
        "normalized_output": "Mira will meet at 08.15.",
        "authoritative_teacher_tokens": 8,
    }
    tokenizer = _fixed_tokenizer([row], tokenizer_type, format_version)
    inventory = [{**row, "ir_record_id": f"r{index}"} for index in range(7000)]
    examples = _pointer_examples(inventory, tokenizer, eos_id, {"maximum_source_lexemes": 64, "maximum_target_actions": 64})
    example = examples[0]
    assert example["target_actions"][-1] == eos_id
    assert example["pointer_actions"] == 4
    assert any(action >= tokenizer.vocab_size for action in example["target_actions"])
    assert tokenizer.decode_actions(example["target_actions"], tokenizer.split(row["normalized_acquisition_prompt"])) == row["normalized_output"].encode("utf-8")
