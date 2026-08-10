from abi.capability_compiler_repetition_v2 import repetition_collapse_v2, repetition_tokens


def test_tokenizer_keeps_punctuation_loops_visible():
    assert repetition_tokens("8: 8:") == ["8", ":", "8", ":"]
    assert repetition_collapse_v2("8: " * 5)
    assert repetition_collapse_v2("." * 40)


def test_common_words_in_coherent_prose_are_not_collapse():
    value = (
        "I hope this message finds you well. I wanted to thank you for the parcel and ask "
        "whether you could send the invoice by Thursday. Thank you again for your help."
    )
    assert not repetition_collapse_v2(value)


def test_repeated_multitoken_loop_is_collapse():
    assert repetition_collapse_v2("the answer is blue. " * 4)
