from abi.capability_compiler_phase3_host_interface_audit import (
    portable_plan_parameter_count,
    raw_concatenative_v3_compatible,
)


def test_v3_compatibility_requires_raw_graph():
    base = {
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {"type": "BPE", "dropout": None, "unk_token": "[UNK]", "continuing_subword_prefix": None, "end_of_word_suffix": None, "byte_fallback": False, "ignore_merges": False},
    }
    assert raw_concatenative_v3_compatible(base)
    base["decoder"] = {"type": "Sequence"}
    assert not raw_concatenative_v3_compatible(base)


def test_parameter_accounting_matches_v70():
    assert portable_plan_parameter_count(4999) == 4_174_280
    assert portable_plan_parameter_count(32015) == 14_575_440
