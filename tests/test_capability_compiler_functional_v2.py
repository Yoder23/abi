from abi.capability_compiler_functional_v2 import evaluate_functional_v2


def test_number_words_are_surface_equivalent():
    assert evaluate_functional_v2("Two parcels arrived.", {"kind": "contains_all", "values": ["2", "parcels"]}, "fluent_realization")


def test_requested_abstention_phrase_is_accepted():
    assert evaluate_functional_v2("The answer cannot be known from the information given.", {"kind": "contains_any", "values": ["cannot know"]}, "abstention")
