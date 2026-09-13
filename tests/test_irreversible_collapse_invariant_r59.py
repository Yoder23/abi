from experiments.irreversible_collapse_invariant_r59.screen_v1 import (
    _irreversible_collapse_reason,
)


def test_invariant_stops_exactly_at_sixth_identical_token():
    assert _irreversible_collapse_reason([7] * 5, "word " * 5, "prompt") is None
    assert (
        _irreversible_collapse_reason([7] * 6, "word " * 6, "prompt")
        == "maximum_identical_token_run"
    )


def test_invariant_ignores_valid_and_prompt_copied_repetition():
    varied = list(range(30))
    assert _irreversible_collapse_reason(
        varied, "A complete varied response with no looping.", "prompt"
    ) is None
    copied = "send the note today now " * 5
    assert _irreversible_collapse_reason(
        varied,
        copied,
        "Please copy: " + "send the note today now " * 2,
    ) is None


def test_invariant_stops_at_locked_novel_lexical_boundary():
    output = "please send the note today " * 5
    assert (
        _irreversible_collapse_reason(list(range(30)), output, "Different prompt")
        == "repeated_novel_lexical_fourgrams"
    )
