from pathlib import Path
import sys

from abi.capability_compiler_phase3_unicode_span_copy_feasibility import (
    _architecture_grid,
    _build_tokenizer,
    _encode,
)


def _types():
    root = Path(__file__).resolve().parents[2] / "layercake_release"
    sys.path.insert(0, str(root))
    from layercake.portable_token_plan import PortableTokenPlan
    from layercake_extensions.unicode_direct_neural_core import UnicodeAtomicLexemePointerTokenizer

    return PortableTokenPlan, UnicodeAtomicLexemePointerTokenizer


def test_unicode_span_copy_is_exact_for_arbitrary_valid_utf8_and_has_matched_grid():
    model_type, tokenizer_type = _types()
    rows = [
        {
            "prompt": "Repeat identifier Alpha_77 after mojibake тАЬ and emoji 🧪.",
            "output": "Identifier: Alpha_77.",
        }
    ]
    tokenizer = _build_tokenizer(rows, tokenizer_type)
    encoded = _encode(tokenizer, rows[0]["prompt"], rows[0]["output"])
    assert encoded["roundtrip"] is True
    assert encoded["pointer_actions"] == 1
    grid = _architecture_grid(model_type, tokenizer.vocab_size, 64)
    assert len(grid) == 12
    assert all(row["parameters"] > 0 for row in grid)
