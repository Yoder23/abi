import json
from pathlib import Path
import sys

from abi.capability_compiler_phase3_boundary_bpe_feasibility import BoundaryBpeTokenizer, fit
from abi.capability_compiler_phase3_combined_bpe_feasibility import _evaluate


def _types():
    root = Path(__file__).resolve().parents[2] / "layercake_release"
    sys.path.insert(0, str(root))
    from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
    from layercake_extensions.unicode_direct_neural_core import UnicodeAtomicLexemePointerTokenizer

    return Utf8ConcatenativeBpeTokenizer, UnicodeAtomicLexemePointerTokenizer.split


def test_boundary_bpe_keeps_identifier_copyable_and_exact():
    raw_type, split = _types()
    strings = ["Training тАЬ text with Alpha_77.", "Return Alpha_77 clearly."]
    fitted = fit(strings, 256, split)
    tokenizer = BoundaryBpeTokenizer(raw_type(json.loads(fitted.to_str())), split)
    rows = [{"record_id": "x", "capability": "instruction_following", "prompt": "Repeat Beta_99 exactly.", "output": "Beta_99"}]
    report = _evaluate(tokenizer, rows)
    assert report["roundtrip_failures"] == 0
    assert report["per_capability"]["instruction_following"]["records_with_pointers"] == 1
