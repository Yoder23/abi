import json
from pathlib import Path
import sys

from abi.capability_compiler_phase3_boundary_bpe_feasibility import BoundaryBpeTokenizer
from abi.capability_compiler_phase3_combined_bpe_feasibility import _evaluate, fit
from abi.capability_compiler_phase3_selective_boundary_bpe import selective_split


def test_selective_boundary_protects_identifiers_but_compresses_ordinary_text():
    root = Path(__file__).resolve().parents[2] / "layercake_release"; sys.path.insert(0, str(root))
    from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
    from layercake_extensions.unicode_direct_neural_core import UnicodeAtomicLexemePointerTokenizer
    fitted = fit(["ordinary English text Alpha_77 ordinary English text", "return Alpha_77"] * 20, 256)
    raw = Utf8ConcatenativeBpeTokenizer(json.loads(fitted.to_str()))
    split = lambda value: selective_split(value, UnicodeAtomicLexemePointerTokenizer.split, raw.split)
    tokenizer = BoundaryBpeTokenizer(raw, split)
    report = _evaluate(tokenizer, [{"record_id": "x", "capability": "instruction_following", "prompt": "Repeat Beta_99 exactly.", "output": "Beta_99"}])
    assert report["roundtrip_failures"] == 0
    assert report["per_capability"]["instruction_following"]["records_with_pointers"] == 1
    assert len(split("ordinary English text")) <= len(UnicodeAtomicLexemePointerTokenizer.split("ordinary English text"))
