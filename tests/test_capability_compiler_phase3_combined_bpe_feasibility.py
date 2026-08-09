import json
from pathlib import Path
import sys

from abi.capability_compiler_phase3_combined_bpe_feasibility import _evaluate, fit


def _tokenizer_type():
    root = Path(__file__).resolve().parents[2] / "layercake_release"
    sys.path.insert(0, str(root))
    from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer

    return Utf8ConcatenativeBpeTokenizer


def test_combined_training_alphabet_preserves_unicode_and_unseen_ascii_targets():
    tokenizer_type = _tokenizer_type()
    training = ["Prompt тАЬ Alpha_77", "Return Alpha_77 clearly."]
    fitted = fit(training, 256)
    tokenizer = tokenizer_type(json.loads(fitted.to_str()))
    rows = [
        {
            "record_id": "heldout-1",
            "capability": "instruction_following",
            "prompt": "Repeat Beta_99 twice.",
            "output": "Beta_99 Beta_99",
        }
    ]
    report = _evaluate(tokenizer, rows)
    assert report["roundtrip_failures"] == 0
    assert report["maximum_target_actions"] <= 320
