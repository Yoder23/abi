import json
from pathlib import Path
import sys

from abi.capability_compiler_phase3_bpe_pointer_resilience import _pointer_targets


def test_v5_tokenizer_supports_exact_pointer_targets():
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str((root / "../layercake_release").resolve()))
    from layercake_extensions.selective_boundary_bpe_direct_neural_core import SelectiveBoundaryBpeTokenizer
    tokenizer = SelectiveBoundaryBpeTokenizer(json.loads((root / "results/abi_capability_compiler_phase3/selective_boundary_bpe_v168/tokenizer.json").read_text(encoding="utf-8")))
    source = tokenizer.split("Repeat UNIQUE_90210 exactly."); output = tokenizer.split("UNIQUE_90210")
    actions = _pointer_targets(source, output, tokenizer.vocab_size, tokenizer)
    assert tokenizer.decode_actions(actions, source) == b"UNIQUE_90210"
    assert any(action >= tokenizer.vocab_size for action in actions)
