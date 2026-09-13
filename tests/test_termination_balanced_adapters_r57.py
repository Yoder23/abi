import json
from pathlib import Path

import pytest

from experiments.broad_payload_reconstruction_r51.screen_v1a import ScreenError
from experiments.termination_balanced_adapters_r57.screen_v1 import _preflight


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "results" / "termination_balanced_adapters_r57" / "candidate_v1"


def test_r57_endpoint_satisfies_bound_training_preflight():
    _preflight(CANDIDATE)


def test_r57_preflight_rejects_disabled_terminal_objective(tmp_path):
    metadata = json.loads((CANDIDATE / "metadata.json").read_text(encoding="utf-8"))
    metadata["training"]["balanced_terminal_loss"] = False
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ScreenError, match="terminal-balanced"):
        _preflight(candidate)
