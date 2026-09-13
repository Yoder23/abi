import json
from pathlib import Path

import pytest

from experiments.broad_payload_reconstruction_r51.screen_v1a import ScreenError
from experiments.sparse_termination_controller_r58.screen_v1 import _preflight


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "results" / "deep_sparse_adapters_r55" / "candidate_v1"
CONTROLLER = ROOT / "results" / "sparse_termination_controller_r58" / "controller_v1a"


def test_r58_endpoint_satisfies_bound_preflight():
    _preflight(HOST, CONTROLLER)


def test_r58_preflight_rejects_token_modification_claim(tmp_path):
    metadata = json.loads((CONTROLLER / "metadata.json").read_text(encoding="utf-8"))
    metadata["controller"]["can_modify_token_logits"] = True
    candidate = tmp_path / "controller"
    candidate.mkdir()
    (candidate / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (candidate / "termination_controller.safetensors").write_bytes(
        (CONTROLLER / "termination_controller.safetensors").read_bytes()
    )
    with pytest.raises(ScreenError):
        _preflight(HOST, candidate)
