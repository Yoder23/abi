import hashlib
import json
from pathlib import Path

import pytest

from abi.final_mile import FinalMileError, canonical_json_bytes
from abi.final_mile_host_portability import run_structural_screen

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/ABI_FINAL_MILE_HOST_PORTABILITY_V1.json"


def test_exact_product_fails_cross_architecture_native_installation(tmp_path):
    result = run_structural_screen(ROOT, CONTRACT, output_dir=tmp_path / "screen")
    assert result["status"] == "HOST_INDEPENDENCE_FAILED"
    assert result["receivers_passing"] == 1
    assert result["receivers_required"] == 3
    assert result["one_bounded_repair_consumed"] is True
    assert result["model_inference_performed"] is False


def test_portability_evidence_is_immutable(tmp_path):
    output = tmp_path / "screen"
    run_structural_screen(ROOT, CONTRACT, output_dir=output)
    with pytest.raises(FinalMileError, match="already exists"):
        run_structural_screen(ROOT, CONTRACT, output_dir=output)


def test_rescreen_evidence_hash_replays(tmp_path):
    output = tmp_path / "screen"
    run_structural_screen(ROOT, CONTRACT, output_dir=output)
    value = json.loads((output / "repair_rescreen.json").read_text(encoding="utf-8"))
    claimed = value.pop("evidence_sha256")
    assert hashlib.sha256(canonical_json_bytes(value)).hexdigest() == claimed
