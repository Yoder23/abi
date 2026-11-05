import ast

import pytest

from experiments.grounded_english_r99_fp32 import core as c
from experiments.grounded_english_r99_fp32.run import require_numerical_gate


def test_all_data_and_quality_rules_are_inherited_unchanged():
    for key in ("SEED", "STEPS", "BATCH", "CONTEXT", "MAX_NEW", "PARENT_SHA", "HOST_COMMIT"):
        assert getattr(c, key) == getattr(c.original, key)
    for key in ("screen", "source_gate", "normalize", "model_prompt", "target"):
        assert getattr(c, key) is getattr(c.original, key)
    assert c.OUT != c.original.OUT


def test_actual_numerical_failure_is_recomputable():
    require_numerical_gate()


def test_fp32_loop_contains_no_autocast_or_scaler():
    tree = ast.parse((c.HERE / "train.py").read_text(encoding="utf-8"))
    calls = [
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "autocast" not in calls and "GradScaler" not in calls
    assert "backward" in calls


def test_missing_frozen_input_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "OUT", tmp_path / "out")
    c.write_once(c.OUT / "freeze.json", {"files": {"absent": "0" * 64}})
    with pytest.raises(RuntimeError, match="changed or missing"):
        c.verify_binding()
