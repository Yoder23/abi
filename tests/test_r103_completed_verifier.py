"""Additive post-run hostile tests; frozen R103 experiment files are unchanged."""

import copy
import shutil

import pytest

from experiments.recipient_interface_r103 import core as c
from experiments.recipient_interface_r103 import run


@pytest.fixture
def copied_evidence(tmp_path, monkeypatch):
    original = c.OUT
    for relative in ("freeze.json", "qualification.json", "comparison.json", "decision.json",
                     "qualification/control/raw.jsonl", "qualification/control/receipt.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / relative, destination)
    monkeypatch.setattr(c, "OUT", tmp_path)
    # Full c.binding() remains active: it checks original frozen dependencies.
    return tmp_path


def test_completed_qualification_failure_recomputes(copied_evidence):
    assert run.verify()["verdict"] == "CONTROL_QUALIFICATION_FAILED"


@pytest.mark.parametrize("mutation", ["missing_raw", "missing_receipt", "raw_hash", "missing_row", "text", "trusted_screen", "stale_decision", "forbidden_comparison", "interrupted", "resource_nan"])
def test_actual_completed_verifier_rejects_mutations(copied_evidence, mutation):
    root = copied_evidence
    raw = root / "qualification/control/raw.jsonl"
    receipt_path = raw.with_name("receipt.json")
    receipt = c.read(receipt_path)
    if mutation == "missing_raw":
        raw.rename(raw.with_suffix(".missing"))
    elif mutation == "missing_receipt":
        receipt_path.rename(receipt_path.with_suffix(".missing"))
    elif mutation in ("missing_row", "text", "trusted_screen"):
        observed = c.rows(raw)
        if mutation == "missing_row":
            observed.pop()
        elif mutation == "text":
            observed[0]["output"] += " Changed."
        else:
            failed = next(r for r in observed if not r["screen"]["mechanical_pass"])
            failed["screen"]["mechanical_pass"] = True
        raw.write_bytes(b"".join(c.base.canonical(r) for r in observed))
        receipt["raw_sha256"] = c.sha(raw)
        receipt["calls"] = len(observed)
    elif mutation == "raw_hash":
        receipt["raw_sha256"] = "0" * 64
    elif mutation == "stale_decision":
        path = root / "decision.json"
        value = c.read(path)
        value["verdict"] = "FORCED_SUCCESS"
        path.write_bytes(c.base.canonical(value))
    elif mutation == "forbidden_comparison":
        (root / "comparison").mkdir()
    elif mutation == "interrupted":
        c.write_once(root / "stop.json", {"reason": "incomplete"})
    elif mutation == "resource_nan":
        receipt["wall_seconds"] = float("nan")
    if mutation != "missing_receipt":
        receipt_path.write_bytes(c.base.canonical(receipt))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        run.verify()


def test_frozen_contradiction_proxy_has_a_temporal_limitation():
    # Descriptive regression for an observed limitation, NOT a repaired score.
    population = {r["id"]: r for r in c.catalog("qualification")}
    observed = next(r for r in c.rows(c.OUT / "qualification/control/raw.jsonl")
                    if r["id"] == "r103-qualification-020-tone_control")
    assert "as soon as the details are confirmed" in observed["output"]
    result = c.screen(population[observed["id"]], observed["output"])
    assert result == copy.deepcopy(observed["screen"])
    assert result["explicit_approval_contradiction"]
