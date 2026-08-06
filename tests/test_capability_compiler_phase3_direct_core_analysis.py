import json
from pathlib import Path
import shutil

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_direct_core_analysis import build_decision, verify_decision, wilson


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_CORE_PROTOCOL_V23.json"
CANDIDATE = ROOT / "results/abi_capability_compiler_phase3_direct_core/development_v23/A0-seed240017"
EVALUATION = ROOT / "results/abi_capability_compiler_phase3_direct_core/evaluation_v23/A0-seed240017"
DECISION = ROOT / "results/abi_capability_compiler_phase3_direct_core/direct_core_decision_v23_corrected_v1.json"


def _build(candidate=CANDIDATE, evaluation=EVALUATION):
    return build_decision(root=ROOT, protocol_path=PROTOCOL, candidate_dir=candidate, evaluation_dir=evaluation)


def test_wilson_and_real_v23_raw_evidence_recompute():
    assert wilson(50, 100)["lower_95"] < 0.5 < wilson(50, 100)["upper_95"]
    result = _build()
    assert result["status"] == "FAIL_ABSOLUTE_QUALITY_SCREEN_ARCHITECTURE_CLOSED"
    assert result["candidate"]["functional_passes"] == 504
    assert result["candidate"]["repetition_collapses"] == 77
    assert result["gates"]["matched_causal_controls"] == "NOT_REACHED_PREREGISTERED_EARLY_STOP"
    assert result["ownership"]["abi_acquisition_or_representation_failure"] is True
    assert result["ownership"]["layercake_host_regression"] is False


def test_adversarial_receipt_aggregate_mutation_is_rejected(tmp_path):
    evaluation = tmp_path / "evaluation"
    shutil.copytree(EVALUATION, evaluation)
    path = evaluation / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["functional_passes"] += 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase3Error, match="receipt differs"):
        _build(evaluation=evaluation)


def test_adversarial_raw_output_mutation_is_rejected(tmp_path):
    evaluation = tmp_path / "evaluation"
    shutil.copytree(EVALUATION, evaluation)
    path = evaluation / "development_outputs.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    value = json.loads(lines[0])
    value["output"] += " tampered"
    lines[0] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(Phase3Error, match="output binding failed"):
        _build(evaluation=evaluation)


def test_adversarial_candidate_metadata_mutation_is_rejected(tmp_path):
    candidate = tmp_path / "candidate"
    shutil.copytree(CANDIDATE, candidate)
    path = candidate / "metadata.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["teacher_present_at_inference"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance or ownership"):
        _build(candidate=candidate)


def test_adversarial_decision_mutation_is_rejected(tmp_path):
    path = tmp_path / "decision.json"
    value = json.loads(DECISION.read_text(encoding="utf-8"))
    value["phase3_certified"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase3Error, match="differs from raw-evidence"):
        verify_decision(root=ROOT, protocol_path=PROTOCOL, candidate_dir=CANDIDATE, evaluation_dir=EVALUATION, decision_path=path)
