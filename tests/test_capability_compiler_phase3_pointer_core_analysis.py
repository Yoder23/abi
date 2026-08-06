import json
from pathlib import Path
import shutil

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_pointer_core_analysis import build_decision, verify_decision


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_PROTOCOL_V24.json"
CANDIDATE = ROOT / "results/abi_capability_compiler_phase3_pointer_core/development_v24/P0-seed240017"
EVALUATION = ROOT / "results/abi_capability_compiler_phase3_pointer_core/evaluation_v24/P0-seed240017"
DECISION = ROOT / "results/abi_capability_compiler_phase3_pointer_core/pointer_core_decision_v24.json"


def _build(candidate=CANDIDATE, evaluation=EVALUATION):
    return build_decision(root=ROOT, protocol_path=PROTOCOL, candidate_dir=candidate, evaluation_dir=evaluation)


def test_real_v24_recomputes_and_separates_ownership():
    result = _build()
    assert result["candidate"]["functional_passes"] == 601
    assert result["candidate"]["repetition_collapses"] == 139
    assert result["candidate"]["generation_errors"] == 31
    assert result["candidate"]["best_case_if_every_error_became_a_pass"]["passes"] == 632
    assert result["matched_v23_diagnostic"]["pass_delta"] == 97
    assert result["matched_v23_diagnostic"]["collapse_delta"] == 62
    assert result["ownership"]["abi_acquisition_or_representation_failure"] is True
    assert result["ownership"]["layercake_host_regression"] is False
    assert result["ownership"]["layercake_utf8_validity_gap_exposed"] is True


def test_adversarial_receipt_mutation_is_rejected(tmp_path):
    evaluation = tmp_path / "evaluation"
    shutil.copytree(EVALUATION, evaluation)
    path = evaluation / "receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["generation_errors"] -= 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Phase3Error, match="receipt differs"):
        _build(evaluation=evaluation)


def test_adversarial_output_mutation_is_rejected(tmp_path):
    evaluation = tmp_path / "evaluation"
    shutil.copytree(EVALUATION, evaluation)
    path = evaluation / "development_outputs.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    value = json.loads(lines[0])
    value["functional_pass"] = not value["functional_pass"]
    lines[0] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(Phase3Error, match="output binding failed"):
        _build(evaluation=evaluation)


def test_adversarial_candidate_mutation_is_rejected(tmp_path):
    candidate = tmp_path / "candidate"
    shutil.copytree(CANDIDATE, candidate)
    path = candidate / "metadata.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source_blocks_retained"] = 1
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
