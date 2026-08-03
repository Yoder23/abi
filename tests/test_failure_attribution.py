import json
from copy import deepcopy
from pathlib import Path

from abi.failure_attribution import classify_evidence, verify_contract
from abi.layercake_external_host_control import verify_external_host_evidence


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "ABI_LAYERCAKE_FAILURE_ATTRIBUTION_CONTRACT_V1.json"


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _passing_evidence():
    contract = _contract()
    return {
        "format": "abi-layercake-failure-attribution-evidence/1",
        "controls": {
            "sealed_layercake_native": {
                "executed": True,
                "exact_control_lineage": True,
                "result": "PASS",
            },
            "capability_naive_receiver": {
                "executed": True,
                "english_quality_result": "FAIL",
            },
            "bridge_only": {
                "executed": True,
                "english_quality_result": "FAIL",
            },
            "shuffled_abi_artifact": {
                "executed": True,
                "english_quality_result": "FAIL",
            },
            "native_payload_same_path": {"executed": True, "result": "PASS"},
        },
        "abi_extraction": {
            "executed": True,
            "artifact_sha256_before": "a" * 64,
            "artifact_sha256_after": "a" * 64,
            "gates": {
                gate: "PASS" for gate in contract["required_abi_extraction_gates"]
            },
        },
        "integrated_candidate": {
            "executed": True,
            "exact_layercake_execution_contract": True,
            "canonical_abi_unchanged": True,
            "abi_artifact_unchanged": True,
            "teacher_present_at_inference": False,
            "gates": {
                gate: "PASS" for gate in contract["required_integrated_gates"]
            },
        },
    }


def test_contract_binds_separate_sealed_layercake_repository():
    result = verify_contract(CONTRACT_PATH, layercake_root=ROOT.parent / "layercake_release")
    assert result["status"] == "PASS"
    assert result["external_layercake_control"]["verified"] is True
    assert (
        result["external_layercake_control"]["primary_checkpoint_sha256"]
        == "9e0e6b9add32b4c460f7b570a32584f380e59bf6d631e313ff813069d24e09e1"
    )


def test_exact_native_control_failure_is_layercake_regression():
    evidence = _passing_evidence()
    evidence["controls"]["sealed_layercake_native"]["result"] = "FAIL"
    result = classify_evidence(evidence, _contract())
    assert result["classification"] == "LAYERCAKE_HOST_REGRESSION"
    assert result["owner"] == "LAYERCAKE"


def test_abi_failure_does_not_invalidate_passing_layercake():
    evidence = _passing_evidence()
    evidence["abi_extraction"]["gates"]["english_domain_segregation"] = "FAIL"
    result = classify_evidence(evidence, _contract())
    assert result["classification"] == "ABI_EXTRACTION_FAILURE"
    assert result["owner"] == "ABI"


def test_same_path_failure_is_integration_not_layercake():
    evidence = _passing_evidence()
    evidence["controls"]["native_payload_same_path"]["result"] = "FAIL"
    result = classify_evidence(evidence, _contract())
    assert result["classification"] == "ABI_LAYERCAKE_INTEGRATION_FAILURE"
    assert result["owner"] == "INTEGRATION"


def test_unexpected_bridge_only_english_is_causality_failure():
    evidence = _passing_evidence()
    evidence["controls"]["bridge_only"]["english_quality_result"] = "PASS"
    result = classify_evidence(evidence, _contract())
    assert result["classification"] == "TRANSFER_CAUSALITY_FAILURE"
    assert result["promotion_eligible"] is False


def test_integration_speed_failure_is_not_abi_extraction_or_layercake():
    evidence = _passing_evidence()
    evidence["integrated_candidate"]["gates"]["cpu_throughput"] = "FAIL"
    result = classify_evidence(evidence, _contract())
    assert result["classification"] == "ABI_LAYERCAKE_INTEGRATION_FAILURE"
    assert result["owner"] == "INTEGRATION"


def test_nonexact_layercake_control_remains_unassigned():
    evidence = _passing_evidence()
    evidence["controls"]["sealed_layercake_native"]["exact_control_lineage"] = False
    result = classify_evidence(evidence, _contract())
    assert result["classification"] == "INCOMPLETE_ATTRIBUTION_EVIDENCE"
    assert result["owner"] == "UNASSIGNED"


def test_only_complete_causal_same_artifact_candidate_can_pass():
    evidence = deepcopy(_passing_evidence())
    result = classify_evidence(evidence, _contract())
    assert result == {
        "classification": "PASS",
        "owner": "END_TO_END",
        "promotion_eligible": True,
        "reasons": [],
    }


def test_external_production_host_control_is_bound_but_not_transfer_evidence():
    evidence = (
        ROOT
        / "results/abi_moonshot/external_layercake_controls/phase2-r3-phase8-native-v1.json"
    )
    result = verify_external_host_evidence(
        evidence_path=evidence,
        contract_path=CONTRACT_PATH,
    )
    assert result["status"] == "PASS"
    assert (
        result["checkpoint_sha256"]
        == "9e0e6b9add32b4c460f7b570a32584f380e59bf6d631e313ff813069d24e09e1"
    )
