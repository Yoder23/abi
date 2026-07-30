import json

import pytest

from abi.layercake_domains import DomainConformanceError, _canonical_sha
from abi.layercake_host_reproduction import aggregate_host_reproduction


def _validation(candidate, observation):
    metrics = {
        f"capability-{index}": {
            "observations": 100,
            "layercake_passes": 100,
            "source_passing_regressions": 0,
            "automatic_route_accuracy": 1.0,
            "bounded_zero_regression_pass": True,
        }
        for index in range(14)
    }
    value = {
        "schema_version": "abi-layercake-native-host-semantic-validation/1",
        "status": "PASS",
        "split": "validation",
        "observation_count": 1400,
        "capability_metrics": metrics,
        "observations": [observation],
        "bounded_zero_regression_pass": True,
        "complete_locked_depth": True,
        "host_manifest_sha256": candidate["host_manifest_sha256"],
        "runtime_graph_sha256": candidate["runtime_graph_sha256"],
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "final_test_accessed": False,
        "peak_process_rss_bytes": 123,
    }
    value["evidence_sha256"] = _canonical_sha(value)
    return value


def _fixture(tmp_path):
    candidate = {
        "host_manifest_sha256": "a" * 64,
        "runtime_graph_sha256": "b" * 64,
        "runtime_runner_sha256": "c" * 64,
    }
    initializations = []
    for index in range(3):
        path = tmp_path / f"init-{index}.json"
        value = _validation(candidate, {"output": "same"})
        path.write_text(json.dumps(value), encoding="utf-8")
        initializations.append(
            {"id": f"host-init-{index + 1}", "output": path.name}
        )
    protocol = {
        "format": (
            "abi-layercake-native-host-three-initialization-"
            "reproduction-protocol/1"
        ),
        "status": (
            "PREREGISTERED_AFTER_SINGLE_HOST_PROMOTION_GATES_"
            "BEFORE_REPRODUCTION_RUNS"
        ),
        "candidate": candidate,
        "reproduction_unit": (
            "fresh_process_and_fresh_onnxruntime_session_initialization"
        ),
        "initializations": initializations,
        "per_initialization_gate": {
            "functional_observations": 1400,
            "capabilities": 14,
        },
        "claim_boundary": "Three deployments, not three trained seeds.",
        "final_test_accessed": False,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return protocol_path


def test_aggregate_host_reproduction_passes_three_identical_initializations(
    tmp_path,
):
    protocol = _fixture(tmp_path)
    result = aggregate_host_reproduction(
        protocol_path=protocol,
        output_path=tmp_path / "evidence.json",
    )
    assert result["status"] == "PASS"
    assert result["initialization_count"] == 3
    assert result["byte_identical_semantic_observation_payloads"] is True


def test_aggregate_host_reproduction_rejects_divergent_observations(tmp_path):
    protocol = _fixture(tmp_path)
    path = tmp_path / "init-2.json"
    value = _validation(
        {
            "host_manifest_sha256": "a" * 64,
            "runtime_graph_sha256": "b" * 64,
        },
        {"output": "different"},
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        DomainConformanceError, match="different validation observations"
    ):
        aggregate_host_reproduction(
            protocol_path=protocol,
            output_path=tmp_path / "evidence.json",
        )
