import hashlib

from abi.capability_compiler_phase2_common import canonical_json_bytes
from abi.capability_compiler_phase3_sparse_router_replication import (
    _replication_decision,
)


def test_replication_decision_binds_actual_protocol_path(tmp_path) -> None:
    protocol_path = tmp_path / "replication.json"
    protocol = {
        "training": {"seed": 7},
        "router_gate": {
            "aggregate_point_minimum": 0.9,
            "aggregate_wilson_lower_minimum": 0.88,
            "body_point_minimum": 0.9,
            "per_capability_point_minimum": 0.85,
            "per_capability_wilson_lower_minimum": 0.75,
            "metadata_point_minimum": 0.9,
            "metadata_wilson_lower_minimum": 0.88,
        },
    }
    capabilities = (
        "abstention",
        "clarification",
        "coherence",
        "conversation",
        "email_drafting_from_notes",
        "fact_free_reasoning",
        "fluent_realization",
        "format_control",
        "grammar",
        "instruction_following",
        "prompt_grounding",
        "rewriting",
        "supplied_text_summarization",
        "tone_control",
    )
    rows = []
    for capability in capabilities:
        for index in range(100):
            for variant in ("original", "body"):
                rows.append(
                    {
                        "variant": variant,
                        "capability": capability,
                        "predicted": capability,
                        "correct": True,
                    }
                )
            rows.append(
                {
                    "variant": "metadata",
                    "capability": capability,
                    "predicted": "__metadata__",
                    "correct": True,
                }
            )
    result = _replication_decision(
        protocol_path,
        protocol,
        "protocol-hash",
        {"checkpoint": {"sha256": "checkpoint"}},
        rows,
        "rows-hash",
    )
    assert result["protocol"]["path"] == "replication.json"
    assert result["status"] == "PASS_SPARSE_ROUTER_REPLICATION"
    evidence = result["evidence_sha256"]
    without = dict(result)
    without.pop("evidence_sha256")
    assert evidence == hashlib.sha256(canonical_json_bytes(without)).hexdigest()
