from __future__ import annotations

import pytest

from abi.capability_pipeline import build_source_model_manifest
from abi.semantic_source_artifact import (
    SemanticSourceArtifactError,
    _validated_observations,
)
from abi.semantic_source_qualification import _canonical_sha


def _observation():
    import hashlib

    row = {
        "probe_id": "probe-1",
        "record_id": "record-1",
        "capability": "rewriting",
        "split": "search",
        "raw_prompt_sha256": hashlib.sha256(b"raw prompt").hexdigest(),
        "source_response_sha256": "b" * 64,
        "authoritative_judge_token_ids": [1, 2],
        "judge_tokens": 2,
        "judge_token_counter": "authoritative_generated_token_ids",
        "judge_finish_reason": "eos_token",
        "parsed": True,
        "passed": True,
    }
    row["observation_sha256"] = _canonical_sha(row)
    return row


def test_semantic_observation_tampering_fails_before_artifact_construction():
    observation = _observation()
    judge_manifest = build_source_model_manifest(
        model_id="judge/model",
        revision="a" * 40,
        revision_is_immutable=True,
        architecture="Judge",
        parameter_count=10,
        tokenizer_id="judge/model",
        tokenizer_revision="a" * 40,
        license_id="test",
        weight_files=[
            {"relative_path": "model.bin", "sha256": "c" * 64, "bytes": 10}
        ],
    )
    evidence = {
        "format": "abi-independent-semantic-source-qualification/1",
        "status": "PASS",
        "mode": "full",
        "judge": {
            "source_manifest": judge_manifest,
            "generated_tokens": 2,
            "runtime": {
                "device": "cuda",
                "weight_execution_precision": "bitsandbytes_int8",
                "cpu_offload_enabled": False,
            },
        },
        "observations": [observation],
        "observation_count": 1,
        "parse_count": 1,
        "eos_count": 1,
        "semantic_passes": 1,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    kwargs = {
        "records_by_id": {
            "record-1": {"record_id": "record-1", "output_sha256": "b" * 64}
        },
        "results_by_probe": {"probe-1": {"record_id": "record-1"}},
        "probes_by_id": {
            "probe-1": {
                "probe_id": "probe-1",
                "prompt": "raw prompt",
                "capability": "rewriting",
                "split": "search",
            }
        },
    }
    assert set(_validated_observations(evidence=evidence, **kwargs)) == {
        "probe-1"
    }

    observation["passed"] = False
    with pytest.raises(SemanticSourceArtifactError, match="observation hash"):
        tampered_evidence = dict(evidence)
        tampered_evidence["observations"] = [observation]
        tampered_evidence["evidence_sha256"] = _canonical_sha(
            {
                key: value
                for key, value in tampered_evidence.items()
                if key != "evidence_sha256"
            }
        )
        _validated_observations(
            evidence=tampered_evidence,
            **kwargs,
        )
