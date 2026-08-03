from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from abi.capability_pipeline import read_extraction_bundle
from abi.hf_extraction import load_probe_catalog
from abi.partial_semantic_source_artifact import (
    PartialSemanticSourceArtifactError,
    _validated_complete_observations,
)


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    source = read_extraction_bundle(
        ROOT
        / "results"
        / "abi_moonshot"
        / "reference_artifact_v60"
        / "phi3-grounded-reference-search-validation-survey-v3.abix"
    )
    catalog = load_probe_catalog(
        ROOT / "catalogs" / "grounded_english_reference_search_validation_v3.json"
    )
    evidence = json.loads(
        (
            ROOT
            / "results"
            / "abi_moonshot"
            / "reference_artifact_v62"
            / "phi3-v60-qwen2-7b-semantic-full-v1.json"
        ).read_text(encoding="utf-8")
    )
    records = {row["record_id"]: row for row in source["records"]}
    results = {row["probe_id"]: row for row in source["probe_results"]}
    probes = {row["probe_id"]: row for row in catalog["probes"]}
    return evidence, records, results, probes


def test_v62_fail_evidence_is_complete_and_exactly_bound():
    evidence, records, results, probes = _inputs()
    observations = _validated_complete_observations(
        evidence=evidence,
        records_by_id=records,
        results_by_probe=results,
        probes_by_id=probes,
    )
    assert len(observations) == 3136
    assert sum(row["passed"] for row in observations.values()) == 3098


def test_partial_semantic_observation_tampering_fails_closed():
    evidence, records, results, probes = _inputs()
    tampered = copy.deepcopy(evidence)
    tampered["observations"][0]["source_response_sha256"] = "0" * 64
    with pytest.raises(PartialSemanticSourceArtifactError):
        _validated_complete_observations(
            evidence=tampered,
            records_by_id=records,
            results_by_probe=results,
            probes_by_id=probes,
        )
