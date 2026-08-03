from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from abi.contrastive_source_artifact import (
    ContrastiveSourceArtifactError,
    _validated_observations,
)
from abi.hf_extraction import load_probe_catalog


ROOT = Path(__file__).resolve().parents[1]


def test_v68_full_evidence_binds_every_frozen_catalog_row():
    catalog = load_probe_catalog(
        ROOT / "catalogs" / "natural_grammar_reference_search_validation_v1.json"
    )
    evidence = json.loads(
        (
            ROOT
            / "results"
            / "abi_moonshot"
            / "reference_artifact_v68"
            / "phi3-natural-grammar-contrastive-full-v1.json"
        ).read_text(encoding="utf-8")
    )
    observations = _validated_observations(evidence=evidence, catalog=catalog)
    assert len(observations) == 224
    assert all(row["passed"] is True for row in observations.values())


def test_contrastive_evidence_tampering_fails_closed():
    catalog = load_probe_catalog(
        ROOT / "catalogs" / "natural_grammar_reference_search_validation_v1.json"
    )
    evidence = json.loads(
        (
            ROOT
            / "results"
            / "abi_moonshot"
            / "reference_artifact_v68"
            / "phi3-natural-grammar-contrastive-full-v1.json"
        ).read_text(encoding="utf-8")
    )
    tampered = copy.deepcopy(evidence)
    tampered["observations"][0]["correct_sentence"] += " changed"
    with pytest.raises(ContrastiveSourceArtifactError):
        _validated_observations(evidence=tampered, catalog=catalog)
