from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import load_probe_catalog, probe_label_evidence_sha256
from abi.natural_english_catalog import (
    BUILDERS,
    PROBES_PER_CAPABILITY_SPLIT,
    SPLITS,
    build_catalog,
)


ROOT = Path(__file__).resolve().parents[1]


def test_natural_catalog_is_reproducible_and_has_locked_depth() -> None:
    path = ROOT / "catalogs" / "natural_english_acquisition_v1.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == build_catalog()
    validated = load_probe_catalog(path)
    counts = Counter(
        (probe["split"], probe["capability"])
        for probe in validated["probes"]
    )
    assert len(validated["probes"]) == (
        len(SPLITS) * len(BUILDERS) * PROBES_PER_CAPABILITY_SPLIT
    )
    assert set(counts.values()) == {PROBES_PER_CAPABILITY_SPLIT}


def test_natural_v2_catalog_is_reproducible_and_repairs_only_measured_tasks() -> None:
    path = ROOT / "catalogs" / "natural_english_acquisition_v2.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == build_catalog("v2")
    validated = load_probe_catalog(path)
    assert validated["catalog_id"] == "abi-natural-english-acquisition-v2"
    changed = {"abstention", "domain_independent_reasoning", "format_control"}
    v1 = {
        (probe["split"], probe["capability"], index % 100): probe
        for index, probe in enumerate(build_catalog("v1")["probes"])
    }
    for index, probe in enumerate(validated["probes"]):
        original = v1[(probe["split"], probe["capability"], index % 100)]
        if probe["capability"] not in changed:
            ignored = {"probe_id", "seed", "label_evidence_sha256"}
            assert {
                key: value
                for key, value in probe.items()
                if key not in ignored
            } == {
                key: value
                for key, value in original.items()
                if key not in ignored
            }


def test_natural_v3_changes_only_abstention_and_coherence() -> None:
    path = ROOT / "catalogs" / "natural_english_acquisition_v3.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == build_catalog("v3")
    validated = load_probe_catalog(path)
    assert validated["catalog_id"] == "abi-natural-english-acquisition-v3"
    changed = {"abstention", "coherence"}
    v2 = build_catalog("v2")["probes"]
    for original, probe in zip(v2, validated["probes"], strict=True):
        if probe["capability"] not in changed:
            ignored = {"probe_id", "seed", "label_evidence_sha256"}
            assert {
                key: value
                for key, value in probe.items()
                if key not in ignored
            } == {
                key: value
                for key, value in original.items()
                if key not in ignored
            }


def test_composed_v2_v3_catalog_matches_locked_capability_boundary() -> None:
    path = ROOT / "catalogs" / "natural_english_acquisition_v2_v3.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == build_catalog("v2-v3")
    validated = load_probe_catalog(path)
    assert (
        validated["catalog_id"]
        == "abi-natural-english-acquisition-v2-v3"
    )
    v2 = build_catalog("v2")["probes"]
    v3 = build_catalog("v3")["probes"]
    for v2_probe, v3_probe, composed in zip(
        v2, v3, validated["probes"], strict=True
    ):
        expected = (
            v3_probe
            if composed["capability"] in {"abstention", "coherence"}
            else v2_probe
        )
        assert composed == expected


def test_natural_catalog_has_disjoint_prompt_text_and_valid_labels() -> None:
    catalog = build_catalog()
    prompts = {
        split: {
            probe["prompt"]
            for probe in catalog["probes"]
            if probe["split"] == split
        }
        for split in SPLITS
    }
    assert not (prompts["search"] & prompts["validation"])
    assert not (prompts["search"] & prompts["final_test"])
    assert not (prompts["validation"] & prompts["final_test"])
    for probe in catalog["probes"]:
        assert probe["domain_labels"] == []
        assert probe["domain_claims"] == []
        assert probe["output_introduces_unsupplied_facts"] is False
        assert (
            probe["label_evidence_sha256"]
            == probe_label_evidence_sha256(probe)
        )


def test_collapse_metrics_detect_repetition_without_flagging_normal_text() -> None:
    assert _collapse_metrics([7] * 10, "word " * 10)["collapse_detected"]
    assert _collapse_metrics(
        list(range(24)), "A varied and complete response."
    )["collapse_detected"] is False
