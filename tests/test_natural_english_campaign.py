from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from abi.english_generalization_evaluation import (
    _collapse_metrics,
    _normalize_decoding_contract,
)
from abi.english_realization_scale_catalog import (
    CAPABILITIES as REALIZATION_CAPABILITIES,
    PROBES_PER_CAPABILITY as REALIZATION_PROBES_PER_CAPABILITY,
    build_catalog as build_realization_catalog,
)
from abi.hf_extraction import load_probe_catalog, probe_label_evidence_sha256
from abi.natural_english_catalog import (
    BUILDERS,
    COVERAGE_V4_FAMILIES,
    COVERAGE_V5_FAMILIES,
    PROBES_PER_CAPABILITY_SPLIT,
    SPLITS,
    build_catalog,
)


ROOT = Path(__file__).resolve().parents[1]


def test_decoding_contract_adds_only_safe_defaults_for_older_hosts():
    normalized = _normalize_decoding_contract(
        {
            "algorithm": "greedy",
            "no_repeat_ngram_size": 0,
            "prompt_identity_mixture": True,
        }
    )
    assert normalized["prompt_identity_mixture"] is True
    assert normalized["allow_prompt_ngrams"] is False
    assert normalized["lexical_repetition_truncation_threshold"] == 0


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


def test_successor_final_v4_is_preregistered_final_only_and_disjoint() -> None:
    path = ROOT / "catalogs" / "natural_english_successor_final_v4.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == build_catalog("successor-final-v4")
    validated = load_probe_catalog(path)
    assert validated["status"] == "PREREGISTERED_UNOPENED_SUCCESSOR_FINAL_ONLY"
    assert len(validated["probes"]) == len(BUILDERS) * PROBES_PER_CAPABILITY_SPLIT
    assert {probe["split"] for probe in validated["probes"]} == {"final_test"}
    counts = Counter(probe["capability"] for probe in validated["probes"])
    assert set(counts.values()) == {PROBES_PER_CAPABILITY_SPLIT}
    prior_prompts = {
        probe["prompt"]
        for probe in build_catalog("v2-v3")["probes"]
    }
    assert not (
        prior_prompts
        & {probe["prompt"] for probe in validated["probes"]}
    )


def test_coverage_v4_is_search_only_and_changes_no_evaluator() -> None:
    path = ROOT / "catalogs" / "natural_english_coverage_v4.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == build_catalog("coverage-v4")
    validated = load_probe_catalog(path)
    assert len(validated["probes"]) == 225
    assert {probe["split"] for probe in validated["probes"]} == {"search"}
    base = {
        (
            probe["capability"],
            int(str(probe["probe_id"]).rsplit("-", 2)[-2]),
        ): probe
        for probe in build_catalog("v2-v3")["probes"]
        if probe["split"] == "search"
    }
    observed_families = set()
    for probe in validated["probes"]:
        local_index = int(str(probe["probe_id"]).rsplit("-", 3)[-3])
        original = base[(probe["capability"], local_index)]
        assert probe["evaluator"] == original["evaluator"]
        observed_families.add((probe["capability"], local_index % 4))
    expected_families = {
        (capability, family)
        for capability, families in COVERAGE_V4_FAMILIES.items()
        for family in families
    }
    assert observed_families == expected_families


def test_coverage_v5_repairs_only_failed_v4_families() -> None:
    path = ROOT / "catalogs" / "natural_english_coverage_v5.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == build_catalog("coverage-v5")
    validated = load_probe_catalog(path)
    assert len(validated["probes"]) == 100
    assert {probe["split"] for probe in validated["probes"]} == {"search"}
    base = {
        (
            probe["capability"],
            int(str(probe["probe_id"]).rsplit("-", 2)[-2]),
        ): probe
        for probe in build_catalog("v2-v3")["probes"]
        if probe["split"] == "search"
    }
    observed_families = set()
    for probe in validated["probes"]:
        local_index = int(str(probe["probe_id"]).rsplit("-", 3)[-3])
        original = base[(probe["capability"], local_index)]
        assert probe["evaluator"] == original["evaluator"]
        observed_families.add((probe["capability"], local_index % 4))
    expected_families = {
        (capability, family)
        for capability, families in COVERAGE_V5_FAMILIES.items()
        for family in families
    }
    assert observed_families == expected_families


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
        assert probe["label_method"] == "preregistered_catalog"
        assert probe["output_introduces_unsupplied_facts"] is False
        assert (
            probe["label_evidence_sha256"]
            == probe_label_evidence_sha256(probe)
        )


def test_realization_scale_catalog_is_search_only_labeled_and_distinct() -> None:
    catalog = build_realization_catalog()
    probes = catalog["probes"]
    assert len(probes) == (
        len(REALIZATION_CAPABILITIES)
        * REALIZATION_PROBES_PER_CAPABILITY
    )
    counts = Counter(probe["capability"] for probe in probes)
    assert counts == {
        capability: REALIZATION_PROBES_PER_CAPABILITY
        for capability in REALIZATION_CAPABILITIES
    }
    assert len({probe["prompt"] for probe in probes}) == len(probes)
    for probe in probes:
        assert probe["split"] == "search"
        assert probe["destination_scope"] == "english_core"
        assert probe["domain"] == "domain_independent"
        assert probe["domain_labels"] == []
        assert probe["domain_claims"] == []
        assert probe["label_method"] == "preregistered_catalog"
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
    copied = [1, 2, 3, 4, 1, 2, 3, 4, 9]
    copied_metrics = _collapse_metrics(
        copied,
        "A response that repeats one required supplied identifier.",
        [8, 1, 2, 3, 4, 7],
    )
    assert copied_metrics["repeated_fourgram_occurrences_total"] == 1
    assert copied_metrics["repeated_fourgram_occurrences"] == 0
    assert copied_metrics["collapse_detected"] is False
    repeated_phrase = "please send the note today " * 5
    repeated_metrics = _collapse_metrics(
        list(range(30)),
        repeated_phrase,
    )
    assert (
        repeated_metrics["repeated_lexical_fourgram_occurrences"] >= 4
    )
    assert repeated_metrics["collapse_detected"] is True
