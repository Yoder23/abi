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
