from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from abi.capability_pipeline import read_extraction_bundle
from abi.conditional_choice_source_artifact import (
    COUNTER,
    ConditionalChoiceArtifactError,
    _validated_rows,
    build_conditional_choice_artifact,
)
from abi.hf_extraction import load_probe_catalog
from abi.layercake_host_v3 import (
    LayerCakeHostError,
    _materialize_training_prompt,
    load_english_training_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalogs/robust_choice_reasoning_search_r72_v1.json"
RAW = ROOT / "results/robust_choice_substrate_r72/source_scores_v1/choice_scores.jsonl"
RESULT = ROOT / "results/robust_choice_substrate_r72/source_scores_v1/result.json"
SOURCE = ROOT / "results/abi_moonshot/segregated_acquisition_v3/phi3-broad-natural-conversation-complete-search-training-v3.abix"
ONTOLOGY = ROOT / "evidence/current/segregation/domain_ontology_v1.json"


def _raw_rows() -> list[dict]:
    return [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines()]


def test_r72_raw_scores_recompute_exact_pass_set() -> None:
    passing, families = _validated_rows(
        catalog=load_probe_catalog(CATALOG), raw_rows=_raw_rows()
    )
    assert len(passing) == 2_098
    assert families == {"0": 300, "1": 300, "2": 300, "3": 298, "4": 300, "5": 300, "6": 300}


def test_r73_artifact_is_segregated_and_consumer_bound(tmp_path: Path) -> None:
    output = tmp_path / "reasoning-r73.abix"
    receipt = build_conditional_choice_artifact(
        source_bundle_path=SOURCE,
        catalog_path=CATALOG,
        result_path=RESULT,
        raw_scores_path=RAW,
        domain_ontology_path=ONTOLOGY,
        output_path=output,
        token_counter=lambda _text: 1,
    )
    assert receipt["verified"] is True
    assert receipt["records"] == 2_098
    bundle = read_extraction_bundle(output)
    assert bundle["verification"]["domain_segregation_verified"] is True
    assert {row["teacher_token_counter"] for row in bundle["records"]} == {COUNTER}
    training_rows, _, _ = load_english_training_rows(output, budget_index=-1)
    assert len(training_rows) == 2_098
    assert {row["capability"] for row in training_rows} == {
        "domain_independent_reasoning"
    }
    record = bundle["records"][0]
    result = next(row for row in bundle["probe_results"] if row["record_id"] == record["record_id"])
    assert _materialize_training_prompt(record=record, probe_result=result, ledger=bundle["ledger"]) == record["prompt"]


def test_conditional_consumer_rejects_stale_margin(tmp_path: Path) -> None:
    output = tmp_path / "reasoning-r73.abix"
    build_conditional_choice_artifact(
        source_bundle_path=SOURCE,
        catalog_path=CATALOG,
        result_path=RESULT,
        raw_scores_path=RAW,
        domain_ontology_path=ONTOLOGY,
        output_path=output,
        token_counter=lambda _text: 1,
    )
    bundle = read_extraction_bundle(output)
    record = bundle["records"][0]
    result = next(row for row in bundle["probe_results"] if row["record_id"] == record["record_id"])
    stale = copy.deepcopy(result)
    stale["evaluator"]["top_margin_mean_log_probability"] = 0.0
    with pytest.raises(LayerCakeHostError):
        _materialize_training_prompt(record=record, probe_result=stale, ledger=bundle["ledger"])


def test_raw_score_tamper_fails_closed() -> None:
    rows = _raw_rows()
    rows[0]["selected_output"] = "A10000"
    with pytest.raises(ConditionalChoiceArtifactError):
        _validated_rows(catalog=load_probe_catalog(CATALOG), raw_rows=rows)
