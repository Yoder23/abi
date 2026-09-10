import json
from pathlib import Path

from experiments.factual_semantic_r16.accounting_v2 import account_v2
from experiments.factual_semantic_r16.verify_strict_v3 import verify_v3

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/factual_semantic_r16"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_rendered_prompt_accounting_recomputes() -> None:
    expected = _json(RESULTS / "heldout_v1_accounting_v2_revision_002.json")
    assert account_v2(RESULTS / "heldout_v1") == expected
    assert expected["raw_source_prompt_utf8_bytes"] == 18_334
    assert expected["source_prompt_token_instances"] == 3_206
    assert expected["candidate_token_instances_scored"] == 1_146


def test_strict_v3_reopens_live_and_public_artifacts() -> None:
    source_inventory = _json(
        ROOT
        / "results/preexisting_representation_r15b/heldout_v1_live_v2/receipt.json"
    )["source_snapshot"]
    actual = verify_v3(
        ROOT / "experiments/factual_semantic_r16/configs/heldout_v1.json",
        ROOT / "experiments/factual_semantic_r16/reveals/heldout_v1.json",
        RESULTS / "heldout_v1",
        RESULTS / "heldout_v1_live/live_verification.json",
        RESULTS / "public_v2_requalification_003",
        source_inventory=source_inventory,
    )
    assert actual == _json(RESULTS / "heldout_v1_strict_v3.json")
    assert actual["actual_live_files_rehashed"] == 7
    assert actual["physical_extraction_trees_verified"] == 4
    assert actual["correct_answer_present_once_in_candidate_rows"] == 48
    assert actual["explicit_labeled_answer_fields_in_compiler_bundle"] == 0
