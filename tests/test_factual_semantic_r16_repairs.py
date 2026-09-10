import json
from pathlib import Path

from experiments.factual_semantic_r16.accounting_v2 import account_v2
from experiments.factual_semantic_r16.accounting_v3 import account_v3
from experiments.factual_semantic_r16.verify_strict_v3 import verify_v3
from experiments.factual_semantic_r16.verify_strict_v4 import verify_v4

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


def test_clean_replication_receipts_bind_exact_identity() -> None:
    source_inventory = _json(
        ROOT
        / "results/preexisting_representation_r15b/heldout_v1_live_v2/receipt.json"
    )["source_snapshot"]
    config = ROOT / "experiments/factual_semantic_r16/configs/heldout_v2.json"
    reveal = ROOT / "experiments/factual_semantic_r16/reveals/heldout_v2.json"
    run_dir = RESULTS / "heldout_v2"
    live = RESULTS / "heldout_v2_live/live_verification.json"
    public = RESULTS / "public_v2_requalification_003"
    strict = verify_v4(
        config,
        reveal,
        run_dir,
        live,
        public,
        source_inventory=source_inventory,
    )
    assert strict == _json(RESULTS / "heldout_v2_strict_v4.json")
    assert strict["verified_identity"]["config_sha256"] == (
        "97652ad8ef66eb23ce3172a9da80c11ac68d993f0476af13df94ceb4539a80d5"
    )
    accounting = account_v3(run_dir)
    assert accounting == _json(RESULTS / "heldout_v2_accounting_v3_revision_002.json")
    assert accounting["verified_identity"]["run_receipt_sha256"] == (
        "4db7f748a0a9caa5b2b6fbb47f8d057ba9b61d64b6a6feb5dee508ffd4182a9e"
    )
