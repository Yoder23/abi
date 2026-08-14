from abi.capability_compiler_phase4_b50_grid_verify import (
    FORMAT,
    RUN_RESULT_FORMAT,
    expected_configurations,
    grid_tree_sha256,
    rank_rows,
    result_evidence_digest_valid,
)
from abi.capability_compiler_phase2_common import canonical_json_bytes
import hashlib


def _row(passes, collapses, loss, seconds, identity_exposure):
    return {
        "system": "D0",
        "configuration": {"rank": None, "learning_rate": 3e-5, "exposures": identity_exposure},
        "macro_functional_rate_v1": passes / 140,
        "functional_passes_v1": passes,
        "repetition_collapses_v1": collapses,
        "response_loss": loss,
        "imported_information_scalars": 10,
        "training_seconds": seconds,
    }


def test_grid_verifier_format_and_matrix_are_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-grid-verify/1"
    assert RUN_RESULT_FORMAT == "abi-capability-compiler-phase4-b50-baseline-run-result/1"
    assert {key: len(value) for key, value in expected_configurations().items()} == {
        "L0": 8,
        "L1": 8,
        "D0": 6,
        "D1": 6,
    }


def test_ranking_uses_v1_macro_before_collapse_and_loss():
    higher_with_collapse = _row(1, 9, 9.0, 9.0, 4)
    lower_zero_collapse = _row(0, 0, 0.1, 0.1, 2)
    assert rank_rows([lower_zero_collapse, higher_with_collapse])[0] is higher_with_collapse


def test_ranking_prefers_zero_collapse_then_loss_on_quality_tie():
    collapsed_low_loss = _row(0, 1, 0.1, 1.0, 1)
    zero_high_loss = _row(0, 0, 9.0, 9.0, 2)
    assert rank_rows([collapsed_low_loss, zero_high_loss])[0] is zero_high_loss
    low_loss = _row(0, 2, 0.2, 5.0, 4)
    high_loss = _row(0, 1, 0.3, 1.0, 2)
    assert rank_rows([high_loss, low_loss])[0] is low_loss


def test_grid_tree_rejects_content_mutation_and_extra_evidence(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "phase4_result.json").write_text("one", encoding="utf-8")
    original, files = grid_tree_sha256(tmp_path)
    assert len(files) == 1
    (run / "phase4_result.json").write_text("two", encoding="utf-8")
    mutated, _ = grid_tree_sha256(tmp_path)
    assert mutated != original
    (run / "receipt.json").write_text("extra", encoding="utf-8")
    extra, files = grid_tree_sha256(tmp_path)
    assert extra != mutated and len(files) == 2


def test_result_evidence_digest_rejects_metric_mutation():
    result = {"status": "PASS", "metric": 1}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    assert result_evidence_digest_valid(result)
    result["metric"] = 2
    assert not result_evidence_digest_valid(result)
