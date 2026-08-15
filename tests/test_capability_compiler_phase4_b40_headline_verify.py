import hashlib

import pytest

from abi.capability_compiler_phase2_common import canonical_json_bytes
from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_b40_headline_verify import (
    EXPECTED_OBSERVATIONS_PER_RUN,
    FORMAT,
    SEEDS,
    SYSTEMS,
    expected_configurations,
    headline_tree_sha256,
)
from abi.capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid


def test_b40_headline_verifier_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b40-headline-verify/1"
    assert SYSTEMS == ("L0", "L1", "D0")
    assert SEEDS == (104729, 130363, 155921)
    assert EXPECTED_OBSERVATIONS_PER_RUN == 1400
    assert expected_configurations() == {
        "L0": (16, 1e-4, 4),
        "L1": (8, 1e-4, 4),
        "D0": (None, 3e-5, 4),
    }


def test_tree_digest_rejects_mutation_and_extra_file(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    evidence = run / "phase4_result.json"
    evidence.write_text("one", encoding="utf-8")
    original, files = headline_tree_sha256(tmp_path)
    assert len(files) == 1
    evidence.write_text("two", encoding="utf-8")
    mutated, _ = headline_tree_sha256(tmp_path)
    assert mutated != original
    (run / "unexpected.bin").write_bytes(b"extra")
    extra, files = headline_tree_sha256(tmp_path)
    assert extra != mutated and len(files) == 2


def test_evidence_digest_rejects_metric_mutation():
    result = {"status": "PASS", "metric": 1}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    assert result_evidence_digest_valid(result)
    result["metric"] = 2
    assert not result_evidence_digest_valid(result)


def test_symlink_tree_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("evidence", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        assert not link.exists()
        return
    with pytest.raises(Phase3Error, match="symlink"):
        headline_tree_sha256(tmp_path)
