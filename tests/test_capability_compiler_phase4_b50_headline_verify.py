import hashlib

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase2_common import canonical_json_bytes
from abi.capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from abi.capability_compiler_phase4_b50_headline_verify import (
    EXPECTED_OBSERVATIONS_PER_RUN,
    FORMAT,
    SEEDS,
    SYSTEMS,
    adversarial_test_evidence,
    checkpoint_filename,
    expected_configurations,
    expected_filenames,
    generation_attention_mask_audit,
    headline_tree_sha256,
)


def test_headline_verifier_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-headline-verify/1"
    assert SYSTEMS == ("L0", "L1", "D0", "D1", "D2")
    assert SEEDS == (104729, 130363, 155921)
    assert EXPECTED_OBSERVATIONS_PER_RUN == 1400
    assert expected_configurations() == {
        "L0": (16, 1e-4, 4),
        "L1": (8, 1e-4, 4),
        "D0": (None, 3e-5, 4),
        "D1": (None, 3e-5, 4),
        "D2": (None, 3e-5, 4),
    }


def test_checkpoint_filename_depends_only_on_system_role():
    assert checkpoint_filename("L0") == "adapters.safetensors"
    assert checkpoint_filename("L1") == "adapters.safetensors"
    assert checkpoint_filename("D0") == "student.safetensors"
    assert "adapters.safetensors" in expected_filenames("L0")
    assert "adapters.safetensors" in expected_filenames("L1")
    for system in ("D0", "D1", "D2"):
        assert "student.safetensors" in expected_filenames(system)
    assert all(len(expected_filenames(system)) == 4 for system in SYSTEMS)


def test_headline_tree_rejects_content_mutation_and_extra_file(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "phase4_result.json").write_text("one", encoding="utf-8")
    original, files = headline_tree_sha256(tmp_path)
    assert len(files) == 1
    (run / "phase4_result.json").write_text("two", encoding="utf-8")
    mutated, _ = headline_tree_sha256(tmp_path)
    assert mutated != original
    (run / "unexpected.bin").write_bytes(b"extra")
    extra, files = headline_tree_sha256(tmp_path)
    assert extra != mutated and len(files) == 2
    (tmp_path / "unexpected-root.bin").write_bytes(b"root-extra")
    root_extra, files = headline_tree_sha256(tmp_path)
    assert root_extra != extra and len(files) == 3


def test_result_evidence_digest_rejects_mutation():
    result = {"status": "PASS", "metric": 1}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    assert result_evidence_digest_valid(result)
    result["metric"] = 2
    assert not result_evidence_digest_valid(result)


def test_adversarial_test_evidence_requires_clean_hashed_junit(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="20" failures="0" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8",
    )
    protocol = {
        "adversarial_test_evidence": {
            "path": "report.xml",
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "minimum_tests": 20,
        }
    }
    assert adversarial_test_evidence(tmp_path, protocol)["tests"] == 20
    report.write_text(
        '<testsuites><testsuite tests="20" failures="1" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8",
    )
    protocol["adversarial_test_evidence"]["sha256"] = hashlib.sha256(
        report.read_bytes()
    ).hexdigest()
    with pytest.raises(Phase3Error, match="did not pass cleanly"):
        adversarial_test_evidence(tmp_path, protocol)


def test_attention_mask_audit_rejects_shared_eos_pad_inside_prompt():
    class Encoded:
        def __init__(self, values):
            self.input_ids = values

    class Tokenizer:
        eos_token_id = 7
        pad_token_id = 7

        def apply_chat_template(self, messages, **_):
            return messages[0]["content"]

        def __call__(self, rendered, **_):
            return Encoded([1, 7, 2] if rendered == "unsafe" else [1, 2])

    safe = generation_attention_mask_audit(
        {"p": {"prompt": "safe"}}, Tokenizer()
    )
    unsafe = generation_attention_mask_audit(
        {"p": {"prompt": "unsafe"}}, Tokenizer()
    )
    assert safe["implicit_all_ones_mask_semantically_correct"] is True
    assert unsafe["implicit_all_ones_mask_semantically_correct"] is False
