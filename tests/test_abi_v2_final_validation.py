from pathlib import Path

from abi_v2.build_final_validation_bundle import checklist, environment_lock
from abi_v2.final_validation import (
    shortcut_audit,
    validate_human_packet,
)
from abi_v2.hostile_final_validation import run as hostile_run
from abi_v2.strict_validation import (
    read_json,
    verify_live_causality,
    verify_locked_matrix_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_candidate_pins_exact_architecture_lineage_and_payloads():
    value = read_json(
        ROOT / "results/abi_final_validation_v2/frozen_release_candidate_r6.json"
    )
    assert value["format"] == "abi-v2-repaired-frozen-release-candidate/6"
    assert len(value["technical_proof_commit"]) == 40
    assert len(value["capability_artifacts"]) == 4
    assert value["trusted_scientific_booleans_consumed"] == 0


def test_live_host_causality_uses_eight_fresh_conditions():
    value = verify_live_causality(ROOT)
    assert value["raw_rows"] == 3072
    assert value["cross_host_real_outputs_equal"] == 128
    assert all(
        host["fresh_condition_processes"] == 8 for host in value["hosts"].values()
    )


def test_final_verifier_recomputes_raw_headlines_without_summary_input():
    value = verify_locked_matrix_rows(ROOT)
    assert value["rows_verified"] == 5043
    assert value["cross_host_outputs_equal"] == 1681
    assert value["cross_host_specialist_actions_equal"] == 300


def test_shortcut_human_and_hostile_audits_are_fail_closed():
    shortcut = shortcut_audit(ROOT)
    human = validate_human_packet(ROOT)
    hostile = hostile_run(ROOT)
    assert shortcut["status"] == "PASS_NO_BLOCKING_SHORTCUT_OR_LEAKAGE_PATH"
    assert human["status"] == "TURNKEY_AWAITING_THREE_REAL_INDEPENDENT_HUMANS"
    assert human["ratings_completed_by_codex"] == 0
    assert hostile["status"] == "PASS_ALL_HOSTILE_RELEASE_MUTATIONS_REJECTED"
    assert hostile["mutations_rejected"] == hostile["mutations_required"]
    assert hostile["forbidden_attempts_rejected"] == hostile["forbidden_attempts_required"]


def test_external_package_exposes_exact_turnkey_commands_and_lock():
    value = checklist()
    assert value["status"] == (
        "CLOSED_UNTIL_PUBLIC_RECONSTRUCTION_AND_BLIND_RED_TEAM_PASS"
    )
    assert value["commands"] == [
        "abi-reproduce verify",
        "abi-reproduce certify-hosts",
        "abi-reproduce capability-matrix",
        "abi-reproduce causality",
        "abi-reproduce isolation",
        "abi-reproduce performance",
        "abi-reproduce hostile-audit",
        "abi-reproduce report",
    ]
    assert len(value["prerequisites"]) == 3
    assert environment_lock()["python"] == "3.10"


def test_reviewer_packet_and_claim_document_are_complete():
    expected = (
        "READ_ME_FIRST",
        "CLAIM_MATRIX",
        "ARCHITECTURE",
        "CANONICAL_ABI_SPEC",
        "HOST_CERTIFICATION",
        "CAPABILITY_ARTIFACTS",
        "HOST_CAUSALITY",
        "SEMANTIC_RETENTION",
        "MATHEMATICAL_PORTABILITY",
        "CAPABILITY_ISOLATION",
        "RUNTIME_PERFORMANCE",
        "INFORMATION_ACCOUNTING",
        "HOSTILE_AUDIT",
        "EXTERNAL_REPRODUCTION",
        "HUMAN_EVALUATION",
        "LIMITATIONS",
    )
    assert all((ROOT / "review_packet" / f"{index:02d}_{name}.md").is_file() for index, name in enumerate(expected))
    claims = (ROOT / "docs/ABI_TECHNICAL_CLAIMS.md").read_text(encoding="utf-8")
    assert "standalone capability-runtime" in claims
    assert "Not base-weight tensor transplantation" in claims
    assert "Not independently reproduced" in claims
