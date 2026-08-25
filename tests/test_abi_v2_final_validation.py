from pathlib import Path

from abi_v2.build_final_validation_bundle import checklist, environment_lock
from abi_v2.final_validation import (
    freeze_release_candidate,
    host_causality,
    recompute_headlines,
    shortcut_audit,
    validate_human_packet,
)
from abi_v2.hostile_final_validation import run as hostile_run

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_candidate_pins_exact_architecture_lineage_and_payloads():
    value = freeze_release_candidate(ROOT)
    assert value["technical_proof_commit"] == "acfed2a225a32d36c32b625e35c6ede536cfab01"
    assert value["canonical_abi_version"] == "abi-canonical-host/2"
    assert len(value["capability_artifacts"]) == 4
    assert len(value["host_adapters"]) == 3
    assert all(row["parameters"] == 0 for row in value["host_adapters"].values())


def test_host_causality_falsifies_host_model_semantic_causality():
    value = host_causality(ROOT)
    assert value["status"] == "PASS_WITH_CLAIM_NARROWED_TO_STANDALONE_CAPABILITY_RUNTIME"
    assert value["host_semantic_state_channel_absent"] is True
    assert value["aggregate"]["neutral_stub_exact_outputs"] == value["aggregate"]["neutral_stub_outputs_total"]
    assert value["host_substitution"]["canonical_outputs_identical"] == value["host_substitution"]["canonical_outputs_total"]


def test_final_verifier_recomputes_raw_headlines_without_summary_input():
    value = recompute_headlines(ROOT)
    aggregate = value["aggregate"]
    assert value["summary_files_trusted"] is False
    assert value["headline_constants_embedded"] is False
    assert aggregate["matrix_cells_passed"] == aggregate["matrix_cells_total"]
    assert aggregate["frozen_source_successes"] == aggregate["frozen_source_successes_required"]
    assert aggregate["cross_host_output_equal"] == aggregate["cross_host_output_total"]
    assert aggregate["cross_host_specialist_actions_equal"] == aggregate["cross_host_specialist_actions_total"]


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
