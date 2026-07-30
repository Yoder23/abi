import torch
import torch.nn as nn
from pathlib import Path

from abi.artifacts import (
    build_abi_artifact,
    build_compatibility_certificate,
    build_cost_ledger,
    module_param_count,
    module_state_sha256,
)


def test_module_state_sha256_is_stable_and_weight_sensitive():
    layer = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.25)

    first = module_state_sha256(layer)
    second = module_state_sha256(layer)
    assert first == second

    with torch.no_grad():
        layer.weight[0, 0] = 0.5
    assert module_state_sha256(layer) != first


def test_abi_artifact_verifies_frozen_core_but_allows_domain_ln_change():
    artifact = build_abi_artifact(
        source_model="source",
        target_model="target",
        d_abi=16,
        domain_corpus="wikitext",
        calibration_mode="freeze_domain_net",
        calibration_init="native",
        source_domain_core_sha256="source-core",
        source_domain_full_sha256="source-full",
        rotated_core_initial_sha256=["core-a"],
        rotated_core_final_sha256=["core-a"],
        rotated_full_initial_sha256=["full-a"],
        rotated_full_final_sha256=["full-b"],
        copied_payload_core_params=128,
        copied_payload_full_params=160,
        trainable_groups=[{"name": "proj_in", "params": 64}],
        alignment={"align_map": "procrustes", "final_after_mean": 0.8},
        target_side_components=["proj_in"],
    )

    assert artifact["schema_version"] == "abi-artifact-v1"
    assert artifact["domain_core_frozen_claim"] is True
    assert artifact["domain_core_unchanged_during_calibration"] is True
    assert artifact["full_domain_module_unchanged_during_calibration"] is False
    assert artifact["full_domain_module_change_allowed"] is True


def test_compatibility_certificate_keeps_source_preservation_as_open_gate():
    artifact = build_abi_artifact(
        source_model="source",
        target_model="target",
        d_abi=16,
        domain_corpus="wikitext",
        calibration_mode="freeze_domain_net",
        calibration_init="native",
        source_domain_core_sha256="source-core",
        source_domain_full_sha256="source-full",
        rotated_core_initial_sha256=["core-a"],
        rotated_core_final_sha256=["core-a"],
        rotated_full_initial_sha256=["full-a"],
        rotated_full_final_sha256=["full-a"],
        copied_payload_core_params=128,
        copied_payload_full_params=160,
        trainable_groups=[],
        alignment={"final_after_mean": 0.8},
        target_side_components=[],
    )

    cert = build_compatibility_certificate(
        artifact=artifact,
        alignment={"final_after_mean": 0.8},
        nib_l2={"pass": True},
        posthoc_logit_scale={"applied": False},
        posthoc_logit_bias={"applied": False},
        target_native_oracle_required=True,
    )

    assert cert["schema_version"] == "abi-compatibility-v1"
    assert cert["target_native_oracle_required"] is True
    assert cert["gates"]["target_native_nib_pass"] is True
    assert cert["gates"]["source_preservation_measured"] is False
    assert cert["gates"]["selective_transfer_measured"] is False
    assert cert["gates"]["off_domain_no_leakage_pass"] is None
    assert "does not by itself prove lossless" in cert["claim_boundary"]


def test_compatibility_certificate_marks_source_preservation_when_measured():
    artifact = build_abi_artifact(
        source_model="source",
        target_model="target",
        d_abi=16,
        domain_corpus="wikitext",
        calibration_mode="freeze_domain_net",
        calibration_init="native",
        source_domain_core_sha256="source-core",
        source_domain_full_sha256="source-full",
        rotated_core_initial_sha256=["core-a"],
        rotated_core_final_sha256=["core-a"],
        rotated_full_initial_sha256=["full-a"],
        rotated_full_final_sha256=["full-a"],
        copied_payload_core_params=128,
        copied_payload_full_params=160,
        trainable_groups=[],
        alignment={"final_after_mean": 0.8},
        target_side_components=[],
    )

    cert = build_compatibility_certificate(
        artifact=artifact,
        alignment={"final_after_mean": 0.8},
        nib_l2={"pass": True},
        posthoc_logit_scale={"applied": False},
        posthoc_logit_bias={"applied": False},
        target_native_oracle_required=True,
        source_preservation={"measured": True, "top1_surface_agree": 0.5},
    )

    assert cert["gates"]["source_preservation_measured"] is True
    assert cert["source_preservation"]["top1_surface_agree"] == 0.5


def test_compatibility_certificate_marks_selective_transfer_when_measured():
    artifact = build_abi_artifact(
        source_model="source",
        target_model="target",
        d_abi=16,
        domain_corpus="wikitext",
        calibration_mode="freeze_domain_net",
        calibration_init="native",
        source_domain_core_sha256="source-core",
        source_domain_full_sha256="source-full",
        rotated_core_initial_sha256=["core-a"],
        rotated_core_final_sha256=["core-a"],
        rotated_full_initial_sha256=["full-a"],
        rotated_full_final_sha256=["full-a"],
        copied_payload_core_params=128,
        copied_payload_full_params=160,
        trainable_groups=[],
        alignment={"final_after_mean": 0.8},
        target_side_components=[],
    )

    cert = build_compatibility_certificate(
        artifact=artifact,
        alignment={"final_after_mean": 0.8},
        nib_l2={"pass": True},
        posthoc_logit_scale={"applied": False},
        posthoc_logit_bias={"applied": False},
        target_native_oracle_required=True,
        selective_transfer={
            "enabled": True,
            "off_domain_no_leakage_pass": True,
            "selective_transfer_pass": True,
        },
    )

    assert cert["gates"]["selective_transfer_measured"] is True
    assert cert["gates"]["off_domain_no_leakage_pass"] is True
    assert cert["gates"]["selective_transfer_pass"] is True
    assert cert["selective_transfer"]["enabled"] is True


def test_compatibility_certificate_marks_oracle_light_scope():
    artifact = build_abi_artifact(
        source_model="source",
        target_model="target",
        d_abi=16,
        domain_corpus="wikitext",
        calibration_mode="freeze_domain_net",
        calibration_init="native",
        oracle_mode="target_base_interface",
        source_domain_core_sha256="source-core",
        source_domain_full_sha256="source-full",
        rotated_core_initial_sha256=["core-a"],
        rotated_core_final_sha256=["core-a"],
        rotated_full_initial_sha256=["full-a"],
        rotated_full_final_sha256=["full-a"],
        copied_payload_core_params=128,
        copied_payload_full_params=160,
        trainable_groups=[],
        alignment={"final_after_mean": 0.8},
        target_side_components=[],
    )

    cert = build_compatibility_certificate(
        artifact=artifact,
        alignment={"final_after_mean": 0.8},
        nib_l2={"pass": True},
        posthoc_logit_scale={"applied": False},
        posthoc_logit_bias={"applied": False},
        target_native_oracle_required=False,
        oracle_mode="target_base_interface",
    )

    assert artifact["oracle_mode"] == "target_base_interface"
    assert cert["claim_scope"] == "oracle_light_transfer_probe"
    assert cert["target_native_oracle_required"] is False
    assert cert["gates"]["oracle_light_mode"] is True
    assert cert["gates"]["target_reference_nib_pass"] is True
    assert cert["gates"]["target_native_nib_pass"] is None


def test_cost_ledger_counts_all_declared_phases():
    ledger = build_cost_ledger(
        phase_seconds={"a": 1.5, "c": 2.0, "d": 3.25},
        phase_steps={"source": 10, "target": 20},
        token_counts={"domain": 1000},
        param_counts={"copied_core": 42},
        repeat_count=2,
    )

    assert ledger["schema_version"] == "abi-cost-ledger-v1"
    assert ledger["total_measured_seconds"] == 6.75
    assert ledger["repeat_count"] == 2
    assert ledger["failed_search_runs_counted"] == 0


def test_param_count_helper_matches_torch_module():
    module = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    expected = sum(param.numel() for param in module.parameters())
    assert module_param_count(module) == expected


def test_generic_runner_emits_artifact_certificate_and_cost_schema():
    root = Path(__file__).resolve().parents[1]
    text = (root / "exp_generic_causal_nib_v2.py").read_text(
        encoding="utf-8"
    )

    assert "from abi.artifacts import" in text
    assert "build_abi_artifact" in text
    assert "build_compatibility_certificate" in text
    assert "build_cost_ledger" in text
    assert "compatibility_certificate" in text
    assert "cost_ledger" in text
    assert "ABI_SOURCE_PRESERVATION_EVAL" in text
    assert "source_preservation" in text
    assert "collect_source_preservation_source" in text
    assert "evaluate_source_preservation_target" in text
    assert "ABI_ORACLE_MODE" in text
    assert "target_base_interface" in text
    assert "base_target_reference" in text
    assert "TARGET_REFERENCE_USES_DOMAIN" in text
    assert "TARGET_REFERENCE_BYPASS_ABI" in text
    assert "TARGET_REFERENCE_FORWARD_MODE" in text
    assert 'use_domain == "base"' in text
    assert "ABI_TARGET_INTERFACE_CACHE" in text
    assert "save_target_interface_cache" in text
    assert "load_target_interface_cache" in text
    assert "target_interface_cache" in text
    assert "skipped_phase_c_training" in text
    assert "ABI_SOURCE_COMPLETION_LOSS_WEIGHT" in text
    assert "ABI_SOURCE_COMPLETION_MARGIN_WEIGHT" in text
    assert "ABI_SOURCE_COMPLETION_MARGIN" in text
    assert "ABI_SOURCE_COMPLETION_NLL_WEIGHT" in text
    assert "ABI_SOURCE_COMPLETION_NLL_CAP" in text
    assert "ABI_SOURCE_COMPLETION_LOSS_START_STEP" in text
    assert "ABI_SELECTIVE_TRANSFER_EVAL" in text
    assert "ABI_SELECTIVE_OFF_DOMAIN_CORPUS" in text
    assert "selective_transfer" in text
    assert "off_domain_no_leakage_pass" in text
    assert "ABI_CAL_LR" in text
    assert '"cal_lr": CAL_LR' in text
    assert "target_completion_mean_logprob_tensor" in text
    assert "prepare_source_completion_loss_records" in text
    assert "source_completion_loss" in text
    assert "margin_weight" in text
    assert "nll_weight" in text
    assert "nll_cap" in text
    assert "torch.relu(" in text
    assert "add_special_tokens=False" in text
    assert "torch.cat([prompt_ids, continuation_ids], dim=1)" in text
