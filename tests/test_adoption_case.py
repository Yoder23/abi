import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_summary():
    return json.loads((ROOT / "proof_layers_summary.json").read_text(encoding="utf-8"))


def test_adoption_case_is_generated_from_measured_artifacts():
    summary = load_summary()
    report = (ROOT / "ADOPTION_CASE.md").read_text(encoding="utf-8")
    case = summary["adoption_case"]

    assert "Measured Compute Story" in report
    assert "Measured Accuracy Cost" in report
    assert "No-Leakage Certificate" in report
    assert "Matched Baseline Check" in report
    assert "ABI vs LoRA Frontier" in report
    assert "Flagship GPT-Style Scenario" in report
    assert "North-Star Transfer Gates" in report
    assert "Adoption Gates" in report
    assert "not yet lossless" in report
    assert "Selective-transfer status" in report
    assert "targeted lossless migration" in report
    assert "Margin-hardened source-surface repair" in report
    assert "source-token surface preservation is still low" not in report
    assert "not yet literal lossless token-level migration" in report
    assert "not production-ready" in report
    assert "Phase-C-skipped base-reference bypass" in report
    assert "Non-fixed posthoc base-bypass" in report
    assert case["all_savings_rows_pass"] is True


def test_adoption_case_quantifies_compute_and_accuracy_cost():
    case = load_summary()["adoption_case"]

    assert case["large_target_row_count"] >= 4
    assert case["large_target_max_trainable_fraction"] < 0.003
    assert case["large_target_min_frozen_fraction"] > 0.997
    assert 0.0 < case["mean_ppl_relative_overhead"] < 0.12
    assert 0.0 < case["max_ppl_relative_overhead"] < 0.12


def test_adoption_case_keeps_open_gates_explicit():
    case = load_summary()["adoption_case"]
    best = case["best_hard_direction"]
    wikitext = case["best_generic_wikitext_rank_transfer"]
    gates = " ".join(case["adoption_gates"])
    north_star = case["north_star_gates"]
    selective = case["selective_transfer_gates"]

    assert best["top5"] >= 0.885
    assert best["pass"] is True
    assert wikitext["top5"] >= 0.90
    assert wikitext["pass"] is True
    assert wikitext["entropy_diff"] < 0.35
    assert wikitext["posthoc_logit_scale"]["applied"] is True
    assert case["posthoc_wikitext_pass_count"] >= 2
    assert case["posthoc_wikitext_min_top5"] >= 0.86
    assert case["posthoc_wikitext_max_entropy_diff"] < 0.35
    assert case["reverse_wikitext_pass_count"] >= 2
    assert case["reverse_wikitext_min_top5"] >= 0.86
    assert case["reverse_wikitext_max_entropy_diff"] < 0.35
    assert case["bidirectional_wikitext_pair_count"] >= 3
    assert case["repeat_certified_bidirectional_wikitext_pair_count"] >= 3
    assert case["matched_baseline_row_count"] >= 1
    assert case["best_matched_baseline"]["baseline_type"] == "target_side_lora_kd"
    assert case["matched_withheld_baseline_row_count"] >= 1
    assert case["matched_withheld_baseline_pass_count"] >= 1
    assert case["best_matched_withheld_baseline"]["withheld_nib_eval"] is True
    assert case["best_matched_withheld_baseline"]["pass"] is True
    assert case["withheld_eval_row_count"] >= 10
    assert case["withheld_eval_pass_count"] >= 10
    assert case["withheld_eval_all_split_separated"] is True
    assert case["best_withheld_eval"]["pass"] is True
    assert case["best_withheld_eval"]["top5"] > case["best_matched_withheld_baseline"]["top5"]
    assert case["withheld_eval_min_top5"] >= 0.886
    assert case["withheld_eval_max_entropy_diff"] < 0.35
    assert case["non_gpt_qwen_phi_withheld_pass_count"] >= 4
    assert case["non_gpt_qwen_phi_withheld_min_top5"] >= 0.886
    assert case["non_gpt_qwen_phi_withheld_max_entropy_diff"] < 0.35
    frontier = case["abi_vs_lora_frontier"]
    best_frontier = frontier["best"]
    repeat_frontier = frontier["repeat"]
    heldout_lora = frontier["heldout_lora_baseline"]
    assert frontier["best_beats_heldout_lora_all_metrics"] is True
    assert best_frontier["frontier_role"] == (
        "phi3_qwen_d1024_nativeinit_ema9995_cal14400_seed42off100k"
    )
    assert best_frontier["calibration_steps"] == 14400
    assert best_frontier["seed_offset"] == 100000
    assert best_frontier["top5"] > heldout_lora["top5"]
    assert best_frontier["top1"] > heldout_lora["top1"]
    assert best_frontier["js"] < heldout_lora["js"]
    assert best_frontier["entropy_diff"] < heldout_lora["entropy_diff"]
    assert frontier["repeat_passes_full_nib"] is True
    assert repeat_frontier["frontier_role"] == (
        "phi3_qwen_d1024_nativeinit_ema9995_cal14400_seed42"
    )
    assert repeat_frontier["calibration_steps"] == 14400
    assert repeat_frontier["seed_offset"] == 0
    assert repeat_frontier["pass"] is True
    assert repeat_frontier["top5"] > heldout_lora["top5"]
    assert repeat_frontier["top1"] > heldout_lora["top1"]
    assert frontier["repeat_beats_heldout_lora_all_metrics"] is True
    assert frontier["rank_dominance_repeat_certified"] is True
    phi_frontier = case["phi_vs_lora_frontier"]
    phi_lora = phi_frontier["heldout_lora_baseline"]
    assert phi_lora["target"] == "microsoft/phi-3-mini-4k-instruct"
    assert phi_lora["pass"] is True
    assert phi_lora["stopped_early"] is False
    assert phi_lora["completed_calibration_steps"] == phi_lora["requested_calibration_steps"]
    assert phi_frontier["abi_beats_phi_lora_all_metrics"] is True
    assert phi_frontier["all_metric_repeat_certified"] is True
    assert phi_frontier["best_abi_by_top5"]["top5"] > phi_lora["top5"]
    assert phi_frontier["best_abi_by_top5"]["js"] < phi_lora["js"]
    assert phi_frontier["best_abi_by_top5"]["entropy_diff"] < phi_lora["entropy_diff"]
    assert phi_frontier["best_abi_by_top5"]["top1"] > phi_lora["top1"]
    assert phi_frontier["best_abi_by_top5"]["frontier_role"] == (
        "qwen2_1p5b_d1024_align5000_nativeinit_ema9995_seed42off100k"
    )
    assert phi_frontier["best_abi_by_top1"]["top1"] > phi_lora["top1"]
    assert phi_frontier["best_abi_by_top1"]["frontier_role"] == (
        "qwen2_1p5b_d1024_align5000_nativeinit_ema9995_seed42off100k"
    )
    assert phi_frontier["best_all_metric_winner"] is not None
    assert phi_frontier["best_all_metric_winner"]["top1"] > phi_lora["top1"]
    assert phi_frontier["best_all_metric_winner"]["top5"] > phi_lora["top5"]
    assert phi_frontier["best_all_metric_winner"]["js"] < phi_lora["js"]
    assert (
        phi_frontier["best_all_metric_winner"]["entropy_diff"]
        < phi_lora["entropy_diff"]
    )
    assert phi_frontier["repeat_for_best_all_metric_winner"] is not None
    assert (
        phi_frontier["repeat_for_best_all_metric_winner"]["top1"]
        > phi_lora["top1"]
    )
    assert (
        phi_frontier["repeat_for_best_all_metric_winner"]["top5"]
        > phi_lora["top5"]
    )
    assert (
        phi_frontier["repeat_for_best_all_metric_winner"]["js"]
        < phi_lora["js"]
    )
    assert (
        phi_frontier["repeat_for_best_all_metric_winner"]["entropy_diff"]
        < phi_lora["entropy_diff"]
    )
    assert phi_frontier["repeat_for_best_all_metric_winner"]["frontier_role"] == (
        "qwen2_1p5b_d1024_align5000_nativeinit_ema9995_seed42"
    )
    assert phi_frontier["repeat_beats_phi_lora_all_metrics"] is True
    flagship = case["flagship_gpt_style"]
    assert flagship["scenario"] == "gpt2-medium -> microsoft/phi-3-mini-4k-instruct"
    assert flagship["ready"] is True
    assert flagship["benchmark_ready"] is True
    assert flagship["completion_ready"] is True
    assert flagship["production_ready"] is False
    assert flagship["oracle_light_distribution_pass_count"] >= 2
    assert flagship["repeat_distribution_recipe_count"] >= 1
    assert flagship["repeat_distribution_completion_recipe_count"] >= 1
    assert flagship["base_reference_negative_control_count"] >= 1
    assert flagship["base_reference_negative_control_best_top5"] < 0.95
    assert flagship["base_reference_strict_completion_pass_count"] >= 2
    assert flagship["base_reference_repeat_strict_completion_recipe_count"] >= 1
    assert flagship["base_reference_bypass_strict_completion_pass_count"] >= 2
    assert (
        flagship["base_reference_bypass_repeat_strict_completion_recipe_count"]
        >= 1
    )
    assert flagship["base_reference_bypass_nonfixed_strict_completion_pass_count"] >= 2
    assert (
        flagship[
            "base_reference_bypass_nonfixed_repeat_strict_completion_recipe_count"
        ]
        >= 1
    )
    assert (
        flagship[
            "base_reference_bypass_nonfixed_repeat_strict_completion_domain_count"
        ]
        >= 2
    )
    assert set(
        flagship["base_reference_bypass_nonfixed_repeat_strict_completion_domains"]
    ) >= {"python", "wikitext"}
    assert flagship["base_reference_bypass_nonfixed_best_top5"] >= 0.972
    assert flagship["base_reference_bypass_nonfixed_best_completion"] >= 0.70
    assert (
        flagship["base_reference_bypass_cross_target_strict_completion_pass_count"]
        >= 10
    )
    assert (
        flagship[
            "base_reference_bypass_cross_target_repeat_strict_completion_recipe_count"
        ]
        >= 4
    )
    assert (
        flagship[
            "base_reference_bypass_cross_target_repeat_strict_completion_pair_count"
        ]
        >= 2
    )
    assert set(
        flagship["base_reference_bypass_cross_target_repeat_strict_completion_pairs"]
    ) >= {
        "gpt2-medium -> microsoft/phi-3-mini-4k-instruct",
        "gpt2-medium -> Qwen/Qwen2.5-0.5B",
    }
    assert flagship["base_reference_bypass_cross_target_best_top5"] >= 0.974
    assert flagship["base_reference_bypass_cross_target_best_completion"] >= 0.70
    assert (
        flagship[
            "base_reference_bypass_cross_target_source_surface_repair_pass_count"
        ]
        >= 2
    )
    assert (
        flagship[
            "base_reference_bypass_cross_target_source_surface_repair_repeat_recipe_count"
        ]
        >= 2
    )
    assert (
        flagship[
            "base_reference_bypass_cross_target_source_surface_repair_repeat_pair_count"
        ]
        >= 2
    )
    assert set(
        flagship[
            "base_reference_bypass_cross_target_source_surface_repair_repeat_pairs"
        ]
    ) >= {
        "gpt2-medium -> microsoft/phi-3-mini-4k-instruct",
        "gpt2-medium -> Qwen/Qwen2.5-0.5B",
    }
    assert (
        flagship[
            "base_reference_bypass_cross_target_best_source_top1_surface"
        ]
        >= 0.85
    )
    assert (
        flagship[
            "base_reference_bypass_cross_target_min_repaired_source_top1_surface"
        ]
        >= 0.40
    )
    assert (
        flagship[
            "base_reference_bypass_cross_target_min_repaired_source_top1_in_topk"
        ]
        >= 0.55
    )
    assert (
        flagship["base_reference_bypass_cross_target_min_repaired_completion"]
        >= 0.65
    )
    assert (
        flagship["base_reference_bypass_source_surface_repair_pass_count"]
        >= 12
    )
    assert (
        flagship[
            "base_reference_bypass_source_surface_repair_repeat_recipe_count"
        ]
        >= 5
    )
    assert (
        flagship["base_reference_bypass_source_surface_repair_repeat_pair_count"]
        >= 5
    )
    assert set(
        flagship["base_reference_bypass_source_surface_repair_repeat_pairs"]
    ) >= {
        "gpt2-medium -> microsoft/phi-3-mini-4k-instruct",
        "gpt2-medium -> Qwen/Qwen2.5-0.5B",
        "EleutherAI/gpt-neo-125M -> Qwen/Qwen2.5-0.5B",
        "microsoft/phi-3-mini-4k-instruct -> Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2-1.5B -> microsoft/phi-3-mini-4k-instruct",
    }
    assert (
        set(flagship["base_reference_bypass_source_surface_repair_repeat_sources"])
        >= {
            "gpt2-medium",
            "EleutherAI/gpt-neo-125M",
            "microsoft/phi-3-mini-4k-instruct",
            "Qwen/Qwen2-1.5B",
        }
    )
    assert (
        flagship["base_reference_bypass_source_surface_nll_repair_pass_count"]
        >= 6
    )
    assert (
        flagship["base_reference_bypass_source_surface_min_repaired_top1_in_topk"]
        >= 0.55
    )
    assert (
        flagship["base_reference_bypass_source_surface_min_repaired_completion"]
        >= 0.65
    )
    assert (
        flagship["hard_recipient_ordinary_source_surface_pass_count"]
        >= 2
    )
    assert (
        flagship["hard_recipient_ordinary_source_surface_repeat_recipe_count"]
        >= 1
    )
    assert (
        flagship["hard_recipient_ordinary_source_surface_repeat_pair_count"]
        >= 1
    )
    assert set(flagship["hard_recipient_ordinary_source_surface_repeat_pairs"]) >= {
        "EleutherAI/pythia-410m -> deepseek-ai/deepseek-coder-1.3b-base",
    }
    assert flagship["hard_recipient_ordinary_source_surface_min_top5"] >= 0.86
    assert flagship["hard_recipient_ordinary_source_surface_best_top5"] >= 0.933
    assert flagship["hard_recipient_ordinary_source_surface_best_top1"] >= 0.965
    assert flagship["hard_recipient_ordinary_source_surface_best_js"] <= 0.0054
    assert (
        flagship["hard_recipient_ordinary_source_surface_best_entropy_diff"]
        <= 0.175
    )
    assert (
        flagship["hard_recipient_ordinary_source_surface_min_top1_surface"]
        >= 0.40
    )
    assert (
        flagship["hard_recipient_ordinary_source_surface_min_top1_in_topk"]
        >= 0.55
    )
    assert (
        flagship["hard_recipient_ordinary_source_surface_min_completion"]
        >= 0.65
    )
    assert flagship["base_reference_bypass_best_top5"] >= 0.973
    assert flagship["base_reference_bypass_best_completion"] >= 0.687
    assert flagship["xavier_target_interface_strict_completion_pass_count"] == 0
    assert flagship["zeroout_target_interface_strict_completion_pass_count"] >= 2
    assert flagship["zeroout_target_interface_best_top5"] >= 0.954
    assert flagship["zeroout_target_interface_best_completion"] >= 0.698
    assert (
        flagship["zeroout_target_interface_repeat_strict_completion_recipe_count"]
        >= 1
    )
    assert flagship["target_interface_cache_saved_count"] >= 1
    assert flagship["target_interface_cache_loaded_count"] >= 1
    assert flagship["target_interface_cache_loaded_distribution_pass_count"] >= 1
    assert flagship["target_interface_cache_loaded_strict_completion_pass_count"] >= 2
    assert flagship["target_interface_cache_loaded_best_top5"] >= 0.969
    assert flagship["target_interface_cache_loaded_best_completion"] >= 0.70
    assert flagship["source_completion_loss_row_count"] >= 4
    assert (
        flagship[
            "target_interface_cache_loaded_source_completion_loss_strict_completion_pass_count"
        ]
        >= 2
    )
    assert (
        flagship[
            "target_interface_cache_loaded_source_completion_loss_repeat_recipe_count"
        ]
        >= 1
    )
    assert flagship["native_target_interface_strict_completion_pass_count"] >= 2
    blockers = " ".join(flagship["production_blockers"])
    assert "No GPT5/GPT6" in blockers
    assert "No-base-reference xavier control still fails" in blockers
    assert "Phase-C-skipped base-reference bypass" in blockers
    assert "two GPT2-medium local target pairs" in blockers
    assert "GPT-Neo, Phi-3, and Qwen donors" in blockers
    assert "Qwen/Phi target directions" in blockers
    assert "one local GPT-style source/target pair" not in blockers
    assert "source/target/domain setting" not in blockers
    assert "validation-selected posthoc logit scaling" in blockers
    assert "fixed posthoc logit scaling" not in blockers
    assert "source-completion loss is enabled" in blockers
    strict_flagship = [
        row
        for row in flagship["rows"]
        if row["distribution_near_lossless_pass"]
    ]
    assert len(strict_flagship) >= 2
    assert any(
        row["oracle_mode"] == "base_target_reference"
        and row["target_reference_bypass_abi"] is True
        and row["target_reference_forward_mode"] == "base"
        for row in strict_flagship
    )
    assert any(row["calibration_init"] == "zero_out" for row in strict_flagship)
    assert max(row["js"] for row in strict_flagship) <= 0.005
    assert max(row["entropy_diff"] for row in strict_flagship) <= 0.05
    strict_completion = [
        row
        for row in flagship["rows"]
        if row["distribution_completion_pass"]
    ]
    assert len(strict_completion) >= 2
    assert any(row["calibration_steps"] == 9600 for row in strict_completion)
    assert min(row["source_completion_preferred"] for row in strict_completion) >= 0.50
    assert min(row["top5"] for row in strict_completion) >= 0.95
    base_bypass_strict = [
        row
        for row in strict_completion
        if row["oracle_mode"] == "base_target_reference"
        and row["target_reference_bypass_abi"] is True
        and row["target_reference_forward_mode"] == "base"
    ]
    assert len(base_bypass_strict) >= 2
    assert all(
        (row["source_completion_loss"] or {}).get("enabled")
        for row in base_bypass_strict
    )
    assert all(
        (row["calibration_ema"] or {}).get("restored")
        for row in base_bypass_strict
    )
    nonfixed_base_bypass = [
        row
        for row in base_bypass_strict
        if (row["posthoc_logit_scale"] or {}).get("candidate_count", 1) > 1
    ]
    assert len(nonfixed_base_bypass) >= 2
    assert {
        (row["posthoc_logit_scale"] or {}).get("selection_rule")
        for row in nonfixed_base_bypass
    } >= {"balanced_entropy"}
    cross_target_strict = [
        row
        for row in flagship["base_reference_bypass_cross_target_rows"]
        if row["distribution_completion_pass"]
    ]
    assert {
        row["target"] for row in cross_target_strict
    } >= {"microsoft/phi-3-mini-4k-instruct", "Qwen/Qwen2.5-0.5B"}
    qwen_repeat = [
        row
        for row in cross_target_strict
        if row["target"] == "Qwen/Qwen2.5-0.5B"
    ]
    assert len(qwen_repeat) >= 2
    assert min(row["top5"] for row in qwen_repeat) >= 0.965
    assert max(row["entropy_diff"] for row in qwen_repeat) <= 0.05
    assert min(row["source_completion_preferred"] for row in qwen_repeat) >= 0.63
    repaired_qwen = [
        row
        for row in qwen_repeat
        if row["source_surface_repair_pass"]
        and (row["source_completion_loss"] or {}).get("margin_weight") == 0.5
    ]
    assert len(repaired_qwen) >= 2
    assert min(row["source_preservation_top1"] for row in repaired_qwen) >= 0.40
    assert (
        min(row["source_preservation_top1_in_topk"] for row in repaired_qwen)
        >= 0.55
    )
    assert min(row["source_completion_preferred"] for row in repaired_qwen) >= 0.65
    repaired_phi = [
        row
        for row in cross_target_strict
        if row["target"] == "microsoft/phi-3-mini-4k-instruct"
        and row["source_surface_repair_pass"]
        and (row["source_completion_loss"] or {}).get("weight") == 0.2
        and (row["source_completion_loss"] or {}).get("margin_weight") == 1.0
    ]
    assert len(repaired_phi) >= 2
    assert min(row["top5"] for row in repaired_phi) >= 0.974
    assert max(row["entropy_diff"] for row in repaired_phi) <= 0.05
    assert min(row["source_preservation_top1"] for row in repaired_phi) >= 0.64
    assert (
        min(row["source_preservation_top1_in_topk"] for row in repaired_phi)
        >= 0.79
    )
    assert min(row["source_completion_preferred"] for row in repaired_phi) >= 0.73
    assert "LoRA" in gates
    assert "Phi-3" in gates
    assert "entropy" in gates
    assert "Withheld-domain" in gates
    assert "oracle-light" in gates
    assert "source-preservation" in gates
    assert "selective-transfer" in gates
    assert "off-domain" in gates
    assert north_star["repeat_required_recipe_count"] == 2
    assert north_star["joint_passing_count"] >= 6
    assert north_star["repeat_joint_passing_recipe_count"] >= 3
    assert north_star["repeat_joint_passing_pair_count"] >= 3
    assert selective["repeat_required_recipe_count"] == 2
    assert selective["strict_selective_transfer_pass_count"] == 0
    assert selective["repeat_strict_selective_recipe_count"] == 0
    assert selective["ready"] is False
    hard_pair_native = [
        row
        for row in north_star["rows"]
        if row["source"] == "EleutherAI/pythia-410m"
        and row["target"] == "deepseek-ai/deepseek-coder-1.3b-base"
        and row["calibration_init"] == "native"
        and row["nib_pass"] is True
    ]
    hard_pair_xavier = [
        row
        for row in north_star["rows"]
        if row["source"] == "EleutherAI/pythia-410m"
        and row["target"] == "deepseek-ai/deepseek-coder-1.3b-base"
        and row["calibration_init"] == "xavier"
    ]
    assert len(hard_pair_native) >= 2
    assert min(row["top5"] for row in hard_pair_native) >= 0.876
    assert any(row["nib_pass"] is False for row in hard_pair_xavier)
    assert case["north_star_ready"] == (
        north_star["repeat_joint_passing_recipe_count"]
        >= north_star["repeat_required_recipe_count"]
    )
    assert case["north_star_ready"] is True
    assert "oracle-light source-preservation" in case["north_star_status"]
    assert case["selective_transfer_ready"] is False
    assert "off-domain no-leakage" in case["selective_transfer_status"]
