import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_proof_layers_summary_is_generated_from_current_artifacts():
    summary = load_json("proof_layers_summary.json")
    report = (ROOT / "PROOF_LAYERS.md").read_text(encoding="utf-8")
    assert "Layer 1 - Savings" in report
    assert "Layer 2 - Accuracy Frontier" in report
    assert "Layer 3 - Domain Breadth" in report
    assert "Layer 4 - Matched Baselines" in report
    assert "Layer 4b - ABI vs LoRA Frontier" in report
    assert "Layer 5 - Withheld Evaluation" in report
    assert "Layer 6 - North-Star Gates" in report
    assert "Selective Transfer / Off-Domain No-Leakage" in report
    assert "Layer 7 - Flagship GPT-Style Near-Lossless Scenario" in report
    assert summary["claim_boundary"]["cannot_claim_yet"].startswith("Lossless")
    north_star = summary["north_star_gates"]
    assert north_star["row_count"] >= north_star["oracle_light_row_count"]
    assert north_star["row_count"] >= north_star["source_preservation_row_count"]
    assert north_star["repeat_required_recipe_count"] == 2
    assert north_star["joint_passing_count"] >= 6
    assert north_star["repeat_joint_passing_recipe_count"] >= 3
    assert north_star["repeat_joint_passing_pair_count"] >= 3
    selective = summary["selective_transfer_gates"]
    assert selective["repeat_required_recipe_count"] == 2
    assert selective["ready"] is False
    assert selective["strict_selective_transfer_pass_count"] == 0
    assert selective["repeat_strict_selective_recipe_count"] == 0
    assert "off-domain" in " ".join(selective["open_blockers"])
    hard_pair_native = [
        row
        for row in north_star["rows"]
        if row["source"] == "EleutherAI/pythia-410m"
        and row["target"] == "deepseek-ai/deepseek-coder-1.3b-base"
        and row["calibration_init"] == "native"
        and row["nib_pass"] is True
    ]
    assert len(hard_pair_native) >= 2
    flagship = summary["flagship_gpt_style"]
    assert flagship["ready"] is True
    assert flagship["benchmark_ready"] is True
    assert flagship["completion_ready"] is True
    assert flagship["production_ready"] is False
    assert "not production-ready" in flagship["production_readiness_status"]
    assert flagship["oracle_light_distribution_pass_count"] >= 2
    assert flagship["repeat_distribution_recipe_count"] >= 1
    assert flagship["repeat_distribution_completion_recipe_count"] >= 1
    assert flagship["base_reference_negative_control_count"] >= 1
    assert flagship["base_reference_negative_control_best_top5"] < 0.95
    assert flagship["base_reference_best_top5"] >= 0.973
    assert flagship["base_reference_best_completion"] >= 0.687
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
    assert flagship["xavier_target_interface_best_top5"] < 0.95
    assert flagship["xavier_target_interface_strict_completion_pass_count"] == 0
    assert flagship["zeroout_target_interface_best_top5"] >= 0.954
    assert flagship["zeroout_target_interface_best_completion"] >= 0.698
    assert flagship["zeroout_target_interface_strict_completion_pass_count"] >= 2
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
    assert "GPT5/GPT6" in blockers
    assert "Phase-C-skipped base-reference bypass" in blockers
    assert "two GPT2-medium local target pairs" in blockers
    assert "GPT-Neo, Phi-3, and Qwen donors" in blockers
    assert "Qwen/Phi target directions" in blockers
    assert "one local GPT-style source/target pair" not in blockers
    assert "source/target/domain setting" not in blockers
    assert "validation-selected posthoc logit scaling" in blockers
    assert "fixed posthoc logit scaling" not in blockers
    assert "zero-out init" in blockers
    assert "source-completion loss is enabled" in blockers
    strict_rows = [
        row
        for row in flagship["rows"]
        if row["distribution_near_lossless_pass"]
    ]
    assert len(strict_rows) >= 2
    assert min(row["top5"] for row in strict_rows) >= 0.95
    assert min(row["top1"] for row in strict_rows) >= 0.965
    assert max(row["js"] for row in strict_rows) <= 0.005
    assert max(row["entropy_diff"] for row in strict_rows) <= 0.05
    assert min(row["source_preservation_top1_in_topk"] for row in strict_rows) < 0.30
    strict_completion_rows = [
        row
        for row in flagship["rows"]
        if row["distribution_completion_pass"]
    ]
    assert len(strict_completion_rows) >= 2
    assert min(row["top5"] for row in strict_completion_rows) >= 0.95
    assert min(row["source_completion_preferred"] for row in strict_completion_rows) >= 0.50
    assert any(row["calibration_steps"] == 9600 for row in strict_completion_rows)
    assert any(row["calibration_init"] == "zero_out" for row in strict_completion_rows)
    base_bypass_strict = [
        row
        for row in strict_completion_rows
        if row["oracle_mode"] == "base_target_reference"
        and row["target_reference_bypass_abi"] is True
        and row["target_reference_forward_mode"] == "base"
    ]
    assert len(base_bypass_strict) >= 2
    assert all(row["calibration_init"] == "zero_out" for row in base_bypass_strict)
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
    balanced_nonfixed_base_bypass = [
        row
        for row in nonfixed_base_bypass
        if (row["posthoc_logit_scale"] or {}).get("selection_rule")
        == "balanced_entropy"
    ]
    assert len(balanced_nonfixed_base_bypass) >= 2
    assert min(
        (row["posthoc_logit_scale"] or {}).get("candidate_count", 0)
        for row in balanced_nonfixed_base_bypass
    ) >= 41
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


def test_savings_layer_measures_small_target_side_calibration_fraction():
    summary = load_json("proof_layers_summary.json")
    fractions = [
        row["trainable_fraction_of_target"]
        for row in summary["savings"]["rows"]
        if row["target_params_counted"] is not None
    ]
    assert fractions
    assert min(fractions) < 0.002
    assert max(fractions) < 0.025


def test_base_bypass_recipe_key_tracks_source_completion_margin():
    text = (ROOT / "build_proof_layers.py").read_text(encoding="utf-8")
    assert 'loss.get("margin_weight")' in text
    assert 'loss.get("margin")' in text


def test_accuracy_frontier_records_repeatable_hard_direction_progress():
    summary = load_json("proof_layers_summary.json")
    rows = {
        row["file"]: row for row in summary["accuracy_frontier"]["rows"]
    }
    baseline = rows[
        "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_seed42_results.json"
    ]
    best = rows[
        "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed42_results.json"
    ]
    repeat = rows[
        "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed314_results.json"
    ]
    wider = rows[
        "exp_generic_causal_nib_v2_pythia410_deepseek_d1024_cal12000_topset5_bridge_seed42_results.json"
    ]
    assert best["top5"] > baseline["top5"]
    assert best["top5"] >= 0.885
    assert repeat["pass"] is True
    assert repeat["top5"] >= 0.877
    assert wider["pass"] is False


def test_domain_breadth_records_wikitext_gap_and_posthoc_repair():
    summary = load_json("proof_layers_summary.json")
    breadth = summary["domain_breadth"]
    assert breadth["atlas_domains_passing"] == 4
    assert breadth["atlas_domains_total"] == 4
    assert breadth["atlas_diagonal"]["wikitext"]["top5"] >= 0.95
    generic = breadth["generic_cross_model_wikitext"]
    rank_only = next(
        row
        for row in generic
        if "cal9600_topset5_bridge_wikitext_seed42" in row["file"]
    )
    repaired_rows = [
        row
        for row in generic
        if "posthocscale" in row["file"]
        and row["source"] == "EleutherAI/gpt-neo-125M"
        and row["target"] == "Qwen/Qwen2.5-0.5B"
    ]
    assert rank_only["top5"] >= 0.90
    assert rank_only["pass"] is False
    assert rank_only["entropy_diff"] > 0.35
    assert len(repaired_rows) >= 2
    assert all(row["top5"] >= 0.90 for row in repaired_rows)
    assert all(row["pass"] is True for row in repaired_rows)
    assert all(row["entropy_diff"] < 0.35 for row in repaired_rows)
    assert all(row["posthoc_logit_scale"]["applied"] is True for row in repaired_rows)
    reverse_rows = [
        row
        for row in generic
        if row["source"] == "Qwen/Qwen2.5-0.5B"
        and row["target"] == "EleutherAI/gpt-neo-125M"
    ]
    reverse_passes = [row for row in reverse_rows if row["pass"]]
    assert len(reverse_passes) >= 2
    assert all(row["top5"] >= 0.86 for row in reverse_passes)
    assert all(row["entropy_diff"] < 0.35 for row in reverse_passes)
    assert any(
        row["posthoc_logit_scale"].get("selection_rule") == "minimax_entropy"
        for row in reverse_passes
    )
    phi3_forward = [
        row
        for row in generic
        if row["source"] == "EleutherAI/gpt-neo-125M"
        and row["target"] == "microsoft/phi-3-mini-4k-instruct"
        and row["pass"]
    ]
    phi3_reverse = [
        row
        for row in generic
        if row["source"] == "microsoft/phi-3-mini-4k-instruct"
        and row["target"] == "EleutherAI/gpt-neo-125M"
        and row["pass"]
    ]
    assert len(phi3_forward) >= 2
    assert len(phi3_reverse) >= 2
    assert all(row["top5"] >= 0.86 for row in phi3_forward + phi3_reverse)
    assert all(row["entropy_diff"] < 0.35 for row in phi3_forward + phi3_reverse)
    qwen_phi_forward = [
        row
        for row in generic
        if row["source"] == "Qwen/Qwen2.5-0.5B"
        and row["target"] == "microsoft/phi-3-mini-4k-instruct"
        and row["pass"]
    ]
    qwen_phi_reverse = [
        row
        for row in generic
        if row["source"] == "microsoft/phi-3-mini-4k-instruct"
        and row["target"] == "Qwen/Qwen2.5-0.5B"
        and row["pass"]
    ]
    assert len(qwen_phi_forward) >= 2
    assert len(qwen_phi_reverse) >= 2
    assert all(row["top5"] >= 0.89 for row in qwen_phi_forward + qwen_phi_reverse)
    assert all(row["entropy_diff"] < 0.35 for row in qwen_phi_forward + qwen_phi_reverse)


def test_matched_lora_baseline_is_recorded():
    summary = load_json("proof_layers_summary.json")
    baselines = summary["baselines"]["rows"]
    assert summary["baselines"]["row_count"] >= 1
    assert any(row["baseline_type"] == "target_side_lora_kd" for row in baselines)
    assert any(row["lora_rank"] is not None for row in baselines)
    assert all(row["calibration_trainable_params"] > 0 for row in baselines)
    assert all(row["target_params_counted"] > 0 for row in baselines)
    withheld = [row for row in baselines if row["withheld_nib_eval"]]
    assert withheld
    assert all(row["wikitext_domain_split"] == "train" for row in withheld)
    assert all(row["wikitext_posthoc_split"] == "validation" for row in withheld)
    assert all(row["wikitext_eval_split"] == "test" for row in withheld)
    qwen_withheld = [row for row in withheld if row["target"] == "Qwen/Qwen2.5-0.5B"]
    phi_withheld = [
        row
        for row in withheld
        if row["target"] == "microsoft/phi-3-mini-4k-instruct"
    ]
    assert qwen_withheld
    assert phi_withheld
    assert max(row["top5"] for row in qwen_withheld) >= 0.90
    assert min(row["entropy_diff"] for row in qwen_withheld) > 0.35
    assert any(row["pass"] for row in phi_withheld)
    assert max(row["top5"] for row in phi_withheld) >= 0.875
    assert any(row["stopped_early"] for row in phi_withheld)
    assert any(
        row["completed_calibration_steps"] < row["requested_calibration_steps"]
        for row in phi_withheld
    )


def test_abi_frontier_beats_heldout_lora_once_and_tracks_repeat_limit():
    summary = load_json("proof_layers_summary.json")
    case = summary["adoption_case"]
    frontier = case["abi_vs_lora_frontier"]
    baseline = frontier["heldout_lora_baseline"]
    best = frontier["best"]
    repeat = frontier["repeat"]

    assert summary["abi_lora_frontier"]["row_count"] >= 15
    assert summary["abi_lora_frontier"]["passing_count"] >= 12
    assert frontier["row_count"] >= 15
    assert frontier["pass_count"] >= 12
    assert baseline["withheld_nib_eval"] is True
    assert baseline["file"].startswith("exp_lora_kd_baseline_")
    assert best["frontier_role"] == (
        "phi3_qwen_d1024_nativeinit_ema9995_cal14400_seed42off100k"
    )
    assert best["calibration_steps"] == 14400
    assert best["seed_offset"] == 100000
    assert best["pass"] is True
    assert frontier["best_beats_heldout_lora_all_metrics"] is True
    assert best["top5"] > baseline["top5"]
    assert best["top1"] > baseline["top1"]
    assert best["js"] < baseline["js"]
    assert best["entropy_diff"] < baseline["entropy_diff"]
    assert repeat["frontier_role"] == (
        "phi3_qwen_d1024_nativeinit_ema9995_cal14400_seed42"
    )
    assert repeat["calibration_steps"] == 14400
    assert repeat["seed_offset"] == 0
    assert frontier["repeat_passes_full_nib"] is True
    assert repeat["pass"] is True
    assert repeat["top5"] > baseline["top5"]
    assert repeat["top1"] > baseline["top1"]
    assert repeat["js"] < baseline["js"]
    assert repeat["entropy_diff"] < baseline["entropy_diff"]
    assert frontier["repeat_beats_heldout_lora_all_metrics"] is True
    assert frontier["rank_dominance_repeat_certified"] is True


def test_withheld_wikitext_certificate_is_split_separated():
    summary = load_json("proof_layers_summary.json")
    withheld = summary["withheld_evaluation"]
    rows = withheld["rows"]

    assert withheld["row_count"] >= 10
    assert withheld["passing_count"] >= 10
    assert withheld["all_rows_have_split_separation"] is True
    assert withheld["min_top5"] >= 0.886
    assert withheld["max_entropy_diff"] < 0.35
    assert all(row["withheld_nib_eval"] is True for row in rows)
    assert all(row["wikitext_domain_split"] == "train" for row in rows)
    assert all(row["wikitext_posthoc_split"] == "validation" for row in rows)
    assert all(row["wikitext_eval_split"] == "test" for row in rows)

    qwen_phi = [
        row
        for row in rows
        if {row["source"], row["target"]}
        == {"Qwen/Qwen2.5-0.5B", "microsoft/phi-3-mini-4k-instruct"}
    ]
    assert len(qwen_phi) >= 4
    assert all(row["pass"] for row in qwen_phi)
    assert min(row["top5"] for row in qwen_phi) >= 0.886
    assert max(row["entropy_diff"] for row in qwen_phi) < 0.35
