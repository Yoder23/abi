#!/usr/bin/env python3
"""Generate the measured ABI proof-layer report from result artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

import experiment_data


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "proof_layers_summary.json"
REPORT_PATH = ROOT / "PROOF_LAYERS.md"
ADOPTION_CASE_PATH = ROOT / "ADOPTION_CASE.md"

SAVINGS_RESULTS = [
    "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed42_results.json",
    "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed314_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_phi3_d1024_cal4800_topset5_bridge_seed42_results.json",
    "exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal4800_topset5_bridge_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal4800_topset5_bridge_seed42_results.json",
]

HARD_FRONTIER_RESULTS = [
    "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_seed42_results.json",
    "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal12000_topset5_bridge_seed42_results.json",
    "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed42_results.json",
    "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed314_results.json",
    "exp_generic_causal_nib_v2_pythia410_deepseek_d1024_cal12000_topset5_bridge_seed42_results.json",
]

WIKITEXT_GENERIC_RESULTS = [
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_wikitext_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_posthocscale_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_posthocscale_seed314_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal4800_topset5_bridge_wikitext_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal9600_topset5_bridge_wikitext_posthocscale_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal16000_topset5_bridge_wikitext_posthocscale_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal17000_topset5_bridge_wikitext_posthocscale_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal20000_topset5_bridge_wikitext_posthocscale_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal24000_topset5_bridge_wikitext_posthocscale_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal20000_topset5_bridge_wikitext_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal20000_topset5_bridge_wikitext_posthocminimax_seed314_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_phi3_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_phi3_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed314_results.json",
    "exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal20000_topset5_bridge_wikitext_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal20000_topset5_bridge_wikitext_posthocminimax_seed314_results.json",
    "exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal20000_topset5_bridge_wikitext_posthocbalanced_seed314_results.json",
    "exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal20000_topset5_bridge_wikitext_posthocbalancedw5_seed314_results.json",
    "exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal20000_topset5_bridge_wikitext_ent002_posthocbalanced_seed314_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed314_results.json",
    "exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed314_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_wikitext_ent1_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_wikitext_ent001_seed42_results.json",
    "exp_generic_causal_nib_v2_pythia410_pythia160_d768_cal2400_topk32_wikitext_seed42_results.json",
]

BASELINE_RESULT_GLOB = "exp_lora_kd_baseline_*_results.json"

WITHHELD_WIKITEXT_RESULTS = [
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_nativeinit_ema9995s8000_lrdecay8000x02_cal12000_topset5_bridge_wikitext_train_val_test_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_nativeinit_ema9995s8000_lrdecay8000x02_cal12000_topset5_bridge_wikitext_train_val_d7d416c519_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
    "exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_seed42_results.json",
    "exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_f4fba487b3_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
    "exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
    "exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
]

ABI_LORA_FRONTIER_RESULTS = [
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d512_reference",
        "comparison_note": "Original split-separated ABI certificate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal16000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d512_longer_calibration",
        "comparison_note": "Longer calibration improves rank but remains below the held-out LoRA rank metrics.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal16000_topset10_rankstrong_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d512_rank_strong_ablation",
        "comparison_note": "Over-constrained rank ablation; useful negative control.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d560_cal16000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d560_lora_capacity_match",
        "comparison_note": "Trainable-parameter match to the all-linear r=3 LoRA baseline.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal16000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d768_rank_frontier",
        "comparison_note": "Wider ABI crosses the held-out LoRA top-5 metric but not top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d768_lora_metric_win",
        "comparison_note": "Current held-out ABI frontier; beats the split-separated LoRA baseline on top-5, top-1, JS, and entropy.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "d768_lora_metric_win_repeat",
        "comparison_note": "Shifted-seed repeat passes full NIB but does not repeat the held-out LoRA rank win.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal19000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "d768_longer_repeat_ablation",
        "comparison_note": "Longer repeat-seed calibration stays NIB-passing but moves rank away from the LoRA gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal20000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d768_overrun_ablation",
        "comparison_note": "Longer calibration raises top-1 but breaks the entropy gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d896_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "d896_width_probe_repeat",
        "comparison_note": "Target-width ABI repeat passes NIB and top-1 but misses the LoRA top-5 gate by 0.0003.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d896_cal18000_topsetw6_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "d896_topsetw6_ablation",
        "comparison_note": "Small top-set weight increase hurts rank and entropy relative to the D896 standard recipe.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d1024_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_width_probe_seed42",
        "comparison_note": "Wider ABI passes NIB but seed42 rank falls below the LoRA comparator.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d1024_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "d1024_width_probe_seed314",
        "comparison_note": "Wider ABI beats LoRA strongly under seed314 but does not form a repeated recipe with seed42.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_nativeinit_ema9995s6400_lrdecay6400x02_cal9600_topset5_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "d960_nativeinit_ema9995_cal9600_seed42",
        "comparison_note": "Native target-interface initialization plus EMA reaches the split-separated LoRA top-1 exactly at 9,600 calibration steps while beating LoRA on top-5, JS, and entropy.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_nativeinit_ema9995s8000_lrdecay8000x02_cal12000_topset5_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "d960_nativeinit_ema9995_cal12000_seed42",
        "comparison_note": "Native target-interface initialization plus EMA beats the split-separated LoRA baseline on every reported metric with 12,000 calibration steps instead of the old 18,000-step D960 recipe.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_nativeinit_ema9995s8000_lrdecay8000x02_cal12000_topset5_bridge_wikitext_train_val_d7d416c519_results.json",
        "frontier_role": "d960_nativeinit_ema9995_cal12000_seed42off100k",
        "comparison_note": "Shifted-stream repeat of the 12,000-step native-init/EMA D960 recipe also beats the split-separated LoRA baseline on every reported metric.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d960_lora_repeat_win_seed42",
        "comparison_note": "Mid-width ABI beats the split-separated LoRA baseline on every reported metric.",
    },
    {
        "file": "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "d960_lora_repeat_win_seed314",
        "comparison_note": "Shifted-seed repeat of the D960 recipe beats the split-separated LoRA baseline on every reported metric.",
    },
    {
        "file": "exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s8000_lrdecay8000x02_cal12000_topset5_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "phi3_qwen_d1024_nativeinit_ema9995_cal12000_seed42",
        "comparison_note": "Phi-3 donor into Qwen2.5 improves sharply over the older reverse-direction certificate and beats LoRA on top-5, JS, and entropy, but misses the LoRA top-1 gate by 0.0009.",
    },
    {
        "file": "exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "phi3_qwen_d1024_nativeinit_ema9995_cal14400_seed42",
        "comparison_note": "Extending Phi-3 -> Qwen2.5 native-init/EMA calibration to 14,400 steps clears the split-separated Qwen LoRA comparator on every reported metric.",
    },
    {
        "file": "exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_f4fba487b3_results.json",
        "frontier_role": "phi3_qwen_d1024_nativeinit_ema9995_cal14400_seed42off100k",
        "comparison_note": "Shifted-stream repeat of the 14,400-step Phi-3 -> Qwen2.5 native-init/EMA recipe also beats the split-separated Qwen LoRA baseline on every reported metric.",
    },
]

PHI_ABI_FRONTIER_RESULTS = [
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_seed42_reference",
        "comparison_note": "Original no-leakage Qwen -> Phi ABI certificate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "d1024_seed314_repeat",
        "comparison_note": "Shifted-seed repeat; highest Qwen -> Phi ABI top-1 in this batch.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal7200_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_longer_calibration_seed42",
        "comparison_note": "Longer seed42 calibration improves top-5, JS, and entropy but not the LoRA top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_top1gap2_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_top1_gap_seed42",
        "comparison_note": "Teacher-top1 gap loss improves top-5/JS but still misses the LoRA top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_top1gap5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_top1_gap_strong_seed42",
        "comparison_note": "Stronger teacher-top1 gap hurts top-1 and does not close the Phi LoRA gap.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal7200_topset5_top1gap2_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_longer_top1_gap_seed42",
        "comparison_note": "Combining longer calibration with top1-gap loss hurts rank relative to the standard longer run.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_toplogit_mse_seed42",
        "comparison_note": "Small top-logit MSE probe improves top-5/JS but still misses the LoRA top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_top1ce05_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_top1_ce_seed42",
        "comparison_note": "Teacher-argmax CE improves top-5 and JS but does not close the top-1 gap.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_top1hn5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_top1_hard_negative_seed42",
        "comparison_note": "Top-1 hard-negative pressure over-constrains the rank objective and lowers top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_kd098_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_kd098_seed42",
        "comparison_note": "Higher native-teacher KD weight lowers top-1, so the fixed CE/KD mix is not the bottleneck.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_toplogitmse005_posthocbias300_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_posthoc_bias_seed42",
        "comparison_note": "Validation-only global logit bias overfits and hurts held-out rank.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_validation_selected_toplogit_seed42",
        "comparison_note": "Validation checkpoint selection improves the strict Phi seed42 top-1 frontier but still trails a fully trained Phi LoRA comparator.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_calselect600_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_validation_selected_standard_seed42",
        "comparison_note": "Checkpoint selection without top-logit MSE does not match the selected top-logit recipe.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse0025_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_validation_selected_toplogit0025_seed42",
        "comparison_note": "Lower top-logit weight under validation selection is stable but below the 0.05 frontier.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse01_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_validation_selected_toplogit01_seed42",
        "comparison_note": "Higher top-logit weight over-constrains rank and lowers held-out top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_calselect300_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_validation_selected_fine_probe_seed42",
        "comparison_note": "Finer validation checkpointing overfits the small validation probe and hurts held-out top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_traindomain_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_e2bac75327_results.json",
        "frontier_role": "d1024_train_domain_probe_seed42",
        "comparison_note": "Training the copied domain core doubles trainable parameters and still does not close the Phi LoRA top-1 gap.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_logitres32_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_0805e6200b_results.json",
        "frontier_role": "d1024_logit_residual_probe_seed42",
        "comparison_note": "Rank-32 ABI-to-vocab residual passes NIB but does not improve Phi top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_domainres256_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_tes_c1a09e6c90_results.json",
        "frontier_role": "d1024_domain_residual_probe_seed42",
        "comparison_note": "Small ABI-space domain residual passes NIB but worsens Phi top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_hiddenres256_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_tes_6acbeba56f_results.json",
        "frontier_role": "d1024_hidden_residual_probe_seed42",
        "comparison_note": "Hidden-state target residual passes NIB but does not close the Phi top-1 gap.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_seed42",
        "comparison_note": "Larger Qwen source materially improves Phi top-1 versus Qwen2.5-0.5B.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "qwen2_1p5b_d1024_seed314_align2000",
        "comparison_note": "Previous Qwen2-1.5B Phi top-1 frontier; beats the Phi LoRA seed314 comparator but misses the strongest Phi LoRA seed42 comparator by 0.0008.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_pos_ea0e1c399f_results.json",
        "frontier_role": "qwen2_1p5b_d1024_seed314_align5000_top1_frontier",
        "comparison_note": "Current Phi ABI top-1 frontier; increasing Procrustes alignment to 5000 sentences beats the strongest full-step Phi LoRA comparator on top-1 while preserving better JS and entropy.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_pos_9147e6a278_results.json",
        "frontier_role": "qwen2_1p5b_d1024_seed42_align5000_repeat_probe",
        "comparison_note": "Seed42 repeat of the 5000-alignment recipe passes NIB but does not repeat the Phi LoRA top-1 win.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed314off0_results.json",
        "frontier_role": "qwen2_1p5b_d1024_seed314_offset0_stream_probe",
        "comparison_note": "Seed314 initialization on the seed42 data/eval streams passes NIB but drops to seed42-like top-1, showing the current frontier is not explained by initialization alone.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42off100k_results.json",
        "frontier_role": "qwen2_1p5b_d1024_seed42_offset100k_stream_probe",
        "comparison_note": "Seed42 initialization on the shifted seed314 data/eval streams still misses the LoRA top-1 gate, so the single high frontier is a seed/stream interaction rather than a stable recipe.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calsoup3_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_calsoup3_seed42",
        "comparison_note": "Validation checkpoint soup over the top three seed42 checkpoints improves top-5 and JS substantially but still misses the LoRA top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup3_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup3_seed42",
        "comparison_note": "Applying top-three checkpoint soup to the 5000-alignment seed42 repeat repairs distribution/top-5 but still does not repeat the LoRA top-1 win.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_seed42",
        "comparison_note": "Equal top-two checkpoint soup is a strong seed42 stabilizer, raising top-1 to 0.9199 and top-5 to 0.9052, but still below the Phi LoRA top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_tr_3f2bbd7648_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_weighted_seed42",
        "comparison_note": "Weighted top-two checkpoint soup is the best seed42 stabilization so far, raising top-1 to 0.9228 while preserving strong top-5/JS, but still below the Phi LoRA top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_selaudit8_calselect600_cal4800_topset5_toplogitmse005_bridge_w_9b3a5db194_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_selection_audit_seed42",
        "comparison_note": "Selection-checkpoint NIB audit shows no individual validation checkpoint clears the LoRA top-1 gate; the top-two interpolation is the source of the seed42 gain.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_soupaudit_curve_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_br_73a463fa02_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_soup_curve_seed42",
        "comparison_note": "Held-out diagnostic soup curve confirms the top-two checkpoint interpolation plateaus at 0.9228 top-1 and does not hide a LoRA-clearing state.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_accum2s100k_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_2bb63bdc3b_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_accum2_seed42",
        "comparison_note": "Two-stream gradient accumulation passes NIB but lowers held-out top-1 to 0.9134, so averaging independent calibration streams over-smooths the rank-1 transfer decision.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_top1hn5_calselect600_cal4800_topset5_toplogitmse005_bridge_wik_0e9bc095bb_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_top1hn5_seed42",
        "comparison_note": "Top-1 hard-negative pressure raises validation top-1 but lowers held-out top-1 to 0.9163, showing direct rank-1 pressure reinforces validation overfit.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_ent002_calselect600_cal4800_topset5_toplogitmse005_bridge_wiki_c261359397_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_entropy002_seed42",
        "comparison_note": "Light entropy matching passes NIB but lowers held-out top-1 to 0.9150, so the current Phi blocker is not solved by simple distribution-shape regularization.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_confmargin075_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_brid_d3936d78fb_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_conf_margin_seed42",
        "comparison_note": "Teacher high-margin confidence weighting improves top-5/JS/entropy but lowers top-1 to 0.9215, so high-confidence rank emphasis does not close the Phi gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_conflowmargin075_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_b_7f1f1663d1_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_conf_low_margin_seed42",
        "comparison_note": "Teacher low-margin confidence weighting lowers top-1 to 0.9061, ruling out ambiguous-token emphasis for this Phi seed42 blocker.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_rank5_hard5_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_fb1f28e91d_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_rank5_hard5_seed42",
        "comparison_note": "Cutting rank-margin and hard-negative pressure from 10/10 to 5/5 lowers top-1 to 0.9150, so the current blocker is not caused by excessive pairwise pressure.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_topset3_calsoup2w035065_calselect600_cal4800_toplogitmse005_bridge_wikitext_tr_75edb763f3_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_topset3_seed42",
        "comparison_note": "Narrowing listwise top-set loss from top-5 to top-3 lowers top-1 to 0.9020, so the current top-5 listwise objective remains better.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_instruct_phi3_d1024_align5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wi_bffe331f2e_results.json",
        "frontier_role": "qwen2_1p5b_instruct_d1024_align5000_seed42",
        "comparison_note": "Qwen2-1.5B-Instruct donor passes NIB but lowers held-out top-1 to 0.9142; instruction tuning the source prior does not fix the Phi seed42 LoRA gap.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_abistatemse005_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bri_8a85a5653c_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_abi_state_mse005_seed42",
        "comparison_note": "Post-domain ABI-state MSE against the native target ABI oracle passes NIB but lowers top-1 to 0.9102, so direct ABI-state matching over-constrains the Phi interface.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrcos4800x02to005_cal7200_topset5_toplogitmse005_bridge_wikitext_t_50a00ed4d2_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_ema999_lrcos_seed42",
        "comparison_note": "Cosine annealing the post-4800 LR phase passes NIB but lowers top-1 to 0.9224, so the seed42 plateau is not fixed by smoother late LR decay.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_temporalavg4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_6d6771b128_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_temporalavg4800_seed42",
        "comparison_note": "Validation-independent temporal averaging over late checkpoints passes NIB but lowers top-1 to 0.9195, so broad late checkpoint averaging is too blunt for the Phi rank-1 gap.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_abipremse0005_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_1afc5333cf_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_pre_abi_mse0005_ema_seed42",
        "comparison_note": "A weak pre-domain ABI-state MSE term plus the best EMA curriculum passes NIB but lowers top-1 to 0.9220, so even light direct ABI-coordinate matching does not repair rank generalization.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_stabletop1ce015_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_br_0de1f71cc4_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_stable_top1_ce015_seed42",
        "comparison_note": "Stable-token teacher-argmax CE over native base/domain-agreeing tokens passes NIB but lowers held-out top-1 to 0.9179, so filtered argmax pressure still over-constrains Phi rank transfer.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_domaindeltamse005_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_7d19fee8f5_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_domain_delta_mse005_seed42",
        "comparison_note": "Domain-on/domain-off logit-delta MSE improves calibrated PPL but only reaches top-1 0.9195, so matching the native domain effect is not enough for Phi rank-1 stability.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_uniontopk1_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_68beec8994_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_union_topk1_seed42",
        "comparison_note": "Union top-k KD over teacher top tokens plus current student false positives raises validation top-1 but lowers held-out top-1 to 0.9175, another sign that direct rank pressure is overfitting the proxy.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_alignens3s5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikite_544a860f64_results.json",
        "frontier_role": "qwen2_1p5b_d1024_alignens3_uniform_seed42",
        "comparison_note": "Three independent 5000-sentence Procrustes rotations averaged through copied source domain cores pass NIB and improve top-5 over some seed42 baselines, but top-1 remains 0.9179; uniform alignment ensembling smooths rather than fixes the Phi rank-1 decision.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_alignens3w_s5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wiki_bd015ee788_results.json",
        "frontier_role": "qwen2_1p5b_d1024_alignens3_train_weights_seed42",
        "comparison_note": "Training a three-scalar softmax over the copied rotation ensemble leaves weights nearly uniform and reaches top-1 0.9163, ruling out simple alignment-mixture weighting as the Phi repeat fix.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_aligntrim15k5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wiki_b856823a14_results.json",
        "frontier_role": "qwen2_1p5b_d1024_aligntrim15k_seed42",
        "comparison_note": "Procrustes-trimming a 15000-pair pool raises final alignment cosine to 0.8530 but lowers held-out top-1 to 0.9159, so cleaner sentence-level geometry alone does not solve transfer rank stability.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_d31cd86977_results.json",
        "frontier_role": "qwen2_1p5b_d1024_ema999_lrdecay_seed42off200k",
        "comparison_note": "Second shifted-stream repeat of the EMA/LR-decay recipe passes NIB and beats the partial offset200k LoRA comparator, but top-1 0.9224 does not repeat the full-step Phi LoRA rank win.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_lrdecay4800x02_cal7200_calsoup2w035065_topset5_toplogitmse005_bridge_wikitext_c2ed17f338_results.json",
        "frontier_role": "qwen2_1p5b_d1024_lrdecay_calsoup_seed42off200k",
        "comparison_note": "Disabling forced EMA restore on the second shifted stream still reaches only top-1 0.9220, so the offset200k blocker is not caused by EMA selection alone.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_lrdecay4800x02_cal7200_best1_calselect600_topset5_toplogitmse005_bridge_wikite_615d498354_results.json",
        "frontier_role": "qwen2_1p5b_d1024_lrdecay_best1_seed42off200k",
        "comparison_note": "Restoring the single best validation checkpoint at step 6000 overfits validation and lowers held-out top-1 to 0.9195; the top-two soup remains the better offset200k validation-selected state.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_robustsel5x4min_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_br_a79d1eb32a_results.json",
        "frontier_role": "qwen2_1p5b_d1024_robust_select_5x4_min_seed42",
        "comparison_note": "Five-repeat, four-chunk worst-score validation checkpoint selection chooses steps 3000/4200 and lowers held-out top-1 to 0.9081, ruling out stronger validation sampling as a simple selector fix.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_linearblend025_ridge1_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse_1136590a9c_results.json",
        "frontier_role": "qwen2_1p5b_d1024_linear_blend025_seed42",
        "comparison_note": "A ridge linear/procrustes blended alignment map raises validation top-1 and alignment cosine but falls to held-out top-1 0.9146, so simple scale/shear in the ABI basis map does not repair Phi rank generalization.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2grid_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_7b6bb22a32_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_grid_seed42",
        "comparison_note": "Validation-selected top-two soup grid chooses a 0.60/0.40 blend and passes NIB, but held-out top-1 falls below the manual 0.35/0.65 soup, exposing validation-objective misalignment.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2gridr5min_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_79329f3642_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_grid_r5_min_seed42",
        "comparison_note": "Repeating validation-grid scoring five times with worst-score selection still chooses 0.60/0.40 and does not improve held-out top-1, so validation sample count alone is not the fix.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w030070_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_tr_903aafec1b_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_weight030_seed42",
        "comparison_note": "Manual 0.30/0.70 top-two soup passes and remains close to the weighted frontier, but does not beat the 0.35/0.65 seed42 result.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align10000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_t_df9bc8338d_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align10000_calsoup2_weighted_seed42",
        "comparison_note": "Increasing Procrustes alignment to 10000 sentences lowers alignment cosine and held-out rank, so raw alignment count is not the Phi stability fix.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000min80_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikite_dd5469f7de_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_min80_calsoup2_weighted_seed42",
        "comparison_note": "Filtering Procrustes pairs to longer sentences also lowers alignment cosine and held-out rank, ruling out a simple sentence-length quality filter.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_aligntrim10000to5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_ecabcc581b_results.json",
        "frontier_role": "qwen2_1p5b_d1024_aligntrim10000to5000_calsoup2_weighted_seed42",
        "comparison_note": "Geometry-aware Procrustes trimming raises final alignment cosine sharply but still misses held-out rank, so higher alignment cosine alone is not sufficient.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000zscore_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikit_0497f88804_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_zscore_calsoup2_weighted_seed42",
        "comparison_note": "Z-score normalized Procrustes fit improves validation top-1 at checkpoints but hurts held-out rank, so simple variance-normalized alignment is not the fix.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_tr_5aa3054d33_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_weighted_seed42_offset100k",
        "comparison_note": "Shifted-stream repeat of the best weighted soup passes NIB but drops to top-1 0.9163, so the Phi LoRA all-metric win is still not repeat-certified.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_calselect600_cal7200_topset5_toplogitmse005_bridge_wikitext_tr_9f55f807af_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_weighted_cal7200_seed42",
        "comparison_note": "Longer 7200-step calibration creates high validation top-1 checkpoints but lowers held-out rank, so extra calibration without stronger generalization control overfits.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_calselect600_cal7200_lrdecay4800x02_topset5_toplogitmse005_bri_8773460822_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_weighted_cal7200_lrdecay_seed42",
        "comparison_note": "Decaying the calibration LR after step 4800 reduces longer-calibration overfit and recovers top-1 0.9220/top-5 0.9067, but still trails the best 4800-step seed42 soup and the Phi LoRA top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_9af0dbe9c4_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_ema999_lrdecay_cal7200_seed42",
        "comparison_note": "EMA over the post-4800 LR-decayed calibration phase gives the best seed42 top-5 so far and raises held-out top-1 to 0.9244, but still does not clear the strongest Phi LoRA seed42 top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_e35b341e1b_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_ema999_lrdecay_cal7200_seed42_offset100k",
        "comparison_note": "Shifted-stream EMA repeat clears the strongest Phi LoRA comparator on every reported metric, but the offset-0 EMA run still misses the top-1 gate, so this is not yet a uniform recipe certificate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_finalsoupaudit_selectedema_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogit_d6083f39ef_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_final_soup_audit_seed42",
        "comparison_note": "Final selected/EMA blend audit shows pure EMA remains best at 0.9244 top-1; blending selected soup with EMA does not clear the LoRA gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_711b332ded_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_ema999_lrdecay_cal7200_seed314",
        "comparison_note": "Seed314 repeat of the forced-EMA curriculum passes NIB but drops below the non-EMA seed314 frontier, showing forced EMA is not uniformly stabilizing.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_finalselect_val4r3_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_b_d7aed9f544_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_finalselect_val4r3_seed314",
        "comparison_note": "Validation-only final selector chose EMA over selected soup/best/final on seed314, but held-out top-1 stayed at 0.9167; selector validation is still misaligned with transfer generalization.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_finalselectaudit_val4r3_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse_bbd9590a6c_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_finalselect_audit_seed314",
        "comparison_note": "Final-candidate NIB audit shows selected soup, EMA, best checkpoint, and final checkpoint all miss the Phi LoRA top-1 gate on seed314, so the miss is not only final-state selection.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s5400_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_b8d6a94277_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_ema999_s5400_lrdecay_cal7200_seed42",
        "comparison_note": "Starting EMA later at step 5400 lowers offset-0 held-out top-1 versus the 4800-start EMA, so the useful stabilization window includes the early LR-decayed phase.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4200_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_a54c880214_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_ema999_s4200_lrdecay_cal7200_seed42",
        "comparison_note": "Starting EMA before LR decay at step 4200 lowers offset-0 top-1 to 0.9224, so including the pre-decay plateau does not improve the EMA trajectory.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema9995s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_tra_33028a26d6_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_ema9995_s4800_lrdecay_cal7200_seed42",
        "comparison_note": "Slower EMA decay slightly improves top-5/JS but leaves offset-0 top-1 at 0.9244, so EMA smoothing alone does not close the LoRA top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x01_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_2de76e65c0_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_ema999_lrdecay01_cal7200_seed42",
        "comparison_note": "A stronger LR decay to 0.1 lowers offset-0 top-1 versus the 0.2 decay, so the current late-phase LR factor is not too high.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x02_cal7200_posthocbias200lr001l2p01_topset5_toplogitms_c659a0c4d4_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_ema999_posthoc_bias_seed42",
        "comparison_note": "A conservative validation-only global logit bias after the strongest seed42 EMA recipe collapses top-1 to 0.8927, so target-vocabulary prior correction is too blunt for Phi rank transfer.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_domainres64_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_w_14cb5708c5_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_domainres64_ema_seed42",
        "comparison_note": "A rank-64 ABI-space residual around the copied core improves top-5/JS relative to many probes but only reaches top-1 0.9228, below the EMA frontier and the Phi LoRA gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_freezeallalpha_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridg_45e375849b_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_freezeallalpha_ema_seed42",
        "comparison_note": "Fully freezing the copied domain core and source domain alpha reduces trainable D-phase parameters and reproduces the EMA plateau at top-1 0.9244, but does not clear the Phi LoRA gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_freezeallalpha_selectsoup_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitm_27d3d571c9_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_freezeallalpha_selected_soup_seed42",
        "comparison_note": "Letting the validation-selected EMA/final checkpoint soup stand under the fully frozen copied core reaches top-1 0.9240, so the forced-EMA restore is not hiding the last Phi rank points.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_nativeinit_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wi_9f260e6e3b_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_nativeinit_ema_seed42",
        "comparison_note": "Initializing the Phase D target ABI interface from the native Phase C target oracle is the strongest seed42 move so far: top-1 rises to 0.9264 with top-5 0.9232 and JS 0.0047, but it still misses the strongest Phi LoRA top-1 gate by 0.0008.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_nativeinit_selectsoup_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse00_5f09348af8_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_nativeinit_selected_soup_seed42",
        "comparison_note": "Letting the native-init validation-selected EMA/6000-step soup stand improves top-5/JS but lowers top-1 to 0.9240, so pure EMA remains the best native-init held-out rank state.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_nativeinit_cal600_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_nativeinit_cal600_seed42",
        "comparison_note": "A short 600-step native-init checkpoint passes but drops to top-1 0.9134, ruling out an early-calibration held-out rank peak despite strong early validation rank.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_nativeinit_ema9995s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_w_3d9c1c8b0f_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_nativeinit_ema9995_seed42",
        "comparison_note": "Native target-interface initialization plus slower EMA decay clears the strongest Phi LoRA comparator on every reported metric for seed42: top-1 0.9285, top-5 0.9229, JS 0.0047, entropy diff 0.1466.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_nativeinit_ema9995s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_w_cd56755dbb_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_nativeinit_ema9995_seed42off100k",
        "comparison_note": "Shifted-stream repeat of the native-init/EMA9995 recipe strongly repeats the Phi LoRA all-metric win: top-1 0.9378, top-5 0.9266, JS 0.0046, entropy diff 0.1205.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal6000_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_cal6000_selected_seed42",
        "comparison_note": "The single 6000-step checkpoint reaches validation top-1 0.9370 but held-out top-1 only 0.9110, confirming severe validation-rank overfit.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2_calselect600x4_cal4800_topset5_toplogitmse005_bridge_wikitext_train_v_65c035b778_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_val4_seed42",
        "comparison_note": "Using four validation chunks for selection lowers held-out top-1, so validation-estimate size alone is not the missing stability mechanism.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2_top1gap2_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_a407bebc53_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_calsoup2_top1gap_seed42",
        "comparison_note": "Adding a light top-1 gap objective to the best checkpoint-soup recipe reduces top-1 relative to soup alone.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_dom1000_align5000_calsoup2_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_t_5cf86fa792_results.json",
        "frontier_role": "qwen2_1p5b_d1024_dom1000_align5000_calsoup2_seed42",
        "comparison_note": "Longer source/native domain-core training improves oracle PPL but does not improve held-out rank-1 stability.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_final4800_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_align5000_final_checkpoint_seed42",
        "comparison_note": "Using the final 4800-step checkpoint directly trails the top-two checkpoint soup, confirming soup is the better seed42 stabilizer.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect300_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "qwen2_1p5b_d1024_fine_select_seed314",
        "comparison_note": "Finer validation checkpointing selects the same best state and reproduces the top-1 frontier.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1280_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1280_width_probe_seed42",
        "comparison_note": "Widening the ABI to 1280 dimensions increases trainable parameters but does not improve seed42 top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal7200_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_longer_calibration_seed42",
        "comparison_note": "Longer seed42 calibration improves top-5 to 0.9000 but still misses the Phi LoRA top-1 gate.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "qwen2_1p5b_d1024_no_toplogit_seed314",
        "comparison_note": "Removing centered top-logit MSE slightly lowers the Qwen2-1.5B Phi top-1 frontier.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_no_toplogit_seed42",
        "comparison_note": "Removing centered top-logit MSE in seed42 lowers top-1 relative to the top-logit recipe.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_top1ce025_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_pos_cbf147b87f_results.json",
        "frontier_role": "qwen2_1p5b_d1024_top1ce_seed314",
        "comparison_note": "Adding light teacher-argmax CE lowers top-1 relative to the clean Qwen2-1.5B recipe.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_top1ce025_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_top1ce_seed42",
        "comparison_note": "Adding light teacher-argmax CE to seed42 worsens top-1, confirming that direct argmax pressure is not the missing stability mechanism.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse0075_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json",
        "frontier_role": "qwen2_1p5b_d1024_toplogit0075_seed314",
        "comparison_note": "Increasing centered top-logit MSE to 0.075 lowers top-1 versus the 0.05 frontier.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_logitres32_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_logit_residual_seed42",
        "comparison_note": "A rank-32 ABI-to-vocab residual adds capacity but does not repair seed42 top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse005_posthocbias300_bridge_wikitext_train_val_test_seed42_results.json",
        "frontier_role": "qwen2_1p5b_d1024_posthoc_bias_seed42",
        "comparison_note": "Validation-only global logit bias improves top-5 but does not improve seed42 top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2_7b_phi3_d1024_release_source_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_8e31294bc8_results.json",
        "frontier_role": "qwen2_7b_release_source_probe_seed42",
        "comparison_note": "Sequential source-release mode makes a 7B donor feasible on 16GB, but this donor underperforms Qwen2-1.5B for Phi top-1.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset1_rankpos1_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1024_top1_only_ablation",
        "comparison_note": "Top-1-only rank/listwise ablation hurts both top-1 and top-5.",
    },
    {
        "file": "exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1536_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json",
        "frontier_role": "d1536_width_ablation",
        "comparison_note": "More width hurts this Qwen -> Phi direction.",
    },
]


def load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def metric_block(result: dict) -> dict:
    nib = result["nib_l2"]
    return {
        "js": nib["mean_js"],
        "top1": nib["mean_top1_agree"],
        "top5": nib["mean_top5_overlap"],
        "entropy_diff": nib["mean_entropy_diff"],
        "pass": bool(nib["pass"] and result["overall_pass"]),
    }


def count_safetensors(path: Path) -> int:
    total = 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            try:
                shape = f.get_slice(key).get_shape()
            except AttributeError:
                shape = f.get_tensor(key).shape
            total += math.prod(shape)
    return int(total)


def count_torch_bin(path: Path) -> int:
    state = torch.load(path, map_location="meta", weights_only=True, mmap=True)
    return int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))


def count_model_params(model_id: str) -> int | None:
    model_path = Path(experiment_data.hf_local_path(model_id))
    if not model_path.exists():
        return None

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_path / index_name
        if index_path.exists():
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            total = 0
            for shard in sorted(set(weight_map.values())):
                shard_path = model_path / shard
                total += (
                    count_safetensors(shard_path)
                    if shard_path.suffix == ".safetensors"
                    else count_torch_bin(shard_path)
                )
            return total

    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if safetensors_files:
        return sum(count_safetensors(path) for path in safetensors_files)

    bin_file = model_path / "pytorch_model.bin"
    if bin_file.exists():
        return count_torch_bin(bin_file)

    return None


def savings_rows() -> list[dict]:
    rows = []
    param_cache: dict[str, int | None] = {}
    for name in SAVINGS_RESULTS:
        result = load_json(name)
        target = result["target_model"]
        if target not in param_cache:
            param_cache[target] = count_model_params(target)
        target_params = param_cache[target]
        trainable = result["calibration_trainable_params"]
        fraction = trainable / target_params if target_params else None
        metrics = metric_block(result)
        rows.append(
            {
                "file": name,
                "source": result["source_model"],
                "target": target,
                "domain_corpus": result.get("domain_corpus", "python"),
                "calibration_steps": result["calibration_steps"],
                "elapsed_min": result["elapsed_min"],
                "calibration_trainable_params": trainable,
                "target_params_counted": target_params,
                "trainable_fraction_of_target": fraction,
                "frozen_fraction_of_target": (1 - fraction) if fraction is not None else None,
                **metrics,
                "ppl_native_target": result.get("ppl_native_target"),
                "ppl_calibrated_target": result.get("ppl_calibrated_target"),
            }
        )
    return rows


def hard_frontier_rows() -> list[dict]:
    rows = []
    for name in HARD_FRONTIER_RESULTS:
        result = load_json(name)
        metrics = metric_block(result)
        rows.append(
            {
                "file": name,
                "d_abi": result["d_abi"],
                "seed": result["seed"],
                "seed_offset": result["seed_offset"],
                "calibration_steps": result["calibration_steps"],
                "elapsed_min": result["elapsed_min"],
                "top_logit_mse_weight": result.get("top_logit_mse_weight", 0.0),
                **metrics,
            }
        )
    return rows


def domain_breadth() -> dict:
    atlas = load_json("multi_domain_atlas_results.json")
    diagonal = {}
    for domain, values in atlas["diagonal_l2"].items():
        diagonal[domain] = {
            "js": values.get("mean_js", values.get("js")),
            "top1": values["top1"],
            "top5": values["top5"],
            "entropy_diff": values.get("entropy_diff", values.get("entropy", 0.0)),
            "pass": bool(values["pass"]),
            "r_squared": values["r_squared"],
            "kd_steps": values["kd_steps"],
        }

    generic_wikitext = []
    for name in WIKITEXT_GENERIC_RESULTS:
        result = load_json(name)
        metrics = metric_block(result)
        generic_wikitext.append(
            {
                "file": name,
                "source": result["source_model"],
                "target": result["target_model"],
                "domain_corpus": result.get("domain_corpus"),
                "calibration_steps": result["calibration_steps"],
                "entropy_weight": result.get("entropy_weight", 0.0),
                "posthoc_logit_scale": result.get(
                    "posthoc_logit_scale",
                    {
                        "mode": "none",
                        "applied": False,
                        "scale": 1.0,
                    },
                ),
                **metrics,
            }
        )

    return {
        "atlas_diagonal": diagonal,
        "atlas_domains_passing": sum(1 for row in diagonal.values() if row["pass"]),
        "atlas_domains_total": len(diagonal),
        "generic_cross_model_wikitext": generic_wikitext,
    }


def baseline_rows() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob(BASELINE_RESULT_GLOB)):
        result = load_json(path.name)
        metrics = metric_block(result)
        rows.append(
            {
                "file": path.name,
                "baseline_type": result.get("baseline_type", "unknown"),
                "target": result["target_model"],
                "domain_corpus": result.get("domain_corpus"),
                "wikitext_domain_split": result.get("wikitext_domain_split"),
                "wikitext_posthoc_split": result.get("wikitext_posthoc_split"),
                "wikitext_eval_split": result.get("wikitext_eval_split"),
                "withheld_nib_eval": result.get("withheld_nib_eval", False),
                "domain_train_tokens_target": result.get("domain_train_tokens_target"),
                "posthoc_tokens_target": result.get("posthoc_tokens_target"),
                "nib_eval_tokens_target": result.get("nib_eval_tokens_target"),
                "calibration_steps": result["calibration_steps"],
                "completed_calibration_steps": result.get(
                    "completed_calibration_steps",
                    result["calibration_steps"],
                ),
                "requested_calibration_steps": result.get(
                    "requested_calibration_steps",
                    result["calibration_steps"],
                ),
                "max_train_seconds": result.get("max_train_seconds"),
                "stopped_early": result.get("stopped_early", False),
                "stop_reason": result.get("stop_reason"),
                "elapsed_min": result["elapsed_min"],
                "calibration_trainable_params": result["calibration_trainable_params"],
                "target_params_counted": result.get("target_param_count"),
                "trainable_fraction_of_target": result.get(
                    "trainable_fraction_of_target"
                ),
                "lora_rank": result.get("lora_rank"),
                "lora_targets": result.get("lora_targets"),
                "lora_injected_modules": result.get("lora_injected_modules"),
                "teacher_d_abi": result.get("teacher_d_abi"),
                "posthoc_logit_scale": result.get(
                    "posthoc_logit_scale",
                    {
                        "mode": "none",
                        "applied": False,
                        "scale": 1.0,
                    },
                ),
                **metrics,
                "ppl_native_target": result.get("ppl_native_target"),
                "ppl_lora_target": result.get("ppl_lora_target"),
            }
        )
    return rows


def withheld_rows() -> list[dict]:
    rows = []
    for name in WITHHELD_WIKITEXT_RESULTS:
        result = load_json(name)
        metrics = metric_block(result)
        rows.append(
            {
                "file": name,
                "source": result["source_model"],
                "target": result["target_model"],
                "domain_corpus": result.get("domain_corpus"),
                "wikitext_domain_split": result.get("wikitext_domain_split"),
                "wikitext_align_split": result.get("wikitext_align_split"),
                "wikitext_posthoc_split": result.get("wikitext_posthoc_split"),
                "wikitext_eval_split": result.get("wikitext_eval_split"),
                "withheld_nib_eval": result.get("withheld_nib_eval", False),
                "domain_train_tokens_target": result.get("domain_train_tokens_target"),
                "posthoc_tokens_target": result.get("posthoc_tokens_target"),
                "nib_eval_tokens_target": result.get("nib_eval_tokens_target"),
                "calibration_steps": result["calibration_steps"],
                "elapsed_min": result["elapsed_min"],
                "calibration_trainable_params": result["calibration_trainable_params"],
                "posthoc_logit_scale": result.get(
                    "posthoc_logit_scale",
                    {
                        "mode": "none",
                        "applied": False,
                        "scale": 1.0,
                    },
                ),
                **metrics,
                "ppl_native_target": result.get("ppl_native_target"),
                "ppl_calibrated_target": result.get("ppl_calibrated_target"),
            }
        )
    return rows


def abi_lora_frontier_rows() -> list[dict]:
    rows = []
    for spec in ABI_LORA_FRONTIER_RESULTS:
        result = load_json(spec["file"])
        metrics = metric_block(result)
        rows.append(
            {
                "file": spec["file"],
                "frontier_role": spec["frontier_role"],
                "comparison_note": spec["comparison_note"],
                "source": result["source_model"],
                "target": result["target_model"],
                "domain_corpus": result.get("domain_corpus"),
                "d_abi": result["d_abi"],
                "seed": result["seed"],
                "seed_offset": result["seed_offset"],
                "wikitext_domain_split": result.get("wikitext_domain_split"),
                "wikitext_align_split": result.get("wikitext_align_split"),
                "wikitext_posthoc_split": result.get("wikitext_posthoc_split"),
                "wikitext_eval_split": result.get("wikitext_eval_split"),
                "withheld_nib_eval": result.get("withheld_nib_eval", False),
                "calibration_steps": result["calibration_steps"],
                "calibration_mode": result.get("calibration_mode"),
                "calibration_init": result.get("calibration_init", "xavier"),
                "calibration_selection": result.get("calibration_selection"),
                "calibration_ema": result.get("calibration_ema", {}),
                "cal_select_mode": result.get("calibration_selection", {}).get(
                    "mode", "none"
                ),
                "cal_select_avg_top_n": result.get(
                    "calibration_selection", {}
                ).get("avg_top_n", 1),
                "cal_ema_decay": result.get("calibration_ema", {}).get(
                    "decay", 0.0
                ),
                "cal_ema_start_step": result.get("calibration_ema", {}).get(
                    "start_step", 1
                ),
                "cal_ema_restore": result.get("calibration_ema", {}).get(
                    "restore", False
                ),
                "cal_lr": result.get("cal_lr", 1.0e-4),
                "cal_lr_decay_step": result.get("cal_lr_decay_step", 0),
                "cal_lr_decay_factor": result.get("cal_lr_decay_factor", 1.0),
                "n_align_sentences": result.get("n_align_sentences"),
                "elapsed_min": result["elapsed_min"],
                "calibration_trainable_params": result["calibration_trainable_params"],
                "topk": result.get("topk"),
                "topk_kd_weight": result.get("topk_kd_weight"),
                "rank_margin_weight": result.get("rank_margin_weight"),
                "topset_k": result.get("topset_k"),
                "topset_weight": result.get("topset_weight"),
                "posthoc_logit_scale": result.get(
                    "posthoc_logit_scale",
                    {
                        "mode": "none",
                        "applied": False,
                        "scale": 1.0,
                    },
                ),
                **metrics,
                "ppl_native_target": result.get("ppl_native_target"),
                "ppl_calibrated_target": result.get("ppl_calibrated_target"),
            }
        )
    return rows


def phi_abi_frontier_rows() -> list[dict]:
    rows = []
    for spec in PHI_ABI_FRONTIER_RESULTS:
        result = load_json(spec["file"])
        metrics = metric_block(result)
        rows.append(
            {
                "file": spec["file"],
                "frontier_role": spec["frontier_role"],
                "comparison_note": spec["comparison_note"],
                "source": result["source_model"],
                "target": result["target_model"],
                "domain_corpus": result.get("domain_corpus"),
                "d_abi": result["d_abi"],
                "seed": result["seed"],
                "seed_offset": result.get("seed_offset", 0),
                "calibration_steps": result["calibration_steps"],
                "elapsed_min": result["elapsed_min"],
                "calibration_trainable_params": result["calibration_trainable_params"],
                "n_align_sentences": result.get("n_align_sentences"),
                "calibration_mode": result.get("calibration_mode"),
                "calibration_init": result.get("calibration_init", "xavier"),
                "calibration_selection": result.get("calibration_selection"),
                "calibration_ema": result.get("calibration_ema", {}),
                "calibration_final_selection": result.get(
                    "calibration_final_selection"
                ),
                "calibration_final_audit": result.get("calibration_final_audit"),
                "cal_select_mode": result.get("calibration_selection", {}).get(
                    "mode", "none"
                ),
                "cal_select_avg_top_n": result.get(
                    "calibration_selection", {}
                ).get("avg_top_n", 1),
                "cal_ema_decay": result.get("calibration_ema", {}).get("decay", 0.0),
                "cal_ema_start_step": result.get("calibration_ema", {}).get(
                    "start_step", 1
                ),
                "cal_ema_restore": result.get("calibration_ema", {}).get(
                    "restore", False
                ),
                "kd_weight": result.get("kd_weight"),
                "top1_gap_weight": result.get("top1_gap_weight", 0.0),
                "top1_ce_weight": result.get("top1_ce_weight", 0.0),
                "top1_hard_neg_weight": result.get("top1_hard_neg_weight", 0.0),
                "top_logit_mse_weight": result.get("top_logit_mse_weight", 0.0),
                "topset_k": result.get("topset_k"),
                "posthoc_logit_bias": result.get("posthoc_logit_bias"),
                "domain_residual_rank": result.get("domain_residual_rank", 0),
                "target_residual": result.get("target_residual", "none"),
                "target_residual_rank": result.get("target_residual_rank"),
                "release_source_before_target": result.get(
                    "release_source_before_target", False
                ),
                "wikitext_domain_split": result.get("wikitext_domain_split"),
                "wikitext_posthoc_split": result.get("wikitext_posthoc_split"),
                "wikitext_eval_split": result.get("wikitext_eval_split"),
                "withheld_nib_eval": result.get("withheld_nib_eval", False),
                **metrics,
                "ppl_native_target": result.get("ppl_native_target"),
                "ppl_calibrated_target": result.get("ppl_calibrated_target"),
            }
        )
    return rows


def north_star_gate_rows() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob("exp_generic_causal_nib_v2*_results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cert = result.get("compatibility_certificate")
        if not isinstance(cert, dict):
            continue
        gates = cert.get("gates", {})
        source_preservation = result.get("source_preservation") or cert.get(
            "source_preservation"
        )
        source_measured = bool(gates.get("source_preservation_measured"))
        oracle_light = bool(gates.get("oracle_light_mode"))
        target_reference_pass = bool(gates.get("target_reference_nib_pass"))
        calibration_selection = result.get("calibration_selection") or {}
        calibration_ema = result.get("calibration_ema") or {}
        target_native_oracle_required = bool(
            cert.get(
                "target_native_oracle_required",
                result.get("target_native_oracle_required", True),
            )
        )
        rows.append(
            {
                "file": path.name,
                "source": result.get("source_model"),
                "target": result.get("target_model"),
                "oracle_mode": cert.get("oracle_mode", result.get("oracle_mode")),
                "claim_scope": cert.get("claim_scope"),
                "d_abi": result.get("d_abi"),
                "calibration_steps": result.get("calibration_steps"),
                "calibration_mode": result.get("calibration_mode"),
                "calibration_init": result.get("calibration_init", "xavier"),
                "cal_select_mode": calibration_selection.get(
                    "mode",
                    result.get("cal_select_mode", "none"),
                ),
                "cal_select_avg_top_n": calibration_selection.get(
                    "avg_top_n",
                    result.get("cal_select_avg_top_n", 1),
                ),
                "cal_ema_decay": calibration_ema.get(
                    "decay",
                    result.get("cal_ema_decay", 0.0),
                ),
                "cal_ema_start_step": calibration_ema.get(
                    "start_step",
                    result.get("cal_ema_start_step", 1),
                ),
                "cal_ema_restore": calibration_ema.get(
                    "restore",
                    result.get("cal_ema_restore", False),
                ),
                "cal_lr": result.get("cal_lr", 1.0e-4),
                "cal_lr_decay_step": result.get("cal_lr_decay_step", 0),
                "cal_lr_decay_factor": result.get("cal_lr_decay_factor", 1.0),
                "topset_k": result.get("topset_k"),
                "topset_weight": result.get("topset_weight"),
                "seed": result.get("seed"),
                "seed_offset": result.get("seed_offset", 0),
                "target_native_oracle_required": target_native_oracle_required,
                "oracle_light_mode": oracle_light,
                "target_reference_nib_pass": target_reference_pass,
                "target_native_nib_pass": gates.get("target_native_nib_pass"),
                "source_preservation_measured": source_measured,
                "source_preservation_top1": (
                    source_preservation or {}
                ).get("top1_surface_agree"),
                "source_preservation_top1_in_topk": (
                    source_preservation or {}
                ).get("source_top1_in_target_topk"),
                "source_preservation_topk_overlap": (
                    source_preservation or {}
                ).get("mean_topk_surface_overlap"),
                "nib_pass": bool(result.get("overall_pass", False)),
                "top5": result.get("nib_l2", {}).get("mean_top5_overlap"),
                "top1": result.get("nib_l2", {}).get("mean_top1_agree"),
                "js": result.get("nib_l2", {}).get("mean_js"),
                "entropy_diff": result.get("nib_l2", {}).get(
                    "mean_entropy_diff"
                ),
            }
        )
    return rows


def north_star_joint_pass(row: dict) -> bool:
    return bool(
        row["oracle_light_mode"]
        and row["source_preservation_measured"]
        and row["target_reference_nib_pass"]
        and row["nib_pass"]
    )


def north_star_recipe_key(row: dict) -> tuple:
    return (
        row["source"],
        row["target"],
        row["oracle_mode"],
        row["d_abi"],
        row["calibration_steps"],
        row["calibration_mode"],
        row["calibration_init"],
        row["cal_select_mode"],
        row["cal_select_avg_top_n"],
        row["cal_ema_decay"],
        row["cal_ema_start_step"],
        row["cal_ema_restore"],
        row["cal_lr"],
        row["cal_lr_decay_step"],
        row["cal_lr_decay_factor"],
        row["topset_k"],
        row["topset_weight"],
    )


def repeated_north_star_recipe_count(rows: list[dict]) -> int:
    variants_by_recipe = {}
    for row in rows:
        if not north_star_joint_pass(row):
            continue
        variants_by_recipe.setdefault(north_star_recipe_key(row), set()).add(
            (row["seed"], row["seed_offset"])
        )
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


def repeated_north_star_pair_count(rows: list[dict]) -> int:
    variants_by_pair_and_recipe = {}
    for row in rows:
        if not north_star_joint_pass(row):
            continue
        key = (row["source"], row["target"], north_star_recipe_key(row))
        variants_by_pair_and_recipe.setdefault(key, set()).add(
            (row["seed"], row["seed_offset"])
        )
    repeat_pairs = {
        (source, target)
        for (source, target, _recipe), variants in variants_by_pair_and_recipe.items()
        if len(variants) >= 2
    }
    return len(repeat_pairs)


FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS = {
    "top5": 0.95,
    "top1": 0.965,
    "js": 0.005,
    "entropy_diff": 0.05,
}
FLAGSHIP_COMPLETION_PRESERVATION_THRESHOLDS = {
    "source_top1_completion_preferred": 0.50,
}
FLAGSHIP_SOURCE_SURFACE_REPAIR_THRESHOLDS = {
    "top1_surface_agree": 0.40,
    "source_top1_in_target_topk": 0.55,
    "source_top1_completion_preferred": 0.65,
}
SELECTIVE_TRANSFER_REQUIRED_REPEAT_RECIPES = 2


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def strict_nib_metrics_pass(metrics: dict | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    top5 = metrics.get("mean_top5_overlap")
    top1 = metrics.get("mean_top1_agree")
    js = metrics.get("mean_js")
    entropy = metrics.get("mean_entropy_diff")
    return bool(
        metrics.get("pass")
        and finite_number(top5)
        and finite_number(top1)
        and finite_number(js)
        and finite_number(entropy)
        and top5 >= FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS["top5"]
        and top1 >= FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS["top1"]
        and js <= FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS["js"]
        and entropy <= FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS["entropy_diff"]
    )


def selective_transfer_rows() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob("exp_generic_causal_nib_v2*_results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cert = result.get("compatibility_certificate") or {}
        selective = result.get("selective_transfer") or cert.get(
            "selective_transfer"
        )
        if not isinstance(selective, dict) or not selective.get("enabled"):
            continue
        target_nib = result.get("nib_l2") or {}
        off_domain_nib = selective.get("off_domain_nib_l2") or {}
        target_strict = strict_nib_metrics_pass(target_nib)
        off_domain_strict = strict_nib_metrics_pass(off_domain_nib)
        row = {
            "file": path.name,
            "source": result.get("source_model"),
            "target": result.get("target_model"),
            "domain_corpus": result.get("domain_corpus"),
            "off_domain_corpus": selective.get("off_domain_corpus"),
            "off_domain_reference": selective.get("off_domain_reference"),
            "oracle_mode": result.get("oracle_mode"),
            "d_abi": result.get("d_abi"),
            "calibration_steps": result.get("calibration_steps"),
            "calibration_mode": result.get("calibration_mode"),
            "calibration_init": result.get("calibration_init", "xavier"),
            "seed": result.get("seed"),
            "seed_offset": result.get("seed_offset", 0),
            "source_preservation_measured": bool(
                (result.get("source_preservation") or {}).get("measured")
            ),
            "target_domain_nib_pass": bool(selective.get("target_domain_nib_pass")),
            "off_domain_no_leakage_pass": bool(
                selective.get("off_domain_no_leakage_pass")
            ),
            "selective_transfer_pass": bool(
                selective.get("selective_transfer_pass")
            ),
            "target_domain_strict_pass": target_strict,
            "off_domain_strict_pass": off_domain_strict,
            "strict_selective_transfer_pass": bool(
                target_strict and off_domain_strict
            ),
            "target_top5": target_nib.get("mean_top5_overlap"),
            "target_top1": target_nib.get("mean_top1_agree"),
            "target_js": target_nib.get("mean_js"),
            "target_entropy_diff": target_nib.get("mean_entropy_diff"),
            "off_domain_top5": off_domain_nib.get("mean_top5_overlap"),
            "off_domain_top1": off_domain_nib.get("mean_top1_agree"),
            "off_domain_js": off_domain_nib.get("mean_js"),
            "off_domain_entropy_diff": off_domain_nib.get("mean_entropy_diff"),
            "ppl_off_domain_relative_overhead": selective.get(
                "ppl_off_domain_relative_overhead"
            ),
        }
        rows.append(row)
    return rows


def selective_transfer_recipe_key(row: dict) -> tuple:
    return (
        row["source"],
        row["target"],
        row["domain_corpus"],
        row["off_domain_corpus"],
        row["off_domain_reference"],
        row["oracle_mode"],
        row["d_abi"],
        row["calibration_steps"],
        row["calibration_mode"],
        row["calibration_init"],
    )


def repeated_selective_transfer_recipe_count(rows: list[dict]) -> int:
    variants_by_recipe = {}
    for row in rows:
        if not row["strict_selective_transfer_pass"]:
            continue
        variants_by_recipe.setdefault(selective_transfer_recipe_key(row), set()).add(
            (row["seed"], row["seed_offset"])
        )
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


GPT2_BASE_BYPASS_BREADTH_TARGETS = {
    "microsoft/phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-0.5B",
}
HARD_RECIPIENT_TARGETS = {
    "deepseek-ai/deepseek-coder-1.3b-base",
}


def gpt_style_result_row(path: Path, result: dict) -> dict:
    cert = result.get("compatibility_certificate")
    if not isinstance(cert, dict):
        cert = {}
    gates = cert.get("gates", {})
    source_preservation = result.get("source_preservation") or cert.get(
        "source_preservation"
    )
    source_measured = bool(
        gates.get(
            "source_preservation_measured",
            source_preservation is not None,
        )
    )
    target_native_oracle_required = bool(
        cert.get(
            "target_native_oracle_required",
            result.get("target_native_oracle_required", True),
        )
    )
    oracle_mode = cert.get(
        "oracle_mode",
        result.get(
            "oracle_mode",
            "full_native_target_oracle",
        ),
    )
    metrics = metric_block(result)
    row = {
        "file": path.name,
        "source": result.get("source_model"),
        "target": result.get("target_model"),
        "oracle_mode": oracle_mode,
        "claim_scope": cert.get("claim_scope", "legacy_full_oracle_probe"),
        "target_native_oracle_required": target_native_oracle_required,
        "target_reference_bypass_abi": result.get(
            "target_reference_bypass_abi", False
        ),
        "target_reference_forward_mode": result.get(
            "target_reference_forward_mode"
        ),
        "d_abi": result.get("d_abi"),
        "domain_corpus": result.get("domain_corpus"),
        "domain_steps": result.get("domain_steps"),
        "calibration_steps": result.get("calibration_steps"),
        "cal_lr": result.get("cal_lr", 1.0e-4),
        "cal_lr_decay_step": result.get("cal_lr_decay_step", 0),
        "cal_lr_decay_factor": result.get("cal_lr_decay_factor", 1.0),
        "calibration_mode": result.get("calibration_mode"),
        "calibration_init": result.get("calibration_init", "xavier"),
        "native_domain_seed_base": result.get("native_domain_seed_base"),
        "topk": result.get("topk"),
        "topk_kd_weight": result.get("topk_kd_weight"),
        "rank_margin_weight": result.get("rank_margin_weight", 0.0),
        "rank_top_pos": result.get("rank_top_pos"),
        "rank_neg_k": result.get("rank_neg_k"),
        "rank_margin": result.get("rank_margin"),
        "hard_neg_weight": result.get("hard_neg_weight", 0.0),
        "hard_neg_k": result.get("hard_neg_k"),
        "topset_k": result.get("topset_k"),
        "topset_weight": result.get("topset_weight"),
        "top_logit_mse_weight": result.get("top_logit_mse_weight", 0.0),
        "top_logit_mse_k": result.get("top_logit_mse_k"),
        "entropy_weight": result.get("entropy_weight", 0.0),
        "target_residual": result.get("target_residual", "none"),
        "target_residual_rank": result.get("target_residual_rank"),
        "target_interface_cache": result.get("target_interface_cache", {}),
        "source_completion_loss": result.get("source_completion_loss", {}),
        "calibration_ema": result.get("calibration_ema", {}),
        "posthoc_logit_scale": result.get("posthoc_logit_scale", {}),
        "seed": result.get("seed"),
        "seed_offset": result.get("seed_offset", 0),
        "source_preservation_measured": source_measured,
        "source_preservation_top1": (
            source_preservation or {}
        ).get("top1_surface_agree"),
        "source_preservation_top1_in_topk": (
            source_preservation or {}
        ).get("source_top1_in_target_topk"),
        "source_preservation_topk_overlap": (
            source_preservation or {}
        ).get("mean_topk_surface_overlap"),
        "source_completion_preferred": (
            source_preservation or {}
        ).get("source_top1_completion_preferred"),
        "source_completion_mean_rank": (
            source_preservation or {}
        ).get("mean_source_top1_completion_rank"),
        "source_completion_margin_vs_best": (
            source_preservation or {}
        ).get("mean_source_top1_completion_margin_vs_best"),
        "source_completion_prompt_count": (
            source_preservation or {}
        ).get("completion_prompt_count_measured"),
        "target_reference_nib_pass": bool(
            gates.get("target_reference_nib_pass", metrics["pass"])
        ),
        "nib_pass": metrics["pass"],
        "top5": metrics["top5"],
        "top1": metrics["top1"],
        "js": metrics["js"],
        "entropy_diff": metrics["entropy_diff"],
    }
    row["distribution_near_lossless_pass"] = flagship_distribution_pass(row)
    row["cross_tokenizer_completion_pass"] = flagship_completion_pass(row)
    row["distribution_completion_pass"] = bool(
        row["distribution_near_lossless_pass"]
        and row["cross_tokenizer_completion_pass"]
    )
    row["source_surface_repair_pass"] = flagship_source_surface_repair_pass(row)
    row["ordinary_source_surface_pass"] = flagship_ordinary_source_surface_pass(row)
    return row


def flagship_gpt_style_rows() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob("exp_generic_causal_nib_v2*gpt2med*phi3*results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("source_model") != "gpt2-medium"
            or result.get("target_model") != "microsoft/phi-3-mini-4k-instruct"
        ):
            continue
        rows.append(gpt_style_result_row(path, result))
    return rows


def gpt2_base_bypass_target_breadth_rows() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob("exp_generic_causal_nib_v2*gpt2med*results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("source_model") != "gpt2-medium"
            or result.get("target_model") not in GPT2_BASE_BYPASS_BREADTH_TARGETS
        ):
            continue
        row = gpt_style_result_row(path, result)
        if (
            row["oracle_mode"] == "base_target_reference"
            and row["target_reference_bypass_abi"] is True
            and row["target_reference_forward_mode"] == "base"
        ):
            rows.append(row)
    return rows


def base_bypass_source_surface_breadth_rows() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob("exp_generic_causal_nib_v2*basebypass*results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        row = gpt_style_result_row(path, result)
        if (
            row["oracle_mode"] == "base_target_reference"
            and row["target_reference_bypass_abi"] is True
            and row["target_reference_forward_mode"] == "base"
            and row["source_preservation_measured"]
        ):
            rows.append(row)
    return rows


def flagship_distribution_pass(row: dict) -> bool:
    return bool(
        row["nib_pass"]
        and row["source_preservation_measured"]
        and not row["target_native_oracle_required"]
        and finite_number(row["top5"])
        and finite_number(row["top1"])
        and finite_number(row["js"])
        and finite_number(row["entropy_diff"])
        and row["top5"] >= FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS["top5"]
        and row["top1"] >= FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS["top1"]
        and row["js"] <= FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS["js"]
        and row["entropy_diff"] <= FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS["entropy_diff"]
    )


def flagship_completion_pass(row: dict) -> bool:
    return bool(
        finite_number(row.get("source_completion_preferred"))
        and row["source_completion_preferred"]
        >= FLAGSHIP_COMPLETION_PRESERVATION_THRESHOLDS[
            "source_top1_completion_preferred"
        ]
    )


def flagship_source_surface_repair_pass(row: dict) -> bool:
    return bool(
        row["distribution_near_lossless_pass"]
        and finite_number(row.get("source_preservation_top1"))
        and finite_number(row.get("source_preservation_top1_in_topk"))
        and finite_number(row.get("source_completion_preferred"))
        and row["source_preservation_top1"]
        >= FLAGSHIP_SOURCE_SURFACE_REPAIR_THRESHOLDS["top1_surface_agree"]
        and row["source_preservation_top1_in_topk"]
        >= FLAGSHIP_SOURCE_SURFACE_REPAIR_THRESHOLDS[
            "source_top1_in_target_topk"
        ]
        and row["source_completion_preferred"]
        >= FLAGSHIP_SOURCE_SURFACE_REPAIR_THRESHOLDS[
            "source_top1_completion_preferred"
        ]
    )


def flagship_ordinary_source_surface_pass(row: dict) -> bool:
    return bool(
        row["nib_pass"]
        and finite_number(row.get("source_preservation_top1"))
        and finite_number(row.get("source_preservation_top1_in_topk"))
        and finite_number(row.get("source_completion_preferred"))
        and row["source_preservation_top1"]
        >= FLAGSHIP_SOURCE_SURFACE_REPAIR_THRESHOLDS["top1_surface_agree"]
        and row["source_preservation_top1_in_topk"]
        >= FLAGSHIP_SOURCE_SURFACE_REPAIR_THRESHOLDS[
            "source_top1_in_target_topk"
        ]
        and row["source_completion_preferred"]
        >= FLAGSHIP_SOURCE_SURFACE_REPAIR_THRESHOLDS[
            "source_top1_completion_preferred"
        ]
    )


def flagship_recipe_key(row: dict) -> tuple:
    return (
        row["source"],
        row["target"],
        row["oracle_mode"],
        row["target_native_oracle_required"],
        row["d_abi"],
        row["domain_corpus"],
        row["domain_steps"],
        row["calibration_steps"],
        row["cal_lr"],
        row["cal_lr_decay_step"],
        row["cal_lr_decay_factor"],
        row["calibration_mode"],
        row["calibration_init"],
        row["topk"],
        row["topk_kd_weight"],
        row.get("rank_margin_weight"),
        row.get("rank_top_pos"),
        row.get("rank_neg_k"),
        row.get("rank_margin"),
        row.get("hard_neg_weight"),
        row.get("hard_neg_k"),
        row["topset_k"],
        row["topset_weight"],
        row["top_logit_mse_weight"],
        row["top_logit_mse_k"],
        row["entropy_weight"],
        row["target_residual"],
        row["target_residual_rank"],
    )


def repeated_flagship_distribution_recipe_count(rows: list[dict]) -> int:
    variants_by_recipe = {}
    for row in rows:
        if not row["distribution_near_lossless_pass"]:
            continue
        variants_by_recipe.setdefault(flagship_recipe_key(row), set()).add(
            (row["seed"], row["seed_offset"])
        )
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


def repeated_flagship_distribution_completion_recipe_count(rows: list[dict]) -> int:
    variants_by_recipe = {}
    for row in rows:
        if not row["distribution_completion_pass"]:
            continue
        variants_by_recipe.setdefault(flagship_recipe_key(row), set()).add(
            (row["seed"], row["seed_offset"])
        )
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


def flagship_source_completion_loss_key(row: dict) -> tuple:
    loss = row.get("source_completion_loss") or {}
    ema = row.get("calibration_ema") or {}
    return (
        *flagship_recipe_key(row),
        row.get("native_domain_seed_base"),
        bool((row.get("target_interface_cache") or {}).get("loaded")),
        loss.get("weight"),
        loss.get("every"),
        loss.get("batch"),
        loss.get("prompt_limit"),
        loss.get("candidates"),
        loss.get("temperature"),
        loss.get("start_step"),
        loss.get("margin_weight"),
        loss.get("margin"),
        loss.get("nll_weight"),
        loss.get("nll_cap"),
        ema.get("decay"),
        ema.get("start_step"),
        ema.get("every"),
        ema.get("restored"),
    )


def flagship_base_reference_bypass_key(row: dict) -> tuple:
    loss = row.get("source_completion_loss") or {}
    ema = row.get("calibration_ema") or {}
    posthoc = row.get("posthoc_logit_scale") or {}
    posthoc_candidate_count = posthoc.get("candidate_count", 1)
    selected_scale_key = (
        posthoc.get("scale") if posthoc_candidate_count in (None, 1) else None
    )
    return (
        *flagship_recipe_key(row),
        row.get("target_reference_bypass_abi"),
        row.get("target_reference_forward_mode"),
        loss.get("enabled"),
        loss.get("weight"),
        loss.get("every"),
        loss.get("batch"),
        loss.get("prompt_limit"),
        loss.get("candidates"),
        loss.get("temperature"),
        loss.get("start_step"),
        loss.get("margin_weight"),
        loss.get("margin"),
        loss.get("nll_weight"),
        loss.get("nll_cap"),
        ema.get("decay"),
        ema.get("start_step"),
        ema.get("every"),
        ema.get("restored"),
        posthoc.get("applied"),
        posthoc.get("mode"),
        selected_scale_key,
        posthoc_candidate_count,
        posthoc.get("calibration_chunks"),
        posthoc.get("calibration_repeats"),
        posthoc.get("scale_min"),
        posthoc.get("scale_max"),
        posthoc.get("selection_rule"),
        posthoc.get("signed_entropy_weight"),
    )


def repeated_flagship_base_reference_bypass_completion_recipe_count(
    rows: list[dict],
) -> int:
    variants_by_recipe = {}
    for row in rows:
        if (
            row.get("oracle_mode") != "base_target_reference"
            or not row.get("target_reference_bypass_abi")
            or row.get("target_reference_forward_mode") != "base"
            or not row["distribution_completion_pass"]
        ):
            continue
        variants_by_recipe.setdefault(
            flagship_base_reference_bypass_key(row), set()
        ).add((row["seed"], row["seed_offset"]))
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


def repeated_flagship_base_reference_bypass_nonfixed_completion_recipe_count(
    rows: list[dict],
) -> int:
    variants_by_recipe = {}
    for row in rows:
        posthoc = row.get("posthoc_logit_scale") or {}
        if (
            row.get("oracle_mode") != "base_target_reference"
            or not row.get("target_reference_bypass_abi")
            or row.get("target_reference_forward_mode") != "base"
            or not row["distribution_completion_pass"]
            or posthoc.get("candidate_count", 1) <= 1
        ):
            continue
        variants_by_recipe.setdefault(
            flagship_base_reference_bypass_key(row), set()
        ).add((row["seed"], row["seed_offset"]))
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


def repeated_flagship_base_reference_bypass_nonfixed_completion_domains(
    rows: list[dict],
) -> set[str]:
    variants_by_recipe = {}
    domain_by_recipe = {}
    for row in rows:
        posthoc = row.get("posthoc_logit_scale") or {}
        if (
            row.get("oracle_mode") != "base_target_reference"
            or not row.get("target_reference_bypass_abi")
            or row.get("target_reference_forward_mode") != "base"
            or not row["distribution_completion_pass"]
            or posthoc.get("candidate_count", 1) <= 1
        ):
            continue
        key = flagship_base_reference_bypass_key(row)
        variants_by_recipe.setdefault(key, set()).add((row["seed"], row["seed_offset"]))
        domain_by_recipe[key] = row.get("domain_corpus")
    return {
        domain_by_recipe[key]
        for key, variants in variants_by_recipe.items()
        if len(variants) >= 2 and domain_by_recipe.get(key)
    }


def repeated_base_reference_bypass_target_pairs(rows: list[dict]) -> set[str]:
    variants_by_recipe = {}
    pair_by_recipe = {}
    for row in rows:
        if (
            row.get("oracle_mode") != "base_target_reference"
            or not row.get("target_reference_bypass_abi")
            or row.get("target_reference_forward_mode") != "base"
            or not row["distribution_completion_pass"]
        ):
            continue
        key = flagship_base_reference_bypass_key(row)
        variants_by_recipe.setdefault(key, set()).add((row["seed"], row["seed_offset"]))
        pair_by_recipe[key] = f"{row.get('source')} -> {row.get('target')}"
    return {
        pair_by_recipe[key]
        for key, variants in variants_by_recipe.items()
        if len(variants) >= 2 and pair_by_recipe.get(key)
    }


def repeated_base_reference_bypass_source_surface_repair_recipe_count(
    rows: list[dict],
) -> int:
    variants_by_recipe = {}
    for row in rows:
        if (
            row.get("oracle_mode") != "base_target_reference"
            or not row.get("target_reference_bypass_abi")
            or row.get("target_reference_forward_mode") != "base"
            or not row.get("source_surface_repair_pass")
        ):
            continue
        variants_by_recipe.setdefault(
            flagship_base_reference_bypass_key(row), set()
        ).add((row["seed"], row["seed_offset"]))
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


def repeated_base_reference_bypass_source_surface_repair_pairs(
    rows: list[dict],
) -> set[str]:
    variants_by_recipe = {}
    pair_by_recipe = {}
    for row in rows:
        if (
            row.get("oracle_mode") != "base_target_reference"
            or not row.get("target_reference_bypass_abi")
            or row.get("target_reference_forward_mode") != "base"
            or not row.get("source_surface_repair_pass")
        ):
            continue
        key = flagship_base_reference_bypass_key(row)
        variants_by_recipe.setdefault(key, set()).add((row["seed"], row["seed_offset"]))
        pair_by_recipe[key] = f"{row.get('source')} -> {row.get('target')}"
    return {
        pair_by_recipe[key]
        for key, variants in variants_by_recipe.items()
        if len(variants) >= 2 and pair_by_recipe.get(key)
    }


def repeated_base_reference_bypass_ordinary_source_surface_recipe_count(
    rows: list[dict],
) -> int:
    variants_by_recipe = {}
    for row in rows:
        if (
            row.get("oracle_mode") != "base_target_reference"
            or not row.get("target_reference_bypass_abi")
            or row.get("target_reference_forward_mode") != "base"
            or not row.get("ordinary_source_surface_pass")
        ):
            continue
        variants_by_recipe.setdefault(
            flagship_base_reference_bypass_key(row), set()
        ).add((row["seed"], row["seed_offset"]))
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


def repeated_base_reference_bypass_ordinary_source_surface_pairs(
    rows: list[dict],
) -> set[str]:
    variants_by_recipe = {}
    pair_by_recipe = {}
    for row in rows:
        if (
            row.get("oracle_mode") != "base_target_reference"
            or not row.get("target_reference_bypass_abi")
            or row.get("target_reference_forward_mode") != "base"
            or not row.get("ordinary_source_surface_pass")
        ):
            continue
        key = flagship_base_reference_bypass_key(row)
        variants_by_recipe.setdefault(key, set()).add((row["seed"], row["seed_offset"]))
        pair_by_recipe[key] = f"{row.get('source')} -> {row.get('target')}"
    return {
        pair_by_recipe[key]
        for key, variants in variants_by_recipe.items()
        if len(variants) >= 2 and pair_by_recipe.get(key)
    }


def repeated_flagship_cache_source_completion_loss_recipe_count(
    rows: list[dict],
) -> int:
    variants_by_recipe = {}
    for row in rows:
        loss = row.get("source_completion_loss") or {}
        if (
            not row["distribution_completion_pass"]
            or not (row.get("target_interface_cache") or {}).get("loaded")
            or not loss.get("enabled")
        ):
            continue
        variants_by_recipe.setdefault(
            flagship_source_completion_loss_key(row), set()
        ).add((row["seed"], row["seed_offset"]))
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


def repeated_flagship_zeroout_completion_recipe_count(rows: list[dict]) -> int:
    variants_by_recipe = {}
    for row in rows:
        if (
            row.get("calibration_init") != "zero_out"
            or not row["distribution_completion_pass"]
        ):
            continue
        variants_by_recipe.setdefault(
            flagship_source_completion_loss_key(row), set()
        ).add((row["seed"], row["seed_offset"]))
    return sum(1 for variants in variants_by_recipe.values() if len(variants) >= 2)


def build_summary() -> dict:
    savings = savings_rows()
    hard = hard_frontier_rows()
    breadth = domain_breadth()
    baselines = baseline_rows()
    withheld = withheld_rows()
    abi_lora_frontier = abi_lora_frontier_rows()
    phi_frontier = phi_abi_frontier_rows()
    north_star_rows = north_star_gate_rows()
    selective_rows = selective_transfer_rows()
    selective_repeat_count = repeated_selective_transfer_recipe_count(
        selective_rows
    )
    flagship_rows = flagship_gpt_style_rows()
    gpt2_base_bypass_breadth_rows = gpt2_base_bypass_target_breadth_rows()
    base_bypass_source_surface_rows = base_bypass_source_surface_breadth_rows()
    gpt2_base_bypass_breadth_strict_completion_rows = [
        row
        for row in gpt2_base_bypass_breadth_rows
        if row["distribution_completion_pass"]
    ]
    gpt2_base_bypass_breadth_nonfixed_rows = [
        row
        for row in gpt2_base_bypass_breadth_rows
        if (row.get("posthoc_logit_scale") or {}).get("candidate_count", 1) > 1
    ]
    flagship_distribution_rows = [
        row for row in flagship_rows if row["distribution_near_lossless_pass"]
    ]
    flagship_oracle_light_distribution_rows = [
        row
        for row in flagship_distribution_rows
        if row["oracle_mode"] == "target_base_interface"
    ]
    flagship_distribution_completion_rows = [
        row for row in flagship_rows if row["distribution_completion_pass"]
    ]
    flagship_source_surface = [
        row["source_preservation_top1_in_topk"]
        for row in flagship_rows
        if row["distribution_near_lossless_pass"]
        and row["source_preservation_top1_in_topk"] is not None
    ]
    flagship_completion_scores = [
        row["source_completion_preferred"]
        for row in flagship_rows
        if row["source_completion_preferred"] is not None
    ]
    flagship_base_reference_rows = [
        row for row in flagship_rows if row["oracle_mode"] == "base_target_reference"
    ]
    flagship_base_reference_negative_rows = [
        row for row in flagship_base_reference_rows if row["nib_pass"] is False
    ]
    flagship_base_reference_bypass_rows = [
        row
        for row in flagship_base_reference_rows
        if row.get("target_reference_bypass_abi")
        and row.get("target_reference_forward_mode") == "base"
    ]
    flagship_base_reference_bypass_nonfixed_rows = [
        row
        for row in flagship_base_reference_bypass_rows
        if (row.get("posthoc_logit_scale") or {}).get("candidate_count", 1) > 1
    ]
    flagship_xavier_target_interface_rows = [
        row
        for row in flagship_rows
        if row["oracle_mode"] == "target_base_interface"
        and row["calibration_init"] == "xavier"
    ]
    flagship_zeroout_target_interface_rows = [
        row
        for row in flagship_rows
        if row["oracle_mode"] == "target_base_interface"
        and row["calibration_init"] == "zero_out"
    ]
    flagship_native_strict_completion_rows = [
        row
        for row in flagship_distribution_completion_rows
        if row["oracle_mode"] == "target_base_interface"
        and row["calibration_init"] == "native"
    ]
    flagship_cache_loaded_rows = [
        row
        for row in flagship_rows
        if (row.get("target_interface_cache") or {}).get("loaded")
    ]
    flagship_cache_saved_rows = [
        row
        for row in flagship_rows
        if (row.get("target_interface_cache") or {}).get("saved")
    ]
    flagship_source_completion_loss_rows = [
        row
        for row in flagship_rows
        if (row.get("source_completion_loss") or {}).get("enabled")
    ]
    flagship_cache_source_completion_loss_rows = [
        row
        for row in flagship_source_completion_loss_rows
        if (row.get("target_interface_cache") or {}).get("loaded")
    ]
    flagship_repeat_count = repeated_flagship_distribution_recipe_count(flagship_rows)
    flagship_completion_repeat_count = (
        repeated_flagship_distribution_completion_recipe_count(flagship_rows)
    )
    flagship_cache_source_completion_loss_repeat_count = (
        repeated_flagship_cache_source_completion_loss_recipe_count(flagship_rows)
    )
    flagship_zeroout_completion_repeat_count = (
        repeated_flagship_zeroout_completion_recipe_count(flagship_rows)
    )
    flagship_base_reference_bypass_completion_repeat_count = (
        repeated_flagship_base_reference_bypass_completion_recipe_count(flagship_rows)
    )
    flagship_base_reference_bypass_nonfixed_completion_repeat_count = (
        repeated_flagship_base_reference_bypass_nonfixed_completion_recipe_count(
            flagship_rows
        )
    )
    flagship_base_reference_bypass_nonfixed_completion_domains = (
        repeated_flagship_base_reference_bypass_nonfixed_completion_domains(
            flagship_rows
        )
    )
    gpt2_base_bypass_breadth_repeat_pairs = (
        repeated_base_reference_bypass_target_pairs(gpt2_base_bypass_breadth_rows)
    )
    gpt2_base_bypass_breadth_repeat_count = (
        repeated_flagship_base_reference_bypass_completion_recipe_count(
            gpt2_base_bypass_breadth_rows
        )
    )
    gpt2_base_bypass_breadth_nonfixed_repeat_count = (
        repeated_flagship_base_reference_bypass_nonfixed_completion_recipe_count(
            gpt2_base_bypass_breadth_rows
        )
    )
    gpt2_base_bypass_breadth_surface_repair_repeat_count = (
        repeated_base_reference_bypass_source_surface_repair_recipe_count(
            gpt2_base_bypass_breadth_rows
        )
    )
    gpt2_base_bypass_breadth_surface_repair_pairs = (
        repeated_base_reference_bypass_source_surface_repair_pairs(
            gpt2_base_bypass_breadth_rows
        )
    )
    base_bypass_source_surface_repair_rows = [
        row for row in base_bypass_source_surface_rows if row["source_surface_repair_pass"]
    ]
    base_bypass_source_surface_repair_repeat_count = (
        repeated_base_reference_bypass_source_surface_repair_recipe_count(
            base_bypass_source_surface_rows
        )
    )
    base_bypass_source_surface_repair_pairs = (
        repeated_base_reference_bypass_source_surface_repair_pairs(
            base_bypass_source_surface_rows
        )
    )
    base_bypass_source_surface_repair_sources = {
        pair.split(" -> ", 1)[0]
        for pair in base_bypass_source_surface_repair_pairs
        if " -> " in pair
    }
    hard_recipient_ordinary_source_surface_rows = [
        row
        for row in base_bypass_source_surface_rows
        if row.get("target") in HARD_RECIPIENT_TARGETS
        and row.get("ordinary_source_surface_pass")
    ]
    hard_recipient_ordinary_source_surface_repeat_count = (
        repeated_base_reference_bypass_ordinary_source_surface_recipe_count(
            hard_recipient_ordinary_source_surface_rows
        )
    )
    hard_recipient_ordinary_source_surface_pairs = (
        repeated_base_reference_bypass_ordinary_source_surface_pairs(
            hard_recipient_ordinary_source_surface_rows
        )
    )
    fractions = [
        row["trainable_fraction_of_target"]
        for row in savings
        if row["trainable_fraction_of_target"] is not None
    ]
    summary = {
        "savings": {
            "rows": savings,
            "min_trainable_fraction_of_target": min(fractions) if fractions else None,
            "max_trainable_fraction_of_target": max(fractions) if fractions else None,
        },
        "accuracy_frontier": {
            "hard_direction": "EleutherAI/pythia-410m -> deepseek-ai/deepseek-coder-1.3b-base",
            "rows": hard,
            "best_seed42_top5": max(
                row["top5"] for row in hard if row["seed"] == 42 and row["pass"]
            ),
        },
        "domain_breadth": breadth,
        "baselines": {
            "rows": baselines,
            "row_count": len(baselines),
            "passing_count": sum(1 for row in baselines if row["pass"]),
            "best_top5": max(
                (row["top5"] for row in baselines if math.isfinite(row["top5"])),
                default=None,
            ),
            "min_trainable_fraction_of_target": min(
                (
                    row["trainable_fraction_of_target"]
                    for row in baselines
                    if row["trainable_fraction_of_target"] is not None
                ),
                default=None,
            ),
        },
        "withheld_evaluation": {
            "rows": withheld,
            "row_count": len(withheld),
            "passing_count": sum(1 for row in withheld if row["pass"]),
            "min_top5": min((row["top5"] for row in withheld), default=None),
            "max_entropy_diff": max(
                (row["entropy_diff"] for row in withheld),
                default=None,
            ),
            "all_rows_have_split_separation": all(
                row["withheld_nib_eval"]
                and row["wikitext_domain_split"] != row["wikitext_eval_split"]
                and row["wikitext_posthoc_split"] != row["wikitext_eval_split"]
                for row in withheld
            ),
        },
        "abi_lora_frontier": {
            "rows": abi_lora_frontier,
            "row_count": len(abi_lora_frontier),
            "passing_count": sum(1 for row in abi_lora_frontier if row["pass"]),
            "max_top5": max((row["top5"] for row in abi_lora_frontier), default=None),
            "min_js": min((row["js"] for row in abi_lora_frontier), default=None),
            "min_entropy_diff": min(
                (row["entropy_diff"] for row in abi_lora_frontier),
                default=None,
            ),
        },
        "phi_abi_frontier": {
            "rows": phi_frontier,
            "row_count": len(phi_frontier),
            "passing_count": sum(1 for row in phi_frontier if row["pass"]),
            "max_top5": max((row["top5"] for row in phi_frontier), default=None),
            "max_top1": max((row["top1"] for row in phi_frontier), default=None),
        },
        "north_star_gates": {
            "rows": north_star_rows,
            "row_count": len(north_star_rows),
            "oracle_light_row_count": sum(
                1 for row in north_star_rows if row["oracle_light_mode"]
            ),
            "source_preservation_row_count": sum(
                1 for row in north_star_rows if row["source_preservation_measured"]
            ),
            "joint_oracle_light_source_preservation_count": sum(
                1
                for row in north_star_rows
                if row["oracle_light_mode"]
                and row["source_preservation_measured"]
            ),
            "joint_passing_count": sum(
                1
                for row in north_star_rows
                if north_star_joint_pass(row)
            ),
            "repeat_joint_passing_recipe_count": repeated_north_star_recipe_count(
                north_star_rows
            ),
            "repeat_joint_passing_pair_count": repeated_north_star_pair_count(
                north_star_rows
            ),
            "repeat_required_recipe_count": 2,
        },
        "selective_transfer_gates": {
            "rows": selective_rows,
            "row_count": len(selective_rows),
            "measured_count": len(selective_rows),
            "ordinary_pass_count": sum(
                1 for row in selective_rows if row["selective_transfer_pass"]
            ),
            "target_domain_strict_pass_count": sum(
                1 for row in selective_rows if row["target_domain_strict_pass"]
            ),
            "off_domain_no_leakage_pass_count": sum(
                1 for row in selective_rows if row["off_domain_no_leakage_pass"]
            ),
            "off_domain_strict_pass_count": sum(
                1 for row in selective_rows if row["off_domain_strict_pass"]
            ),
            "strict_selective_transfer_pass_count": sum(
                1
                for row in selective_rows
                if row["strict_selective_transfer_pass"]
            ),
            "repeat_strict_selective_recipe_count": selective_repeat_count,
            "repeat_required_recipe_count": SELECTIVE_TRANSFER_REQUIRED_REPEAT_RECIPES,
            "strict_thresholds": FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS,
            "ready": bool(
                selective_repeat_count >= SELECTIVE_TRANSFER_REQUIRED_REPEAT_RECIPES
            ),
            "open_blockers": [
                "No opt-in selective off-domain audit artifacts have been generated yet."
                if not selective_rows
                else "Strict selective transfer lacks enough repeat-certified recipes.",
                "Need paired target-domain pass and off-domain no-leakage pass for at least two independent seed/stream variants.",
                "Need task-level selected-domain and off-domain evaluations, not only logit-space NIB.",
            ],
        },
        "flagship_gpt_style": {
            "scenario": "gpt2-medium -> microsoft/phi-3-mini-4k-instruct",
            "rows": flagship_rows,
            "row_count": len(flagship_rows),
            "near_lossless_distribution_thresholds": FLAGSHIP_NEAR_LOSSLESS_THRESHOLDS,
            "completion_preservation_thresholds": (
                FLAGSHIP_COMPLETION_PRESERVATION_THRESHOLDS
            ),
            "source_surface_repair_thresholds": (
                FLAGSHIP_SOURCE_SURFACE_REPAIR_THRESHOLDS
            ),
            "distribution_near_lossless_pass_count": len(
                flagship_distribution_rows
            ),
            "oracle_light_distribution_pass_count": len(
                flagship_oracle_light_distribution_rows
            ),
            "cross_tokenizer_completion_measured_count": len(
                flagship_completion_scores
            ),
            "distribution_completion_pass_count": len(
                flagship_distribution_completion_rows
            ),
            "repeat_distribution_recipe_count": flagship_repeat_count,
            "repeat_distribution_completion_recipe_count": (
                flagship_completion_repeat_count
            ),
            "repeat_required_recipe_count": 1,
            "ready": flagship_repeat_count >= 1,
            "completion_ready": flagship_completion_repeat_count >= 1,
            "benchmark_ready": (
                flagship_repeat_count >= 1 and flagship_completion_repeat_count >= 1
            ),
            "base_reference_negative_control_count": sum(
                1 for row in flagship_base_reference_negative_rows
            ),
            "base_reference_negative_control_best_top5": max(
                (row["top5"] for row in flagship_base_reference_negative_rows),
                default=None,
            ),
            "base_reference_best_top5": max(
                (row["top5"] for row in flagship_base_reference_rows),
                default=None,
            ),
            "base_reference_best_completion": max(
                (
                    row["source_completion_preferred"]
                    for row in flagship_base_reference_rows
                    if row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "base_reference_strict_completion_pass_count": sum(
                1 for row in flagship_base_reference_rows
                if row["distribution_completion_pass"]
            ),
            "base_reference_repeat_strict_completion_recipe_count": (
                flagship_base_reference_bypass_completion_repeat_count
            ),
            "base_reference_bypass_strict_completion_pass_count": sum(
                1
                for row in flagship_base_reference_bypass_rows
                if row["distribution_completion_pass"]
            ),
            "base_reference_bypass_repeat_strict_completion_recipe_count": (
                flagship_base_reference_bypass_completion_repeat_count
            ),
            "base_reference_bypass_nonfixed_strict_completion_pass_count": sum(
                1
                for row in flagship_base_reference_bypass_nonfixed_rows
                if row["distribution_completion_pass"]
            ),
            "base_reference_bypass_nonfixed_repeat_strict_completion_recipe_count": (
                flagship_base_reference_bypass_nonfixed_completion_repeat_count
            ),
            "base_reference_bypass_nonfixed_repeat_strict_completion_domain_count": len(
                flagship_base_reference_bypass_nonfixed_completion_domains
            ),
            "base_reference_bypass_nonfixed_repeat_strict_completion_domains": sorted(
                flagship_base_reference_bypass_nonfixed_completion_domains
            ),
            "base_reference_bypass_nonfixed_best_top5": max(
                (row["top5"] for row in flagship_base_reference_bypass_nonfixed_rows),
                default=None,
            ),
            "base_reference_bypass_nonfixed_best_completion": max(
                (
                    row["source_completion_preferred"]
                    for row in flagship_base_reference_bypass_nonfixed_rows
                    if row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_cross_target_rows": (
                gpt2_base_bypass_breadth_rows
            ),
            "base_reference_bypass_cross_target_strict_completion_pass_count": sum(
                1 for row in gpt2_base_bypass_breadth_strict_completion_rows
            ),
            "base_reference_bypass_cross_target_repeat_strict_completion_recipe_count": (
                gpt2_base_bypass_breadth_repeat_count
            ),
            "base_reference_bypass_cross_target_repeat_strict_completion_pair_count": len(
                gpt2_base_bypass_breadth_repeat_pairs
            ),
            "base_reference_bypass_cross_target_repeat_strict_completion_pairs": sorted(
                gpt2_base_bypass_breadth_repeat_pairs
            ),
            "base_reference_bypass_cross_target_nonfixed_strict_completion_pass_count": sum(
                1
                for row in gpt2_base_bypass_breadth_nonfixed_rows
                if row["distribution_completion_pass"]
            ),
            "base_reference_bypass_cross_target_nonfixed_repeat_strict_completion_recipe_count": (
                gpt2_base_bypass_breadth_nonfixed_repeat_count
            ),
            "base_reference_bypass_cross_target_best_top5": max(
                (
                    row["top5"]
                    for row in gpt2_base_bypass_breadth_strict_completion_rows
                ),
                default=None,
            ),
            "base_reference_bypass_cross_target_best_completion": max(
                (
                    row["source_completion_preferred"]
                    for row in gpt2_base_bypass_breadth_strict_completion_rows
                    if row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_cross_target_source_surface_repair_pass_count": sum(
                1
                for row in gpt2_base_bypass_breadth_rows
                if row.get("source_surface_repair_pass")
            ),
            "base_reference_bypass_cross_target_source_surface_repair_repeat_recipe_count": (
                gpt2_base_bypass_breadth_surface_repair_repeat_count
            ),
            "base_reference_bypass_cross_target_source_surface_repair_repeat_pair_count": len(
                gpt2_base_bypass_breadth_surface_repair_pairs
            ),
            "base_reference_bypass_cross_target_source_surface_repair_repeat_pairs": sorted(
                gpt2_base_bypass_breadth_surface_repair_pairs
            ),
            "base_reference_bypass_cross_target_best_source_top1_surface": max(
                (
                    row["source_preservation_top1"]
                    for row in gpt2_base_bypass_breadth_rows
                    if row.get("source_surface_repair_pass")
                    and row["source_preservation_top1"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_cross_target_min_repaired_source_top1_surface": min(
                (
                    row["source_preservation_top1"]
                    for row in gpt2_base_bypass_breadth_rows
                    if row.get("source_surface_repair_pass")
                    and row["source_preservation_top1"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_cross_target_min_repaired_source_top1_in_topk": min(
                (
                    row["source_preservation_top1_in_topk"]
                    for row in gpt2_base_bypass_breadth_rows
                    if row.get("source_surface_repair_pass")
                    and row["source_preservation_top1_in_topk"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_cross_target_min_repaired_completion": min(
                (
                    row["source_completion_preferred"]
                    for row in gpt2_base_bypass_breadth_rows
                    if row.get("source_surface_repair_pass")
                    and row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_source_surface_rows": (
                base_bypass_source_surface_rows
            ),
            "base_reference_bypass_source_surface_repair_pass_count": len(
                base_bypass_source_surface_repair_rows
            ),
            "base_reference_bypass_source_surface_repair_repeat_recipe_count": (
                base_bypass_source_surface_repair_repeat_count
            ),
            "base_reference_bypass_source_surface_repair_repeat_pair_count": len(
                base_bypass_source_surface_repair_pairs
            ),
            "base_reference_bypass_source_surface_repair_repeat_pairs": sorted(
                base_bypass_source_surface_repair_pairs
            ),
            "base_reference_bypass_source_surface_repair_repeat_source_count": len(
                base_bypass_source_surface_repair_sources
            ),
            "base_reference_bypass_source_surface_repair_repeat_sources": sorted(
                base_bypass_source_surface_repair_sources
            ),
            "base_reference_bypass_source_surface_nll_repair_pass_count": sum(
                1
                for row in base_bypass_source_surface_repair_rows
                if (row.get("source_completion_loss") or {}).get("nll_weight", 0)
                > 0
            ),
            "base_reference_bypass_source_surface_best_top1_surface": max(
                (
                    row["source_preservation_top1"]
                    for row in base_bypass_source_surface_repair_rows
                    if row["source_preservation_top1"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_source_surface_min_repaired_top1_surface": min(
                (
                    row["source_preservation_top1"]
                    for row in base_bypass_source_surface_repair_rows
                    if row["source_preservation_top1"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_source_surface_min_repaired_top1_in_topk": min(
                (
                    row["source_preservation_top1_in_topk"]
                    for row in base_bypass_source_surface_repair_rows
                    if row["source_preservation_top1_in_topk"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_source_surface_min_repaired_completion": min(
                (
                    row["source_completion_preferred"]
                    for row in base_bypass_source_surface_repair_rows
                    if row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "hard_recipient_ordinary_source_surface_rows": (
                hard_recipient_ordinary_source_surface_rows
            ),
            "hard_recipient_ordinary_source_surface_pass_count": len(
                hard_recipient_ordinary_source_surface_rows
            ),
            "hard_recipient_ordinary_source_surface_repeat_recipe_count": (
                hard_recipient_ordinary_source_surface_repeat_count
            ),
            "hard_recipient_ordinary_source_surface_repeat_pair_count": len(
                hard_recipient_ordinary_source_surface_pairs
            ),
            "hard_recipient_ordinary_source_surface_repeat_pairs": sorted(
                hard_recipient_ordinary_source_surface_pairs
            ),
            "hard_recipient_ordinary_source_surface_min_top5": min(
                (
                    row["top5"]
                    for row in hard_recipient_ordinary_source_surface_rows
                    if row["top5"] is not None
                ),
                default=None,
            ),
            "hard_recipient_ordinary_source_surface_best_top5": max(
                (
                    row["top5"]
                    for row in hard_recipient_ordinary_source_surface_rows
                    if row["top5"] is not None
                ),
                default=None,
            ),
            "hard_recipient_ordinary_source_surface_best_top1": max(
                (
                    row["top1"]
                    for row in hard_recipient_ordinary_source_surface_rows
                    if row["top1"] is not None
                ),
                default=None,
            ),
            "hard_recipient_ordinary_source_surface_best_js": min(
                (
                    row["js"]
                    for row in hard_recipient_ordinary_source_surface_rows
                    if row["js"] is not None
                ),
                default=None,
            ),
            "hard_recipient_ordinary_source_surface_best_entropy_diff": min(
                (
                    row["entropy_diff"]
                    for row in hard_recipient_ordinary_source_surface_rows
                    if row["entropy_diff"] is not None
                ),
                default=None,
            ),
            "hard_recipient_ordinary_source_surface_min_top1_surface": min(
                (
                    row["source_preservation_top1"]
                    for row in hard_recipient_ordinary_source_surface_rows
                    if row["source_preservation_top1"] is not None
                ),
                default=None,
            ),
            "hard_recipient_ordinary_source_surface_min_top1_in_topk": min(
                (
                    row["source_preservation_top1_in_topk"]
                    for row in hard_recipient_ordinary_source_surface_rows
                    if row["source_preservation_top1_in_topk"] is not None
                ),
                default=None,
            ),
            "hard_recipient_ordinary_source_surface_min_completion": min(
                (
                    row["source_completion_preferred"]
                    for row in hard_recipient_ordinary_source_surface_rows
                    if row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "base_reference_bypass_best_top5": max(
                (row["top5"] for row in flagship_base_reference_bypass_rows),
                default=None,
            ),
            "base_reference_bypass_best_completion": max(
                (
                    row["source_completion_preferred"]
                    for row in flagship_base_reference_bypass_rows
                    if row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "xavier_target_interface_best_top5": max(
                (row["top5"] for row in flagship_xavier_target_interface_rows),
                default=None,
            ),
            "xavier_target_interface_strict_completion_pass_count": sum(
                1
                for row in flagship_xavier_target_interface_rows
                if row["distribution_completion_pass"]
            ),
            "zeroout_target_interface_best_top5": max(
                (row["top5"] for row in flagship_zeroout_target_interface_rows),
                default=None,
            ),
            "zeroout_target_interface_best_completion": max(
                (
                    row["source_completion_preferred"]
                    for row in flagship_zeroout_target_interface_rows
                    if row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "zeroout_target_interface_strict_completion_pass_count": sum(
                1
                for row in flagship_zeroout_target_interface_rows
                if row["distribution_completion_pass"]
            ),
            "zeroout_target_interface_repeat_strict_completion_recipe_count": (
                flagship_zeroout_completion_repeat_count
            ),
            "native_target_interface_strict_completion_pass_count": len(
                flagship_native_strict_completion_rows
            ),
            "target_interface_cache_saved_count": len(flagship_cache_saved_rows),
            "target_interface_cache_loaded_count": len(flagship_cache_loaded_rows),
            "target_interface_cache_loaded_distribution_pass_count": sum(
                1
                for row in flagship_cache_loaded_rows
                if row["distribution_near_lossless_pass"]
            ),
            "target_interface_cache_loaded_strict_completion_pass_count": sum(
                1
                for row in flagship_cache_loaded_rows
                if row["distribution_completion_pass"]
            ),
            "target_interface_cache_loaded_best_top5": max(
                (row["top5"] for row in flagship_cache_loaded_rows),
                default=None,
            ),
            "target_interface_cache_loaded_best_completion": max(
                (
                    row["source_completion_preferred"]
                    for row in flagship_cache_loaded_rows
                    if row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "source_completion_loss_row_count": len(
                flagship_source_completion_loss_rows
            ),
            "source_completion_loss_strict_completion_pass_count": sum(
                1
                for row in flagship_source_completion_loss_rows
                if row["distribution_completion_pass"]
            ),
            "target_interface_cache_loaded_source_completion_loss_count": len(
                flagship_cache_source_completion_loss_rows
            ),
            "target_interface_cache_loaded_source_completion_loss_strict_completion_pass_count": sum(
                1
                for row in flagship_cache_source_completion_loss_rows
                if row["distribution_completion_pass"]
            ),
            "target_interface_cache_loaded_source_completion_loss_repeat_recipe_count": (
                flagship_cache_source_completion_loss_repeat_count
            ),
            "target_interface_cache_loaded_source_completion_loss_best_completion": max(
                (
                    row["source_completion_preferred"]
                    for row in flagship_cache_source_completion_loss_rows
                    if row["source_completion_preferred"] is not None
                ),
                default=None,
            ),
            "source_surface_min_top1_in_topk": (
                min(flagship_source_surface) if flagship_source_surface else None
            ),
            "source_surface_max_top1_in_topk": (
                max(flagship_source_surface) if flagship_source_surface else None
            ),
            "source_completion_min_preferred": (
                min(flagship_completion_scores) if flagship_completion_scores else None
            ),
            "source_completion_max_preferred": (
                max(flagship_completion_scores) if flagship_completion_scores else None
            ),
            "production_ready": False,
            "production_readiness_status": (
                "local benchmark-ready but not production-ready"
            ),
            "production_blockers": [
                "No GPT5/GPT6 private-model evaluation in this local repo.",
                "Strict distribution + completion now repeat-certifies Phase-C-skipped base-reference bypass across two GPT2-medium local target pairs, but still not on private GPT5/GPT6-scale models.",
                "No-base-reference xavier control still fails NIB top-5.",
                "Xavier target-interface control passes ordinary NIB but misses strict distribution + completion.",
                "The passing no-Phase-C base-bypass recipe still requires zero-out init, EMA restore, source-completion loss, and validation-selected posthoc logit scaling.",
                "Reusable target-interface cache load now repeat-clears strict distribution + completion only when the source-completion loss is enabled.",
                "Source-token surface repair is repeat-certified across two GPT2-medium local target pairs and now extends to GPT-Neo, Phi-3, and Qwen donors across Qwen/Phi target directions, but it is not yet lossless, private-model-scale, or universal across arbitrary source/target families.",
                "Selective-transfer off-domain no-leakage is now a first-class gate, but no opt-in selective audit artifacts have repeat-certified strict passes yet.",
            ],
        },
        "claim_boundary": {
            "can_claim": "Scoped domain-module migration with frozen copied cores and small target-side ABI calibration where certificates pass.",
            "cannot_claim_yet": "Lossless migration of arbitrary GPT5 knowledge into arbitrary GPT6 targets, or replacement of GPT6 base training.",
        },
    }
    summary["adoption_case"] = adoption_case(summary)
    return summary


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def beats_lora_all_metrics(candidate: dict | None, baseline: dict | None) -> bool:
    if candidate is None or baseline is None:
        return False
    required = ("top5", "top1", "js", "entropy_diff")
    if not all(
        math.isfinite(candidate.get(metric, float("nan")))
        and math.isfinite(baseline.get(metric, float("nan")))
        for metric in required
    ):
        return False
    return bool(
        candidate["pass"]
        and candidate["top5"] > baseline["top5"]
        and candidate["top1"] > baseline["top1"]
        and candidate["js"] < baseline["js"]
        and candidate["entropy_diff"] < baseline["entropy_diff"]
    )


def metric_margins_vs_lora(candidate: dict | None, baseline: dict | None) -> dict:
    if candidate is None or baseline is None:
        return {}
    return {
        "top5_margin": candidate["top5"] - baseline["top5"],
        "top1_margin": candidate["top1"] - baseline["top1"],
        "js_reduction": baseline["js"] - candidate["js"],
        "entropy_diff_reduction": baseline["entropy_diff"]
        - candidate["entropy_diff"],
    }


def summarize_direction(rows: list[dict]) -> dict:
    passes = [row for row in rows if row["pass"]]
    return {
        "pass_count": len(passes),
        "min_top5": min((row["top5"] for row in passes), default=None),
        "max_entropy_diff": max((row["entropy_diff"] for row in passes), default=None),
        "best_top5": max((row["top5"] for row in rows), default=None),
        "best_entropy_diff": min((row["entropy_diff"] for row in rows), default=None),
    }


def bidirectional_wikitext_pairs(rows: list[dict]) -> list[dict]:
    pairs: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for row in rows:
        key = tuple(sorted((row["source"], row["target"])))
        direction = f"{row['source']} -> {row['target']}"
        pairs.setdefault(key, {}).setdefault(direction, []).append(row)

    certified = []
    for key, directions in pairs.items():
        if len(directions) != 2:
            continue
        direction_summaries = {
            direction: summarize_direction(direction_rows)
            for direction, direction_rows in directions.items()
        }
        if all(summary["pass_count"] > 0 for summary in direction_summaries.values()):
            certified.append(
                {
                    "pair": " <-> ".join(key),
                    "repeat_certified": all(
                        summary["pass_count"] >= 2
                        for summary in direction_summaries.values()
                    ),
                    "directions": direction_summaries,
                }
            )
    return sorted(certified, key=lambda row: row["pair"])


def adoption_case(summary: dict) -> dict:
    savings = summary["savings"]["rows"]
    hard_rows = summary["accuracy_frontier"]["rows"]
    baseline_rows_ = summary.get("baselines", {}).get("rows", [])
    withheld_rows_ = summary.get("withheld_evaluation", {}).get("rows", [])
    abi_lora_frontier_rows_ = summary.get("abi_lora_frontier", {}).get("rows", [])
    phi_abi_frontier_rows_ = summary.get("phi_abi_frontier", {}).get("rows", [])
    north_star = summary.get("north_star_gates", {})
    selective_transfer = summary.get("selective_transfer_gates", {})
    flagship = summary.get("flagship_gpt_style", {})
    passing_hard = [row for row in hard_rows if row["pass"] and math.isfinite(row["top5"])]
    best_hard = max(passing_hard, key=lambda row: row["top5"])

    target_counted = [row for row in savings if row["target_params_counted"] is not None]
    large_target_rows = [
        row for row in target_counted if row["target_params_counted"] >= 400_000_000
    ]
    ppl_overheads = []
    for row in savings:
        ratio = safe_ratio(row.get("ppl_calibrated_target"), row.get("ppl_native_target"))
        if ratio is not None:
            ppl_overheads.append(ratio - 1.0)

    wikitext_rows = summary["domain_breadth"]["generic_cross_model_wikitext"]
    finite_wikitext = [
        row
        for row in wikitext_rows
        if math.isfinite(row["top5"]) and math.isfinite(row["entropy_diff"])
    ]
    passing_wikitext = [row for row in finite_wikitext if row["pass"]]
    posthoc_wikitext_passes = [
        row
        for row in passing_wikitext
        if row["posthoc_logit_scale"]["applied"]
    ]
    reverse_wikitext = [
        row
        for row in finite_wikitext
        if row["source"] == "Qwen/Qwen2.5-0.5B"
        and row["target"] == "EleutherAI/gpt-neo-125M"
    ]
    reverse_wikitext_passes = [row for row in reverse_wikitext if row["pass"]]
    certified_pairs = bidirectional_wikitext_pairs(finite_wikitext)
    best_wikitext = max(
        passing_wikitext if passing_wikitext else finite_wikitext,
        key=lambda row: row["top5"],
    )
    finite_baselines = [
        row
        for row in baseline_rows_
        if math.isfinite(row["top5"]) and math.isfinite(row["entropy_diff"])
    ]
    passing_baselines = [row for row in finite_baselines if row["pass"]]
    withheld_baselines = [
        row for row in finite_baselines if row.get("withheld_nib_eval") is True
    ]
    passing_withheld_baselines = [row for row in withheld_baselines if row["pass"]]
    qwen_withheld_baselines = [
        row for row in withheld_baselines if row["target"] == "Qwen/Qwen2.5-0.5B"
    ]
    phi_withheld_baselines = [
        row
        for row in withheld_baselines
        if row["target"] == "microsoft/phi-3-mini-4k-instruct"
    ]
    best_baseline = max(
        passing_baselines if passing_baselines else finite_baselines,
        key=lambda row: row["top5"],
        default=None,
    )
    best_withheld_baseline = max(
        passing_withheld_baselines if passing_withheld_baselines else withheld_baselines,
        key=lambda row: row["top5"],
        default=None,
    )
    best_qwen_withheld_baseline = max(
        [row for row in qwen_withheld_baselines if row["pass"]]
        if any(row["pass"] for row in qwen_withheld_baselines)
        else qwen_withheld_baselines,
        key=lambda row: row["top5"],
        default=None,
    )
    best_phi_withheld_baseline = max(
        [row for row in phi_withheld_baselines if row["pass"]]
        if any(row["pass"] for row in phi_withheld_baselines)
        else phi_withheld_baselines,
        key=lambda row: row["top5"],
        default=None,
    )
    finite_withheld = [
        row
        for row in withheld_rows_
        if math.isfinite(row["top5"]) and math.isfinite(row["entropy_diff"])
    ]
    passing_withheld = [row for row in finite_withheld if row["pass"]]
    qwen_phi_withheld = [
        row
        for row in passing_withheld
        if {row["source"], row["target"]}
        == {"Qwen/Qwen2.5-0.5B", "microsoft/phi-3-mini-4k-instruct"}
    ]
    best_withheld = max(
        passing_withheld if passing_withheld else finite_withheld,
        key=lambda row: row["top5"],
        default=None,
    )
    finite_abi_lora_frontier = [
        row
        for row in abi_lora_frontier_rows_
        if math.isfinite(row["top5"]) and math.isfinite(row["entropy_diff"])
    ]
    abi_lora_metric_winners = [
        row
        for row in finite_abi_lora_frontier
        if beats_lora_all_metrics(row, best_qwen_withheld_baseline)
    ]
    best_abi_lora_frontier = max(
        abi_lora_metric_winners if abi_lora_metric_winners else finite_abi_lora_frontier,
        key=lambda row: row["top5"],
        default=None,
    )
    abi_lora_frontier_repeat = None
    if best_abi_lora_frontier is not None:
        repeat_candidates = [
            row
            for row in finite_abi_lora_frontier
            if row["source"] == best_abi_lora_frontier["source"]
            and row["target"] == best_abi_lora_frontier["target"]
            and row["d_abi"] == best_abi_lora_frontier["d_abi"]
            and row["calibration_steps"] == best_abi_lora_frontier["calibration_steps"]
            and row.get("n_align_sentences")
            == best_abi_lora_frontier.get("n_align_sentences")
            and row.get("calibration_mode")
            == best_abi_lora_frontier.get("calibration_mode")
            and row.get("calibration_init", "xavier")
            == best_abi_lora_frontier.get("calibration_init", "xavier")
            and row.get("cal_select_mode", "none")
            == best_abi_lora_frontier.get("cal_select_mode", "none")
            and row.get("cal_select_avg_top_n", 1)
            == best_abi_lora_frontier.get("cal_select_avg_top_n", 1)
            and row.get("cal_ema_decay", 0.0)
            == best_abi_lora_frontier.get("cal_ema_decay", 0.0)
            and row.get("cal_ema_start_step", 1)
            == best_abi_lora_frontier.get("cal_ema_start_step", 1)
            and row.get("cal_ema_restore", False)
            == best_abi_lora_frontier.get("cal_ema_restore", False)
            and row.get("cal_lr_decay_step", 0)
            == best_abi_lora_frontier.get("cal_lr_decay_step", 0)
            and row.get("cal_lr_decay_factor", 1.0)
            == best_abi_lora_frontier.get("cal_lr_decay_factor", 1.0)
            and row.get("topk") == best_abi_lora_frontier.get("topk")
            and row.get("topk_kd_weight")
            == best_abi_lora_frontier.get("topk_kd_weight")
            and row.get("rank_margin_weight")
            == best_abi_lora_frontier.get("rank_margin_weight")
            and row.get("topset_k") == best_abi_lora_frontier.get("topset_k")
            and row.get("topset_weight")
            == best_abi_lora_frontier.get("topset_weight")
            and (
                row["seed"] != best_abi_lora_frontier["seed"]
                or row.get("seed_offset", 0)
                != best_abi_lora_frontier.get("seed_offset", 0)
            )
        ]
        abi_lora_frontier_repeat = max(
            repeat_candidates,
            key=lambda row: row["top5"],
            default=None,
        )
    best_beats_lora = beats_lora_all_metrics(
        best_abi_lora_frontier,
        best_qwen_withheld_baseline,
    )
    repeat_beats_lora = beats_lora_all_metrics(
        abi_lora_frontier_repeat,
        best_qwen_withheld_baseline,
    )
    finite_phi_frontier = [
        row
        for row in phi_abi_frontier_rows_
        if math.isfinite(row["top5"]) and math.isfinite(row["entropy_diff"])
    ]
    phi_metric_winners = [
        row
        for row in finite_phi_frontier
        if beats_lora_all_metrics(row, best_phi_withheld_baseline)
    ]
    best_phi_abi_by_top5 = max(
        finite_phi_frontier,
        key=lambda row: row["top5"],
        default=None,
    )
    best_phi_abi_by_top1 = max(
        finite_phi_frontier,
        key=lambda row: row["top1"],
        default=None,
    )
    best_phi_metric_winner = max(
        phi_metric_winners,
        key=lambda row: row["top5"],
        default=None,
    )
    phi_metric_winner_repeat = None
    if best_phi_metric_winner is not None:
        phi_repeat_candidates = [
            row
            for row in finite_phi_frontier
            if row["source"] == best_phi_metric_winner["source"]
            and row["target"] == best_phi_metric_winner["target"]
            and row["d_abi"] == best_phi_metric_winner["d_abi"]
            and row.get("n_align_sentences")
            == best_phi_metric_winner.get("n_align_sentences")
            and row.get("cal_select_avg_top_n", 1)
            == best_phi_metric_winner.get("cal_select_avg_top_n", 1)
            and row.get("cal_select_mode", "none")
            == best_phi_metric_winner.get("cal_select_mode", "none")
            and row["calibration_steps"] == best_phi_metric_winner["calibration_steps"]
            and row.get("calibration_mode")
            == best_phi_metric_winner.get("calibration_mode")
            and row.get("calibration_init", "xavier")
            == best_phi_metric_winner.get("calibration_init", "xavier")
            and row.get("cal_ema_decay", 0.0)
            == best_phi_metric_winner.get("cal_ema_decay", 0.0)
            and row.get("cal_ema_start_step", 1)
            == best_phi_metric_winner.get("cal_ema_start_step", 1)
            and row.get("cal_ema_restore", False)
            == best_phi_metric_winner.get("cal_ema_restore", False)
            and row.get("top1_gap_weight", 0.0)
            == best_phi_metric_winner.get("top1_gap_weight", 0.0)
            and row.get("top1_ce_weight", 0.0)
            == best_phi_metric_winner.get("top1_ce_weight", 0.0)
            and row.get("top1_hard_neg_weight", 0.0)
            == best_phi_metric_winner.get("top1_hard_neg_weight", 0.0)
            and row.get("top_logit_mse_weight", 0.0)
            == best_phi_metric_winner.get("top_logit_mse_weight", 0.0)
            and row.get("topset_k") == best_phi_metric_winner.get("topset_k")
            and row.get("target_residual")
            == best_phi_metric_winner.get("target_residual")
            and row.get("domain_residual_rank")
            == best_phi_metric_winner.get("domain_residual_rank")
            and (
                row["seed"] != best_phi_metric_winner["seed"]
                or row.get("seed_offset", 0)
                != best_phi_metric_winner.get("seed_offset", 0)
            )
        ]
        phi_metric_winner_repeat = max(
            phi_repeat_candidates,
            key=lambda row: row["top5"],
            default=None,
        )
    phi_repeat_beats_lora = beats_lora_all_metrics(
        phi_metric_winner_repeat,
        best_phi_withheld_baseline,
    )
    north_star_repeat_count = north_star.get("repeat_joint_passing_recipe_count", 0)
    north_star_required_count = north_star.get("repeat_required_recipe_count", 2)
    north_star_ready = north_star_repeat_count >= north_star_required_count
    selective_repeat_count = selective_transfer.get(
        "repeat_strict_selective_recipe_count", 0
    )
    selective_required_count = selective_transfer.get(
        "repeat_required_recipe_count", SELECTIVE_TRANSFER_REQUIRED_REPEAT_RECIPES
    )
    selective_ready = bool(selective_transfer.get("ready", False))

    return {
        "decision_claim": (
            "ABI is ready to be benchmarked as a frozen-core, target-side interface "
            "calibration path for scoped domain migration. The current evidence does "
            "not yet justify replacing base-model training or claiming lossless "
            "arbitrary-model transfer, and it has not yet proven lossless selective "
            "domain transfer with off-domain noninterference."
        ),
        "all_savings_rows_pass": all(row["pass"] for row in savings),
        "local_large_target_threshold_params": 400_000_000,
        "large_target_row_count": len(large_target_rows),
        "large_target_mean_trainable_fraction": mean(
            [row["trainable_fraction_of_target"] for row in large_target_rows]
        ),
        "large_target_max_trainable_fraction": max(
            (row["trainable_fraction_of_target"] for row in large_target_rows),
            default=None,
        ),
        "large_target_min_frozen_fraction": min(
            (row["frozen_fraction_of_target"] for row in large_target_rows),
            default=None,
        ),
        "mean_ppl_relative_overhead": mean(ppl_overheads),
        "max_ppl_relative_overhead": max(ppl_overheads) if ppl_overheads else None,
        "best_hard_direction": best_hard,
        "best_generic_wikitext_rank_transfer": best_wikitext,
        "posthoc_wikitext_pass_count": len(posthoc_wikitext_passes),
        "posthoc_wikitext_min_top5": min(
            (row["top5"] for row in posthoc_wikitext_passes),
            default=None,
        ),
        "posthoc_wikitext_max_entropy_diff": max(
            (row["entropy_diff"] for row in posthoc_wikitext_passes),
            default=None,
        ),
        "reverse_wikitext_pass_count": len(reverse_wikitext_passes),
        "reverse_wikitext_min_top5": min(
            (row["top5"] for row in reverse_wikitext_passes),
            default=None,
        ),
        "reverse_wikitext_max_entropy_diff": max(
            (row["entropy_diff"] for row in reverse_wikitext_passes),
            default=None,
        ),
        "bidirectional_wikitext_pair_count": len(certified_pairs),
        "repeat_certified_bidirectional_wikitext_pair_count": sum(
            1 for pair in certified_pairs if pair["repeat_certified"]
        ),
        "bidirectional_wikitext_pairs": certified_pairs,
        "matched_baseline_row_count": len(finite_baselines),
        "matched_baseline_pass_count": len(passing_baselines),
        "best_matched_baseline": best_baseline,
        "matched_withheld_baseline_row_count": len(withheld_baselines),
        "matched_withheld_baseline_pass_count": len(passing_withheld_baselines),
        "best_matched_withheld_baseline": best_withheld_baseline,
        "matched_withheld_baseline_max_top5": max(
            (row["top5"] for row in withheld_baselines),
            default=None,
        ),
        "matched_withheld_baseline_min_entropy_diff": min(
            (row["entropy_diff"] for row in withheld_baselines),
            default=None,
        ),
        "matched_baseline_min_top5": min(
            (row["top5"] for row in finite_baselines),
            default=None,
        ),
        "matched_baseline_max_top5": max(
            (row["top5"] for row in finite_baselines),
            default=None,
        ),
        "matched_baseline_min_trainable_fraction": min(
            (
                row["trainable_fraction_of_target"]
                for row in finite_baselines
                if row["trainable_fraction_of_target"] is not None
            ),
            default=None,
        ),
        "withheld_eval_row_count": len(finite_withheld),
        "withheld_eval_pass_count": len(passing_withheld),
        "withheld_eval_all_split_separated": summary.get(
            "withheld_evaluation", {}
        ).get("all_rows_have_split_separation", False),
        "best_withheld_eval": best_withheld,
        "withheld_eval_min_top5": min(
            (row["top5"] for row in passing_withheld),
            default=None,
        ),
        "withheld_eval_max_entropy_diff": max(
            (row["entropy_diff"] for row in passing_withheld),
            default=None,
        ),
        "non_gpt_qwen_phi_withheld_pass_count": len(qwen_phi_withheld),
        "non_gpt_qwen_phi_withheld_min_top5": min(
            (row["top5"] for row in qwen_phi_withheld),
            default=None,
        ),
        "non_gpt_qwen_phi_withheld_max_entropy_diff": max(
            (row["entropy_diff"] for row in qwen_phi_withheld),
            default=None,
        ),
        "abi_vs_lora_frontier": {
            "row_count": len(finite_abi_lora_frontier),
            "pass_count": sum(1 for row in finite_abi_lora_frontier if row["pass"]),
            "heldout_lora_baseline": best_qwen_withheld_baseline,
            "best": best_abi_lora_frontier,
            "repeat": abi_lora_frontier_repeat,
            "best_beats_heldout_lora_all_metrics": best_beats_lora,
            "repeat_beats_heldout_lora_all_metrics": repeat_beats_lora,
            "repeat_passes_full_nib": bool(
                abi_lora_frontier_repeat and abi_lora_frontier_repeat["pass"]
            ),
            "rank_dominance_repeat_certified": bool(
                best_beats_lora and repeat_beats_lora
            ),
            "best_metric_margins_vs_heldout_lora": metric_margins_vs_lora(
                best_abi_lora_frontier,
                best_qwen_withheld_baseline,
            ),
            "repeat_metric_margins_vs_heldout_lora": metric_margins_vs_lora(
                abi_lora_frontier_repeat,
                best_qwen_withheld_baseline,
            ),
        },
        "phi_vs_lora_frontier": {
            "row_count": len(finite_phi_frontier),
            "pass_count": sum(1 for row in finite_phi_frontier if row["pass"]),
            "heldout_lora_baseline": best_phi_withheld_baseline,
            "best_abi_by_top5": best_phi_abi_by_top5,
            "best_abi_by_top1": best_phi_abi_by_top1,
            "best_all_metric_winner": best_phi_metric_winner,
            "repeat_for_best_all_metric_winner": phi_metric_winner_repeat,
            "abi_beats_phi_lora_all_metrics": bool(best_phi_metric_winner),
            "repeat_beats_phi_lora_all_metrics": phi_repeat_beats_lora,
            "all_metric_repeat_certified": bool(
                best_phi_metric_winner and phi_repeat_beats_lora
            ),
            "best_top5_margins_vs_lora": metric_margins_vs_lora(
                best_phi_abi_by_top5,
                best_phi_withheld_baseline,
            ),
            "best_all_metric_margins_vs_lora": metric_margins_vs_lora(
                best_phi_metric_winner,
                best_phi_withheld_baseline,
            ),
            "repeat_metric_margins_vs_lora": metric_margins_vs_lora(
                phi_metric_winner_repeat,
                best_phi_withheld_baseline,
            ),
            "best_top1_margins_vs_lora": metric_margins_vs_lora(
                best_phi_abi_by_top1,
                best_phi_withheld_baseline,
            ),
        },
        "flagship_gpt_style": flagship,
        "north_star_gates": north_star,
        "north_star_ready": north_star_ready,
        "north_star_status": (
            "repeat-certified oracle-light source-preservation gate passed"
            if north_star_ready
            else (
                "open: repeat-certified oracle-light source-preservation recipes "
                f"{north_star_repeat_count}/{north_star_required_count}"
            )
        ),
        "selective_transfer_gates": selective_transfer,
        "selective_transfer_ready": selective_ready,
        "selective_transfer_status": (
            "repeat-certified strict selective transfer gate passed"
            if selective_ready
            else (
                "open: repeat-certified strict selective transfer recipes "
                f"{selective_repeat_count}/{selective_required_count}; "
                "off-domain no-leakage evidence is required before claiming "
                "targeted lossless migration"
            )
        ),
        "atlas_pass_fraction": safe_ratio(
            summary["domain_breadth"]["atlas_domains_passing"],
            summary["domain_breadth"]["atlas_domains_total"],
        ),
        "adoption_gates": [
            "Extend the repeat-certified Phase-C-skipped base-reference bypass from the current GPT2-medium local target pairs to private-model-scale GPT5/GPT6-style evaluations before making GPT5-to-GPT6 migration claims.",
            "Require at least two repeat-certified oracle-light or base-bypass recipes with source-preservation measured, target-reference NIB pass, and independent seed or stream variants.",
            "Treat the GPT2-medium local target-pair repairs as near-lossless distribution plus source-surface repair results; extend beyond GPT2-medium donor and private-model-scale targets before using lossless wording.",
            "Extend the native-init/EMA repeat-certified LoRA/KD all-metric wins beyond the current Qwen2-1.5B -> Phi-3, GPT-Neo-125M -> Qwen2.5-0.5B, and Phi-3 -> Qwen2.5-0.5B WikiText recipes.",
            "Complete the matched baseline matrix against classic adapters, full/partial fine-tuning, and conventional distillation at equal trainable-parameter and wall-time budgets.",
            "Extend bidirectional rank/entropy WikiText certification to additional model pairs and domains.",
            "Extend Withheld-domain and no-leakage evaluations that are not tuned on the same corpus used to define the ABI bridge.",
            "Add selective-transfer audits that migrate one selected domain while preserving off-domain target behavior against the frozen target base/reference, with repeat-certified strict passes.",
            "Larger target families and private-model-style evals, including safety and refusal behavior, before making GPT5-to-GPT6 claims.",
        ],
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.4f}%"


def fmt_num(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def render_markdown(summary: dict) -> str:
    lines = [
        "# ABI Proof Layers",
        "",
        "Generated from local result JSON files by `build_proof_layers.py`.",
        "",
        "## Claim Boundary",
        "",
        f"- Supported: {summary['claim_boundary']['can_claim']}",
        f"- Not yet supported: {summary['claim_boundary']['cannot_claim_yet']}",
        "",
        "## Layer 1 - Savings",
        "",
        "Measured target-side calibration parameters are a small fraction of counted target model parameters where local weights are available.",
        "",
        "| Result | Target | Trainable | Target params | Trainable % | Frozen % | Time min | Top-5 | JS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["savings"]["rows"]:
        lines.append(
            "| `{file}` | {target} | {trainable:,} | {target_params} | {train_pct} | {frozen_pct} | {elapsed:.1f} | {top5:.4f} | {js:.5f} |".format(
                file=row["file"],
                target=row["target"],
                trainable=row["calibration_trainable_params"],
                target_params=(
                    f"{row['target_params_counted']:,}"
                    if row["target_params_counted"]
                    else "n/a"
                ),
                train_pct=fmt_pct(row["trainable_fraction_of_target"]),
                frozen_pct=fmt_pct(row["frozen_fraction_of_target"]),
                elapsed=row["elapsed_min"],
                top5=row["top5"],
                js=row["js"],
            )
        )
    lines.extend(
        [
            "",
            "## Layer 2 - Accuracy Frontier",
            "",
            f"Hard direction: {summary['accuracy_frontier']['hard_direction']}.",
            "",
            "| Result | D_ABI | Seed | Cal steps | Pass | Top-5 | Top-1 | JS | Entropy diff |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["accuracy_frontier"]["rows"]:
        lines.append(
            f"| `{row['file']}` | {row['d_abi']} | {row['seed']} | {row['calibration_steps']} | {row['pass']} | {fmt_num(row['top5'])} | {fmt_num(row['top1'])} | {fmt_num(row['js'])} | {fmt_num(row['entropy_diff'])} |"
        )
    lines.extend(
        [
            "",
            "## Layer 3 - Domain Breadth",
            "",
            "The atlas artifact certifies multiple domain charts. Generic cross-model WikiText runs are recorded separately because they track the entropy-calibration gap and the post-hoc scale repair.",
            "",
            "| Atlas domain | Pass | Top-5 | Top-1 | JS | Entropy diff | R^2 | KD steps |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, row in summary["domain_breadth"]["atlas_diagonal"].items():
        lines.append(
            f"| {domain} | {row['pass']} | {fmt_num(row['top5'])} | {fmt_num(row['top1'])} | {fmt_num(row['js'])} | {fmt_num(row['entropy_diff'])} | {fmt_num(row['r_squared'])} | {row['kd_steps']} |"
        )
    lines.extend(
        [
            "",
            "| Generic WikiText transfer | Pass | Top-5 | Top-1 | JS | Entropy diff | Notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["domain_breadth"]["generic_cross_model_wikitext"]:
        notes = "entropy calibration gap"
        if row["pass"] and row["posthoc_logit_scale"]["applied"]:
            notes = "post-hoc entropy-scale certificate"
        elif row["pass"]:
            notes = "certified"
        if math.isnan(row["js"]) or math.isnan(row["entropy_diff"]):
            notes = "unstable entropy-loss ablation"
        lines.append(
            f"| `{row['file']}` | {row['pass']} | {fmt_num(row['top5'])} | {fmt_num(row['top1'])} | {fmt_num(row['js'])} | {fmt_num(row['entropy_diff'])} | {notes} |"
        )
    lines.extend(
        [
            "",
            "## Layer 4 - Matched Baselines",
            "",
            "LoRA/KD comparator rows are target-side PEFT controls. They do not copy a frozen source domain core, but they test whether ordinary target-side adaptation reaches the same NIB certificate under a similar parameter budget.",
            "",
            "| Baseline result | Target | LoRA | Split | Trainable | Trainable % | Cal steps | Pass | Top-5 | Top-1 | JS | Entropy diff |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("baselines", {}).get("rows", []):
        lora = f"{row['lora_targets']} r={row['lora_rank']}"
        split = "same"
        if row.get("withheld_nib_eval"):
            split = (
                f"{row.get('wikitext_domain_split')}/"
                f"{row.get('wikitext_posthoc_split')}/"
                f"{row.get('wikitext_eval_split')}"
            )
        lines.append(
            f"| `{row['file']}` | {row['target']} | {lora} | {split} | {row['calibration_trainable_params']:,} | {fmt_pct(row['trainable_fraction_of_target'])} | {row['calibration_steps']} | {row['pass']} | {fmt_num(row['top5'])} | {fmt_num(row['top1'])} | {fmt_num(row['js'])} | {fmt_num(row['entropy_diff'])} |"
        )
    lines.extend(
        [
            "",
            "## Layer 4b - ABI vs LoRA Frontier",
            "",
            "These rows track the held-out GPT-Neo-125M -> Qwen2.5-0.5B ABI recipe against the split-separated all-linear r=3 LoRA/KD comparator.",
            "",
            "| ABI frontier result | D_ABI | Seed | Split | Trainable | Cal steps | Pass | Top-5 | Top-1 | JS | Entropy diff | Note |",
            "| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary.get("abi_lora_frontier", {}).get("rows", []):
        split = (
            f"{row['wikitext_domain_split']}/"
            f"{row['wikitext_posthoc_split']}/"
            f"{row['wikitext_eval_split']}"
        )
        lines.append(
            f"| `{row['file']}` | {row['d_abi']} | {row['seed']} | {split} | {row['calibration_trainable_params']:,} | {row['calibration_steps']} | {row['pass']} | {fmt_num(row['top5'])} | {fmt_num(row['top1'])} | {fmt_num(row['js'])} | {fmt_num(row['entropy_diff'])} | {row['comparison_note']} |"
        )
    lines.extend(
        [
            "",
            "## Layer 4c - Phi ABI vs LoRA Frontier",
            "",
            "These rows track Qwen-family -> Phi-3 ABI probes against the split-separated Phi-3 attention-LoRA comparator.",
            "",
            "| Phi ABI frontier result | D_ABI | Seed | Trainable | Cal steps | Pass | Top-5 | Top-1 | JS | Entropy diff | Note |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary.get("phi_abi_frontier", {}).get("rows", []):
        lines.append(
            f"| `{row['file']}` | {row['d_abi']} | {row['seed']} | {row['calibration_trainable_params']:,} | {row['calibration_steps']} | {row['pass']} | {fmt_num(row['top5'])} | {fmt_num(row['top1'])} | {fmt_num(row['js'])} | {fmt_num(row['entropy_diff'])} | {row['comparison_note']} |"
        )
    lines.extend(
        [
            "",
            "## Layer 5 - Withheld Evaluation",
            "",
            "Withheld rows separate source/native/calibration training, post-hoc scale selection, and final NIB evaluation onto distinct WikiText splits.",
            "",
            "| Withheld result | Splits train/posthoc/eval | Trainable | Cal steps | Pass | Top-5 | Top-1 | JS | Entropy diff |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("withheld_evaluation", {}).get("rows", []):
        splits = (
            f"{row['wikitext_domain_split']}/"
            f"{row['wikitext_posthoc_split']}/"
            f"{row['wikitext_eval_split']}"
        )
        lines.append(
            f"| `{row['file']}` | {splits} | {row['calibration_trainable_params']:,} | {row['calibration_steps']} | {row['pass']} | {fmt_num(row['top5'])} | {fmt_num(row['top1'])} | {fmt_num(row['js'])} | {fmt_num(row['entropy_diff'])} |"
        )
    north_star = summary.get("north_star_gates", {})
    selective = summary.get("selective_transfer_gates", {})
    lines.extend(
        [
            "",
            "## Layer 6 - North-Star Gates",
            "",
            "This layer tracks the evidence required before escalating from scoped ABI transfer to GPT5-to-GPT6-style domain migration claims.",
            "",
            (
                f"- Certificate-bearing ABI artifacts: "
                f"{north_star.get('row_count', 0)}."
            ),
            (
                f"- Oracle-light artifacts: "
                f"{north_star.get('oracle_light_row_count', 0)}."
            ),
            (
                f"- Source-preservation measured artifacts: "
                f"{north_star.get('source_preservation_row_count', 0)}."
            ),
            (
                f"- Joint oracle-light + source-preservation artifacts: "
                f"{north_star.get('joint_oracle_light_source_preservation_count', 0)}."
            ),
            (
                f"- Passing joint artifacts: "
                f"{north_star.get('joint_passing_count', 0)}."
            ),
            (
                f"- Repeat-certified joint recipes: "
                f"{north_star.get('repeat_joint_passing_recipe_count', 0)}/"
                f"{north_star.get('repeat_required_recipe_count', 2)} required."
            ),
            (
                f"- Repeat-certified model pairs: "
                f"{north_star.get('repeat_joint_passing_pair_count', 0)}."
            ),
            "",
            "| Result | Source -> Target | Oracle | Init | Seed/off | Pass | Top-5 | Top-1 | JS | Entropy | Src top1 in tgt top-k |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in north_star.get("rows", []):
        pair = f"{row['source']} -> {row['target']}"
        seed = f"{row['seed']}/{row['seed_offset']}"
        lines.append(
            f"| `{row['file']}` | {pair} | {row['oracle_mode']} | {row['calibration_init']} | {seed} | {row['nib_pass']} | {fmt_num(row['top5'])} | {fmt_num(row['top1'])} | {fmt_num(row['js'])} | {fmt_num(row['entropy_diff'])} | {fmt_num(row['source_preservation_top1_in_topk'])} |"
        )
    lines.extend(
        [
            "",
            "### Selective Transfer / Off-Domain No-Leakage",
            "",
            (
                f"- Selective-transfer audit artifacts: "
                f"{selective.get('row_count', 0)}."
            ),
            (
                f"- Ordinary selected-domain + off-domain no-leakage passes: "
                f"{selective.get('ordinary_pass_count', 0)}."
            ),
            (
                f"- Strict selected-domain passes: "
                f"{selective.get('target_domain_strict_pass_count', 0)}."
            ),
            (
                f"- Strict off-domain no-leakage passes: "
                f"{selective.get('off_domain_strict_pass_count', 0)}."
            ),
            (
                f"- Strict selective-transfer passes: "
                f"{selective.get('strict_selective_transfer_pass_count', 0)}."
            ),
            (
                f"- Repeat-certified strict selective recipes: "
                f"{selective.get('repeat_strict_selective_recipe_count', 0)}/"
                f"{selective.get('repeat_required_recipe_count', SELECTIVE_TRANSFER_REQUIRED_REPEAT_RECIPES)} "
                "required."
            ),
            f"- Selective-transfer ready: {selective.get('ready', False)}.",
        ]
    )
    for blocker in selective.get("open_blockers", []):
        lines.append(f"- Selective-transfer blocker: {blocker}")
    if selective.get("rows"):
        lines.extend(
            [
                "",
                "| Selective audit | Domain -> off-domain | Seed/off | Strict selected | Strict off-domain | Top-5/off top-5 | JS/off JS |",
                "| --- | --- | --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in selective["rows"]:
            seed = f"{row['seed']}/{row['seed_offset']}"
            lines.append(
                f"| `{row['file']}` | {row['domain_corpus']} -> {row['off_domain_corpus']} | {seed} | {row['target_domain_strict_pass']} | {row['off_domain_strict_pass']} | {fmt_num(row['target_top5'])}/{fmt_num(row['off_domain_top5'])} | {fmt_num(row['target_js'])}/{fmt_num(row['off_domain_js'])} |"
            )
    flagship = summary.get("flagship_gpt_style", {})
    thresholds = flagship.get("near_lossless_distribution_thresholds", {})
    completion_thresholds = flagship.get("completion_preservation_thresholds", {})
    lines.extend(
        [
            "",
            "## Layer 7 - Flagship GPT-Style Near-Lossless Scenario",
            "",
            "This layer narrows the north-star question to a single GPT-style donor and a larger successor-style target: `gpt2-medium -> microsoft/phi-3-mini-4k-instruct`.",
            "",
            (
                f"- Strict distribution thresholds: top-5 >= "
                f"{fmt_num(thresholds.get('top5'))}, top-1 >= "
                f"{fmt_num(thresholds.get('top1'))}, JS <= "
                f"{fmt_num(thresholds.get('js'))}, entropy diff <= "
                f"{fmt_num(thresholds.get('entropy_diff'))}."
            ),
            (
                f"- Cross-tokenizer completion threshold: source top-1 preferred "
                f">= {fmt_num(completion_thresholds.get('source_top1_completion_preferred'))} "
                f"among source top-k continuations."
            ),
            (
                f"- Near-lossless distribution passes: "
                f"{flagship.get('distribution_near_lossless_pass_count', 0)}."
            ),
            (
                f"- Oracle-light near-lossless distribution passes: "
                f"{flagship.get('oracle_light_distribution_pass_count', 0)}."
            ),
            (
                f"- Repeat-certified flagship recipes: "
                f"{flagship.get('repeat_distribution_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)} required."
            ),
            (
                f"- Repeat-certified strict distribution + completion recipes: "
                f"{flagship.get('repeat_distribution_completion_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)} required."
            ),
            (
                f"- Native target-interface strict+completion passes: "
                f"{flagship.get('native_target_interface_strict_completion_pass_count', 0)}."
            ),
            (
                f"- Xavier target-interface strict+completion passes: "
                f"{flagship.get('xavier_target_interface_strict_completion_pass_count', 0)} "
                f"(best top-5 {fmt_num(flagship.get('xavier_target_interface_best_top5'))})."
            ),
            (
                f"- Zero-out target-interface strict+completion passes: "
                f"{flagship.get('zeroout_target_interface_strict_completion_pass_count', 0)} "
                f"(repeat-certified recipes "
                f"{flagship.get('zeroout_target_interface_repeat_strict_completion_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)}, "
                f"best top-5 {fmt_num(flagship.get('zeroout_target_interface_best_top5'))}, "
                f"best completion "
                f"{fmt_num(flagship.get('zeroout_target_interface_best_completion'))})."
            ),
            (
                f"- Reusable target-interface cache: saved "
                f"{flagship.get('target_interface_cache_saved_count', 0)}, loaded "
                f"{flagship.get('target_interface_cache_loaded_count', 0)}; "
                f"loaded strict-distribution passes "
                f"{flagship.get('target_interface_cache_loaded_distribution_pass_count', 0)}, "
                f"loaded strict+completion passes "
                f"{flagship.get('target_interface_cache_loaded_strict_completion_pass_count', 0)} "
                f"(best top-5 "
                f"{fmt_num(flagship.get('target_interface_cache_loaded_best_top5'))}, "
                f"best completion "
                f"{fmt_num(flagship.get('target_interface_cache_loaded_best_completion'))})."
            ),
            (
                f"- Cache-load + source-completion-loss strict+completion passes: "
                f"{flagship.get('target_interface_cache_loaded_source_completion_loss_strict_completion_pass_count', 0)}/"
                f"{flagship.get('target_interface_cache_loaded_source_completion_loss_count', 0)}; "
                f"repeat-certified recipes "
                f"{flagship.get('target_interface_cache_loaded_source_completion_loss_repeat_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)} required "
                f"(best completion "
                f"{fmt_num(flagship.get('target_interface_cache_loaded_source_completion_loss_best_completion'))})."
            ),
            (
                f"- Phase-C-skipped base-reference bypass strict+completion passes: "
                f"{flagship.get('base_reference_bypass_strict_completion_pass_count', 0)} "
                f"(repeat-certified recipes "
                f"{flagship.get('base_reference_bypass_repeat_strict_completion_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)}, "
                f"best top-5 "
                f"{fmt_num(flagship.get('base_reference_bypass_best_top5'))}, "
                f"best completion "
                f"{fmt_num(flagship.get('base_reference_bypass_best_completion'))})."
            ),
            (
                f"- Non-fixed posthoc base-bypass strict+completion passes: "
                f"{flagship.get('base_reference_bypass_nonfixed_strict_completion_pass_count', 0)} "
                f"(repeat-certified recipes "
                f"{flagship.get('base_reference_bypass_nonfixed_repeat_strict_completion_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)}, "
                f"repeat-certified domains "
                f"{flagship.get('base_reference_bypass_nonfixed_repeat_strict_completion_domain_count', 0)}, "
                f"best top-5 "
                f"{fmt_num(flagship.get('base_reference_bypass_nonfixed_best_top5'))}, "
                f"best completion "
                f"{fmt_num(flagship.get('base_reference_bypass_nonfixed_best_completion'))})."
            ),
            (
                f"- Cross-target GPT2-medium base-bypass strict+completion passes: "
                f"{flagship.get('base_reference_bypass_cross_target_strict_completion_pass_count', 0)} "
                f"(repeat-certified recipes "
                f"{flagship.get('base_reference_bypass_cross_target_repeat_strict_completion_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)}, "
                f"repeat-certified target pairs "
                f"{flagship.get('base_reference_bypass_cross_target_repeat_strict_completion_pair_count', 0)}, "
                f"best top-5 "
                f"{fmt_num(flagship.get('base_reference_bypass_cross_target_best_top5'))}, "
                f"best completion "
                f"{fmt_num(flagship.get('base_reference_bypass_cross_target_best_completion'))})."
            ),
            (
                f"- Margin-hardened source-surface repair passes: "
                f"{flagship.get('base_reference_bypass_cross_target_source_surface_repair_pass_count', 0)} "
                f"(repeat-certified recipes "
                f"{flagship.get('base_reference_bypass_cross_target_source_surface_repair_repeat_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)}, "
                f"repeat-certified target pairs "
                f"{flagship.get('base_reference_bypass_cross_target_source_surface_repair_repeat_pair_count', 0)}, "
                f"best top-1 surface "
                f"{fmt_num(flagship.get('base_reference_bypass_cross_target_best_source_top1_surface'))}, "
                f"min repaired top-1 in target top-k "
                f"{fmt_num(flagship.get('base_reference_bypass_cross_target_min_repaired_source_top1_in_topk'))}, "
                f"min repaired completion "
                f"{fmt_num(flagship.get('base_reference_bypass_cross_target_min_repaired_completion'))})."
            ),
            (
                f"- Cross-donor base-bypass source-surface repair passes: "
                f"{flagship.get('base_reference_bypass_source_surface_repair_pass_count', 0)} "
                f"(repeat-certified recipes "
                f"{flagship.get('base_reference_bypass_source_surface_repair_repeat_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)}, "
                f"repeat-certified model pairs "
                f"{flagship.get('base_reference_bypass_source_surface_repair_repeat_pair_count', 0)}, "
                f"repeat-certified source models "
                f"{flagship.get('base_reference_bypass_source_surface_repair_repeat_source_count', 0)}, "
                f"NLL-repair passes "
                f"{flagship.get('base_reference_bypass_source_surface_nll_repair_pass_count', 0)}, "
                f"best top-1 surface "
                f"{fmt_num(flagship.get('base_reference_bypass_source_surface_best_top1_surface'))}, "
                f"min repaired top-1 in target top-k "
                f"{fmt_num(flagship.get('base_reference_bypass_source_surface_min_repaired_top1_in_topk'))}, "
                f"min repaired completion "
                f"{fmt_num(flagship.get('base_reference_bypass_source_surface_min_repaired_completion'))})."
            ),
            (
                f"- Hard-recipient ordinary NIB + source-surface passes: "
                f"{flagship.get('hard_recipient_ordinary_source_surface_pass_count', 0)} "
                f"(repeat-certified recipes "
                f"{flagship.get('hard_recipient_ordinary_source_surface_repeat_recipe_count', 0)}/"
                f"{flagship.get('repeat_required_recipe_count', 1)}, "
                f"repeat-certified pairs "
                f"{flagship.get('hard_recipient_ordinary_source_surface_repeat_pair_count', 0)}, "
                f"best top-5 "
                f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_best_top5'))}, "
                f"best top-1 "
                f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_best_top1'))}, "
                f"best JS "
                f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_best_js'))}, "
                f"best entropy diff "
                f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_best_entropy_diff'))}, "
                f"min top-5 "
                f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_min_top5'))}, "
                f"min top-1 surface "
                f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_min_top1_surface'))}, "
                f"min top-1 in target top-k "
                f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_min_top1_in_topk'))}, "
                f"min completion "
                f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_min_completion'))})."
            ),
            (
                f"- Base-reference xavier negative controls: "
                f"{flagship.get('base_reference_negative_control_count', 0)} "
                f"(best failed top-5 "
                f"{fmt_num(flagship.get('base_reference_negative_control_best_top5'))})."
            ),
            (
                f"- Source-token surface preservation, top-1 in target top-k: "
                f"{fmt_num(flagship.get('source_surface_min_top1_in_topk'))}-"
                f"{fmt_num(flagship.get('source_surface_max_top1_in_topk'))}."
            ),
            (
                f"- Cross-tokenizer source top-1 continuation preferred: "
                f"{fmt_num(flagship.get('source_completion_min_preferred'))}-"
                f"{fmt_num(flagship.get('source_completion_max_preferred'))}."
            ),
            (
                f"- Production readiness: "
                f"{flagship.get('production_readiness_status', 'unknown')}."
            ),
            "",
        ]
    )
    for blocker in flagship.get("production_blockers", []):
        lines.append(f"- Production blocker: {blocker}")
    if flagship.get("production_blockers"):
        lines.append("")
    lines.extend(
        [
            "| Result | Oracle | Ref bypass | Ref fwd | Init | Cal | Seed/off | Pass | Strict dist | Strict+completion | Top-5 | Top-1 | JS | Entropy | Src top1 in tgt top-k | Src completion pref |",
            "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in flagship.get("rows", []):
        seed = f"{row['seed']}/{row['seed_offset']}"
        lines.append(
            f"| `{row['file']}` | {row['oracle_mode']} | {row.get('target_reference_bypass_abi', False)} | {row.get('target_reference_forward_mode') or ''} | {row['calibration_init']} | {row['calibration_steps']} | {seed} | {row['nib_pass']} | {row['distribution_near_lossless_pass']} | {row['distribution_completion_pass']} | {fmt_num(row['top5'])} | {fmt_num(row['top1'])} | {fmt_num(row['js'])} | {fmt_num(row['entropy_diff'])} | {fmt_num(row['source_preservation_top1_in_topk'])} | {fmt_num(row['source_completion_preferred'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_adoption_case(summary: dict) -> str:
    case = summary["adoption_case"]
    best = case["best_hard_direction"]
    wikitext = case["best_generic_wikitext_rank_transfer"]
    baseline = case.get("best_matched_baseline")
    withheld_baseline = case.get("best_matched_withheld_baseline")
    withheld = case.get("best_withheld_eval")
    abi_lora = case.get("abi_vs_lora_frontier", {})
    abi_lora_best = abi_lora.get("best")
    abi_lora_repeat = abi_lora.get("repeat")
    phi_lora = case.get("phi_vs_lora_frontier", {})
    phi_lora_baseline = phi_lora.get("heldout_lora_baseline")
    phi_best_top5 = phi_lora.get("best_abi_by_top5")
    phi_best_top1 = phi_lora.get("best_abi_by_top1")
    north_star = case.get("north_star_gates", {})
    selective = case.get("selective_transfer_gates", {})
    flagship = case.get("flagship_gpt_style", {})
    flagship_distribution_rows = [
        row
        for row in flagship.get("rows", [])
        if row.get("distribution_near_lossless_pass")
    ]
    flagship_distribution_completion_rows = [
        row
        for row in flagship.get("rows", [])
        if row.get("distribution_completion_pass")
    ]
    flagship_base_reference_failures = [
        row
        for row in flagship.get("rows", [])
        if row.get("oracle_mode") == "base_target_reference"
        and row.get("nib_pass") is False
    ]
    posthoc_scale = wikitext["posthoc_logit_scale"]
    gates = case["adoption_gates"]
    large_mean = case["large_target_mean_trainable_fraction"]
    large_max = case["large_target_max_trainable_fraction"]
    large_frozen_min = case["large_target_min_frozen_fraction"]
    repeat_pair_names = ", ".join(
        pair["pair"]
        for pair in case["bidirectional_wikitext_pairs"]
        if pair["repeat_certified"]
    )

    lines = [
        "# ABI Adoption Case",
        "",
        "Generated from local result JSON files by `build_proof_layers.py`.",
        "",
        "## Decision Claim",
        "",
        case["decision_claim"],
        "",
        "## Measured Compute Story",
        "",
        (
            f"- All {len(summary['savings']['rows'])} savings-layer transfer rows pass "
            f"the current NIB thresholds: {case['all_savings_rows_pass']}."
        ),
        (
            f"- On local targets >= {case['local_large_target_threshold_params']:,} "
            f"parameters, ABI calibrates an average of {fmt_pct(large_mean)} and at "
            f"most {fmt_pct(large_max)} of target parameters."
        ),
        (
            f"- The least-frozen local target in that group is still "
            f"{fmt_pct(large_frozen_min)} frozen."
        ),
        (
            f"- The hard Pythia-410M -> DeepSeek-1.3B run calibrates "
            f"{best['calibration_steps']:,} steps in {best['elapsed_min']:.1f} "
            f"minutes with top-5 {fmt_num(best['top5'])}, top-1 "
            f"{fmt_num(best['top1'])}, and JS {fmt_num(best['js'])}."
        ),
        "",
        "## Measured Accuracy Cost",
        "",
        (
            f"- Mean calibrated-target perplexity overhead across savings rows is "
            f"{fmt_pct(case['mean_ppl_relative_overhead'])}; worst measured overhead "
            f"is {fmt_pct(case['max_ppl_relative_overhead'])}."
        ),
        (
            f"- The hard direction improves from top-5 0.8679 at 7,200 calibration "
            f"steps to {fmt_num(best['top5'])} at {best['calibration_steps']:,} "
            f"steps, with a shifted-seed repeat still passing at top-5 0.8776."
        ),
        (
            "- This is not yet lossless. The current case is efficient, certified "
            "scoped transfer with measurable quality cost."
        ),
        "",
        "## Domain Breadth",
        "",
        (
            f"- The multi-domain atlas passes "
            f"{summary['domain_breadth']['atlas_domains_passing']}/"
            f"{summary['domain_breadth']['atlas_domains_total']} diagonal domains "
            f"(pass fraction {fmt_pct(case['atlas_pass_fraction'])})."
        ),
        (
            f"- Generic cross-model WikiText now reaches top-5 "
            f"{fmt_num(wikitext['top5'])}, JS {fmt_num(wikitext['js'])}, and "
            f"entropy diff {fmt_num(wikitext['entropy_diff'])} with pass="
            f"{wikitext['pass']}."
        ),
        (
            f"- The passing WikiText run uses post-hoc logit scale "
            f"{fmt_num(posthoc_scale.get('scale', 1.0))} in "
            f"{posthoc_scale.get('mode', 'none')} mode. Across "
            f"{case['posthoc_wikitext_pass_count']} post-hoc WikiText passes, "
            f"minimum top-5 is {fmt_num(case['posthoc_wikitext_min_top5'])} and "
            f"maximum entropy diff is "
            f"{fmt_num(case['posthoc_wikitext_max_entropy_diff'])}."
        ),
        (
            f"- Bidirectional WikiText is repeat-certified for "
            f"{case['repeat_certified_bidirectional_wikitext_pair_count']} "
            f"model pairs: {repeat_pair_names}."
        ),
        "",
        "## No-Leakage Certificate",
        "",
    ]
    if withheld is None:
        lines.extend(
            [
                "- No withheld-split ABI certificate has been generated yet.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"- Withheld WikiText artifacts: {case['withheld_eval_row_count']}; "
                    f"passing NIB: {case['withheld_eval_pass_count']}; "
                    f"split-separated: {case['withheld_eval_all_split_separated']}."
                ),
                (
                    f"- Best withheld run is `{withheld['file']}` with "
                    f"train/posthoc/eval splits "
                    f"{withheld['wikitext_domain_split']}/"
                    f"{withheld['wikitext_posthoc_split']}/"
                    f"{withheld['wikitext_eval_split']}: top-5 "
                    f"{fmt_num(withheld['top5'])}, JS {fmt_num(withheld['js'])}, "
                    f"entropy diff {fmt_num(withheld['entropy_diff'])}, "
                    f"pass={withheld['pass']}."
                ),
                (
                    f"- Direct non-GPT Qwen2.5 <-> Phi-3 withheld passes: "
                    f"{case['non_gpt_qwen_phi_withheld_pass_count']}; minimum "
                    f"top-5 {fmt_num(case['non_gpt_qwen_phi_withheld_min_top5'])}; "
                    f"maximum entropy diff "
                    f"{fmt_num(case['non_gpt_qwen_phi_withheld_max_entropy_diff'])}."
                ),
                "",
            ]
        )
    lines.extend(
        [
        "## Matched Baseline Check",
        "",
        ]
    )
    if baseline is None:
        lines.extend(
            [
                "- No matched LoRA/KD comparator artifact has been generated yet.",
                "- This remains the first adoption gate because it tests whether ABI is stronger than ordinary target-side PEFT under the same NIB certificate.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"- Current matched baseline artifacts: "
                    f"{case['matched_baseline_row_count']}; passing NIB: "
                    f"{case['matched_baseline_pass_count']}."
                ),
                (
                    f"- Best LoRA/KD baseline is `{baseline['file']}` on "
                    f"{baseline['target']} with {baseline['calibration_trainable_params']:,} "
                    f"trainable parameters ({fmt_pct(baseline['trainable_fraction_of_target'])}), "
                    f"top-5 {fmt_num(baseline['top5'])}, JS {fmt_num(baseline['js'])}, "
                    f"entropy diff {fmt_num(baseline['entropy_diff'])}, pass={baseline['pass']}."
                ),
                (
                    "- This is a target-side adaptation control, not source-core migration; "
                    "it strengthens the adoption case only when interpreted against the ABI "
                    "portability rows above."
                ),
            "",
            ]
        )
    if withheld_baseline is not None:
        split = (
            f"{withheld_baseline.get('wikitext_domain_split')}/"
            f"{withheld_baseline.get('wikitext_posthoc_split')}/"
            f"{withheld_baseline.get('wikitext_eval_split')}"
        )
        lines.extend(
            [
                (
                    f"- Split-separated LoRA/KD baselines: "
                    f"{case['matched_withheld_baseline_row_count']}; passing NIB: "
                    f"{case['matched_withheld_baseline_pass_count']}."
                ),
                (
                    f"- Best split-separated LoRA/KD baseline is "
                    f"`{withheld_baseline['file']}` with {split} splits, "
                    f"top-5 {fmt_num(withheld_baseline['top5'])}, JS "
                    f"{fmt_num(withheld_baseline['js'])}, entropy diff "
                    f"{fmt_num(withheld_baseline['entropy_diff'])}, "
                    f"pass={withheld_baseline['pass']}."
                ),
                "",
            ]
        )
    if abi_lora_best is not None and withheld_baseline is not None:
        lines.extend(
            [
                "## ABI vs LoRA Frontier",
                "",
                (
                    f"- ABI frontier artifacts: {abi_lora['row_count']}; passing "
                    f"NIB: {abi_lora['pass_count']}."
                ),
                (
                    f"- Best held-out ABI frontier is `{abi_lora_best['file']}` "
                    f"with {abi_lora_best['calibration_trainable_params']:,} "
                    f"trainable parameters and "
                    f"{abi_lora_best['calibration_steps']} calibration steps, "
                    f"top-5 {fmt_num(abi_lora_best['top5'])}, "
                    f"top-1 {fmt_num(abi_lora_best['top1'])}, JS "
                    f"{fmt_num(abi_lora_best['js'])}, entropy diff "
                    f"{fmt_num(abi_lora_best['entropy_diff'])}, pass="
                    f"{abi_lora_best['pass']}."
                ),
                (
                    f"- Against the split-separated LoRA/KD baseline, that best "
                    f"ABI run beats top-5, top-1, JS, and entropy together: "
                    f"{abi_lora['best_beats_heldout_lora_all_metrics']}."
                ),
            ]
        )
        if abi_lora_repeat is not None:
            lines.extend(
                [
                    (
                        f"- The same D_ABI/calibration recipe under an independent "
                        f"seed/stream is "
                        f"`{abi_lora_repeat['file']}`: top-5 "
                        f"{fmt_num(abi_lora_repeat['top5'])}, top-1 "
                        f"{fmt_num(abi_lora_repeat['top1'])}, JS "
                        f"{fmt_num(abi_lora_repeat['js'])}, entropy diff "
                        f"{fmt_num(abi_lora_repeat['entropy_diff'])}, "
                        f"calibration steps "
                        f"{abi_lora_repeat['calibration_steps']}, pass="
                        f"{abi_lora_repeat['pass']}."
                    ),
                    (
                        f"- Full-NIB repeat passes: "
                        f"{abi_lora['repeat_passes_full_nib']}; LoRA rank "
                        f"dominance repeat-certified: "
                        f"{abi_lora['rank_dominance_repeat_certified']}."
                    ),
                ]
            )
        lines.append("")
    if phi_lora_baseline is not None and phi_best_top5 is not None:
        completed = phi_lora_baseline.get("completed_calibration_steps")
        requested = phi_lora_baseline.get("requested_calibration_steps")
        lines.extend(
            [
                "## Phi ABI vs LoRA Frontier",
                "",
                (
                    f"- Phi-3 LoRA/KD baseline `{phi_lora_baseline['file']}` "
                    f"uses {phi_lora_baseline['calibration_trainable_params']:,} "
                    f"trainable parameters and completed {completed}/"
                    f"{requested} requested LoRA steps under a "
                    f"{fmt_num(phi_lora_baseline.get('max_train_seconds'))}s "
                    f"train cap; pass={phi_lora_baseline['pass']}."
                ),
                (
                    f"- Best Qwen -> Phi ABI by top-5 is "
                    f"`{phi_best_top5['file']}`: top-5 "
                    f"{fmt_num(phi_best_top5['top5'])}, top-1 "
                    f"{fmt_num(phi_best_top5['top1'])}, JS "
                    f"{fmt_num(phi_best_top5['js'])}, entropy diff "
                    f"{fmt_num(phi_best_top5['entropy_diff'])}, pass="
                    f"{phi_best_top5['pass']}."
                ),
            ]
        )
        if phi_best_top1 is not None and phi_best_top1["file"] != phi_best_top5["file"]:
            lines.append(
                f"- Best Qwen -> Phi ABI by top-1 is `{phi_best_top1['file']}` "
                f"with top-1 {fmt_num(phi_best_top1['top1'])}."
            )
        lines.extend(
            [
                (
                    f"- ABI beats the Phi LoRA baseline on every reported metric: "
                    f"{phi_lora['abi_beats_phi_lora_all_metrics']}."
                ),
                (
                    f"- Phi all-metric repeat-certified: "
                    f"{phi_lora['all_metric_repeat_certified']}."
                ),
                (
                    "- Current Phi status: Qwen2-1.5B -> Phi ABI now has a "
                    "same-recipe repeat-certified all-metric win over the "
                    "full-step Phi LoRA comparator on WikiText. This is a "
                    "flagship scoped result, not yet a production-readiness or "
                    "universal/lossless transfer claim."
                ),
                "",
            ]
        )
    if flagship.get("row_count", 0):
        thresholds = flagship.get("near_lossless_distribution_thresholds", {})
        completion_thresholds = flagship.get("completion_preservation_thresholds", {})
        lines.extend(
            [
                "## Flagship GPT-Style Scenario",
                "",
                (
                    f"- Scenario: {flagship.get('scenario')}; artifacts: "
                    f"{flagship.get('row_count', 0)}."
                ),
                (
                    f"- Near-lossless distribution threshold: top-5 >= "
                    f"{fmt_num(thresholds.get('top5'))}, top-1 >= "
                    f"{fmt_num(thresholds.get('top1'))}, JS <= "
                    f"{fmt_num(thresholds.get('js'))}, entropy diff <= "
                    f"{fmt_num(thresholds.get('entropy_diff'))}."
                ),
                (
                    f"- Cross-tokenizer completion threshold: source top-1 "
                    f"preferred >= "
                    f"{fmt_num(completion_thresholds.get('source_top1_completion_preferred'))} "
                    f"among source top-k continuations."
                ),
                (
                    f"- Oracle-light near-lossless distribution passes: "
                    f"{flagship.get('oracle_light_distribution_pass_count', 0)}; "
                    f"repeat-certified recipes: "
                    f"{flagship.get('repeat_distribution_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)}."
                ),
                (
                    f"- Repeat-certified strict distribution + completion recipes: "
                    f"{flagship.get('repeat_distribution_completion_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)}."
                ),
                (
                    f"- Native target-interface strict+completion passes: "
                    f"{flagship.get('native_target_interface_strict_completion_pass_count', 0)}; "
                    f"xavier target-interface strict+completion passes: "
                    f"{flagship.get('xavier_target_interface_strict_completion_pass_count', 0)} "
                    f"(best top-5 "
                    f"{fmt_num(flagship.get('xavier_target_interface_best_top5'))})."
                ),
                (
                    f"- Zero-out target-interface strict+completion passes: "
                    f"{flagship.get('zeroout_target_interface_strict_completion_pass_count', 0)} "
                    f"(repeat-certified recipes "
                    f"{flagship.get('zeroout_target_interface_repeat_strict_completion_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)}, "
                    f"best top-5 {fmt_num(flagship.get('zeroout_target_interface_best_top5'))}, "
                    f"best completion "
                    f"{fmt_num(flagship.get('zeroout_target_interface_best_completion'))})."
                ),
                (
                    f"- Reusable target-interface cache: saved "
                    f"{flagship.get('target_interface_cache_saved_count', 0)}, loaded "
                    f"{flagship.get('target_interface_cache_loaded_count', 0)}; "
                    f"loaded strict-distribution passes "
                    f"{flagship.get('target_interface_cache_loaded_distribution_pass_count', 0)}, "
                    f"loaded strict+completion passes "
                    f"{flagship.get('target_interface_cache_loaded_strict_completion_pass_count', 0)} "
                    f"(best top-5 "
                    f"{fmt_num(flagship.get('target_interface_cache_loaded_best_top5'))}, "
                    f"best completion "
                    f"{fmt_num(flagship.get('target_interface_cache_loaded_best_completion'))})."
                ),
                (
                    f"- Cache-load + source-completion-loss strict+completion passes: "
                    f"{flagship.get('target_interface_cache_loaded_source_completion_loss_strict_completion_pass_count', 0)}/"
                    f"{flagship.get('target_interface_cache_loaded_source_completion_loss_count', 0)}; "
                    f"repeat-certified recipes "
                    f"{flagship.get('target_interface_cache_loaded_source_completion_loss_repeat_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)} required "
                    f"(best completion "
                    f"{fmt_num(flagship.get('target_interface_cache_loaded_source_completion_loss_best_completion'))})."
                ),
                (
                    f"- Phase-C-skipped base-reference bypass strict+completion passes: "
                    f"{flagship.get('base_reference_bypass_strict_completion_pass_count', 0)} "
                    f"(repeat-certified recipes "
                    f"{flagship.get('base_reference_bypass_repeat_strict_completion_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)}, "
                    f"best top-5 "
                    f"{fmt_num(flagship.get('base_reference_bypass_best_top5'))}, "
                    f"best completion "
                    f"{fmt_num(flagship.get('base_reference_bypass_best_completion'))})."
                ),
                (
                    f"- Non-fixed posthoc base-bypass strict+completion passes: "
                    f"{flagship.get('base_reference_bypass_nonfixed_strict_completion_pass_count', 0)} "
                    f"(repeat-certified recipes "
                    f"{flagship.get('base_reference_bypass_nonfixed_repeat_strict_completion_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)}, "
                    f"repeat-certified domains "
                    f"{flagship.get('base_reference_bypass_nonfixed_repeat_strict_completion_domain_count', 0)}, "
                    f"best top-5 "
                    f"{fmt_num(flagship.get('base_reference_bypass_nonfixed_best_top5'))}, "
                    f"best completion "
                    f"{fmt_num(flagship.get('base_reference_bypass_nonfixed_best_completion'))})."
                ),
                (
                    f"- Cross-target GPT2-medium base-bypass strict+completion passes: "
                    f"{flagship.get('base_reference_bypass_cross_target_strict_completion_pass_count', 0)} "
                    f"(repeat-certified recipes "
                    f"{flagship.get('base_reference_bypass_cross_target_repeat_strict_completion_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)}, "
                    f"repeat-certified target pairs "
                    f"{flagship.get('base_reference_bypass_cross_target_repeat_strict_completion_pair_count', 0)}, "
                    f"best top-5 "
                    f"{fmt_num(flagship.get('base_reference_bypass_cross_target_best_top5'))}, "
                    f"best completion "
                    f"{fmt_num(flagship.get('base_reference_bypass_cross_target_best_completion'))})."
                ),
                (
                    f"- Margin-hardened source-surface repair passes: "
                    f"{flagship.get('base_reference_bypass_cross_target_source_surface_repair_pass_count', 0)} "
                    f"(repeat-certified recipes "
                    f"{flagship.get('base_reference_bypass_cross_target_source_surface_repair_repeat_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)}, "
                    f"repeat-certified target pairs "
                    f"{flagship.get('base_reference_bypass_cross_target_source_surface_repair_repeat_pair_count', 0)}, "
                    f"best top-1 surface "
                    f"{fmt_num(flagship.get('base_reference_bypass_cross_target_best_source_top1_surface'))}, "
                    f"min repaired top-1 in target top-k "
                    f"{fmt_num(flagship.get('base_reference_bypass_cross_target_min_repaired_source_top1_in_topk'))}, "
                    f"min repaired completion "
                    f"{fmt_num(flagship.get('base_reference_bypass_cross_target_min_repaired_completion'))})."
                ),
                (
                    f"- Cross-donor base-bypass source-surface repair passes: "
                    f"{flagship.get('base_reference_bypass_source_surface_repair_pass_count', 0)} "
                    f"(repeat-certified recipes "
                    f"{flagship.get('base_reference_bypass_source_surface_repair_repeat_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)}, "
                    f"repeat-certified model pairs "
                    f"{flagship.get('base_reference_bypass_source_surface_repair_repeat_pair_count', 0)}, "
                    f"repeat-certified source models "
                    f"{flagship.get('base_reference_bypass_source_surface_repair_repeat_source_count', 0)}, "
                    f"NLL-repair passes "
                    f"{flagship.get('base_reference_bypass_source_surface_nll_repair_pass_count', 0)}, "
                    f"best top-1 surface "
                    f"{fmt_num(flagship.get('base_reference_bypass_source_surface_best_top1_surface'))}, "
                    f"min repaired top-1 in target top-k "
                    f"{fmt_num(flagship.get('base_reference_bypass_source_surface_min_repaired_top1_in_topk'))}, "
                    f"min repaired completion "
                    f"{fmt_num(flagship.get('base_reference_bypass_source_surface_min_repaired_completion'))})."
                ),
                (
                    f"- Hard-recipient ordinary NIB + source-surface passes: "
                    f"{flagship.get('hard_recipient_ordinary_source_surface_pass_count', 0)} "
                    f"(repeat-certified recipes "
                    f"{flagship.get('hard_recipient_ordinary_source_surface_repeat_recipe_count', 0)}/"
                    f"{flagship.get('repeat_required_recipe_count', 1)}, "
                    f"repeat-certified pairs "
                    f"{flagship.get('hard_recipient_ordinary_source_surface_repeat_pair_count', 0)}, "
                    f"best top-5 "
                    f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_best_top5'))}, "
                    f"best top-1 "
                    f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_best_top1'))}, "
                    f"best JS "
                    f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_best_js'))}, "
                    f"best entropy diff "
                    f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_best_entropy_diff'))}, "
                    f"min top-5 "
                    f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_min_top5'))}, "
                    f"min top-1 surface "
                    f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_min_top1_surface'))}, "
                    f"min top-1 in target top-k "
                    f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_min_top1_in_topk'))}, "
                    f"min completion "
                    f"{fmt_num(flagship.get('hard_recipient_ordinary_source_surface_min_completion'))})."
                ),
                (
                    f"- Base-reference xavier negative controls: "
                    f"{flagship.get('base_reference_negative_control_count', 0)} "
                    f"(best failed top-5 "
                    f"{fmt_num(flagship.get('base_reference_negative_control_best_top5'))})."
                ),
                (
                    f"- Production readiness: "
                    f"{flagship.get('production_readiness_status', 'unknown')}."
                ),
            ]
        )
        for blocker in flagship.get("production_blockers", []):
            lines.append(f"- Production blocker: {blocker}")
        for row in flagship_distribution_completion_rows or flagship_distribution_rows:
            lines.append(
                f"- Passing repeat `{row['file']}`: top-5 "
                f"{fmt_num(row['top5'])}, top-1 {fmt_num(row['top1'])}, "
                f"JS {fmt_num(row['js'])}, entropy diff "
                f"{fmt_num(row['entropy_diff'])}, source top-1 in target top-k "
                f"{fmt_num(row['source_preservation_top1_in_topk'])}, "
                f"cross-tokenizer source top-1 preferred "
                f"{fmt_num(row['source_completion_preferred'])}."
            )
        if flagship_base_reference_failures:
            best_failure = max(
                flagship_base_reference_failures,
                key=lambda row: row["top5"],
            )
            lines.append(
                f"- Base-reference negative control `{best_failure['file']}` "
                f"fails NIB at top-5 {fmt_num(best_failure['top5'])}, "
                f"showing that the corrected base-reference path still needs the "
                f"strong zero-out/EMA/source-completion-loss recipe rather than "
                f"xavier initialization."
            )
        lines.extend(
            [
                (
                    "- Interpretation: this is the strongest GPT-style result so far "
                    "for near-lossless distribution matching and it now has a "
                    "repeat-certified cross-tokenizer continuation-preservation "
                    "signal. The margin-hardened GPT2-medium local target paths "
                    "repeat-repair source-token surface preservation, and the "
                    "NLL-hardened GPT-Neo, Phi-3, and Qwen donor paths now repeat-repair the same "
                    "gate across Qwen/Phi target directions, but this is not yet literal lossless "
                    "token-level migration across arbitrary source families or private "
                    "model-scale targets."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## North-Star Transfer Gates",
            "",
            (
                f"- Certificate-bearing ABI artifacts: "
                f"{north_star.get('row_count', 0)}."
            ),
            (
                f"- Oracle-light artifacts: "
                f"{north_star.get('oracle_light_row_count', 0)}."
            ),
            (
                f"- Source-preservation measured artifacts: "
                f"{north_star.get('source_preservation_row_count', 0)}."
            ),
            (
                f"- Joint oracle-light + source-preservation artifacts: "
                f"{north_star.get('joint_oracle_light_source_preservation_count', 0)}."
            ),
            (
                f"- Passing joint artifacts: "
                f"{north_star.get('joint_passing_count', 0)}."
            ),
            (
                f"- Repeat-certified joint recipes: "
                f"{north_star.get('repeat_joint_passing_recipe_count', 0)}/"
                f"{north_star.get('repeat_required_recipe_count', 2)} required."
            ),
            (
                f"- Repeat-certified model pairs: "
                f"{north_star.get('repeat_joint_passing_pair_count', 0)}."
            ),
            f"- Current status: {case['north_star_status']}.",
            (
                f"- Selective-transfer audit artifacts: "
                f"{selective.get('row_count', 0)}."
            ),
            (
                f"- Strict selective-transfer passes: "
                f"{selective.get('strict_selective_transfer_pass_count', 0)}."
            ),
            (
                f"- Repeat-certified strict selective recipes: "
                f"{selective.get('repeat_strict_selective_recipe_count', 0)}/"
                f"{selective.get('repeat_required_recipe_count', SELECTIVE_TRANSFER_REQUIRED_REPEAT_RECIPES)} "
                "required."
            ),
            f"- Selective-transfer status: {case['selective_transfer_status']}.",
            (
                "- The local oracle-light source-preservation gate is now satisfied "
                "for the configured evidence threshold."
                if case["north_star_ready"]
                else "- Production GPT5-to-GPT6 domain-migration readiness remains "
                "an open gate until this section has repeated oracle-light "
                "source-preservation wins."
            ),
            (
                "- This still does not establish arbitrary lossless GPT5-to-GPT6 "
                "migration or targeted lossless selective transfer; it establishes "
                "repeat-certified scoped transfer under the current local model-pair "
                "suite."
            ),
            "",
        ]
    )
    for blocker in selective.get("open_blockers", []):
        lines.append(f"- Selective-transfer blocker: {blocker}")
    lines.append("")
    lines.extend(
        [
        "## Why This Merits Serious Benchmarking",
        "",
        (
            "The evidence is strongest where the claim is scoped: freeze the target "
            "backbone, migrate a domain operator, and calibrate a small ABI interface. "
            "The measured local runs show sub-0.3% target-side trainable fractions "
            "on 0.49B-3.82B targets while preserving NIB pass status, and the latest "
            "WikiText results convert prior rank/entropy tradeoffs into repeated "
            "bidirectional NIB passes. "
            "If that envelope holds under matched baselines and larger private-model "
            "evaluations, the training-time and compute-savings hypothesis becomes "
            "hard to dismiss."
        ),
        "",
        "## Adoption Gates",
        "",
        ]
    )
    lines.extend(f"- {gate}" for gate in gates)
    lines.extend(
        [
            "",
            "## Current Claim Boundary",
            "",
            f"- Supported: {summary['claim_boundary']['can_claim']}",
            f"- Not yet supported: {summary['claim_boundary']['cannot_claim_yet']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_markdown(summary), encoding="utf-8")
    ADOPTION_CASE_PATH.write_text(render_adoption_case(summary), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH.name}")
    print(f"Wrote {REPORT_PATH.name}")
    print(f"Wrote {ADOPTION_CASE_PATH.name}")


if __name__ == "__main__":
    main()
