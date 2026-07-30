import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_result(name):
    with (ROOT / name).open(encoding="utf-8") as f:
        return json.load(f)


def assert_nib_passes(result):
    nib = result["nib_l2"]
    assert nib["pass"] is True
    assert result["overall_pass"] is True
    assert nib["mean_js"] < result["thresholds"]["js_threshold"]
    assert nib["mean_top1_agree"] >= result["thresholds"]["top1_threshold"]
    assert nib["mean_top5_overlap"] >= result["thresholds"]["top5_threshold"]
    assert nib["mean_entropy_diff"] < result["thresholds"]["entropy_diff_threshold"]


def test_deepseek_d256_baseline_records_top5_failure():
    result = load_result("exp_deepseek_1p3b_nib_results.json")
    nib = result["nib_l2"]
    assert result["d_abi"] == 256
    assert nib["mean_top5_overlap"] < result["thresholds"]["top5_threshold"]
    assert nib["top5_pass"] is False
    assert result["overall_pass"] is False


def test_qwen_strict_locked_corpus_followup_passes():
    result = load_result(
        "exp_qwen_1p5b_nib_v2_d256_cal1200_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["d_abi"] == 256
    assert result["calibration_mode"] == "freeze_domain_net"
    assert "exp_*_nib_v2.py" in result["corpus_exclude_globs"]
    assert_nib_passes(result)


def test_deepseek_strict_locked_corpus_followup_passes():
    result = load_result(
        "exp_deepseek_1p3b_nib_v2_d512_cal2400_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["d_abi"] == 512
    assert result["calibration_mode"] == "freeze_domain_net"
    assert "exp_*_nib_v2.py" in result["corpus_exclude_globs"]
    assert_nib_passes(result)


def test_deepseek_width_without_depth_still_fails_top5():
    result = load_result(
        "exp_deepseek_1p3b_nib_v2_d512_cal1200_freeze_domain_net_alpha_stable_results.json"
    )
    nib = result["nib_l2"]
    assert result["d_abi"] == 512
    assert result["calibration_steps"] == 1200
    assert result["calibration_mode"] == "freeze_domain_net"
    assert "tests/test_model_agnostic_followups.py" in result["corpus_exclude_globs"]
    assert nib["top5_pass"] is False
    assert nib["mean_top5_overlap"] < result["thresholds"]["top5_threshold"]
    assert result["overall_pass"] is False


def test_deepseek_depth_without_width_still_fails_top5():
    result = load_result(
        "exp_deepseek_1p3b_nib_v2_d256_cal2400_freeze_domain_net_alpha_stable_results.json"
    )
    nib = result["nib_l2"]
    assert result["d_abi"] == 256
    assert result["calibration_steps"] == 2400
    assert result["calibration_mode"] == "freeze_domain_net"
    assert "tests/test_model_agnostic_followups.py" in result["corpus_exclude_globs"]
    assert nib["top5_pass"] is False
    assert nib["mean_top5_overlap"] < result["thresholds"]["top5_threshold"]
    assert result["overall_pass"] is False


def test_pythia_strict_low_capacity_followup_fails_top5():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia_410m_d256_cal1200_freeze_domain_net_alpha_stable_results.json"
    )
    nib = result["nib_l2"]
    assert result["target_model_type"] == "gpt_neox"
    assert result["d_abi"] == 256
    assert result["calibration_mode"] == "freeze_domain_net"
    assert nib["top5_pass"] is False
    assert result["overall_pass"] is False


def test_pythia_strict_full_width_followup_still_fails_top5():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia_410m_d1024_cal2400_freeze_domain_net_alpha_stable_results.json"
    )
    nib = result["nib_l2"]
    assert result["target_model_type"] == "gpt_neox"
    assert result["d_abi"] == 1024
    assert result["calibration_mode"] == "freeze_domain_net"
    assert nib["top5_pass"] is False
    assert result["overall_pass"] is False


def test_pythia_train_domain_diagnostic_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia_410m_d1024_cal2400_train_domain_alpha_stable_results.json"
    )
    assert result["target_model_type"] == "gpt_neox"
    assert result["d_abi"] == 1024
    assert result["calibration_mode"] == "train_domain"
    assert_nib_passes(result)


def test_pythia_same_model_strict_followup_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia_410m_to_pythia_410m_d1024_dom500_cal2400_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["source_model_type"] == "gpt_neox"
    assert result["target_model_type"] == "gpt_neox"
    assert result["source_model"] == result["target_model"]
    assert result["calibration_mode"] == "freeze_domain_net"
    assert_nib_passes(result)


def test_pythia_larger_to_smaller_strict_followup_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia_410m_to_pythia_160m_d768_dom500_cal2400_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["source_model"] == "EleutherAI/pythia-410m"
    assert result["target_model"] == "EleutherAI/pythia-160m"
    assert result["source_model_type"] == "gpt_neox"
    assert result["target_model_type"] == "gpt_neox"
    assert result["calibration_mode"] == "freeze_domain_net"
    assert_nib_passes(result)


def test_pythia_smaller_to_larger_topk_strict_followup_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia_160m_to_pythia_410m_d1024_dom500_cal2400_topk32w1_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["source_model"] == "EleutherAI/pythia-160m"
    assert result["target_model"] == "EleutherAI/pythia-410m"
    assert result["source_model_type"] == "gpt_neox"
    assert result["target_model_type"] == "gpt_neox"
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 1.0
    assert result["topk"] == 32
    assert_nib_passes(result)


def test_phi3_cross_family_topk_strict_followup_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_gpt2med_to_phi3_mini_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["source_model"] == "gpt2-medium"
    assert result["target_model"] == "microsoft/phi-3-mini-4k-instruct"
    assert result["source_model_type"] == "gpt2"
    assert result["target_model_type"] == "phi3"
    assert result["source_d_model"] == 1024
    assert result["target_d_model"] == 3072
    assert result["d_abi"] == 1024
    assert result["calibration_steps"] == 2400
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 1.0
    assert result["topk"] == 32
    assert result["torch_dtype"] == "float16"
    assert result["batch"] == 1
    assert_nib_passes(result)


def test_phi3_cross_family_shifted_seed_repeat_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_gpt2med_to_phi3_mini_d1024_dom500_cal2400_topk32w1_fp16_b1_seed314off100k_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["source_model"] == "gpt2-medium"
    assert result["target_model"] == "microsoft/phi-3-mini-4k-instruct"
    assert result["source_model_type"] == "gpt2"
    assert result["target_model_type"] == "phi3"
    assert result["seed"] == 314
    assert result["seed_offset"] == 100000
    assert result["source_domain_seed_base"] == 105000
    assert result["native_domain_seed_base"] == 105000
    assert result["calibration_seed_base"] == 107000
    assert result["nib_seed"] == 107777
    assert result["topk_kd_weight"] == 1.0
    assert result["torch_dtype"] == "float16"
    assert result["batch"] == 1
    assert_nib_passes(result)


def test_qwen2_to_phi3_topk_strict_followup_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_qwen2_1p5b_to_phi3_mini_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["source_model"] == "Qwen/Qwen2-1.5B"
    assert result["target_model"] == "microsoft/phi-3-mini-4k-instruct"
    assert result["source_model_type"] == "qwen2"
    assert result["target_model_type"] == "phi3"
    assert result["source_d_model"] == 1536
    assert result["target_d_model"] == 3072
    assert result["d_abi"] == 1024
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 1.0
    assert result["topk"] == 32
    assert result["torch_dtype"] == "float16"
    assert result["batch"] == 1
    assert_nib_passes(result)


def test_phi3_to_qwen2_topk_strict_followup_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_phi3_mini_to_qwen2_1p5b_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["source_model"] == "microsoft/phi-3-mini-4k-instruct"
    assert result["target_model"] == "Qwen/Qwen2-1.5B"
    assert result["source_model_type"] == "phi3"
    assert result["target_model_type"] == "qwen2"
    assert result["source_d_model"] == 3072
    assert result["target_d_model"] == 1536
    assert result["d_abi"] == 1024
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 1.0
    assert result["topk"] == 32
    assert result["torch_dtype"] == "float16"
    assert result["batch"] == 1
    assert_nib_passes(result)


def test_deepseek_to_pythia_topk_strict_followup_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_deepseek_1p3b_to_pythia_410m_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json"
    )
    assert result["source_model"] == "deepseek-ai/deepseek-coder-1.3b-base"
    assert result["target_model"] == "EleutherAI/pythia-410m"
    assert result["source_model_type"] == "llama"
    assert result["target_model_type"] == "gpt_neox"
    assert result["d_abi"] == 1024
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 1.0
    assert result["topk"] == 32
    assert_nib_passes(result)


def test_pythia_to_deepseek_topk_strict_followup_still_fails_top5():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia_410m_to_deepseek_1p3b_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json"
    )
    nib = result["nib_l2"]
    assert result["source_model"] == "EleutherAI/pythia-410m"
    assert result["target_model"] == "deepseek-ai/deepseek-coder-1.3b-base"
    assert result["source_model_type"] == "gpt_neox"
    assert result["target_model_type"] == "llama"
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 1.0
    assert nib["js_pass"] is True
    assert nib["top1_pass"] is True
    assert nib["entropy_pass"] is True
    assert nib["top5_pass"] is False
    assert result["overall_pass"] is False


def test_pythia_to_deepseek_best_bridge_repair_still_fails_top5():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia_410m_to_deepseek_1p3b_d512_dom500_cal4800_topk128w2_rankm10_hardneg10_bridge_fp16_b1_freeze_domain_net_alpha_stable_results.json"
    )
    nib = result["nib_l2"]
    assert result["source_model"] == "EleutherAI/pythia-410m"
    assert result["target_model"] == "deepseek-ai/deepseek-coder-1.3b-base"
    assert result["d_abi"] == 512
    assert result["calibration_steps"] == 4800
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 2.0
    assert result["topk"] == 128
    assert result["rank_margin_weight"] == 10.0
    assert result["hard_neg_weight"] == 10.0
    assert result["domain_bridge"] == "linear"
    assert nib["mean_top5_overlap"] > 0.85
    assert nib["mean_top5_overlap"] < result["thresholds"]["top5_threshold"]
    assert nib["top5_pass"] is False
    assert result["overall_pass"] is False


def test_pythia_to_deepseek_train_domain_diagnostic_still_fails_top5():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia_410m_to_deepseek_1p3b_d512_dom500_cal2400_topk32w1_train_domain_alpha_stable_results.json"
    )
    nib = result["nib_l2"]
    assert result["calibration_mode"] == "train_domain"
    assert result["topk_kd_weight"] == 1.0
    assert nib["top5_pass"] is False
    assert result["overall_pass"] is False


def assert_pythia_to_deepseek_topset_repair(result):
    assert result["source_model"] == "EleutherAI/pythia-410m"
    assert result["target_model"] == "deepseek-ai/deepseek-coder-1.3b-base"
    assert result["source_model_type"] == "gpt_neox"
    assert result["target_model_type"] == "llama"
    assert result["d_abi"] == 512
    assert result["calibration_steps"] == 7200
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 2.0
    assert result["topk"] == 128
    assert result["rank_margin_weight"] == 10.0
    assert result["rank_neg_k"] == 128
    assert result["hard_neg_weight"] == 10.0
    assert result["hard_neg_k"] == 64
    assert result["topset_weight"] == 5.0
    assert result["topset_k"] == 5
    assert result["domain_bridge"] == "linear"
    assert result["torch_dtype"] == "float16"
    assert result["batch"] == 1
    assert_nib_passes(result)


def assert_pythia_to_deepseek_long_cal_recipe(result, cal_steps):
    assert result["source_model"] == "EleutherAI/pythia-410m"
    assert result["target_model"] == "deepseek-ai/deepseek-coder-1.3b-base"
    assert result["source_model_type"] == "gpt_neox"
    assert result["target_model_type"] == "llama"
    assert result["d_abi"] == 512
    assert result["calibration_steps"] == cal_steps
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 2.0
    assert result["topk"] == 128
    assert result["rank_margin_weight"] == 10.0
    assert result["rank_neg_k"] == 128
    assert result["hard_neg_weight"] == 10.0
    assert result["hard_neg_k"] == 64
    assert result["topset_weight"] == 5.0
    assert result["topset_k"] == 5
    assert result["domain_bridge"] == "linear"
    assert result["torch_dtype"] == "float16"
    assert result["batch"] == 1


def test_pythia_to_deepseek_topset_bridge_repair_seed42_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_seed42_results.json"
    )
    assert result["seed"] == 42
    assert result["seed_offset"] == 0
    assert_pythia_to_deepseek_topset_repair(result)


def test_pythia_to_deepseek_topset_bridge_repair_shifted_seed_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_seed314_results.json"
    )
    assert result["seed"] == 314
    assert result["seed_offset"] == 100000
    assert result["nib_seed"] == 107777
    assert_pythia_to_deepseek_topset_repair(result)


def test_pythia_to_deepseek_top_logit_mse_ablation_fails_top5():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_toplogitmse05_seed42_results.json"
    )
    assert_pythia_to_deepseek_long_cal_recipe(result, 7200)
    assert result["top_logit_mse_weight"] == 0.5
    assert result["top_logit_mse_k"] == 64
    assert result["nib_l2"]["top5_pass"] is False
    assert result["overall_pass"] is False


def test_pythia_to_deepseek_longer_calibration_improves_hard_direction():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed42_results.json"
    )
    baseline = load_result(
        "exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_seed42_results.json"
    )
    assert_pythia_to_deepseek_long_cal_recipe(result, 16000)
    assert result["top_logit_mse_weight"] == 0.0
    assert result["nib_l2"]["mean_top5_overlap"] > baseline["nib_l2"]["mean_top5_overlap"]
    assert result["nib_l2"]["mean_top5_overlap"] >= 0.885
    assert result["nib_l2"]["mean_top1_agree"] >= 0.946
    assert_nib_passes(result)


def test_pythia_to_deepseek_wider_abi_ablation_fails_top5():
    result = load_result(
        "exp_generic_causal_nib_v2_pythia410_deepseek_d1024_cal12000_topset5_bridge_seed42_results.json"
    )
    assert result["source_model"] == "EleutherAI/pythia-410m"
    assert result["target_model"] == "deepseek-ai/deepseek-coder-1.3b-base"
    assert result["d_abi"] == 1024
    assert result["calibration_steps"] == 12000
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 2.0
    assert result["topset_weight"] == 5.0
    assert result["domain_bridge"] == "linear"
    assert result["nib_l2"]["top5_pass"] is False
    assert result["overall_pass"] is False


def assert_strong_topset_bridge_recipe(result):
    assert result["d_abi"] == 512
    assert result["calibration_steps"] == 4800
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 2.0
    assert result["topk"] == 128
    assert result["rank_margin_weight"] == 10.0
    assert result["rank_neg_k"] == 128
    assert result["hard_neg_weight"] == 10.0
    assert result["hard_neg_k"] == 64
    assert result["topset_weight"] == 5.0
    assert result["topset_k"] == 5
    assert result["domain_bridge"] == "linear"
    assert result["torch_dtype"] == "float16"
    assert result["batch"] == 1
    assert_nib_passes(result)


def test_gptneo_to_qwen25_topset_bridge_certification_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_seed42_results.json"
    )
    assert result["source_model"] == "EleutherAI/gpt-neo-125M"
    assert result["source_tokenizer"] == "gpt2"
    assert result["target_model"] == "Qwen/Qwen2.5-0.5B"
    assert result["source_model_type"] == "gpt_neo"
    assert result["target_model_type"] == "qwen2"
    assert_strong_topset_bridge_recipe(result)


def test_qwen25_to_gptneo_topset_bridge_certification_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal4800_topset5_bridge_seed42_results.json"
    )
    assert result["source_model"] == "Qwen/Qwen2.5-0.5B"
    assert result["target_model"] == "EleutherAI/gpt-neo-125M"
    assert result["target_tokenizer"] == "gpt2"
    assert result["source_model_type"] == "qwen2"
    assert result["target_model_type"] == "gpt_neo"
    assert_strong_topset_bridge_recipe(result)


def test_gptneo_to_phi3_topset_bridge_certification_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_gptneo125m_phi3_d1024_cal4800_topset5_bridge_seed42_results.json"
    )
    assert result["source_model"] == "EleutherAI/gpt-neo-125M"
    assert result["source_tokenizer"] == "gpt2"
    assert result["target_model"] == "microsoft/phi-3-mini-4k-instruct"
    assert result["source_model_type"] == "gpt_neo"
    assert result["target_model_type"] == "phi3"
    assert result["source_d_model"] == 768
    assert result["target_d_model"] == 3072
    assert result["d_abi"] == 1024
    assert result["calibration_steps"] == 4800
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 2.0
    assert result["topk"] == 128
    assert result["rank_margin_weight"] == 10.0
    assert result["rank_neg_k"] == 128
    assert result["hard_neg_weight"] == 10.0
    assert result["hard_neg_k"] == 64
    assert result["topset_weight"] == 5.0
    assert result["topset_k"] == 5
    assert result["domain_bridge"] == "linear"
    assert result["torch_dtype"] == "float16"
    assert result["batch"] == 1
    assert_nib_passes(result)


def test_phi3_to_gptneo_topset_bridge_certification_passes():
    result = load_result(
        "exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal4800_topset5_bridge_seed42_results.json"
    )
    assert result["source_model"] == "microsoft/phi-3-mini-4k-instruct"
    assert result["target_model"] == "EleutherAI/gpt-neo-125M"
    assert result["target_tokenizer"] == "gpt2"
    assert result["source_model_type"] == "phi3"
    assert result["target_model_type"] == "gpt_neo"
    assert result["source_d_model"] == 3072
    assert result["target_d_model"] == 768
    assert result["d_abi"] == 1024
    assert result["calibration_steps"] == 4800
    assert result["calibration_mode"] == "freeze_domain_net"
    assert result["topk_kd_weight"] == 2.0
    assert result["topk"] == 128
    assert result["rank_margin_weight"] == 10.0
    assert result["rank_neg_k"] == 128
    assert result["hard_neg_weight"] == 10.0
    assert result["hard_neg_k"] == 64
    assert result["topset_weight"] == 5.0
    assert result["topset_k"] == 5
    assert result["domain_bridge"] == "linear"
    assert result["torch_dtype"] == "float16"
    assert result["batch"] == 1
    assert_nib_passes(result)
