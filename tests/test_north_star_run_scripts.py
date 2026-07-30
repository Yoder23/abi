from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oraclelight_preservation_run_script_targets_north_star_gate():
    text = (
        ROOT
        / "run_logs"
        / "run_oraclelight_preserve_phi3_qwen_d1024_cal14400_seed42off100k.ps1"
    ).read_text(encoding="utf-8")

    assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
    assert '$env:ABI_CAL_INIT = "native"' in text
    assert '$env:ABI_SOURCE_PRESERVATION_EVAL = "true"' in text
    assert '$env:ABI_SOURCE_PRESERVATION_PROMPTS = "64"' in text
    assert '$env:ABI_WIKITEXT_DOMAIN_SPLIT = "train"' in text
    assert '$env:ABI_WIKITEXT_POSTHOC_SPLIT = "validation"' in text
    assert '$env:ABI_WIKITEXT_EVAL_SPLIT = "test"' in text
    assert "oraclelight_preserve" in text
    assert "exp_generic_causal_nib_v2.py" in text


def test_qwen_phi_oraclelight_repeat_scripts_target_second_recipe():
    scripts = [
        ROOT / "run_logs" / "run_ns_qwen_phi3_seed42.ps1",
        ROOT / "run_logs" / "run_ns_qwen_phi3_seed42off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "Qwen/Qwen2-1.5B"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_EVAL = "true"' in text
        assert '$env:ABI_CAL_INIT = "native"' in text
        assert '$env:ABI_CAL_STEPS = "7200"' in text
        assert "oraclelight_preserve_qwen_phi3" in text


def test_pythia_deepseek_oraclelight_scripts_target_hard_pair():
    scripts = [
        ROOT / "run_logs" / "run_ns_pythia410_deepseek_d512_cal16000_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_ns_pythia410_deepseek_d512_cal16000_seed314off100k.ps1",
        ROOT
        / "run_logs"
        / "run_ns_pythia410_deepseek_d512_nativeinit_cal16000_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_ns_pythia410_deepseek_d512_nativeinit_cal16000_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "EleutherAI/pythia-410m"' in text
        assert (
            '$env:ABI_TARGET_MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"'
            in text
        )
        assert '$env:ABI_D_ABI = "512"' in text
        assert '$env:ABI_CAL_STEPS = "16000"' in text
        assert any(
            init in text
            for init in [
                '$env:ABI_CAL_INIT = "xavier"',
                '$env:ABI_CAL_INIT = "native"',
            ]
        )
        assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_EVAL = "true"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "python"' in text
        assert "oraclelight_preserve" in text
        assert "pythia410_deepseek" in text


def test_flagship_gpt_style_scripts_encode_positive_and_negative_controls():
    positive_scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_nativeinit_cal2400_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_nativeinit_cal2400_seed314off100k.ps1",
    ]

    for script in positive_scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_D_ABI = "1024"' in text
        assert '$env:ABI_CAL_STEPS = "2400"' in text
        assert '$env:ABI_CAL_INIT = "native"' in text
        assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "python"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_EVAL = "true"' in text
        assert "flagship_oraclelight_preserve_nativeinit" in text

    negative_text = (
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_baseref_xavier_cal2400_seed42.ps1"
    ).read_text(encoding="utf-8")
    assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in negative_text
    assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in negative_text
    assert '$env:ABI_D_ABI = "1024"' in negative_text
    assert '$env:ABI_CAL_STEPS = "2400"' in negative_text
    assert '$env:ABI_CAL_INIT = "xavier"' in negative_text
    assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in negative_text
    assert '$env:ABI_SOURCE_PRESERVATION_EVAL = "true"' in negative_text
    assert "flagship_baseref_preserve_xavier" in negative_text


def test_tuned_flagship_gpt_style_scripts_encode_completion_preservation_recipe():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_nativeinit_seqpreserve_topset_cal3600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_nativeinit_seqpreserve_topset_cal3600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_D_ABI = "1024"' in text
        assert '$env:ABI_CAL_STEPS = "3600"' in text
        assert '$env:ABI_CAL_LR_DECAY_STEP = "2400"' in text
        assert '$env:ABI_CAL_LR_DECAY_FACTOR = "0.2"' in text
        assert '$env:ABI_CAL_INIT = "native"' in text
        assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
        assert '$env:ABI_TOPSET_WEIGHT = "0.25"' in text
        assert '$env:ABI_TOPSET_K = "5"' in text
        assert '$env:ABI_TOP_LOGIT_MSE_WEIGHT = "0.005"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_EVAL = "true"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text
        assert "flagship_oraclelight_seqpreserve_nativeinit" in text


def test_tuned_flagship_gpt_style_scripts_encode_production_boundary_controls():
    controls = [
        (
            ROOT
            / "run_logs"
            / "run_flagship_gpt2med_phi3_baseref_xavier_seqpreserve_topset_cal3600_seed42.ps1",
            '$env:ABI_ORACLE_MODE = "base_target_reference"',
        ),
        (
            ROOT
            / "run_logs"
            / "run_flagship_gpt2med_phi3_oraclelight_xavier_seqpreserve_topset_cal3600_seed42.ps1",
            '$env:ABI_ORACLE_MODE = "target_base_interface"',
        ),
    ]

    for script, oracle_line in controls:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_STEPS = "3600"' in text
        assert '$env:ABI_CAL_LR_DECAY_STEP = "2400"' in text
        assert '$env:ABI_CAL_INIT = "xavier"' in text
        assert oracle_line in text
        assert '$env:ABI_TOPSET_WEIGHT = "0.25"' in text
        assert '$env:ABI_TOP_LOGIT_MSE_WEIGHT = "0.005"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text


def test_zeroout_flagship_scripts_target_native_init_replacement():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_zeroout_seqpreserve_topset_cal7200_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_zeroout_seqpreserve_topset_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_zeroout_seqpreserve_top5select_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_zeroout_hiddenres128_seqpreserve_top5select_cal9600_seed42.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
        assert '$env:ABI_TOPSET_WEIGHT = "0.25"' in text
        assert '$env:ABI_TOP_LOGIT_MSE_WEIGHT = "0.005"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    selected = scripts[2].read_text(encoding="utf-8")
    assert '$env:ABI_CAL_SELECT = "validation_top1"' in selected
    assert '$env:ABI_CAL_SELECT_TOP5_WEIGHT = "1.0"' in selected

    residual = scripts[3].read_text(encoding="utf-8")
    assert '$env:ABI_TARGET_RESIDUAL = "hidden"' in residual
    assert '$env:ABI_TARGET_RESIDUAL_RANK = "128"' in residual


def test_flagship_cache_scripts_encode_reusable_target_interface_prefit():
    save_script = (
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_nativeinit_cacheauto_cal3600_seed42.ps1"
    ).read_text(encoding="utf-8")
    load_script = (
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_nativeinit_cacheload_cal3600_seed314off100k.ps1"
    ).read_text(encoding="utf-8")

    for text in [save_script, load_script]:
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_INIT = "native"' in text
        assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
        assert '$env:ABI_TARGET_INTERFACE_CACHE_PATH = "target_interface_cache\\phi3_d1024_target_base_interface_python_nativebase5000.pt"' in text
        assert '$env:ABI_NATIVE_DOMAIN_SEED_BASE = "5000"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    assert '$env:ABI_TARGET_INTERFACE_CACHE = "auto"' in save_script
    assert '$env:ABI_TARGET_INTERFACE_CACHE = "load"' in load_script
    assert '$env:ABI_SEED_OFFSET = "100000"' in load_script


def test_flagship_cache_source_completion_loss_scripts_encode_repeat_recipe():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_nativeinit_cacheload_sourcecomploss_w005_cal3600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_nativeinit_cacheload_sourcecomploss_w005_cal3600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_INIT = "native"' in text
        assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
        assert '$env:ABI_TARGET_INTERFACE_CACHE = "load"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_EVERY = "4"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_PROMPTS = "64"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_zeroout_source_completion_loss_script_keeps_nonnative_control_scope():
    base_text = (
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_zeroout_sourcecomploss_w005_topset_cal9600_seed42.ps1"
    ).read_text(encoding="utf-8")
    stronger_topset_text = (
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_zeroout_sourcecomploss_w005_topsetw05_cal9600_seed42.ps1"
    ).read_text(encoding="utf-8")

    for text in [base_text, stronger_topset_text]:
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
        assert '$env:ABI_CAL_STEPS = "9600"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_EVERY = "4"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    assert '$env:ABI_TOPSET_WEIGHT = "0.25"' in base_text
    assert '$env:ABI_TOPSET_WEIGHT = "0.50"' in stronger_topset_text


def test_zeroout_ema_source_completion_loss_scripts_encode_repeat_recipe():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_zeroout_ema9995_sourcecomploss_w005_topset_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_oraclelight_zeroout_ema9995_sourcecomploss_w005_topset_cal9600_seed314off100k_nativebase5000.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "target_base_interface"' in text
        assert '$env:ABI_CAL_EMA_DECAY = "0.9995"' in text
        assert '$env:ABI_CAL_EMA_RESTORE = "true"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_TOPSET_WEIGHT = "0.25"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat
    assert '$env:ABI_NATIVE_DOMAIN_SEED_BASE = "5000"' in repeat


def test_basebypass_zeroout_ema_source_completion_loss_scripts_encode_repeat_recipe():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_scale09810_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_scale09810_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert "ABI_TARGET_INTERFACE_CACHE" not in text
        assert "ABI_NATIVE_DOMAIN_SEED_BASE" not in text
        assert '$env:ABI_CAL_STEPS = "9600"' in text
        assert '$env:ABI_CAL_EMA_DECAY = "0.9995"' in text
        assert '$env:ABI_CAL_EMA_RESTORE = "true"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_EVERY = "4"' in text
        assert '$env:ABI_TOPSET_WEIGHT = "0.25"' in text
        assert '$env:ABI_POSTHOC_LOGIT_SCALE = "entropy_grid"' in text
        assert '$env:ABI_POSTHOC_SCALE_MIN = "0.9810"' in text
        assert '$env:ABI_POSTHOC_SCALE_MAX = "0.9810"' in text
        assert '$env:ABI_POSTHOC_SELECTION = "minimax_entropy"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_basebypass_balanced_posthoc_scripts_keep_scale_selection_nonfixed():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_balancedposthoc_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_balancedposthoc_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert "ABI_TARGET_INTERFACE_CACHE" not in text
        assert "ABI_NATIVE_DOMAIN_SEED_BASE" not in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_CAL_EMA_RESTORE = "true"' in text
        assert '$env:ABI_POSTHOC_LOGIT_SCALE = "entropy_grid"' in text
        assert '$env:ABI_POSTHOC_SCALE_MIN = "0.94"' in text
        assert '$env:ABI_POSTHOC_SCALE_MAX = "1.02"' in text
        assert '$env:ABI_POSTHOC_SCALE_STEPS = "41"' in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text
        assert '$env:ABI_POSTHOC_SIGNED_ENTROPY_WEIGHT = "1.0"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_phi3_margin_source_completion_scripts_target_surface_repair_repeat():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent015_balancedposthoc_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent015_balancedposthoc_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.15"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "0.50"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN = "0.10"' in text
        assert "marginw05m010" in text
        assert "ent015" in text
        assert '$env:ABI_POSTHOC_SCALE_MIN = "0.94"' in text
        assert '$env:ABI_POSTHOC_SCALE_MAX = "1.02"' in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_phi3_wikitext_margin_scripts_keep_split_separation():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent015_balancedposthoc_wikitext_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent015_balancedposthoc_wikitext_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "wikitext"' in text
        assert '$env:ABI_WIKITEXT_DOMAIN_SPLIT = "train"' in text
        assert '$env:ABI_WIKITEXT_ALIGN_SPLIT = "train"' in text
        assert '$env:ABI_WIKITEXT_POSTHOC_SPLIT = "validation"' in text
        assert '$env:ABI_WIKITEXT_EVAL_SPLIT = "test"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.15"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "0.50"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN = "0.10"' in text
        assert "marginw05m010" in text
        assert "ent015" in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_phi3_wikitext_stronger_source_completion_repair_scripts_repeat():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w020_marginw1_topset_ent020_balancedposthoc_wikitext_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w020_marginw1_topset_ent020_balancedposthoc_wikitext_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "wikitext"' in text
        assert '$env:ABI_WIKITEXT_DOMAIN_SPLIT = "train"' in text
        assert '$env:ABI_WIKITEXT_POSTHOC_SPLIT = "validation"' in text
        assert '$env:ABI_WIKITEXT_EVAL_SPLIT = "test"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.20"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.20"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_EVERY = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_BATCH = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "1.00"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN = "0.10"' in text
        assert "sourcecomplossw020_marginw1m010_every2b2_ent020" in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_basebypass_balanced_posthoc_wikitext_scripts_are_split_separated():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_balancedposthoc_wikitext_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_balancedposthoc_wikitext_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "wikitext"' in text
        assert '$env:ABI_WIKITEXT_DOMAIN_SPLIT = "train"' in text
        assert '$env:ABI_WIKITEXT_ALIGN_SPLIT = "train"' in text
        assert '$env:ABI_WIKITEXT_POSTHOC_SPLIT = "validation"' in text
        assert '$env:ABI_WIKITEXT_EVAL_SPLIT = "test"' in text
        assert '$env:ABI_POSTHOC_SCALE_MIN = "0.94"' in text
        assert '$env:ABI_POSTHOC_SCALE_MAX = "1.02"' in text
        assert '$env:ABI_POSTHOC_SCALE_STEPS = "41"' in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_basebypass_entropy_regularized_wikitext_scripts_target_strict_entropy_repeat():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_ent002_balancedposthoc_wikitext_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_phi3_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_ent002_balancedposthoc_wikitext_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "wikitext"' in text
        assert '$env:ABI_WIKITEXT_DOMAIN_SPLIT = "train"' in text
        assert '$env:ABI_WIKITEXT_POSTHOC_SPLIT = "validation"' in text
        assert '$env:ABI_WIKITEXT_EVAL_SPLIT = "test"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.02"' in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text
        assert '$env:ABI_POSTHOC_SCALE_STEPS = "41"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_gpt2med_qwen_basebypass_scripts_target_second_pair_breadth():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_balancedposthoc_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_balancedposthoc_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert "ABI_TARGET_INTERFACE_CACHE" not in text
        assert "ABI_NATIVE_DOMAIN_SEED_BASE" not in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_CAL_EMA_RESTORE = "true"' in text
        assert '$env:ABI_POSTHOC_LOGIT_SCALE = "entropy_grid"' in text
        assert '$env:ABI_POSTHOC_SCALE_MIN = "0.55"' in text
        assert '$env:ABI_POSTHOC_SCALE_MAX = "1.10"' in text
        assert '$env:ABI_POSTHOC_SCALE_STEPS = "41"' in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_gpt2med_qwen_entropy_regularized_basebypass_scripts_target_strict_entropy():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_ent002_balancedposthoc_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_ent002_balancedposthoc_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.02"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text
        assert '$env:ABI_POSTHOC_SCALE_STEPS = "41"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_gpt2med_qwen_stronger_entropy_basebypass_scripts_target_repeat():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_ent010_balancedposthoc_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_topset_ent010_balancedposthoc_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.10"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_CAL_EMA_RESTORE = "true"' in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text
        assert '$env:ABI_POSTHOC_SCALE_STEPS = "41"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_gpt2med_qwen_margin_source_completion_scripts_are_separate_recipe():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent010_balancedposthoc_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent010_balancedposthoc_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.10"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "0.50"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN = "0.10"' in text
        assert "marginw05m010" in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text
        assert '$env:ABI_POSTHOC_SCALE_STEPS = "41"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_gpt2med_qwen_margin_entropy015_scripts_target_strict_repeat():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent015_balancedposthoc_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gpt2med_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent015_balancedposthoc_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.15"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "0.50"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN = "0.10"' in text
        assert "marginw05m010" in text
        assert "ent015" in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_gptneo_qwen_margin_entropy015_scripts_target_donor_family_breadth():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gptneo_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent015_balancedposthoc_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gptneo_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_topset_ent015_balancedposthoc_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "EleutherAI/gpt-neo-125M"' in text
        assert '$env:ABI_SOURCE_TOKENIZER_ID = "gpt2"' in text
        assert '$env:ABI_SOURCE_LABEL = "gptneo125m"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_D_ABI = "1024"' in text
        assert '$env:ABI_CAL_STEPS = "9600"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.15"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "0.50"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN = "0.10"' in text
        assert "marginw05m010" in text
        assert "ent015" in text
        assert '$env:ABI_POSTHOC_SELECTION = "balanced_entropy"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_gptneo_qwen_nll_surface_scripts_target_full_vocab_repair():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_gptneo_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_nllw05_every2b2_topset_ent015_balancedposthoc_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_gptneo_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_nllw05_every2b2_topset_ent015_balancedposthoc_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "EleutherAI/gpt-neo-125M"' in text
        assert '$env:ABI_SOURCE_TOKENIZER_ID = "gpt2"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_EVERY = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_BATCH = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "0.50"' in text
        assert '$env:ABI_SOURCE_COMPLETION_NLL_WEIGHT = "0.50"' in text
        assert "nllw05" in text
        assert "every2b2" in text
        assert "ent015" in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_phi3_qwen_nll_surface_scripts_target_modern_donor_repair():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_phi3_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_nllw05_every2b2_topset_ent015_balancedposthoc_wikitext_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_phi3_qwen_basebypass_zeroout_ema9995_sourcecomploss_w005_marginw05_nllw05_every2b2_topset_ent015_balancedposthoc_wikitext_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "wikitext"' in text
        assert '$env:ABI_WIKITEXT_DOMAIN_SPLIT = "train"' in text
        assert '$env:ABI_WIKITEXT_POSTHOC_SPLIT = "validation"' in text
        assert '$env:ABI_WIKITEXT_EVAL_SPLIT = "test"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_EVERY = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_BATCH = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "0.50"' in text
        assert '$env:ABI_SOURCE_COMPLETION_NLL_WEIGHT = "0.50"' in text
        assert '$env:ABI_RELEASE_SOURCE_BEFORE_TARGET = "true"' in text
        assert "nllw05" in text
        assert "every2b2" in text
        assert "wikitext" in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_phi3_qwen_stronger_nll_surface_scripts_target_completion_repair():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_phi3_qwen_basebypass_zeroout_ema9995_sourcecomploss_w020_marginw1_nllw1_every2b2_topset_ent020_balancedposthoc_wikitext_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_phi3_qwen_basebypass_zeroout_ema9995_sourcecomploss_w020_marginw1_nllw1_every2b2_topset_ent020_balancedposthoc_wikitext_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "wikitext"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.20"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.20"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_EVERY = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_BATCH = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "1.00"' in text
        assert '$env:ABI_SOURCE_COMPLETION_NLL_WEIGHT = "1.00"' in text
        assert '$env:ABI_RELEASE_SOURCE_BEFORE_TARGET = "true"' in text
        assert "sourcecomplossw020_marginw1m010_nllw1_every2b2_ent020" in text
        assert "separatecont" in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_qwen_phi3_basebypass_nll_surface_scripts_target_reverse_modern_pair():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_qwen_phi3_basebypass_zeroout_ema9995_sourcecomploss_w020_marginw1_nllw1_every2b2_topset_ent020_balancedposthoc_wikitext_align5000_cal9600_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_qwen_phi3_basebypass_zeroout_ema9995_sourcecomploss_w020_marginw1_nllw1_every2b2_topset_ent020_balancedposthoc_wikitext_align5000_cal9600_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "Qwen/Qwen2-1.5B"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_N_ALIGN_SENTENCES = "5000"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "wikitext"' in text
        assert '$env:ABI_ENTROPY_WEIGHT = "0.20"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.20"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_EVERY = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_BATCH = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "1.00"' in text
        assert '$env:ABI_SOURCE_COMPLETION_NLL_WEIGHT = "1.00"' in text
        assert '$env:ABI_RELEASE_SOURCE_BEFORE_TARGET = "true"' in text
        assert "align5000" in text
        assert "separatecont" in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_pythia_deepseek_basebypass_nll_surface_scripts_target_hard_coder_recipient():
    scripts = [
        ROOT
        / "run_logs"
        / "run_flagship_pythia410_deepseek_basebypass_xavier_targetheavy_topsetw10_sourcecomploss_w005_marginw05_nllw025cap125_every2b1_python_cal16000_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_flagship_pythia410_deepseek_basebypass_xavier_targetheavy_topsetw10_sourcecomploss_w005_marginw05_nllw025cap125_every2b1_python_cal16000_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "EleutherAI/pythia-410m"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"' in text
        assert '$env:ABI_D_ABI = "512"' in text
        assert '$env:ABI_CAL_STEPS = "16000"' in text
        assert '$env:ABI_CAL_LR_DECAY_STEP = "8000"' in text
        assert '$env:ABI_CAL_INIT = "xavier"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "python"' in text
        assert '$env:ABI_DOMAIN_BRIDGE = "none"' in text
        assert '$env:ABI_TOPK = "128"' in text
        assert '$env:ABI_TOPK_KD_WEIGHT = "2.0"' in text
        assert '$env:ABI_RANK_MARGIN_WEIGHT = "10.0"' in text
        assert '$env:ABI_HARD_NEG_WEIGHT = "10.0"' in text
        assert '$env:ABI_TOPSET_WEIGHT = "10.0"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_WEIGHT = "0.05"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_EVERY = "2"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_BATCH = "1"' in text
        assert '$env:ABI_SOURCE_COMPLETION_LOSS_START_STEP = "8000"' in text
        assert '$env:ABI_SOURCE_COMPLETION_MARGIN_WEIGHT = "0.50"' in text
        assert '$env:ABI_SOURCE_COMPLETION_NLL_WEIGHT = "0.25"' in text
        assert '$env:ABI_SOURCE_COMPLETION_NLL_CAP = "1.25"' in text
        assert '$env:ABI_RELEASE_SOURCE_BEFORE_TARGET = "true"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text
        assert "basebypass_xavier_targetheavy_topsetw10" in text
        assert "sourcecomplossw005_start8000_marginw05m010_nllw025cap125_every2b1" in text
        assert "separatecont" in text
        assert "python" in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat


def test_selective_transfer_scripts_add_off_domain_no_leakage_audit():
    scripts = [
        ROOT / "run_logs" / "run_selective_gpt2med_phi3_python_offwiki_seed42.ps1",
        ROOT
        / "run_logs"
        / "run_selective_gpt2med_phi3_python_offwiki_seed314off100k.ps1",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '$env:ABI_SOURCE_MODEL_ID = "gpt2-medium"' in text
        assert '$env:ABI_TARGET_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"' in text
        assert '$env:ABI_DOMAIN_CORPUS = "python"' in text
        assert '$env:ABI_ORACLE_MODE = "base_target_reference"' in text
        assert '$env:ABI_CAL_INIT = "zero_out"' in text
        assert '$env:ABI_SELECTIVE_TRANSFER_EVAL = "true"' in text
        assert '$env:ABI_SELECTIVE_OFF_DOMAIN_CORPUS = "wikitext"' in text
        assert '$env:ABI_SELECTIVE_OFF_DOMAIN_WIKITEXT_SPLIT = "test"' in text
        assert '$env:ABI_SELECTIVE_OFF_DOMAIN_REFERENCE = "base"' in text
        assert '$env:ABI_SOURCE_PRESERVATION_COMPLETION_EVAL = "true"' in text
        assert "selective_gpt2med_phi3_python_offwiki" in text

    repeat = scripts[1].read_text(encoding="utf-8")
    assert '$env:ABI_SEED = "314"' in repeat
    assert '$env:ABI_SEED_OFFSET = "100000"' in repeat
