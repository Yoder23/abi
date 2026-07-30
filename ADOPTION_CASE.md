# ABI Adoption Case

Generated from local result JSON files by `build_proof_layers.py`.

## Decision Claim

ABI is ready to be benchmarked as a frozen-core, target-side interface calibration path for scoped domain migration. The current evidence does not yet justify replacing base-model training or claiming lossless arbitrary-model transfer, and it has not yet proven lossless selective domain transfer with off-domain noninterference.

## Measured Compute Story

- All 6 savings-layer transfer rows pass the current NIB thresholds: True.
- On local targets >= 400,000,000 parameters, ABI calibrates an average of 0.2254% and at most 0.2923% of target parameters.
- The least-frozen local target in that group is still 99.7077% frozen.
- The hard Pythia-410M -> DeepSeek-1.3B run calibrates 16,000 steps in 17.2 minutes with top-5 0.8850, top-1 0.9463, and JS 0.0120.

## Measured Accuracy Cost

- Mean calibrated-target perplexity overhead across savings rows is 8.4431%; worst measured overhead is 11.3507%.
- The hard direction improves from top-5 0.8679 at 7,200 calibration steps to 0.8850 at 16,000 steps, with a shifted-seed repeat still passing at top-5 0.8776.
- This is not yet lossless. The current case is efficient, certified scoped transfer with measurable quality cost.

## Domain Breadth

- The multi-domain atlas passes 4/4 diagonal domains (pass fraction 100.0000%).
- Generic cross-model WikiText now reaches top-5 0.9072, JS 0.0095, and entropy diff 0.2718 with pass=True.
- The passing WikiText run uses post-hoc logit scale 0.8909 in entropy_grid mode. Across 12 post-hoc WikiText passes, minimum top-5 is 0.8659 and maximum entropy diff is 0.3489.
- Bidirectional WikiText is repeat-certified for 3 model pairs: EleutherAI/gpt-neo-125M <-> Qwen/Qwen2.5-0.5B, EleutherAI/gpt-neo-125M <-> microsoft/phi-3-mini-4k-instruct, Qwen/Qwen2.5-0.5B <-> microsoft/phi-3-mini-4k-instruct.

## No-Leakage Certificate

- Withheld WikiText artifacts: 14; passing NIB: 14; split-separated: True.
- Best withheld run is `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_f4fba487b3_results.json` with train/posthoc/eval splits train/validation/test: top-5 0.9163, JS 0.0080, entropy diff 0.2654, pass=True.
- Direct non-GPT Qwen2.5 <-> Phi-3 withheld passes: 6; minimum top-5 0.8862; maximum entropy diff 0.2909.

## Matched Baseline Check

- Current matched baseline artifacts: 9; passing NIB: 4.
- Best LoRA/KD baseline is `exp_lora_kd_baseline_phi3_attn_r11_wikitext_train_val_test_topset5_balanced_cal4800_cap1800s_seed42_fullrerun_results.json` on microsoft/phi-3-mini-4k-instruct with 6,488,064 trainable parameters (0.1698%), top-5 0.8920, JS 0.0105, entropy diff 0.2398, pass=True.
- This is a target-side adaptation control, not source-core migration; it strengthens the adoption case only when interpreted against the ABI portability rows above.

- Split-separated LoRA/KD baselines: 6; passing NIB: 4.
- Best split-separated LoRA/KD baseline is `exp_lora_kd_baseline_phi3_attn_r11_wikitext_train_val_test_topset5_balanced_cal4800_cap1800s_seed42_fullrerun_results.json` with train/validation/test splits, top-5 0.8920, JS 0.0105, entropy diff 0.2398, pass=True.

## ABI vs LoRA Frontier

- ABI frontier artifacts: 21; passing NIB: 19.
- Best held-out ABI frontier is `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_f4fba487b3_results.json` with 3,936,257 trainable parameters and 14400 calibration steps, top-5 0.9163, top-1 0.9167, JS 0.0080, entropy diff 0.2654, pass=True.
- Against the split-separated LoRA/KD baseline, that best ABI run beats top-5, top-1, JS, and entropy together: True.
- The same D_ABI/calibration recipe under an independent seed/stream is `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_seed42_results.json`: top-5 0.9159, top-1 0.9167, JS 0.0085, entropy diff 0.2909, calibration steps 14400, pass=True.
- Full-NIB repeat passes: True; LoRA rank dominance repeat-certified: True.

## Phi ABI vs LoRA Frontier

- Phi-3 LoRA/KD baseline `exp_lora_kd_baseline_phi3_attn_r11_wikitext_train_val_test_topset5_balanced_cal4800_cap1800s_seed42_fullrerun_results.json` uses 6,488,064 trainable parameters and completed 4800/4800 requested LoRA steps under a 1800.0000s train cap; pass=True.
- Best Qwen -> Phi ABI by top-5 is `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_nativeinit_ema9995s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_w_cd56755dbb_results.json`: top-5 0.9266, top-1 0.9378, JS 0.0046, entropy diff 0.1205, pass=True.
- ABI beats the Phi LoRA baseline on every reported metric: True.
- Phi all-metric repeat-certified: True.
- Current Phi status: Qwen2-1.5B -> Phi ABI now has a same-recipe repeat-certified all-metric win over the full-step Phi LoRA comparator on WikiText. This is a flagship scoped result, not yet a production-readiness or universal/lossless transfer claim.

## Flagship GPT-Style Scenario

- Scenario: gpt2-medium -> microsoft/phi-3-mini-4k-instruct; artifacts: 51.
- Near-lossless distribution threshold: top-5 >= 0.9500, top-1 >= 0.9650, JS <= 0.0050, entropy diff <= 0.0500.
- Cross-tokenizer completion threshold: source top-1 preferred >= 0.5000 among source top-k continuations.
- Oracle-light near-lossless distribution passes: 11; repeat-certified recipes: 6/1.
- Repeat-certified strict distribution + completion recipes: 5/1.
- Native target-interface strict+completion passes: 4; xavier target-interface strict+completion passes: 0 (best top-5 0.9436).
- Zero-out target-interface strict+completion passes: 2 (repeat-certified recipes 1/1, best top-5 0.9540, best completion 0.6984).
- Reusable target-interface cache: saved 1, loaded 3; loaded strict-distribution passes 3, loaded strict+completion passes 2 (best top-5 0.9697, best completion 0.7143).
- Cache-load + source-completion-loss strict+completion passes: 2/2; repeat-certified recipes 1/1 required (best completion 0.7143).
- Phase-C-skipped base-reference bypass strict+completion passes: 13 (repeat-certified recipes 4/1, best top-5 0.9764, best completion 0.7344).
- Non-fixed posthoc base-bypass strict+completion passes: 11 (repeat-certified recipes 3/1, repeat-certified domains 2, best top-5 0.9764, best completion 0.7344).
- Cross-target GPT2-medium base-bypass strict+completion passes: 18 (repeat-certified recipes 6/1, repeat-certified target pairs 2, best top-5 0.9765, best completion 1.0000).
- Margin-hardened source-surface repair passes: 6 (repeat-certified recipes 2/1, repeat-certified target pairs 2, best top-1 surface 0.8525, min repaired top-1 in target top-k 0.5763, min repaired completion 0.6833).
- Cross-donor base-bypass source-surface repair passes: 12 (repeat-certified recipes 5/1, repeat-certified model pairs 5, repeat-certified source models 4, NLL-repair passes 6, best top-1 surface 0.8525, min repaired top-1 in target top-k 0.5763, min repaired completion 0.6719).
- Hard-recipient ordinary NIB + source-surface passes: 8 (repeat-certified recipes 1/1, repeat-certified pairs 1, best top-5 0.9339, best top-1 0.9650, best JS 0.0053, best entropy diff 0.1724, min top-5 0.8802, min top-1 surface 0.4098, min top-1 in target top-k 0.5873, min completion 0.7937).
- Base-reference xavier negative controls: 2 (best failed top-5 0.8520).
- Production readiness: local benchmark-ready but not production-ready.
- Production blocker: No GPT5/GPT6 private-model evaluation in this local repo.
- Production blocker: Strict distribution + completion now repeat-certifies Phase-C-skipped base-reference bypass across two GPT2-medium local target pairs, but still not on private GPT5/GPT6-scale models.
- Production blocker: No-base-reference xavier control still fails NIB top-5.
- Production blocker: Xavier target-interface control passes ordinary NIB but misses strict distribution + completion.
- Production blocker: The passing no-Phase-C base-bypass recipe still requires zero-out init, EMA restore, source-completion loss, and validation-selected posthoc logit scaling.
- Production blocker: Reusable target-interface cache load now repeat-clears strict distribution + completion only when the source-completion loss is enabled.
- Production blocker: Source-token surface repair is repeat-certified across two GPT2-medium local target pairs and now extends to GPT-Neo, Phi-3, and Qwen donors across Qwen/Phi target directions, but it is not yet lossless, private-model-scale, or universal across arbitrary source/target families.
- Production blocker: Selective-transfer off-domain no-leakage is now a first-class gate, but no opt-in selective audit artifacts have repeat-certified strict passes yet.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005_marginw05m010_every4_ent_9d7ba60a27_results.json`: top-5 0.9746, top-1 0.9907, JS 0.0004, entropy diff 0.0245, source top-1 in target top-k 0.4375, cross-tokenizer source top-1 preferred 0.7188.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005_marginw05m010_every4_ent_ae4de1e1b8_results.json`: top-5 0.9764, top-1 0.9789, JS 0.0006, entropy diff 0.0364, source top-1 in target top-k 0.7812, cross-tokenizer source top-1 preferred 0.6250.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005every4_balancedposthoc_lr_5872bec9c5_results.json`: top-5 0.9720, top-1 0.9809, JS 0.0007, entropy diff 0.0409, source top-1 in target top-k 0.5156, cross-tokenizer source top-1 preferred 0.7031.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005every4_balancedposthoc_lr_6cb73fe878_results.json`: top-5 0.9668, top-1 0.9768, JS 0.0009, entropy diff 0.0427, source top-1 in target top-k 0.5312, cross-tokenizer source top-1 preferred 0.6562.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005every4_balancedposthoc_lr_fc65cfb059_results.json`: top-5 0.9737, top-1 0.9659, JS 0.0008, entropy diff 0.0470, source top-1 in target top-k 0.7812, cross-tokenizer source top-1 preferred 0.6406.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005every4_ent002_balancedpos_2fcb7f06fa_results.json`: top-5 0.9733, top-1 0.9736, JS 0.0008, entropy diff 0.0481, source top-1 in target top-k 0.7812, cross-tokenizer source top-1 preferred 0.6250.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005every4_ent002_balancedpos_57fb6faaa7_results.json`: top-5 0.9749, top-1 0.9707, JS 0.0007, entropy diff 0.0430, source top-1 in target top-k 0.7812, cross-tokenizer source top-1 preferred 0.6406.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005every4_posthocentropy_lrd_1259c90b50_results.json`: top-5 0.9737, top-1 0.9825, JS 0.0008, entropy diff 0.0439, source top-1 in target top-k 0.5469, cross-tokenizer source top-1 preferred 0.6875.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005every4_scale09810_lrdecay_1a03a7404b_results.json`: top-5 0.9663, top-1 0.9732, JS 0.0012, entropy diff 0.0498, source top-1 in target top-k 0.5156, cross-tokenizer source top-1 preferred 0.6719.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw005every4_scale09810_lrdecay_df8c8474d3_results.json`: top-5 0.9733, top-1 0.9825, JS 0.0008, entropy diff 0.0403, source top-1 in target top-k 0.5469, cross-tokenizer source top-1 preferred 0.6875.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw020_marginw1m010_every2b2_en_06bb9d3240_results.json`: top-5 0.9740, top-1 0.9748, JS 0.0008, entropy diff 0.0389, source top-1 in target top-k 0.8281, cross-tokenizer source top-1 preferred 0.7344.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_basebypass_zeroout_ema9995s4800restore_sourcecomplossw020_marginw1m010_every2b2_en_c6f0613322_results.json`: top-5 0.9746, top-1 0.9748, JS 0.0006, entropy diff 0.0326, source top-1 in target top-k 0.7969, cross-tokenizer source top-1 preferred 0.7344.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_oraclelight_seqpreserve_nativeinit_cacheload_sourcecomplossw005every4_lrdecay2400x_259537583d_results.json`: top-5 0.9666, top-1 0.9866, JS 0.0007, entropy diff 0.0292, source top-1 in target top-k 0.5312, cross-tokenizer source top-1 preferred 0.6719.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_oraclelight_seqpreserve_nativeinit_cacheload_sourcecomplossw005every4_lrdecay2400x_49d4320d2d_results.json`: top-5 0.9697, top-1 0.9793, JS 0.0008, entropy diff 0.0352, source top-1 in target top-k 0.5079, cross-tokenizer source top-1 preferred 0.7143.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_oraclelight_seqpreserve_nativeinit_lrdecay2400x02_cal3600_topset5_toplogitmse005_p_c9b439ed0b_results.json`: top-5 0.9690, top-1 0.9809, JS 0.0008, entropy diff 0.0343, source top-1 in target top-k 0.2812, cross-tokenizer source top-1 preferred 0.5156.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_oraclelight_seqpreserve_nativeinit_lrdecay2400x02_cal3600_topset5_toplogitmse005_python_seed42_results.json`: top-5 0.9720, top-1 0.9862, JS 0.0004, entropy diff 0.0243, source top-1 in target top-k 0.2833, cross-tokenizer source top-1 preferred 0.5167.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_oraclelight_seqpreserve_zeroout_ema9995s4800restore_sourcecomplossw005every4_lrdec_b9f5339acd_results.json`: top-5 0.9513, top-1 0.9821, JS 0.0007, entropy diff 0.0323, source top-1 in target top-k 0.5397, cross-tokenizer source top-1 preferred 0.6984.
- Passing repeat `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_oraclelight_seqpreserve_zeroout_ema9995s4800restore_sourcecomplossw005every4_nativ_01ea1f7e82_results.json`: top-5 0.9540, top-1 0.9837, JS 0.0011, entropy diff 0.0363, source top-1 in target top-k 0.4762, cross-tokenizer source top-1 preferred 0.6984.
- Passing repeat `exp_generic_causal_nib_v2_selective_gpt2med_phi3_python_offwiki_base_seed42_results.json`: top-5 0.9707, top-1 0.9756, JS 0.0011, entropy diff 0.0494, source top-1 in target top-k 0.4531, cross-tokenizer source top-1 preferred 0.7031.
- Base-reference negative control `exp_generic_causal_nib_v2_gpt2med_phi3_d1024_flagship_baseref_seqpreserve_xavier_lrdecay2400x02_cal3600_topset5_toplogitmse005_python_seed42_results.json` fails NIB at top-5 0.8520, showing that the corrected base-reference path still needs the strong zero-out/EMA/source-completion-loss recipe rather than xavier initialization.
- Interpretation: this is the strongest GPT-style result so far for near-lossless distribution matching and it now has a repeat-certified cross-tokenizer continuation-preservation signal. The margin-hardened GPT2-medium local target paths repeat-repair source-token surface preservation, and the NLL-hardened GPT-Neo, Phi-3, and Qwen donor paths now repeat-repair the same gate across Qwen/Phi target directions, but this is not yet literal lossless token-level migration across arbitrary source families or private model-scale targets.

## North-Star Transfer Gates

- Certificate-bearing ABI artifacts: 91.
- Oracle-light artifacts: 91.
- Source-preservation measured artifacts: 91.
- Joint oracle-light + source-preservation artifacts: 91.
- Passing joint artifacts: 80.
- Repeat-certified joint recipes: 12/2 required.
- Repeat-certified model pairs: 6.
- Current status: repeat-certified oracle-light source-preservation gate passed.
- Selective-transfer audit artifacts: 1.
- Strict selective-transfer passes: 0.
- Repeat-certified strict selective recipes: 0/2 required.
- Selective-transfer status: open: repeat-certified strict selective transfer recipes 0/2; off-domain no-leakage evidence is required before claiming targeted lossless migration.
- The local oracle-light source-preservation gate is now satisfied for the configured evidence threshold.
- This still does not establish arbitrary lossless GPT5-to-GPT6 migration or targeted lossless selective transfer; it establishes repeat-certified scoped transfer under the current local model-pair suite.

- Selective-transfer blocker: Strict selective transfer lacks enough repeat-certified recipes.
- Selective-transfer blocker: Need paired target-domain pass and off-domain no-leakage pass for at least two independent seed/stream variants.
- Selective-transfer blocker: Need task-level selected-domain and off-domain evaluations, not only logit-space NIB.

## Why This Merits Serious Benchmarking

The evidence is strongest where the claim is scoped: freeze the target backbone, migrate a domain operator, and calibrate a small ABI interface. The measured local runs show sub-0.3% target-side trainable fractions on 0.49B-3.82B targets while preserving NIB pass status, and the latest WikiText results convert prior rank/entropy tradeoffs into repeated bidirectional NIB passes. If that envelope holds under matched baselines and larger private-model evaluations, the training-time and compute-savings hypothesis becomes hard to dismiss.

## Adoption Gates

- Extend the repeat-certified Phase-C-skipped base-reference bypass from the current GPT2-medium local target pairs to private-model-scale GPT5/GPT6-style evaluations before making GPT5-to-GPT6 migration claims.
- Require at least two repeat-certified oracle-light or base-bypass recipes with source-preservation measured, target-reference NIB pass, and independent seed or stream variants.
- Treat the GPT2-medium local target-pair repairs as near-lossless distribution plus source-surface repair results; extend beyond GPT2-medium donor and private-model-scale targets before using lossless wording.
- Extend the native-init/EMA repeat-certified LoRA/KD all-metric wins beyond the current Qwen2-1.5B -> Phi-3, GPT-Neo-125M -> Qwen2.5-0.5B, and Phi-3 -> Qwen2.5-0.5B WikiText recipes.
- Complete the matched baseline matrix against classic adapters, full/partial fine-tuning, and conventional distillation at equal trainable-parameter and wall-time budgets.
- Extend bidirectional rank/entropy WikiText certification to additional model pairs and domains.
- Extend Withheld-domain and no-leakage evaluations that are not tuned on the same corpus used to define the ABI bridge.
- Add selective-transfer audits that migrate one selected domain while preserving off-domain target behavior against the frozen target base/reference, with repeat-certified strict passes.
- Larger target families and private-model-style evals, including safety and refusal behavior, before making GPT5-to-GPT6 claims.

## Current Claim Boundary

- Supported: Scoped domain-module migration with frozen copied cores and small target-side ABI calibration where certificates pass.
- Not yet supported: Lossless migration of arbitrary GPT5 knowledge into arbitrary GPT6 targets, or replacement of GPT6 base training.
