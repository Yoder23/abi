# Model-Agnostic ABI Follow-Ups

These runs were added after the original new-experiment batch showed:

- Qwen2-1.5B, `D_ABI=256`: PASS
- Qwen2-7B, `D_ABI=256`: PASS
- DeepSeek-Coder-1.3B, `D_ABI=256`: FAIL on top-5 overlap

The DeepSeek failure is preserved in `exp_deepseek_1p3b_nib_results.json`.
It should be treated as evidence against the strongest fixed-256 universal claim.

## Stricter Copy/Paste Protocol

The v2 runners use `calibration_mode="freeze_domain_net"`:

- Train source ABI domain module.
- Rotate the source domain MLP into the target ABI basis with Procrustes.
- Freeze the rotated domain MLP core during target calibration.
- Train only target interface projections, ABI LayerNorm, domain LayerNorm, and
  the scalar domain gate.

The v2 runners also lock the local Python corpus by excluding `exp_*_nib_v2.py`
from corpus loading, so adding follow-up scripts does not change future inputs.

## Locked-Corpus Results

| Result | Target family | D_ABI | Calibration | Domain MLP core | Top-5 | Overall |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `exp_qwen_1p5b_nib_v2_d256_cal1200_freeze_domain_net_alpha_stable_results.json` | Qwen2 | 256 | 1200 | frozen | 0.8655 | PASS |
| `exp_deepseek_1p3b_nib_v2_d512_cal2400_freeze_domain_net_alpha_stable_results.json` | Llama/DeepSeek | 512 | 2400 | frozen | 0.8646 | PASS |

## DeepSeek Ablations

| Result | D_ABI | Calibration | Top-5 | Overall | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| `exp_deepseek_1p3b_nib_results.json` | 256 | 1200 | 0.8437 | FAIL | Original fixed-width baseline fails. |
| `exp_deepseek_1p3b_nib_v2_d512_cal1200_freeze_domain_net_alpha_stable_results.json` | 512 | 1200 | 0.8301 | FAIL | Width alone does not rescue DeepSeek. |
| `exp_deepseek_1p3b_nib_v2_d256_cal2400_freeze_domain_net_alpha_stable_results.json` | 256 | 2400 | 0.8519 | FAIL | Depth alone does not rescue fixed 256. |
| `exp_deepseek_1p3b_nib_v2_d512_cal2400_freeze_domain_net_alpha_stable_results.json` | 512 | 2400 | 0.8646 | PASS | Llama-family target needs both wider ABI and longer calibration. |

## Current Claim Posture

Supported: ABI transfer works across multiple directed source/target pairs
spanning Qwen2, DeepSeek/Llama-family, Pythia/GPT-NeoX, and Phi-3 under NIB
when the ABI dimension, calibration budget, and rank-calibration loss are
allowed to scale for the harder target, while keeping the rotated domain MLP
core frozen.

Not supported: a universal fixed `D_ABI=256` claim. DeepSeek-Coder-1.3B failed
that original protocol at top-5 overlap 0.8437 versus the 0.86 threshold.

Working design rule: use a strict frozen-domain-core protocol, size the ABI
adapter/capacity to the target family, and add interface-only top-k KD when
top-5/rank fidelity is the limiting metric. For DeepSeek/Llama at
`d_model=2048`, the observed working point is `D_ABI=512` with 2400 calibration
steps. Qwen2-1.5B still clears NIB at `D_ABI=256` with 1200 calibration steps.
Phi-3-mini at `d_model=3072` clears strongly at `D_ABI=1024` with 2400
calibration steps, top-k KD, fp16 target weights, and batch 1.

Still not supported: an unlimited "any model, any size" theorem or an
exhaustive proof over every possible directed model pair. The evidence is now
cross-family, bidirectional for Phi-3/Qwen2, repaired for the previous
Pythia-410M -> DeepSeek-Coder hard direction, and repeat-checked under shifted
seeds for both GPT-2-medium -> Phi-3 and Pythia-410M -> DeepSeek-Coder, but it
remains empirical and protocol-specific.

## Pythia/GPT-NeoX Probe

`exp_generic_causal_nib_v2.py` adds a reusable decoder-only runner for cached
Hugging Face targets. It keeps the same strict default: rotate the copied domain
MLP into the target ABI basis, freeze the domain MLP core, and calibrate only
the target interface plus normalization/gate.

| Result | Target | D_ABI | Calibration mode | Top-5 | Overall | Interpretation |
| --- | --- | ---: | --- | ---: | --- | --- |
| `exp_generic_causal_nib_v2_pythia_410m_d256_cal1200_freeze_domain_net_alpha_stable_results.json` | Pythia-410M | 256 | frozen domain core | 0.8143 | FAIL | Low-capacity strict transfer fails rank/top-5. |
| `exp_generic_causal_nib_v2_pythia_410m_d512_cal2400_freeze_domain_net_alpha_stable_results.json` | Pythia-410M | 512 | frozen domain core | 0.8516 | FAIL | DeepSeek working point is not enough for Pythia. |
| `exp_generic_causal_nib_v2_pythia_410m_d1024_cal2400_freeze_domain_net_alpha_stable_results.json` | Pythia-410M | 1024 | frozen domain core | 0.8499 | FAIL | Full target-width ABI still does not port GPT-2 domain core cleanly. |
| `exp_generic_causal_nib_v2_pythia_410m_d1024_cal2400_train_domain_alpha_stable_results.json` | Pythia-410M | 1024 | train domain core | 0.8830 | PASS | ABI placement can match Pythia; frozen-core portability is the failing part. |
| `exp_generic_causal_nib_v2_pythia_410m_to_pythia_410m_d1024_dom500_cal2400_freeze_domain_net_alpha_stable_results.json` | Pythia-410M -> Pythia-410M | 1024 | frozen domain core | 0.8602 | PASS | Same-model frozen-core protocol can pass with very strong Procrustes alignment. |
| `exp_generic_causal_nib_v2_pythia_160m_to_pythia_410m_d512_cal2400_freeze_domain_net_alpha_stable_results.json` | Pythia-160M -> Pythia-410M | 512 | frozen domain core | 0.8387 | FAIL | Smaller-to-larger Pythia transfer fails despite same tokenizer/family. |
| `exp_generic_causal_nib_v2_pythia_160m_to_pythia_410m_d1024_cal2400_freeze_domain_net_alpha_stable_results.json` | Pythia-160M -> Pythia-410M | 1024 | frozen domain core | 0.8494 | FAIL | Full-width smaller-to-larger still misses top-5. |
| `exp_generic_causal_nib_v2_pythia_160m_to_pythia_410m_d1024_dom2000_cal2400_freeze_domain_net_alpha_stable_results.json` | Pythia-160M -> Pythia-410M | 1024 | frozen domain core | 0.7363 | FAIL | Longer source/native domain training made the target oracle too strong for the transferred source core. |
| `exp_generic_causal_nib_v2_pythia_410m_to_pythia_160m_d768_dom500_cal2400_freeze_domain_net_alpha_stable_results.json` | Pythia-410M -> Pythia-160M | 768 | frozen domain core | 0.9437 | PASS | Larger-to-smaller Pythia transfer works strongly. |
| `exp_generic_causal_nib_v2_pythia_160m_to_pythia_410m_d1024_dom500_cal2400_topk32w1_freeze_domain_net_alpha_stable_results.json` | Pythia-160M -> Pythia-410M | 1024 | frozen domain core + interface top-k KD | 0.8603 | PASS | Top-k interface KD fixes smaller-to-larger rank fidelity without training the copied domain core. |

Current diagnosis: Pythia/GPT-NeoX transfer is rank-fidelity sensitive. Plain
KD aligns JS/top-1/entropy but misses top-5 for smaller-to-larger Pythia. Adding
an interface-only top-k KD term fixes the smaller-to-larger case while keeping
the copied domain MLP core frozen. A universal claim should therefore specify
the stricter frozen-domain-core protocol plus a top-k/rank calibration term for
families where top-5 is the limiting metric.

## DeepSeek/GPT-NeoX Directed Probe

The Pythia/GPT-NeoX <-> DeepSeek/Llama probe exposed a directional gap. DeepSeek
can donate a copied frozen domain core into a Pythia target with the standard
top-k protocol. Pythia-410M -> DeepSeek-Coder initially failed, then crossed
NIB after adding a target-side linear domain bridge, stronger rank controls, a
full-vocab top-set loss, and longer calibration. The copied source domain MLP
core remains frozen in the repaired runs.

| Result | Source -> Target | D_ABI | Calibration mode | Rank controls | Top-5 | Overall | Interpretation |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| `exp_generic_causal_nib_v2_deepseek_1p3b_to_pythia_410m_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json` | DeepSeek-Coder-1.3B -> Pythia-410M | 1024 | frozen domain core + top-k KD | top-k 32 | 0.9107 | PASS | Llama-family source can transfer into GPT-NeoX strongly. |
| `exp_generic_causal_nib_v2_pythia_410m_to_deepseek_1p3b_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 1024 | frozen domain core + top-k KD | top-k 32 | 0.8002 | FAIL | Reverse direction fails top-5 despite JS/top1/entropy passing. |
| `exp_generic_causal_nib_v2_pythia_410m_to_deepseek_1p3b_d512_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + top-k KD | top-k 32 | 0.8083 | FAIL | Narrower ABI improves target oracle PPL but still fails rank fidelity. |
| `exp_generic_causal_nib_v2_pythia_410m_to_deepseek_1p3b_d2048_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 2048 | frozen domain core + top-k KD | top-k 32 | 0.7744 | FAIL | Width alone worsens the transfer. |
| `exp_generic_causal_nib_v2_pythia_410m_to_deepseek_1p3b_d512_dom500_cal2400_topk32w1_rankm5_fp16_b1_freeze_domain_net_alpha_stable_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + top-k KD | teacher rank margin | 0.8359 | FAIL | Rank loss moves in the right direction but does not clear NIB. |
| `exp_generic_causal_nib_v2_pythia_410m_to_deepseek_1p3b_d512_dom500_cal4800_topk128w2_rankm10_hardneg10_bridge_fp16_b1_freeze_domain_net_alpha_stable_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + bridge | top-k 128, rank margin, student hard negatives | 0.8527 | FAIL | Close but still below the 0.86 top-5 threshold without the top-set loss. |
| `exp_generic_causal_nib_v2_pythia_410m_to_deepseek_1p3b_d512_dom500_cal2400_topk32w1_train_domain_alpha_stable_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | train domain core | top-k 32 | 0.8182 | FAIL | Simply unfreezing the copied domain net for 2400 steps does not solve this direction. |
| `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_seed42_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + bridge | top-k 128, rank margin, hard negatives, top-set loss | 0.8679 | PASS | Final repaired recipe clears NIB on the original seed. |
| `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_seed314_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + bridge | top-k 128, rank margin, hard negatives, top-set loss | 0.8674 | PASS | Same repaired recipe clears NIB under shifted init, train batches, PPL batches, and NIB chunks. |
| `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_toplogitmse05_seed42_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + bridge | top-k 128, rank margin, hard negatives, top-set loss, centered top-logit MSE | 0.8588 | FAIL | Matching top-64 centered logits at weight 0.5 over-constrains the interface and hurts top-5. |
| `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal9600_topset5_bridge_seed42_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + bridge | top-k 128, rank margin, hard negatives, top-set loss, longer calibration | 0.8736 | PASS | Longer target-side calibration improves the hard-direction top-5 frontier without changing the frozen copied core. |
| `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal12000_topset5_bridge_seed42_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + bridge | top-k 128, rank margin, hard negatives, top-set loss, longer calibration | 0.8742 | PASS | Current best hard-direction top-5 certificate; top-1 also rises to 0.9419. |
| `exp_generic_causal_nib_v2_pythia410_deepseek_d1024_cal12000_topset5_bridge_seed42_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 1024 | frozen domain core + bridge | top-k 128, rank margin, hard negatives, top-set loss, longer calibration | 0.8403 | FAIL | More ABI width hurts this direction, so the working `D_ABI=512` appears to be a useful alignment regularizer rather than a capacity ceiling. |
| `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed42_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + bridge | top-k 128, rank margin, hard negatives, top-set loss, longer calibration | 0.8850 | PASS | New hard-direction frontier: longer target-side interface calibration improves top-5, top-1, and JS without changing the frozen copied core. |
| `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed314_results.json` | Pythia-410M -> DeepSeek-Coder-1.3B | 512 | frozen domain core + bridge | top-k 128, rank margin, hard negatives, top-set loss, longer calibration | 0.8776 | PASS | Shifted-seed repeat validates the 16000-step recipe above threshold with changed init, train batches, PPL batches, and NIB chunks. |

Current diagnosis: Pythia-410M -> DeepSeek-Coder requires listwise top-set
calibration in addition to top-k KD, teacher-rank loss, student hard-negative
suppression, longer calibration, and an identity-initialized target domain
bridge around the frozen core. Extending `D_ABI=512` calibration from 7200 to
16000 steps continues improving rank fidelity on the hardest direction, but
raising the ABI width to 1024 and adding a centered top-logit MSE term both
hurt top-5. The current best recipe therefore uses a regularized `D_ABI=512`
interface, long target-side calibration, and no direct top-logit MSE. The
repair keeps the copied source domain MLP core frozen, so this remains
copy/paste transfer with a stronger target-side ABI interface.

## Proof-Layer Report

`build_proof_layers.py` generates `PROOF_LAYERS.md`,
`ADOPTION_CASE.md`, and `proof_layers_summary.json` from the local result
artifacts. It separates three claims that need to stay distinct:

- savings: measured target-side trainable parameters versus counted local target
  model parameters where weights are available;
- accuracy frontier: hard-direction Pythia-410M -> DeepSeek-Coder progress and
  repeatability;
- domain breadth: certified atlas domains versus still-open generic cross-model
  WikiText entropy failures.

The current generated report shows target-side calibration uses 0.1948% of the
counted DeepSeek target parameters for the hard direction, while the copied
source domain core remains frozen.

The adoption case is intentionally stricter than the proof-layer report: it
summarizes measured compute savings, measured perplexity overhead, the generic
WikiText entropy blocker, and the matched-baseline gates needed before claiming
GPT5-to-GPT6-style domain migration as a training replacement.

## Generic WikiText Domain Probe

`ABI_DOMAIN_CORPUS=wikitext` switches the generic runner from the local Python
corpus to WikiText-2 for domain training and NIB evaluation. This gives a direct
second-domain probe rather than only using WikiText for alignment.

| Result | Source -> Target | Calibration | Top-5 | Entropy diff | Overall | Interpretation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_wikitext_seed42_results.json` | GPT-Neo-125M -> Qwen2.5-0.5B | 4800 | 0.8972 | 0.6567 | FAIL | Rank transfer works on WikiText, but distribution entropy is too far from the native target oracle. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_seed42_results.json` | GPT-Neo-125M -> Qwen2.5-0.5B | 9600 | 0.9069 | 0.6961 | FAIL | Longer calibration improves top-5 but does not fix entropy. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_posthocscale_seed42_results.json` | GPT-Neo-125M -> Qwen2.5-0.5B | 9600 | 0.9072 | 0.2718 | PASS | Post-hoc entropy-grid logit scaling converts the prior rank-only WikiText signal into a full NIB pass. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_posthocscale_seed314_results.json` | GPT-Neo-125M -> Qwen2.5-0.5B | 9600 | 0.9007 | 0.2919 | PASS | Shifted-seed repeat confirms the post-hoc entropy-grid repair above threshold. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal4800_topset5_bridge_wikitext_seed42_results.json` | Qwen2.5-0.5B -> GPT-Neo-125M | 4800 | 0.8322 | 0.8003 | FAIL | Reverse direction fails rank and entropy. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal9600_topset5_bridge_wikitext_posthocscale_seed42_results.json` | Qwen2.5-0.5B -> GPT-Neo-125M | 9600 | 0.8521 | 0.3734 | FAIL | Longer reverse calibration nearly reaches both thresholds but still misses top-5 and entropy. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal16000_topset5_bridge_wikitext_posthocscale_seed42_results.json` | Qwen2.5-0.5B -> GPT-Neo-125M | 16000 | 0.8576 | 0.3447 | FAIL | Entropy passes, but top-5 remains just below threshold. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal17000_topset5_bridge_wikitext_posthocscale_seed42_results.json` | Qwen2.5-0.5B -> GPT-Neo-125M | 17000 | 0.8670 | 0.4563 | FAIL | Top-5 passes, but entropy regresses. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal20000_topset5_bridge_wikitext_posthocscale_seed42_results.json` | Qwen2.5-0.5B -> GPT-Neo-125M | 20000 | 0.8659 | 0.3868 | FAIL | Top-5 passes, but entropy misses. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal24000_topset5_bridge_wikitext_posthocscale_seed42_results.json` | Qwen2.5-0.5B -> GPT-Neo-125M | 24000 | 0.8743 | 0.4207 | FAIL | More calibration improves rank but worsens entropy. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal20000_topset5_bridge_wikitext_posthocminimax_seed42_results.json` | Qwen2.5-0.5B -> GPT-Neo-125M | 20000 | 0.8659 | 0.3489 | PASS | Multi-seed minimax post-hoc scale repairs entropy while preserving rank. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal20000_topset5_bridge_wikitext_posthocminimax_seed314_results.json` | Qwen2.5-0.5B -> GPT-Neo-125M | 20000 | 0.8794 | 0.3485 | PASS | Shifted-seed repeat confirms the reverse WikiText repair. |
| `exp_generic_causal_nib_v2_gptneo125m_phi3_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed42_results.json` | GPT-Neo-125M -> Phi-3-mini | 4800 | 0.8926 | 0.1547 | PASS | Larger Phi-3 target passes WikiText with the copied domain core frozen. |
| `exp_generic_causal_nib_v2_gptneo125m_phi3_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed314_results.json` | GPT-Neo-125M -> Phi-3-mini | 4800 | 0.8877 | 0.1663 | PASS | Shifted-seed repeat confirms the larger Phi-3 target direction. |
| `exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal20000_topset5_bridge_wikitext_posthocminimax_seed42_results.json` | Phi-3-mini -> GPT-Neo-125M | 20000 | 0.8665 | 0.3247 | PASS | Reverse Phi-3 -> GPT-Neo passes with the minimax post-hoc scale. |
| `exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal20000_topset5_bridge_wikitext_posthocminimax_seed314_results.json` | Phi-3-mini -> GPT-Neo-125M | 20000 | 0.8712 | 0.3942 | FAIL | Shifted seed preserves rank but misses entropy before entropy-stabilized calibration. |
| `exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal20000_topset5_bridge_wikitext_ent002_posthocbalanced_seed314_results.json` | Phi-3-mini -> GPT-Neo-125M | 20000 | 0.8759 | 0.3453 | PASS | Stable fp32 entropy matching plus balanced post-hoc scale repairs the shifted-seed reverse direction. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed42_results.json` | Qwen2.5-0.5B -> Phi-3-mini | 4800 | 0.8907 | 0.1563 | PASS | Direct Qwen -> Phi-3 WikiText transfer passes without a GPT-Neo endpoint. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed314_results.json` | Qwen2.5-0.5B -> Phi-3-mini | 4800 | 0.8922 | 0.1638 | PASS | Shifted-seed repeat confirms the direct Qwen -> Phi-3 direction. |
| `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed42_results.json` | Phi-3-mini -> Qwen2.5-0.5B | 4800 | 0.9018 | 0.2448 | PASS | Reverse direct Phi-3 -> Qwen direction passes strongly. |
| `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_cal4800_topset5_bridge_wikitext_posthocminimax_seed314_results.json` | Phi-3-mini -> Qwen2.5-0.5B | 4800 | 0.8988 | 0.2459 | PASS | Shifted-seed repeat completes the direct Qwen/Phi-3 bidirectional certificate. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_wikitext_ent1_seed42_results.json` | GPT-Neo-125M -> Qwen2.5-0.5B | 4800 | 0.0000 | NaN | FAIL | Direct entropy-loss weight 1.0 destabilizes calibration. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_wikitext_ent001_seed42_results.json` | GPT-Neo-125M -> Qwen2.5-0.5B | 4800 | 0.0000 | NaN | FAIL | Direct entropy-loss weight 0.01 still destabilizes calibration. |
| `exp_generic_causal_nib_v2_pythia410_pythia160_d768_cal2400_topk32_wikitext_seed42_results.json` | Pythia-410M -> Pythia-160M | 2400 | 0.4600 | 0.7248 | FAIL | This setup is not a useful WikiText target: the native Pythia-160M oracle PPL is very poor. |

Current diagnosis: generic cross-model WikiText is now repeat-certified in both
directions for three pairs: GPT-Neo-125M <-> Qwen2.5-0.5B, GPT-Neo-125M <->
Phi-3-mini, and Qwen2.5-0.5B <-> Phi-3-mini. The direct Qwen/Phi-3 pair removes
GPT-Neo as an endpoint and still passes in both directions under shifted seeds.
GPT-Neo targets are the harder entropy case: Qwen2.5 -> GPT-Neo needs the
stronger multi-seed minimax scale selector, while Phi-3 -> GPT-Neo also needs
the stable fp32 entropy-matching term on the shifted seed. The copied source
domain MLP core remains frozen in all passing directions. The separate
`multi_domain_atlas_results` artifact remains the broader certified
multi-domain evidence: Python, WikiText, Markdown, and SQL all pass diagonal
domain-chart checks.

## No-Leakage WikiText Certificate

The generic runner now supports explicit WikiText split controls:
`ABI_WIKITEXT_DOMAIN_SPLIT`, `ABI_WIKITEXT_ALIGN_SPLIT`,
`ABI_WIKITEXT_POSTHOC_SPLIT`, and `ABI_WIKITEXT_EVAL_SPLIT`. This separates
source/native domain training, Procrustes alignment, post-hoc entropy-scale
selection, and final NIB evaluation.

For the clean certificate below, the ABI source/native/calibration phases use
WikiText train, the post-hoc scale is selected on WikiText validation, and the
final NIB is evaluated on WikiText test. The copied source domain MLP core
remains frozen.

| Result | Source -> Target | Split train/posthoc/eval | Top-5 | Top-1 | JS | Entropy diff | Overall |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | GPT-Neo-125M -> Qwen2.5-0.5B | train/validation/test | 0.8972 | 0.8963 | 0.0107 | 0.2992 | PASS |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | GPT-Neo-125M -> Qwen2.5-0.5B | train/validation/test | 0.8920 | 0.8988 | 0.0106 | 0.2838 | PASS |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | Qwen2.5-0.5B -> Phi-3-mini | train/validation/test | 0.8885 | 0.9012 | 0.0104 | 0.1853 | PASS |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | Qwen2.5-0.5B -> Phi-3-mini | train/validation/test | 0.8880 | 0.9146 | 0.0111 | 0.1686 | PASS |
| `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | Phi-3-mini -> Qwen2.5-0.5B | train/validation/test | 0.8958 | 0.8927 | 0.0103 | 0.2837 | PASS |
| `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | Phi-3-mini -> Qwen2.5-0.5B | train/validation/test | 0.8862 | 0.8927 | 0.0112 | 0.2763 | PASS |
| `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_seed42_results.json` | Phi-3-mini -> Qwen2.5-0.5B | train/validation/test | 0.9159 | 0.9167 | 0.0085 | 0.2909 | PASS |
| `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_f4fba487b3_results.json` | Phi-3-mini -> Qwen2.5-0.5B | train/validation/test | 0.9163 | 0.9167 | 0.0080 | 0.2654 | PASS |

Current diagnosis: WikiText transfer now survives a production-style no-leakage
split under shifted-seed repeats for GPT-Neo-125M -> Qwen2.5-0.5B and for the
direct non-GPT Qwen2.5-0.5B <-> Phi-3-mini pair. Native target-interface
initialization plus EMA now raises the strict Phi-3 -> Qwen2.5 direction from a
plain NIB pass into a Qwen-LoRA-clearing result under two streams. This does
not prove arbitrary GPT5 -> GPT6 migration, but it removes a major reviewer
objection: the final NIB pass is no longer tuned on the same WikiText split
used to train or select the post-hoc scale, and the split-separated evidence is
no longer dependent on GPT-Neo as an endpoint.

## Matched LoRA/KD Baseline

`exp_lora_kd_baseline.py` adds a target-side PEFT comparator: train a native
target ABI oracle, inject LoRA into the frozen target backbone, train LoRA
against the oracle with the same KD/rank/top-set losses, and evaluate by the
same NIB L2 certificate. LoRA does not copy a frozen source domain core, so this
is an efficiency/control baseline rather than ABI portability evidence.

| Result | Target | LoRA | Trainable | Top-5 | Top-1 | JS | Entropy diff | Overall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `exp_lora_kd_baseline_qwen2p5_0p5b_attn_r11_wikitext_cal9600_seed42_results.json` | Qwen2.5-0.5B | attention r=11 | 1,486,848 | 0.8918 | 0.8780 | 0.0176 | 0.4516 | FAIL |
| `exp_lora_kd_baseline_qwen2p5_0p5b_attn_r11_wikitext_ent002_balanced_cal9600_seed42_results.json` | Qwen2.5-0.5B | attention r=11 + entropy | 1,486,848 | 0.8922 | 0.8785 | 0.0170 | 0.4347 | FAIL |
| `exp_lora_kd_baseline_qwen2p5_0p5b_all_r3_wikitext_ent002_balanced_cal9600_seed42_results.json` | Qwen2.5-0.5B | all-linear r=3 | 1,649,664 | 0.9107 | 0.9134 | 0.0157 | 0.4484 | FAIL |
| `exp_lora_kd_baseline_qwen2p5_0p5b_all_r3_wikitext_train_val_test_ent002_balanced_cal9600_seed42_results.json` | Qwen2.5-0.5B | all-linear r=3, train/validation/test | 1,649,664 | 0.9085 | 0.9102 | 0.0169 | 0.5030 | FAIL |

Current diagnosis: matched LoRA/KD can achieve strong rank agreement, and the
all-linear variants beat the ABI transfer on top-5 for this target. They still
miss the entropy/distribution gate by a wide margin, including under the same
train/validation/test split used by the ABI no-leakage certificate, while ABI
passes the full rank-plus-entropy certificate. This is useful adoption evidence
because it shows ABI is not just reproducing the easiest behavior of ordinary
target-side PEFT.

## Held-Out ABI vs LoRA Frontier

The first LoRA objection was real: the split-separated all-linear LoRA/KD
baseline had better rank metrics than the original held-out D512 ABI run,
although it failed the entropy gate. The follow-up frontier below keeps the
same train/validation/test split separation and tracks whether ABI can beat
that LoRA baseline without relaxing the NIB certificate.

| Result | D_ABI | Seed | Trainable | Top-5 | Top-1 | JS | Entropy diff | Overall | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `exp_lora_kd_baseline_qwen2p5_0p5b_all_r3_wikitext_train_val_test_ent002_balanced_cal9600_seed42_results.json` | n/a | 42 | 1,649,664 | 0.9085 | 0.9102 | 0.0169 | 0.5030 | FAIL | Split-separated all-linear LoRA/KD comparator; strong rank but fails entropy. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal9600_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | 512 | 42 | 1,443,841 | 0.8972 | 0.8963 | 0.0107 | 0.2992 | PASS | Original held-out ABI certificate passes NIB but loses to LoRA rank. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal16000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | 512 | 42 | 1,443,841 | 0.9031 | 0.8972 | 0.0111 | 0.3272 | PASS | Longer calibration improves top-5 but still loses LoRA rank. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d560_cal16000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | 560 | 42 | 1,632,961 | 0.9036 | 0.8907 | 0.0112 | 0.3323 | PASS | Capacity-matched ABI still does not beat LoRA rank. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal16000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | 768 | 42 | 2,558,977 | 0.9089 | 0.9045 | 0.0104 | 0.3240 | PASS | Wider ABI beats held-out LoRA top-5 and distribution metrics, but not top-1. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | 768 | 42 | 2,558,977 | 0.9094 | 0.9114 | 0.0104 | 0.3303 | PASS | First held-out ABI run to beat LoRA on all reported metrics. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | 768 | 314 | 2,558,977 | 0.9067 | 0.9093 | 0.0099 | 0.3059 | PASS | Shifted-seed repeat passes full NIB and beats LoRA JS/entropy, but does not repeat the LoRA rank win. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal19000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | 768 | 314 | 2,558,977 | 0.9060 | 0.9012 | 0.0106 | 0.3295 | PASS | More calibration on D768 moves rank in the wrong direction. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d768_cal20000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | 768 | 42 | 2,558,977 | 0.9080 | 0.9110 | 0.0113 | 0.3602 | FAIL | Longer calibration overfits rank/top-1 and breaks entropy. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d896_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | 896 | 314 | 3,214,849 | 0.9082 | 0.9106 | 0.0100 | 0.3118 | PASS | Target-width ABI fixes top-1 but misses the LoRA top-5 gate by 0.0003. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d896_cal18000_topsetw6_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | 896 | 314 | 3,214,849 | 0.9071 | 0.9073 | 0.0109 | 0.3383 | PASS | Increasing top-set weight hurts the repeat-seed rank metrics. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d1024_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | 1024 | 42 | 3,936,257 | 0.9076 | 0.9061 | 0.0104 | 0.3339 | PASS | Wider ABI passes NIB but loses the seed42 LoRA rank gate. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d1024_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | 1024 | 314 | 3,936,257 | 0.9106 | 0.9171 | 0.0095 | 0.2990 | PASS | D1024 beats LoRA strongly on seed314 but is not the repeated recipe. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_nativeinit_ema9995s6400_lrdecay6400x02_cal9600_topset5_bridge_wikitext_train_val_test_seed42_results.json` | 960 | 42 | 3,567,361 | 0.9140 | 0.9102 | 0.0084 | 0.2812 | PASS | Native target-interface initialization plus EMA matches the split-separated LoRA top-1 at 9.6k steps and beats LoRA on top-5, JS, and entropy. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_nativeinit_ema9995s8000_lrdecay8000x02_cal12000_topset5_bridge_wikitext_train_val_test_seed42_results.json` | 960 | 42 | 3,567,361 | 0.9162 | 0.9114 | 0.0084 | 0.2845 | PASS | Native target-interface initialization plus EMA beats LoRA on every reported metric at 12k steps, reducing the old repeated D960 calibration budget by one third. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_nativeinit_ema9995s8000_lrdecay8000x02_cal12000_topset5_bridge_wikitext_train_val_d7d416c519_results.json` | 960 | 42+100k | 3,567,361 | 0.9150 | 0.9142 | 0.0077 | 0.2512 | PASS | Shifted-stream repeat of the 12k native-init/EMA recipe also beats LoRA on every reported metric. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | 960 | 42 | 3,567,361 | 0.9110 | 0.9167 | 0.0105 | 0.3340 | PASS | Mid-width recipe beats LoRA on all reported metrics. |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d960_cal18000_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | 960 | 314 | 3,567,361 | 0.9113 | 0.9236 | 0.0100 | 0.3148 | PASS | Shifted-seed repeat of the D960 recipe also beats LoRA on all reported metrics. |
| `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s8000_lrdecay8000x02_cal12000_topset5_bridge_wikitext_train_val_test_seed42_results.json` | 1024 | 42 | 3,936,257 | 0.9140 | 0.9093 | 0.0085 | 0.2856 | PASS | Phi-3 donor into Qwen2.5 improves sharply over the older reverse-direction certificate and beats LoRA on top-5, JS, and entropy, but misses the LoRA top-1 gate by 0.0009. |
| `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_seed42_results.json` | 1024 | 42 | 3,936,257 | 0.9159 | 0.9167 | 0.0085 | 0.2909 | PASS | Extending Phi-3 -> Qwen2.5 native-init/EMA calibration to 14.4k clears the split-separated Qwen LoRA comparator on every reported metric. |
| `exp_generic_causal_nib_v2_phi3_qwen2p5_0p5b_d1024_nativeinit_ema9995s9600_lrdecay9600x02_cal14400_topset5_bridge_wikitext_train_val_test_f4fba487b3_results.json` | 1024 | 42+100k | 3,936,257 | 0.9163 | 0.9167 | 0.0080 | 0.2654 | PASS | Shifted-stream repeat of the 14.4k Phi-3 -> Qwen2.5 native-init/EMA recipe also beats LoRA on every reported metric. |

Current diagnosis: ABI now has a repeat-certified held-out recipe that beats
the split-separated LoRA/KD baseline across every reported metric while
preserving the full NIB pass. GPT-Neo -> Qwen now repeats the D960 all-metric
win at 12k calibration steps, down from the older 18k recipe. Phi-3 -> Qwen is
now an even stronger Qwen-target frontier: the 14.4k native-init/EMA recipe
reaches top-1 `0.9167`, top-5 `0.9159`, JS `0.0085`, and entropy diff `0.2909`
under seed42, and top-1 `0.9167`, top-5 `0.9163`, JS `0.0080`, and entropy diff
`0.2654` under shifted-stream seed42+offset100k. This is a materially stronger
adoption signal than the original D512 certificate, the first D768 one-off LoRA
win, and the previous 18k D960 repeat because it shows the native-init/EMA
mechanism carrying to another source family. The claim is still scoped: this
proves repeat-certified ABI-over-LoRA wins for recorded WikiText transfers, not
universal arbitrary-model domain migration.

## Phi-3 ABI vs LoRA Frontier

The next matched-baseline target is Phi-3. This is a harder adoption gate than
Qwen because the target is larger, the architecture is different, and the
target-side LoRA baseline now passes the full NIB certificate when trained to
the requested 4800 calibration steps.

| Result | Method | Seed | Trainable | Steps | Top-5 | Top-1 | JS | Entropy diff | Overall | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `exp_lora_kd_baseline_phi3_attn_r11_wikitext_train_val_test_topset5_balanced_cal4800_cap1800s_seed42_results.json` | Phi attention LoRA r=11 | 42 | 6,488,064 | 907/4800 | 0.8757 | 0.9118 | 0.0130 | 0.2430 | PASS | Historical early-stopped comparator; superseded by current full-step reruns below. |
| `exp_lora_kd_baseline_phi3_attn_r11_wikitext_train_val_test_topset5_balanced_cal4800_cap1800s_seed42_fullrerun_results.json` | Phi attention LoRA r=11 | 42 | 6,488,064 | 4800/4800 | 0.8920 | 0.9272 | 0.0105 | 0.2398 | PASS | Current fair seed42 comparator; raises the Phi top-1 gate substantially. |
| `exp_lora_kd_baseline_phi3_attn_r11_wikitext_train_val_test_topset5_balanced_cal4800_cap1800s_seed314_results.json` | Phi attention LoRA r=11 | 314 | 6,488,064 | 4800/4800 | 0.8902 | 0.9244 | 0.0119 | 0.2353 | PASS | Shifted-seed LoRA repeat confirms strong target-side PEFT rank transfer. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI D1024 | 42 | 8,392,705 | 4800 | 0.8885 | 0.9012 | 0.0104 | 0.1853 | PASS | ABI beats LoRA on top-5, JS, entropy, and elapsed time, but misses top-1. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | ABI D1024 | 314 | 8,392,705 | 4800 | 0.8880 | 0.9146 | 0.0111 | 0.1686 | PASS | Best ABI top-1 so far, but still below the current full-step LoRA comparator. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal7200_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI D1024 longer cal | 42 | 8,392,705 | 7200 | 0.8979 | 0.9053 | 0.0080 | 0.1692 | PASS | Best top-5/JS Phi ABI result so far, but still below LoRA top-1. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_top1gap2_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI top1-gap probe | 42 | 8,392,705 | 4800 | 0.8945 | 0.9073 | 0.0089 | 0.1739 | PASS | Surgical top-1 gap term improves top-5/JS but still misses LoRA top-1. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_kd098_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI KD/top-logit probe | 42 | 8,392,705 | 4800 | 0.8993 | 0.9053 | 0.0085 | 0.1733 | PASS | Best ABI top-5 against current Phi LoRA; still misses top-1 by 0.0219. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI validation-selected top-logit | 42 | 8,392,705 | selected/4800 | 0.8973 | 0.9093 | 0.0073 | 0.1403 | PASS | Best strict seed42 top-1 frontier; validation selection helped but did not close the LoRA gap. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset5_top1ce05_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI top1-CE probe | 42 | 8,392,705 | 4800 | 0.8977 | 0.9069 | 0.0085 | 0.1691 | PASS | Direct teacher-argmax CE improves top-5/JS but not enough rank-1 agreement. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_traindomain_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_e2bac75327_results.json` | ABI train-domain probe | 42 | 16,786,433 | 4800 | 0.8952 | 0.9061 | 0.0082 | 0.1677 | PASS | Unfreezing the copied domain core doubles trainable parameters and still misses LoRA top-1. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_logitres32_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_0805e6200b_results.json` | ABI logit residual r=32 | 42 | 9,451,521 | 4800 | 0.8956 | 0.9069 | 0.0091 | 0.1814 | PASS | Small ABI-to-vocab residual does not improve top-1. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_domainres256_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_tes_c1a09e6c90_results.json` | ABI domain residual r=256 | 42 | 8,916,993 | 4800 | 0.8908 | 0.9033 | 0.0097 | 0.1846 | PASS | ABI-space residual around the copied domain core worsens top-1. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_hiddenres256_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_tes_6acbeba56f_results.json` | ABI hidden residual r=256 | 42 | 9,965,569 | 4800 | 0.8963 | 0.9053 | 0.0084 | 0.1710 | PASS | Hidden target residual does not close the top-1 gap. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI Qwen2-1.5B source | 42 | 8,392,705 | 4800 | 0.8993 | 0.9171 | 0.0073 | 0.1508 | PASS | Larger Qwen source materially improves held-out Phi transfer. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed314_results.json` | ABI Qwen2-1.5B source | 314 | 8,392,705 | 4800 | 0.8952 | 0.9264 | 0.0084 | 0.1458 | PASS | Previous best Phi ABI top-1; beats Phi LoRA seed314 but misses the strongest Phi LoRA seed42 by 0.0008. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_pos_ea0e1c399f_results.json` | ABI Qwen2-1.5B source + 5000 align pairs | 314 | 8,392,705 | 4800 | 0.8951 | 0.9276 | 0.0084 | 0.1425 | PASS | Current Phi ABI top-1 frontier; beats the strongest full-step Phi LoRA comparator on top-1 while preserving better JS and entropy. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_pos_9147e6a278_results.json` | ABI Qwen2-1.5B 5000-align repeat | 42 | 8,392,705 | 4800 | 0.8924 | 0.9049 | 0.0080 | 0.1586 | PASS | Seed42 repeat of the 5000-align recipe passes NIB but does not repeat the LoRA top-1 win. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed314off0_results.json` | ABI Qwen2-1.5B seed314 + seed42 streams | 314 | 8,392,705 | 4800 | 0.8953 | 0.9028 | 0.0078 | 0.1607 | PASS | Seed314 initialization on seed42 data/eval streams drops to seed42-like top-1, so the high frontier is not explained by initialization alone. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42off100k_results.json` | ABI Qwen2-1.5B seed42 + shifted streams | 42 | 8,392,705 | 4800 | 0.8819 | 0.9081 | 0.0111 | 0.1512 | PASS | Seed42 initialization on shifted data/eval streams also misses LoRA top-1; the current high run is a seed/stream interaction. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calsoup3_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json` | ABI Qwen2-1.5B checkpoint soup top3 | 42 | 8,392,705 | 4800 | 0.9087 | 0.9163 | 0.0062 | 0.1485 | PASS | Validation checkpoint soup improves seed42 top-5/JS strongly, but not enough top-1. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup3_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json` | ABI Qwen2-1.5B 5000-align soup top3 | 42 | 8,392,705 | 4800 | 0.9063 | 0.9163 | 0.0060 | 0.1442 | PASS | Checkpoint soup repairs the 5000-align seed42 distribution/top-5 gap but still misses LoRA top-1. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json` | ABI Qwen2-1.5B 5000-align soup top2 | 42 | 8,392,705 | 4800 | 0.9052 | 0.9199 | 0.0062 | 0.1450 | PASS | Equal top-two soup is a strong seed42 stabilizer but still below the Phi LoRA seed42 top-1 of 0.9272. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_tr_3f2bbd7648_results.json` | ABI Qwen2-1.5B 5000-align soup top2 weighted | 42 | 8,392,705 | 4800 | 0.9046 | 0.9228 | 0.0062 | 0.1448 | PASS | Best seed42 stabilization so far; still misses Phi LoRA seed42 top-1 by 0.0044. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_selaudit8_calselect600_cal4800_topset5_toplogitmse005_bridge_w_9b3a5db194_results.json` | ABI Qwen2-1.5B 5000-align checkpoint audit | 42 | 8,392,705 | 4800 | 0.9046 | 0.9228 | 0.0062 | 0.1448 | PASS | Diagnostic audit shows no individual validation checkpoint clears LoRA; the interpolation is doing the useful work. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_soupaudit_curve_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_br_73a463fa02_results.json` | ABI Qwen2-1.5B 5000-align soup curve audit | 42 | 8,392,705 | 4800 | 0.9046 | 0.9228 | 0.0062 | 0.1448 | PASS | Held-out diagnostic weight sweep peaks at the same 0.35/0.65 soup and does not hide a LoRA-clearing blend. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_accum2s100k_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_2bb63bdc3b_results.json` | ABI Qwen2-1.5B 5000-align two-stream accumulation | 42 | 8,392,705 | 4800 | 0.9020 | 0.9134 | 0.0068 | 0.1566 | PASS | Averaging independent calibration streams over-smooths held-out rank and falls below the weighted-soup frontier. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_top1hn5_calselect600_cal4800_topset5_toplogitmse005_bridge_wik_0e9bc095bb_results.json` | ABI Qwen2-1.5B 5000-align top1-hard-neg | 42 | 8,392,705 | 4800 | 0.9004 | 0.9163 | 0.0069 | 0.1545 | PASS | Direct rank-1 hard-negative pressure raises validation top-1 but hurts held-out transfer. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_ent002_calselect600_cal4800_topset5_toplogitmse005_bridge_wiki_c261359397_results.json` | ABI Qwen2-1.5B 5000-align entropy regularized | 42 | 8,392,705 | 4800 | 0.9020 | 0.9150 | 0.0070 | 0.1532 | PASS | Light entropy matching does not repair the Phi seed42 repeat; distribution shape is already good enough that rank remains the blocker. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_confmargin075_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_brid_d3936d78fb_results.json` | ABI Qwen2-1.5B high-margin weighted aux | 42 | 8,392,705 | 4800 | 0.9091 | 0.9215 | 0.0057 | 0.1290 | PASS | Teacher high-margin weighting improves top-5/JS/entropy but lowers top-1 versus the weighted-soup frontier. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_conflowmargin075_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_b_7f1f1663d1_results.json` | ABI Qwen2-1.5B low-margin weighted aux | 42 | 8,392,705 | 4800 | 0.8966 | 0.9061 | 0.0074 | 0.1536 | PASS | Emphasizing ambiguous low-margin teacher tokens sharply lowers held-out rank. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_rank5_hard5_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_fb1f28e91d_results.json` | ABI Qwen2-1.5B lower rank/hard pressure | 42 | 8,392,705 | 4800 | 0.9038 | 0.9150 | 0.0067 | 0.1527 | PASS | Reducing pairwise pressure from 10/10 to 5/5 lowers top-1, so the baseline is not simply over-constrained. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_topset3_calsoup2w035065_calselect600_cal4800_toplogitmse005_bridge_wikitext_tr_75edb763f3_results.json` | ABI Qwen2-1.5B top-set K=3 | 42 | 8,392,705 | 4800 | 0.8959 | 0.9020 | 0.0088 | 0.1853 | PASS | Narrowing the listwise top-set from 5 to 3 hurts both rank and distribution metrics. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_instruct_phi3_d1024_align5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wi_bffe331f2e_results.json` | ABI Qwen2-1.5B-Instruct donor | 42 | 8,392,705 | 4800 | 0.9028 | 0.9142 | 0.0065 | 0.1480 | PASS | Instruction-tuned source prior passes NIB but lowers top-1 versus the base Qwen2-1.5B weighted-soup frontier, so it does not fix the Phi seed42 LoRA gap. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_abistatemse005_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bri_8a85a5653c_results.json` | ABI Qwen2-1.5B post-domain ABI-state MSE | 42 | 8,392,705 | 4800 | 0.8985 | 0.9102 | 0.0072 | 0.1567 | PASS | Matching the native target ABI post-domain state over-constrains the interface and worsens held-out rank versus the weighted-soup frontier. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrcos4800x02to005_cal7200_topset5_toplogitmse005_bridge_wikitext_t_50a00ed4d2_results.json` | ABI Qwen2-1.5B EMA + late cosine LR | 42 | 8,392,705 | 7200 | 0.9106 | 0.9224 | 0.0060 | 0.1523 | PASS | Smoother post-4800 LR annealing improves top-5 but lowers top-1 versus the step-decay EMA frontier. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_temporalavg4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_6d6771b128_results.json` | ABI Qwen2-1.5B temporal avg after 4800 | 42 | 8,392,705 | 7200 | 0.9101 | 0.9195 | 0.0061 | 0.1522 | PASS | Validation-independent late checkpoint averaging is too broad and lowers held-out top-1. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_abipremse0005_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_1afc5333cf_results.json` | ABI Qwen2-1.5B pre-domain ABI MSE + EMA | 42 | 8,392,705 | 7200 | 0.9099 | 0.9220 | 0.0059 | 0.1505 | PASS | Light pre-domain ABI-coordinate matching still lowers top-1 versus the EMA frontier. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_stabletop1ce015_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_br_0de1f71cc4_results.json` | ABI Qwen2-1.5B stable-token top1 CE | 42 | 8,392,705 | 4800 | 0.9037 | 0.9179 | 0.0065 | 0.1474 | PASS | Filtering argmax CE to native base/domain-agreeing tokens still over-constrains rank and lowers top-1. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_domaindeltamse005_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_7d19fee8f5_results.json` | ABI Qwen2-1.5B domain-delta logit MSE | 42 | 8,392,705 | 4800 | 0.9028 | 0.9195 | 0.0063 | 0.1459 | PASS | Matching native domain-on/domain-off logit deltas improves PPL but does not fix held-out rank-1 stability. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_uniontopk1_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_68beec8994_results.json` | ABI Qwen2-1.5B union top-k KD | 42 | 8,392,705 | 4800 | 0.9045 | 0.9175 | 0.0062 | 0.1398 | PASS | Adding current student top candidates to KD raises validation top-1 but lowers held-out top-1, reinforcing the validation-overfit diagnosis. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2grid_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_7b6bb22a32_results.json` | ABI Qwen2-1.5B 5000-align validation soup grid | 42 | 8,392,705 | 4800 | 0.9035 | 0.9187 | 0.0063 | 0.1460 | PASS | Validation selected 0.60/0.40, but held-out top-1 trailed the manual 0.35/0.65 soup. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2gridr5min_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_79329f3642_results.json` | ABI Qwen2-1.5B 5000-align validation soup grid r5 min | 42 | 8,392,705 | 4800 | 0.9035 | 0.9187 | 0.0063 | 0.1460 | PASS | Five repeated validation samples with worst-score selection still choose 0.60/0.40, so validation sample count alone is not the fix. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w030070_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_tr_903aafec1b_results.json` | ABI Qwen2-1.5B 5000-align soup top2 weighted 0.30/0.70 | 42 | 8,392,705 | 4800 | 0.9051 | 0.9220 | 0.0062 | 0.1449 | PASS | Narrow weight probe stays close to the frontier but does not beat 0.35/0.65. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align10000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_t_df9bc8338d_results.json` | ABI Qwen2-1.5B 10000-align soup top2 weighted | 42 | 8,392,705 | 4800 | 0.9007 | 0.9150 | 0.0066 | 0.1499 | PASS | More raw alignment pairs lowers Procrustes cosine and held-out rank, so count alone is not the fix. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000min80_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikite_dd5469f7de_results.json` | ABI Qwen2-1.5B 5000-align min80 soup top2 weighted | 42 | 8,392,705 | 4800 | 0.9050 | 0.9150 | 0.0067 | 0.1521 | PASS | Longer-sentence alignment filter also lowers Procrustes cosine and top-1, so simple length filtering is not the fix. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_aligntrim10000to5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_ecabcc581b_results.json` | ABI Qwen2-1.5B trim 10000->5000 soup top2 weighted | 42 | 8,392,705 | 4800 | 0.8998 | 0.9163 | 0.0069 | 0.1526 | PASS | Geometry-aware trim raises final Procrustes cosine to 0.8264 but still misses held-out rank, so cosine alone is not sufficient. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000zscore_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikit_0497f88804_results.json` | ABI Qwen2-1.5B z-score Procrustes soup top2 weighted | 42 | 8,392,705 | 4800 | 0.9024 | 0.9126 | 0.0067 | 0.1517 | PASS | Variance-normalized alignment improves validation checkpoint top-1 but hurts held-out rank. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_tr_5aa3054d33_results.json` | ABI Qwen2-1.5B 5000-align soup top2 weighted, shifted streams | 42 | 8,392,705 | 4800 | 0.8973 | 0.9163 | 0.0093 | 0.1452 | PASS | Shifted-stream repeat passes NIB but does not repeat the LoRA top-1 win. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_calselect600_cal7200_topset5_toplogitmse005_bridge_wikitext_tr_9f55f807af_results.json` | ABI Qwen2-1.5B 5000-align soup top2 weighted, 7200 steps | 42 | 8,392,705 | 7200 | 0.9031 | 0.9142 | 0.0066 | 0.1539 | PASS | Longer calibration creates high validation top-1 checkpoints but lowers held-out rank. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2w035065_calselect600_cal7200_lrdecay4800x02_topset5_toplogitmse005_bri_8773460822_results.json` | ABI Qwen2-1.5B 5000-align soup top2 weighted, 7200 steps + LR decay | 42 | 8,392,705 | 7200 | 0.9067 | 0.9220 | 0.0060 | 0.1511 | PASS | LR decay after step 4800 reduces longer-calibration overfit, but still trails the best 4800-step seed42 soup and Phi LoRA top-1 gate. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_9af0dbe9c4_results.json` | ABI Qwen2-1.5B 5000-align EMA after LR decay | 42 | 8,392,705 | 7200 | 0.9096 | 0.9244 | 0.0059 | 0.1502 | PASS | EMA over the post-4800 LR-decayed phase gives the best seed42 top-5 so far and nearly closes the LoRA seed314 top-1 gate, but still misses the strongest Phi LoRA seed42 top-1. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_e35b341e1b_results.json` | ABI Qwen2-1.5B 5000-align EMA after LR decay, shifted streams | 42 | 8,392,705 | 7200 | 0.9119 | 0.9329 | 0.0065 | 0.1300 | PASS | Shifted-stream EMA repeat beats the strongest Phi LoRA seed42 comparator on top-1, top-5, JS, and entropy, but offset-0 EMA still sits below the top-1 gate. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_finalsoupaudit_selectedema_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogit_d6083f39ef_results.json` | ABI Qwen2-1.5B selected/EMA blend audit | 42 | 8,392,705 | 7200 | 0.9096 | 0.9244 | 0.0059 | 0.1502 | PASS | Diagnostic final-candidate blend shows pure EMA is the best selected/EMA blend; interpolation does not clear LoRA. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_711b332ded_results.json` | ABI Qwen2-1.5B 5000-align EMA after LR decay, seed314 | 314 | 8,392,705 | 7200 | 0.9067 | 0.9167 | 0.0061 | 0.1537 | PASS | Forced EMA hurts the seed314 repeat relative to the non-EMA seed314 frontier, so EMA is not a uniform stabilization mechanism. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_finalselect_val4r3_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_b_d7aed9f544_results.json` | ABI Qwen2-1.5B final selector over soup/EMA/best/final | 314 | 8,392,705 | 7200 | 0.9067 | 0.9167 | 0.0061 | 0.1537 | PASS | Validation-only final selector chose EMA, but held-out stayed poor; selector validation remains misaligned with transfer generalization. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_finalselectaudit_val4r3_ema999s4800_lrdecay4800x02_cal7200_topset5_toplogitmse_bbd9590a6c_results.json` | ABI Qwen2-1.5B final selector audit | 314 | 8,392,705 | 7200 | 0.9067 | 0.9167 | 0.0061 | 0.1537 | PASS | Diagnostic NIB audit shows selected soup 0.9142, EMA 0.9167, best checkpoint 0.9106, and final checkpoint 0.9122 top-1; the seed314 miss is not only final-state selection. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4200_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_a54c880214_results.json` | ABI Qwen2-1.5B EMA before LR decay | 42 | 8,392,705 | 7200 | 0.9089 | 0.9224 | 0.0061 | 0.1526 | PASS | Starting EMA at step 4200 includes the pre-decay plateau but lowers held-out top-1 versus the step-4800 EMA. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s5400_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_b8d6a94277_results.json` | ABI Qwen2-1.5B 5000-align EMA after LR decay, later start | 42 | 8,392,705 | 7200 | 0.9096 | 0.9236 | 0.0061 | 0.1527 | PASS | Starting EMA at step 5400 lowers offset-0 top-1 versus step 4800, so the useful smoothing window includes the early LR-decayed phase. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema9995s4800_lrdecay4800x02_cal7200_topset5_toplogitmse005_bridge_wikitext_tra_33028a26d6_results.json` | ABI Qwen2-1.5B 5000-align EMA after LR decay, slower decay | 42 | 8,392,705 | 7200 | 0.9106 | 0.9244 | 0.0058 | 0.1462 | PASS | Slower EMA decay improves top-5/JS slightly but leaves top-1 unchanged below the strongest Phi LoRA gate. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_ema999s4800_lrdecay4800x01_cal7200_topset5_toplogitmse005_bridge_wikitext_trai_2de76e65c0_results.json` | ABI Qwen2-1.5B 5000-align EMA after stronger LR decay | 42 | 8,392,705 | 7200 | 0.9102 | 0.9236 | 0.0059 | 0.1499 | PASS | Decaying LR to 0.1 after step 4800 lowers top-1 versus the 0.2 factor, so the remaining offset-0 gap is not fixed by a smaller late LR. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calselect600_cal6000_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json` | ABI Qwen2-1.5B 5000-align selected checkpoint, 6000 steps | 42 | 8,392,705 | 6000 | 0.8980 | 0.9110 | 0.0076 | 0.1567 | PASS | The validation top-1 spike at step 6000 does not generalize to held-out test. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2_calselect600x4_cal4800_topset5_toplogitmse005_bridge_wikitext_train_v_65c035b778_results.json` | ABI Qwen2-1.5B 5000-align soup top2, val x4 | 42 | 8,392,705 | 4800 | 0.8997 | 0.9102 | 0.0069 | 0.1521 | PASS | More validation chunks worsen held-out top-1, so selection-estimate size alone is not the fix. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_calsoup2_top1gap2_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_a407bebc53_results.json` | ABI Qwen2-1.5B 5000-align soup top2 + top1 gap | 42 | 8,392,705 | 4800 | 0.9028 | 0.9142 | 0.0068 | 0.1535 | PASS | Light top-1 margin pressure hurts relative to soup alone. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_dom1000_align5000_calsoup2_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_t_5cf86fa792_results.json` | ABI Qwen2-1.5B 1000 domain steps + soup top2 | 42 | 8,392,705 | 4800 | 0.9029 | 0.9118 | 0.0067 | 0.1618 | PASS | Stronger domain-core training improves oracle PPL but not rank-1 stability. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_align5000_final4800_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json` | ABI Qwen2-1.5B 5000-align final checkpoint | 42 | 8,392,705 | 4800 | 0.8974 | 0.9154 | 0.0072 | 0.1500 | PASS | The final checkpoint trails top-two checkpoint soup, confirming soup is the better seed42 stabilizer. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1280_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI Qwen2-1.5B D1280 width probe | 42 | 11,146,241 | 4800 | 0.8908 | 0.9122 | 0.0082 | 0.1599 | PASS | More ABI width adds capacity but does not repair seed42 top-1. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal7200_topset5_toplogitmse005_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI Qwen2-1.5B longer cal | 42 | 8,392,705 | 7200 | 0.9000 | 0.9093 | 0.0075 | 0.1591 | PASS | Longer calibration improves top-5 to 0.9000 but still misses LoRA top-1. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI Qwen2-1.5B no top-logit | 42 | 8,392,705 | 4800 | 0.8962 | 0.9126 | 0.0076 | 0.1590 | PASS | Removing centered top-logit MSE lowers seed42 top-1 relative to the top-logit recipe. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_top1ce025_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json` | ABI Qwen2-1.5B top1-CE | 42 | 8,392,705 | 4800 | 0.8911 | 0.9041 | 0.0084 | 0.1648 | PASS | Direct teacher-argmax CE worsens seed42 top-1, so the issue is not solved by stronger argmax pressure. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_logitres32_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_seed42_results.json` | ABI Qwen2-1.5B logit residual r=32 | 42 | 9,451,521 | 4800 | 0.8932 | 0.9098 | 0.0080 | 0.1600 | PASS | A small ABI-to-vocab residual does not repair seed42 top-1. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_phi3_d1024_calselect600_cal4800_topset5_toplogitmse005_posthocbias300_bridge_wikitext_train_val_test_seed42_results.json` | ABI Qwen2-1.5B posthoc bias | 42 | 8,392,705 | 4800 | 0.8995 | 0.9093 | 0.0091 | 0.2051 | PASS | Validation-only global logit bias improves top-5 but not seed42 top-1. |
| `exp_generic_causal_nib_v2_qwen2_7b_phi3_d1024_release_source_calselect600_cal4800_topset5_toplogitmse005_bridge_wikitext_train_val_test_8e31294bc8_results.json` | ABI Qwen2-7B source | 42 | 8,392,705 | 4800 | 0.8869 | 0.8996 | 0.0090 | 0.1729 | PASS | Sequential source-release makes the 7B donor feasible on 16GB, but this donor underperforms Qwen2-1.5B. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1024_cal4800_topset1_rankpos1_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI top1-only ablation | 42 | 8,392,705 | 4800 | 0.8662 | 0.8687 | 0.0165 | 0.2588 | PASS | Over-focusing top-1 harms both top-1 and top-5. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_phi3_d1536_cal4800_topset5_bridge_wikitext_train_val_test_posthocminimax_seed42_results.json` | ABI D1536 width probe | 42 | 14,161,921 | 4800 | 0.8834 | 0.8878 | 0.0117 | 0.1876 | PASS | More width hurts this direction. |

Current diagnosis: the immediate Phi-3 LoRA top-1 gap is closed by one held-out
frontier run, but not repeat-certified. Increasing sentence-level Procrustes
alignment from 2000 to 5000 pairs moves Qwen2-1.5B -> Phi-3 seed314 from top-1
0.9264 to 0.9276, narrowly above the strongest full-step Phi LoRA seed42 top-1
of 0.9272 while staying stronger on JS and entropy. The same 5000-align recipe
at seed42 reaches only top-1 0.9049. Crossed seed/stream diagnostics show the
frontier is not explained by initialization alone: seed314 on the seed42
streams reaches top-1 0.9028, while seed42 on the shifted streams reaches top-1
0.9081. Validation checkpoint soup is the first stability mechanism that moves
the repeat materially in the right direction: the best seed42 soup is now a
0.35/0.65 weighted average of the top validation checkpoints and reaches top-1
0.9228, top-5 0.9046, JS 0.0062, and entropy diff 0.1448. It still misses the
full-step Phi LoRA seed42 top-1 of 0.9272 by 0.0044. New negative controls make
that gap more concrete: two-stream gradient accumulation falls to top-1 0.9134,
top1-hard-negative pressure falls to 0.9163 despite stronger validation top-1,
light entropy matching falls to 0.9150, high-margin confidence weighting falls
to 0.9215, low-margin confidence weighting falls to 0.9061, rank/hard pressure
5/5 falls to 0.9150, and top-set K=3 falls to 0.9020. Those results rule out
simple batch smoothing, direct rank-1 pressure, gross entropy mismatch,
teacher-margin reweighting, lower pairwise pressure, and a narrower top-set as
the missing mechanism. A Qwen2-1.5B-Instruct donor also passes NIB but drops to
top-1 0.9142, ruling out instruction-tuned source prior as the missing Phi
stability mechanism. A post-domain ABI-state MSE constraint against the native
target ABI oracle drops further to top-1 0.9102, showing that direct internal
state matching at weight 0.05 over-constrains the copied-core interface rather
than improving rank generalization. Three late-phase stabilization probes also
miss: cosine annealing the post-4800 LR phase reaches top-1 0.9224,
validation-independent temporal averaging over the late checkpoints reaches
0.9195, and weak pre-domain ABI-coordinate matching plus EMA reaches 0.9220.
That rules out smooth late LR decay, broad temporal checkpoint averaging, and
light direct ABI-coordinate matching as sufficient fixes. Three newer
mechanism probes also miss: stable-token top1 CE reaches 0.9179, domain-delta
logit MSE reaches 0.9195 while improving calibrated PPL, and union top-k KD
reaches 0.9175 despite higher validation top-1. Those runs rule out filtered
argmax pressure, direct domain-effect logit matching, and student-false-positive
union KD as sufficient Phi rank-generalization fixes. Alignment-ensemble probes
also miss: uniform three-rotation averaging reaches top-1 0.9179/top-5 0.9062,
and trainable three-scalar ensemble weighting reaches top-1 0.9163 while
keeping weights nearly uniform. Validation-checkpoint audit shows the best individual
checkpoint reaches only held-out top-1 0.9146, so the 0.9228 gain is coming
from interpolation rather than a hidden checkpoint. A held-out diagnostic sweep
over the top-two checkpoint soup peaks at the existing 0.35/0.65 blend and does
not contain a LoRA-clearing state. Validation-selected
soup-weight grids chose 0.60/0.40 on validation and fell to held-out top-1
0.9187; repeating the validation grid five times with worst-score selection
still chose the same held-out-weaker blend. A nearby manual 0.30/0.70 probe
reached 0.9220. Raising raw Procrustes alignment count to 10000 lowered the
alignment cosine and held-out top-1 to 0.9150; filtering to longer sentences
also lowered cosine and top-1 to 0.9150. These probes show that naive alignment
count or length filters are not the missing stability mechanism. Geometry-aware
Procrustes trimming raises final alignment cosine to 0.8530 but still lands at
top-1 0.9159, so alignment cosine alone is not sufficient. Z-score normalized
Procrustes fit reaches strong validation top-1 but falls to held-out top-1
0.9126. Longer 7200-step calibration creates a validation top-1 spike at step
6000, but the 6000 checkpoint itself reaches only held-out top-1 0.9110 and the
7200 checkpoint soup reaches 0.9142. A calibration LR decay after step 4800
recovers much of the lost held-out rank, reaching top-1 0.9220, top-5 0.9067,
JS 0.0060, and entropy diff 0.1511, but it still trails the best 4800-step
seed42 weighted soup and the Phi LoRA top-1 gate. EMA over the post-4800
LR-decayed calibration phase improves further to top-1 0.9244, top-5 0.9096,
JS 0.0059, and entropy diff 0.1502. This is the best seed42 top-5 result so
far and a real stability gain, but it still misses the strongest full-step Phi
LoRA seed42 top-1 of 0.9272. The same EMA curriculum under shifted streams
reaches top-1 0.9329, top-5 0.9119, JS 0.0065, and entropy diff 0.1300,
clearing the strongest Phi LoRA comparator on every reported metric in that
repeat. This is the clearest evidence so far that the hard Phi gap can be
crossed without loosening the frozen-core protocol. It is still not a uniform
recipe certificate, because offset-0 EMA remains below the top-1 gate and the
second shifted-stream repeat at offset200k reaches only top-1 0.9224 with
forced EMA, 0.9220 without forced EMA, and 0.9195 when restoring the single
best validation checkpoint. A matched offset200k LoRA run stopped early at
1230/4800 steps under the train cap, so it is a partial comparator only; ABI
beats that partial run, but this does not create a full-step LoRA repeat
certificate. Starting the
offset-0 EMA earlier at step 4200 drops held-out top-1 to 0.9224, while
starting later at step 5400 drops it to 0.9236; the best window is still start
4800. Slowing EMA decay to 0.9995 improves top-5/JS slightly but leaves top-1
at 0.9244, and selected/EMA final-candidate blending also peaks at pure EMA.
Lowering the post-4800 LR decay factor from 0.2 to 0.1 drops top-1 to 0.9236,
so the remaining offset-0 gap is not due to excessive late learning rate alone.
A validation-only final selector over selected soup, EMA, best checkpoint, and
final checkpoint chose EMA on seed314, but held-out top-1 remained 0.9167;
therefore the validation selector itself is still misaligned with the transfer
metric. The diagnostic final NIB audit confirms the deeper problem: selected
soup, EMA, best checkpoint, and final checkpoint all miss the LoRA top-1 gate
on seed314, so that run needs a different calibration trajectory rather than
only a better final-state selector.
Repeating checkpoint validation five times with four chunks per repeat and
ranking by worst score selects steps 3000/4200 and drops held-out top-1 to
0.9081, so more validation sampling alone does not align the selector with NIB.
Relaxing the ABI basis map from pure Procrustes to a 0.25 ridge-linear blend
raises alignment cosine to 0.7484 and validation top-1 to 0.9217, but held-out
top-1 is only 0.9146; simple scale/shear in the alignment map is not the
missing mechanism either.
Extra validation chunks, top1-gap pressure, longer domain-core training, and
final-checkpoint selection all trail the weighted top-two soup. Seed42 width,
longer calibration, no-toplogit, top1-CE, logit-residual, validation global
bias, and rank-64 ABI residual probes also fail to close the top-1 gate. The
global-bias run falls sharply to top-1 0.8927, while the rank-64 residual
reaches top-1 0.9228/top-5 0.9111/JS 0.0060 and still misses both the EMA
frontier and the strongest Phi LoRA gate. The stricter copied-core branch
(`freeze_all_domain` plus frozen `domain_alpha`) reproduces the EMA plateau at
top-1 0.9244 with fewer D-phase trainable parameters, and its validation-selected
EMA/final soup reaches only top-1 0.9240, so copied-core amplitude retuning is
not hiding the remaining rank points. Native target-interface initialization
(`ABI_CAL_INIT=native`) is the first recent vertical improvement: the forced-EMA
seed42 run reaches top-1 0.9264, top-5 0.9232, JS 0.0047, and entropy diff
0.1468. It is still 0.0008 below the strongest Phi LoRA top-1 gate, while the
native-init selected soup drops to top-1 0.9240 and the native-init 600-step
checkpoint drops to top-1 0.9134. Slower EMA then closes the scoped Phi gate:
native init plus EMA decay 0.9995 reaches top-1 0.9285, top-5 0.9229, JS
0.0047, and entropy diff 0.1466 under seed42, and the shifted-stream repeat
reaches top-1 0.9378, top-5 0.9266, JS 0.0046, and entropy diff 0.1205. This is
a repeat-certified all-metric ABI-over-LoRA win for Qwen2-1.5B -> Phi-3
WikiText. It is real progress, not a production-readiness claim. The next
blocker is broader architectural and domain generalization: one follow-up now
shows the native-init/EMA recipe also reduces the GPT-Neo -> Qwen D960
calibration budget from 18k to 12k while preserving the repeat-certified
all-metric LoRA win, and another follow-up turns Phi-3 -> Qwen2.5 into a
repeat-certified Qwen-LoRA-clearing transfer at 14.4k steps. The repo still
needs non-WikiText domains and broader matched baseline classes before any
production-readiness claim.

## GPT-Neo/Qwen2.5 Directed Probe

The generic runner also certified a smaller bidirectional cross-family pair
that was not covered by the original hand-written experiments: GPT-Neo-125M
and Qwen2.5-0.5B. GPT-Neo uses the GPT-2 tokenizer family, so the runner uses
the cached GPT-2 tokenizer override for GPT-Neo while keeping the model weights
and ABI transfer protocol unchanged.

Both directions use the current strong frozen-core certification recipe:
`D_ABI=512`, 500 source/native domain steps, 4800 target calibration steps,
top-k KD (`k=128`, weight 2.0), teacher-rank margin, student hard-negative
suppression, full-vocabulary teacher top-set loss, and an identity-initialized
target-side domain bridge.

| Result | Source -> Target | D_ABI | Calibration mode | Top-5 | Overall | Interpretation |
| --- | --- | ---: | --- | ---: | --- | --- |
| `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_seed42_results.json` | GPT-Neo-125M -> Qwen2.5-0.5B | 512 | frozen domain core + bridge/top-set | 0.9154 | PASS | GPT-Neo family source transfers strongly into Qwen2.5 target. |
| `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal4800_topset5_bridge_seed42_results.json` | Qwen2.5-0.5B -> GPT-Neo-125M | 512 | frozen domain core + bridge/top-set | 0.8821 | PASS | Qwen2.5 source transfers back into GPT-Neo target. |

## Phi-3 Probe

The generic runner now supports configurable target dtype, experiment batch
size, and repeat-run seed controls so larger cached decoder-only targets can be
tested without changing the copy/paste ABI protocol. Phi-3 was run with a
frozen rotated domain MLP core, `D_ABI=1024`, 500 source/native domain steps,
2400 target calibration steps, top-k KD, fp16 model weights, and batch 1.

| Result | Source -> Target | D_ABI | Calibration mode | Top-5 | Overall | Interpretation |
| --- | --- | ---: | --- | ---: | --- | --- |
| `exp_generic_causal_nib_v2_gpt2med_to_phi3_mini_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json` | GPT-2-medium -> Phi-3-mini-4k-instruct | 1024 | frozen domain core + interface top-k KD | 0.9296 | PASS | New-family Phi-3 transfer passes with very high rank fidelity while the copied domain MLP core remains frozen. |
| `exp_generic_causal_nib_v2_gpt2med_to_phi3_mini_d1024_dom500_cal2400_topk32w1_fp16_b1_seed314off100k_freeze_domain_net_alpha_stable_results.json` | GPT-2-medium -> Phi-3-mini-4k-instruct | 1024 | frozen domain core + interface top-k KD | 0.9319 | PASS | Shifted initialization, train batches, PPL batches, and NIB chunks still pass. |
| `exp_generic_causal_nib_v2_qwen2_1p5b_to_phi3_mini_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json` | Qwen2-1.5B -> Phi-3-mini-4k-instruct | 1024 | frozen domain core + interface top-k KD | 0.9264 | PASS | Phi-3 can receive a copied domain core from a second non-GPT source family. |
| `exp_generic_causal_nib_v2_phi3_mini_to_qwen2_1p5b_d1024_dom500_cal2400_topk32w1_fp16_b1_freeze_domain_net_alpha_stable_results.json` | Phi-3-mini-4k-instruct -> Qwen2-1.5B | 1024 | frozen domain core + interface top-k KD | 0.9312 | PASS | Phi-3 can also act as the source domain core into a Qwen2 target. |
| `exp_generic_causal_nib_v2_gptneo125m_phi3_d1024_cal4800_topset5_bridge_seed42_results.json` | GPT-Neo-125M -> Phi-3-mini-4k-instruct | 1024 | frozen domain core + bridge/top-set | 0.9150 | PASS | GPT-Neo uses the GPT-2 tokenizer override but a distinct GPT-Neo source backbone; it transfers strongly into Phi-3 under the stronger certification recipe. |
| `exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal4800_topset5_bridge_seed42_results.json` | Phi-3-mini-4k-instruct -> GPT-Neo-125M | 1024 | frozen domain core + bridge/top-set | 0.8831 | PASS | Reverse GPT-Neo/Phi-3 direction also passes, making this pair bidirectional under the strict frozen-core certification recipe. |

This is the strongest cross-family evidence so far because it includes a larger
Phi-3 architecture (`d_model=3072`, model type `phi3`), a shifted-seed repeat,
three source families feeding Phi-3 plus reverse Phi-3 -> Qwen2 and
Phi-3 -> GPT-Neo directions.
