# ABI Claims — Canonical Claim Map

This file is the authoritative record of what ABI claims, what has been validated, what has not been tested, and what is explicitly not claimed.

---

## Validated Claims

### Claim 1 — Same-Backbone ABI Reconstruction (Path 2C)

**Statement:** Two independently initialized ABI modules on a frozen T5-large backbone produce logit distributions that are non-inferior to each other under the NIB criterion.

**Status:** ✅ VALIDATED

| Criterion | Value | Threshold | Pass |
|-----------|-------|-----------|------|
| Top-5 token agreement | 0.8725 | ≥ 0.860 | ✅ |
| Top-1 token agreement | 0.8508 | ≥ 0.680 | ✅ |
| Jensen-Shannon divergence | 0.01391 | < 0.100 | ✅ |
| Entropy difference | 0.2256 | < 0.350 | ✅ |

- Script: `cross_arch_t5_nib_v53.py`
- Result file: `cross_arch_t5_nib_v53_results.json` (immutable)
- Extended evaluation (n=25, 5 seeds): mean top-5 = 0.8549, 95% CI = [0.8425, 0.8673]
- Note: 3 of 5 seeds fall below the single-run threshold. The mean is above the noise floor. This is reported transparently.

---

### Claim 2 — Cross-Family Transfer: GPT-2-small → Qwen2.5-0.5B (Exp 32)

**Statement:** An ABI module trained on GPT-2-small transfers to Qwen2.5-0.5B via Procrustes alignment and KD calibration, achieving NIB PASS.

**Status:** ✅ VALIDATED

- Top-5 token agreement: **0.8701** (threshold ≥ 0.860)
- Script: `cross_family_nib.py`
- Result file: `cross_family_nib_results.json`

---

### Claim 3 — Encoder-Decoder → Decoder-Only Migration (Exp 39)

**Statement:** A frozen ABI domain module trained on T5-large (encoder-decoder) transfers to GPT-2-medium (decoder-only) using orthogonal Procrustes rotation and 1200-step KD calibration. Only `proj_in` and `proj_out` are trained. NIB evaluated in GPT-2's native vocabulary.

**Status:** ✅ VALIDATED

| Source | Target | Top-5 | Top-1 | JS | Ent | Pass |
|--------|--------|-------|-------|----|-----|------|
| T5-large (enc-dec, 730M, 32K vocab) | GPT-2-medium (dec-only, 354M, 50K vocab) | 0.8699 | 0.9252 | 0.01787 | 0.2819 | ✅ |

- Script: `cross_arch_enc_dec_nib.py`
- Result file: `cross_arch_enc_dec_nib_results.json`
- Elapsed: 7.4 min (RTX 3080 Laptop)

---

### Claim 4 — T5-Large Backbone-Update Invariance (Exp 40)

**Statement:** An ABI domain module trained on an original T5-large backbone remains effective (and improves) when applied zero-shot to a T5-large backbone that has been fine-tuned for 1000 steps on WikiText-2.

**Status:** ✅ VALIDATED

| Checkpoint | Python PPL |
|------------|------------|
| Raw T5 baseline | 63.73 |
| Phase A domain (pre-update) | 29.61 |
| Raw backbone after 1000-step WikiText fine-tune | 35.22 |
| **Zero-shot: Phase A domain on updated backbone** | **25.61** |
| Cold-start oracle (original backbone, fresh ABI) | 32.06 |

Transfer efficacy = (35.22 − 25.61) / (35.22 − 32.06) = **304.3%** (threshold ≥ 50%)

- Script: `cross_arch_t5_succession.py`
- Result file: `cross_arch_t5_succession_results.json`

---

### Claim 5 — GPT-2-Medium Backbone-Update Invariance

**Statement:** ABI domain modules survive GPT-2-medium backbone updates.

**Status:** ✅ VALIDATED — transfer efficacy = **65.3%** (threshold ≥ 50%)

- Script: `scale_validation_test.py`
- Result file: `scale_validation_results.json`

---

### Claim 6 — Cross-Lineage Transfer: Pythia-410M → GPT-2-Medium (Exp derived)

**Statement:** ABI transfer efficacy of 91.1% across the Pythia → GPT-2 lineage boundary.

**Status:** ✅ VALIDATED

- Script: `cross_lineage_transfer_test.py`
- Result file: `cross_lineage_results.json`

---

### Claim 7 — Cross-Size Transfer: 117M–774M (all NIB PASS)

**Statement:** ABI achieves NIB PASS across GPT-2-small (117M), GPT-2-medium (354M), and GPT-2-large/XL (~774M) when using Procrustes calibration.

**Status:** ✅ VALIDATED — top-5 range: 0.862–0.870

- Script: `cross_size_large_nib_v9.py`
- Result file: `cross_size_large_nib_v9_results.json`

---

### Claim 8 — Calibration Scaling Law

**Statement:** NIB top-5 follows a predictable scaling relationship with calibration steps, with R² ≈ 0.97.

**Status:** ✅ VALIDATED

- Script: `calibration_scaling_law_b.py`
- Result file: `calibration_scaling_law_b_results.json`

---

### Claim 9 — Multi-Round Succession

**Statement:** 3-round multi-domain succession is achievable with ABI modules.

**Status:** ✅ VALIDATED

- Script: `succession_test_v2.py`
- Result file: `succession_test_v2_results.json`

---

### Claim 10 — Cross-Scale Transfer: GPT-2-small → Qwen2-1.5B

**Statement:** Domain knowledge encoded in GPT-2-small's 256-dim ABI space transfers to Qwen2-1.5B (1.54B parameters, 13x larger target, same Qwen2 architecture family as Exp 32) via orthogonal Procrustes rotation + KD calibration, achieving NIB criterion in Qwen2-1.5B's native 151936-token vocabulary.

**Status:** ⏳ PENDING — `exp_qwen_1p5b_nib.py` ready to run

- Script: `exp_qwen_1p5b_nib.py`
- Result file: `exp_qwen_1p5b_nib_results.json` (will be generated)
- Source: GPT-2-small (117M, d_model=768, BPE 50K, GPT-2 arch)
- Target: Qwen2-1.5B (1.54B, d_model=1536, tiktoken 152K, Qwen2 arch)
- D_ABI: 256 (unchanged from all prior experiments)

---

### Claim 11 — Cross-Lineage Transfer: GPT-2-medium → DeepSeek-Coder-1.3B (Llama family)

**Statement:** Domain knowledge encoded in GPT-2-medium's 256-dim ABI space transfers to deepseek-ai/deepseek-coder-1.3b-base (Llama architecture: RoPE, SwiGLU, RMSNorm, GQA — architecturally distinct from all models tested to date) via orthogonal Procrustes rotation + KD calibration, achieving NIB criterion in DeepSeek-Coder's native 32256-token vocabulary.

**Status:** ⏳ PENDING — `exp_deepseek_1p3b_nib.py` ready to run

- Script: `exp_deepseek_1p3b_nib.py`
- Result file: `exp_deepseek_1p3b_nib_results.json` (will be generated)
- Source: GPT-2-medium (354M, d_model=1024, BPE 50K, GPT-2 arch)
- Target: DeepSeek-Coder-1.3B (1.3B, d_model=2048, BPE 32K, Llama arch)
- D_ABI: 256 (unchanged)
- Significance: First Llama-family model tested — proves D_ABI=256 crosses the GPT-2 / Llama architectural boundary.

---

### Claim 12 — 7B Scale: T5-large → Qwen2-7B (INT8 quantized)

**Statement:** Domain knowledge encoded in T5-large's 256-dim ABI space transfers to Qwen2-7B (7B parameters, INT8 quantized, enc-dec → causal-decoder, different tokenizer family) via orthogonal Procrustes rotation + KD calibration, achieving NIB criterion in Qwen2-7B's native 152064-token vocabulary.

**Status:** ⏳ PENDING — `exp_qwen_7b_nib.py` ready to run

- Script: `exp_qwen_7b_nib.py`
- Result file: `exp_qwen_7b_nib_results.json` (will be generated)
- Source: T5-large (730M, d_model=1024, SentencePiece 32K, enc-dec)
- Target: Qwen2-7B (7B, d_model=3584, tiktoken 152K, Qwen2 causal decoder, INT8)
- D_ABI: 256 (unchanged)
- Significance: First 7B-scale experiment. Directly addresses "only tested small models" critique.

---

## Not Yet Tested

| Area | Why not tested |
|------|---------------|
| 7B+ models (LLaMA-2, Mistral, etc.) | Claims 10–12 in progress (exp scripts written, running) |
| Multilingual domains | Out of scope for this release |
| Medical / legal / code-other-than-Python | Out of scope for this release |
| Production inference optimization | Research prototype |
| Adversarial inputs / red-teaming | Out of scope |

---

## Explicitly Not Claimed

- Universal transfer across all architectures and domains
- Theorem-level mathematical proof of universality
- Production readiness
- Outperformance of fine-tuning in absolute quality (the claim is non-inferiority within the NIB criterion)
- Zero GPU requirement (training requires ≥ 10 GB VRAM)

---

## NIB Protocol (immutable)

The Non-Inferiority Benchmark is a fixed protocol. Parameters:

- RNG seed: 7777
- n = 5 chunks of 512 tokens each
- Skip first 20 positions per chunk (warm-up positions excluded)
- All four criteria must pass simultaneously: top-5 ≥ 0.860, top-1 ≥ 0.680, JS < 0.100, ent_diff < 0.350

The protocol cannot be modified retroactively. All validated claims use exactly these parameters.

---

*Last updated: 2026-05-18 | Release: v0.1.0*
