# ABI Experiments Index

Complete map of every validated claim to its experiment script, result file, and key metric.
All experiments share the same NIB protocol (rng=7777, n=5 chunks × 512 tokens, top-5 ≥ 0.860, top-1 ≥ 0.680, JS < 0.100, ent_diff < 0.350) unless noted.

---

## Part I — Core Validated Claims (All LOCKED)

### Claim 1 — Same-Backbone NIB (T5-large, Path 2C / Experiment 45AS)

| Field | Value |
|-------|-------|
| **Script** | `cross_arch_t5_nib_v53.py` |
| **Result file** | `cross_arch_t5_nib_v53_results.json` |
| **NIB top-5** | **0.8725** ✅ PASS |
| NIB top-1 | 0.8508 ✅ |
| NIB JS | 0.01391 ✅ |
| NIB ent_diff | 0.2256 ✅ |
| Extended NIB (n=25) | mean=0.8549, 95% CI [0.8425, 0.8673] |
| ABI architecture | 6-tap decoder [19-24], per-tap LN, D_ABI=4096 |
| Calibration steps | 16,000 (4-phase corrMSE) |
| Elapsed | 237.4 min (RTX 3080 Laptop) |

**Claim:** A T5-large model with fully frozen backbone and independently randomized ABI weights achieves non-inferior top-5 token distribution vs. the anchor — using only a correction to the residual stream.

---

### Claim 2 — Cross-Family Decoder-Only Transfer (Experiment 32)

| Field | Value |
|-------|-------|
| **Script** | `cross_family_nib.py` |
| **Result file** | `cross_family_nib_results.json` |
| **NIB top-5** | **0.8701** ✅ PASS |
| NIB top-1 | 0.9057 ✅ |
| NIB JS | 0.01123 ✅ |
| NIB ent_diff | 0.2348 ✅ |
| Transfer pair | GPT-2-small (OpenAI, BPE 50K) → Qwen2.5-0.5B (Alibaba, tiktoken 152K) |
| Method | Sentence-level orthogonal Procrustes + 1200-step KD |
| D_ABI | 256 |
| Elapsed | 15.6 min |

**Claim:** Domain knowledge transfers across genuinely different model families — different organization, architecture (absolute position vs. RoPE+GQA), tokenizer vocabulary (50K vs. 152K), and training data. NIB evaluated in Qwen's native 151,936-token vocabulary.

---

### Claim 3 — Cross-Architecture Enc-Dec → Dec-Only (Experiment 39)

| Field | Value |
|-------|-------|
| **Script** | `cross_arch_enc_dec_nib.py` |
| **Result file** | `cross_arch_enc_dec_nib_results.json` |
| **NIB top-5** | **0.8699** ✅ PASS |
| NIB top-1 | 0.9252 ✅ |
| NIB JS | 0.01787 ✅ |
| NIB ent_diff | 0.2819 ✅ |
| Source | T5-large (730M, enc-dec, SentencePiece 32K, relative position, cross-attention) |
| Target | GPT-2-medium (354M, dec-only, BPE 50K, absolute position, causal MHA) |
| Method | Prefix-LM T5 mode + sentence-level Procrustes + 1200-step KD |
| D_ABI | 256 |
| Elapsed | 7.4 min |

**Claim:** Encoder-decoder ↔ decoder-only frozen-module migration is validated. The orthogonal Procrustes map on sentence-level mean-pooled d=256 representations bridges the full architectural divide.

---

### Claim 4 — Backbone-Update Invariance for T5 (Enc-Dec) (Experiment 40)

| Field | Value |
|-------|-------|
| **Script** | `cross_arch_t5_succession.py` |
| **Result file** | `cross_arch_t5_succession_results.json` |
| **Transfer efficacy** | **304.3%** ✅ PASS (threshold ≥ 50%) |
| Zero-shot PPL (Python) | 25.61 |
| Cold-start oracle PPL | 32.06 |
| Raw backbone PPL (post-update) | 35.22 |
| Update | 1000-step WikiText-2 fine-tune with ABI stability constraint |
| Stability | Pre-computed h_abi reference cache + frozen proj_in |
| Elapsed | 9.9 min |

**Claim:** A domain module trained on T5-large before a backbone update continues to function — and outperforms a cold-start oracle — after 1000 steps of WikiText-2 fine-tuning. Backbone-update invariance confirmed for encoder-decoder architectures.

---

### Claim 5 — Backbone-Update Invariance, GPT-2-Medium (Scale Validation)

| Field | Value |
|-------|-------|
| **Script** | `scale_validation_test.py` |
| **Result file** | `scale_validation_results.json` |
| **Transfer efficacy** | **65.3%** ✅ PASS (threshold ≥ 50%) |
| Architecture | GPT-2-medium (354M, dec-only) |
| Update corpus | WikiText-2 (public benchmark) |
| Update steps | 1,000 |
| ABI alignment | 323× random noise floor post-update |
| Tests | S1–S5 all PASS |

**Claim:** Domain modules survive 1000-step WikiText-2 backbone updates at 354M scale with 65.3% zero-shot efficacy vs. native cold-start. Above the peer reviewer's stated "people will pay attention" threshold (50–70%).

---

### Claim 6 — Repeated Succession Transfer (3 Rounds, 2 Domains)

| Field | Value |
|-------|-------|
| **Script** | `succession_test_v2.py` |
| **Result file** | `succession_results_v2.json` |
| Architecture | GPT-2-medium (354M) |
| Domains | Python code + Markdown prose (simultaneously) |
| Total update steps | 3,000 (3 rounds × 1,000) |
| Signal at every checkpoint | +9.3% to +40.9% zero-shot gain both domains |
| ABI alignment | 13–14× random noise floor through all 3 rounds |

**Claim:** Domain module zero-shot signal persists across 3 successive backbone update rounds and two domain types. Transfer efficacy decays from 65% at 1,000 steps to 19–44% at 3,000 steps (backbone drift increases denominator); signal is positive at every checkpoint.

---

### Claim 7 — Calibration Scaling Law (Experiment 35b)

| Field | Value |
|-------|-------|
| **Script** | `calibration_scaling_law_b.py` |
| **Result file** | `calibration_scaling_law_b_results.json` |
| **R²** | **≈ 0.97** |
| Relationship | `floor_steps ∝ 1 / margin_median` |
| Model sizes tested | d_model 768 and 1280 (identical floors at identical margins) |
| Decision rule | margin > 0.002 → ≤ 800 steps; margin ≈ 0.001 → 2,000 steps; margin < 0.0003 → flag as hard domain |

**Claim:** Calibration budget is predicted by the native token-margin geometry of the domain, not model size. This provides a practical budget estimator for new domains.

---

### Claim 8 — Cross-Lineage Transfer (Pythia → GPT-2)

| Field | Value |
|-------|-------|
| **Script** | `cross_lineage_transfer_test.py` |
| **Result file** | `cross_lineage_results.json` |
| **Transfer efficacy** | **91.1%** |
| Source | Pythia-410m (EleutherAI, GPT-NeoX arch, The Pile data, 50,254-token vocab) |
| Target | GPT-2-medium (OpenAI, GPT-2 arch, WebText data, 50,257-token vocab) |
| Method | Sentence mean-pool MSE alignment + proj_out fine-tune |
| ABI alignment | cos_sim 0→0.859 (14× random noise floor) |

**Claim:** Domain knowledge transfers across different organizations, architectures (NeoX rotary vs. GPT-2 absolute), and training datasets. The fixed d=256 ABI bottleneck is architecture-family-agnostic for decoder-only models.

---

### Claim 9 — Cross-Size Transfer (117M → 774M, All Sizes Validated)

| Field | Value |
|-------|-------|
| **Script** | `cross_size_large_nib_v9.py` |
| **Result file** | `cross_size_large_nib_v9_results.json` |
| Sizes validated | GPT-2-small (117M) ✅, GPT-Neo-125M ✅, GPT-2-medium (354M) ✅, GPT-2-large (774M) ✅ |
| top-5 range | 0.862–0.870 (all ≥ 0.860 threshold) |
| Key finding | Correct d_abi ratio is 0.5× d_model for large models (d_abi=640 for 1280-dim GPT-2-large) |

**Claim:** NIB equivalence holds at all tested decoder-only sizes (117M–774M). The earlier apparent GPT-2-large barrier was an ABI capacity artefact — using the correct depth-ratio (d_abi = 0.5 × d_model) resolves it.

---

### Additional: Cross-Size Efficacy Test

| Field | Value |
|-------|-------|
| **Script** | `cross_size_transfer_test.py` |
| **Result file** | `cross_size_transfer_results.json` |
| **Transfer efficacy** | **88.2%** (117M module → 354M backbone) |
| Method | Sentence mean-pool MSE alignment + proj_out fine-tune |

---

### Additional: Multi-Domain Atlas

| Field | Value |
|-------|-------|
| **Script** | `multi_domain_atlas.py` |
| **Result file** | `multi_domain_atlas_results.json` |
| Domains | Python, WikiText, SQL |
| Key finding | Routing required — no single rotation achieves multi-domain parity simultaneously (locality ratio 25.4×) |

---

## Part II — Supporting Experiments

These experiments validate robustness, protocol integrity, and design choices.

| Script | Result file | What it validates |
|--------|-------------|-------------------|
| `abi_ablation_test.py` | `abi_ablation_results.json` | Objective ablation: corrMSE is uniquely optimal vs. KL, logit-MSE |
| `knowledge_non_interference.py` | `knowledge_non_interference_results.json` | Domain modules do not corrupt general backbone capability |
| `non_inferiority_benchmark.py` | `non_inferiority_results.json` | NIB protocol reference implementation and baseline |
| `nib_geometry_diagnostic.py` | `nib_geometry_diagnostic_results.json` | ABI representation geometry (cos_sim, alignment structure) |
| `generation_equivalence_test.py` | `generation_equivalence_results.json` | Generation-level equivalence (G1–G5): syntax, keyword density, coherence, cross-PPL, functional |
| `precision_parity.py` | `precision_parity_results.json` | fp32 vs bf16 parity — NIB results are precision-stable |
| `method_robustness_sweep.py` | `method_robustness_results.json` | Robustness across seeds and hyperparameters |
| `ranking_quality_analysis.py` | `ranking_quality_results.json` | Top-k ranking quality analysis |
| `transition_zone_multiseed.py` | `transition_zone_results.json` | Calibration step transition zone (multi-seed) |
| `abi_scaling_law.py` | `abi_scaling_results.json` | ABI capacity vs. model size |
| `analytical_calibration.py` | `analytical_calibration_results.json` | Analytical step count predictor |
| `calibration_budget_floor.py` | `calibration_budget_floor_results.json` | Minimum calibration floor per domain type |
| `procrustes_full_nib.py` | `procrustes_nib_results.json` | Procrustes alignment protocol NIB baseline |

---

## Part III — Production Package

| File | Purpose |
|------|---------|
| `run_abi.py` | Full Path 2C training + NIB evaluation (main entry point) |
| `verify_result.py` | Standalone result verifier — no GPU, < 5 s, checks all locked result values |
| `reproduce_abi.py` | End-to-end single-command reproduction of 4 core claims |
| `wikitext_cache.py` | WikiText-2 data loader used by all experiment scripts |
| `baseline_transformer_lm.py` | Baseline transformer used in comparison experiments |
| `abi/__init__.py` | Package entry point |
| `abi/models.py` | `AnchorABI`, `CandidateABI`, `DomainModule` |
| `abi/training.py` | Stage A (domain), Stage C (KD calibration), Stage D (Procrustes) |
| `abi/evaluation.py` | NIB evaluation logic |

---

## Complete Validated Claim Ladder

| # | Claim | Script | Key Metric | Status |
|---|-------|--------|------------|--------|
| 1 | Same-backbone NIB (T5-large) | `cross_arch_t5_nib_v53.py` | top-5 = 0.8725 | **LOCKED** |
| 2 | Cross-family decoder-only (GPT-2 → Qwen2.5) | `cross_family_nib.py` | top-5 = 0.8701 | **LOCKED** |
| 3 | Cross-arch enc-dec → dec-only | `cross_arch_enc_dec_nib.py` | top-5 = 0.8699 | **LOCKED** |
| 4 | Backbone-update invariance (T5, enc-dec) | `cross_arch_t5_succession.py` | efficacy = 304.3% | **LOCKED** |
| 5 | Backbone-update invariance (GPT-2, dec-only) | `scale_validation_test.py` | efficacy = 65.3% | PASS |
| 6 | 3-round succession, 2 domains | `succession_test_v2.py` | signal positive all rounds | PASS |
| 7 | Calibration scaling law | `calibration_scaling_law_b.py` | R² ≈ 0.97 | PASS |
| 8 | Cross-lineage (Pythia → GPT-2) | `cross_lineage_transfer_test.py` | efficacy = 91.1% | PASS |
| 9 | Cross-size (117M–774M, all NIB PASS) | `cross_size_large_nib_v9.py` | top-5 = 0.862–0.870 | PASS |
