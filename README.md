# ABI: Frozen-Module Domain Transfer Across LLM Architectures
## Verified Experimental Results — Research Preview Release

> **Frozen ABI domain modules migrate across tested LLM architecture boundaries with NIB-verified behavioral equivalence, while keeping backbones entirely frozen and calibrating only interface projections.** Validated across decoder-only and encoder-decoder architectures, 4 model families, 117M–774M parameter scale, with domain modules surviving backbone updates at 65–304% transfer efficacy.

---

## What This Does Not Claim

Before anything else:

| Not claimed | Why |
|-------------|-----|
| Universal transfer across all possible models | Only tested: T5-large, GPT-2 family, Qwen2.5-0.5B, Pythia-410m |
| 7B+ scale | Not yet evaluated |
| All domains | Core results are Python-focused; WikiText used for backbone-update tests |
| Zero calibration | Cross-architecture transfer requires KD/projection calibration steps |
| Production-ready deployment | Research prototype — no inference optimisation, no serving infrastructure |
| Theorem-level mathematical proof | Formal verification artifact for the specific Path 2C claim; broader results are empirically validated |

This is a **research preview**: the goal is that strangers can verify the result in 5 seconds, reproduce it in a few hours, and understand exactly what is and is not claimed.

---

## The Core Claim

**A model can transfer domain knowledge to a second, independently initialized model by learning only a correction to the shared backbone's residual stream — with the backbone entirely frozen in both models.**

This is Autonomous Basis Injection (ABI). The two models share a frozen T5-large backbone but have completely different ABI module weights (different random seeds). After calibration, the candidate's logit distributions are non-inferior to the anchor's. No teacher forward pass is needed at inference time. No logit-level distillation is required. No backbone weights are modified at any stage.

The formal criterion is the Non-Inferiority Benchmark (NIB):

| Criterion | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| **Top-5 token agreement** | **0.8725** | >= 0.860 | **PASS** |
| Top-1 token agreement | 0.8508 | >= 0.680 | PASS |
| Jensen-Shannon divergence | 0.01391 | < 0.100 | PASS |
| Entropy difference | 0.2256 | < 0.350 | PASS |

All four criteria pass simultaneously. The result holds at extended evaluation (n=25, 5 independent RNG seeds): mean top-5 = 0.8549, 95% CI = [0.8425, 0.8673].

### Cross-Architecture Transfer — Also Validated

Exp 39 (result file: `cross_arch_enc_dec_nib_results.json`) extends the core result across the encoder-decoder ↔ decoder-only architectural boundary:

| Source | Target | Method | Top-5 | Top-1 | JS | Ent | Status |
|--------|--------|--------|-------|-------|----|-----|--------|
| T5-large (enc-dec, 32K vocab) | GPT-2-medium (dec-only, 50K vocab) | Procrustes + KD | **0.8699** | 0.9252 | 0.01787 | 0.2819 | **PASS** |

The source model (T5-large, 730M, SentencePiece, relative position encoding, cross-attention) and target model (GPT-2-medium, 354M, BPE, absolute position encoding, causal MHA) differ in architecture class, tokenizer family, vocabulary size, and position encoding scheme. The domain module trained on T5-large is transferred to GPT-2-medium via orthogonal Procrustes rotation on sentence-level mean-pooled ABI representations and 1200-step KD calibration. NIB is evaluated entirely in GPT-2's native 50257-token vocabulary. The domain module weights are not modified during calibration — only the projection matrices `proj_in` and `proj_out` are trained. Elapsed: 7.4 min (RTX 3080 Laptop).

**Encoder-decoder ↔ decoder-only frozen-module migration: VALIDATED.**

### Backbone-Update Invariance — Also Validated

Exp 40 (result file: `cross_arch_t5_succession_results.json`) confirms that domain modules survive backbone fine-tuning for encoder-decoder architectures:

| Checkpoint | Python PPL |
|------------|------------|
| Raw T5 baseline | 63.73 |
| Phase A domain (pre-update) | 29.61 |
| Raw backbone after 1000-step WikiText fine-tune | 35.22 |
| **Zero-shot: Phase A domain on updated backbone** | **25.61** |
| Cold-start oracle (original backbone, fresh ABI) | 32.06 |

Transfer efficacy = (35.22 − 25.61) / (35.22 − 32.06) = **304.3%** (threshold ≥ 50%). The zero-shot PPL is lower than the cold-start oracle because the WikiText update improved the backbone's general representations, which the pre-trained domain module can exploit even more effectively. Backbone-update invariance is now validated for both decoder-only (GPT-2-medium, 65% efficacy) and encoder-decoder (T5-large, 304% efficacy) architectures.

---

## Instant Verification (< 5 seconds, no GPU required)

```powershell
python verify_result.py
```

Expected output: all checks green, final line `Path 2C -- T5-large ABI domain reconstruction -- VERIFIED`.

The result file `cross_arch_t5_nib_v53_results.json` is the immutable published record. `verify_result.py` checks every value against embedded expected constants.

---

## Full Reproduction (no pre-computed results needed)

```powershell
# One-time: download t5-large (~2.9 GB)
python -c "from transformers import T5ForConditionalGeneration, T5TokenizerFast; T5ForConditionalGeneration.from_pretrained('t5-large'); T5TokenizerFast.from_pretrained('t5-large'); print('ready')"

# Full training + evaluation (~4 hours on RTX 3080 Laptop)
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
python run_abi.py
```

Results are written to `abi_result.json`.

---

## How It Works

T5-large's decoder has 24 transformer layers. ABI taps the last 6 layers (19-24) simultaneously, applies per-layer normalization, and learns a correction vector in the backbone's residual stream:

```
For each tap k in [19, 20, 21, 22, 23, 24]:
    h_k' = LayerNorm_k(hidden_states[k])       per-tap normalisation

h_tap    = concat([h_19', ..., h_24'])          [B, T, 6144]
h_abi    = LayerNorm(Linear_6144->4096(h_tap))
h_out    = h_abi + alpha * Domain(h_abi)        learnable domain gate
correction = Linear_4096->1024(h_out)

h_final  = correction + h_24                   residual injection
h_final *= d_model^-0.5                        T5 tie_word_embeddings scaling
logits   = lm_head(h_final)                    frozen backbone head
```

The backbone -- all 730M parameters of T5-large -- is never updated. Only the ABI module (163.6M parameters) is trained.

### Why D_ABI = 4096?

`proj_out` maps 4096 -> 1024. This map has a 3072-dimensional null space. After convergence, residual calibration error lives in directions orthogonal to the top-5 token embedding rows in the lm_head matrix -- producing zero logit change for those tokens. This null-space geometry is why corrMSE = 0.003047 produces top-5 agreement of 0.8725 rather than the ~0.848 a naive model would predict.

### Why corrMSE?

Seven objectives were tested. corrMSE (MSE of the hidden-space correction vector) is the only one that works:

- **Raw KL**: 12.7% lower loss than corrMSE -- yet NIB top-5 is **0.030 lower**. Logit-level supervision causes gradient interference with top-5 geometry.
- **Weighted corrMSE**: any per-position weighting destroys easy-seed performance. Uniform is optimal.
- **logit-MSE**: catastrophic (top-5 drops to ~0.6).

---

## Repository Structure

```
abi/                                   <- production package
    __init__.py
    models.py                          <- AnchorABI, CandidateABI, DomainModule
    training.py                        <- Stages A, C, D; data pipeline
    evaluation.py                      <- NIB evaluation (official + extended)

run_abi.py                             <- entry point: full training + evaluation
verify_result.py                       <- standalone result verifier (no GPU)
reproduce_abi.py                       <- one-command reproduction of 4 core claims
wikitext_cache.py                      <- WikiText-2 data loader
baseline_transformer_lm.py             <- baseline transformer

# Core validated claims (9 experiments — each has a .py script + _results.json)
cross_arch_t5_nib_v53.py              <- Claim 1: Path 2C (top-5=0.8725)
cross_family_nib.py                    <- Claim 2: GPT-2 → Qwen2.5 (top-5=0.8701)
cross_arch_enc_dec_nib.py              <- Claim 3: T5-large → GPT-2-medium (top-5=0.8699)
cross_arch_t5_succession.py           <- Claim 4: T5 backbone-update (efficacy=304%)
scale_validation_test.py               <- Claim 5: GPT-2-medium backbone-update (65%)
succession_test_v2.py                  <- Claim 6: 3-round multi-domain succession
calibration_scaling_law_b.py           <- Claim 7: calibration scaling law (R²=0.97)
cross_lineage_transfer_test.py         <- Claim 8: Pythia → GPT-2 (91.1%)
cross_size_large_nib_v9.py             <- Claim 9: cross-size 117M–774M (all PASS)

experiments/                           <- supporting validation experiments (13 scripts)
    abi_ablation_test.py               <- objective ablation
    knowledge_non_interference.py      <- general capability preservation
    procrustes_full_nib.py             <- Procrustes alignment baseline
    generation_equivalence_test.py     <- generation-level equivalence
    precision_parity.py                <- fp32/bf16 parity
    ... (+ 8 more, each with _results.json)

README.md                              <- this file
PROOF.md                               <- verified experimental proof artifact (Path 2C)
ABI_ARCHITECTURE.md                    <- full technical specification
ABI_EXPERIMENTS.md                     <- complete experimental ledger
ABI_REPRODUCE.md                       <- step-by-step replication guide
ABI_START_HERE.md                      <- developer onboarding (10 steps)
EXPERIMENTS_INDEX.md                   <- map of every claim to script/result/metric
```

---

## Documentation Map

| Document | Audience | Contents |
|----------|----------|----------|
| [PROOF.md](PROOF.md) | Researchers, potential partners | Formal claim, proof structure, what was ruled out |
| [ABI_ARCHITECTURE.md](ABI_ARCHITECTURE.md) | Engineers | NIB math, model code, training theory, corrMSE floor |
| [ABI_EXPERIMENTS.md](ABI_EXPERIMENTS.md) | Engineers | All 8 experiments with results and conclusions |
| [ABI_REPRODUCE.md](ABI_REPRODUCE.md) | Anyone | Exact commands and expected output at every stage |
| [ABI_START_HERE.md](ABI_START_HERE.md) | New developers | 10-step onboarding from zero |

---

## Requirements

```
torch>=2.1  (with CUDA)
transformers>=4.38
sentencepiece>=0.1.99
numpy
tqdm
```

GPU with >= 10 GB VRAM required for training. `verify_result.py` runs on CPU only.

---

## What This Means

The standard approach to knowledge transfer in AI is fine-tuning: update a pretrained model's weights on new data. ABI demonstrates a different regime: **the backbone's weights need never change**. Two models can agree on domain-specific predictions while maintaining completely independent learned representations -- connected only through a correction to the shared residual stream.

This has direct implications for:

- **Multi-tenant deployment**: one frozen backbone, many independent ABI modules, each domain-specialized, hot-swappable at zero cost.
- **Continual learning**: domain knowledge can be injected and revoked without touching the core model.
- **Federated settings**: independent agents can align their domain representations without sharing raw data or model weights.
- **Auditable AI**: because the backbone is frozen, any behavioral change is fully attributable to the ABI module. The boundary of responsibility is exact.

The corrMSE calibration objective requires no labeled data, no teacher network at inference time, and no vocabulary-level supervision. The data pipeline is entirely self-contained.
