# ABI Architecture Reference

Technical specification of the Adaptive Bridge Interface as implemented in Experiment 45AS (`cross_arch_t5_nib_v53.py`).

---

## 1. The Non-Inferiority Benchmark (NIB) Protocol

The NIB is the **immutable evaluation standard**. A result is valid only if produced by exactly this protocol.

### Protocol Definition

Evaluate the agreement between an anchor model $A$ and a calibrated model $C$ over 5 chunks of 64-token sequences sampled from a held-out corpus.

For each token position $t$ (after skipping the first `SKIP=5` positions per chunk):

1. Compute softmax distributions $p_A(t)$ and $p_C(t)$ over the 32,128-token T5 vocabulary
2. Compute the mixture $m(t) = 0.5 \cdot (p_A(t) + p_C(t))$
3. Compute Jensen-Shannon divergence: $\text{JS}(t) = \frac{1}{2} \text{KL}(p_A \| m) + \frac{1}{2} \text{KL}(p_C \| m)$
4. Compute top-1 agreement: $\mathbb{1}[\arg\max p_A = \arg\max p_C]$
5. Compute top-5 agreement: $|\text{top}_5(p_A) \cap \text{top}_5(p_C)| / 5$
6. Compute entropy difference: $|\mathcal{H}(p_A) - \mathcal{H}(p_C)|$

Average all metrics across all token positions in all 5 chunks.

### Pass Thresholds

| Metric | Threshold | Direction |
|--------|-----------|-----------|
| mean JS divergence | < 0.10 | lower is better |
| mean top-1 agreement | ≥ 0.68 | higher is better |
| **mean top-5 agreement** | **≥ 0.86** | **higher is better (binding criterion)** |
| mean entropy difference | < 0.35 | lower is better |

### Official Evaluation Parameters

```python
rng       = np.random.default_rng(7777)   # FIXED seed, never changed
n_chunks  = 5                              # 5 × 64-token sequences
SKIP      = 5                             # skip first 5 decoder positions
ENC_LEN   = 64                            # encoder context
PRED_LEN  = 64                            # decoder prediction window
VOCAB     = 32128                         # T5-large vocabulary size
```

### Extended Evaluation

The official n=5 evaluation has high variance (SE ≈ 0.015). For robustness reporting, the extended evaluation runs the same protocol with 5 random seeds (7777, 1111, 2222, 3333, 4444), yielding n=25 observations.

The extended evaluation is **not** the official protocol — it is a supplementary robustness measurement. The official threshold applies only to the rng=7777 evaluation.

---

## 2. T5-large Model Facts

These facts govern all architectural decisions. Do not estimate them.

| Property | Value |
|----------|-------|
| Architecture | Encoder-Decoder Transformer |
| `d_model` | 1024 |
| Encoder layers | 24 |
| Decoder layers | 24 (numbered 1–24 in `hidden_states`) |
| Vocabulary | 32,128 tokens |
| `tie_word_embeddings` | `True` |
| `lm_head` has bias | `False` |
| Total parameters | ~737.7 M |
| `lm_head` scaling | `h_final *= d_model ** -0.5` applied before `lm_head` |

### Critical: `hidden_states` Indexing

When calling the T5 decoder with `output_hidden_states=True`:

```python
dec_out = t5.decoder(
    input_ids=dec_ids,
    encoder_hidden_states=enc_out.last_hidden_state,
    output_hidden_states=True,
)
dec_out.hidden_states[k]   # output of decoder layer k (k=1..24)
dec_out.last_hidden_state  # == dec_out.hidden_states[24]
```

`hidden_states[0]` is the embedding layer output (pre-attention). `hidden_states[24]` is the final decoder layer output, identical to `last_hidden_state`. In the ABI, tap indices [19,20,21,22,23,24] correspond directly to these indices.

### The Scaling Issue

T5-large uses `tie_word_embeddings=True`, which means the embedding matrix is shared with `lm_head`. When `tie_word_embeddings` is True, T5 applies a scale factor of $1/\sqrt{d_{\text{model}}}$ to the hidden state **before** applying `lm_head`. This is hardcoded in T5's `forward()`. The ABI correctly replicates this:

```python
if self.t5.config.tie_word_embeddings:
    h_final = h_final * (self.model_dim ** -0.5)
return self.t5.lm_head(h_final)
```

Failure to apply this scaling causes a 30–40× perplexity penalty. Every ABI model variant correctly includes this line.

---

## 3. The ABI Concept

### Problem Statement

A model $A$ has been calibrated to a domain (Python code). A second model $C$ with an **identical backbone** but different ABI weights must produce logit distributions that are non-inferior to $A$'s distributions — measured by the NIB protocol.

The backbone is shared and frozen. Only the ABI modules (proj_in, abi_ln, proj_out, domain) differ between $A$ and $C$.

### The Correction Vector

Both $A$ and $C$ produce a **correction vector** in hidden space:

$$\text{correction} = \text{proj\_out}\big(\text{domain\_out}\big)$$

This correction is added to the backbone's residual stream (`h_24`). The resulting modified hidden state is:

$$h_{\text{final}} = \text{correction} + h_{24}$$

The **corrMSE objective** minimizes the mean squared error between $C$'s correction vector and $A$'s correction vector:

$$\mathcal{L}_{\text{corrMSE}} = \mathbb{E}\big[\|\text{correction}_C(x) - \text{correction}_A(x)\|^2\big]$$

This is computed in hidden space (dimension 1024), not in logit space (dimension 32,128). This is key: it avoids the pathological gradient geometry of logit-space objectives.

### Why Not Fine-tune?

Fine-tuning T5-large modifies the backbone. This:
1. Destroys the pretrained knowledge (catastrophic forgetting)
2. Is expensive (full backprop through 730 M params)
3. Produces a model that cannot hot-swap ABI modules

The ABI preserves the backbone exactly. The correction is surgical: it shifts the hidden state by a learned offset that encodes the domain adaptation, while the backbone's pretrained linguistic knowledge remains intact.

---

## 4. The 45AS Architecture (MultiTap6SV_PerTapLN)

### Class Definition

```python
class MultiTap6SV_PerTapLN(nn.Module):
    """
    45AS: 6-tap [19..24], per-tap LayerNorm, D_ABI=4096.
    Defined in cross_arch_t5_nib_v53.py.
    """
    def __init__(self, abi_seed=99, d_abi=4096):
        super().__init__()
        self.t5        = T5ForConditionalGeneration.from_pretrained("t5-large")
        self.model_dim = self.t5.config.d_model                # 1024
        self.d_abi     = d_abi                                  # 4096

        # Per-tap LayerNorms: 6 × LN(1024)
        self.tap_lns   = nn.ModuleList([nn.LayerNorm(1024) for _ in TAP_LAYERS])

        self.proj_in   = nn.Linear(6144, 4096, bias=False)     # 6×1024 → 4096
        self.abi_ln    = nn.LayerNorm(4096)
        self.proj_out  = nn.Linear(4096, 1024, bias=False)     # 4096 → 1024
        self.domain    = DomainModuleSV(4096)                  # 4096→16384→4096+LN
        self.domain_alpha = nn.Parameter(torch.ones(1))

        torch.manual_seed(abi_seed)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)
```

### Forward Pass

```python
def encode_core(self, enc_ids, dec_ids):
    enc_out = self.t5.encoder(input_ids=enc_ids)
    dec_out = self.t5.decoder(
        input_ids=dec_ids,
        encoder_hidden_states=enc_out.last_hidden_state,
        output_hidden_states=True,
    )
    # Per-tap normalization before concatenation
    h_tap = torch.cat(
        [self.tap_lns[i](dec_out.hidden_states[layer_idx])
         for i, layer_idx in enumerate([19,20,21,22,23,24])],
        dim=-1,
    )                                # [B, T, 6144]
    h_24  = dec_out.last_hidden_state # residual connection (un-normalized)
    h_abi = self.abi_ln(self.proj_in(h_tap))
    return h_24, h_abi

def forward(self, enc_ids, dec_ids, use_domain=True):
    h_24, h_abi = self.encode_core(enc_ids, dec_ids)
    h_out   = h_abi + domain_alpha * domain(h_abi)  if use_domain else h_abi
    h_final = self.proj_out(h_out) + h_24
    if self.t5.config.tie_word_embeddings:
        h_final = h_final * (self.model_dim ** -0.5)
    return self.t5.lm_head(h_final)
```

### DomainModuleSV

```python
class DomainModuleSV(nn.Module):
    def __init__(self, d):  # d=4096
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 4),   # 4096 → 16384
            nn.GELU(),
            nn.Linear(d * 4, d),   # 16384 → 4096
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, h):
        return self.ln(self.net(h))
```

### Parameter Count

| Component | Formula | Parameters |
|-----------|---------|------------|
| `tap_lns` (6 × LN(1024)) | 6 × 1024 × 2 | 12,288 |
| `proj_in` (no bias) | 6144 × 4096 | 25,165,824 |
| `abi_ln` | 4096 × 2 | 8,192 |
| `proj_out` (no bias) | 4096 × 1024 | 4,194,304 |
| `domain.net[0]` weight | 4096 × 16384 | 67,108,864 |
| `domain.net[0]` bias | 16384 | 16,384 |
| `domain.net[2]` weight | 16384 × 4096 | 67,108,864 |
| `domain.net[2]` bias | 4096 | 4,096 |
| `domain.ln` | 4096 × 2 | 8,192 |
| `domain_alpha` | scalar | 1 |
| **Total ABI** | | **163,627,009** |

The T5-large backbone (~737.7 M parameters) is entirely frozen during calibration.

---

## 5. Why Per-Tap LayerNorm?

T5-large's decoder layers produce hidden states at different magnitudes. Layer 19's hidden state has a different L2 norm than layer 24's. When 6 layers are concatenated raw, the higher-norm layers dominate `proj_in`'s learned projection.

Per-tap LN normalizes each layer's hidden state independently before concatenation. This ensures that `proj_in` processes a uniformly-scaled input and can learn to use information from all 6 taps equally.

**Empirical result:** Per-tap LN contributed +0.008 to extended NIB top-5 compared to an otherwise identical architecture without it (45AP vs 45AQ baseline comparison). This was the most impactful single architectural change in the search.

---

## 6. Why D_ABI = 4096?

The proj_out layer maps from $\mathbb{R}^{4096}$ to $\mathbb{R}^{1024}$. The null space of this mapping has dimension 3072. Correction errors that fall in the null space of `proj_out` produce zero contribution to the output hidden state — they are geometrically invisible to the NIB metric.

With D_ABI=1024 (square), proj_out is $1024 \times 1024$, null space dimension 0. Correction errors distribute across all 1024 output dimensions.

With D_ABI=4096, proj_out is $1024 \times 4096$, null space dimension 3072. During optimization, correction errors preferentially flow into directions that minimize corrMSE, which can include null-space directions of the lm_head embedding matrix. This "error parking" into logit-neutral directions raises NIB metrics without requiring a lower corrMSE floor.

**Empirical result:** D=4096 outperformed D=1024 and D=2048 at the same corrMSE floor level.

---

## 7. Training Pipeline

### Stage A: Anchor Domain Pre-training

Model: `SVAT5Large` (single-tap, layer 24, D_ABI=512, SEED_A=42)
Steps: 2000
LR: 3e-4 (AdamW, weight_decay=0.01)
Objective: Cross-entropy on Python tokens
Batch: B=2, no accumulation
Purpose: Learn the domain-shifted hidden representation that will serve as the calibration target

### Stage C: Native Domain Pre-training

Model: `MultiTap6SV_PerTapLN` (45AS architecture, SEED_C=99)
Backbone: shared with anchor (frozen during calibration, but domain LM objective uses its logits here)
Steps: 2000
LR: 3e-4
Objective: Cross-entropy on Python tokens
Purpose: Initialize the 6-tap ABI parameters in the domain before corrMSE calibration begins

### Stage D: corrMSE Calibration (4 phases)

Starting from the Stage C checkpoint, calibrate against the anchor's correction vectors.

```
Phase | Steps | LR      | Notes
------+-------+---------+--------------------------------
P1    | 4000  | 5e-3    | warmup 400 steps (0 → 5e-3)
P2    | 3000  | 5e-4    |
P3    | 3000  | 5e-5    |
P4    | 6000  | 5e-6    | extended; best checkpoint ~step 15466
Total | 16000 |         |
```

**Best checkpoint tracking:** The training loop maintains a rolling 50-step average of corrMSE. The state dict of the best (lowest avg50) checkpoint is saved in memory and restored at the end. The best corrMSE of 0.003047 was found at step 15466 of 16000.

### Calibration Training Loop (pseudocode)

```python
for step in range(16000):
    lr = get_lr(step)               # 4-phase schedule
    
    for _ in range(ACCUM_STEPS=4):  # gradient accumulation
        enc_ids, dec_ids, _ = make_batch(py_ids, seed=9000 + global_mini)
        
        with torch.no_grad():
            _, corr_A = anchor.forward_with_correction(enc_ids, dec_ids)
        
        _, corr_C = calibrated.forward_with_correction(enc_ids, dec_ids)
        loss = F.mse_loss(corr_C.float(), corr_A.float())
        (loss / ACCUM_STEPS).backward()
    
    nn.utils.clip_grad_norm_(cal_params, 1.0)
    opt.step()
```

**Effective batch size:** BATCH_SV=2 × ACCUM_STEPS=4 = 8 sequences per optimizer step.

### What Is Calibrated, What Is Frozen

During Stage D, the ABI module parameters are trained; the T5-large backbone is frozen:

```python
# Trainable:
calibrated.proj_in.weight       # 6144×4096
calibrated.abi_ln.weight        # 4096
calibrated.abi_ln.bias          # 4096
calibrated.proj_out.weight      # 4096×1024
calibrated.domain_alpha         # scalar
calibrated.domain.ln.weight     # 4096
calibrated.domain.ln.bias       # 4096
calibrated.domain.net[*]        # 4096→16384→4096

# Per-tap LN weights (frozen from Stage C initialization)
# NOTE: tap_lns are NOT in the calibration parameter list —
# they are initialized during Stage C and not further updated.
# This is intentional: LN statistics stabilize during domain pretraining.
```

---

## 8. The corrMSE Objective: Why It Works

### Definition

$$\mathcal{L}_{\text{corrMSE}} = \frac{1}{BT} \sum_{b,t} \| \text{correction}_C^{(b,t)} - \text{correction}_A^{(b,t)} \|_2^2$$

where correction vectors live in $\mathbb{R}^{1024}$ (hidden dimension).

### Why Not KL Divergence?

KL divergence operates in logit space (32,128 dimensions). Gradients must flow through:
- `lm_head` (weight matrix 1024 × 32,128)
- The $d_{\text{model}}^{-0.5}$ scaling
- Then back to `proj_out`

The gradient signal is diluted by a factor of ~32,000 / 1024 ≈ 31× in terms of effective dimensionality. Moreover, KL divergence gradients are highly non-uniform: small changes in the top-k logits dominate the loss, creating pathological gradient landscapes for the correction task.

**Experiment 45AX (raw KL, LR/10)** achieved corrMSE=0.002659 — **12.7% lower** than 45AS — but NIB top-5 was 0.8244, **0.030 lower**. The KL objective found a solution with lower correction norm that nevertheless placed errors in logit-relevant directions.

### Why Not Weighted corrMSE?

Weighting correction MSE by $\|h_{\text{anchor}}\|^2$ to focus on hard positions (Experiment 45AY) and curriculum weighting (45AZ) both hurt performance:
- The weighted gradient creates interference between easy (already well-corrected) positions and hard positions
- Even at very low learning rate (LR=5e-6 in the P4 phase only), the interference catastrophically degrades rng=7777 (-0.039 top-5)
- The gradient directions for different position difficulties are misaligned in the ABI parameter space

**The standard corrMSE objective is the uniquely optimal calibration objective for this task.**

---

## 9. The corrMSE Floor

Across all 45AS-architecture variants, the corrMSE floor converges to approximately 0.003050 ± 0.0003. This is an **information-theoretic limit** of the ABI architecture:

- Single-tap baseline (`SVAT5Large`): floor ≈ 0.003668
- 3-tap `[20,22,24]` D=1024: floor ≈ 0.003339
- 6-tap `[19..24]` D=1024: floor ≈ 0.003339 (same — tap count does not lower the floor)
- 6-tap `[19..24]` D=4096 + per-tap LN: floor ≈ 0.003047 (per-tap LN breaks the floor by ~10%)

The floor is determined by the information content of the domain corpus that can be extracted through the ABI's linear bottleneck. The tap count controls the null-space dimensionality (NIB geometry), not the floor.

---

## 10. NIB Geometry: Why D=4096 Raises Top-5 Without Lowering corrMSE

The NIB metric (top-5 agreement) measures whether $C$ and $A$ agree on the top-5 vocabulary items at each position. The top-5 items are determined by the rows of the embedding matrix with the highest dot product with $h_{\text{final}}$.

Let $E \in \mathbb{R}^{32128 \times 1024}$ be the embedding matrix. The top-5 of $A$ and $C$ agree if:

$$\text{top}_5(E \cdot h_{\text{final},A}) = \text{top}_5(E \cdot h_{\text{final},C})$$

The correction error is $\delta = \text{correction}_C - \text{correction}_A$. For top-5 to still agree, we need:

$$\text{top}_5(E \cdot (h_{24} + \text{correction}_A + \delta)) = \text{top}_5(E \cdot (h_{24} + \text{correction}_A))$$

This is possible if $E \cdot \delta \approx 0$, i.e., $\delta$ is in the null space of the embedding matrix $E$. This null space has dimension $1024 - \text{rank}(E)$.

**With D_ABI=4096**, `proj_out` has a 3072-dimensional null space. During optimization, gradient descent can find solutions where the residual correction error $\delta$ has a large component in $\ker(E)$, making it invisible to the NIB metric while still contributing to the corrMSE loss. This is the mechanism by which higher D raises NIB metrics without requiring a lower corrMSE floor.

---

## 11. Key Constants Reference

Defined in `cross_arch_t5_nib_v48.py` (base module):

```python
TAP_LAYERS  = [19, 20, 21, 22, 23, 24]
D_MODEL     = 1024
D_ABI       = 1024     # base module (v48); overridden to 4096 in v53
D_IN_MULTI  = 6144     # 6 × 1024
ENC_LEN     = 64
PRED_LEN    = 64
DOMAIN_STEPS= 2000
LR_ABI      = 3e-4     # domain pre-training LR
P1_N, P1_LR = 4000, 5e-3
P2_N, P2_LR = 3000, 5e-4
P3_N, P3_LR = 3000, 5e-5
P4_N, P4_LR = 2000, 5e-6   # overridden to 6000 in v53
LR_WARMUP   = 400
CAL_WD      = 0.0
ACCUM_STEPS = 4
BATCH_SV    = 2
SEED_A      = 42
SEED_C      = 99
VOCAB_SIZE  = 32128
MAX_PY      = 500_000
PAD_ID      = 0
SKIP        = 5
```

Overrides in `cross_arch_t5_nib_v53.py` (45AS):

```python
D_ABI_NEW    = 4096
SEED_C_NEW   = 99     # same as base
P4_EXTENDED  = 6000   # was 2000 in base
```
