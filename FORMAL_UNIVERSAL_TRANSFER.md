# Formal ABI Transfer Theorem

This document is the formal claim boundary for "universal domain transfer."
It separates three statements that are easy to conflate:

1. An assumption-free theorem over all possible models.
2. A conditional theorem over ABI-compatible source/target pairs.
3. An empirical certificate that a concrete pair belongs to that ABI-compatible
   regime.

The first statement is impossible. The second is the theorem ABI can support.
The third is what the experiment suite is designed to certify for real model
pairs.

---

## 1. Definitions

Let a frozen language model with an ABI interface be a tuple

```
M = (B, E, D, P, H)
```

where:

- `B` is the frozen backbone.
- `E_M(x) in R^d` is the ABI encoder/interface representation for input
  position `x`.
- `D_M: R^d -> R^d` is the domain operator.
- `P_M: R^d -> R^{h_M}` is the ABI projection back to the model residual
  stream.
- `H_M: R^{h_M} -> R^{|V_M|}` is the frozen output head.

For a target model `T`, define the target native domain oracle logits as

```
L_T^*(x) = H_T(h_T(x) + P_T(z_T(x) + D_T(z_T(x))))
```

where `z_T(x) = E_T(x)`.

A transferred ABI module uses a source domain operator `D_S`, an alignment map
`R: R^d -> R^d`, and target-side bridge/interface parameters `A_T`:

```
L_{S->T}(x) = H_T(h_T(x) + A_T(z_T(x), R D_S R^{-1}))
```

The copied source domain core `D_S` is considered frozen if its internal MLP
weights are unchanged during target calibration. Target-side projections,
normalization, scalar gates, rank losses, and coordinate bridges are interface
calibration, not retraining of the copied domain core.

---

## 2. Impossibility of Assumption-Free Universality

**Theorem 1 (No assumption-free universal transfer).**
No algorithm can guarantee non-inferior domain transfer from an arbitrary source
model to an arbitrary target model for every possible model pair, domain, and
token distribution using only a portable module and finite calibration.

**Proof.**
Assume such an algorithm `A` exists.

Construct a target model `T_bad` whose frozen output head ignores every ABI
correction:

```
H_bad(h + P(z)) = c
```

for a constant logit vector `c`, independent of `z`, `P`, and the transferred
domain module. Let the source domain oracle have two inputs `x_1, x_2` whose
top-5 token sets differ from the constant top-5 set of `c`.

For every possible adapter, bridge, rotation, calibration objective, and domain
module, `T_bad` returns the same logits. Therefore the transferred model cannot
match the source/domain oracle top-5 sets on both inputs. NIB non-inferiority is
impossible.

This contradicts the assumed universal guarantee. Therefore no theorem over all
possible models can exist without assumptions on target expressivity,
observable ABI coordinates, and output-head sensitivity.

The same conclusion follows from finite calibration: for any finite calibration
set, one can construct two target oracles that agree on the calibration set and
disagree on held-out inputs. No finite calibration transcript can distinguish
them, so no algorithm can guarantee held-out transfer for both.

QED.

**Consequence.** The correct formal target is not "all possible models." The
correct target is "all models satisfying an ABI compatibility certificate."

---

## 3. ABI Compatibility Assumptions

A source/target pair `(S, T)` is ABI-compatible on domain distribution `mu` if
there exist constants

```
epsilon_align, epsilon_domain, epsilon_cal, L_head, gamma, rho
```

such that the following hold on at least a `1 - rho` fraction of positions
sampled from `mu`.

### A1. Coordinate alignment

There exists an approximately orthogonal map `R` such that

```
|| z_T(x) - R z_S(x) ||_2 <= epsilon_align
```

for aligned source/target sentence positions.

### A2. Domain equivariance

The target native domain operator and rotated source domain operator agree up
to bounded error:

```
|| D_T(z_T(x)) - R D_S(z_S(x)) ||_2 <= epsilon_domain
```

This is the mathematical statement that the source domain operator represents
portable domain knowledge rather than architecture-specific noise.

### A3. Interface calibration

After calibrating only target-side ABI interface parameters, the residual stream
correction error is bounded:

```
|| Delta_T^*(x) - Delta_{S->T}(x) ||_2 <= epsilon_cal
```

where `Delta_T^*` is the target native oracle correction and `Delta_{S->T}` is
the transferred correction.

### A4. Output-head Lipschitzness

The frozen target output head is locally Lipschitz:

```
|| H_T(u) - H_T(v) ||_infty <= L_head || u - v ||_2
```

for corrections reached by the ABI interface.

### A5. Top-k margin

Let `K = 5`. For the target native oracle logits `L_T^*(x)`, define

```
margin_5(x) =
  min_{i in top5(L_T^*(x)), j not in top5(L_T^*(x))}
  L_T^*(x)_i - L_T^*(x)_j
```

The certified positions satisfy

```
margin_5(x) > 2 delta
```

where

```
delta = L_head * (epsilon_align + epsilon_domain + epsilon_cal)
```

Positions failing this margin condition have total mass at most `rho`.

---

## 4. Universal Conditional Transfer Theorem

**Theorem 2 (Universal ABI transfer over the compatible class).**
For every source/target pair `(S, T)` satisfying assumptions A1-A5 on domain
distribution `mu`, the transferred ABI module preserves the target native
oracle top-5 set on all certified positions. Therefore the expected top-5 set
agreement is at least

```
1 - rho
```

and NIB top-5 passes whenever

```
1 - rho >= 0.860.
```

If the JS and entropy certificate bounds also satisfy the NIB thresholds, then
the transferred ABI module is non-inferior to the target native oracle under
the full NIB protocol.

**Proof.**
By A1-A3, the transferred residual correction differs from the target native
oracle correction by at most

```
epsilon_align + epsilon_domain + epsilon_cal
```

in the ABI-reachable residual stream. By A4, every target-vocabulary logit
changes by at most

```
delta = L_head * (epsilon_align + epsilon_domain + epsilon_cal).
```

Take any certified position `x`. Let `i` be a token in the native oracle top-5
set and `j` be any token outside that set. By A5,

```
L_T^*(x)_i - L_T^*(x)_j > 2 delta.
```

After transfer, the worst case is that token `i` decreases by `delta` and token
`j` increases by `delta`. Therefore

```
L_{S->T}(x)_i - L_{S->T}(x)_j
  >= L_T^*(x)_i - L_T^*(x)_j - 2 delta
  > 0.
```

Thus every native top-5 token remains above every non-top-5 token in the
transferred logits. The unordered top-5 set is identical.

This holds on all certified positions. Since uncertified or low-margin
positions have mass at most `rho`, expected top-5 set agreement is at least
`1 - rho`. If `1 - rho >= 0.860`, the NIB top-5 criterion passes. The remaining
NIB criteria follow from the separately certified JS and entropy bounds.

QED.

---

## 5. What a GPT-5 -> GPT-6 Claim Would Need

The theorem does not require the source and target to share architecture,
tokenizer, parameter count, or vocabulary. It requires a certificate that the
target exposes an ABI-compatible coordinate system for the source domain
operator.

For a future source/target pair such as "GPT-5 domain module -> GPT-6 target,"
the required artifact is not a retraining run. It is a transfer certificate:

```
{
  "source_model": "...",
  "target_model": "...",
  "domain": "...",
  "source_domain_core_sha256": "...",
  "target_backbone_sha256": "...",
  "domain_core_frozen": true,
  "trainable_target_side_components": [
    "proj_in",
    "abi_ln",
    "proj_out",
    "domain_ln",
    "domain_alpha",
    "optional_domain_bridge"
  ],
  "alignment_certificate": {
    "method": "sentence_mean_pool_procrustes",
    "n_pairs": 2000,
    "post_alignment_cosine": "..."
  },
  "calibration_certificate": {
    "steps": "...",
    "losses": ["KD", "top-k KD", "rank margin", "top-set"],
    "max_or_empirical_residual_error": "..."
  },
  "nib_certificate": {
    "mean_js": "...",
    "mean_top1_agree": "...",
    "mean_top5_overlap": "...",
    "mean_entropy_diff": "...",
    "pass": true
  }
}
```

If that certificate passes, then the practical claim is:

> The source domain core was copied into the target ABI interface without
> full target retraining, and the calibrated target-side interface is
> non-inferior to the target native domain oracle under NIB.

That is the precise version of "move domain knowledge from one model generation
to the next."

---

## 6. Current Empirical Support

The current experiments do not prove all future model pairs. They show that the
ABI compatibility assumptions are not vacuous: they have been satisfied by many
real cached model pairs spanning GPT-2, GPT-Neo, Qwen2/Qwen2.5,
DeepSeek/Llama, Pythia/GPT-NeoX, Phi-3, and T5-style artifacts.

The important recent stress case was Pythia-410M -> DeepSeek-Coder-1.3B. It
initially failed top-5 despite passing JS/top1/entropy. The repair that passed
on two seeds was:

- frozen copied source domain MLP core
- `D_ABI=512`
- target-side linear domain bridge
- top-k KD with `k=128`
- teacher-rank margin loss
- student hard-negative suppression
- full-vocabulary teacher top-set loss
- 7200 target calibration steps

The two result files are:

- `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_seed42_results.json`
- `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal7200_topset5_bridge_seed314_results.json`

The current highest top-5 hard-direction certificate extends the same recipe to
16000 calibration steps while keeping the copied source domain core frozen, and
the shifted-seed repeat remains above the NIB threshold:

- `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed42_results.json`
- `exp_generic_causal_nib_v2_pythia410_deepseek_d512_cal16000_topset5_bridge_seed314_results.json`

The same strong frozen-core recipe also certifies a bidirectional GPT-Neo/Qwen2.5
pair:

- `exp_generic_causal_nib_v2_gptneo125m_qwen2p5_0p5b_d512_cal4800_topset5_bridge_seed42_results.json`
- `exp_generic_causal_nib_v2_qwen2p5_0p5b_gptneo125m_d512_cal4800_topset5_bridge_seed42_results.json`

It also certifies a bidirectional GPT-Neo-125M/Phi-3-mini-4k-instruct pair
with the same frozen-core, bridge, rank, and top-set controls at `D_ABI=1024`:

- `exp_generic_causal_nib_v2_gptneo125m_phi3_d1024_cal4800_topset5_bridge_seed42_results.json`
- `exp_generic_causal_nib_v2_phi3_gptneo125m_d1024_cal4800_topset5_bridge_seed42_results.json`

These are not a mathematical proof of every future model. They are evidence
that the compatibility class covered by Theorem 2 includes non-trivial,
cross-family, directed model pairs.

---

## 7. Claim Boundary

What is formally proved:

- Assumption-free universal transfer over all possible models is impossible.
- Universal transfer is guaranteed over the ABI-compatible class A1-A5.
- A concrete model-pair claim is valid only when it comes with a certificate
  showing that the pair satisfies the theorem's conditions or empirically
  passes the NIB certificate.

What is not yet proved:

- That every future frontier model pair automatically satisfies A1-A5.
- That a particular unreleased target model exposes an ABI-compatible coordinate
  system.
- That Python-domain transfer implies medical, legal, multilingual, or all
  other domains.

This is the formal path to the larger win: prove the theorem once, then use
each certificate to transfer without full target retraining.
