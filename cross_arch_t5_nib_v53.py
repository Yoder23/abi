#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 45AS -- T5-large: Per-Tap LayerNorm + D_ABI=4096 + Extended P4
==========================================================================

SITUATION AFTER 45AP / 45AQ:
==============================
45AP SEED_C=99  official (n=5):  top5=0.8603  PASS  (lucky: 2/5 chunks below 0.860)
45AQ SEED_C=456 official (n=5):  top5=0.8556  FAIL
45AQ SEED_C=456 extended (n=25): true_mean=0.8464, 95% CI=[0.833, 0.859]  BELOW 0.860

TRUE GAP: architecture mean is ~0.846, threshold is 0.860, shortfall = 0.014
D-scaling is showing severe diminishing returns (D=1024→2048 gave +0.001 official,
essentially zero in true mean).

TWO NEW INTERVENTIONS THIS EXPERIMENT:
=======================================

INTERVENTION 1: Per-tap LayerNorm before concatenation
-------------------------------------------------------
Current:  h_tap = [h_19 || h_20 || h_21 || h_22 || h_23 || h_24]  (raw, unscaled)
Proposed: h_tap = [LN(h_19) || LN(h_20) || ... || LN(h_24)]

Motivation: T5-large decoder layers do NOT normalize their outputs to a consistent
scale. Each layer's hidden states have different mean and variance:
  h_19: earlier in the stack, lower L2 norm on average
  h_24: final layer, typically higher L2 norm
When proj_in(6144→D_ABI) processes the raw concatenation, the later (higher-norm)
tap layers dominate the linear combination, and the earlier layers' information is
effectively down-weighted. Per-tap LN equalizes the contribution of all 6 tap layers.

This is analogous to what LayerNorm inside Transformers does for residual streams:
it prevents one pathway from dominating due to scale differences. With 6 taps at
potentially different scales, this is the key missing normalization.

Expected effect: lower corrMSE floor by 5-15%, because proj_in gets a cleaner,
more uniformly-weighted input signal.

INTERVENTION 2: D_ABI = 4096
------------------------------
Same mechanism as D=1024→2048 (more null-space dimensions in proj_out), but larger.
proj_out(4096→1024) has a 3072-dim null space in the input space vs 1024-dim for D=2048.
More geometry leverage to distribute correction errors into logit-neutral directions.

INTERVENTION 3: Extended P4 (6000 steps)
-----------------------------------------
Best checkpoint consistently found at step 11800/12000 in ALL prior runs.
This strong signal: the optimizer has NOT converged at 12000 steps.
Extend P4 from 2000 to 6000 steps to allow further convergence.
Total cal steps: P1=4000 + P2=3000 + P3=3000 + P4=6000 = 16000.

EVALUATION: EXTENDED NIB (25 positions)
-----------------------------------------
The official n=5 evaluation has SE=0.015 and CI width ±0.030 — too noisy to
distinguish 0.846 from 0.860. We use the extended protocol (5 rng seeds × 5 chunks)
for the PRIMARY verdict. Official n=5 is also reported for protocol compliance.

Decision rule:
  ROBUST PASS if: official top5 ≥ 0.860 AND extended mean top5 ≥ 0.860
  MARGINAL if:    official top5 ≥ 0.860 but extended mean < 0.860 (like 45AP)
  FAIL if:        official top5 < 0.860

ARCHITECTURE:
  Taps:     [19,20,21,22,23,24]           unchanged
  D_IN:     6144 (6 × 1024)              unchanged
  D_ABI:    4096                          NEW (was 2048)
  Per-tap LN: LN(1024) applied to each tap before concatenation  NEW
  proj_in:  Linear(6144→4096, no bias)   NEW
  abi_ln:   LayerNorm(4096)
  proj_out: Linear(4096→1024, no bias)   NEW (3072-dim null space)
  domain:   Linear(4096→16384)+GELU+Linear(16384→4096)+LN(4096)  NEW
  Obj:      corrMSE                       unchanged
  LR:       P1=4000@5e-3 P2=3000@5e-4 P3=3000@5e-5 P4=6000@5e-6  P4 extended
  Pretrain: 2000 domain steps             unchanged

ABI param count:
  per_tap_lns: 6 × 1024 × 2 = 12288  (≈0)
  proj_in:   6144×4096 = 25.2M
  abi_ln:    4096×2 = 8192
  proj_out:  4096×1024 = 4.2M
  domain:    4096×16384 + 16384×4096 + 4096×2 = 67.1M + 67.1M + 8192 = 134.2M
  domain_alpha: 1
  Total: ~163.6M ABI params

SEED_C: 99 (same as 45AP for direct comparison of architecture improvement)
OUTPUT: cross_arch_t5_nib_v53_results.json
"""

import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cross_arch_t5_nib_v48 as base
import cross_arch_t5_nib_v51 as repro_base

EXPERIMENT      = "45AS"
D_ABI_NEW       = 4096
SEED_C_NEW      = 99           # same as 45AP for direct architecture comparison
TAP_LAYERS      = [19, 20, 21, 22, 23, 24]
N_TAPS          = len(TAP_LAYERS)
P4_EXTENDED     = 6000         # was 2000

OFFICIAL_RNG    = 7777
EXTENDED_RNGS   = [1111, 2222, 3333, 4444]

# Reference values
TOP5_45AP_OFFICIAL = 0.8603    # SEED_C=99, n=5 (pass, but within noise)
TOP5_45AQ_EXTENDED = 0.8464    # SEED_C=456, n=25 (true mean, fail)
CORRM_45AQ         = 0.003195


class MultiTap6SV_PerTapLN(nn.Module):
    """
    6-tap ABI with per-tap LayerNorm before concatenation.

    KEY CHANGE vs MultiTap6SV (v48):
      Before: h_tap = cat([h_19, h_20, ..., h_24], dim=-1)  raw, unscaled
      After:  h_tap = cat([LN(h_19), LN(h_20), ..., LN(h_24)], dim=-1)

    Motivation: T5-large decoder layers produce hidden states at different
    scales. Per-tap LN equalises these scales before proj_in processes them,
    giving the optimizer a cleaner input and preventing later (higher-norm)
    layers from dominating.
    """
    def __init__(self, abi_seed=99, d_abi=4096):
        super().__init__()
        self.t5           = base.T5ForConditionalGeneration.from_pretrained("t5-large")
        self.model_dim    = self.t5.config.d_model        # 1024
        self.d_abi        = d_abi

        # Per-tap LayerNorms (one per tap layer, shared weight across batch/seq)
        self.tap_lns      = nn.ModuleList([nn.LayerNorm(self.model_dim) for _ in TAP_LAYERS])

        self.proj_in      = nn.Linear(base.D_IN_MULTI, d_abi, bias=False)  # 6144→4096
        self.abi_ln       = nn.LayerNorm(d_abi)
        self.proj_out     = nn.Linear(d_abi, self.model_dim, bias=False)   # 4096→1024
        self.domain       = base.DomainModuleSV(d_abi)
        self.domain_alpha = nn.Parameter(torch.ones(1))

        torch.manual_seed(abi_seed)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def retie_weights(self):
        self.t5.tie_weights()

    def encode_core(self, enc_ids, dec_ids):
        enc_out = self.t5.encoder(input_ids=enc_ids)
        dec_out = self.t5.decoder(
            input_ids=dec_ids,
            encoder_hidden_states=enc_out.last_hidden_state,
            output_hidden_states=True,
        )
        # Per-tap normalisation before concatenation
        h_tap = torch.cat(
            [self.tap_lns[i](dec_out.hidden_states[layer_idx])
             for i, layer_idx in enumerate(TAP_LAYERS)],
            dim=-1,
        )   # [B, T, 6144]
        h_24  = dec_out.last_hidden_state   # residual connection (un-normalised)
        h_abi = self.abi_ln(self.proj_in(h_tap))
        return h_24, h_abi

    def forward(self, enc_ids, dec_ids, use_domain=True):
        h_24, h_abi = self.encode_core(enc_ids, dec_ids)
        h_out       = h_abi + self.domain_alpha * self.domain(h_abi) if use_domain else h_abi
        h_final     = self.proj_out(h_out) + h_24
        if self.t5.config.tie_word_embeddings:
            h_final = h_final * (self.model_dim ** -0.5)
        return self.t5.lm_head(h_final)

    def forward_with_correction(self, enc_ids, dec_ids, use_domain=True):
        h_24, h_abi = self.encode_core(enc_ids, dec_ids)
        h_out       = h_abi + self.domain_alpha * self.domain(h_abi) if use_domain else h_abi
        correction  = self.proj_out(h_out)
        h_final     = correction + h_24
        if self.t5.config.tie_word_embeddings:
            h_final = h_final * (self.model_dim ** -0.5)
        return self.t5.lm_head(h_final), correction


def train_native_as(anchor, py_ids):
    """Train the 45AS native model (replaces base.train_native for this architecture)."""
    print(f"  [C] 6-tap per-tap-LN D_ABI={D_ABI_NEW} native "
          f"(SEED_C={SEED_C_NEW}, layers={TAP_LAYERS}, "
          f"D_IN=6144→{D_ABI_NEW}, {base.DOMAIN_STEPS} steps)...")
    t0     = time.time()
    native = MultiTap6SV_PerTapLN(abi_seed=SEED_C_NEW, d_abi=D_ABI_NEW).to(base.DEVICE)
    native.t5 = anchor.t5   # share frozen backbone
    base.freeze_backbone(native)
    abi_params = [p for p in native.parameters() if p.requires_grad]
    n_params   = sum(p.numel() for p in abi_params)
    print(f"      ABI params: {n_params:,}")
    print(f"      [NEW] per_tap_lns: {sum(p.numel() for p in native.tap_lns.parameters()):,}")
    opt = torch.optim.AdamW(abi_params, lr=base.LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(base.DOMAIN_STEPS):
        enc_ids, dec_ids, labels = base.make_batch(py_ids, base_seed=6500, step=step)
        opt.zero_grad()
        F.cross_entropy(native(enc_ids, dec_ids).reshape(-1, base.VOCAB_SIZE),
                        labels.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(abi_params, 1.0)
        opt.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                p_val = base.ppl(native, py_ids, seed_base=82000)
            print(f"      step {step+1}/{base.DOMAIN_STEPS}  ppl={p_val:.1f}", flush=True)
    native.eval()
    for p in native.parameters():
        p.requires_grad_(False)
    ppl_val = base.ppl(native, py_ids)
    print(f"  [C] {time.time()-t0:.0f}s  ppl_c={ppl_val:.1f}")
    return native, ppl_val


def train_calibration_extended(anchor, native, py_ids):
    """
    corrMSE calibration with extended P4 (6000 steps total P4 instead of 2000).
    Total: P1=4000 + P2=3000 + P3=3000 + P4=6000 = 16000 steps.
    """
    P4_N_EXT  = P4_EXTENDED    # 6000
    CAL_EXT   = base.P1_N + base.P2_N + base.P3_N + P4_N_EXT  # 16000

    def get_lr_ext(step):
        if step < base.LR_WARMUP:
            return base.P1_LR * (step + 1) / base.LR_WARMUP
        if step < base.P1_N:
            return base.P1_LR
        if step < base.P1_N + base.P2_N:
            return base.P2_LR
        if step < base.P1_N + base.P2_N + base.P3_N:
            return base.P3_LR
        return base.P4_LR   # same P4 LR, just more steps

    def phase_name_ext(step):
        if step < base.P1_N:                               return "P1"
        if step < base.P1_N + base.P2_N:                  return "P2"
        if step < base.P1_N + base.P2_N + base.P3_N:      return "P3"
        return "P4"

    t0 = time.time()
    print(f"  [D] corrMSE Calibration 6-tap per-tap-LN D_ABI={D_ABI_NEW} ({CAL_EXT} steps)")
    print(f"      P1={base.P1_N}@{base.P1_LR:.0e}  P2={base.P2_N}@{base.P2_LR:.0e}  "
          f"P3={base.P3_N}@{base.P3_LR:.0e}  P4={P4_N_EXT}@{base.P4_LR:.0e}  [P4 EXTENDED]")
    print(f"      NEW: per-tap LN before concat | D=4096 null-space | best@step~11800→extended")

    calibrated = copy.deepcopy(native).to(base.DEVICE)
    calibrated.retie_weights()
    for p in calibrated.parameters():
        p.requires_grad_(False)
    # Unlock all ABI-specific params
    for nm, p in calibrated.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain", "tap_lns")):
            p.requires_grad_(True)

    cal_params = [p for p in calibrated.parameters() if p.requires_grad]
    n_cal = sum(p.numel() for p in cal_params)
    print(f"      Cal params: {n_cal:,}")

    opt = torch.optim.AdamW(cal_params, lr=base.P1_LR, weight_decay=base.CAL_WD)
    anchor.eval()
    calibrated.train()

    corr_hist    = []
    best_corrMSE = float("inf")
    best_step    = None
    best_state   = None
    global_mini  = 0
    phase_floors = {}
    prev_phase   = "P1"

    for step in range(CAL_EXT):
        lr = get_lr_ext(step)
        for g in opt.param_groups:
            g["lr"] = lr

        ph = phase_name_ext(step)
        if ph != prev_phase:
            floor = float(np.mean(corr_hist[-200:])) if len(corr_hist) >= 200 \
                else float(np.mean(corr_hist))
            phase_floors[prev_phase] = floor
            delta = floor - base.FLOOR_45AI
            print(f"      ── {ph} start (step {step+1})  "
                  f"{prev_phase} corrMSE_floor={floor:.6f}  vs_45AI: {delta:+.6f}  "
                  f"LR: →{lr:.0e}")
            prev_phase = ph

        step_loss = 0.0
        opt.zero_grad()
        for _ in range(base.ACCUM_STEPS):
            mini_seed = 9000 + global_mini
            global_mini += 1
            enc_ids, dec_ids, _ = base.make_batch(py_ids, base_seed=0, step=mini_seed)
            with torch.no_grad():
                _, corr_A = anchor.forward_with_correction(enc_ids, dec_ids)
            _, corr_C = calibrated.forward_with_correction(enc_ids, dec_ids)
            loss = F.mse_loss(corr_C.float(), corr_A.float())
            step_loss += loss.item()
            (loss / base.ACCUM_STEPS).backward()

        nn.utils.clip_grad_norm_(cal_params, 1.0)
        opt.step()
        corr_hist.append(step_loss / base.ACCUM_STEPS)

        if len(corr_hist) >= 50:
            avg50 = float(np.mean(corr_hist[-50:]))
            if avg50 < best_corrMSE:
                best_corrMSE = avg50
                best_step    = step + 1
                best_state   = copy.deepcopy(calibrated.state_dict())

        if (step + 1) % 500 == 0:
            avg50     = float(np.mean(corr_hist[-50:]))
            delta_aq  = avg50 - CORRM_45AQ
            flags = ""
            if avg50 < base.CORRM_NEEDED:   flags += " NIB_PASS_RATE?"
            elif avg50 < base.FLOOR_45AI:   flags += " BELOW_45AI✓✓"
            elif avg50 < base.FLOOR_45AG:   flags += " below_45AG✓"
            elif avg50 < base.FLOOR_SINGLE: flags += " >single✓"
            print(f"      [{ph}] step {step+1:6d}/{CAL_EXT}  "
                  f"lr={lr:.2e}  corrMSE={avg50:.6f}  vs_45AQ: {delta_aq:+.6f}{flags}",
                  flush=True)

    floor = float(np.mean(corr_hist[-200:])) if len(corr_hist) >= 200 \
        else float(np.mean(corr_hist))
    phase_floors[prev_phase] = floor

    if best_state is not None:
        print(f"      Restoring best checkpoint (step={best_step}, corrMSE={best_corrMSE:.6f})")
        calibrated.load_state_dict(best_state)
    calibrated.eval()
    for p in calibrated.parameters():
        p.requires_grad_(False)

    ppl_cal   = base.ppl(calibrated, py_ids)
    pred_top5 = base.TOP5_SINGLE + (base.FLOOR_SINGLE - best_corrMSE) * base.RATE_EFF
    delta_aq  = best_corrMSE - CORRM_45AQ

    print(f"  [D] {time.time()-t0:.0f}s  cal_ppl={ppl_cal:.1f}  "
          f"best_corrMSE={best_corrMSE:.6f}@{best_step}  vs_45AQ: {delta_aq:+.6f}")
    for ph_k, fl in phase_floors.items():
        print(f"       {ph_k} corrMSE_floor={fl:.6f}  vs_45AQ: {fl-CORRM_45AQ:+.6f}")

    if best_corrMSE < CORRM_45AQ - 0.000050:
        pct = (CORRM_45AQ - best_corrMSE) / CORRM_45AQ * 100
        print(f"       *** PER-TAP LN + D=4096 BROKE THE D=2048 FLOOR! {pct:.1f}% reduction ***")
    else:
        pct_diff = (best_corrMSE - CORRM_45AQ) / CORRM_45AQ * 100
        print(f"       corrMSE vs 45AQ: {pct_diff:+.1f}%")

    return calibrated, best_corrMSE, best_step, pred_top5, phase_floors


def main():
    t_global = time.time()

    # Patch base constants
    base.REGISTRY.update({
        "experiment": EXPERIMENT,
        "hypothesis": "per_tap_LN_D4096_corrMSE_extended_P4",
        "d_abi":      D_ABI_NEW,
        "n_taps":     N_TAPS,
    })
    base.D_ABI  = D_ABI_NEW
    base.SEED_C = SEED_C_NEW
    assert base.D_IN_MULTI == 6144
    assert base.TAP_LAYERS  == TAP_LAYERS

    base.banner(
        "Experiment 45AS -- T5-large: Per-Tap LN + D_ABI=4096 + Extended P4"
    )
    print(f"  Device:   {base.DEVICE}")
    print(f"  Taps:     {TAP_LAYERS}  D_IN={base.D_IN_MULTI}  D_ABI={D_ABI_NEW}")
    print(f"  SEED_C:   {SEED_C_NEW}")
    print()
    print(f"  WHY THIS EXPERIMENT:")
    print(f"    45AP (SEED_C=99, D=2048):  official=0.8603 PASS (lucky), extended=?")
    print(f"    45AQ (SEED_C=456, D=2048): official=0.8556 FAIL, extended=0.8464")
    print(f"    True mean ~0.846, need +0.014 to robustly pass 0.860")
    print(f"  TWO NEW INTERVENTIONS:")
    print(f"    1. Per-tap LN: equalise tap scales before proj_in (untested, high potential)")
    print(f"    2. D=4096: larger null space in proj_out(4096→1024)")
    print(f"    3. Extended P4: 6000 steps (best always found at step ~11800/12000)")
    print(f"  DECISION: based on extended NIB (n=25) — not just official n=5")

    print("\n  [Data] Loading Python corpus...")
    tok = base.T5TokenizerFast.from_pretrained("t5-large")
    parts, chars = [], 0
    for fp in base.ROOT.rglob("*.py"):
        try:
            s = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        parts.append(s)
        chars += len(s)
        if chars >= 2_000_000:
            break
    py_raw = "\n".join(parts)
    py_ids = tok(
        py_raw,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )["input_ids"].squeeze(0)[:base.MAX_PY]
    print(f"  [Data] py={len(py_ids):,}")

    base.banner(f"Step A: Anchor (single-tap D_ABI=512, SEED_A={base.SEED_A})")
    anchor, ppl_a = base.train_anchor(py_ids)

    base.banner(
        f"Step C: 6-Tap Per-Tap-LN D_ABI={D_ABI_NEW} Native Domain Training"
    )
    native, ppl_c = train_native_as(anchor, py_ids)
    gap = abs(ppl_a - ppl_c) / min(ppl_a, ppl_c)
    print(f"\n  ppl_A={ppl_a:.2f}  ppl_C={ppl_c:.2f}  gap={gap:.3f}")

    base.banner(
        f"Step D: corrMSE Calibration Per-Tap-LN D_ABI={D_ABI_NEW} Extended P4"
    )
    calibrated, best_corrMSE, step_best, pred_top5, phase_floors = \
        train_calibration_extended(anchor, native, py_ids)

    # Official NIB
    base.banner("NIB Evaluation (OFFICIAL): rng=7777, n=5")
    nib_official, t5_official = repro_base.nib_eval_with_seed(
        anchor, calibrated, py_ids, rng_seed=OFFICIAL_RNG, label="AS-official"
    )
    overall_official = bool(nib_official["pass"])

    # Extended NIB
    base.banner("NIB Evaluation (EXTENDED): 4 additional rng seeds")
    extended_results = {}
    # Store per-CHUNK means (not per-position) for conservative CI matching 45AQ methodology
    positions_per_chunk = base.PRED_LEN - base.SKIP   # 59
    n_chunks_per_seed   = base.REGISTRY["n_logit_chunks"]  # 5

    def t5_to_chunk_means(t5_positions):
        """Convert flat per-position t5 list → list of per-chunk means (n=5 each)."""
        return [
            float(np.mean(t5_positions[i*positions_per_chunk:(i+1)*positions_per_chunk]))
            for i in range(n_chunks_per_seed)
        ]

    chunk_means_official = t5_to_chunk_means(t5_official)   # 5 chunk means
    all_chunk_means_combined = list(chunk_means_official)    # starts with official 5

    for rng_seed in EXTENDED_RNGS:
        print(f"\n  -- rng_seed={rng_seed} --")
        nib_ext, t5_ext = repro_base.nib_eval_with_seed(
            anchor, calibrated, py_ids, rng_seed=rng_seed, label=f"AS-{rng_seed}"
        )
        extended_results[str(rng_seed)] = nib_ext
        all_chunk_means_combined.extend(t5_to_chunk_means(t5_ext))  # +5 chunk means each
        print(f"  rng={rng_seed}: top5={nib_ext['mean_top5']:.4f}  "
              f"pass={'YES ✓' if nib_ext['pass'] else 'NO ✗'}")

    # Combined statistics over 25 chunk means (matches 45AQ methodology exactly)
    n_combined      = len(all_chunk_means_combined)   # should be 25
    mean_combined   = sum(all_chunk_means_combined) / n_combined
    std_combined    = math.sqrt(
        sum((x - mean_combined)**2 for x in all_chunk_means_combined) / (n_combined - 1)
    )
    se_combined     = std_combined / math.sqrt(n_combined)
    ci_low          = mean_combined - 1.96 * se_combined
    ci_high         = mean_combined + 1.96 * se_combined

    rng_means = {str(OFFICIAL_RNG): nib_official["mean_top5"]}
    for s in EXTENDED_RNGS:
        rng_means[str(s)] = extended_results[str(s)]["mean_top5"]

    elapsed = (time.time() - t_global) / 60.0
    delta_corrMSE_vs_AQ = best_corrMSE - CORRM_45AQ
    delta_top5_vs_AP    = nib_official["mean_top5"] - TOP5_45AP_OFFICIAL
    architecture_confirmed = overall_official and mean_combined >= 0.860

    print()
    print("=" * 72)
    print("  45AS FINAL VERDICT")
    print("=" * 72)
    print()
    print(f"  OFFICIAL NIB (rng=7777, n=5):")
    print(f"    top5={nib_official['mean_top5']:.4f}  "
          f"JS={nib_official['mean_js']:.5f}  "
          f"top1={nib_official['mean_top1']:.4f}  "
          f"ent={nib_official['mean_ent']:.4f}  "
          f"pass={'YES ✓' if overall_official else 'NO ✗'}")
    print()
    print(f"  EXTENDED NIB (rng∈{{1111,2222,3333,4444}}, n=5 each):")
    for s in EXTENDED_RNGS:
        r = extended_results[str(s)]
        print(f"    rng={s}: top5={r['mean_top5']:.4f}  "
              f"pass={'YES ✓' if r['pass'] else 'NO ✗'}")
    print()
    print(f"  COMBINED (25 positions):  mean={mean_combined:.4f}  "
          f"std={std_combined:.4f}  SE={se_combined:.4f}")
    print(f"  95% CI: [{ci_low:.4f},  {ci_high:.4f}]  "
          f"vs 0.860: {ci_low-0.860:+.4f} (lower bound)")
    print()
    print(f"  ARCHITECTURE CONFIRMATION:")
    print(f"    Official pass:        {'YES ✓' if overall_official else 'NO ✗'}")
    print(f"    Extended mean ≥ 0.860: {'YES ✓' if mean_combined >= 0.860 else 'NO ✗'}  "
          f"(mean={mean_combined:.4f})")
    print(f"    ROBUST PASS:          {'YES ✓✓' if architecture_confirmed else 'NOT YET'}")
    print()
    print(f"  corrMSE={best_corrMSE:.6f}  vs 45AQ: {delta_corrMSE_vs_AQ:+.6f}")
    print(f"  vs 45AP official: {delta_top5_vs_AP:+.4f}")

    results = {
        "experiment":                 EXPERIMENT,
        "model":                      "t5-large",
        "protocol":                   "6tap_perTapLN_D4096_corrMSE_extP4",
        "tap_layers":                 TAP_LAYERS,
        "n_taps":                     N_TAPS,
        "d_abi":                      D_ABI_NEW,
        "d_in":                       base.D_IN_MULTI,
        "per_tap_ln":                 True,
        "p4_steps":                   P4_EXTENDED,
        "total_cal_steps":            base.P1_N + base.P2_N + base.P3_N + P4_EXTENDED,
        "seed_a":                     base.SEED_A,
        "seed_c":                     SEED_C_NEW,
        "best_corrMSE":               best_corrMSE,
        "best_step":                  step_best,
        "delta_corrMSE_vs_45AQ":      round(delta_corrMSE_vs_AQ, 6),
        "pred_top5_rate":             round(pred_top5, 4),
        "phase_corrMSE_floors":       phase_floors,
        "nib_official": {
            "rng_seed":               OFFICIAL_RNG,
            "n_chunks":               base.REGISTRY["n_logit_chunks"],
            **nib_official,
        },
        "nib_extended":               extended_results,
        "nib_combined": {
            "n_positions":            n_combined,
            "mean_top5":              round(mean_combined, 4),
            "std_top5":               round(std_combined, 4),
            "se_top5":                round(se_combined, 4),
            "ci_95_low":              round(ci_low, 4),
            "ci_95_high":             round(ci_high, 4),
            "ci_margin_vs_0860":      round(ci_low - 0.860, 4),
            "per_rng_mean_top5":      rng_means,
        },
        "architecture_confirmed":     architecture_confirmed,
        "delta_top5_official_vs_45AP": round(delta_top5_vs_AP, 4),
        "elapsed_min":                round(elapsed, 1),
        "reference": {
            "45AP_D2048_seed99_official": {
                "top5_official": TOP5_45AP_OFFICIAL,
                "corrMSE": 0.003203, "per_tap_ln": False,
            },
            "45AQ_D2048_seed456_extended": {
                "top5_extended_mean": TOP5_45AQ_EXTENDED,
                "ci_95": [0.833, 0.859],
                "corrMSE": CORRM_45AQ, "per_tap_ln": False,
            },
        },
        "registry": base.REGISTRY,
    }
    out_path = base.ROOT / "cross_arch_t5_nib_v53_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path.name}")
    print(f"  Elapsed: {elapsed:.1f} min")

    if architecture_confirmed:
        print()
        print("  " + "=" * 68)
        print("  *** ROBUST NIB PASS CONFIRMED ***")
        print(f"  Official top5={nib_official['mean_top5']:.4f} ≥ 0.860 AND")
        print(f"  Extended mean={mean_combined:.4f} ≥ 0.860 (n=25 positions)")
        print(f"  The T5-large universal knowledge transfer claim is REPRODUCIBLE")
        print(f"  and STATISTICALLY ROBUST.")
        print("  " + "=" * 68)
    elif overall_official:
        print()
        print(f"  Official pass but extended mean={mean_combined:.4f} < 0.860")
        print(f"  Same marginal situation as 45AP. Need further architecture improvement.")
    else:
        gap_ext = 0.860 - mean_combined
        print()
        print(f"  Extended gap remaining: {gap_ext:.4f}")
        if gap_ext < 0.010:
            print(f"  Very close. Next: 45AT — second seed (SEED_C=456) with same architecture")
        else:
            print(f"  Gap too large for minor tweaks. Need fundamentally different approach.")


if __name__ == "__main__":
    main()
