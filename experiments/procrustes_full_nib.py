#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procrustes Full NIB  — Domain-Restricted Parity via Analytical Calibration
=============================================================
Replaces Step D (800-step KD SGD) with a single closed-form Procrustes
solve in ABI space, then runs the complete 5-level NIB evaluation.

Hypothesis: if the Procrustes solve passes all 5 levels, it proves that
  (a) domain-restricted behavioral parity — not just distributional parity — is achievable
      without a training loop, and
  (b) the 800-step Step D was solving a problem that has a closed-form
      solution — implying this domain's parity is largely geometric, not a
      learning problem.

Protocol: A → B → C → Procrustes(D) → L2 + L3 + L4a + L4b + L4c

R² of the linear fit is measured separately; if R²≈1.0 all five levels pass.
If top-5 fails while JS/top-1 pass, the residual non-linearity is confirmed
to live in the high-entropy ranking positions (as shown by ranking_quality_analysis).

Results: procrustes_nib_results.json
"""

import copy
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import shared infrastructure from NIB (backbone frozen, same probes) ─────
# Safe to import: NIB has `if __name__ == "__main__":` guard.
from non_inferiority_benchmark import (
    PROBE_BANK, ADV_VARIANTS, REGISTRY,
    SVGPT2, DomainModuleSV,
    DEVICE, D_ABI, SEQ_LEN, DOMAIN_STEPS, UPDATE_STEPS,
    LR_ABI, LR_BACKBONE, LR_CAL, ALPHA,
    MAX_PY_SV, MAX_WIKI_SV, BATCH_SV, SEED, ROOT,
    make_batch_sv, ppl_sv, generate_nib, evaluate_probe,
    bootstrap_ci, noninferior, jaccard,
    l2_logit_test, l3_decoding_test, l4a_l4b_functional_test, l4c_adversarial_test,
)
from transformers import GPT2TokenizerFast
from wikitext_cache import load_wikitext_split

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

N_COLLECT_BATCHES = 200   # 200 × 8 × 128 = 204,800 ABI vector pairs

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = 10**30

    py_parts, py_chars = [], 0
    for p in ROOT.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            py_parts.append(txt); py_chars += len(txt)
            if py_chars >= MAX_PY_SV * 4: break
        except Exception:
            continue
    py_ids = tok("\n".join(py_parts), return_tensors="pt",
                 truncation=False)["input_ids"].squeeze(0)[:MAX_PY_SV]

    wiki_ds  = load_wikitext_split("wikitext-2-raw-v1", "train")
    wiki_raw = "\n".join(r["text"] for r in wiki_ds if r["text"].strip())
    wiki_ids = tok(wiki_raw, return_tensors="pt",
                   truncation=False)["input_ids"].squeeze(0)[:MAX_WIKI_SV]

    print(f"  py_ids={len(py_ids):,}  wiki_ids={len(wiki_ids):,}")
    return tok, py_ids, wiki_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEPS A → B → C  (identical to NIB run_training_protocol, no Step D yet)
# ══════════════════════════════════════════════════════════════════════════════

def run_abc(py_ids, wiki_ids):
    """Run Steps A, B, C and return (transferred_state, native, ppl_nat)."""

    # ── A: anchor on Python (ABI only, backbone frozen) ─────────────────────
    print("  [A] anchor (500 steps Python)...")
    t0 = time.time()
    anchor = SVGPT2().to(DEVICE)
    for p in anchor.parameters(): p.requires_grad_(False)
    for nm, p in anchor.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_a = torch.optim.AdamW([p for p in anchor.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    anchor.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch_sv(py_ids, seed=5000 + step)
        opt_a.zero_grad()
        F.cross_entropy(anchor(x).reshape(-1, 50257), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(anchor.parameters(), 1.0)
        opt_a.step()
    anchor.eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    saved_dom = copy.deepcopy(anchor.domain.state_dict())
    print(f"  [A] {time.time()-t0:.0f}s  ppl={ppl_sv(anchor, py_ids):.2f}")

    # ── B: backbone drift on WikiText ────────────────────────────────────────
    print("  [B] backbone drift (1000 steps WikiText)...")
    t1 = time.time()
    transferred = copy.deepcopy(anchor).to(DEVICE)
    for p in transferred.parameters(): p.requires_grad_(False)
    for nm, p in transferred.named_parameters():
        if "backbone" in nm or "proj_in" in nm or "abi_ln" in nm:
            p.requires_grad_(True)
    opt_b = torch.optim.AdamW([p for p in transferred.parameters() if p.requires_grad],
                               lr=LR_BACKBONE, weight_decay=0.01)
    transferred.train()
    for step in range(UPDATE_STEPS):
        x, y = make_batch_sv(wiki_ids, seed=9000 + step)
        opt_b.zero_grad()
        h, h_abi = transferred.encode_core(x)
        logits = transferred.lm_head(transferred.proj_out(h_abi) + h)
        ll = F.cross_entropy(logits.reshape(-1, 50257), y.reshape(-1))
        with torch.no_grad():
            _, h_aa = anchor.encode_core(x)
        sl = F.mse_loss(h_abi, h_aa)
        (ll + ALPHA * sl).backward()
        nn.utils.clip_grad_norm_(transferred.parameters(), 1.0)
        opt_b.step()
    transferred.domain.load_state_dict(saved_dom)
    transferred.eval()
    for p in transferred.parameters(): p.requires_grad_(False)
    transferred_state = copy.deepcopy(transferred.state_dict())
    print(f"  [B] {time.time()-t1:.0f}s")

    # ── C: native oracle (fresh ABI, PyTorch corpus) ─────────────────────────
    print("  [C] native oracle (500 steps Python)...")
    t2 = time.time()
    native = copy.deepcopy(transferred).to(DEVICE)
    nn.init.xavier_uniform_(native.proj_in.weight)
    nn.init.xavier_uniform_(native.proj_out.weight)
    nn.init.ones_(native.abi_ln.weight); nn.init.zeros_(native.abi_ln.bias)
    native.domain = DomainModuleSV(D_ABI).to(DEVICE)
    native.domain_alpha.data.fill_(1.0)
    for p in native.parameters(): p.requires_grad_(False)
    for nm, p in native.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_c = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch_sv(py_ids, seed=5000 + step)
        opt_c.zero_grad()
        F.cross_entropy(native(x).reshape(-1, 50257), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt_c.step()
    native.eval()
    for p in native.parameters(): p.requires_grad_(False)
    ppl_nat = ppl_sv(native, py_ids)
    print(f"  [C] {time.time()-t2:.0f}s  ppl_nat={ppl_nat:.2f}")

    return transferred_state, native, ppl_nat


# ══════════════════════════════════════════════════════════════════════════════
# PROCRUSTES STEP D  (0 SGD steps — closed-form coordinate rotation)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def procrustes_step_d(transferred_state, native, py_ids):
    """
    Collect N=204,800 ABI vector pairs (h_full_cal, h_full_nat).
    Solve A* = lstsq(H_cal, H_nat).
    Bake: proj_out_new.weight = proj_out_nat.weight @ A*.T
    Returns (calibrated_model, r_squared, cond_number).
    """
    t0 = time.time()
    print(f"  [Procrustes] Collecting {N_COLLECT_BATCHES}×{BATCH_SV}×{SEQ_LEN}"
          f"={N_COLLECT_BATCHES*BATCH_SV*SEQ_LEN:,} ABI pairs...")

    # Build transferred model for collection
    transferred = SVGPT2().to(DEVICE)
    transferred.load_state_dict(transferred_state)
    transferred.eval()

    H_cal_list, H_nat_list = [], []
    for i in range(N_COLLECT_BATCHES):
        x, _ = make_batch_sv(py_ids, seed=3000 + i)

        _, h_abi_cal = transferred.encode_core(x)   # [B, T, D_ABI]
        _, h_abi_nat = native.encode_core(x)          # [B, T, D_ABI]

        h_full_cal = (h_abi_cal
                      + transferred.domain_alpha * transferred.domain(h_abi_cal))
        h_full_nat = (h_abi_nat
                      + native.domain_alpha      * native.domain(h_abi_nat))

        H_cal_list.append(h_full_cal.reshape(-1, D_ABI).cpu().float())
        H_nat_list.append(h_full_nat.reshape(-1, D_ABI).cpu().float())

    H_cal = torch.cat(H_cal_list, dim=0)   # [N, D_ABI]
    H_nat = torch.cat(H_nat_list, dim=0)   # [N, D_ABI]

    # Least-squares solve: H_cal @ A* ≈ H_nat
    # Match analytical_calibration.py: truncated SVD is more stable for the
    # moderately ill-conditioned ABI design matrix than the default driver.
    A_star = torch.linalg.lstsq(
        H_cal, H_nat, rcond=1e-4, driver="gelsd").solution  # [D_ABI, D_ABI]

    # R² of linear fit
    H_nat_pred = H_cal @ A_star
    ss_res = float(((H_nat - H_nat_pred) ** 2).sum())
    ss_tot = float(((H_nat - H_nat.mean(0)) ** 2).sum())
    r_squared = round(1.0 - ss_res / ss_tot, 5) if ss_tot > 0 else 0.0

    # Condition number of H_cal
    sv = torch.linalg.svdvals(H_cal)
    cond = float(sv.max() / sv.min().clamp(min=1e-12))

    # Build calibrated model: bake proj_out_new.weight = proj_out_nat.weight @ A*.T
    calibrated = SVGPT2().to(DEVICE)
    calibrated.load_state_dict(transferred_state)

    new_proj_out_w = (native.proj_out.weight.cpu().float() @ A_star.T)
    calibrated.proj_out.weight.data.copy_(
        new_proj_out_w.to(DEVICE).to(calibrated.proj_out.weight.dtype))

    # Copy output-side calibration params from native
    calibrated.domain_alpha.data.copy_(native.domain_alpha.data)
    calibrated.domain.ln.weight.data.copy_(native.domain.ln.weight.data)
    calibrated.domain.ln.bias.data.copy_(native.domain.ln.bias.data)

    for p in calibrated.parameters(): p.requires_grad_(False)
    calibrated.eval()

    ppl_cal = ppl_sv(calibrated, py_ids)
    efficacy = (ppl_cal / ppl_sv(native, py_ids)) * 100
    print(f"  [Procrustes] R²={r_squared:.5f}  cond={cond:.1f}"
          f"  ppl_cal={ppl_cal:.2f}  efficacy={efficacy:.1f}%"
          f"  ({time.time()-t0:.0f}s)")

    return calibrated, r_squared, cond, ppl_cal, efficacy


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def banner(msg):
    w = 72
    print()
    print("=" * w)
    print(f"  {msg}")
    print("=" * w)


def main():
    t_global = time.time()
    banner("Procrustes Full NIB  —  Domain-Restricted Parity via Analytical Calibration")
    print(f"  device: {DEVICE}")
    print(f"  D_ABI: {D_ABI}   collect_batches: {N_COLLECT_BATCHES}")
    print(f"  Hypothesis: 0-SGD Procrustes passes all 5 NIB levels")

    tok, py_ids, wiki_ids = load_data()

    banner("Training Protocol: A → B → C → Procrustes(D)")
    transferred_state, native, ppl_nat = run_abc(py_ids, wiki_ids)

    banner("Procrustes Step D  (closed-form, 0 SGD steps)")
    calibrated, r_squared, cond_number, ppl_cal, efficacy = procrustes_step_d(
        transferred_state, native, py_ids)

    # ── L2: Distributional Equivalence ───────────────────────────────────────
    banner("L2 — Distributional Equivalence")
    t_l2 = time.time()
    print("  Running 5 × 512-token forward passes...")
    l2_res = l2_logit_test(native, calibrated, py_ids, REGISTRY)
    print(f"\n  mean_JS     = {l2_res['mean_js']:.5f}  "
          f"(thr < {REGISTRY['js_threshold']})  {'PASS' if l2_res['js_pass'] else 'FAIL'}")
    print(f"  top-1 agree = {l2_res['mean_top1_agree']:.4f}  "
          f"(thr >= {REGISTRY['top1_threshold']})  {'PASS' if l2_res['top1_pass'] else 'FAIL'}")
    print(f"  top-5 overlap = {l2_res['mean_top5_overlap']:.4f}  "
          f"(thr >= {REGISTRY['top5_threshold']})  {'PASS' if l2_res['top5_pass'] else 'FAIL'}")
    print(f"  entropy diff = {l2_res['mean_entropy_diff']:.4f}  "
          f"(thr < {REGISTRY['entropy_diff_threshold']})  "
          f"{'PASS' if l2_res['entropy_pass'] else 'FAIL'}")
    print(f"\n  [L2] {'PASS' if l2_res['pass'] else 'FAIL'}  ({time.time()-t_l2:.0f}s)")

    # ── L4a / L4b: Functional Non-Inferiority + Error Identity ───────────────
    banner("L4a/L4b — Functional Non-Inferiority + Error Identity (60 probes × 3 seeds)")
    t_l4 = time.time()
    l4ab_res = l4a_l4b_functional_test(native, calibrated, tok, PROBE_BANK, REGISTRY)
    print(f"\n  native pass:       {l4ab_res['nat_pass_pp']:.1f}%  "
          f"CI=[{l4ab_res['nat_ci_95'][0]:.1f}%, {l4ab_res['nat_ci_95'][1]:.1f}%]")
    print(f"  calibrated pass:   {l4ab_res['cal_pass_pp']:.1f}%  "
          f"CI=[{l4ab_res['cal_ci_95'][0]:.1f}%, {l4ab_res['cal_ci_95'][1]:.1f}%]")
    print(f"  NI threshold:      {l4ab_res['ni_threshold_pp']:.1f}%  "
          f"NI={'PASS' if l4ab_res['ni_pass'] else 'FAIL'}")
    print(f"  Failure Jaccard:   {l4ab_res['failure_jaccard']:.3f}  "
          f"{'PASS' if l4ab_res['failure_jaccard_pass'] else 'FAIL'}")
    print(f"  Pass Jaccard:      {l4ab_res['pass_jaccard']:.3f}  "
          f"{'PASS' if l4ab_res['pass_jaccard_pass'] else 'FAIL'}")
    for d, dr in l4ab_res["by_difficulty"].items():
        print(f"  Diff-{d} ({dr['n_probes']} probes):  "
              f"nat={dr['nat_pass_pp']:.1f}%  cal={dr['cal_pass_pp']:.1f}%")
    print(f"\n  [L4a/L4b] {'PASS' if l4ab_res['pass'] else 'FAIL'}  ({time.time()-t_l4:.0f}s)")

    # ── L3: Decoding Equivalence ──────────────────────────────────────────────
    banner("L3 — Decoding Equivalence (greedy / low-temp / high-temp)")
    t_l3 = time.time()
    l3_res = l3_decoding_test(native, calibrated, tok, PROBE_BANK, REGISTRY)
    print(f"\n  [L3] {'PASS' if l3_res['pass'] else 'FAIL'}  ({time.time()-t_l3:.0f}s)")

    # ── L4c: Adversarial ─────────────────────────────────────────────────────
    banner("L4c — Adversarial Prompt Perturbations (30 variants)")
    t_l4c = time.time()
    l4c_res = l4c_adversarial_test(native, calibrated, tok, ADV_VARIANTS, PROBE_BANK, REGISTRY)
    print(f"\n  native adv:    {l4c_res['nat_pass_pp']:.1f}%")
    print(f"  calibrated adv:{l4c_res['cal_pass_pp']:.1f}%  "
          f"CI_lo={l4c_res['cal_ci_95_lower_pp']:.1f}%")
    print(f"  [L4c] {'PASS' if l4c_res['pass'] else 'FAIL'}  ({time.time()-t_l4c:.0f}s)")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    tests = {
        "L2_distributional":  l2_res["pass"],
        "L4a_functional_NI":  l4ab_res["ni_pass"] or l4ab_res["floor_skip"],
        "L4b_error_identity": l4ab_res["failure_jaccard_pass"],
        "L3_decoding":        l3_res["pass"],
        "L4c_adversarial":    l4c_res["pass"],
    }
    n_pass = sum(tests.values())

    verdicts = {
        5: "DOMAIN-RESTRICTED PARITY CONFIRMED - PROCRUSTES PASSES ALL 5 LEVELS",
        4: "STRONG PARITY EVIDENCE (4/5 Procrustes)",
        3: "PARTIAL PARITY (3/5 Procrustes)",
    }
    verdict = verdicts.get(n_pass, f"PARTIAL ({n_pass}/5 Procrustes)")

    banner(f"VERDICT: {verdict}")
    for name, passed in tests.items():
        sym = "✓" if passed else "✗"
        print(f"  {sym} {name}")

    print(f"\n  n_pass    = {n_pass}/5")
    print(f"  R²        = {r_squared:.5f}  (1.0 = perfectly linear correction)")
    print(f"  cond(H)   = {cond_number:.1f}")
    print(f"  ppl_cal   = {ppl_cal:.2f}  ppl_nat = {ppl_nat:.2f}  efficacy = {efficacy:.1f}%")
    print(f"  Total runtime = {(time.time()-t_global)/60:.1f} min")

    if n_pass == 5:
        print("\n  *** STEP D IS A COORDINATE ROTATION FOR THIS DOMAIN ***")
        print(f"  *** Procrustes (0 SGD steps) achieves full 5-level parity ***")

    output = {
        "verdict":      verdict,
        "n_pass":       n_pass,
        "n_total":      5,
        "r_squared":    r_squared,
        "cond_number":  round(cond_number, 1),
        "ppl_cal":      round(ppl_cal, 3),
        "ppl_nat":      round(ppl_nat, 3),
        "efficacy_pct": round(efficacy, 2),
        "L2":           l2_res,
        "L3":           l3_res,
        "L4ab":         l4ab_res,
        "L4c":          l4c_res,
        "test_summary": tests,
        "registry":     REGISTRY,
    }
    out_path = ROOT / "procrustes_nib_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
