#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Non-Interference Test
=================================
Central claim: ABI knowledge transfer is ADDITIVE — calibrating a model to
match a Python domain oracle does not degrade its general language capability.

A strong separability claim requires not just that domain knowledge transfers IN,
but that general knowledge is NOT displaced.  This experiment measures both.

Protocol:
  Same A → B → C → D as NIB.
  Evaluate FOUR models on TWO corpora:

    Models:
      (a) base_gpt2m      — GPT-2-medium straight from HuggingFace (no training)
      (b) transferred     — post-Step-B  (backbone drifted to WikiText)
      (c) native          — post-Step-C  (Python oracle, ABI trained on Python)
      (d) calibrated      — post-Step-D  (Procrustes analytical calibration)

    Corpora:
      (i)  Python code    — domain corpus (same py_ids as NIB)
      (ii) WikiText-2     — non-domain corpus (same wiki_ids as NIB)

  Metrics per (model, corpus) pair:
    - PPL  (language quality)
    - Mean JS divergence vs native on domain-matched corpus (calibrated vs native only)
    - Top-5 overlap vs native

  Key predictions:
    1. calibrated ≈ native on Python (domain parity) ← already proven
    2. calibrated ≈ transferred on WikiText (no degradation)
       i.e., Step D does NOT hurt non-domain performance
    3. wiki_ppl(calibrated) < wiki_ppl(native)
       because native re-trained ABI on Python disrupts backbone for non-domain use
       while calibrated retains the backbone optimised for WikiText in Step B

  If prediction 2 holds: knowledge transfer is non-destructive.
  If prediction 3 holds: Procrustes calibration is more knowledge-conservative
    than standard SGD Step D.

  Double-calibration test (new):
    Takes calibrated (Python-specific) and runs a SECOND Step D targeting
    WikiText (domain_B). Asks: does calibrating for Wiki destroy Python parity?
    If Python NIB L2 still passes after double-calibration → knowledge transfer
    is truly modular / composable.

Results: knowledge_non_interference_results.json
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

from non_inferiority_benchmark import (
    SVGPT2, DomainModuleSV,
    DEVICE, D_ABI, SEQ_LEN, DOMAIN_STEPS, UPDATE_STEPS,
    LR_ABI, LR_BACKBONE, LR_CAL, ALPHA,
    MAX_PY_SV, MAX_WIKI_SV, BATCH_SV, SEED, ROOT,
    REGISTRY,
    make_batch_sv, ppl_sv, l2_logit_test,
)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wikitext_cache import load_wikitext_split

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

N_COLLECT        = 200   # Procrustes collection batches
DOUBLE_CAL_STEPS = 400   # second calibration budget (domain_B = WikiText)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
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
# BASE GPT-2-MEDIUM PPL (no ABI, raw HuggingFace model)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def base_gpt2_ppl(tok, ids, n_batches=50):
    """Evaluate raw GPT-2-medium (no ABI) perplexity on token sequence."""
    from transformers import GPT2LMHeadModel  # local import to avoid shadowing
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(DEVICE).eval()
    tot, n = 0.0, 0
    max_start = max(len(ids) - SEQ_LEN - 1, 1)
    rng = torch.Generator()
    for i in range(n_batches):
        rng.manual_seed(80000 + i)
        starts = torch.randint(0, max_start, (BATCH_SV,), generator=rng)
        x = torch.stack([ids[s:s+SEQ_LEN]     for s in starts]).to(DEVICE)
        y = torch.stack([ids[s+1:s+SEQ_LEN+1] for s in starts]).to(DEVICE)
        logits = model(x).logits
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                y.reshape(-1)).item()
        n += 1
    del model
    torch.cuda.empty_cache()
    return math.exp(tot / n)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING STEPS A → B → C
# ══════════════════════════════════════════════════════════════════════════════

def run_abc(py_ids, wiki_ids):
    t0 = time.time()
    print("  [A] anchor (500 steps Python, ABI only)...")
    anchor = SVGPT2().to(DEVICE)
    for p in anchor.parameters(): p.requires_grad_(False)
    for nm, p in anchor.named_parameters():
        if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain")):
            p.requires_grad_(True)
    opt_a = torch.optim.AdamW([p for p in anchor.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    anchor.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch_sv(py_ids, seed=5000+step)
        opt_a.zero_grad()
        F.cross_entropy(anchor(x).reshape(-1,50257), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(anchor.parameters(), 1.0)
        opt_a.step()
    anchor.eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    saved_dom = copy.deepcopy(anchor.domain.state_dict())
    print(f"  [A] {time.time()-t0:.0f}s")

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
        x, y = make_batch_sv(wiki_ids, seed=9000+step)
        opt_b.zero_grad()
        h, h_abi = transferred.encode_core(x)
        logits = transferred.lm_head(transferred.proj_out(h_abi)+h)
        ll = F.cross_entropy(logits.reshape(-1,50257), y.reshape(-1))
        with torch.no_grad(): _, h_aa = anchor.encode_core(x)
        (ll + ALPHA*F.mse_loss(h_abi, h_aa)).backward()
        nn.utils.clip_grad_norm_(transferred.parameters(), 1.0)
        opt_b.step()
    transferred.domain.load_state_dict(saved_dom)
    transferred.eval()
    for p in transferred.parameters(): p.requires_grad_(False)
    transferred_state = copy.deepcopy(transferred.state_dict())
    print(f"  [B] {time.time()-t1:.0f}s")

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
        if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain")):
            p.requires_grad_(True)
    opt_c = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch_sv(py_ids, seed=5000+step)
        opt_c.zero_grad()
        F.cross_entropy(native(x).reshape(-1,50257), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt_c.step()
    native.eval()
    for p in native.parameters(): p.requires_grad_(False)
    ppl_nat = ppl_sv(native, py_ids)
    print(f"  [C] {time.time()-t2:.0f}s  ppl_nat={ppl_nat:.2f}")
    return transferred_state, native, ppl_nat


# ══════════════════════════════════════════════════════════════════════════════
# STEP D: PROCRUSTES ANALYTICAL CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def procrustes_step_d(transferred_state, native, py_ids):
    transferred = SVGPT2().to(DEVICE)
    transferred.load_state_dict(transferred_state)
    transferred.eval()
    H_cal_list, H_nat_list = [], []
    max_start = max(len(py_ids)-SEQ_LEN-1, 1)
    rng = torch.Generator()
    for i in range(N_COLLECT):
        rng.manual_seed(30000+i)
        starts = torch.randint(0, max_start, (BATCH_SV,), generator=rng)
        x = torch.stack([py_ids[s:s+SEQ_LEN] for s in starts]).to(DEVICE)
        _, h_abi_c = transferred.encode_core(x)
        _, h_abi_n = native.encode_core(x)
        hf_c = h_abi_c + transferred.domain_alpha * transferred.domain(h_abi_c)
        hf_n = h_abi_n + native.domain_alpha * native.domain(h_abi_n)
        H_cal_list.append(hf_c.reshape(-1, D_ABI).cpu().float())
        H_nat_list.append(hf_n.reshape(-1, D_ABI).cpu().float())
    H_cal = torch.cat(H_cal_list)
    H_nat = torch.cat(H_nat_list)
    A_star = torch.linalg.lstsq(H_cal, H_nat, rcond=None).solution
    calibrated = SVGPT2().to(DEVICE)
    calibrated.load_state_dict(transferred_state)
    new_w = (native.proj_out.weight.cpu().float() @ A_star.T)
    calibrated.proj_out.weight.data.copy_(
        new_w.to(DEVICE).to(calibrated.proj_out.weight.dtype))
    calibrated.domain_alpha.data.copy_(native.domain_alpha.data)
    calibrated.domain.ln.weight.data.copy_(native.domain.ln.weight.data)
    calibrated.domain.ln.bias.data.copy_(native.domain.ln.bias.data)
    for p in calibrated.parameters(): p.requires_grad_(False)
    calibrated.eval()
    return calibrated


# ══════════════════════════════════════════════════════════════════════════════
# DOUBLE CALIBRATION  (calibrate for WikiText after Python calibration)
# ══════════════════════════════════════════════════════════════════════════════

def double_calibration(calibrated_python, wiki_ids):
    """
    Take the Python-calibrated model and run a second KD pass targeting WikiText.
    The "teacher" is a freshly trained WikiText oracle (new native for domain_B).
    Ask: does Python parity (L2 on py_ids) survive this second calibration?
    """
    print("  [Double-Cal] Building WikiText oracle (domain_B native)...")
    # Build wiki oracle: ABI trained on WikiText
    wiki_oracle = copy.deepcopy(calibrated_python).to(DEVICE)
    nn.init.xavier_uniform_(wiki_oracle.proj_in.weight)
    nn.init.xavier_uniform_(wiki_oracle.proj_out.weight)
    nn.init.ones_(wiki_oracle.abi_ln.weight); nn.init.zeros_(wiki_oracle.abi_ln.bias)
    wiki_oracle.domain = DomainModuleSV(D_ABI).to(DEVICE)
    wiki_oracle.domain_alpha.data.fill_(1.0)
    for p in wiki_oracle.parameters(): p.requires_grad_(False)
    for nm, p in wiki_oracle.named_parameters():
        if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain")):
            p.requires_grad_(True)
    opt_w = torch.optim.AdamW([p for p in wiki_oracle.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    wiki_oracle.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch_sv(wiki_ids, seed=5500+step)
        opt_w.zero_grad()
        F.cross_entropy(wiki_oracle(x).reshape(-1,50257), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(wiki_oracle.parameters(), 1.0)
        opt_w.step()
    wiki_oracle.eval()
    for p in wiki_oracle.parameters(): p.requires_grad_(False)

    print("  [Double-Cal] Running second KD calibration (→ WikiText)...")
    double_cal = copy.deepcopy(calibrated_python).to(DEVICE)
    for p in double_cal.parameters(): p.requires_grad_(False)
    double_cal.proj_in.weight.requires_grad_(True)
    double_cal.proj_out.weight.requires_grad_(True)
    double_cal.domain_alpha.requires_grad_(True)
    double_cal.domain.ln.weight.requires_grad_(True)
    double_cal.domain.ln.bias.requires_grad_(True)
    params = [double_cal.proj_in.weight, double_cal.proj_out.weight,
              double_cal.domain_alpha,
              double_cal.domain.ln.weight, double_cal.domain.ln.bias]
    opt_d2 = torch.optim.AdamW(params, lr=LR_CAL, weight_decay=0.01)
    kd_w = REGISTRY["kd_weight"]; kd_t = REGISTRY["kd_temp"]; ce_w = 1-kd_w
    double_cal.train()
    for step in range(DOUBLE_CAL_STEPS):
        x, y = make_batch_sv(wiki_ids, seed=7500+step)
        opt_d2.zero_grad()
        cal_lo = double_cal(x)
        with torch.no_grad(): nat_lo = wiki_oracle(x)
        V = cal_lo.shape[-1]
        kd = F.kl_div(F.log_softmax(cal_lo.reshape(-1,V)/kd_t, dim=-1),
                      F.softmax(nat_lo.reshape(-1,V)/kd_t, dim=-1),
                      reduction='batchmean') * (kd_t**2)
        ce = F.cross_entropy(cal_lo.reshape(-1,V), y.reshape(-1))
        (kd_w*kd + ce_w*ce).backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt_d2.step()
    double_cal.eval()
    for p in double_cal.parameters(): p.requires_grad_(False)
    return double_cal, wiki_oracle


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-CORPUS L2 METRICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def cross_corpus_l2(model_a, model_b, ids, label="", n_chunks=5, chunk_size=512):
    """
    Measure JS/top-1/top-5 distributional similarity between model_a and model_b
    on the given token sequence.  model_a = "native" reference; model_b = calibrated.
    """
    eps = 1e-12
    rng = np.random.default_rng(7777)
    js_list, top1_list, top5_list = [], [], []
    max_start = max(len(ids)-chunk_size, 1)
    skip = 20

    for _ in range(n_chunks):
        start = int(rng.integers(0, max_start))
        chunk = ids[start:start+chunk_size].unsqueeze(0).to(DEVICE)
        lo_a  = model_a(chunk, use_domain=True)[0, skip:, :]
        lo_b  = model_b(chunk, use_domain=True)[0, skip:, :]
        p_a   = F.softmax(lo_a, dim=-1).cpu().float().numpy()
        p_b   = F.softmax(lo_b, dim=-1).cpu().float().numpy()
        T     = p_a.shape[0]
        m    = 0.5*(p_a+p_b)
        p_ac = np.clip(p_a, eps, 1.0)
        p_bc = np.clip(p_b, eps, 1.0)
        m_c  = np.clip(m, eps, 1.0)
        kl_am = (p_ac*np.log(p_ac/m_c)).sum(1)
        kl_bm = (p_bc*np.log(p_bc/m_c)).sum(1)
        js_list.extend(np.clip(0.5*(kl_am+kl_bm), 0, None).tolist())
        top1_list.extend((p_a.argmax(1) == p_b.argmax(1)).tolist())
        nat5 = np.argpartition(p_a,-5,axis=1)[:,-5:]
        cal5 = np.argpartition(p_b,-5,axis=1)[:,-5:]
        for t in range(T):
            top5_list.append(len(set(nat5[t])&set(cal5[t]))/5.)

    return {
        "label":          label,
        "mean_js":        round(float(np.mean(js_list)), 5),
        "mean_top1":      round(float(np.mean(top1_list)), 4),
        "mean_top5":      round(float(np.mean(top5_list)), 4),
        "js_pass":        float(np.mean(js_list)) < REGISTRY["js_threshold"],
        "top5_pass":      float(np.mean(top5_list)) >= REGISTRY["top5_threshold"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_global = time.time()
    print("=" * 72)
    print("  KNOWLEDGE NON-INTERFERENCE TEST")
    print("  Claim: ABI knowledge transfer is ADDITIVE, not destructive")
    print(f"  device: {DEVICE}")
    print("=" * 72)

    tok, py_ids, wiki_ids = load_data()

    # Baseline: raw GPT-2-medium
    print("\n  [Base] GPT-2-medium PPL on both corpora...")
    base_py_ppl   = base_gpt2_ppl(tok, py_ids)
    base_wiki_ppl = base_gpt2_ppl(tok, wiki_ids)
    print(f"  base GPT-2m: py_ppl={base_py_ppl:.2f}  wiki_ppl={base_wiki_ppl:.2f}")

    # Train
    print("\n  Running A→B→C→D...")
    transferred_state, native, ppl_nat = run_abc(py_ids, wiki_ids)

    # Build transferred (post-B, no domain calibration)
    transferred_eval = SVGPT2().to(DEVICE)
    transferred_eval.load_state_dict(transferred_state)
    for p in transferred_eval.parameters(): p.requires_grad_(False)
    transferred_eval.eval()

    print("  [D] Procrustes calibration...")
    t = time.time()
    calibrated = procrustes_step_d(transferred_state, native, py_ids)
    print(f"    done ({time.time()-t:.0f}s)")

    # PPL for all four models on both corpora
    print("\n  Computing PPL on Python corpus...")
    ppl_results = {
        "base_gpt2m":   {"py_ppl": round(base_py_ppl, 3),   "wiki_ppl": round(base_wiki_ppl, 3)},
        "transferred":  {"py_ppl": round(ppl_sv(transferred_eval,  py_ids),   3),
                         "wiki_ppl": round(ppl_sv(transferred_eval, wiki_ids),  3)},
        "native":       {"py_ppl": round(ppl_nat, 3),
                         "wiki_ppl": round(ppl_sv(native, wiki_ids), 3)},
        "calibrated":   {"py_ppl": round(ppl_sv(calibrated, py_ids), 3),
                         "wiki_ppl": round(ppl_sv(calibrated, wiki_ids), 3)},
    }

    print("\n  PPL Summary:")
    print(f"  {'Model':<20}  {'py_ppl':>8}  {'wiki_ppl':>10}  wiki_delta_vs_transferred")
    t_wiki = ppl_results["transferred"]["wiki_ppl"]
    for name, d in ppl_results.items():
        delta = d["wiki_ppl"] - t_wiki
        print(f"  {name:<20}  {d['py_ppl']:8.2f}  {d['wiki_ppl']:10.2f}  {delta:+.3f}")

    # Cross-corpus L2: calibrated vs native on BOTH corpora
    print("\n  Cross-corpus L2 (calibrated vs native):")
    py_l2   = cross_corpus_l2(native, calibrated, py_ids,   label="python")
    wiki_l2 = cross_corpus_l2(native, calibrated, wiki_ids, label="wikitext")
    print(f"  Python domain:    "
          f"JS={py_l2['mean_js']:.5f}  top1={py_l2['mean_top1']:.4f}  "
          f"top5={py_l2['mean_top5']:.4f}  "
          f"{'PASS' if py_l2['js_pass'] and py_l2['top5_pass'] else 'FAIL'}")
    print(f"  WikiText domain:  "
          f"JS={wiki_l2['mean_js']:.5f}  top1={wiki_l2['mean_top1']:.4f}  "
          f"top5={wiki_l2['mean_top5']:.4f}  "
          f"{'PASS' if wiki_l2['js_pass'] and wiki_l2['top5_pass'] else 'FAIL'}")

    # Double-calibration test
    print("\n  Running Double-Calibration (Python-cal → WikiText-cal)...")
    t = time.time()
    double_cal, wiki_oracle = double_calibration(calibrated, wiki_ids)
    print(f"  Double-cal done ({time.time()-t:.0f}s)")

    # After double-cal: does Python parity survive?
    py_l2_after_double = cross_corpus_l2(native, double_cal, py_ids,
                                          label="python_after_double_cal")
    wiki_l2_after_double = cross_corpus_l2(wiki_oracle, double_cal, wiki_ids,
                                            label="wiki_after_double_cal")
    py_ppl_double = ppl_sv(double_cal, py_ids)
    wiki_ppl_double = ppl_sv(double_cal, wiki_ids)
    print(f"  After double-cal:")
    print(f"    vs native (Python):    "
          f"JS={py_l2_after_double['mean_js']:.5f}  "
          f"top5={py_l2_after_double['mean_top5']:.4f}  "
          f"{'PASS' if py_l2_after_double['js_pass'] and py_l2_after_double['top5_pass'] else 'FAIL'}")
    print(f"    vs wiki_oracle (Wiki): "
          f"JS={wiki_l2_after_double['mean_js']:.5f}  "
          f"top5={wiki_l2_after_double['mean_top5']:.4f}  "
          f"{'PASS' if wiki_l2_after_double['js_pass'] and wiki_l2_after_double['top5_pass'] else 'FAIL'}")
    print(f"    py_ppl={py_ppl_double:.2f}  wiki_ppl={wiki_ppl_double:.2f}")

    # Key interference metrics
    wiki_ppl_delta    = ppl_results["calibrated"]["wiki_ppl"] - ppl_results["transferred"]["wiki_ppl"]
    py_parity_intact  = py_l2["js_pass"] and py_l2["top5_pass"]
    wiki_no_harm      = abs(wiki_ppl_delta) < 0.5   # < 0.5 PPL degradation is immaterial
    dual_parity_intact = (py_l2_after_double["js_pass"] and
                          wiki_l2_after_double["js_pass"])

    print("\n" + "=" * 72)
    print("  NON-INTERFERENCE VERDICT")
    print("=" * 72)
    print(f"  [1] Python domain parity (calibrated vs native):      "
          f"{'PASS' if py_parity_intact else 'FAIL'}"
          f"  JS={py_l2['mean_js']:.5f}  top5={py_l2['mean_top5']:.4f}")
    print(f"  [2] WikiText non-degradation (calibrated vs xferred):  "
          f"{'PASS' if wiki_no_harm else 'FAIL'}"
          f"  Δppl={wiki_ppl_delta:+.3f}")
    print(f"  [3] Generalization of rotation to WikiText:            "
          f"JS={wiki_l2['mean_js']:.5f}  top5={wiki_l2['mean_top5']:.4f}")
    print(f"  [4] Double-cal Python parity survives:                 "
          f"{'PASS' if py_l2_after_double['js_pass'] else 'FAIL'}"
          f"  JS={py_l2_after_double['mean_js']:.5f}")
    print(f"  [5] Double-cal WikiText parity achieved:               "
          f"{'PASS' if wiki_l2_after_double['js_pass'] else 'FAIL'}"
          f"  JS={wiki_l2_after_double['mean_js']:.5f}")

    if py_parity_intact and wiki_no_harm:
        print("\n  *** KNOWLEDGE TRANSFER IS ADDITIVE ***")
        print("  *** Python calibration preserves general language capability ***")
    if dual_parity_intact:
        print("  *** MODULAR COMPOSABILITY CONFIRMED ***")
        print("  *** Python + WikiText parity coexist after double-calibration ***")

    print(f"\n  Total runtime: {(time.time()-t_global)/60:.1f} min")

    output = {
        "ppl": ppl_results,
        "cross_domain_l2": {
            "calibrated_vs_native": {
                "python":   py_l2,
                "wikitext": wiki_l2,
            },
        },
        "double_calibration": {
            "py_parity_after_double_cal":   py_l2_after_double,
            "wiki_parity_after_double_cal": wiki_l2_after_double,
            "ppl_py":   round(py_ppl_double, 3),
            "ppl_wiki": round(wiki_ppl_double, 3),
        },
        "interference_summary": {
            "wiki_ppl_delta_vs_transferred": round(wiki_ppl_delta, 4),
            "python_parity_intact":    py_parity_intact,
            "wiki_non_degradation":    wiki_no_harm,
            "dual_parity_composable":  dual_parity_intact,
        },
    }
    out_path = ROOT / "knowledge_non_interference_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
