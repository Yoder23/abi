#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 36 -- Cross-Size NIB v9: GPT-2-large, d_abi=640 (larger ABI)
=========================================================================
Post-mortem of Exp 34 failure:

Exp 27 (standard KD, 1200 steps, d_abi=320):  top-5 = 0.8446
Exp 34 (Top-K KD,   2000 steps, d_abi=320):   top-5 = 0.8402  ← REGRESSION

The Top-K KD hypothesis was wrong: restricting to teacher's top-100 did NOT help.
Diagnosis: Top-K KD pushes the student to agree on teacher's top-100 tokens, but
does NOT penalise the student for ranking tokens OUTSIDE teacher's top-100 highly.
Those uninhibited student-preferred tokens can still block the teacher's rank 2-5
tokens from entering the top-5 overlap set.

Pattern in all 8 large-model experiments (Exps 21-34):
  - top-1 agreement: ~0.91  (far above 0.68 threshold)
  - top-5 overlap:   ~0.84  (0.02 below 0.86 threshold)
  - JS divergence:   ~0.02  (far below 0.10 threshold)
  - entropy diff:    ~0.25  (far below 0.35 threshold)

Interpretation: The model matches the dominant token (top-1) very well and overall
distribution shape (JS, entropy) is excellent. But tokens 2-5 have systematic rank
disagreement. This is consistent with an ABI BOTTLENECK hypothesis:

d_abi = 320 = d_model // 4 = 1280 // 4

The 320-dim ABI is sufficient to encode the primary token direction (top-1 = 0.91)
but loses fine-grained rank information for positions 2-5. For GPT-2-small (768→192),
the same ratio works because the model has fewer layers (12) and less representational
complexity. For GPT-2-large (1280→320 with 36 layers), the rank-2 through rank-5
token directions cannot be distinguished through this bottleneck.

Fix: Double the ABI capacity:
  d_abi = 640 = d_model // 2

This gives 2× more dimensions for the ABI to encode rank-ordered token preferences,
which should directly enable the calibrated model to preserve tokens 2-5 ordering.

KD method: Standard full-vocab KD (reverting from Top-K, since Top-K regressed).
Standard KD's full-vocab gradient is diluted but still correct — the bottleneck was
never the KD method, it was the ABI's capacity to represent rank ordering.

Calibration steps: 2000 (same as Exp 34, adequate for convergence).

All other hyperparameters identical to Exp 27/34.

Result: cross_size_large_nib_v9_results.json
Exp: 36
Baseline to beat: Exp 27, top-5 = 0.8446 (standard KD, d_abi=320, 1200 steps)
"""

import copy
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wikitext_cache import load_wikitext_split

sys.stdout.reconfigure(line_buffering=True)

ROOT   = pathlib.Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Pre-registered NIB thresholds (identical to all prior experiments) ────────
REGISTRY = {
    "js_threshold":           0.10,
    "top1_threshold":         0.68,
    "top5_threshold":         0.86,
    "entropy_diff_threshold": 0.35,
    "n_logit_chunks":         5,
    "calibration_steps":      2000,
    "kd_weight":              0.90,
    "kd_temp":                2.0,
}

# KEY CHANGE: d_abi = d_model // 2 = 640 (was d_model // 4 = 320 in Exps 21-34)
D_MODEL      = 1280
D_ABI        = D_MODEL // 2        # 640  ← doubled from 320
SEQ_LEN      = 128
DOMAIN_STEPS = 500
UPDATE_STEPS = 1000
LR_ABI       = 3e-4
LR_BACKBONE  = 5e-5
LR_CAL       = 1e-4
ALPHA        = 1.0
BATCH_SV     = 4
SEED         = 42
VOCAB_SIZE   = 50257
MAX_PY       = 500_000
MAX_WIKI     = 600_000

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Model architecture (d_abi=640 is the only change from Exp 34) ─────────────

class DomainModuleSV(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ln  = nn.LayerNorm(d)
    def forward(self, h):
        return self.ln(self.net(h))


class SVGPT2Large(nn.Module):
    """GPT-2-large (774M, d_model=1280) with ABI wrapper, d_abi=640."""
    def __init__(self):
        super().__init__()
        g             = GPT2LMHeadModel.from_pretrained("gpt2-large")
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        self.d_model  = g.config.n_embd   # 1280
        self.proj_in  = nn.Linear(self.d_model, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, self.d_model, bias=False)
        self.domain   = DomainModuleSV(D_ABI)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def encode_core(self, x):
        h     = self.backbone(x).last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        return h, h_abi

    def forward(self, x, use_domain=True):
        h, h_abi = self.encode_core(x)
        if use_domain:
            h_out = h_abi + self.domain_alpha * self.domain(h_abi)
        else:
            h_out = h_abi
        return self.lm_head(self.proj_out(h_out) + h)


# ── Utilities ─────────────────────────────────────────────────────────────────

def make_batch(tokens, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    starts = torch.randint(0, max_start, (BATCH_SV,), generator=rng)
    x = torch.stack([tokens[s : s + SEQ_LEN]     for s in starts]).to(DEVICE)
    y = torch.stack([tokens[s+1 : s + SEQ_LEN+1] for s in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def ppl(model, tokens, use_domain=True, n_batches=50, seed_offset=0):
    model.eval()
    tot, n = 0.0, 0
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    rng = torch.Generator()
    for i in range(n_batches):
        rng.manual_seed(80000 + seed_offset + i)
        starts = torch.randint(0, max_start, (BATCH_SV,), generator=rng)
        x = torch.stack([tokens[s : s + SEQ_LEN]     for s in starts]).to(DEVICE)
        y = torch.stack([tokens[s+1 : s + SEQ_LEN+1] for s in starts]).to(DEVICE)
        logits = model(x, use_domain=use_domain)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1)).item()
        n   += 1
    return math.exp(tot / n)


# ── Training protocol A->B->C->D (standard KD, no Top-K restriction) ─────────

def run_protocol(py_ids, wiki_ids):
    kd_weight = REGISTRY["kd_weight"]
    kd_temp   = REGISTRY["kd_temp"]
    cal_steps = REGISTRY["calibration_steps"]

    # Step A: anchor on Python (ABI only, backbone frozen)
    print("  [A] Anchor training (500 steps Python, ABI only)...")
    t0 = time.time()
    anchor = SVGPT2Large().to(DEVICE)
    for p in anchor.parameters():
        p.requires_grad_(False)
    for nm, p in anchor.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_a = torch.optim.AdamW(
        [p for p in anchor.parameters() if p.requires_grad],
        lr=LR_ABI, weight_decay=0.01)
    anchor.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids, seed=5000 + step)
        opt_a.zero_grad()
        F.cross_entropy(anchor(x, use_domain=True).reshape(-1, VOCAB_SIZE),
                        y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(anchor.parameters(), 1.0)
        opt_a.step()
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    saved_dom = copy.deepcopy(anchor.domain.state_dict())
    ppl_a = ppl(anchor, py_ids, use_domain=True)
    print(f"  [A] {time.time()-t0:.0f}s  ppl={ppl_a:.1f}")

    # Step B: backbone drift on WikiText (ABI stability loss)
    print("  [B] Backbone update (1000 steps WikiText-2)...")
    t1 = time.time()
    transferred = copy.deepcopy(anchor).to(DEVICE)
    for p in transferred.parameters():
        p.requires_grad_(False)
    for nm, p in transferred.named_parameters():
        if "backbone" in nm or "proj_in" in nm or "abi_ln" in nm:
            p.requires_grad_(True)
    transferred.proj_out.requires_grad_(False)
    opt_b = torch.optim.AdamW(
        [p for p in transferred.parameters() if p.requires_grad],
        lr=LR_BACKBONE, weight_decay=0.01)
    transferred.train()
    for step in range(UPDATE_STEPS):
        x, y = make_batch(wiki_ids, seed=9000 + step)
        opt_b.zero_grad()
        h, h_abi = transferred.encode_core(x)
        logits = transferred.lm_head(transferred.proj_out(h_abi) + h)
        ll = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        with torch.no_grad():
            _, h_aa = anchor.encode_core(x)
        sl = F.mse_loss(h_abi, h_aa)
        (ll + ALPHA * sl).backward()
        nn.utils.clip_grad_norm_(transferred.parameters(), 1.0)
        opt_b.step()
    transferred.domain.load_state_dict(saved_dom)
    transferred.eval()
    for p in transferred.parameters():
        p.requires_grad_(False)
    ppl_b = ppl(transferred, py_ids, use_domain=False)
    print(f"  [B] {time.time()-t1:.0f}s  no-domain ppl={ppl_b:.1f}")

    # Step C: native oracle (fresh ABI on transferred backbone)
    print("  [C] Native oracle (500 steps Python, fresh ABI)...")
    t2 = time.time()
    native = copy.deepcopy(transferred).to(DEVICE)
    nn.init.xavier_uniform_(native.proj_in.weight)
    nn.init.xavier_uniform_(native.proj_out.weight)
    nn.init.ones_(native.abi_ln.weight)
    nn.init.zeros_(native.abi_ln.bias)
    native.domain = DomainModuleSV(D_ABI).to(DEVICE)
    native.domain_alpha.data.fill_(1.0)
    for p in native.parameters():
        p.requires_grad_(False)
    for nm, p in native.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_c = torch.optim.AdamW(
        [p for p in native.parameters() if p.requires_grad],
        lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids, seed=5000 + step)
        opt_c.zero_grad()
        F.cross_entropy(native(x, use_domain=True).reshape(-1, VOCAB_SIZE),
                        y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt_c.step()
    native.eval()
    for p in native.parameters():
        p.requires_grad_(False)
    ppl_nat = ppl(native, py_ids, use_domain=True)
    print(f"  [C] {time.time()-t2:.0f}s  native ppl={ppl_nat:.1f}")

    # Step D: Standard full-vocab KD calibration (reverted from Top-K)
    print(f"  [D] Standard KD calibration ({cal_steps} steps)...")
    print(f"      kd_weight={kd_weight}  kd_temp={kd_temp}  (full-vocab KL)")
    t3 = time.time()
    calibrated = copy.deepcopy(transferred).to(DEVICE)
    for p in calibrated.parameters():
        p.requires_grad_(False)
    calibrated.proj_in.weight.requires_grad_(True)
    calibrated.proj_out.weight.requires_grad_(True)
    calibrated.domain_alpha.requires_grad_(True)
    calibrated.domain.ln.weight.requires_grad_(True)
    calibrated.domain.ln.bias.requires_grad_(True)
    for p in calibrated.domain.net.parameters():
        p.requires_grad_(True)
    cal_params = (
        [calibrated.proj_in.weight,
         calibrated.proj_out.weight,
         calibrated.domain_alpha,
         calibrated.domain.ln.weight,
         calibrated.domain.ln.bias]
        + list(calibrated.domain.net.parameters())
    )
    opt_d = torch.optim.AdamW(cal_params, lr=LR_CAL, weight_decay=0.01)

    native.eval()
    calibrated.train()
    for step in range(cal_steps):
        x, y = make_batch(py_ids, seed=7000 + step)
        opt_d.zero_grad()
        cal_logits = calibrated(x, use_domain=True)
        with torch.no_grad():
            nat_logits = native(x, use_domain=True)
        V = cal_logits.shape[-1]
        # Standard full-vocab KD (same as Exp 27 which got 0.8446)
        nat_soft = F.softmax(nat_logits.reshape(-1, V).float() / kd_temp, dim=-1)
        cal_log  = F.log_softmax(cal_logits.reshape(-1, V).float() / kd_temp, dim=-1)
        kd_loss  = F.kl_div(cal_log, nat_soft, reduction="batchmean") * (kd_temp ** 2)
        ce_loss  = F.cross_entropy(cal_logits.reshape(-1, V), y.reshape(-1))
        total_loss = (kd_weight * kd_loss) + ((1 - kd_weight) * ce_loss)
        total_loss.backward()
        nn.utils.clip_grad_norm_(cal_params, 1.0)
        opt_d.step()
        if (step + 1) % 200 == 0:
            print(f"      step {step+1}/{cal_steps}  kd={kd_loss.item():.4f}  ce={ce_loss.item():.4f}")
    calibrated.eval()
    for p in calibrated.parameters():
        p.requires_grad_(False)
    ppl_cal = ppl(calibrated, py_ids, use_domain=True)
    print(f"  [D] {time.time()-t3:.0f}s  cal ppl={ppl_cal:.1f}")

    return native, calibrated, ppl_nat, ppl_cal


# ── L2 logit NIB test (identical to all prior experiments) ───────────────────

@torch.no_grad()
def l2_logit_test(native, calibrated, py_ids):
    native.eval(); calibrated.eval()
    CHUNK = 512
    SKIP  = 20
    rng   = np.random.default_rng(7777)
    js_list, top1_list, top5_list, ent_list = [], [], [], []
    n_chunks  = REGISTRY["n_logit_chunks"]
    max_start = max(len(py_ids) - CHUNK, 1)

    for ci in range(n_chunks):
        start = int(rng.integers(0, max_start))
        chunk = py_ids[start : start + CHUNK].unsqueeze(0).to(DEVICE)
        nat_logits = native(chunk, use_domain=True)[0, SKIP:, :]
        cal_logits = calibrated(chunk, use_domain=True)[0, SKIP:, :]
        nat_p = F.softmax(nat_logits, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_logits, dim=-1).cpu().float().numpy()
        T   = nat_p.shape[0]
        eps = 1e-12
        m   = 0.5 * (nat_p + cal_p)
        kl_n = (np.clip(nat_p, eps, 1) * np.log(np.clip(nat_p, eps, 1) / np.clip(m, eps, 1))).sum(1)
        kl_c = (np.clip(cal_p, eps, 1) * np.log(np.clip(cal_p, eps, 1) / np.clip(m, eps, 1))).sum(1)
        js_list.extend(np.clip(0.5 * (kl_n + kl_c), 0, None).tolist())
        top1_list.extend((nat_p.argmax(1) == cal_p.argmax(1)).tolist())
        n5 = np.argpartition(nat_p, -5, axis=1)[:, -5:]
        c5 = np.argpartition(cal_p, -5, axis=1)[:, -5:]
        for t in range(T):
            top5_list.append(len(set(n5[t]) & set(c5[t])) / 5.0)
        Hn = -(np.clip(nat_p, eps, 1) * np.log(np.clip(nat_p, eps, 1))).sum(1)
        Hc = -(np.clip(cal_p, eps, 1) * np.log(np.clip(cal_p, eps, 1))).sum(1)
        ent_list.extend(np.abs(Hn - Hc).tolist())
        print(f"    chunk {ci+1}/{n_chunks}: JS={float(np.mean(js_list)):.4f} "
              f"top1={float(np.mean(top1_list)):.3f} top5={float(np.mean(top5_list)):.3f}")

    mj  = float(np.mean(js_list))
    mt1 = float(np.mean(top1_list))
    mt5 = float(np.mean(top5_list))
    me  = float(np.mean(ent_list))
    jp  = mj  <  REGISTRY["js_threshold"]
    t1p = mt1 >= REGISTRY["top1_threshold"]
    t5p = mt5 >= REGISTRY["top5_threshold"]
    ep  = me  <  REGISTRY["entropy_diff_threshold"]
    return {
        "n_positions":        len(js_list),
        "mean_js":            round(mj,  5),
        "mean_top1_agree":    round(mt1, 4),
        "mean_top5_overlap":  round(mt5, 4),
        "mean_entropy_diff":  round(me,  4),
        "js_pass":            jp,
        "top1_pass":          t1p,
        "top5_pass":          t5p,
        "entropy_pass":       ep,
        "pass":               jp and t1p and t5p and ep,
        "thresholds": {
            "js":           REGISTRY["js_threshold"],
            "top1":         REGISTRY["top1_threshold"],
            "top5":         REGISTRY["top5_threshold"],
            "entropy_diff": REGISTRY["entropy_diff_threshold"],
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def banner(msg):
    print()
    print("=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def main():
    t_global = time.time()
    banner("Experiment 36 -- GPT-2-large: d_abi=640 (doubled ABI capacity)")
    print(f"  Device:      {DEVICE}")
    print(f"  Model:       GPT-2-large (774M, d_model=1280, 36 layers)")
    print(f"  D_ABI:       {D_ABI}  (= d_model // 2 = 640)  ← KEY CHANGE from 320")
    print(f"  KD method:   Standard full-vocab KD (reverted from Top-K)")
    print(f"  Cal steps:   {REGISTRY['calibration_steps']}")
    print()
    print("  Hypothesis: d_abi=320 bottleneck can't encode rank-2..5 token order.")
    print("  d_abi=640 (2x) gives ABI sufficient dimensions to preserve top-5 order.")
    print("  Standard KD reverted: Top-K KD (Exp 34) regressed vs Exp 27 baseline.")
    print()
    print("  Exp history (d_abi=320 all): top-5 range 0.821-0.8446 (all FAIL)")
    print(f"  Exp 27 baseline (best):      top-5 = 0.8446")
    print(f"  Exp 34 (Top-K KD attempt):   top-5 = 0.8402 (regression)")
    print(f"  This exp (d_abi=640):        target top-5 >= 0.8600")
    print()

    # Data
    print("  [Data] Loading corpora...")
    t_data = time.time()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = sys.maxsize
    wiki_raw = "\n".join(
        r["text"] for r in load_wikitext_split("wikitext-2-raw-v1", "train")
        if r["text"].strip())
    py_parts, py_chars = [], 0
    for p in ROOT.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            py_parts.append(txt)
            py_chars += len(txt)
            if py_chars >= MAX_PY * 4:
                break
        except Exception:
            continue
    py_raw   = "\n".join(py_parts)
    py_ids   = tok(py_raw,   return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_PY]
    wiki_ids = tok(wiki_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_WIKI]
    print(f"  [Data] {time.time()-t_data:.1f}s  py={len(py_ids):,}  wiki={len(wiki_ids):,}")
    print()

    # Protocol
    banner(f"Training Protocol: A -> B -> C -> D  (GPT-2-large, d_abi={D_ABI})")
    native, calibrated, ppl_nat, ppl_cal = run_protocol(py_ids, wiki_ids)

    # L2 NIB
    banner("L2 -- Distributional Equivalence")
    print("  Running 5 x 512-token forward passes...")
    t_l2 = time.time()
    l2 = l2_logit_test(native, calibrated, py_ids)
    print()
    print(f"  mean_JS          = {l2['mean_js']:.5f}   (thr < {REGISTRY['js_threshold']})   "
          f"{'PASS' if l2['js_pass'] else 'FAIL'}")
    print(f"  mean_top1_agree  = {l2['mean_top1_agree']:.4f}  (thr >= {REGISTRY['top1_threshold']})  "
          f"{'PASS' if l2['top1_pass'] else 'FAIL'}")
    print(f"  mean_top5_overlap= {l2['mean_top5_overlap']:.4f}  (thr >= {REGISTRY['top5_threshold']})  "
          f"{'PASS' if l2['top5_pass'] else 'FAIL'}")
    print(f"  mean_entropy_diff= {l2['mean_entropy_diff']:.4f}  (thr < {REGISTRY['entropy_diff_threshold']})  "
          f"{'PASS' if l2['entropy_pass'] else 'FAIL'}")
    print()
    overall    = l2["pass"]
    status_str = "PASS" if overall else "FAIL"
    print(f"  L2 NIB overall: {status_str}")
    print(f"  L2 eval time: {time.time()-t_l2:.1f}s")

    # Comparison summary
    print()
    print("=" * 72)
    print("  Experiment Progression (GPT-2-large, all on Python domain)")
    print("=" * 72)
    history = [
        ("Exp 27", "standard KD",    320, 1200, 0.8446),
        ("Exp 34", "Top-K KD K=100", 320, 2000, 0.8402),
        ("Exp 36", f"standard KD",   640, 2000, l2["mean_top5_overlap"]),
    ]
    print(f"  {'Exp':<8} {'KD method':<22} {'d_abi':>6} {'cal_steps':>10} {'top-5':>8}  Status")
    print("  " + "-" * 65)
    for name, method, dabi, steps, top5 in history:
        s = "PASS" if top5 >= 0.86 else "FAIL"
        marker = " ← THIS" if name == "Exp 36" else ""
        print(f"  {name:<8} {method:<22} {dabi:>6} {steps:>10} {top5:>8.4f}  {s}{marker}")
    print()
    gap = 0.86 - l2["mean_top5_overlap"]
    if overall:
        print("  *** TOP-5 THRESHOLD CROSSED — GPT-2-LARGE NIB PASS ***")
        print("  *** All model sizes 117M-774M now PASS 4/4 NIB thresholds ***")
    elif gap < 0.005:
        print(f"  Gap = {gap:.4f} — within 0.005. Consider increasing cal_steps to 4000.")
    elif gap < 0.010:
        print(f"  Gap = {gap:.4f} — try d_abi=768 (= d_model) or 4000 cal_steps.")
    else:
        print(f"  Gap = {gap:.4f} — ABI capacity alone insufficient. ")
        print(f"  Next hypothesis: backbone drift in step B is too large for 36 layers.")
        print(f"  Suggestion: reduce LR_BACKBONE (5e-5→2e-5) or UPDATE_STEPS (1000→500).")

    elapsed = time.time() - t_global

    results = {
        "experiment":        36,
        "name":              "cross_size_large_nib_v9",
        "model":             "gpt2-large",
        "model_params_M":    774,
        "d_model":           D_MODEL,
        "d_abi":             D_ABI,
        "d_abi_rule":        "d_model // 2  (doubled from prior experiments)",
        "abi_ratio":         round(D_ABI / D_MODEL, 4),
        "n_layers":          36,
        "seed":              SEED,
        "batch_size":        BATCH_SV,
        "domain_steps":      DOMAIN_STEPS,
        "update_steps":      UPDATE_STEPS,
        "calibration_steps": REGISTRY["calibration_steps"],
        "kd_method":         "standard_full_vocab",
        "kd_weight":         REGISTRY["kd_weight"],
        "kd_temp":           REGISTRY["kd_temp"],
        "ppl_native":        round(ppl_nat, 3),
        "ppl_calibrated":    round(ppl_cal, 3),
        "l2_nib":            l2,
        "nib_pass":          overall,
        "total_runtime_s":   round(elapsed, 1),
        "experiment_history": {
            "exp27": {"kd": "standard", "d_abi": 320, "cal_steps": 1200, "top5": 0.8446},
            "exp34": {"kd": "top_k_k100", "d_abi": 320, "cal_steps": 2000, "top5": 0.8402},
            "exp36": {"kd": "standard", "d_abi": D_ABI, "cal_steps": REGISTRY["calibration_steps"],
                      "top5": l2["mean_top5_overlap"]},
        },
    }
    out = ROOT / "cross_size_large_nib_v9_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    banner(f"Exp 36 complete: {status_str}  --  {elapsed/60:.1f} min")
    print(f"  Results -> {out.name}")


if __name__ == "__main__":
    main()
