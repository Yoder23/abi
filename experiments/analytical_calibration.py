#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytical Calibration Experiment
==================================
HYPOTHESIS: Step D (800-step KD calibration) can be replaced by a single
closed-form linear solve — a Procrustes alignment in ABI hidden space.

If the error after A→B→C is purely a linear coordinate mismatch (not a
representational gap), then one matrix solve produces equivalent transfer
quality to 800 gradient steps.

PROTOCOL:
  A → B → C  (same as NIB run 8 / abi_scaling_law.py)
  Then compare FIVE methods for Step D:

  1. post_c_raw      — transferred model, no calibration (0 steps)
  2. analytical      — Procrustes linear solve (0 SGD steps)
  3. anal_plus_50    — analytical init + 50 KD steps
  4. anal_plus_200   — analytical init + 200 KD steps
  5. sgd_800_scratch — KD from scratch, 800 steps (the proven baseline)

ANALYTICAL SOLVE:
  Collect N (h_full_cal, h_full_nat) pairs in ABI bottleneck space (d_abi=256).
  h_full = h_abi + alpha * domain(h_abi)  for each model respectively.
  Backbone hidden state h is identical for both (shared drifted backbone).

  Solve: A* = lstsq(H_cal, H_nat)  [d_abi × d_abi]
  s.t. H_cal @ A* ≈ H_nat

  Bake into transferred model:
    proj_out_new.weight = proj_out_nat.weight @ A*.T   [d_model × d_abi]
  → one forward pass, no training.

  R² of the linear fit measures whether the residual error is linear.

Results: analytical_calibration_results.json
"""

import copy
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wikitext_cache import load_wikitext_split

ROOT   = pathlib.Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_ABI        = 256
SEQ_LEN      = 128
DOMAIN_STEPS = 500
UPDATE_STEPS = 1000
LR_ABI       = 3e-4
LR_BACKBONE  = 5e-5
LR_CAL       = 1e-4
KD_WEIGHT    = 0.90
KD_TEMP      = 2.0
ALPHA        = 1.0
MAX_PY_SV    = 500_000
MAX_WIKI_SV  = 600_000
BATCH_SV     = 8
SEED         = 42

N_LOGIT_CHUNKS     = 5
CHUNK_SIZE         = 512
SKIP_POS           = 20
N_COLLECT_BATCHES  = 200    # 200 × 8 × 128 = 204,800 ABI pair samples for lstsq

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DomainModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        self.ln  = nn.LayerNorm(d)
    def forward(self, h): return self.ln(self.net(h))


class ABI_GPT2(nn.Module):
    def __init__(self, d_abi=D_ABI):
        super().__init__()
        g             = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        self.d_model  = g.config.n_embd
        self.d_abi    = d_abi
        self.proj_in  = nn.Linear(self.d_model, d_abi, bias=False)
        self.abi_ln   = nn.LayerNorm(d_abi)
        self.proj_out = nn.Linear(d_abi, self.d_model, bias=False)
        self.domain   = DomainModule(d_abi)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def encode_core(self, x):
        h = self.backbone(x).last_hidden_state
        return h, self.abi_ln(self.proj_in(h))

    def abi_full(self, x):
        """Return (h_backbone, h_full_abi) where h_full includes domain."""
        h, h_abi = self.encode_core(x)
        h_full = h_abi + self.domain_alpha * self.domain(h_abi)
        return h, h_full

    def forward(self, x, use_domain=True):
        h, h_abi = self.encode_core(x)
        h_out = h_abi + self.domain_alpha * self.domain(h_abi) if use_domain else h_abi
        return self.lm_head(self.proj_out(h_out) + h)


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
    py_raw = "\n".join(py_parts)
    py_ids = tok(py_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_PY_SV]
    wiki_ds  = load_wikitext_split("wikitext-2-raw-v1", "train")
    wiki_raw = "\n".join(r["text"] for r in wiki_ds if r["text"].strip())
    wiki_ids = tok(wiki_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_WIKI_SV]
    print(f"  py_ids={len(py_ids):,}  wiki_ids={len(wiki_ids):,}")
    return tok, py_ids, wiki_ids


def make_batch(tokens, seed):
    rng = torch.Generator(); rng.manual_seed(seed)
    max_s = max(len(tokens) - SEQ_LEN - 1, 1)
    starts = torch.randint(0, max_s, (BATCH_SV,), generator=rng)
    x = torch.stack([tokens[s:s+SEQ_LEN]     for s in starts]).to(DEVICE)
    y = torch.stack([tokens[s+1:s+SEQ_LEN+1] for s in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def ppl(model, tokens, n=50):
    model.eval()
    losses = []
    for i in range(n):
        x, y = make_batch(tokens, seed=8000+i)
        losses.append(F.cross_entropy(model(x).reshape(-1,50257), y.reshape(-1)).item())
    return float(np.exp(np.mean(losses)))


# ══════════════════════════════════════════════════════════════════════════════
# SHARED A → B → C  (run once)
# ══════════════════════════════════════════════════════════════════════════════

def run_abc(py_ids, wiki_ids):
    """Returns (transferred_state_dict, native, ppl_nat)."""
    t0 = time.time()

    # A
    print("  [A] anchor...")
    anchor = ABI_GPT2().to(DEVICE)
    for p in anchor.parameters(): p.requires_grad_(False)
    for nm, p in anchor.named_parameters():
        if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain")): p.requires_grad_(True)
    opt_a = torch.optim.AdamW([p for p in anchor.parameters() if p.requires_grad], lr=LR_ABI, weight_decay=0.01)
    anchor.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids, seed=5000+step)
        opt_a.zero_grad()
        F.cross_entropy(anchor(x).reshape(-1,50257), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(anchor.parameters(), 1.0); opt_a.step()
    anchor.eval(); [p.requires_grad_(False) for p in anchor.parameters()]
    saved_dom = copy.deepcopy(anchor.domain.state_dict())
    print(f"  [A] {time.time()-t0:.0f}s")

    # B
    t1 = time.time()
    print("  [B] backbone drift...")
    transferred = copy.deepcopy(anchor).to(DEVICE)
    for p in transferred.parameters(): p.requires_grad_(False)
    for nm, p in transferred.named_parameters():
        if "backbone" in nm or "proj_in" in nm or "abi_ln" in nm: p.requires_grad_(True)
    opt_b = torch.optim.AdamW([p for p in transferred.parameters() if p.requires_grad], lr=LR_BACKBONE, weight_decay=0.01)
    transferred.train()
    for step in range(UPDATE_STEPS):
        x, y = make_batch(wiki_ids, seed=9000+step)
        opt_b.zero_grad()
        h, h_abi = transferred.encode_core(x)
        logits = transferred.lm_head(transferred.proj_out(h_abi)+h)
        ll = F.cross_entropy(logits.reshape(-1,50257), y.reshape(-1))
        with torch.no_grad(): _, h_aa = anchor.encode_core(x)
        sl = F.mse_loss(h_abi, h_aa)
        (ll + ALPHA*sl).backward()
        nn.utils.clip_grad_norm_(transferred.parameters(), 1.0); opt_b.step()
    transferred.domain.load_state_dict(saved_dom)
    transferred.eval(); [p.requires_grad_(False) for p in transferred.parameters()]
    transferred_state = copy.deepcopy(transferred.state_dict())
    print(f"  [B] {time.time()-t1:.0f}s")

    # C — native oracle
    t2 = time.time()
    print("  [C] native oracle...")
    native = copy.deepcopy(transferred).to(DEVICE)
    nn.init.xavier_uniform_(native.proj_in.weight); nn.init.xavier_uniform_(native.proj_out.weight)
    nn.init.ones_(native.abi_ln.weight); nn.init.zeros_(native.abi_ln.bias)
    native.domain = DomainModule(D_ABI).to(DEVICE); native.domain_alpha.data.fill_(1.0)
    for p in native.parameters(): p.requires_grad_(False)
    for nm, p in native.named_parameters():
        if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain")): p.requires_grad_(True)
    opt_c = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad], lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids, seed=5000+step)
        opt_c.zero_grad()
        F.cross_entropy(native(x).reshape(-1,50257), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0); opt_c.step()
    native.eval(); [p.requires_grad_(False) for p in native.parameters()]
    ppl_nat = ppl(native, py_ids)
    print(f"  [C] {time.time()-t2:.0f}s  ppl_nat={ppl_nat:.3f}")
    print(f"  A→B→C: {(time.time()-t0)/60:.1f} min")
    return transferred_state, native, ppl_nat


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL CALIBRATION (Procrustes in ABI space)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def solve_analytical_d(transferred_state, native, py_ids):
    """
    Collect (h_full_cal, h_full_nat) pairs in ABI space.
    Solve A* = lstsq(H_cal, H_nat).
    Bake A* into proj_out: proj_out_new.weight = proj_out_nat.weight @ A*.T
    Return (analytically_calibrated_model, r_squared, cond_number).
    """
    # Reconstruct transferred model from saved state
    transferred = ABI_GPT2().to(DEVICE)
    transferred.load_state_dict(transferred_state)
    transferred.eval(); [p.requires_grad_(False) for p in transferred.parameters()]

    all_h_cal, all_h_nat = [], []
    t_col = time.time()

    for i in range(N_COLLECT_BATCHES):
        x, _ = make_batch(py_ids, seed=3000+i)
        # Backbone output h is identical for both (same weights from Step B)
        h, h_abi_cal = transferred.encode_core(x)
        h_full_cal   = h_abi_cal + transferred.domain_alpha * transferred.domain(h_abi_cal)

        _, h_abi_nat = native.encode_core(x)
        h_full_nat   = h_abi_nat + native.domain_alpha * native.domain(h_abi_nat)

        all_h_cal.append(h_full_cal.reshape(-1, D_ABI).cpu().float())
        all_h_nat.append(h_full_nat.reshape(-1, D_ABI).cpu().float())

    H_cal = torch.cat(all_h_cal, dim=0)   # N × d_abi
    H_nat = torch.cat(all_h_nat, dim=0)   # N × d_abi
    N     = len(H_cal)
    print(f"  [Analytical] collected {N:,} ABI pairs in {time.time()-t_col:.0f}s")

    # Solve: H_cal @ A* ≈ H_nat  →  A* in R^{d_abi × d_abi}
    # torch.linalg.lstsq: A (m,n) B (m,k) → X (n,k) s.t. A@X ≈ B
    result  = torch.linalg.lstsq(H_cal, H_nat, rcond=1e-4, driver='gelsd')
    A_star  = result.solution   # d_abi × d_abi

    # Condition number of H_cal (squared matrix HH^T)
    sv = torch.linalg.svdvals(H_cal[:min(N, D_ABI*4), :D_ABI])   # fast on small slice
    cond = float(sv.max() / (sv.min() + 1e-12))

    # R² of linear fit
    H_nat_pred = H_cal @ A_star
    SS_res = ((H_nat - H_nat_pred)**2).sum().item()
    SS_tot = ((H_nat - H_nat.mean(0))**2).sum().item()
    r2     = float(1.0 - SS_res / (SS_tot + 1e-12))
    print(f"  [Analytical] R²={r2:.4f}  cond(H_cal)={cond:.1f}")

    # Bake A* into proj_out:
    # Forward: h_full_cal → (h_full_cal @ A*) → proj_out_nat → + h → lm_head
    # Equivalent: proj_out_new s.t. proj_out_new(h) = proj_out_nat(h @ A*)
    # proj_out_new.weight = proj_out_nat.weight @ A*.T  (both d_model × d_abi)
    new_proj_out_w = native.proj_out.weight.data.cpu().float() @ A_star.cpu().float().T  # d_model × d_abi

    # Build the analytically-calibrated model (copy of transferred with updated proj_out)
    anal = ABI_GPT2().to(DEVICE)
    anal.load_state_dict(transferred_state)
    anal.proj_out.weight.data.copy_(new_proj_out_w.to(DEVICE))
    anal.eval(); [p.requires_grad_(False) for p in anal.parameters()]

    return anal, float(r2), float(cond), A_star


# ══════════════════════════════════════════════════════════════════════════════
# STEP D — KD calibration with variable budget and optional analytical init
# ══════════════════════════════════════════════════════════════════════════════

def run_kd_calibration(start_state_dict, native, py_ids, n_steps):
    """KD calibration from start_state_dict for n_steps. Returns calibrated model."""
    if n_steps == 0:
        model = ABI_GPT2().to(DEVICE)
        model.load_state_dict(start_state_dict)
        model.eval(); [p.requires_grad_(False) for p in model.parameters()]
        return model

    model = ABI_GPT2().to(DEVICE)
    model.load_state_dict(start_state_dict)
    for p in model.parameters(): p.requires_grad_(False)

    _params = [model.proj_in.weight, model.proj_out.weight,
               model.domain_alpha, model.domain.ln.weight, model.domain.ln.bias]
    for p in _params: p.requires_grad_(True)
    opt    = torch.optim.AdamW(_params, lr=LR_CAL, weight_decay=0.01)
    ce_w   = 1.0 - KD_WEIGHT
    native.eval(); model.train()

    for step in range(n_steps):
        x, y = make_batch(py_ids, seed=7000+step)
        opt.zero_grad()
        cal_lo = model(x)
        with torch.no_grad(): nat_lo = native(x)
        V  = cal_lo.shape[-1]
        kd = F.kl_div(F.log_softmax(cal_lo.reshape(-1,V)/KD_TEMP, dim=-1),
                      F.softmax(nat_lo.reshape(-1,V)/KD_TEMP, dim=-1), reduction='batchmean') * (KD_TEMP**2)
        ce = F.cross_entropy(cal_lo.reshape(-1,V), y.reshape(-1))
        (KD_WEIGHT*kd + ce_w*ce).backward()
        nn.utils.clip_grad_norm_(_params, 1.0); opt.step()

    model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    return model


# ══════════════════════════════════════════════════════════════════════════════
# L2 METRICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def l2_metrics(native, model, py_ids, label=""):
    native.eval(); model.eval()
    rng = np.random.default_rng(7777)
    js_list, top1_list, top5_list, ent_list = [], [], [], []
    max_start = max(len(py_ids) - CHUNK_SIZE, 1)
    for ci in range(N_LOGIT_CHUNKS):
        start  = int(rng.integers(0, max_start))
        chunk  = py_ids[start:start+CHUNK_SIZE].unsqueeze(0).to(DEVICE)
        nat_lo = native(chunk)[0, SKIP_POS:, :]
        mod_lo = model (chunk)[0, SKIP_POS:, :]
        nat_p  = F.softmax(nat_lo, dim=-1).cpu().float().numpy()
        mod_p  = F.softmax(mod_lo, dim=-1).cpu().float().numpy()
        T, eps = nat_p.shape[0], 1e-12
        m      = 0.5*(nat_p+mod_p)
        nat_pc = np.clip(nat_p, eps, 1.0)
        mod_pc = np.clip(mod_p, eps, 1.0)
        m_c    = np.clip(m, eps, 1.0)
        kl1    = (nat_pc*np.log(nat_pc/m_c)).sum(1)
        kl2    = (mod_pc*np.log(mod_pc/m_c)).sum(1)
        js_list.extend(np.clip(0.5*(kl1+kl2),0,None).tolist())
        top1_list.extend((nat_p.argmax(1)==mod_p.argmax(1)).tolist())
        nat5 = np.argpartition(nat_p,-5,axis=1)[:,-5:]
        mod5 = np.argpartition(mod_p,-5,axis=1)[:,-5:]
        for t in range(T): top5_list.append(len(set(nat5[t])&set(mod5[t]))/5.)
        H_nat = -(nat_pc*np.log(nat_pc)).sum(1)
        H_mod = -(mod_pc*np.log(mod_pc)).sum(1)
        ent_list.extend(np.abs(H_nat-H_mod).tolist())
    d = {
        "mean_js":          round(float(np.mean(js_list)),5),
        "mean_top1_agree":  round(float(np.mean(top1_list)),4),
        "mean_top5_overlap":round(float(np.mean(top5_list)),4),
        "mean_entropy_diff":round(float(np.mean(ent_list)),4),
    }
    passes_all = (d["mean_js"] < 0.10 and d["mean_top1_agree"] >= 0.68
                  and d["mean_top5_overlap"] >= 0.86 and d["mean_entropy_diff"] < 0.35)
    verdict = "PASS" if passes_all else "FAIL"
    print(f"  [{label}] JS={d['mean_js']:.4f}  top1={d['mean_top1_agree']:.3f}  "
          f"top5={d['mean_top5_overlap']:.3f}  ent={d['mean_entropy_diff']:.3f}  [{verdict}]")
    d["l2_pass"] = passes_all
    return d


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_global = time.time()
    print("=" * 70)
    print("  ANALYTICAL CALIBRATION EXPERIMENT")
    print(f"  d_abi={D_ABI}  kd_weight={KD_WEIGHT}  T={KD_TEMP}")
    print(f"  device: {DEVICE}")
    print("  Hypothesis: Step D is a linear coordinate correction in ABI space.")
    print("=" * 70)

    _tok, py_ids, wiki_ids = load_data()

    print("\n  Running shared A → B → C...")
    transferred_state, native, ppl_nat = run_abc(py_ids, wiki_ids)
    print(f"  ppl_nat = {ppl_nat:.3f}")

    results = {}

    # ── Method 1: Post-C raw (no calibration)
    print("\n  Method 1: post_c_raw (no calibration)")
    t = time.time()
    raw = ABI_GPT2().to(DEVICE); raw.load_state_dict(transferred_state)
    raw.eval(); [p.requires_grad_(False) for p in raw.parameters()]
    ppl_raw = ppl(raw, py_ids)
    l2_raw  = l2_metrics(native, raw, py_ids, label="post_c_raw")
    results["post_c_raw"] = {
        "method": "post_c_raw (0 steps, no analytical)",
        "sgd_steps": 0, "analytical_init": False,
        "ppl_cal": round(ppl_raw,3), "ppl_nat": round(ppl_nat,3),
        "efficacy": round(ppl_raw/ppl_nat*100, 2),
        "runtime_sec": round(time.time()-t, 1), "L2": l2_raw,
    }

    # ── Method 2: Analytical D (Procrustes, 0 SGD)
    print("\n  Method 2: analytical (Procrustes solve, 0 SGD steps)")
    t = time.time()
    anal, r2, cond, A_star = solve_analytical_d(transferred_state, native, py_ids)
    ppl_anal = ppl(anal, py_ids)
    l2_anal  = l2_metrics(native, anal, py_ids, label="analytical")
    anal_state = copy.deepcopy(anal.state_dict())   # save for fine-tuning
    results["analytical"] = {
        "method": "Procrustes linear solve (0 SGD steps)",
        "sgd_steps": 0, "analytical_init": True,
        "r_squared": round(r2, 5),
        "cond_number_H_cal": round(cond, 1),
        "ppl_cal": round(ppl_anal,3), "ppl_nat": round(ppl_nat,3),
        "efficacy": round(ppl_anal/ppl_nat*100, 2),
        "runtime_sec": round(time.time()-t, 1), "L2": l2_anal,
    }

    # ── Method 3: Analytical + 50 SGD steps
    print("\n  Method 3: analytical_init + 50 KD steps")
    t = time.time()
    anal50 = run_kd_calibration(anal_state, native, py_ids, n_steps=50)
    ppl_a50 = ppl(anal50, py_ids)
    l2_a50  = l2_metrics(native, anal50, py_ids, label="anal+50")
    results["anal_plus_50"] = {
        "method": "analytical init + 50 KD steps",
        "sgd_steps": 50, "analytical_init": True,
        "ppl_cal": round(ppl_a50,3), "ppl_nat": round(ppl_nat,3),
        "efficacy": round(ppl_a50/ppl_nat*100, 2),
        "runtime_sec": round(time.time()-t, 1), "L2": l2_a50,
    }

    # ── Method 4: Analytical + 200 SGD steps
    print("\n  Method 4: analytical_init + 200 KD steps")
    t = time.time()
    anal200 = run_kd_calibration(anal_state, native, py_ids, n_steps=200)
    ppl_a200 = ppl(anal200, py_ids)
    l2_a200  = l2_metrics(native, anal200, py_ids, label="anal+200")
    results["anal_plus_200"] = {
        "method": "analytical init + 200 KD steps",
        "sgd_steps": 200, "analytical_init": True,
        "ppl_cal": round(ppl_a200,3), "ppl_nat": round(ppl_nat,3),
        "efficacy": round(ppl_a200/ppl_nat*100, 2),
        "runtime_sec": round(time.time()-t, 1), "L2": l2_a200,
    }

    # ── Method 5: Full SGD from scratch (800 steps, baseline)
    print("\n  Method 5: sgd_800_scratch (standard Step D baseline)")
    t = time.time()
    sgd800 = run_kd_calibration(transferred_state, native, py_ids, n_steps=800)
    ppl_sgd = ppl(sgd800, py_ids)
    l2_sgd  = l2_metrics(native, sgd800, py_ids, label="sgd_800")
    results["sgd_800_scratch"] = {
        "method": "Full SGD 800 steps from transferred (standard baseline)",
        "sgd_steps": 800, "analytical_init": False,
        "ppl_cal": round(ppl_sgd,3), "ppl_nat": round(ppl_nat,3),
        "efficacy": round(ppl_sgd/ppl_nat*100, 2),
        "runtime_sec": round(time.time()-t, 1), "L2": l2_sgd,
    }

    # ── Summary table
    print(f"\n{'='*70}")
    print("  ANALYTICAL CALIBRATION COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Method':<30} {'steps':>6} {'JS':>8} {'top-1':>7} {'top-5':>7} "
          f"{'ent':>7} {'eff%':>7} {'L2':>6}")
    print("-" * 80)
    METHOD_LABELS = [
        ("post_c_raw",     "post-C raw (no D)"),
        ("analytical",     "Procrustes (0 SGD)"),
        ("anal_plus_50",   "Analytical + 50 SGD"),
        ("anal_plus_200",  "Analytical + 200 SGD"),
        ("sgd_800_scratch","Full SGD 800 (baseline)"),
    ]
    for key, label in METHOD_LABELS:
        r = results[key]; l = r["L2"]
        print(f"  {label:<30} {r['sgd_steps']:>6} "
              f"{l['mean_js']:>8.4f} {l['mean_top1_agree']:>7.3f} "
              f"{l['mean_top5_overlap']:>7.3f} {l['mean_entropy_diff']:>7.3f} "
              f"{r['efficacy']:>6.1f}% {'PASS' if l['l2_pass'] else 'FAIL':>6}")

    print(f"\n  R² of Procrustes linear fit: {results['analytical']['r_squared']:.4f}")
    print(f"  Cond(H_cal): {results['analytical']['cond_number_H_cal']:.1f}")
    print()
    if results["analytical"]["L2"]["l2_pass"]:
        print("  *** ANALYTICAL CALIBRATION PASSES L2 (0 SGD steps) ***")
        print("  → Confirms: Step D is a linear coordinate correction, not learning.")
    elif results["anal_plus_50"]["L2"]["l2_pass"]:
        print("  Analytical alone FAILS; Analytical + 50 steps PASSES.")
        print(f"  → 50 SGD steps on analytical init ≡ 800 SGD steps from scratch")
        print(f"  → 16× training cost reduction.")
    elif results["anal_plus_200"]["L2"]["l2_pass"]:
        print("  Analytical + 200 steps PASSES; need fewer than 800 SGD steps.")
        print(f"  → 4× training cost reduction vs full baseline.")
    else:
        print("  No analytical method beats 800-step baseline on all thresholds.")
        print(f"  → The residual error is non-linear; SGD is load-bearing.")

    out = ROOT / "analytical_calibration_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out}")
    print(f"  Total runtime: {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    main()
