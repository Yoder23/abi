#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transition Zone Multi-Seed Validation  (Exp 15 supplement)
===========================================================
Exp 15 found a non-monotone pattern in the transition zone:
  d_abi=4  PASS  (top5=0.861, +0.001 above threshold, single seed)
  d_abi=8  FAIL  (top5=0.848, -0.012 below threshold, single seed)
  d_abi=16 PASS  (top5=0.867, +0.007 above threshold, single seed)

This is suspicious: better R2 at d_abi=8 (0.765) than d_abi=4 (0.565)
but worse top5. The non-monotone pattern could be:
  (a) Genuine: low-d_abi forces alignment to shared principal directions
  (b) Sampling noise: 3 independent seeds should resolve this

This script runs 3 seeds for d_abi in {4, 8, 16} Procrustes-only.
Seeds: 42, 137, 999 (same as Exp 12 robustness sweep for consistency).

Output: transition_zone_results.json
Checkpoint: transition_zone_checkpoint.json
Runtime: ~1.6 hours (9 runs * ~11 min each)
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

sys.stdout.reconfigure(line_buffering=True)

from non_inferiority_benchmark import (
    REGISTRY, DEVICE, SEQ_LEN,
    DOMAIN_STEPS, UPDATE_STEPS,
    LR_ABI, LR_BACKBONE, LR_CAL,
    BATCH_SV, SEED, ROOT,
)
from multi_domain_atlas import (
    load_py_ids, load_wiki_ids,
    N_COLLECT, N_L2_CHUNKS, L2_SKIP,
)

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

D_ABI_TRANSITION = [4, 8, 16]
SEEDS            = [42, 137, 999]

NIB_JS        = REGISTRY["js_threshold"]
NIB_TOP1      = REGISTRY["top1_threshold"]
NIB_TOP5      = REGISTRY["top5_threshold"]
NIB_ENT_DIFF  = REGISTRY["entropy_diff_threshold"]

CHECKPOINT_FILE = ROOT / "transition_zone_checkpoint.json"
RESULT_FILE     = ROOT / "transition_zone_results.json"


# ══════════════════════════════════════════════════════════════════════════════
# Parameterized model (identical to abi_collapse_search.py)
# ══════════════════════════════════════════════════════════════════════════════

class DomainModuleABI(nn.Module):
    def __init__(self, d_abi: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_abi, d_abi * 4),
            nn.GELU(),
            nn.Linear(d_abi * 4, d_abi),
        )
        self.ln = nn.LayerNorm(d_abi)

    def forward(self, h):
        return self.ln(self.net(h))


class ABIModel(nn.Module):
    def __init__(self, d_abi: int):
        super().__init__()
        g              = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        self.backbone  = g.transformer
        self.lm_head   = g.lm_head
        self.d_model   = g.config.n_embd
        self.d_abi     = d_abi
        self.proj_in   = nn.Linear(self.d_model, d_abi, bias=False)
        self.abi_ln    = nn.LayerNorm(d_abi)
        self.proj_out  = nn.Linear(d_abi, self.d_model, bias=False)
        self.domain    = DomainModuleABI(d_abi)
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


def make_batch(tokens, seed: int):
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    starts = torch.randint(0, max_start, (BATCH_SV,), generator=rng)
    x = torch.stack([tokens[s : s + SEQ_LEN]     for s in starts]).to(DEVICE)
    y = torch.stack([tokens[s+1 : s + SEQ_LEN+1] for s in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def model_ppl(model, tokens, n_batches: int = 50) -> float:
    model.eval()
    tot, n = 0.0, 0
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    rng = torch.Generator()
    for i in range(n_batches):
        rng.manual_seed(50000 + i)
        starts = torch.randint(0, max_start, (BATCH_SV,), generator=rng)
        x = torch.stack([tokens[s : s + SEQ_LEN]     for s in starts]).to(DEVICE)
        y = torch.stack([tokens[s+1 : s + SEQ_LEN+1] for s in starts]).to(DEVICE)
        logits = model(x, use_domain=True)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n   += 1
    return math.exp(tot / n)


# ══════════════════════════════════════════════════════════════════════════════
# Training (seeded per-run)
# ══════════════════════════════════════════════════════════════════════════════

def step_A(d_abi: int, py_ids: torch.Tensor, seed_offset: int) -> ABIModel:
    torch.manual_seed(SEED + seed_offset)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + seed_offset)
    model = ABIModel(d_abi).to(DEVICE)
    for p in model.backbone.parameters(): p.requires_grad_(False)
    for p in model.lm_head.parameters():  p.requires_grad_(False)
    params = [model.proj_in.weight, model.proj_out.weight,
              *model.abi_ln.parameters(), *model.domain.parameters(), model.domain_alpha]
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR_ABI, weight_decay=0.01)
    model.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids, seed=step + seed_offset * 10000)
        opt.zero_grad()
        logits = model(x, use_domain=True)
        F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def step_B(anchor: ABIModel, wiki_ids: torch.Tensor, seed_offset: int) -> ABIModel:
    d_abi = anchor.d_abi
    torch.manual_seed(SEED + seed_offset + 1000)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + seed_offset + 1000)
    model = ABIModel(d_abi).to(DEVICE)
    model.load_state_dict(copy.deepcopy(anchor.state_dict()))
    for p in model.backbone.parameters(): p.requires_grad_(True)
    for p in model.lm_head.parameters():  p.requires_grad_(True)
    for p in model.proj_in.parameters():  p.requires_grad_(False)
    for p in model.proj_out.parameters(): p.requires_grad_(False)
    for p in model.abi_ln.parameters():   p.requires_grad_(False)
    for p in model.domain.parameters():   p.requires_grad_(False)
    model.domain_alpha.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR_ABI * 0.1667, weight_decay=0.01)  # LR_BACKBONE
    model.train()
    for step in range(UPDATE_STEPS):
        x, y = make_batch(wiki_ids, seed=1000 + step + seed_offset * 10000)
        opt.zero_grad()
        logits = model(x, use_domain=False)
        F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def step_C(drifted: ABIModel, py_ids: torch.Tensor, seed_offset: int) -> ABIModel:
    d_abi = drifted.d_abi
    torch.manual_seed(SEED + seed_offset + 2000)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + seed_offset + 2000)
    native = ABIModel(d_abi).to(DEVICE)
    native.load_state_dict(copy.deepcopy(drifted.state_dict()))
    for p in native.backbone.parameters(): p.requires_grad_(False)
    for p in native.lm_head.parameters():  p.requires_grad_(False)
    params = [native.proj_in.weight, native.proj_out.weight,
              *native.abi_ln.parameters(), *native.domain.parameters(), native.domain_alpha]
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids, seed=3000 + step + seed_offset * 10000)
        opt.zero_grad()
        logits = native(x, use_domain=True)
        F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    native.eval()
    for p in native.parameters(): p.requires_grad_(False)
    return native


def procrustes(drifted: ABIModel, native: ABIModel, py_ids: torch.Tensor) -> tuple:
    d_abi = drifted.d_abi
    drifted.eval(); native.eval()
    H_cal_list, H_nat_list = [], []
    with torch.no_grad():
        for i in range(N_COLLECT):
            x, _ = make_batch(py_ids, seed=2000 + i)
            _, h_abi_cal = drifted.encode_core(x)
            _, h_abi_nat = native.encode_core(x)
            h_full_cal = h_abi_cal + drifted.domain_alpha * drifted.domain(h_abi_cal)
            h_full_nat = h_abi_nat + native.domain_alpha  * native.domain(h_abi_nat)
            H_cal_list.append(h_full_cal.reshape(-1, d_abi).cpu().float())
            H_nat_list.append(h_full_nat.reshape(-1, d_abi).cpu().float())
    H_cal = torch.cat(H_cal_list); H_nat = torch.cat(H_nat_list)
    A_star, _, _, _ = torch.linalg.lstsq(H_cal, H_nat, rcond=None)
    H_nat_pred = H_cal @ A_star
    ss_res = float(((H_nat - H_nat_pred) ** 2).sum())
    ss_tot = float(((H_nat - H_nat.mean(0)) ** 2).sum())
    r_sq   = round(1.0 - ss_res / ss_tot, 5) if ss_tot > 0 else 0.0
    calibrated = ABIModel(d_abi).to(DEVICE)
    calibrated.load_state_dict(copy.deepcopy(drifted.state_dict()))
    calibrated.proj_out.weight.data.copy_(
        (native.proj_out.weight.cpu().float() @ A_star.T).to(DEVICE).to(calibrated.proj_out.weight.dtype))
    calibrated.domain_alpha.data.copy_(native.domain_alpha.data)
    calibrated.domain.ln.weight.data.copy_(native.domain.ln.weight.data)
    calibrated.domain.ln.bias.data.copy_(native.domain.ln.bias.data)
    for p in calibrated.parameters(): p.requires_grad_(False)
    calibrated.eval()
    return calibrated, r_sq


@torch.no_grad()
def nib_l2_eval(native: ABIModel, calibrated: ABIModel, domain_ids: torch.Tensor) -> dict:
    native.eval(); calibrated.eval()
    CHUNK = 512; SKIP = L2_SKIP; eps = 1e-12
    rng = np.random.default_rng(7777)
    js_list, top1_list, top5_list, ent_list = [], [], [], []
    max_start = max(len(domain_ids) - CHUNK, 1)
    for ci in range(N_L2_CHUNKS):
        start = int(rng.integers(0, max_start))
        chunk = domain_ids[start : start + CHUNK].unsqueeze(0).to(DEVICE)
        nat_logits = native(chunk, use_domain=True)[0, SKIP:, :]
        cal_logits = calibrated(chunk, use_domain=True)[0, SKIP:, :]
        nat_p = F.softmax(nat_logits, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_logits, dim=-1).cpu().float().numpy()
        m = 0.5 * (nat_p + cal_p)
        nat_safe = np.clip(nat_p, eps, 1.0); cal_safe = np.clip(cal_p, eps, 1.0)
        m_safe   = np.clip(m,     eps, 1.0)
        kl_nm = (nat_p * (np.log(nat_safe) - np.log(m_safe))).sum(1)
        kl_cm = (cal_p * (np.log(cal_safe) - np.log(m_safe))).sum(1)
        js_list.extend(np.clip(0.5 * (kl_nm + kl_cm), 0, None).tolist())
        top1_list.extend((nat_p.argmax(1) == cal_p.argmax(1)).tolist())
        nat5 = np.argpartition(nat_p, -5, axis=1)[:, -5:]
        cal5 = np.argpartition(cal_p, -5, axis=1)[:, -5:]
        T = nat_p.shape[0]
        for t in range(T):
            top5_list.append(len(set(nat5[t]) & set(cal5[t])) / 5.0)
        H_nat = -(nat_p * np.log(nat_p + eps)).sum(1)
        H_cal = -(cal_p * np.log(cal_p + eps)).sum(1)
        ent_list.extend(np.abs(H_nat - H_cal).tolist())
    mean_js   = float(np.mean(js_list))
    mean_top1 = float(np.mean(top1_list))
    mean_top5 = float(np.mean(top5_list))
    mean_ent  = float(np.mean(ent_list))
    nib_pass = (mean_js < NIB_JS and mean_top1 >= NIB_TOP1 and
                mean_top5 >= NIB_TOP5 and mean_ent < NIB_ENT_DIFF)
    return {
        "mean_js":       round(mean_js, 5),
        "mean_top1":     round(mean_top1, 5),
        "mean_top5":     round(mean_top5, 5),
        "mean_ent_diff": round(mean_ent, 5),
        "pass":          nib_pass,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_one(d_abi: int, seed_offset: int,
            py_ids: torch.Tensor, wiki_ids: torch.Tensor) -> dict:
    print(f"  [d_abi={d_abi} seed_offset={seed_offset}] A...", flush=True)
    t0 = time.time()
    anchor     = step_A(d_abi, py_ids, seed_offset)
    print(f"  [d_abi={d_abi} seed_offset={seed_offset}] B...", flush=True)
    drifted    = step_B(anchor, wiki_ids, seed_offset)
    del anchor; torch.cuda.empty_cache()
    print(f"  [d_abi={d_abi} seed_offset={seed_offset}] C...", flush=True)
    native     = step_C(drifted, py_ids, seed_offset)
    print(f"  [d_abi={d_abi} seed_offset={seed_offset}] Proc...", flush=True)
    calibrated, r_sq = procrustes(drifted, native, py_ids)
    del drifted; torch.cuda.empty_cache()
    l2 = nib_l2_eval(native, calibrated, py_ids)
    del native, calibrated; torch.cuda.empty_cache()
    elapsed = round(time.time() - t0, 1)
    print(f"  [d_abi={d_abi} seed_offset={seed_offset}] "
          f"top5={l2['mean_top5']:.4f} R2={r_sq:.4f} PASS={l2['pass']} ({elapsed:.0f}s)")
    return {"d_abi": d_abi, "seed_offset": seed_offset, "r_squared": r_sq,
            "l2": l2, "nib_pass": l2["pass"], "elapsed_s": elapsed}


def main():
    t_global = time.time()
    print("=" * 70)
    print("  Transition Zone Multi-Seed Validation  (Exp 15 supplement)")
    print("=" * 70)
    print(f"  d_abi sweep: {D_ABI_TRANSITION}")
    print(f"  Seeds (as offset): {SEEDS}")
    print(f"  NIB thresholds: JS<{NIB_JS}  top5>={NIB_TOP5}  ent<{NIB_ENT_DIFF}")
    print(f"  Device: {DEVICE}")

    # Load checkpoint
    completed = set()
    all_results = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
        all_results = ckpt
        for k, v in ckpt.items():
            if isinstance(v, dict):
                completed.add(k)
        print(f"  [resume] Loaded {len(completed)} completed: {sorted(completed)}")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    py_ids   = load_py_ids(tok)
    wiki_ids = load_wiki_ids(tok)
    print(f"  py_ids: {len(py_ids):,}  wiki_ids: {len(wiki_ids):,}")

    for d_abi in D_ABI_TRANSITION:
        for seed_offset in SEEDS:
            key = f"d{d_abi}_s{seed_offset}"
            if key in completed:
                print(f"  [skip] {key}")
                continue
            print(f"\n{'='*70}")
            print(f"  d_abi={d_abi}  seed_offset={seed_offset}")
            print(f"{'='*70}")
            try:
                result = run_one(d_abi, seed_offset, py_ids, wiki_ids)
                all_results[key] = result
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"  [ckpt] {key} saved")
            except Exception as exc:
                print(f"  [ERROR] {key}: {exc}")
                import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY  --  Transition Zone (Procrustes-only, 3 seeds each)")
    print(f"{'='*70}")
    print(f"  {'d_abi':<6}  {'seed':<6}  {'top5':<8}  {'R2':<8}  {'PASS'}")
    print(f"  {'-'*44}")
    for d_abi in D_ABI_TRANSITION:
        top5s, passes = [], []
        for seed_offset in SEEDS:
            key = f"d{d_abi}_s{seed_offset}"
            r = all_results.get(key)
            if r:
                t5 = r["l2"]["mean_top5"]
                p  = r["nib_pass"]
                print(f"  {d_abi:<6}  {seed_offset:<6}  {t5:<8.4f}  "
                      f"{r['r_squared']:<8.5f}  {'PASS' if p else 'FAIL'}")
                top5s.append(t5); passes.append(p)
        if len(top5s) == 3:
            mean5 = np.mean(top5s)
            ci95  = 1.96 * np.std(top5s, ddof=1) / math.sqrt(3)
            pr    = sum(passes) / 3
            print(f"  {'':6}  {'mean':6}  {mean5:<8.4f} +-{ci95:.4f}  pass_rate={pr:.3f}")
        print()

    output = {
        "d_abi_transition": D_ABI_TRANSITION,
        "seeds": SEEDS,
        "nib_thresholds": {"js": NIB_JS, "top1": NIB_TOP1,
                           "top5": NIB_TOP5, "entropy_diff": NIB_ENT_DIFF},
        "results": all_results,
        "total_elapsed_s": round(time.time() - t_global, 1),
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved -> {RESULT_FILE}")
    print(f"  Total runtime: {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    main()
