#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precision Parity Analysis  — Per-Position JS CDF
=================================================
Moves beyond aggregate metrics (mean JS=0.006) to ask: at what fraction of
token positions are the calibrated and native models *statistically identical*?

Metrics (5,120 positions = 10 × 512-token chunks):
  - JS CDF at thresholds: 0.001, 0.005, 0.01, 0.02, 0.05, 0.10
  - Percentiles: p50, p90, p95, p99, p999, max
  - Top-1 exact agreement fraction
  - Perfect top-5 fraction (all 5 tokens identical)
  - Per-entropy-quintile breakdown (where does imprecision cluster?)

Compares three calibration methods on the same A→B→C backbone:
  (a)  No Step D      — baseline (expected: high JS, many failures)
  (b)  Procrustes D   — 0 SGD; expected to dominate
  (c)  SGD 800        — full standard baseline

The key question: what fraction of positions does Procrustes render truly
identical to native (JS < 0.001)?

Results: precision_parity_results.json
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
    make_batch_sv, ppl_sv,
)
from transformers import GPT2TokenizerFast
from wikitext_cache import load_wikitext_split

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

N_EVAL_CHUNKS    = 10      # 10 × 512 = 5,120 positions (2× NIB for precision)
CHUNK_SIZE       = 512
SKIP_POS         = 20
N_COLLECT        = 200     # Procrustes collection batches
JS_THRESHOLDS    = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


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
    return py_ids, wiki_ids


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING STEPS A → B → C
# ══════════════════════════════════════════════════════════════════════════════

def run_abc(py_ids, wiki_ids):
    t0 = time.time()
    print("  [A] anchor (500 steps Python)...")
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
    print(f"  [C] {time.time()-t2:.0f}s  ppl_nat={ppl_sv(native, py_ids):.2f}")
    return transferred_state, native


# ══════════════════════════════════════════════════════════════════════════════
# BUILD CALIBRATED MODELS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_procrustes(transferred_state, native, py_ids):
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


def build_sgd_800(transferred_state, native, py_ids):
    calibrated = SVGPT2().to(DEVICE)
    calibrated.load_state_dict(transferred_state)
    for p in calibrated.parameters(): p.requires_grad_(False)
    calibrated.proj_in.weight.requires_grad_(True)
    calibrated.proj_out.weight.requires_grad_(True)
    calibrated.domain_alpha.requires_grad_(True)
    calibrated.domain.ln.weight.requires_grad_(True)
    calibrated.domain.ln.bias.requires_grad_(True)
    params = [calibrated.proj_in.weight, calibrated.proj_out.weight,
              calibrated.domain_alpha,
              calibrated.domain.ln.weight, calibrated.domain.ln.bias]
    opt = torch.optim.AdamW(params, lr=LR_CAL, weight_decay=0.01)
    kd_w = REGISTRY["kd_weight"]; kd_t = REGISTRY["kd_temp"]; ce_w = 1-kd_w
    calibrated.train()
    for step in range(REGISTRY["calibration_steps"]):
        x, y = make_batch_sv(py_ids, seed=7000+step)
        opt.zero_grad()
        cal_lo = calibrated(x)
        with torch.no_grad(): nat_lo = native(x)
        V = cal_lo.shape[-1]
        kd = F.kl_div(F.log_softmax(cal_lo.reshape(-1,V)/kd_t, dim=-1),
                      F.softmax(nat_lo.reshape(-1,V)/kd_t, dim=-1),
                      reduction='batchmean') * (kd_t**2)
        ce = F.cross_entropy(cal_lo.reshape(-1,V), y.reshape(-1))
        (kd_w*kd + ce_w*ce).backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    calibrated.eval()
    for p in calibrated.parameters(): p.requires_grad_(False)
    return calibrated


def build_raw(transferred_state):
    """No calibration — transferred state as-is (post-C)."""
    m = SVGPT2().to(DEVICE)
    m.load_state_dict(transferred_state)
    for p in m.parameters(): p.requires_grad_(False)
    m.eval()
    return m


# ══════════════════════════════════════════════════════════════════════════════
# PRECISION PARITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def precision_metrics(native, calibrated, py_ids, label=""):
    """
    Compute per-position JS over N_EVAL_CHUNKS × CHUNK_SIZE positions.
    Returns full per-position JS array + derived statistics.
    """
    native.eval(); calibrated.eval()
    js_all, top1_all, top5_perfect_all, entropy_all = [], [], [], []
    eps = 1e-12
    rng = np.random.default_rng(8888)
    max_start = max(len(py_ids) - CHUNK_SIZE, 1)

    for ci in range(N_EVAL_CHUNKS):
        start = int(rng.integers(0, max_start))
        chunk = py_ids[start:start+CHUNK_SIZE].unsqueeze(0).to(DEVICE)

        nat_lo = native(chunk, use_domain=True)[0, SKIP_POS:, :]
        cal_lo = calibrated(chunk, use_domain=True)[0, SKIP_POS:, :]

        nat_p = F.softmax(nat_lo, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_lo, dim=-1).cpu().float().numpy()
        T = nat_p.shape[0]

        # JS divergence per position
        m      = 0.5*(nat_p + cal_p)
        nat_pc = np.clip(nat_p, eps, 1.0)
        cal_pc = np.clip(cal_p, eps, 1.0)
        m_c    = np.clip(m, eps, 1.0)
        kl_nm  = (nat_pc*np.log(nat_pc/m_c)).sum(1)
        kl_cm  = (cal_pc*np.log(cal_pc/m_c)).sum(1)
        js = np.clip(0.5*(kl_nm+kl_cm), 0, None)
        js_all.extend(js.tolist())

        # Top-1 agreement
        top1_all.extend((nat_p.argmax(1) == cal_p.argmax(1)).tolist())

        # Perfect top-5 (all 5 tokens identical)
        nat5 = np.argpartition(nat_p, -5, axis=1)[:,-5:]
        cal5 = np.argpartition(cal_p, -5, axis=1)[:,-5:]
        for t in range(T):
            top5_perfect_all.append(len(set(nat5[t]) & set(cal5[t])) == 5)

        # Entropy of native
        H_nat = -(nat_pc * np.log(nat_pc)).sum(1)
        entropy_all.extend(H_nat.tolist())

    js_arr     = np.array(js_all)
    ent_arr    = np.array(entropy_all)
    n          = len(js_arr)

    # CDF
    cdf = {f"frac_js_lt_{t}": round(float((js_arr < t).mean()), 5)
           for t in JS_THRESHOLDS}

    # Percentiles
    pctiles = {}
    for pct in [50, 90, 95, 99, 99.9]:
        pctiles[f"p{str(pct).replace('.','_')}"] = round(float(np.percentile(js_arr, pct)), 6)
    pctiles["max"] = round(float(js_arr.max()), 6)
    pctiles["mean"] = round(float(js_arr.mean()), 6)

    # Entropy-quintile JS breakdown
    quintile_edges = np.percentile(ent_arr, [0,20,40,60,80,100])
    entropy_bins = {}
    for qi in range(5):
        lo, hi = quintile_edges[qi], quintile_edges[qi+1]
        mask = (ent_arr >= lo) & (ent_arr <= hi)
        if mask.sum() > 0:
            bin_js = js_arr[mask]
            entropy_bins[f"Q{qi+1}"] = {
                "entropy_range": [round(lo,3), round(hi,3)],
                "n": int(mask.sum()),
                "mean_js": round(float(bin_js.mean()), 6),
                "frac_identical": round(float((bin_js < 0.001).mean()), 4),
                "p90_js": round(float(np.percentile(bin_js, 90)), 6),
            }

    result = {
        "label": label,
        "n_positions": n,
        "mean_js": round(float(js_arr.mean()), 6),
        "mean_top1_agree": round(float(np.mean(top1_all)), 5),
        "frac_perfect_top5": round(float(np.mean(top5_perfect_all)), 5),
        "js_cdf": cdf,
        "js_percentiles": pctiles,
        "entropy_quintile_js": entropy_bins,
    }

    # Summary print
    print(f"  [{label}]  mean_JS={result['mean_js']:.5f}"
          f"  top1={result['mean_top1_agree']:.4f}"
          f"  perf_top5={result['frac_perfect_top5']:.4f}")
    print(f"    JS CDF: "
          + "  ".join(f"<{t:.3f}={cdf[f'frac_js_lt_{t}']:.3f}" for t in JS_THRESHOLDS))
    print(f"    pctiles: p50={pctiles['p50']:.5f}"
          f"  p90={pctiles['p90']:.5f}"
          f"  p99={pctiles['p99']:.5f}"
          f"  max={pctiles['max']:.5f}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_global = time.time()
    print("=" * 72)
    print("  PRECISION PARITY ANALYSIS — Per-Position JS CDF")
    print(f"  {N_EVAL_CHUNKS} chunks × {CHUNK_SIZE} positions = "
          f"{N_EVAL_CHUNKS*CHUNK_SIZE:,} total positions")
    print(f"  device: {DEVICE}")
    print("=" * 72)

    py_ids, wiki_ids = load_data()

    print("\n  Running A→B→C...")
    transferred_state, native = run_abc(py_ids, wiki_ids)

    # Build calibrated models
    print("\n  [Build] Raw (no D)...")
    t = time.time()
    raw = build_raw(transferred_state)
    print(f"    done ({time.time()-t:.0f}s)")

    print("  [Build] Procrustes D...")
    t = time.time()
    proc = build_procrustes(transferred_state, native, py_ids)
    print(f"    done ({time.time()-t:.0f}s)")

    print("  [Build] SGD-800 D...")
    t = time.time()
    sgd800 = build_sgd_800(transferred_state, native, py_ids)
    print(f"    done ({time.time()-t:.0f}s)")

    # Evaluate precision parity
    print("\n" + "=" * 72)
    print("  PRECISION PARITY METRICS  (higher fraction = more identical)")
    print("=" * 72)
    results = {}

    for label, model in [("raw_no_D", raw), ("procrustes_D", proc), ("sgd_800", sgd800)]:
        results[label] = precision_metrics(native, model, py_ids, label=label)

    # Summary comparison table
    print("\n  Summary — frac positions with JS < threshold:")
    print(f"  {'Method':<18}", end="")
    for t in JS_THRESHOLDS:
        print(f"  JS<{t:.3f}", end="")
    print(f"  {'max_JS':>9}  {'p99_JS':>9}  top1  perf_top5")
    for label, res in results.items():
        print(f"  {label:<18}", end="")
        for t in JS_THRESHOLDS:
            print(f"  {res['js_cdf'][f'frac_js_lt_{t}']:8.4f}", end="")
        print(f"  {res['js_percentiles']['max']:9.5f}"
              f"  {res['js_percentiles']['p99']:9.5f}"
              f"  {res['mean_top1_agree']:.4f}"
              f"  {res['frac_perfect_top5']:.4f}")

    print(f"\n  Total runtime: {(time.time()-t_global)/60:.1f} min")

    out_path = ROOT / "precision_parity_results.json"
    out_path.write_text(json.dumps({
        "n_eval_chunks": N_EVAL_CHUNKS,
        "chunk_size": CHUNK_SIZE,
        "n_positions": N_EVAL_CHUNKS * (CHUNK_SIZE - SKIP_POS),
        "js_thresholds": JS_THRESHOLDS,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
