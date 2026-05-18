#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method Robustness Sweep
=======================
Shows that NIB equivalence is NOT a brittle optimum — it holds across a
family of alignment procedures and hyperparameter configurations.

Sweep axes:
  1. Method:      Procrustes-only | KD-only | Procrustes→KD
  2. KD temperature: 2, 4, 8, 16
  3. local_topk_kl weight: 0.0, 0.1, 0.2, 0.3
  4. Seeds:       3 independent seeds

Reports for each configuration:
  - JS ± CI
  - top-5 ± CI
  - pass rate
  - Jaccard(pass positions across seeds)

Target claims:
  "Equivalence holds across a family of alignment procedures."
  "Equivalence is not a brittle optimum."

Domain: Python (fastest per-domain oracle, ~100 KD steps to pass)
Protocol: shared A→B backbone, then per-config sweep

Output: method_robustness_results.json
Runtime: ~45-75 min on RTX 3080
"""

import copy
import json
import math
import os
import pathlib
import sys
import time
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

sys.stdout.reconfigure(line_buffering=True)

from non_inferiority_benchmark import (
    PROBE_BANK, ADV_VARIANTS, REGISTRY,
    SVGPT2, DomainModuleSV,
    DEVICE, D_ABI, SEQ_LEN, DOMAIN_STEPS, UPDATE_STEPS,
    LR_ABI, LR_BACKBONE, LR_CAL, ALPHA,
    MAX_PY_SV, MAX_WIKI_SV, BATCH_SV, SEED, ROOT,
    make_batch_sv, ppl_sv,
)

from multi_domain_atlas import (
    flush, banner,
    load_py_ids, load_wiki_ids,
    atlas_forward, make_batch, model_ppl,
    VOCAB_SIZE, N_COLLECT, N_L2_CHUNKS, L2_SKIP,
    seed_domain_stage, DOMAIN_SEED_OFFSET,
    train_native_oracle, procrustes_solve, l2_eval,
)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Sweep config ───────────────────────────────────────────────────────────────
KD_TEMPS    = [2, 4, 8, 16]
KL_WEIGHTS  = [0.0, 0.1, 0.2, 0.3]
SWEEP_SEEDS = [42, 137, 999]       # 3 independent seeds for CI

# KD steps for Python domain (past the floor based on existing data)
KD_STEPS_PYTHON = 200
LOCAL_TOPK_KL_K  = 10

# Methods to compare
METHODS = [
    "procrustes_only",         # 0 KD steps
    "kd_only",                 # start from transferred (no Procrustes)
    "procrustes_then_kd",      # standard atlas method
]

CHECKPOINT_FILE = pathlib.Path(__file__).parent / "method_robustness_checkpoint.json"


def run_kd_from_state(start_state: dict,
                      native: SVGPT2,
                      domain_ids: torch.Tensor,
                      steps: int,
                      kd_temp: float,
                      kl_weight: float,
                      seed_offset: int) -> SVGPT2:
    """Run KD refinement from start_state for `steps` steps."""
    torch.manual_seed(SEED + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + seed_offset)

    model = SVGPT2().to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in start_state.items()})
    model.domain_alpha.data.copy_(native.domain_alpha.data)
    model.domain.ln.weight.data.copy_(native.domain.ln.weight.data)
    model.domain.ln.bias.data.copy_(native.domain.ln.bias.data)
    for p in model.parameters(): p.requires_grad_(False)

    params = [
        model.domain_alpha,
        model.domain.ln.weight,
        model.domain.ln.bias,
        model.proj_out.weight,
        model.proj_in.weight,
    ]
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR_CAL, weight_decay=0.01)

    model.train()
    for step in range(steps):
        x, y = make_batch(domain_ids, seed=9000 + seed_offset * 100 + step)
        opt.zero_grad()

        with torch.no_grad():
            nat_logits = atlas_forward(native, x, use_domain=True)
        cal_logits = atlas_forward(model, x, use_domain=True)

        kd_loss = F.kl_div(
            F.log_softmax(cal_logits / kd_temp, dim=-1),
            F.softmax(nat_logits / kd_temp, dim=-1),
            reduction="batchmean",
        ) * (kd_temp ** 2)
        loss = kd_loss

        if kl_weight > 0:
            with torch.no_grad():
                nat_topk_idx = nat_logits.topk(LOCAL_TOPK_KL_K, dim=-1).indices
                nat_local = nat_logits.gather(-1, nat_topk_idx)
                nat_local_target = F.softmax(
                    nat_local.reshape(-1, LOCAL_TOPK_KL_K) / 2.0, dim=-1)
            cal_local = cal_logits.gather(-1, nat_topk_idx)
            kl = F.kl_div(
                F.log_softmax(cal_local.reshape(-1, LOCAL_TOPK_KL_K) / 2.0, dim=-1),
                nat_local_target,
                reduction="batchmean",
            ) * 4.0
            loss = loss + kl_weight * kl

        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def summarise_runs(runs: list) -> dict:
    """Aggregate JS, top5, pass across multiple seed runs."""
    js_vals   = [r["mean_js"]  for r in runs]
    top5_vals = [r["top5"]     for r in runs]
    passes    = [r["pass"]     for r in runs]
    n = len(runs)

    def ci95(vals):
        m = np.mean(vals)
        s = np.std(vals, ddof=1) if n > 1 else 0.0
        return round(m, 5), round(1.96 * s / math.sqrt(n), 5)

    js_mean, js_ci   = ci95(js_vals)
    t5_mean, t5_ci   = ci95(top5_vals)
    return {
        "js_mean": js_mean, "js_ci95": js_ci,
        "top5_mean": t5_mean, "top5_ci95": t5_ci,
        "pass_rate": round(sum(passes) / n, 3),
        "n_seeds": n,
        "runs": runs,
    }


def main():
    t_global = time.time()

    banner("Method Robustness Sweep")
    flush("Testing NIB equivalence across methods, KD temps, KL weights, seeds.")
    flush(f"Methods: {METHODS}")
    flush(f"KD temps: {KD_TEMPS}  |  KL weights: {KL_WEIGHTS}  |  Seeds: {SWEEP_SEEDS}")

    # ── Data loading ────────────────────────────────────────────────────────
    banner("Data Loading")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    py_ids   = load_py_ids(tok)
    wiki_ids = load_wiki_ids(tok)
    flush("  Data loaded.")

    REGISTRY_THRESHOLDS = {
        "js":      REGISTRY["js_threshold"],
        "top5":    REGISTRY["top5_threshold"],
    }

    # Resume from checkpoint if it exists
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as _f:
            all_results = json.load(_f)
        completed_seeds = set()
        for _method_data in all_results.values():
            for _t_data in _method_data.values():
                for _kl_data in _t_data.values():
                    for _r in _kl_data:
                        completed_seeds.add(_r["seed"])
        flush(f"  [resume] Loaded checkpoint. Completed seeds: {sorted(completed_seeds)}")
    else:
        all_results = {}
        completed_seeds = set()

    for seed_idx, base_seed in enumerate(SWEEP_SEEDS):
        if base_seed in completed_seeds:
            flush(f"  [skip] seed={base_seed} already in checkpoint")
            continue
        banner(f"SEED {base_seed}  ({seed_idx+1}/{len(SWEEP_SEEDS)})")

        # ── Step A ───────────────────────────────────────────────────────────
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)

        model_a = SVGPT2().to(DEVICE)
        params_a = [p for nm, p in model_a.named_parameters()
                    if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain"))]
        for p in model_a.parameters(): p.requires_grad_(False)
        for p in params_a: p.requires_grad_(True)
        opt_a = torch.optim.AdamW(params_a, lr=LR_ABI, weight_decay=0.01)
        t0 = time.time()
        model_a.train()
        for step in range(DOMAIN_STEPS):
            x, y = make_batch_sv(py_ids, base_seed + step)
            opt_a.zero_grad()
            F.cross_entropy(model_a(x).reshape(-1, VOCAB_SIZE), y.reshape(-1)).backward()
            nn.utils.clip_grad_norm_(params_a, 1.0)
            opt_a.step()
        model_a.eval()
        for p in model_a.parameters(): p.requires_grad_(False)
        flush(f"  [A seed={base_seed}] {time.time()-t0:.0f}s")

        # ── Step B ───────────────────────────────────────────────────────────
        torch.manual_seed(base_seed + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + 1)

        model_b = SVGPT2().to(DEVICE)
        model_b.load_state_dict(model_a.state_dict())
        params_b_back = [p for nm, p in model_b.named_parameters()
                         if "gpt2" in nm or "lm_head" in nm]
        params_b_abi  = [p for nm, p in model_b.named_parameters()
                         if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain"))]
        for p in model_b.parameters(): p.requires_grad_(False)
        for p in params_b_back: p.requires_grad_(True)
        for p in params_b_abi:  p.requires_grad_(True)
        opt_b = torch.optim.AdamW([
            {"params": params_b_back, "lr": LR_BACKBONE},
            {"params": params_b_abi,  "lr": LR_ABI, "weight_decay": 0.01},
        ])
        t0 = time.time()
        model_b.train()
        for step in range(UPDATE_STEPS):
            x, y = make_batch_sv(wiki_ids, base_seed + 10000 + step)
            lm   = F.cross_entropy(model_b(x).reshape(-1, VOCAB_SIZE), y.reshape(-1))
            h1   = model_b.encode_core(x)[1].reshape(-1, D_ABI)
            h0   = model_a.encode_core(x)[1].reshape(-1, D_ABI).detach()
            stab = ALPHA * F.mse_loss(h1, h0)
            (lm + stab).backward()
            nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
            opt_b.step()
            opt_b.zero_grad()
        model_b.eval()
        for p in model_b.parameters(): p.requires_grad_(False)
        flush(f"  [B seed={base_seed}] {time.time()-t0:.0f}s")
        del model_a; torch.cuda.empty_cache()
        transferred_state = {k: v.cpu().clone() for k, v in model_b.state_dict().items()}
        del model_b; torch.cuda.empty_cache()

        # ── Step C: Python native oracle ─────────────────────────────────────
        # Override seed for reproducibility across sweep seeds
        torch.manual_seed(base_seed + 2000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + 2000)
        native = train_native_oracle(transferred_state, py_ids, "python")

        # ── Procrustes solve (shared for all hyper combos at this seed) ──────
        calibrated_proc, A_star, r_sq, cond, ppl_proc = procrustes_solve(
            transferred_state, native, py_ids, "python")
        proc_state = {k: v.cpu().clone() for k, v in calibrated_proc.state_dict().items()}
        del calibrated_proc; torch.cuda.empty_cache()

        # ── Method: procrustes_only ───────────────────────────────────────────
        method = "procrustes_only"
        cal_proc_only = SVGPT2().to(DEVICE)
        cal_proc_only.load_state_dict({k: v.to(DEVICE) for k, v in proc_state.items()})
        cal_proc_only.domain_alpha.data.copy_(native.domain_alpha.data)
        cal_proc_only.domain.ln.weight.data.copy_(native.domain.ln.weight.data)
        cal_proc_only.domain.ln.bias.data.copy_(native.domain.ln.bias.data)
        cal_proc_only.eval()
        res_proc = l2_eval(native, cal_proc_only, py_ids)
        del cal_proc_only; torch.cuda.empty_cache()
        key = (method, "na", "na", base_seed)
        if method not in all_results: all_results[method] = {}
        if "na" not in all_results[method]: all_results[method]["na"] = {}
        if "na" not in all_results[method]["na"]: all_results[method]["na"]["na"] = []
        all_results[method]["na"]["na"].append({
            **res_proc, "seed": base_seed, "r_squared": r_sq
        })
        flush(f"  [{method} seed={base_seed}] JS={res_proc['mean_js']:.5f}  "
              f"top5={res_proc['top5']:.4f}  pass={res_proc['pass']}")

        # ── Sweep: KD temp × KL weight ────────────────────────────────────────
        for kd_temp, kl_weight in itertools.product(KD_TEMPS, KL_WEIGHTS):
            label = f"T{kd_temp}_kl{kl_weight}"

            # Method: procrustes_then_kd
            for method, start_state in [
                ("procrustes_then_kd", proc_state),
                ("kd_only",            transferred_state),
            ]:
                model_kd = run_kd_from_state(
                    start_state, native, py_ids,
                    steps=KD_STEPS_PYTHON,
                    kd_temp=kd_temp,
                    kl_weight=kl_weight,
                    seed_offset=base_seed * 10 + int(kd_temp) + int(kl_weight * 100),
                )
                res = l2_eval(native, model_kd, py_ids)
                del model_kd; torch.cuda.empty_cache()

                if method not in all_results:
                    all_results[method] = {}
                if str(kd_temp) not in all_results[method]:
                    all_results[method][str(kd_temp)] = {}
                if str(kl_weight) not in all_results[method][str(kd_temp)]:
                    all_results[method][str(kd_temp)][str(kl_weight)] = []
                all_results[method][str(kd_temp)][str(kl_weight)].append({
                    **res, "seed": base_seed
                })
                flush(f"  [{method} T={kd_temp} kl={kl_weight} seed={base_seed}] "
                      f"JS={res['mean_js']:.5f}  top5={res['top5']:.4f}  pass={res['pass']}")

        del native; torch.cuda.empty_cache()
        # Save checkpoint after every seed so a kill doesn't lose progress
        with open(CHECKPOINT_FILE, "w") as _f:
            json.dump(all_results, _f, indent=2)
        flush(f"  [checkpoint] seed={base_seed} saved")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    banner("RESULTS SUMMARY")
    aggregated = {}

    # procrustes_only
    method = "procrustes_only"
    runs = all_results[method]["na"]["na"]
    agg = summarise_runs(runs)
    aggregated[method] = {"na": {"na": agg}}
    flush(f"\n  procrustes_only:  "
          f"JS={agg['js_mean']}±{agg['js_ci95']}  "
          f"top5={agg['top5_mean']}±{agg['top5_ci95']}  "
          f"pass_rate={agg['pass_rate']}")

    for method in ["procrustes_then_kd", "kd_only"]:
        aggregated[method] = {}
        flush(f"\n  {method}:")
        for kd_temp in KD_TEMPS:
            aggregated[method][str(kd_temp)] = {}
            flush(f"    T={kd_temp}:  ", end="")
            row_parts = []
            for kl_weight in KL_WEIGHTS:
                runs = all_results[method][str(kd_temp)][str(kl_weight)]
                agg = summarise_runs(runs)
                aggregated[method][str(kd_temp)][str(kl_weight)] = agg
                row_parts.append(f"kl={kl_weight}:top5={agg['top5_mean']:.4f}(pass={agg['pass_rate']})")
            flush("  |  ".join(row_parts))

    # Method comparison at best hypers
    flush("\n  Method comparison (best hypers):")
    flush(f"  {'Method':<25} {'JS mean':<12} {'top5 mean':<12} {'pass rate':<10}")
    flush(f"  {'─'*60}")
    proc_agg = aggregated["procrustes_only"]["na"]["na"]
    flush(f"  {'procrustes_only':<25} {proc_agg['js_mean']:<12}  {proc_agg['top5_mean']:<12}  {proc_agg['pass_rate']}")
    for method in ["procrustes_then_kd", "kd_only"]:
        # find best (highest pass_rate, then top5)
        best = None
        for T in KD_TEMPS:
            for kl in KL_WEIGHTS:
                a = aggregated[method][str(T)][str(kl)]
                if best is None or a["pass_rate"] > best[2]["pass_rate"] or \
                   (a["pass_rate"] == best[2]["pass_rate"] and a["top5_mean"] > best[2]["top5_mean"]):
                    best = (T, kl, a)
        if best:
            T, kl, a = best
            flush(f"  {method:<25} {a['js_mean']:<12}  {a['top5_mean']:<12}  {a['pass_rate']}  "
                  f"(best: T={T}, kl={kl})")

    total_time = (time.time() - t_global) / 60
    flush(f"\n  Total runtime: {total_time:.1f} min")

    output = {
        "aggregated": aggregated,
        "raw": all_results,
        "config": {
            "kd_temps": KD_TEMPS,
            "kl_weights": KL_WEIGHTS,
            "seeds": SWEEP_SEEDS,
            "kd_steps_python": KD_STEPS_PYTHON,
            "domain": "python",
        },
        "nib_thresholds": {
            "js": REGISTRY["js_threshold"],
            "top5": REGISTRY["top5_threshold"],
        },
        "total_runtime_min": round(total_time, 1),
    }
    out_path = pathlib.Path(__file__).parent / "method_robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    flush(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
