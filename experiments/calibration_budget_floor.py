#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Budget Floor
========================
For each domain, find the MINIMUM number of KD refinement steps required to
pass all NIB thresholds (JS < 0.10, top-1 >= 0.68, top-5 >= 0.86,
entropy_diff < 0.35), starting from a pure Procrustes solution.

Two results this produces:
  A. "Budget floor" per domain: minimum steps vs top-5 / JS curve
  B. Margin correlation: native_margin_5_6 correlates with required steps
     → Claim: calibration cost is governed by probability margin geometry

Protocol: identical A→B shared backbone as multi_domain_atlas.py, then
per-domain C + Procrustes + incremental KD checkpoint sweep.

KD budget schedule per domain:
  steps tested: [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, 9600]
  Stop at first PASS; report that as the floor.

Output: calibration_budget_floor_results.json
Runtime: ~60-90 min on RTX 3080
"""

import copy
import json
import math
import os
import pathlib
import sys
import time

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

# ── shared corpus + model code (import from atlas) ────────────────────────────
from multi_domain_atlas import (
    flush, banner,
    load_py_ids, load_wiki_ids, load_md_ids, load_sql_ids,
    atlas_forward, make_batch, model_ppl,
    train_native_oracle, procrustes_solve, l2_eval,
    VOCAB_SIZE, N_COLLECT, N_L2_CHUNKS, L2_SKIP,
    seed_domain_stage, DOMAIN_SEED_OFFSET,
)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── KD budget schedule ─────────────────────────────────────────────────────────
BUDGET_STEPS = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, 9600]

CHECKPOINT_FILE = pathlib.Path(__file__).parent / "calibration_budget_floor_checkpoint.json"

# Use the same KD hypers as atlas retry-3 (which passed all 4 domains)
KD_WEIGHT = 1.0
KD_TEMP   = 8.0
LOCAL_TOPK_KL_WEIGHT = 0.15
LOCAL_TOPK_KL_TEMP   = 4.0
LOCAL_TOPK_KL_K      = 10
SWA_EVERY  = 400


def _kd_refine_to_budget(calibrated_state: dict,
                          native: SVGPT2,
                          domain_ids: torch.Tensor,
                          dname: str,
                          target_steps: int,
                          current_model: SVGPT2 = None) -> SVGPT2:
    """
    Continue (or start) KD refinement from current_model (or a fresh
    calibrated copy) up to target_steps total.
    """
    if current_model is None:
        model = SVGPT2().to(DEVICE)
        model.load_state_dict({k: v.to(DEVICE) for k, v in calibrated_state.items()})
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
        step_offset = 0
    else:
        # Continuing from previous checkpoint (model is already in train mode)
        # Rebuild optimizer over the same params
        model = current_model
        params = [
            model.domain_alpha,
            model.domain.ln.weight,
            model.domain.ln.bias,
            model.proj_out.weight,
            model.proj_in.weight,
        ]
        for p in model.parameters(): p.requires_grad_(False)
        for p in params: p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=LR_CAL, weight_decay=0.01)
        step_offset = 0  # always train from scratch per budget point for reproducibility

    # Reset to clean calibrated state for each budget level
    model = SVGPT2().to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in calibrated_state.items()})
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

    seed_domain_stage(dname, 5000 + target_steps)
    model.train()
    swa_state = {}
    swa_count = 0

    for step in range(target_steps):
        x, y = make_batch(domain_ids, seed=7000 + step)
        opt.zero_grad()

        with torch.no_grad():
            nat_logits = atlas_forward(native, x, use_domain=True)
        cal_logits = atlas_forward(model, x, use_domain=True)

        kd_loss = F.kl_div(
            F.log_softmax(cal_logits / KD_TEMP, dim=-1),
            F.softmax(nat_logits / KD_TEMP, dim=-1),
            reduction="batchmean",
        ) * (KD_TEMP ** 2)

        loss = KD_WEIGHT * kd_loss

        # local_topk_kl
        with torch.no_grad():
            nat_topk_idx = nat_logits.topk(LOCAL_TOPK_KL_K, dim=-1).indices
            nat_local = nat_logits.gather(-1, nat_topk_idx)
            nat_local_target = F.softmax(
                nat_local.reshape(-1, LOCAL_TOPK_KL_K) / LOCAL_TOPK_KL_TEMP, dim=-1)
        cal_local = cal_logits.gather(-1, nat_topk_idx)
        local_kl = F.kl_div(
            F.log_softmax(cal_local.reshape(-1, LOCAL_TOPK_KL_K) / LOCAL_TOPK_KL_TEMP, dim=-1),
            nat_local_target,
            reduction="batchmean",
        ) * (LOCAL_TOPK_KL_TEMP ** 2)
        loss = loss + LOCAL_TOPK_KL_WEIGHT * local_kl

        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        # SWA
        done = step + 1
        if done >= 2400 and (done - 2400) % SWA_EVERY == 0:
            for name, param in [
                ("proj_out.weight", model.proj_out.weight),
                ("proj_in.weight", model.proj_in.weight),
                ("domain_alpha", model.domain_alpha),
            ]:
                t = param.detach().cpu().float()
                if name not in swa_state:
                    swa_state[name] = t.clone()
                else:
                    swa_state[name].mul_(swa_count / (swa_count + 1.0)).add_(t / (swa_count + 1.0))
            swa_count += 1

    model.eval()
    if swa_state:
        with torch.no_grad():
            for name, t in swa_state.items():
                parts = name.split(".")
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                getattr(obj, parts[-1]).data.copy_(t.to(DEVICE).to(getattr(obj, parts[-1]).dtype))

    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def measure_margin(native: SVGPT2, domain_ids: torch.Tensor,
                   n_chunks: int = 5, seed: int = 7777) -> dict:
    """Measure native_margin_5_6 distribution statistics."""
    native.eval()
    CHUNK = 512
    SKIP = L2_SKIP
    rng = np.random.default_rng(seed)
    margins = []
    max_start = max(len(domain_ids) - CHUNK, 1)
    for _ in range(n_chunks):
        start = int(rng.integers(0, max_start))
        chunk = domain_ids[start:start+CHUNK].unsqueeze(0).to(DEVICE)
        logits = atlas_forward(native, chunk, use_domain=True)[0, SKIP:, :]
        probs = F.softmax(logits, dim=-1).cpu().float().numpy()
        sorted10 = np.sort(np.partition(probs, -10, axis=1)[:, -10:], axis=1)[:, ::-1]
        m = sorted10[:, 4] - sorted10[:, 5]
        margins.extend(m.tolist())
    return {
        "mean": float(np.mean(margins)),
        "median": float(np.median(margins)),
        "p10": float(np.percentile(margins, 10)),
        "p25": float(np.percentile(margins, 25)),
    }


def main():
    t_global = time.time()

    banner("Calibration Budget Floor")
    flush("Measures minimum KD steps per domain to pass NIB.")
    flush(f"Budget schedule: {BUDGET_STEPS}")
    flush(f"Device: {DEVICE}")

    # ── Data loading ────────────────────────────────────────────────────────
    banner("Data Loading")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    py_ids   = load_py_ids(tok)
    wiki_ids = load_wiki_ids(tok)
    md_ids   = load_md_ids(tok)
    sql_ids  = load_sql_ids(tok)

    domain_ids_map = {
        "python":   py_ids,
        "wikitext": wiki_ids,
        "markdown": md_ids,
        "sql":      sql_ids,
    }

    REGISTRY_THRESHOLDS = {
        "js":      REGISTRY["js_threshold"],
        "top1":    REGISTRY["top1_threshold"],
        "top5":    REGISTRY["top5_threshold"],
        "entropy": REGISTRY["entropy_diff_threshold"],
    }
    flush(f"NIB thresholds: {REGISTRY_THRESHOLDS}")

    # ── Step A: Python anchor ────────────────────────────────────────────────
    banner("Step A  --  Anchor (Python, ABI-only)")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model_a = SVGPT2().to(DEVICE)
    params_a = [p for nm, p in model_a.named_parameters()
                if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain"))]
    for p in model_a.parameters(): p.requires_grad_(False)
    for p in params_a: p.requires_grad_(True)
    opt_a = torch.optim.AdamW(params_a, lr=LR_ABI, weight_decay=0.01)
    t0 = time.time()
    model_a.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch_sv(py_ids, SEED + step)
        opt_a.zero_grad()
        F.cross_entropy(model_a(x).reshape(-1, VOCAB_SIZE), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(params_a, 1.0)
        opt_a.step()
    model_a.eval()
    for p in model_a.parameters(): p.requires_grad_(False)
    flush(f"  [A] {time.time()-t0:.0f}s  ppl={model_ppl(model_a, py_ids):.2f}")

    # ── Step B: WikiText backbone drift ─────────────────────────────────────
    banner("Step B  --  Backbone drift (WikiText)")
    torch.manual_seed(SEED + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 1)
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
        x, y = make_batch_sv(wiki_ids, SEED + 10000 + step)
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
    flush(f"  [B] {time.time()-t0:.0f}s")
    del model_a; torch.cuda.empty_cache()
    transferred_state = {k: v.cpu().clone() for k, v in model_b.state_dict().items()}
    del model_b; torch.cuda.empty_cache()

    # ── Per-domain budget sweep ──────────────────────────────────────────────
    banner("Per-Domain Calibration Budget Sweep")
    # Resume from checkpoint if it exists
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as _f:
            results = json.load(_f)
        flush(f"  [resume] Loaded checkpoint with {len(results)} completed domains: {list(results.keys())}")
    else:
        results = {}

    for dname, domain_ids in domain_ids_map.items():
        if dname in results:
            flush(f"  [skip] {dname} already in checkpoint — skipping")
            continue
        banner(f"DOMAIN: {dname}")
        domain_results = {"domain": dname, "budget_curve": [], "margin_stats": {}}

        # Step C: native oracle
        native = train_native_oracle(transferred_state, domain_ids, dname)

        # Margin stats on native
        margin_stats = measure_margin(native, domain_ids)
        domain_results["margin_stats"] = margin_stats
        flush(f"  [margin_{dname}] mean={margin_stats['mean']:.5f}  "
              f"median={margin_stats['median']:.5f}  p10={margin_stats['p10']:.5f}")

        # Procrustes (0 steps)
        calibrated_0, A_star, r_sq, cond, ppl_cal = procrustes_solve(
            transferred_state, native, domain_ids, dname)
        calibrated_state_0 = {k: v.cpu().clone() for k, v in calibrated_0.state_dict().items()}
        del calibrated_0; torch.cuda.empty_cache()

        # Evaluate pure Procrustes (step=0)
        cal_0 = SVGPT2().to(DEVICE)
        cal_0.load_state_dict({k: v.to(DEVICE) for k, v in calibrated_state_0.items()})
        cal_0.domain_alpha.data.copy_(native.domain_alpha.data)
        cal_0.domain.ln.weight.data.copy_(native.domain.ln.weight.data)
        cal_0.domain.ln.bias.data.copy_(native.domain.ln.bias.data)
        cal_0.eval()
        res_0 = l2_eval(native, cal_0, domain_ids)
        del cal_0; torch.cuda.empty_cache()
        flush(f"  [budget_{dname}] steps=0  JS={res_0['mean_js']:.5f}  "
              f"top5={res_0['top5']:.4f}  pass={res_0['pass']}")
        domain_results["budget_curve"].append({
            "steps": 0,
            "js": res_0["mean_js"],
            "top1": res_0["top1"],
            "top5": res_0["top5"],
            "entropy_diff": res_0.get("entropy", 0.0),
            "pass": res_0["pass"],
            "r_squared": r_sq,
        })
        floor_steps = 0 if res_0["pass"] else None

        if floor_steps is None:
            for budget in BUDGET_STEPS[1:]:  # skip 0, already done
                flush(f"  [budget_{dname}] training {budget} KD steps...")
                model_kd = _kd_refine_to_budget(
                    calibrated_state_0, native, domain_ids, dname, budget)
                res = l2_eval(native, model_kd, domain_ids)
                del model_kd; torch.cuda.empty_cache()
                flush(f"  [budget_{dname}] steps={budget}  JS={res['mean_js']:.5f}  "
                      f"top5={res['top5']:.4f}  pass={res['pass']}")
                domain_results["budget_curve"].append({
                    "steps": budget,
                    "js": res["mean_js"],
                    "top1": res["top1"],
                    "top5": res["top5"],
                    "entropy_diff": res.get("entropy", 0.0),
                    "pass": res["pass"],
                    "r_squared": r_sq,
                })
                if res["pass"] and floor_steps is None:
                    floor_steps = budget
                    flush(f"  [budget_{dname}] *** FLOOR = {budget} steps ***")
                    # Continue to get the full curve even after passing
                if budget >= 6400 and floor_steps is None:
                    flush(f"  [budget_{dname}] WARNING: still failing at {budget} steps")

        domain_results["floor_steps"] = floor_steps
        domain_results["margin_mean"] = margin_stats["mean"]
        domain_results["r_squared"] = r_sq

        flush(f"\n  === {dname}: floor={floor_steps} steps, "
              f"margin_mean={margin_stats['mean']:.5f} ===\n")
        del native; torch.cuda.empty_cache()
        results[dname] = domain_results
        # Save checkpoint after every domain so a kill doesn't lose progress
        with open(CHECKPOINT_FILE, "w") as _f:
            json.dump(results, _f, indent=2)
        flush(f"  [checkpoint] saved {list(results.keys())}")

    # ── Margin correlation ──────────────────────────────────────────────────
    banner("Margin-Cost Correlation")
    domains_with_floor = [d for d, r in results.items() if r["floor_steps"] is not None]
    if len(domains_with_floor) >= 2:
        margins = [results[d]["margin_mean"] for d in domains_with_floor]
        floors  = [results[d]["floor_steps"] for d in domains_with_floor]
        log_floors = [math.log(max(s, 1)) for s in floors]
        log_margins = [math.log(max(m, 1e-8)) for m in margins]
        # Pearson r between log(margin) and log(floor)
        n = len(margins)
        mx, my = np.mean(log_margins), np.mean(log_floors)
        cov = sum((lm - mx) * (lf - my) for lm, lf in zip(log_margins, log_floors)) / n
        sx = math.sqrt(sum((lm - mx)**2 for lm in log_margins) / n)
        sy = math.sqrt(sum((lf - my)**2 for lf in log_floors) / n)
        pearson_r = cov / (sx * sy + 1e-12)
        flush(f"\n  Domain      margin_mean   floor_steps")
        for d in domains_with_floor:
            flush(f"  {d:<12}  {results[d]['margin_mean']:.5f}       {results[d]['floor_steps']}")
        flush(f"\n  Pearson r (log-margin vs log-floor): {pearson_r:.4f}")
        flush(f"  Interpretation: negative r -> higher margin = fewer steps needed")
        correlation = {
            "domains": domains_with_floor,
            "margin_means": margins,
            "floor_steps": floors,
            "pearson_r_log": round(pearson_r, 4),
        }
    else:
        correlation = {"note": "insufficient data for correlation"}
        pearson_r = float("nan")

    # ── Summary ─────────────────────────────────────────────────────────────
    banner("RESULTS SUMMARY")
    flush(f"\n  Domain        | margin_mean  | floor_steps | Procrustes R²")
    flush(f"  {'─'*60}")
    for dname, r in results.items():
        flush(f"  {dname:<13} | {r['margin_mean']:.5f}      | "
              f"{str(r['floor_steps']):<11} | {r['r_squared']:.5f}")
    flush(f"\n  Margin-cost Pearson r (log-log): {pearson_r:.4f}")
    flush(f"  Claim: calibration cost is governed by probability margin geometry")

    total_time = (time.time() - t_global) / 60
    flush(f"\n  Total runtime: {total_time:.1f} min")

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "domains": results,
        "budget_schedule": BUDGET_STEPS,
        "kd_config": {
            "kd_weight": KD_WEIGHT,
            "kd_temp": KD_TEMP,
            "local_topk_kl_weight": LOCAL_TOPK_KL_WEIGHT,
            "local_topk_kl_temp": LOCAL_TOPK_KL_TEMP,
            "local_topk_kl_k": LOCAL_TOPK_KL_K,
            "swa_every": SWA_EVERY,
            "swa_start": 2400,
        },
        "nib_thresholds": REGISTRY_THRESHOLDS,
        "correlation": correlation,
        "total_runtime_min": round(total_time, 1),
    }
    out_path = pathlib.Path(__file__).parent / "calibration_budget_floor_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    flush(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
