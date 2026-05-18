#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 33 — NIB Geometry Diagnostic
========================================
Answers the question: WHY does GPT-2-large top-5 saturate at 0.845 while
GPT-2-small/medium PASS? Is this a method problem or a geometric inevitability?

Approach:
  1. Load GPT-2-small (117M), GPT-2-medium (354M), GPT-2-large (774M)
  2. Run each on the same WikiText-2 validation sentences
  3. Measure per-token distribution geometry:
       - mean entropy H (lower = sharper model)
       - mean top-1 probability P(rank-1)
       - top-5 margin = P(rank-5) - P(rank-6) (smaller = more fragile top-5)
       - top-5 fragility: fraction of positions where rank-5 and rank-6 are
         within τ of each other (i.e., rank swap is easy)
  4. Self-perturbation upper bound: add Gaussian noise σ to logits,
     measure top-5 Jaccard agreement with unperturbed output.
     This gives the MAXIMUM top-5 score achievable by any calibration that
     introduces this level of distributional noise.
  5. Threshold analysis: what σ of noise would reduce top-5 from 1.0 to 0.86?
     Does GPT-2-large reach this σ with smaller perturbations than GPT-2-medium?

Result: nib_geometry_diagnostic_results.json
Runtime: ~5-10 min on GPU (no training, inference only)

This diagnostic tells us:
  - If large model has LOWER fragility threshold → the NIB top-5 threshold
    is intrinsically harder to achieve → document as scaling boundary
  - If large model has SAME fragility → the problem is in calibration method
    → justify Exp 34 (ranking-aware KD)
"""

import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.stdout.reconfigure(line_buffering=True)

ROOT   = pathlib.Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = ROOT / "nib_geometry_diagnostic_results.json"

# -- Config ------------------------------------------------------------------
SEED      = 42
CHUNK     = 512   # tokens per forward pass
SKIP      = 20    # skip first tokens (context warmup)
N_CHUNKS  = 8     # number of chunks per model
SIGMA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]  # logit noise σ
N_NOISE_REPS = 4  # repeats per σ value for noise averaging

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODELS = [
    ("gpt2",        "GPT-2-small",  117,  768,  12),
    ("gpt2-medium", "GPT-2-medium", 354, 1024,  24),
    ("gpt2-large",  "GPT-2-large",  774, 1280,  36),
]


# -- Data --------------------------------------------------------------------
def load_wikitext_val_tokens(tok):
    """Load WikiText-2 validation set tokens."""
    try:
        from wikitext_cache import load_wikitext_split
        val_data = load_wikitext_split("wikitext-2-raw-v1", "validation")
        text = "\n".join(r["text"] for r in val_data if r["text"].strip())
    except Exception:
        # Fallback: generate deterministic pseudo-text
        print("  [warn] wikitext_cache unavailable, using wiki train fallback")
        from wikitext_cache import load_wikitext_split
        val_data = load_wikitext_split("wikitext-2-raw-v1", "train")
        text = "\n".join(r["text"] for r in val_data if r["text"].strip())[:200_000]
    ids = tok(text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
    return ids[:100_000]  # 100K tokens is plenty


# -- Distribution metrics ----------------------------------------------------
@torch.no_grad()
def get_distribution_stats(model, tokens, rng_seed=42):
    """
    Run N_CHUNKS forward passes and collect per-token distribution statistics.
    Returns dicts of arrays: entropy, top1_prob, top5_margin, rank5_prob, rank6_prob
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)
    max_start = max(len(tokens) - CHUNK, 1)

    entropy_list     = []
    top1_prob_list   = []
    top5_margin_list = []   # P(rank5) - P(rank6)
    rank5_prob_list  = []
    rank6_prob_list  = []

    for _ in range(N_CHUNKS):
        start = int(rng.integers(0, max_start))
        chunk = tokens[start : start + CHUNK].unsqueeze(0).to(DEVICE)
        logits = model(chunk).logits[0, SKIP:, :]          # [T, V]
        probs  = F.softmax(logits, dim=-1).cpu().float()   # [T, V]

        # entropy
        eps = 1e-12
        H = -(probs * (probs + eps).log()).sum(dim=-1).numpy()  # [T]
        entropy_list.extend(H.tolist())

        # top-1 prob
        p_top1 = probs.max(dim=-1).values.numpy()
        top1_prob_list.extend(p_top1.tolist())

        # top-5/6 margin
        # topk returns sorted descending
        topk6 = probs.topk(6, dim=-1).values  # [T, 6]
        rank5_prob = topk6[:, 4].numpy()       # 5th ranked (0-indexed)
        rank6_prob = topk6[:, 5].numpy()       # 6th ranked
        margin     = (rank5_prob - rank6_prob)

        rank5_prob_list.extend(rank5_prob.tolist())
        rank6_prob_list.extend(rank6_prob.tolist())
        top5_margin_list.extend(margin.tolist())

    return {
        "entropy":     np.array(entropy_list),
        "top1_prob":   np.array(top1_prob_list),
        "top5_margin": np.array(top5_margin_list),
        "rank5_prob":  np.array(rank5_prob_list),
        "rank6_prob":  np.array(rank6_prob_list),
    }


# -- Self-perturbation upper bound ------------------------------------------
@torch.no_grad()
def perturbation_top5_curve(model, tokens, sigma_grid, rng_seed=99):
    """
    For each σ in sigma_grid, add N(0, σ²) noise to logits and measure
    the top-5 Jaccard agreement with the clean logits.

    Returns: {sigma: mean_top5_agree}
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)
    max_start = max(len(tokens) - CHUNK, 1)

    # Pre-collect clean logits for N_CHUNKS chunks
    clean_logits_list = []
    chunk_starts = []
    for _ in range(N_CHUNKS):
        start = int(rng.integers(0, max_start))
        chunk_starts.append(start)
        chunk = tokens[start : start + CHUNK].unsqueeze(0).to(DEVICE)
        clean_logits = model(chunk).logits[0, SKIP:, :].cpu()  # [T, V]
        clean_logits_list.append(clean_logits)

    results = {}
    for sigma in sigma_grid:
        top5_agrees = []
        for ci, clean_logits in enumerate(clean_logits_list):
            T, V = clean_logits.shape
            for _ in range(N_NOISE_REPS):
                noise = torch.randn(T, V) * sigma
                noisy_logits = clean_logits + noise
                # top-5 sets
                clean_top5 = clean_logits.topk(5, dim=-1).indices.numpy()  # [T, 5]
                noisy_top5 = noisy_logits.topk(5, dim=-1).indices.numpy()  # [T, 5]
                for t in range(T):
                    agree = len(set(clean_top5[t]) & set(noisy_top5[t])) / 5.0
                    top5_agrees.append(agree)
        results[sigma] = float(np.mean(top5_agrees))
        print(f"      σ={sigma:.3f}  top5_agree={results[sigma]:.4f}")

    return results


# -- Fragility threshold -----------------------------------------------------
def find_fragility_sigma(perturbation_curve, target_top5=0.86):
    """Find the σ at which top-5 agreement crosses the NIB threshold (0.86)."""
    sorted_items = sorted(perturbation_curve.items())
    for i in range(len(sorted_items) - 1):
        s0, v0 = sorted_items[i]
        s1, v1 = sorted_items[i + 1]
        if v0 >= target_top5 >= v1:
            # Linear interpolation
            frac = (v0 - target_top5) / (v0 - v1) if (v0 - v1) > 1e-10 else 0
            return s0 + frac * (s1 - s0)
    return None  # threshold not crossed in range


# -- Summary stats helper ----------------------------------------------------
def summarise(arr, name):
    return {
        f"{name}_mean":   float(np.mean(arr)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_p10":    float(np.percentile(arr, 10)),
        f"{name}_p25":    float(np.percentile(arr, 25)),
        f"{name}_p75":    float(np.percentile(arr, 75)),
        f"{name}_p90":    float(np.percentile(arr, 90)),
    }


# -- Main --------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 72)
    print("  Experiment 33 — NIB Geometry Diagnostic")
    print("=" * 72)
    print(f"  Device:  {DEVICE}")
    print(f"  CHUNK:   {CHUNK}   N_CHUNKS: {N_CHUNKS}   SKIP: {SKIP}")
    print(f"  σ grid:  {SIGMA_GRID}")
    print()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = sys.maxsize

    print("  [Data] Loading WikiText-2 validation tokens...")
    wiki_tokens = load_wikitext_val_tokens(tok)
    print(f"  [Data] {len(wiki_tokens):,} tokens loaded")
    print()

    all_results = {}

    for model_id, model_name, params_M, d_model, n_layers in MODELS:
        print("=" * 72)
        print(f"  Model: {model_name} ({params_M}M, d_model={d_model}, {n_layers} layers)")
        print("=" * 72)
        t_model = time.time()

        # Load model
        model = GPT2LMHeadModel.from_pretrained(model_id).to(DEVICE)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Loaded {n_params/1e6:.1f}M parameters in {time.time()-t_model:.1f}s")

        # Distribution stats
        print(f"\n  [1/2] Measuring distribution geometry ({N_CHUNKS} chunks × {CHUNK} tokens)...")
        t_dist = time.time()
        stats = get_distribution_stats(model, wiki_tokens)
        print(f"        Done in {time.time()-t_dist:.1f}s")
        print(f"        mean_entropy    = {stats['entropy'].mean():.4f}  nat")
        print(f"        mean_top1_prob  = {stats['top1_prob'].mean():.4f}")
        print(f"        mean_top5_margin= {stats['top5_margin'].mean():.5f}")
        print(f"        frac_tight_top5 = {(stats['top5_margin'] < 0.001).mean():.4f}  "
              f"(positions where rank-5/6 prob diff < 0.001)")

        # Self-perturbation curve
        print(f"\n  [2/2] Self-perturbation top-5 curve ({N_NOISE_REPS} reps per σ)...")
        t_pert = time.time()
        curve = perturbation_top5_curve(model, wiki_tokens, SIGMA_GRID)
        print(f"        Done in {time.time()-t_pert:.1f}s")

        # Fragility threshold
        frag_sigma = find_fragility_sigma(curve, target_top5=0.86)
        if frag_sigma is not None:
            print(f"\n  *** top-5 drops below NIB threshold (0.86) at σ ≈ {frag_sigma:.4f} ***")
        else:
            print(f"\n  *** top-5 remains above 0.86 for all tested σ values ***")

        # Collect results
        summary = {
            "model_id":     model_id,
            "model_name":   model_name,
            "params_M":     params_M,
            "d_model":      d_model,
            "n_layers":     n_layers,
            "fragility_sigma_at_0.86": round(frag_sigma, 5) if frag_sigma else None,
        }
        summary.update(summarise(stats["entropy"],     "entropy"))
        summary.update(summarise(stats["top1_prob"],   "top1_prob"))
        summary.update(summarise(stats["top5_margin"], "top5_margin"))
        summary["tight_top5_fraction"] = float((stats["top5_margin"] < 0.001).mean())
        summary["very_tight_top5_fraction"] = float((stats["top5_margin"] < 0.0001).mean())
        summary["perturbation_curve"] = {str(k): round(v, 5) for k, v in curve.items()}

        all_results[model_name] = summary
        print(f"\n  {model_name} done in {time.time()-t_model:.1f}s")
        print()

        # Free VRAM before loading next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Cross-model comparison ---------------------------------------------
    print("=" * 72)
    print("  Cross-Model Summary")
    print("=" * 72)
    header = f"  {'Model':<20} {'entropy':>10} {'top1':>8} {'top5_margin':>12} {'tight%':>8} {'frag_σ':>8}"
    print(header)
    print("  " + "-" * 70)
    for mname in ["GPT-2-small", "GPT-2-medium", "GPT-2-large"]:
        r = all_results[mname]
        fs = r["fragility_sigma_at_0.86"]
        fs_str = f"{fs:.4f}" if fs is not None else ">0.40"
        print(f"  {mname:<20} {r['entropy_mean']:>10.4f} {r['top1_prob_mean']:>8.4f} "
              f"{r['top5_margin_mean']:>12.5f} {r['tight_top5_fraction']*100:>7.1f}% {fs_str:>8}")

    # -- Interpretation -------------------------------------------------------
    print()
    print("  Key question: does GPT-2-large's fragility_sigma < GPT-2-medium's?")
    print()
    fs_small  = all_results["GPT-2-small"]["fragility_sigma_at_0.86"]
    fs_medium = all_results["GPT-2-medium"]["fragility_sigma_at_0.86"]
    fs_large  = all_results["GPT-2-large"]["fragility_sigma_at_0.86"]
    print(f"  GPT-2-small  fragility σ: {fs_small}")
    print(f"  GPT-2-medium fragility σ: {fs_medium}")
    print(f"  GPT-2-large  fragility σ: {fs_large}")
    print()
    if fs_large is not None and fs_medium is not None and fs_large < fs_medium:
        print("  CONCLUSION: GPT-2-large is MORE fragile (lower σ threshold).")
        print("  The top-5 rank-5/6 gap is smaller → any calibration error shuffles rank 5.")
        print("  This is a GEOMETRIC CONSTRAINT — method improvement (Exp 34 top-K KD)")
        print("  may still help by reducing calibration error, but the ceiling is lower.")
        interpretation = "geometric_constraint"
    elif fs_large is None or (fs_medium is not None and fs_large >= fs_medium):
        print("  CONCLUSION: GPT-2-large is NOT more fragile than GPT-2-medium.")
        print("  The failure is in the calibration METHOD, not model geometry.")
        print("  Top-K restricted KD (Exp 34) should be able to fix this.")
        interpretation = "method_problem"
    else:
        print("  CONCLUSION: Inconclusive — need more data points.")
        interpretation = "inconclusive"

    all_results["interpretation"] = interpretation
    all_results["config"] = {
        "seed": SEED, "chunk": CHUNK, "skip": SKIP,
        "n_chunks": N_CHUNKS, "sigma_grid": SIGMA_GRID,
        "n_noise_reps": N_NOISE_REPS,
    }

    elapsed = time.time() - t0
    all_results["elapsed_s"] = round(elapsed, 1)
    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    with open(OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {OUTPUT.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
