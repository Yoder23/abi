#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abi/evaluation.py
=================
NIB (Non-Inferiority Benchmark) evaluation.

The NIB protocol is the formal, immutable criterion for claiming that a
CandidateABI model has achieved universal knowledge transfer with respect
to the AnchorABI.

Protocol specification (Path 2C, immutable)
--------------------------------------------
  Given: anchor A, candidate C, token sequence T, RNG seed s, n=5 chunks.

  For each of n randomly drawn 64-token prediction windows:
    1. Compute logit vectors a_t = A(x)[t] and c_t = C(x)[t]  for t > SKIP=5
    2. Convert to probability distributions: p_a = softmax(a_t), p_c = softmax(c_t)
    3. Compute:
         JS(t)      = JensenShannon(p_a, p_c)     [0 .. 1]
         top1(t)    = 1 if argmax(p_a) == argmax(p_c) else 0
         top5(t)    = |Top5(p_a) ∩ Top5(p_c)| / 5
         ent_diff(t)= |H(p_a) - H(p_c)|

  Aggregate: mean over all (n × T) token positions.

  Pass criteria (all four must hold simultaneously):
    mean_JS        < 0.10
    mean_top1      ≥ 0.68
    mean_top5      ≥ 0.86     ← primary criterion for Path 2C
    mean_ent_diff  < 0.35

  Official evaluation: rng=7777, n=5 (single run — defines pass/fail).
  Extended evaluation:  rng ∈ {7777, 1111, 2222, 3333, 4444}, n=5 each
                        → 25 positions → SE ≈ std/√25 → 95% CI ≈ mean ± 2·SE.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .training import ENC_LEN, PRED_LEN, PAD_ID, SKIP, DEVICE

PASS_THRESHOLDS = {
    "mean_js":   ("lt", 0.10),
    "mean_top1": ("ge", 0.68),
    "mean_top5": ("ge", 0.86),
    "mean_ent":  ("lt", 0.35),
}


@torch.no_grad()
def nib_eval(anchor, candidate, tokens: torch.Tensor,
             rng_seed: int = 7777, n_chunks: int = 5,
             label: str = "") -> dict:
    """
    Run one NIB evaluation pass and return a result dict.

    Parameters
    ----------
    anchor      : AnchorABI    (frozen, eval mode)
    candidate   : CandidateABI (frozen, eval mode)
    tokens      : 1-D int64 tensor
    rng_seed    : RNG seed for chunk position sampling
    n_chunks    : number of 64-token windows to evaluate
    label       : short string printed alongside chunk progress

    Returns
    -------
    dict with keys: mean_js, mean_top1, mean_top5, mean_ent, pass,
                    rng_seed, n_chunks, raw_top5 (list of per-token scores)
    """
    anchor.eval()
    candidate.eval()
    rng       = np.random.default_rng(rng_seed)
    total     = ENC_LEN + PRED_LEN
    max_start = max(len(tokens) - total, 1)
    js_l, t1_l, t5_l, ent_l = [], [], [], []

    for ci in range(n_chunks):
        start   = int(rng.integers(0, max_start))
        enc_ids = tokens[start:start + ENC_LEN].unsqueeze(0).to(DEVICE)
        cont    = tokens[start + ENC_LEN:start + total]
        pad_col = torch.full((1, 1), PAD_ID, dtype=torch.long)
        dec_ids = torch.cat([pad_col, cont[:-1].unsqueeze(0)], dim=1).to(DEVICE)

        anc_l = anchor(enc_ids, dec_ids)[0, SKIP:, :]
        cal_l = candidate(enc_ids, dec_ids)[0, SKIP:, :]
        anc_p = F.softmax(anc_l, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_l, dim=-1).cpu().float().numpy()

        T, eps = anc_p.shape[0], 1e-12
        m    = 0.5 * (anc_p + cal_p)
        kla  = (np.clip(anc_p, eps, 1) *
                np.log(np.clip(anc_p, eps, 1) / np.clip(m, eps, 1))).sum(1)
        klc  = (np.clip(cal_p, eps, 1) *
                np.log(np.clip(cal_p, eps, 1) / np.clip(m, eps, 1))).sum(1)
        js_l.extend(np.clip(0.5 * (kla + klc), 0, None).tolist())
        t1_l.extend((anc_p.argmax(1) == cal_p.argmax(1)).tolist())

        n5  = np.argpartition(anc_p, -5, axis=1)[:, -5:]
        c5  = np.argpartition(cal_p, -5, axis=1)[:, -5:]
        chunk_t5 = [len(set(n5[t]) & set(c5[t])) / 5.0 for t in range(T)]
        t5_l.extend(chunk_t5)

        Ha  = -(np.clip(anc_p, eps, 1) * np.log(np.clip(anc_p, eps, 1))).sum(1)
        Hc  = -(np.clip(cal_p, eps, 1) * np.log(np.clip(cal_p, eps, 1))).sum(1)
        ent_l.extend(np.abs(Ha - Hc).tolist())

        pfx = f"[{label}:rng={rng_seed}] " if label else f"[rng={rng_seed}] "
        print(f"    {pfx}chunk {ci + 1}/{n_chunks}: "
              f"JS={float(np.mean(js_l[-T:])):.4f}  "
              f"top1={float(np.mean(t1_l[-T:])):.3f}  "
              f"top5={float(np.mean(chunk_t5)):.3f}  "
              f"ent={float(np.mean(ent_l[-T:])):.4f}", flush=True)

    mj, mt1, mt5, me = (float(np.mean(js_l)), float(np.mean(t1_l)),
                        float(np.mean(t5_l)), float(np.mean(ent_l)))
    passed = mj < 0.10 and mt1 >= 0.68 and mt5 >= 0.86 and me < 0.35
    return {
        "mean_js":   round(mj,  5),
        "mean_top1": round(mt1, 4),
        "mean_top5": round(mt5, 4),
        "mean_ent":  round(me,  4),
        "pass":      passed,
        "rng_seed":  rng_seed,
        "n_chunks":  n_chunks,
        "raw_top5":  [round(v, 4) for v in t5_l],
    }


def nib_eval_extended(anchor, candidate, tokens: torch.Tensor,
                      official_seed: int = 7777,
                      extended_seeds: tuple = (1111, 2222, 3333, 4444)) -> dict:
    """
    Full NIB evaluation: official pass (rng=7777) + extended robustness (4 seeds).

    Returns
    -------
    dict:
      nib_official   : official result (rng=7777) — defines pass/fail
      nib_extended   : per-seed results for extended seeds
      nib_combined   : aggregate over all 25 positions with 95% CI
    """
    # Official evaluation
    print(f"\n  NIB — Official evaluation (rng={official_seed}, n=5):")
    official = nib_eval(anchor, candidate, tokens,
                        rng_seed=official_seed, n_chunks=5, label="official")
    status = "PASS" if official["pass"] else "FAIL"
    print(f"  → {status}  JS={official['mean_js']}  top1={official['mean_top1']}  "
          f"top5={official['mean_top5']}  ent={official['mean_ent']}")

    # Extended evaluation
    print(f"\n  NIB — Extended evaluation ({len(extended_seeds)} additional seeds):")
    ext_results = {}
    all_top5    = list(official["raw_top5"])
    for seed in extended_seeds:
        r = nib_eval(anchor, candidate, tokens,
                     rng_seed=seed, n_chunks=5, label=f"ext")
        ext_results[str(seed)] = r
        all_top5.extend(r["raw_top5"])
        print(f"    rng={seed}: top5={r['mean_top5']:.4f}  "
              f"{'PASS' if r['pass'] else 'fail'}")

    # Aggregate statistics
    all_top5_arr = np.array(all_top5)
    n    = len(all_top5_arr)
    mean = float(np.mean(all_top5_arr))
    std  = float(np.std(all_top5_arr, ddof=1))
    se   = std / np.sqrt(n)
    ci_lo = round(mean - 1.96 * se, 4)
    ci_hi = round(mean + 1.96 * se, 4)

    per_seed_means = {str(official_seed): official["mean_top5"]}
    per_seed_means.update({str(s): ext_results[str(s)]["mean_top5"]
                           for s in extended_seeds})

    combined = {
        "n_positions":       n,
        "mean_top5":         round(mean, 4),
        "std_top5":          round(std, 4),
        "se_top5":           round(se, 4),
        "ci_95_low":         ci_lo,
        "ci_95_high":        ci_hi,
        "ci_margin_vs_0860": round(ci_lo - 0.860, 4),
        "per_rng_mean_top5": per_seed_means,
    }

    print(f"\n  NIB — Combined (n={n}):  "
          f"mean={mean:.4f}  std={std:.4f}  95% CI=[{ci_lo}, {ci_hi}]")
    return {
        "nib_official": official,
        "nib_extended": ext_results,
        "nib_combined": combined,
    }
