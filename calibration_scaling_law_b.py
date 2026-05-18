#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 35 — Calibration Scaling Law
=========================================
Builds the predictive model:  log10(floor_steps + 1) = α + β·log10(d_model) + γ·margin

Existing data from prior experiments (GPT-2-medium, Exp 11):
  python:   floor=0,    margin_median=0.000819, d_model=1024
  wikitext: floor=0,    margin_median=0.002700, d_model=1024
  markdown: floor=0,    margin_median=0.000864, d_model=1024
  sql:      floor=8000, margin_median=0.000012, d_model=1024

This gives us only ONE model size (1024). We need additional data points at:
  GPT-2-small  (d_model=768)  — for each of Python and WikiText
  GPT-2-large  (d_model=1280) — for each of Python and WikiText

Protocol for each model × domain:
  1. Run A→B→C (anchor, backbone drift, native oracle) — same as prior exps
  2. Measure native oracle margin statistics (P(rank5) - P(rank6)) on domain eval set
  3. Run top-K restricted KD (from Exp 34) at budget schedule:
       [0, 50, 200, 800, 2000, 4000] steps
  4. Stop at first PASS; record floor_steps and margin_median

After collecting all data points (3 models × 2 domains = 6 points + SQL from Exp 11 = 7):
  Fit: log10(floor+1) = α + β·log10(d_model) + γ·log10(margin+ε)
  Evaluate: LOOCV MAE, R², decision rule

Note: SQL domain requires 8000 steps on GPT-2-medium, confirming its structural outlier
status — it is NOT included in the primary fit (documented separately).

Output: calibration_scaling_law_results.json
Exp: 35
Runtime: ~3-4 hours on RTX 3080 Laptop.
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
OUTPUT = ROOT / "calibration_scaling_law_b_results.json"

# ── Pre-registered NIB thresholds ─────────────────────────────────────────────
REGISTRY = {
    "js_threshold":           0.10,
    "top1_threshold":         0.68,
    "top5_threshold":         0.86,
    "entropy_diff_threshold": 0.35,
}

# ── Budget schedule to probe ──────────────────────────────────────────────────
BUDGET_SCHEDULE = [0, 50, 200, 800, 2000, 4000]

# ── KD params: standard full-vocab KD (Top-K KD proven to regress in Exp 34)
# Exp 35 used Top-K KD and d_abi=320 for large — both confounders fixed here.
KD_TOPK      = None   # unused — standard full-vocab KD
LAMBDA_RANK5 = None   # unused
KD_WEIGHT    = 0.90
KD_TEMP      = 2.0
LR_CAL       = 1e-4

# ── Model configs to sweep ────────────────────────────────────────────────────
MODELS_TO_TEST = [
    # (hf_id,        label,         d_model, d_abi,  n_layers, batch)
    # d_abi rule: 0.25 for <=12 layers; 0.50 for 36 layers (proven in Exp 36)
    ("gpt2",         "gpt2-small",  768,     192,    12,       8),
    ("gpt2-large",   "gpt2-large",  1280,    640,    36,       4),  # 640 = d_model//2 (Exp 36)
]
DOMAINS_TO_TEST = ["python", "wikitext"]

# ── Existing data from Exp 11 (GPT-2-medium, d_model=1024) ───────────────────
EXP11_DATA = {
    "python":   {"d_model": 1024, "floor_steps": 0,    "margin_median": 0.000819},
    "wikitext": {"d_model": 1024, "floor_steps": 0,    "margin_median": 0.002700},
    "markdown": {"d_model": 1024, "floor_steps": 0,    "margin_median": 0.000864},
    # SQL excluded from primary fit (structural outlier, confirmed in Exp 13)
}

SEQ_LEN      = 128
DOMAIN_STEPS = 500
UPDATE_STEPS = 1000
LR_ABI       = 3e-4
LR_BACKBONE  = 5e-5
ALPHA        = 1.0
SEED         = 42
VOCAB_SIZE   = 50257
MAX_PY       = 500_000
MAX_WIKI     = 600_000

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Model wrapper (generic GPT-2 family) ─────────────────────────────────────

class DomainModuleSV(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ln  = nn.LayerNorm(d)
    def forward(self, h):
        return self.ln(self.net(h))


class SVModelWrapper(nn.Module):
    """Generic GPT-2 family wrapper with ABI interface."""
    def __init__(self, hf_id, d_abi):
        super().__init__()
        g             = GPT2LMHeadModel.from_pretrained(hf_id)
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        self.d_model  = g.config.n_embd
        self.proj_in  = nn.Linear(self.d_model, d_abi, bias=False)
        self.abi_ln   = nn.LayerNorm(d_abi)
        self.proj_out = nn.Linear(d_abi, self.d_model, bias=False)
        self.domain   = DomainModuleSV(d_abi)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        self._d_abi   = d_abi
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

def make_batch(tokens, seed, batch_size, seq_len=SEQ_LEN):
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(tokens) - seq_len - 1, 1)
    starts = torch.randint(0, max_start, (batch_size,), generator=rng)
    x = torch.stack([tokens[s : s + seq_len]     for s in starts]).to(DEVICE)
    y = torch.stack([tokens[s+1 : s + seq_len+1] for s in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def ppl(model, tokens, batch_size, use_domain=True, n_batches=50, seed_offset=0):
    model.eval()
    tot, n = 0.0, 0
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    rng = torch.Generator()
    for i in range(n_batches):
        rng.manual_seed(80000 + seed_offset + i)
        starts = torch.randint(0, max_start, (batch_size,), generator=rng)
        x = torch.stack([tokens[s : s + SEQ_LEN]     for s in starts]).to(DEVICE)
        y = torch.stack([tokens[s+1 : s + SEQ_LEN+1] for s in starts]).to(DEVICE)
        logits = model(x, use_domain=use_domain)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1)).item()
        n   += 1
    return math.exp(tot / n)


@torch.no_grad()
def measure_margin(model, tokens, batch_size, n_batches=30):
    """Measure mean/median P(rank-5) - P(rank-6) for the model on this domain."""
    model.eval()
    margins = []
    rng = torch.Generator()
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    for i in range(n_batches):
        rng.manual_seed(40000 + i)
        starts = torch.randint(0, max_start, (batch_size,), generator=rng)
        x = torch.stack([tokens[s : s + SEQ_LEN] for s in starts]).to(DEVICE)
        logits = model(x, use_domain=True)
        probs  = F.softmax(logits, dim=-1).cpu().float()
        topk6  = probs.topk(6, dim=-1).values         # [B, T, 6]
        margin = (topk6[:, :, 4] - topk6[:, :, 5])    # [B, T]
        margins.extend(margin.reshape(-1).tolist())
    margins = np.array(margins)
    return float(np.mean(margins)), float(np.median(margins))


def topk_restricted_kd_loss(cal_logits_flat, nat_logits_flat, K, temp, lambda_rank5):
    """Top-K restricted KD loss (same as Exp 34)."""
    _, topk_idx = nat_logits_flat.topk(K, dim=-1)
    nat_topk = nat_logits_flat.gather(-1, topk_idx) / temp
    cal_topk = cal_logits_flat.gather(-1, topk_idx) / temp
    topk_kd = F.kl_div(
        F.log_softmax(cal_topk, dim=-1),
        F.softmax(nat_topk,     dim=-1),
        reduction="batchmean",
    ) * (temp ** 2)
    _, top5_idx = nat_logits_flat.topk(5, dim=-1)
    nat_top5 = nat_logits_flat.gather(-1, top5_idx) / temp
    cal_top5 = cal_logits_flat.gather(-1, top5_idx) / temp
    rank5_kd = F.kl_div(
        F.log_softmax(cal_top5, dim=-1),
        F.softmax(nat_top5,     dim=-1),
        reduction="batchmean",
    ) * (temp ** 2)
    return (1.0 - lambda_rank5) * topk_kd + lambda_rank5 * rank5_kd


@torch.no_grad()
def nib_eval(native, calibrated, domain_ids, n_chunks=5):
    """Run NIB evaluation and return result dict."""
    native.eval(); calibrated.eval()
    CHUNK = 512; SKIP = 20
    rng = np.random.default_rng(7777)
    js_list, top1_list, top5_list, ent_list = [], [], [], []
    max_start = max(len(domain_ids) - CHUNK, 1)
    for _ in range(n_chunks):
        start = int(rng.integers(0, max_start))
        chunk = domain_ids[start : start + CHUNK].unsqueeze(0).to(DEVICE)
        nat_l = native(chunk, use_domain=True)[0, SKIP:, :]
        cal_l = calibrated(chunk, use_domain=True)[0, SKIP:, :]
        nat_p = F.softmax(nat_l, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_l, dim=-1).cpu().float().numpy()
        T = nat_p.shape[0]; eps = 1e-12
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
    mj = float(np.mean(js_list)); mt1 = float(np.mean(top1_list))
    mt5 = float(np.mean(top5_list)); me = float(np.mean(ent_list))
    return {
        "mean_js": round(mj, 5), "mean_top1": round(mt1, 4),
        "mean_top5": round(mt5, 4), "mean_entropy_diff": round(me, 4),
        "pass": (mj < REGISTRY["js_threshold"] and
                 mt1 >= REGISTRY["top1_threshold"] and
                 mt5 >= REGISTRY["top5_threshold"] and
                 me < REGISTRY["entropy_diff_threshold"]),
    }


# ── A→B→C protocol (shared, produces native + transferred) ───────────────────

def run_abc(hf_id, d_abi, batch_size, py_ids, wiki_ids, domain_label):
    """Run Steps A, B, C for one model. Returns (native, transferred, anchor)."""
    domain_ids = py_ids if domain_label == "python" else wiki_ids
    wiki_for_b = wiki_ids if domain_label == "python" else py_ids

    # Step A: anchor
    print(f"    [A] Anchor ({DOMAIN_STEPS} steps on {domain_label})...")
    t0 = time.time()
    anchor = SVModelWrapper(hf_id, d_abi).to(DEVICE)
    for p in anchor.parameters():
        p.requires_grad_(False)
    for nm, p in anchor.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_a = torch.optim.AdamW([p for p in anchor.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    anchor.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(domain_ids, seed=5000 + step, batch_size=batch_size)
        opt_a.zero_grad()
        F.cross_entropy(anchor(x, True).reshape(-1, VOCAB_SIZE), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(anchor.parameters(), 1.0)
        opt_a.step()
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    saved_dom = copy.deepcopy(anchor.domain.state_dict())
    print(f"    [A] {time.time()-t0:.0f}s")

    # Step B: backbone drift
    print(f"    [B] Backbone drift ({UPDATE_STEPS} steps)...")
    t1 = time.time()
    transferred = copy.deepcopy(anchor).to(DEVICE)
    for p in transferred.parameters():
        p.requires_grad_(False)
    for nm, p in transferred.named_parameters():
        if "backbone" in nm or "proj_in" in nm or "abi_ln" in nm:
            p.requires_grad_(True)
    opt_b = torch.optim.AdamW([p for p in transferred.parameters() if p.requires_grad],
                               lr=LR_BACKBONE, weight_decay=0.01)
    transferred.train()
    for step in range(UPDATE_STEPS):
        x, y = make_batch(wiki_for_b, seed=9000 + step, batch_size=batch_size)
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
    print(f"    [B] {time.time()-t1:.0f}s")

    # Step C: native oracle
    print(f"    [C] Native oracle ({DOMAIN_STEPS} steps)...")
    t2 = time.time()
    native = copy.deepcopy(transferred).to(DEVICE)
    nn.init.xavier_uniform_(native.proj_in.weight)
    nn.init.xavier_uniform_(native.proj_out.weight)
    nn.init.ones_(native.abi_ln.weight); nn.init.zeros_(native.abi_ln.bias)
    native.domain = DomainModuleSV(d_abi).to(DEVICE)
    native.domain_alpha.data.fill_(1.0)
    for p in native.parameters():
        p.requires_grad_(False)
    for nm, p in native.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_c = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(domain_ids, seed=5000 + step, batch_size=batch_size)
        opt_c.zero_grad()
        F.cross_entropy(native(x, True).reshape(-1, VOCAB_SIZE), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt_c.step()
    native.eval()
    for p in native.parameters():
        p.requires_grad_(False)
    print(f"    [C] {time.time()-t2:.0f}s")

    return native, transferred


# ── Step D sweep: find floor steps ───────────────────────────────────────────

def find_floor_steps(native, transferred, d_abi, batch_size, domain_ids, budget_schedule):
    """
    Calibrate in increasing step increments; stop at first NIB PASS.
    Returns floor_steps and the full budget curve.
    """
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
        [calibrated.proj_in.weight, calibrated.proj_out.weight,
         calibrated.domain_alpha, calibrated.domain.ln.weight,
         calibrated.domain.ln.bias]
        + list(calibrated.domain.net.parameters())
    )
    opt_d = torch.optim.AdamW(cal_params, lr=LR_CAL, weight_decay=0.01)
    native.eval()

    budget_curve = []
    total_steps_done = 0
    floor_steps = None

    for budget in budget_schedule:
        # Train from current state to 'budget' total steps
        steps_needed = budget - total_steps_done
        if steps_needed > 0:
            calibrated.train()
            for step_idx in range(steps_needed):
                x, y = make_batch(domain_ids,
                                  seed=7000 + total_steps_done + step_idx,
                                  batch_size=batch_size)
                opt_d.zero_grad()
                cal_logits = calibrated(x, use_domain=True)
                with torch.no_grad():
                    nat_logits = native(x, use_domain=True)
                V = cal_logits.shape[-1]
                # Standard full-vocab KD (reverted from Top-K — Top-K regressed in Exp 34)
                nat_soft = F.softmax(nat_logits.reshape(-1, V).float() / KD_TEMP, dim=-1)
                cal_log  = F.log_softmax(cal_logits.reshape(-1, V).float() / KD_TEMP, dim=-1)
                kd_loss  = F.kl_div(cal_log, nat_soft, reduction="batchmean") * (KD_TEMP ** 2)
                ce_loss = F.cross_entropy(cal_logits.reshape(-1, V), y.reshape(-1))
                ((KD_WEIGHT * kd_loss) + ((1 - KD_WEIGHT) * ce_loss)).backward()
                nn.utils.clip_grad_norm_(cal_params, 1.0)
                opt_d.step()
            total_steps_done = budget
            calibrated.eval()
            for p in calibrated.parameters():
                p.requires_grad_(False)

        # Eval
        nib = nib_eval(native, calibrated, domain_ids)
        entry = {"steps": budget, **nib}
        budget_curve.append(entry)
        status = "PASS" if nib["pass"] else "FAIL"
        print(f"      steps={budget:>5d}: top5={nib['mean_top5']:.4f} "
              f"js={nib['mean_js']:.5f} [{status}]")

        if nib["pass"] and floor_steps is None:
            floor_steps = budget
            print(f"    *** FLOOR = {floor_steps} steps ***")
            break

        # Re-enable grad for continued training
        for p in cal_params:
            p.requires_grad_(True)
        calibrated.train()

    if floor_steps is None:
        floor_steps = budget_schedule[-1] + 1  # did not converge in schedule
        print(f"    *** No PASS in budget schedule; floor > {budget_schedule[-1]} ***")

    return floor_steps, budget_curve


# ── Statistical fitting ───────────────────────────────────────────────────────

def fit_scaling_law(data_points):
    """
    Fit: log10(floor+1) = α + β·log10(d_model) + γ·log10(margin_median+1e-9)

    data_points: list of {"d_model": int, "floor_steps": int, "margin_median": float, "label": str}
    """
    import numpy as np

    # Exclude points where floor is marked as not-converged (very high values)
    fit_pts = [p for p in data_points if p["floor_steps"] <= 4000]
    if len(fit_pts) < 3:
        return None

    Y = np.array([math.log10(p["floor_steps"] + 1) for p in fit_pts])
    X = np.column_stack([
        np.ones(len(fit_pts)),
        [math.log10(p["d_model"])                     for p in fit_pts],
        [math.log10(p["margin_median"] + 1e-9)        for p in fit_pts],
    ])

    result = np.linalg.lstsq(X, Y, rcond=None)
    coeffs = result[0]
    alpha, beta, gamma = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    Y_pred = X @ coeffs
    ss_res = float(np.sum((Y - Y_pred) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    # LOOCV
    loo_errors = []
    for i in range(len(fit_pts)):
        Xi = np.delete(X, i, axis=0)
        Yi = np.delete(Y, i)
        ci, _, _, _ = np.linalg.lstsq(Xi, Yi, rcond=None) if len(Xi) >= 3 else (coeffs, None, None, None)
        pred_log = float(X[i] @ ci)
        pred_steps = max(0, round(10 ** pred_log - 1))
        loo_errors.append(abs(pred_steps - fit_pts[i]["floor_steps"]))

    return {
        "alpha": round(alpha, 4), "beta": round(beta, 4), "gamma": round(gamma, 4),
        "r2": round(r2, 4),
        "loocv_mae_steps": round(float(np.mean(loo_errors)), 1),
        "n_points": len(fit_pts),
        "formula": f"log10(floor+1) = {alpha:.4f} + {beta:.4f}·log10(d_model) + {gamma:.4f}·log10(margin)",
    }


def predict_steps(fit, d_model, margin_median):
    """Predict floor steps given fit result."""
    log_val = (fit["alpha"]
               + fit["beta"]  * math.log10(d_model)
               + fit["gamma"] * math.log10(margin_median + 1e-9))
    return max(0, round(10 ** log_val - 1))


# ── Main ──────────────────────────────────────────────────────────────────────

def banner(msg):
    print(); print("=" * 72); print(f"  {msg}"); print("=" * 72)


def main():
    t_global = time.time()
    banner("Experiment 35b — Calibration Scaling Law (corrected: standard KD, d_abi=640 for large)")
    print(f"  Device:  {DEVICE}")
    print(f"  Models:  GPT-2-small (768), GPT-2-large (1280)")
    print(f"  Domains: python, wikitext")
    print(f"  Budget:  {BUDGET_SCHEDULE}")
    print(f"  KD:      Top-K (K={KD_TOPK}, λ={LAMBDA_RANK5}) — Exp 34 method")
    print()

    # Load corpora once
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = sys.maxsize

    print("  [Data] Loading Python and WikiText-2 corpora...")
    t_data = time.time()
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

    # Collect new data points
    new_data = {}

    for hf_id, label, d_model, d_abi, n_layers, batch_size in MODELS_TO_TEST:
        for domain_label in DOMAINS_TO_TEST:
            run_key = f"{label}/{domain_label}"
            banner(f"{run_key}  (d_model={d_model}, d_abi={d_abi}, batch={batch_size})")
            t_run = time.time()

            domain_ids = py_ids if domain_label == "python" else wiki_ids

            # A→B→C
            native, transferred = run_abc(hf_id, d_abi, batch_size, py_ids, wiki_ids, domain_label)

            # Measure native margin
            print(f"    [margin] Measuring native token margin...")
            m_mean, m_median = measure_margin(native, domain_ids, batch_size)
            print(f"    [margin] mean={m_mean:.6f}  median={m_median:.6f}")

            # Step D floor sweep
            print(f"    [D] Budget sweep {BUDGET_SCHEDULE}...")
            floor_steps, budget_curve = find_floor_steps(
                native, transferred, d_abi, batch_size, domain_ids, BUDGET_SCHEDULE)

            result = {
                "model":         label,
                "d_model":       d_model,
                "d_abi":         d_abi,
                "domain":        domain_label,
                "margin_mean":   round(m_mean,   6),
                "margin_median": round(m_median, 6),
                "floor_steps":   floor_steps,
                "budget_curve":  budget_curve,
                "runtime_s":     round(time.time() - t_run, 1),
            }
            new_data[run_key] = result
            print(f"  --> floor_steps={floor_steps}  margin_median={m_median:.6f}")

            # Free VRAM
            del native, transferred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Combine with Exp 11 data
    all_data_points = []

    # Existing GPT-2-medium data from Exp 11
    for domain, info in EXP11_DATA.items():
        all_data_points.append({
            "label":         f"gpt2-medium/{domain}",
            "d_model":       info["d_model"],
            "floor_steps":   info["floor_steps"],
            "margin_median": info["margin_median"],
            "source":        "exp11",
        })

    # New data from this experiment
    for run_key, data in new_data.items():
        all_data_points.append({
            "label":         run_key,
            "d_model":       data["d_model"],
            "floor_steps":   data["floor_steps"],
            "margin_median": data["margin_median"],
            "source":        "exp35",
        })

    # Fit the scaling law
    banner("Fitting Calibration Scaling Law")
    fit = fit_scaling_law(all_data_points)

    print(f"\n  All data points:")
    print(f"  {'Label':<30} {'d_model':>8} {'margin_med':>12} {'floor':>8}")
    print("  " + "-" * 62)
    for pt in all_data_points:
        print(f"  {pt['label']:<30} {pt['d_model']:>8} {pt['margin_median']:>12.6f} {pt['floor_steps']:>8}")

    if fit:
        print(f"\n  Fitted formula:")
        print(f"    {fit['formula']}")
        print(f"    R² = {fit['r2']:.4f}   LOOCV MAE = {fit['loocv_mae_steps']:.1f} steps")

        # Print predictions for each model size
        print(f"\n  Predictions for typical margin (median=0.001):")
        for d_model_test in [768, 1024, 1280, 2048, 4096]:
            pred = predict_steps(fit, d_model_test, 0.001)
            print(f"    d_model={d_model_test:>5}: predicted floor ≈ {pred} steps")

        print(f"\n  Decision rule:")
        print(f"    margin_median >= 0.01 → floor ≈ {predict_steps(fit, 1024, 0.01)} steps")
        print(f"    margin_median >= 0.001 → floor ≈ {predict_steps(fit, 1024, 0.001)} steps")
        print(f"    margin_median < 0.0001 → floor ≈ {predict_steps(fit, 1024, 0.0001)} steps")
    else:
        print(f"\n  Not enough data points to fit the scaling law.")
        fit = {"note": "insufficient_data"}

    # Save results
    elapsed = time.time() - t_global
    results = {
        "experiment":     "35b",
        "name":           "calibration_scaling_law_b",
        "note":           "corrected: standard KD (not Top-K), d_abi=640 for gpt2-large (not 320)",
        "new_data":       new_data,
        "all_data_points": all_data_points,
        "exp11_source":   EXP11_DATA,
        "scaling_law_fit": fit,
        "total_runtime_s": round(elapsed, 1),
        "config": {
            "budget_schedule":  BUDGET_SCHEDULE,
            "kd_topk":          KD_TOPK,
            "lambda_rank5":     LAMBDA_RANK5,
            "kd_weight":        KD_WEIGHT,
            "kd_temp":          KD_TEMP,
        },
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    banner(f"Exp 35 complete — {elapsed/60:.1f} min")
    print(f"  Results -> {OUTPUT.name}")


if __name__ == "__main__":
    main()
