#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ranking Quality Analysis
========================
QUESTION: Why is top-5 overlap the binding constraint in calibration?

After running the full A→B→C→D protocol (d_abi=256, 800 KD steps), computes:

  1. Spearman ρ — rank correlation over the top-100 native tokens per position
     Captures how well relative ordering is preserved within the high-prob region.

  2. Kendall τ — pairwise concordance within top-20 native tokens
     Robust to outliers; directly measures fraction of correctly ordered pairs.

  3. Top-1 margin analysis — p_nat[rank=1] - p_nat[rank=2] distributions
     Hypothesis: positions with small margin (uncertain) fail more on top-5.

  4. Rank displacement — for each native top-5 token, what rank does calibrated
     assign?  Distribution of displacements reveals degree of misalignment.

  5. Entropy-binned failure rate — fail rate on top-5 vs H(p_nat) quintile.
     Hypothesis: high-entropy (uncertain) positions drive failure.

  6. Margin × failure cross-table — joint analysis of margin × entropy × top-5 fail.

Results: ranking_quality_results.json
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
CAL_STEPS    = 800
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

N_LOGIT_CHUNKS  = 5
CHUNK_SIZE      = 512
SKIP_POS        = 20
SPEARMAN_TOP_K  = 100   # tokens for Spearman ρ
KENDALL_TOP_K   = 20    # tokens for Kendall τ (O(K²) per position)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (identical to NIB / abi_scaling_law)
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
# TRAINING PROTOCOL  (same as NIB run 8)
# ══════════════════════════════════════════════════════════════════════════════

def run_protocol(py_ids, wiki_ids):
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

    # C
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
    print(f"  [C] {time.time()-t2:.0f}s  ppl_nat={ppl_nat:.2f}")

    # D
    t3 = time.time()
    print(f"  [D] KD calibration ({CAL_STEPS} steps)...")
    calibrated = ABI_GPT2().to(DEVICE)
    calibrated.load_state_dict(transferred_state)
    for p in calibrated.parameters(): p.requires_grad_(False)
    _params = [calibrated.proj_in.weight, calibrated.proj_out.weight,
               calibrated.domain_alpha, calibrated.domain.ln.weight, calibrated.domain.ln.bias]
    for p in _params: p.requires_grad_(True)
    opt_d = torch.optim.AdamW(_params, lr=LR_CAL, weight_decay=0.01)
    ce_w  = 1.0 - KD_WEIGHT
    native.eval(); calibrated.train()
    for step in range(CAL_STEPS):
        x, y = make_batch(py_ids, seed=7000+step)
        opt_d.zero_grad()
        cal_lo = calibrated(x)
        with torch.no_grad(): nat_lo = native(x)
        V  = cal_lo.shape[-1]
        kd = F.kl_div(F.log_softmax(cal_lo.reshape(-1,V)/KD_TEMP, dim=-1),
                      F.softmax(nat_lo.reshape(-1,V)/KD_TEMP, dim=-1), reduction='batchmean') * (KD_TEMP**2)
        ce = F.cross_entropy(cal_lo.reshape(-1,V), y.reshape(-1))
        (KD_WEIGHT*kd + ce_w*ce).backward()
        nn.utils.clip_grad_norm_(_params, 1.0); opt_d.step()
    calibrated.eval(); [p.requires_grad_(False) for p in calibrated.parameters()]
    ppl_cal = ppl(calibrated, py_ids)
    print(f"  [D] {time.time()-t3:.0f}s  ppl_cal={ppl_cal:.2f}  efficacy={ppl_cal/ppl_nat*100:.1f}%")
    print(f"  Total A→B→C→D: {(time.time()-t0)/60:.1f} min")
    return calibrated, native, ppl_cal, ppl_nat


# ══════════════════════════════════════════════════════════════════════════════
# RANKING METRICS
# ══════════════════════════════════════════════════════════════════════════════

def rankdata_desc(a):
    """Rank array descending (highest value = rank 1). Handles ties by average."""
    n = len(a)
    sorter = np.argsort(-a, kind='mergesort')
    inv    = np.empty(n, dtype=int); inv[sorter] = np.arange(n)
    a_s = (-a)[sorter]
    obs = np.r_[True, a_s[1:] != a_s[:-1], True]
    dense = obs[:-1].cumsum()[inv]
    count = np.where(obs)[0]
    return 0.5 * (count[dense-1] + count[dense] + 1)


def spearman_top_k(p_nat, p_cal, K=100):
    """Spearman ρ between native and calibrated over top-K native tokens."""
    top_idx = np.argpartition(p_nat, -K)[-K:]
    r_nat   = rankdata_desc(p_nat[top_idx])
    r_cal   = rankdata_desc(p_cal[top_idx])
    rn_c    = r_nat - r_nat.mean()
    rc_c    = r_cal - r_cal.mean()
    denom   = np.linalg.norm(rn_c) * np.linalg.norm(rc_c)
    return float(np.dot(rn_c, rc_c) / (denom + 1e-12))


def kendall_top_k(p_nat, p_cal, K=20):
    """Kendall τ_b between native and calibrated over top-K native tokens."""
    top_idx = np.argpartition(p_nat, -K)[-K:]
    r_nat   = rankdata_desc(p_nat[top_idx])
    r_cal   = rankdata_desc(p_cal[top_idx])
    con, dis, tied_n, tied_c, tied_both = 0, 0, 0, 0, 0
    for i in range(K):
        for j in range(i+1, K):
            sn = np.sign(r_nat[i] - r_nat[j])
            sc = np.sign(r_cal[i] - r_cal[j])
            if sn == 0 and sc == 0:   tied_both += 1
            elif sn == 0:             tied_n    += 1
            elif sc == 0:             tied_c    += 1
            elif sn == sc:            con       += 1
            else:                     dis       += 1
    n0 = K*(K-1)//2
    denom = np.sqrt((n0-tied_n)*(n0-tied_c)+1e-12)
    return float((con - dis) / denom)


def rank_displacement_of_topk(p_nat, p_cal, K=5):
    """For each of the native top-K tokens, return its rank in calibrated (1-indexed)."""
    top_nat_idx = np.argpartition(p_nat, -K)[-K:]
    # Build calibrated rank lookup via argsort (1 = highest prob)
    cal_rank_order = np.argsort(-p_cal, kind='stable')
    cal_rank_lookup = np.empty(len(p_cal), dtype=np.int32)
    cal_rank_lookup[cal_rank_order] = np.arange(1, len(p_cal)+1, dtype=np.int32)
    return cal_rank_lookup[top_nat_idx].tolist()   # list of K ranks


def top1_margin(p_nat):
    """p_nat[rank1] - p_nat[rank2]: how confident is the native model?"""
    sorted_p = np.sort(p_nat)[::-1]
    return float(sorted_p[0] - sorted_p[1])


@torch.no_grad()
def run_ranking_analysis(native, calibrated, py_ids):
    native.eval(); calibrated.eval()
    rng = np.random.default_rng(7777)
    max_start = max(len(py_ids) - CHUNK_SIZE, 1)

    # Per-position records
    spearman_vals  = []
    kendall_vals   = []
    margins_nat    = []
    displacements  = []     # all rank-displacement values
    top5_overlap   = []     # fraction in [0,1]
    entropy_nat    = []     # H(p_nat)
    top5_fail      = []     # bool: top5_overlap < 1.0
    margin_bins    = []     # 0-4 quintile index of margin

    eps = 1e-12

    for ci in range(N_LOGIT_CHUNKS):
        start  = int(rng.integers(0, max_start))
        chunk  = py_ids[start:start+CHUNK_SIZE].unsqueeze(0).to(DEVICE)
        nat_lo = native    (chunk)[0, SKIP_POS:, :]     # T, V
        cal_lo = calibrated(chunk)[0, SKIP_POS:, :]
        nat_p  = F.softmax(nat_lo, dim=-1).cpu().float().numpy()
        cal_p  = F.softmax(cal_lo, dim=-1).cpu().float().numpy()
        T      = nat_p.shape[0]

        # Pre-compute calibrated rank lookup for entire chunk (double argsort)
        # Shape T × V; rk_cal[t, v] = rank of token v at position t (1 = highest)
        # This is O(T × V log V) — acceptable for T=492, V=50257
        rk_cal_order = np.argsort(-cal_p, axis=1, kind='stable')
        rk_cal = np.empty_like(rk_cal_order)
        rk_cal[np.arange(T)[:, None], rk_cal_order] = np.arange(1, cal_p.shape[1]+1)

        for t in range(T):
            pn = nat_p[t]; pc = cal_p[t]

            # Spearman
            spearman_vals.append(spearman_top_k(pn, pc, K=SPEARMAN_TOP_K))

            # Kendall (expensive so only every 4th position)
            if t % 4 == 0:
                kendall_vals.append(kendall_top_k(pn, pc, K=KENDALL_TOP_K))

            # Top-5 overlap
            top5_nat = np.argpartition(pn, -5)[-5:]
            top5_cal = np.argpartition(pc, -5)[-5:]
            ov = len(set(top5_nat) & set(top5_cal)) / 5.
            top5_overlap.append(ov)
            top5_fail.append(float(ov < 1.0))

            # Rank displacement of native top-5 tokens in calibrated
            displacements.extend(rk_cal[t, top5_nat].tolist())

            # Entropy of native
            H = float(-(pn * np.log(pn + eps)).sum())
            entropy_nat.append(H)

            # Top-1 margin
            margins_nat.append(top1_margin(pn))

        print(f"  chunk {ci+1}/{N_LOGIT_CHUNKS}: "
              f"spearman={np.mean(spearman_vals[-T:]):.4f}  "
              f"top5_fail_rate={np.mean(top5_fail[-T:]):.3f}")

    # Entropy-binned failure rate
    ent_arr    = np.array(entropy_nat)
    fail_arr   = np.array(top5_fail)
    quintiles  = np.percentile(ent_arr, [0,20,40,60,80,100])
    ent_bins   = {}
    for i in range(5):
        lo, hi = quintiles[i], quintiles[i+1]
        mask   = (ent_arr >= lo) & (ent_arr <= hi)
        ent_bins[f"Q{i+1} H=[{lo:.2f},{hi:.2f}]"] = {
            "n":            int(mask.sum()),
            "fail_rate":    round(float(fail_arr[mask].mean()), 4),
            "mean_entropy": round(float(ent_arr[mask].mean()), 4),
        }

    # Margin-binned failure rate
    margin_arr = np.array(margins_nat)
    m_quintiles = np.percentile(margin_arr, [0,20,40,60,80,100])
    margin_bins_out = {}
    for i in range(5):
        lo, hi = m_quintiles[i], m_quintiles[i+1]
        mask   = (margin_arr >= lo) & (margin_arr <= hi)
        margin_bins_out[f"Q{i+1} margin=[{lo:.4f},{hi:.4f}]"] = {
            "n":            int(mask.sum()),
            "fail_rate":    round(float(fail_arr[mask].mean()), 4),
            "mean_margin":  round(float(margin_arr[mask].mean()), 4),
        }

    # Rank displacement distribution
    disp_arr = np.array(displacements)

    return {
        "n_positions":          len(spearman_vals),
        "spearman": {
            "mean": round(float(np.mean(spearman_vals)), 4),
            "std":  round(float(np.std(spearman_vals)),  4),
            "p10":  round(float(np.percentile(spearman_vals, 10)), 4),
            "p50":  round(float(np.percentile(spearman_vals, 50)), 4),
            "p90":  round(float(np.percentile(spearman_vals, 90)), 4),
        },
        "kendall": {
            "mean": round(float(np.mean(kendall_vals)), 4),
            "std":  round(float(np.std(kendall_vals)),  4),
            "p10":  round(float(np.percentile(kendall_vals, 10)), 4),
            "p50":  round(float(np.percentile(kendall_vals, 50)), 4),
        },
        "top5_overlap": {
            "mean":      round(float(np.mean(top5_overlap)), 4),
            "perfect_fraction": round(float(np.mean(np.array(top5_overlap)==1.0)), 4),
        },
        "rank_displacement": {
            "mean":          round(float(disp_arr.mean()), 2),
            "std":           round(float(disp_arr.std()),  2),
            "frac_in_top5":  round(float((disp_arr <= 5).mean()), 4),
            "frac_in_top10": round(float((disp_arr <= 10).mean()), 4),
            "frac_in_top50": round(float((disp_arr <= 50).mean()), 4),
            "p50_displacement": round(float(np.percentile(disp_arr, 50)), 1),
            "p90_displacement": round(float(np.percentile(disp_arr, 90)), 1),
            "p99_displacement": round(float(np.percentile(disp_arr, 99)), 1),
        },
        "entropy_binned_failure": ent_bins,
        "margin_binned_failure":  margin_bins_out,
        "global_top5_fail_rate":  round(float(np.mean(top5_fail)), 4),
        "interpreting_metric":    "spearman_rho measures rank order; "
                                  "displacement shows how far native tokens drift in calibrated; "
                                  "entropy/margin bins test whether uncertainty drives failure",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_global = time.time()
    print("=" * 70)
    print("  RANKING QUALITY ANALYSIS")
    print(f"  d_abi={D_ABI}, cal_steps={CAL_STEPS}, KD w={KD_WEIGHT} T={KD_TEMP}")
    print(f"  device: {DEVICE}")
    print("=" * 70)

    _tok, py_ids, wiki_ids = load_data()

    print("\n  Running A→B→C→D protocol...")
    calibrated, native, ppl_cal, ppl_nat = run_protocol(py_ids, wiki_ids)

    print(f"\n  ppl_nat={ppl_nat:.3f}  ppl_cal={ppl_cal:.3f}  "
          f"efficacy={ppl_cal/ppl_nat*100:.1f}%")

    print("\n  Running ranking analysis (this may take a few minutes)...")
    t_rank = time.time()
    ranking = run_ranking_analysis(native, calibrated, py_ids)
    print(f"  Done in {time.time()-t_rank:.0f}s")

    # Print summary
    print(f"\n{'='*70}")
    print("  RANKING QUALITY SUMMARY")
    print(f"{'='*70}")

    sp = ranking["spearman"]
    kd = ranking["kendall"]
    rd = ranking["rank_displacement"]
    print(f"  Spearman ρ (top-{SPEARMAN_TOP_K} native tokens):")
    print(f"    mean={sp['mean']:.4f}  std={sp['std']:.4f}  "
          f"p10={sp['p10']:.4f}  p50={sp['p50']:.4f}  p90={sp['p90']:.4f}")

    print(f"  Kendall τ (top-{KENDALL_TOP_K} native tokens):")
    print(f"    mean={kd['mean']:.4f}  std={kd['std']:.4f}  "
          f"p10={kd['p10']:.4f}  p50={kd['p50']:.4f}")

    print(f"  Rank displacement of native top-5 tokens in calibrated:")
    print(f"    mean rank = {rd['mean']:.1f}  (std={rd['std']:.1f})")
    print(f"    frac in top-5  = {rd['frac_in_top5']:.3f}")
    print(f"    frac in top-10 = {rd['frac_in_top10']:.3f}")
    print(f"    frac in top-50 = {rd['frac_in_top50']:.3f}")
    print(f"    p50 disp = {rd['p50_displacement']:.1f}  p90 disp = {rd['p90_displacement']:.1f}  "
          f"p99 disp = {rd['p99_displacement']:.1f}")

    print(f"\n  Entropy-binned top-5 fail rate:")
    for k, v in ranking["entropy_binned_failure"].items():
        print(f"    {k}: n={v['n']:5d}  fail={v['fail_rate']:.3f}  H={v['mean_entropy']:.3f}")

    print(f"\n  Margin-binned top-5 fail rate (p_nat[1] - p_nat[2]):")
    for k, v in ranking["margin_binned_failure"].items():
        print(f"    {k}: n={v['n']:5d}  fail={v['fail_rate']:.3f}  margin={v['mean_margin']:.4f}")

    print(f"\n  Global top-5 fail rate: {ranking['global_top5_fail_rate']:.4f}")
    print(f"  Perfect top-5 positions: {ranking['top5_overlap']['perfect_fraction']:.3f}")

    results = {
        "config": {
            "d_abi": D_ABI, "cal_steps": CAL_STEPS,
            "kd_weight": KD_WEIGHT, "kd_temp": KD_TEMP,
        },
        "ppl_nat":   round(ppl_nat, 3),
        "ppl_cal":   round(ppl_cal, 3),
        "efficacy":  round(ppl_cal/ppl_nat*100, 2),
        "ranking":   ranking,
    }
    out = ROOT / "ranking_quality_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out}")
    print(f"  Total runtime: {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    main()
