#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABI Scaling Law Experiment
==========================
Sweeps d_abi ∈ {64, 128, 256, 512, 1024} and measures how L2 distributional
metrics (JS divergence, top-1/5 agreement, entropy diff) and PPL efficacy
converge as the ABI bottleneck capacity increases.

Protocol per config: A → B → C → D (same as NIB run 8, but only L2 + PPL).
Skips L4 probe battery to keep each config ~6 min.

Results: abi_scaling_results.json
"""

import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wikitext_cache import load_wikitext_split

ROOT   = pathlib.Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Sweep config ─────────────────────────────────────────────────────────────
D_ABI_VALUES   = [64, 128, 256, 512, 1024]

# Fixed training hyper-params (same as NIB run 8)
SEQ_LEN      = 128
DOMAIN_STEPS = 500      # Step A + Step C
UPDATE_STEPS = 1000     # Step B
CAL_STEPS    = 800      # Step D
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

# L2 evaluation config
N_LOGIT_CHUNKS = 5
CHUNK_SIZE     = 512
SKIP_POS       = 20     # skip first 20 positions (low-context)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL — parameterised by d_abi
# ══════════════════════════════════════════════════════════════════════════════

class DomainModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ln  = nn.LayerNorm(d)
    def forward(self, h):
        return self.ln(self.net(h))


class ABI_GPT2(nn.Module):
    """GPT-2-medium with ABI bottleneck of dimension d_abi."""
    def __init__(self, d_abi: int):
        super().__init__()
        g             = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        self.d_model  = g.config.n_embd    # 1024
        self.d_abi    = d_abi
        self.proj_in  = nn.Linear(self.d_model, d_abi, bias=False)
        self.abi_ln   = nn.LayerNorm(d_abi)
        self.proj_out = nn.Linear(d_abi, self.d_model, bias=False)
        self.domain   = DomainModule(d_abi)
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


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = 10**30

    # Python — scan local .py files in the workspace (same as NIB)
    py_parts, py_chars = [], 0
    for p in ROOT.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            py_parts.append(txt); py_chars += len(txt)
            if py_chars >= MAX_PY_SV * 4: break
        except Exception:
            continue
    py_raw  = "\n".join(py_parts)
    py_ids  = tok(py_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_PY_SV]

    # WikiText-2
    wiki_ds   = load_wikitext_split("wikitext-2-raw-v1", "train")
    wiki_raw  = "\n".join(r["text"] for r in wiki_ds if r["text"].strip())
    wiki_ids  = tok(wiki_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_WIKI_SV]

    print(f"  py_ids={len(py_ids):,}  wiki_ids={len(wiki_ids):,}")
    return tok, py_ids, wiki_ids


def make_batch(tokens, seed):
    rng = torch.Generator(); rng.manual_seed(seed)
    max_s = max(len(tokens) - SEQ_LEN - 1, 1)
    starts = torch.randint(0, max_s, (BATCH_SV,), generator=rng)
    x = torch.stack([tokens[s : s+SEQ_LEN]     for s in starts]).to(DEVICE)
    y = torch.stack([tokens[s+1 : s+SEQ_LEN+1] for s in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def ppl(model, tokens, use_domain=True, n_batches=50):
    model.eval()
    losses = []
    for i in range(n_batches):
        x, y = make_batch(tokens, seed=8000 + i)
        logits = model(x, use_domain=use_domain)
        losses.append(F.cross_entropy(logits.reshape(-1, 50257), y.reshape(-1)).item())
    return float(np.exp(np.mean(losses)))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PROTOCOL (A → B → C → D, same as NIB run 8)
# ══════════════════════════════════════════════════════════════════════════════

def run_protocol(d_abi: int, py_ids, wiki_ids):
    """Return (calibrated, native, cal_alpha, ppl_cal, ppl_nat)."""
    t_start = time.time()

    # ── Step A: anchor on Python (ABI only, backbone frozen) ─────────────────
    print(f"  [A] d_abi={d_abi} anchor ({DOMAIN_STEPS} steps)...")
    anchor = ABI_GPT2(d_abi).to(DEVICE)
    for p in anchor.parameters(): p.requires_grad_(False)
    for nm, p in anchor.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_a = torch.optim.AdamW([p for p in anchor.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    anchor.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids, seed=5000 + step)
        opt_a.zero_grad()
        F.cross_entropy(anchor(x, use_domain=True).reshape(-1, 50257),
                        y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(anchor.parameters(), 1.0)
        opt_a.step()
    anchor.eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    saved_dom = copy.deepcopy(anchor.domain.state_dict())
    ppl_a = ppl(anchor, py_ids, use_domain=True)
    print(f"  [A] {time.time()-t_start:.0f}s  ppl={ppl_a:.2f}")

    # ── Step B: backbone drift on WikiText ────────────────────────────────────
    t1 = time.time()
    print(f"  [B] backbone update ({UPDATE_STEPS} steps WikiText)...")
    transferred = copy.deepcopy(anchor).to(DEVICE)
    for p in transferred.parameters(): p.requires_grad_(False)
    for nm, p in transferred.named_parameters():
        if "backbone" in nm or "proj_in" in nm or "abi_ln" in nm:
            p.requires_grad_(True)
    transferred.proj_out.requires_grad_(False)
    opt_b = torch.optim.AdamW([p for p in transferred.parameters() if p.requires_grad],
                               lr=LR_BACKBONE, weight_decay=0.01)
    transferred.train()
    for step in range(UPDATE_STEPS):
        x, y = make_batch(wiki_ids, seed=9000 + step)
        opt_b.zero_grad()
        h, h_abi = transferred.encode_core(x)
        logits   = transferred.lm_head(transferred.proj_out(h_abi) + h)
        ll = F.cross_entropy(logits.reshape(-1, 50257), y.reshape(-1))
        with torch.no_grad():
            _, h_aa = anchor.encode_core(x)
        sl = F.mse_loss(h_abi, h_aa)
        (ll + ALPHA * sl).backward()
        nn.utils.clip_grad_norm_(transferred.parameters(), 1.0)
        opt_b.step()
    transferred.domain.load_state_dict(saved_dom)
    transferred.eval()
    for p in transferred.parameters(): p.requires_grad_(False)
    print(f"  [B] {time.time()-t1:.0f}s")

    # ── Step C: native oracle (trained first — KD teacher) ───────────────────
    t2 = time.time()
    print(f"  [C] native oracle ({DOMAIN_STEPS} steps)...")
    native = copy.deepcopy(transferred).to(DEVICE)
    nn.init.xavier_uniform_(native.proj_in.weight)
    nn.init.xavier_uniform_(native.proj_out.weight)
    nn.init.ones_(native.abi_ln.weight); nn.init.zeros_(native.abi_ln.bias)
    native.domain = DomainModule(d_abi).to(DEVICE)
    native.domain_alpha.data.fill_(1.0)
    for p in native.parameters(): p.requires_grad_(False)
    for nm, p in native.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_c = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids, seed=5000 + step)
        opt_c.zero_grad()
        F.cross_entropy(native(x, use_domain=True).reshape(-1, 50257),
                        y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt_c.step()
    native.eval()
    for p in native.parameters(): p.requires_grad_(False)
    ppl_nat = ppl(native, py_ids, use_domain=True)
    print(f"  [C] {time.time()-t2:.0f}s  native ppl={ppl_nat:.2f}")

    # ── Step D: KD calibration ────────────────────────────────────────────────
    t3 = time.time()
    print(f"  [D] KD calibration ({CAL_STEPS} steps, w={KD_WEIGHT}, T={KD_TEMP})...")
    calibrated = copy.deepcopy(transferred).to(DEVICE)
    for p in calibrated.parameters(): p.requires_grad_(False)
    calibrated.proj_in.weight.requires_grad_(True)
    calibrated.proj_out.weight.requires_grad_(True)
    calibrated.domain_alpha.requires_grad_(True)
    calibrated.domain.ln.weight.requires_grad_(True)
    calibrated.domain.ln.bias.requires_grad_(True)
    _params = [calibrated.proj_in.weight, calibrated.proj_out.weight,
               calibrated.domain_alpha,
               calibrated.domain.ln.weight, calibrated.domain.ln.bias]
    opt_d = torch.optim.AdamW(_params, lr=LR_CAL, weight_decay=0.01)
    ce_weight = 1.0 - KD_WEIGHT
    native.eval()
    calibrated.train()
    for step in range(CAL_STEPS):
        x, y       = make_batch(py_ids, seed=7000 + step)
        opt_d.zero_grad()
        cal_logits = calibrated(x, use_domain=True)
        with torch.no_grad():
            nat_logits = native(x, use_domain=True)
        V       = cal_logits.shape[-1]
        kd_loss = F.kl_div(
            F.log_softmax(cal_logits.reshape(-1, V) / KD_TEMP, dim=-1),
            F.softmax(nat_logits.reshape(-1, V)     / KD_TEMP, dim=-1),
            reduction='batchmean') * (KD_TEMP ** 2)
        ce_loss = F.cross_entropy(cal_logits.reshape(-1, V), y.reshape(-1))
        (KD_WEIGHT * kd_loss + ce_weight * ce_loss).backward()
        nn.utils.clip_grad_norm_(_params, 1.0)
        opt_d.step()
    calibrated.eval()
    for p in calibrated.parameters(): p.requires_grad_(False)
    cal_alpha     = float(calibrated.domain_alpha.item())
    ppl_cal_final = ppl(calibrated, py_ids, use_domain=True)
    efficacy      = ppl_cal_final / ppl_nat * 100
    print(f"  [D] {time.time()-t3:.0f}s  cal ppl={ppl_cal_final:.2f}  "
          f"alpha={cal_alpha:.4f}  efficacy={efficacy:.1f}%")
    print(f"  Total A→B→C→D: {(time.time()-t_start)/60:.1f} min")
    return calibrated, native, cal_alpha, ppl_cal_final, ppl_nat


# ══════════════════════════════════════════════════════════════════════════════
# L2 DISTRIBUTIONAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def l2_metrics(native, calibrated, py_ids):
    native.eval(); calibrated.eval()
    rng = np.random.default_rng(7777)
    js_list, top1_list, top5_list, ent_list = [], [], [], []
    max_start = max(len(py_ids) - CHUNK_SIZE, 1)

    for ci in range(N_LOGIT_CHUNKS):
        start  = int(rng.integers(0, max_start))
        chunk  = py_ids[start : start + CHUNK_SIZE].unsqueeze(0).to(DEVICE)
        nat_lo = native(chunk,     use_domain=True)[0, SKIP_POS:, :]
        cal_lo = calibrated(chunk, use_domain=True)[0, SKIP_POS:, :]
        nat_p  = F.softmax(nat_lo, dim=-1).cpu().float().numpy()
        cal_p  = F.softmax(cal_lo, dim=-1).cpu().float().numpy()
        T = nat_p.shape[0]

        # JS
        m      = 0.5 * (nat_p + cal_p)
        eps    = 1e-12
        nat_pc = np.clip(nat_p, eps, 1.0)
        cal_pc = np.clip(cal_p, eps, 1.0)
        m_c    = np.clip(m, eps, 1.0)
        kl_nm  = (nat_pc * np.log(nat_pc / m_c)).sum(1)
        kl_cm  = (cal_pc * np.log(cal_pc / m_c)).sum(1)
        js     = np.clip(0.5*(kl_nm+kl_cm), 0, None)
        js_list.extend(js.tolist())

        # top-1
        top1_list.extend((nat_p.argmax(1) == cal_p.argmax(1)).tolist())

        # top-5
        nat5 = np.argpartition(nat_p, -5, axis=1)[:, -5:]
        cal5 = np.argpartition(cal_p, -5, axis=1)[:, -5:]
        for t in range(T):
            top5_list.append(len(set(nat5[t]) & set(cal5[t])) / 5.)

        # entropy
        H_nat = -(nat_pc * np.log(nat_pc)).sum(1)
        H_cal = -(cal_pc * np.log(cal_pc)).sum(1)
        ent_list.extend(np.abs(H_nat - H_cal).tolist())

        print(f"    [L2] chunk {ci+1}/{N_LOGIT_CHUNKS}: "
              f"JS={float(np.mean(js)):.4f}  "
              f"top1={float(np.mean(nat_p.argmax(1)==cal_p.argmax(1))):.3f}  "
              f"top5={float(np.mean([len(set(nat5[t])&set(cal5[t]))/5. for t in range(T)])):.3f}")

    return {
        "mean_js":          round(float(np.mean(js_list)),   5),
        "mean_top1_agree":  round(float(np.mean(top1_list)), 4),
        "mean_top5_overlap":round(float(np.mean(top5_list)), 4),
        "mean_entropy_diff":round(float(np.mean(ent_list)),  4),
        "n_positions":      len(js_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_global = time.time()
    print("=" * 70)
    print("  ABI SCALING LAW EXPERIMENT")
    print(f"  d_abi sweep: {D_ABI_VALUES}")
    print(f"  device: {DEVICE}")
    print("=" * 70)

    tok, py_ids, wiki_ids = load_data()

    results = {}
    for d_abi in D_ABI_VALUES:
        print(f"\n{'='*70}")
        print(f"  d_abi = {d_abi}  (ABI bottleneck dim)")
        print(f"{'='*70}")
        t_cfg = time.time()

        calibrated, native, cal_alpha, ppl_cal, ppl_nat = run_protocol(
            d_abi, py_ids, wiki_ids)

        print(f"\n  [L2] Computing distributional metrics for d_abi={d_abi}...")
        l2 = l2_metrics(native, calibrated, py_ids)

        efficacy = ppl_cal / ppl_nat * 100
        results[d_abi] = {
            "d_abi":              d_abi,
            "ppl_cal":            round(ppl_cal, 3),
            "ppl_nat":            round(ppl_nat, 3),
            "ppl_efficacy_pct":   round(efficacy, 2),
            "domain_alpha":       round(cal_alpha, 4),
            "runtime_min":        round((time.time() - t_cfg) / 60, 1),
            "L2":                 l2,
        }
        print(f"  d_abi={d_abi}: JS={l2['mean_js']:.4f}  "
              f"top1={l2['mean_top1_agree']:.3f}  "
              f"top5={l2['mean_top5_overlap']:.3f}  "
              f"entropy={l2['mean_entropy_diff']:.3f}  "
              f"efficacy={efficacy:.1f}%")

        # Save incrementally
        out = ROOT / "abi_scaling_results.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  Saved → {out}")

    # Summary table
    print(f"\n{'='*70}")
    print("  ABI SCALING LAW SUMMARY")
    print(f"{'='*70}")
    print(f"{'d_abi':>6} {'JS':>8} {'top-1':>7} {'top-5':>7} "
          f"{'entropy':>9} {'efficacy':>10}")
    print("-" * 60)
    for d in D_ABI_VALUES:
        if d not in results:
            continue
        r = results[d]; l = r["L2"]
        print(f"{d:>6} {l['mean_js']:>8.4f} {l['mean_top1_agree']:>7.3f} "
              f"{l['mean_top5_overlap']:>7.3f} {l['mean_entropy_diff']:>9.3f} "
              f"{r['ppl_efficacy_pct']:>9.1f}%")

    print(f"\n  Total runtime: {(time.time()-t_global)/60:.1f} min")
    print(f"  Results saved → {ROOT / 'abi_scaling_results.json'}")


if __name__ == "__main__":
    main()
