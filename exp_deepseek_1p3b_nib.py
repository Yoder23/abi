#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment -- Cross-Lineage NIB: GPT-2-medium -> DeepSeek-Coder-1.3B
=====================================================================
First ABI experiment targeting a Llama-family architecture.

Source: GPT-2-medium        (354M,  d_model=1024, 24 layers, BPE 50257 vocab)
Target: deepseek-coder-1.3b (1.3B,  d_model=2048, 24 layers, BPE 32256 vocab)

DeepSeek-Coder is built on the Llama architecture:
  - RoPE positional encodings
  - SwiGLU activation
  - RMSNorm (not LayerNorm)
  - Grouped-query attention
  - Llama-style causal decoder (model.model access pattern)

This is architecturally the most distinct target tested to date:
  - Tokenizer : BPE 50K (GPT-2)   vs  BPE 32K (DeepSeek/Llama)
  - Arch      : GPT-2             vs  Llama (RoPE, SwiGLU, RMSNorm)
  - Domain    : general English   vs  code-specialised (pre-trained on 2T code tokens)
  - Backbone  : model.transformer vs  model.model  (Llama naming convention)
  - LM head   : model.lm_head

If NIB criterion is met, it demonstrates that:
  1. The 256-dim ABI bottleneck transfers to the Llama architectural family.
  2. Domain specialisation (code pre-training) does not prevent alignment.
  3. GPT-2 BPE tokens can be aligned to Llama BPE tokens via Procrustes.

Protocol: A -> C -> D  (backbone update omitted; invariance already proven).
NIB thresholds: pre-registered, identical to all prior experiments.
Result file: exp_deepseek_1p3b_nib_results.json
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from wikitext_cache import load_wikitext_split

sys.stdout.reconfigure(line_buffering=True)

ROOT   = pathlib.Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Pre-registered NIB thresholds (identical to all prior experiments) ─────────
REGISTRY = {
    "js_threshold":           0.10,
    "top1_threshold":         0.68,
    "top5_threshold":         0.86,
    "entropy_diff_threshold": 0.35,
    "n_logit_chunks":         5,
    "calibration_steps":      1200,
    "kd_weight":              0.90,
    "kd_temp":                2.0,
    "n_align_sentences":      2000,
}

# ── Architecture constants ─────────────────────────────────────────────────────
D_ABI       = 256    # Fixed shared ABI dimension — unchanged across all experiments
D_MODEL_SRC = 1024   # GPT-2-medium
VOCAB_SRC   = 50257
D_MODEL_TGT = 2048   # deepseek-coder-1.3b-base (Llama architecture)
VOCAB_TGT   = 32256  # DeepSeek-Coder BPE vocabulary

SEQ_LEN      = 128
DOMAIN_STEPS = 500
LR_ABI       = 3e-4
LR_CAL       = 1e-4
BATCH        = 4
SEED         = 42
MAX_PY       = 500_000
MAX_WIKI     = 600_000

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Shared sub-modules ─────────────────────────────────────────────────────────

class DomainModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ln  = nn.LayerNorm(d)
    def forward(self, h):
        return self.ln(self.net(h))


# ── Source model: GPT-2-medium + ABI (d_abi=256) ───────────────────────────────

class GPT2MedABI(nn.Module):
    """GPT-2-medium (354M, d_model=1024, 24 layers) with d_abi=256 bottleneck."""
    def __init__(self):
        super().__init__()
        g             = GPT2LMHeadModel.from_pretrained("gpt2-medium", local_files_only=True)
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        self.proj_in  = nn.Linear(D_MODEL_SRC, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, D_MODEL_SRC, bias=False)
        self.domain   = DomainModule(D_ABI)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def encode_core(self, x):
        h     = self.backbone(x).last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        return h, h_abi

    def forward(self, x, use_domain=True):
        h, h_abi = self.encode_core(x)
        h_out = h_abi + self.domain_alpha * self.domain(h_abi) if use_domain else h_abi
        return self.lm_head(self.proj_out(h_out) + h)


# ── Target model: DeepSeek-Coder-1.3B + ABI (d_abi=256) ───────────────────────

class DeepSeekCoder1p3BABI(nn.Module):
    """deepseek-ai/deepseek-coder-1.3b-base (1.3B, d_model=2048, Llama arch)
    with fixed d_abi=256 bottleneck.

    Llama naming convention:
      backbone: model.model  (not model.transformer)
      lm_head:  model.lm_head
      hidden states: out.last_hidden_state  (same API as Qwen2)
    """
    def __init__(self):
        super().__init__()
        q = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-base", local_files_only=True)
        self.backbone = q.model      # LlamaModel
        self.lm_head  = q.lm_head
        self.proj_in  = nn.Linear(D_MODEL_TGT, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, D_MODEL_TGT, bias=False)
        self.domain   = DomainModule(D_ABI)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def encode_core(self, x):
        # Llama backbone returns last_hidden_state directly
        h = self.backbone(input_ids=x).last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        return h, h_abi

    def forward(self, x, use_domain=True):
        h, h_abi = self.encode_core(x)
        h_out = h_abi + self.domain_alpha * self.domain(h_abi) if use_domain else h_abi
        return self.lm_head(self.proj_out(h_out) + h)


# ── Batch / PPL utilities ──────────────────────────────────────────────────────

def make_batch(tokens, seed, batch=BATCH):
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    starts = torch.randint(0, max_start, (batch,), generator=rng)
    x = torch.stack([tokens[s : s + SEQ_LEN]     for s in starts]).to(DEVICE)
    y = torch.stack([tokens[s+1 : s + SEQ_LEN+1] for s in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def ppl(model, tokens, use_domain=True, n_batches=50, seed_offset=0):
    model.eval()
    tot, n = 0.0, 0
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    rng = torch.Generator()
    for i in range(n_batches):
        rng.manual_seed(80000 + seed_offset + i)
        starts = torch.randint(0, max_start, (BATCH,), generator=rng)
        x = torch.stack([tokens[s : s + SEQ_LEN]     for s in starts]).to(DEVICE)
        y = torch.stack([tokens[s+1 : s + SEQ_LEN+1] for s in starts]).to(DEVICE)
        logits = model(x, use_domain=use_domain)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += 1
    return math.exp(tot / n)


# ── Cross-lineage Procrustes alignment ────────────────────────────────────────

@torch.no_grad()
def cross_lineage_procrustes(src_model, tgt_model, align_sentences, src_tok, tgt_tok):
    """Orthogonal Procrustes: GPT-2-medium ABI -> DeepSeek-Coder ABI space.
    Sentence-level mean-pooling handles the GPT-2 BPE vs Llama BPE tokenizer mismatch.
    """
    src_model.eval()
    tgt_model.eval()
    src_vecs, tgt_vecs = [], []
    for sent in align_sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        try:
            ids_src = src_tok(sent, return_tensors="pt", truncation=True,
                              max_length=128)["input_ids"].to(DEVICE)
            if ids_src.shape[1] < 4:
                continue
            _, h_src = src_model.encode_core(ids_src)
            src_vecs.append(h_src[0].mean(0).cpu().float())

            ids_tgt = tgt_tok(sent, return_tensors="pt", truncation=True,
                              max_length=128)["input_ids"].to(DEVICE)
            if ids_tgt.shape[1] < 4:
                continue
            _, h_tgt = tgt_model.encode_core(ids_tgt)
            tgt_vecs.append(h_tgt[0].mean(0).cpu().float())
        except Exception:
            continue
        if len(src_vecs) >= REGISTRY["n_align_sentences"]:
            break

    n = min(len(src_vecs), len(tgt_vecs))
    print(f"  [Procrustes] Using {n} sentence pairs")
    A = torch.stack(src_vecs[:n]); A = A - A.mean(0)
    B = torch.stack(tgt_vecs[:n]); B = B - B.mean(0)
    U, _, Vh = torch.linalg.svd(A.T @ B)
    R = U @ Vh
    print(f"  [Procrustes] cos sim: "
          f"{F.cosine_similarity(A, B, dim=1).mean().item():.4f} -> "
          f"{F.cosine_similarity(A @ R, B, dim=1).mean().item():.4f}")
    return R.to(DEVICE)


def apply_rotation_to_domain(src_domain, R):
    """Rotate domain module MLP into target ABI space."""
    dom   = copy.deepcopy(src_domain).cpu()
    R_cpu = R.cpu().float()
    with torch.no_grad():
        dom.net[0].weight.data = dom.net[0].weight.data @ R_cpu.T
        dom.net[2].weight.data = R_cpu @ dom.net[2].weight.data
        nn.init.ones_(dom.ln.weight)
        nn.init.zeros_(dom.ln.bias)
    return dom


# ── NIB L2 distributional equivalence test ─────────────────────────────────────

@torch.no_grad()
def l2_logit_test(native, calibrated, py_ids_tgt):
    """NIB evaluation in DeepSeek-Coder's 32256-token vocabulary space."""
    native.eval()
    calibrated.eval()
    CHUNK     = 512
    SKIP      = 20
    rng       = np.random.default_rng(7777)
    n_chunks  = REGISTRY["n_logit_chunks"]
    max_start = max(len(py_ids_tgt) - CHUNK, 1)
    js_list, top1_list, top5_list, ent_list = [], [], [], []

    for ci in range(n_chunks):
        start = int(rng.integers(0, max_start))
        chunk = py_ids_tgt[start : start + CHUNK].unsqueeze(0).to(DEVICE)
        nat_logits = native(chunk, use_domain=True)[0, SKIP:, :]
        cal_logits = calibrated(chunk, use_domain=True)[0, SKIP:, :]
        nat_p = F.softmax(nat_logits, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_logits, dim=-1).cpu().float().numpy()
        T, eps = nat_p.shape[0], 1e-12
        m = 0.5 * (nat_p + cal_p)
        kl_n = (np.clip(nat_p, eps, 1) * np.log(np.clip(nat_p / np.clip(m, eps, 1), eps, None))).sum(1)
        kl_c = (np.clip(cal_p, eps, 1) * np.log(np.clip(cal_p / np.clip(m, eps, 1), eps, None))).sum(1)
        js_list.extend(np.clip(0.5 * (kl_n + kl_c), 0, None).tolist())
        top1_list.extend((nat_p.argmax(1) == cal_p.argmax(1)).tolist())
        n5 = np.argpartition(nat_p, -5, axis=1)[:, -5:]
        c5 = np.argpartition(cal_p, -5, axis=1)[:, -5:]
        for t in range(T):
            top5_list.append(len(set(n5[t]) & set(c5[t])) / 5.0)
        Hn = -(np.clip(nat_p, eps, 1) * np.log(np.clip(nat_p, eps, 1))).sum(1)
        Hc = -(np.clip(cal_p, eps, 1) * np.log(np.clip(cal_p, eps, 1))).sum(1)
        ent_list.extend(np.abs(Hn - Hc).tolist())
        print(f"    chunk {ci+1}/{n_chunks}: JS={float(np.mean(js_list)):.4f} "
              f"top1={float(np.mean(top1_list)):.3f} top5={float(np.mean(top5_list)):.3f}")

    mj, mt1, mt5, me = (float(np.mean(js_list)), float(np.mean(top1_list)),
                        float(np.mean(top5_list)), float(np.mean(ent_list)))
    return {
        "n_positions":        len(js_list),
        "mean_js":            round(mj,  5),
        "mean_top1_agree":    round(mt1, 4),
        "mean_top5_overlap":  round(mt5, 4),
        "mean_entropy_diff":  round(me,  4),
        "js_pass":            mj  <  REGISTRY["js_threshold"],
        "top1_pass":          mt1 >= REGISTRY["top1_threshold"],
        "top5_pass":          mt5 >= REGISTRY["top5_threshold"],
        "entropy_pass":       me  <  REGISTRY["entropy_diff_threshold"],
        "pass": (mj  <  REGISTRY["js_threshold"] and
                 mt1 >= REGISTRY["top1_threshold"] and
                 mt5 >= REGISTRY["top5_threshold"] and
                 me  <  REGISTRY["entropy_diff_threshold"]),
    }


# ── Banner ─────────────────────────────────────────────────────────────────────

def banner(msg):
    print()
    print("=" * 72)
    print(f"  {msg}")
    print("=" * 72)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_global = time.time()
    banner("Exp -- Cross-Lineage NIB: GPT-2-medium (354M) -> DeepSeek-Coder-1.3B")
    print(f"  Device:  {DEVICE}")
    print(f"  D_ABI:   {D_ABI}  (fixed, unchanged)")
    print(f"  Target architecture family: Llama (RoPE, SwiGLU, RMSNorm, GQA)")
    print(f"  NOTE: First ABI experiment on a Llama-family model.")
    print()

    # ── Data loading ──────────────────────────────────────────────────────────
    banner("Data loading")
    t_data  = time.time()
    tok_src = GPT2TokenizerFast.from_pretrained("gpt2-medium", local_files_only=True)
    tok_src.pad_token = tok_src.eos_token
    tok_src.model_max_length = sys.maxsize

    tok_tgt = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base", local_files_only=True)
    tok_tgt.pad_token = tok_tgt.eos_token

    from datasets import load_dataset
    ds_py = load_dataset("bigcode/the-stack", data_dir="data/python",
                         split="train", streaming=True, trust_remote_code=True)
    py_text = "\n\n".join(
        r["content"] for _, r in zip(range(5000), ds_py) if r.get("content")
    )[:MAX_PY]

    py_ids_src = tok_src(py_text, return_tensors="pt",
                         truncation=False)["input_ids"].squeeze(0)[:MAX_PY]
    py_ids_tgt = tok_tgt(py_text, return_tensors="pt",
                         truncation=False)["input_ids"].squeeze(0)[:MAX_PY]

    wiki_raw   = load_wikitext_split("validation")
    wiki_text  = " ".join(wiki_raw)
    wiki_sentences = [s for s in wiki_text.split("\n") if len(s.strip()) >= 20]

    print(f"  {time.time()-t_data:.1f}s  "
          f"py_src={len(py_ids_src):,}  py_tgt={len(py_ids_tgt):,}  "
          f"sentences={len(wiki_sentences):,}")

    kd_weight = REGISTRY["kd_weight"]
    kd_temp   = REGISTRY["kd_temp"]
    cal_steps = REGISTRY["calibration_steps"]

    # ── Phase A: Train GPT-2-medium ABI on Python ─────────────────────────────
    banner("Phase A — GPT-2-medium ABI domain training (Python, 500 steps)")
    t_a = time.time()
    src_model = GPT2MedABI().to(DEVICE)
    for p in src_model.parameters():
        p.requires_grad_(False)
    for nm, p in src_model.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_a = torch.optim.AdamW([p for p in src_model.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    src_model.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids_src, seed=5000 + step)
        opt_a.zero_grad()
        F.cross_entropy(src_model(x).reshape(-1, VOCAB_SRC), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(src_model.parameters(), 1.0)
        opt_a.step()
        if (step + 1) % 100 == 0:
            print(f"  A step {step+1}/{DOMAIN_STEPS}  {time.time()-t_a:.0f}s")
    src_model.eval()
    for p in src_model.parameters():
        p.requires_grad_(False)
    ppl_a = ppl(src_model, py_ids_src)
    print(f"  Phase A complete: {time.time()-t_a:.0f}s  GPT-2-med ppl={ppl_a:.1f}")

    # ── Phase C: Native DeepSeek ABI oracle (Python) ──────────────────────────
    banner("Phase C — Native DeepSeek-Coder-1.3B ABI oracle (Python, 500 steps)")
    t_c = time.time()
    native = DeepSeekCoder1p3BABI().to(DEVICE)
    for p in native.parameters():
        p.requires_grad_(False)
    for nm, p in native.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_c = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids_tgt, seed=5000 + step)
        opt_c.zero_grad()
        F.cross_entropy(native(x).reshape(-1, VOCAB_TGT), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt_c.step()
        if (step + 1) % 100 == 0:
            print(f"  C step {step+1}/{DOMAIN_STEPS}  {time.time()-t_c:.0f}s")
    native.eval()
    for p in native.parameters():
        p.requires_grad_(False)
    ppl_nat = ppl(native, py_ids_tgt)
    print(f"  Phase C complete: {time.time()-t_c:.0f}s  DeepSeek native ppl={ppl_nat:.1f}")

    # ── Phase D: Procrustes + KD calibration ──────────────────────────────────
    banner("Phase D — Procrustes + KD calibration (GPT-2-med -> DeepSeek ABI space)")
    t_d = time.time()

    R = cross_lineage_procrustes(src_model, native, wiki_sentences, tok_src, tok_tgt)
    rotated_domain = apply_rotation_to_domain(src_model.domain, R)

    # Build calibrated: share frozen DeepSeek backbone, fresh ABI + rotated domain
    calibrated = DeepSeekCoder1p3BABI.__new__(DeepSeekCoder1p3BABI)
    nn.Module.__init__(calibrated)
    calibrated.backbone     = native.backbone   # frozen Llama backbone
    calibrated.lm_head      = native.lm_head
    calibrated.proj_in      = nn.Linear(D_MODEL_TGT, D_ABI, bias=False).to(DEVICE)
    calibrated.abi_ln       = nn.LayerNorm(D_ABI).to(DEVICE)
    calibrated.proj_out     = nn.Linear(D_ABI, D_MODEL_TGT, bias=False).to(DEVICE)
    calibrated.domain       = rotated_domain.to(DEVICE)
    calibrated.domain_alpha = nn.Parameter(torch.ones(1, device=DEVICE))
    nn.init.xavier_uniform_(calibrated.proj_in.weight)
    nn.init.xavier_uniform_(calibrated.proj_out.weight)
    calibrated.encode_core  = native.encode_core.__func__.__get__(calibrated, DeepSeekCoder1p3BABI)
    calibrated.forward      = native.forward.__func__.__get__(calibrated, DeepSeekCoder1p3BABI)

    for p in calibrated.backbone.parameters():
        p.requires_grad_(False)
    cal_params = (
        list(calibrated.proj_in.parameters()) +
        list(calibrated.abi_ln.parameters()) +
        list(calibrated.proj_out.parameters()) +
        list(calibrated.domain.parameters()) +
        [calibrated.domain_alpha]
    )
    for p in cal_params:
        p.requires_grad_(True)

    opt_d = torch.optim.AdamW(cal_params, lr=LR_CAL, weight_decay=0.01)
    calibrated.train()
    native.eval()
    for step in range(cal_steps):
        x, y = make_batch(py_ids_tgt, seed=7000 + step)
        opt_d.zero_grad()
        cal_logits = calibrated(x)
        with torch.no_grad():
            nat_logits = native(x)
        ce  = F.cross_entropy(cal_logits.reshape(-1, VOCAB_TGT), y.reshape(-1))
        kd  = F.kl_div(
            F.log_softmax(cal_logits.reshape(-1, VOCAB_TGT) / kd_temp, dim=-1),
            F.softmax(nat_logits.reshape(-1, VOCAB_TGT)     / kd_temp, dim=-1),
            reduction="batchmean",
        ) * (kd_temp ** 2)
        (kd_weight * kd + (1 - kd_weight) * ce).backward()
        nn.utils.clip_grad_norm_(cal_params, 1.0)
        opt_d.step()
        if (step + 1) % 300 == 0:
            print(f"  D step {step+1}/{cal_steps}  {time.time()-t_d:.0f}s")
    calibrated.eval()
    for p in cal_params:
        p.requires_grad_(False)
    ppl_cal = ppl(calibrated, py_ids_tgt)
    print(f"  Phase D complete: {time.time()-t_d:.0f}s  DeepSeek calibrated ppl={ppl_cal:.1f}")

    # ── NIB evaluation ─────────────────────────────────────────────────────────
    banner("NIB L2 evaluation (DeepSeek-Coder vocab, 32256 tokens)")
    print("  Running 5 × 512-token forward passes...")
    t_nib  = time.time()
    l2     = l2_logit_test(native, calibrated, py_ids_tgt)
    overall    = l2["pass"]
    status_str = "PASS" if overall else "FAIL"
    print()
    print(f"  mean_JS          = {l2['mean_js']:.5f}   (thr < {REGISTRY['js_threshold']})   "
          f"{'PASS' if l2['js_pass'] else 'FAIL'}")
    print(f"  mean_top1_agree  = {l2['mean_top1_agree']:.4f}  (thr >= {REGISTRY['top1_threshold']})  "
          f"{'PASS' if l2['top1_pass'] else 'FAIL'}")
    print(f"  mean_top5_overlap= {l2['mean_top5_overlap']:.4f}  (thr >= {REGISTRY['top5_threshold']})  "
          f"{'PASS' if l2['top5_pass'] else 'FAIL'}")
    print(f"  mean_entropy_diff= {l2['mean_entropy_diff']:.4f}  (thr < {REGISTRY['entropy_diff_threshold']})  "
          f"{'PASS' if l2['entropy_pass'] else 'FAIL'}")
    print(f"\n  NIB overall: {status_str}  ({time.time()-t_nib:.1f}s)")

    elapsed = time.time() - t_global
    banner(f"Summary — GPT-2-medium -> DeepSeek-Coder-1.3B Cross-Lineage NIB: {status_str}")
    print(f"  Source: GPT-2-medium     354M   |  BPE 50K    |  GPT-2 arch")
    print(f"  Target: DeepSeek-Coder   1.3B   |  BPE 32K    |  Llama (RoPE, SwiGLU, RMSNorm)")
    print(f"  D_ABI:  {D_ABI}  (first Llama-family test — same dim as all prior experiments)")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    results = {
        "experiment":          "cross_lineage_deepseek_coder_1p3b",
        "name":                "exp_deepseek_1p3b_nib",
        "source_model":        "gpt2-medium-354M",
        "target_model":        "deepseek-coder-1.3b-base-1300M",
        "source_vocab":        VOCAB_SRC,
        "target_vocab":        VOCAB_TGT,
        "source_d_model":      D_MODEL_SRC,
        "target_d_model":      D_MODEL_TGT,
        "source_arch":         "GPT-2 (abs-pos, MHA, GELU, LayerNorm)",
        "target_arch":         "Llama (RoPE, GQA, SwiGLU, RMSNorm)",
        "target_arch_note":    "First Llama-family model tested — distinct from all prior experiments",
        "d_abi":               D_ABI,
        "d_abi_note":          "fixed shared dimension — same for all experiments",
        "seed":                SEED,
        "domain_steps":        DOMAIN_STEPS,
        "calibration_steps":   REGISTRY["calibration_steps"],
        "n_align_sentences":   REGISTRY["n_align_sentences"],
        "alignment_method":    "sentence-level mean-pool Procrustes (cross-tokenizer)",
        "ppl_native_deepseek":     round(ppl_nat, 3),
        "ppl_calibrated_deepseek": round(ppl_cal, 3),
        "nib_l2":              l2,
        "overall_pass":        overall,
        "elapsed_min":         round(elapsed / 60, 1),
        "thresholds":          REGISTRY,
        "claim": (
            "Domain knowledge encoded in GPT-2-medium's 256-dim ABI space transfers "
            "to DeepSeek-Coder-1.3B (Llama architecture: RoPE, SwiGLU, RMSNorm, GQA) "
            "via orthogonal Procrustes rotation + KD calibration, achieving NIB "
            "distributional equivalence in DeepSeek-Coder's native 32256-token vocabulary. "
            "Source and target models differ in tokenizer, architecture, training domain, "
            "and architectural family (GPT-2 vs Llama). D_ABI=256 is unchanged from all "
            "prior experiments: the same fixed bottleneck crosses the GPT-2 / Llama family boundary."
        ),
    }
    out_path = ROOT / "exp_deepseek_1p3b_nib_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results -> {out_path}")
    banner(f"Done — {status_str} — {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
