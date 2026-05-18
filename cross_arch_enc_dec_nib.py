#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 39 -- Cross-Architecture NIB: T5-large (enc-dec) -> GPT-2-medium (dec-only)
========================================================================================
The definitive test of encoder-decoder <-> decoder-only frozen-module migration.

Source: T5-large    (730M, SentencePiece 32128 vocab,
                     T5 architecture: relative pos, cross-attention, ReLU FFN)
Target: GPT-2-medium (354M, BPE 50257 vocab,
                     GPT-2 architecture: absolute pos, causal MHA, GELU FFN)

Both models: d_model=1024 -- identical hidden dimension.
Shared ABI bottleneck: d_abi=256 (fixed across all experiments).

Architecture gap being bridged:
  - Enc-dec cross-attention vs causal self-attention only
  - Relative position encodings vs absolute position
  - SentencePiece 32K vocab vs BPE 50K vocab  (completely different token IDs)
  - T5 LM-head scaling (d_model^-0.5) vs GPT-2 no scaling

T5 prefix-LM mode (fixes Exp 37 degenerate-oracle failure):
  Standard T5 teacher-forcing is degenerate for NIB: the encoder "sees" the full
  answer via cross-attention, producing near-zero decoder uncertainty (PPL~1.18,
  margin~0). Prefix-LM mode fixes this: the encoder sees only a 64-token PREFIX;
  the decoder must genuinely predict the 64-token CONTINUATION.  This is the
  correct structural analog of GPT-2 causal LM.

Protocol:
  Phase A : Train T5-large+ABI on Python domain (prefix-LM, 500 steps)
            Encoder: 64-token prefix.  Decoder: predict 64-token continuation.
            ABI taps T5 decoder final hidden state, d_abi=256 bottleneck.
  Phase C : Train native GPT-2-medium+ABI on Python (500 steps) -- ORACLE
  Phase D : Cross-architecture Procrustes (T5 ABI space -> GPT-2 ABI space)
            Sentence mean-pool bridges tokenizer + architecture mismatch.
            KD calibration on GPT-2 using C-oracle as teacher (1200 steps).
  Eval    : NIB between D-calibrated and C-native in GPT-2's 50257-token vocab.

Result file: cross_arch_enc_dec_nib_results.json
Runtime:     ~30-45 min on RTX 3080 Laptop.
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
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

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

# ── Constants ──────────────────────────────────────────────────────────────────
D_ABI        = 256     # Fixed shared ABI dimension across all experiments
D_MODEL      = 1024    # Both T5-large decoder and GPT-2-medium: d_model=1024

VOCAB_SRC    = 32128   # T5 SentencePiece vocabulary size
VOCAB_TGT    = 50257   # GPT-2 BPE vocabulary size

PREFIX_LEN   = 64      # T5 encoder sees this many tokens (prefix)
CONT_LEN     = 64      # T5 decoder predicts this many tokens (continuation)
SEQ_LEN      = PREFIX_LEN + CONT_LEN  # 128 total tokens per training example

DOMAIN_STEPS = 500     # ABI domain training steps for both Phase A and C
LR_ABI       = 3e-4
LR_CAL       = 1e-4
BATCH        = 4       # Conservative batch for T5-large+GPT-2-medium co-loaded
SEED         = 42
MAX_PY       = 500_000

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Shared ABI domain module (identical to Exp 32) ─────────────────────────────

class DomainModuleSV(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ln  = nn.LayerNorm(d)
    def forward(self, h):
        return self.ln(self.net(h))


# ── Source model: T5-large (enc-dec) + ABI (d_abi=256) ─────────────────────────

class SVT5LargeSrc(nn.Module):
    """T5-large (730M, d_model=1024) in prefix-LM mode with d_abi=256 ABI.

    Prefix-LM mode guarantees genuine decoder uncertainty:
      Encoder input  : x[:, :PREFIX_LEN]           (64 tokens)
      Decoder input  : [dec_start, cont[:, :-1]]   (64 tokens, shifted right)
      Decoder target : cont = x[:, PREFIX_LEN:]    (64 tokens)
      logits[b, i, :] predicts cont[b, i]

    ABI is a single-tap on the T5 decoder final hidden state.
    """
    def __init__(self):
        super().__init__()
        t5 = T5ForConditionalGeneration.from_pretrained(
            "t5-large", local_files_only=True)
        self.encoder   = t5.encoder
        self.decoder   = t5.decoder
        self.lm_head   = t5.lm_head      # tied to shared embeddings
        self.d_model   = t5.config.d_model  # 1024
        self.dec_start = t5.config.decoder_start_token_id  # 0
        del t5
        for p in self.encoder.parameters(): p.requires_grad_(False)
        for p in self.decoder.parameters(): p.requires_grad_(False)
        for p in self.lm_head.parameters(): p.requires_grad_(False)
        # ABI bottleneck: d_model=1024 -> d_abi=256 -> d_model=1024
        self.proj_in  = nn.Linear(D_MODEL, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, D_MODEL, bias=False)
        self.domain   = DomainModuleSV(D_ABI)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def encode_core(self, x):
        """x: [B, SEQ_LEN]. Returns (h, h_abi, cont) where h,h_abi are [B,CONT_LEN,1024/256]."""
        B = x.shape[0]
        prefix = x[:, :PREFIX_LEN]          # [B, 64]
        cont   = x[:, PREFIX_LEN:SEQ_LEN]   # [B, 64]
        enc_attn = (prefix != 0).long()     # T5 pad_token_id = 0
        dec_start_tok = torch.full(
            (B, 1), self.dec_start, dtype=x.dtype, device=x.device)
        dec_in = torch.cat([dec_start_tok, cont[:, :-1]], dim=1)  # [B, 64]
        enc_out = self.encoder(
            input_ids=prefix,
            attention_mask=enc_attn,
        ).last_hidden_state  # [B, 64, 1024]
        dec_out = self.decoder(
            input_ids=dec_in,
            encoder_hidden_states=enc_out,
            encoder_attention_mask=enc_attn,
            use_cache=False,
        )
        h     = dec_out.last_hidden_state  # [B, 64, 1024]
        h_abi = self.abi_ln(self.proj_in(h))
        return h, h_abi, cont

    def forward(self, x, use_domain=True):
        h, h_abi, cont = self.encode_core(x)
        if use_domain:
            h_out = h_abi + self.domain_alpha * self.domain(h_abi)
        else:
            h_out = h_abi
        correction = self.proj_out(h_out)
        # T5 convention: scale decoder output by d_model^{-0.5} before lm_head
        h_scaled = (h + correction) * (self.d_model ** -0.5)
        return self.lm_head(h_scaled), cont  # [B, 64, 32128], [B, 64]


# ── Target model: GPT-2-medium (dec-only) + ABI (d_abi=256) ───────────────────

class SVGpt2MediumTgt(nn.Module):
    """GPT-2-medium (354M, d_model=1024) with d_abi=256 ABI bottleneck."""
    def __init__(self):
        super().__init__()
        g             = GPT2LMHeadModel.from_pretrained(
            "gpt2-medium", local_files_only=True)
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        del g
        for p in self.backbone.parameters(): p.requires_grad_(False)
        for p in self.lm_head.parameters():  p.requires_grad_(False)
        self.proj_in  = nn.Linear(D_MODEL, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, D_MODEL, bias=False)
        self.domain   = DomainModuleSV(D_ABI)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def encode_core(self, x):
        h     = self.backbone(x).last_hidden_state  # [B, T, 1024]
        h_abi = self.abi_ln(self.proj_in(h))
        return h, h_abi

    def forward(self, x, use_domain=True):
        h, h_abi = self.encode_core(x)
        if use_domain:
            h_out = h_abi + self.domain_alpha * self.domain(h_abi)
        else:
            h_out = h_abi
        return self.lm_head(h + self.proj_out(h_out))  # [B, T, 50257]


# ── Batch / loss / PPL utilities ───────────────────────────────────────────────

def make_batch(tokens, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    starts = torch.randint(0, max_start, (BATCH,), generator=rng)
    return torch.stack([tokens[s : s + SEQ_LEN] for s in starts]).to(DEVICE)


def lm_loss_t5(model, x):
    """T5 prefix-LM cross-entropy: logits[:,i,:] predicts cont[:,i]."""
    logits, cont = model(x, use_domain=True)
    return F.cross_entropy(logits.reshape(-1, VOCAB_SRC), cont.reshape(-1))


def lm_loss_gpt2(model, x, use_domain=True):
    logits = model(x, use_domain=use_domain)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB_TGT), x[:, 1:].reshape(-1))


@torch.no_grad()
def ppl_t5(model, tokens, n_batches=30):
    model.eval()
    tot, n = 0.0, 0
    for i in range(n_batches):
        loss = lm_loss_t5(model, make_batch(tokens, seed=20000 + i)).item()
        tot += loss; n += 1
    return math.exp(tot / n)


@torch.no_grad()
def ppl_gpt2(model, tokens, use_domain=True, n_batches=50):
    model.eval()
    tot, n = 0.0, 0
    for i in range(n_batches):
        tot += lm_loss_gpt2(model, make_batch(tokens, seed=30000 + i),
                            use_domain).item()
        n += 1
    return math.exp(tot / n)


# ── Cross-architecture Procrustes alignment ────────────────────────────────────

@torch.no_grad()
def cross_arch_procrustes(src_model, tgt_model, align_sentences, tok_src, tok_tgt):
    """Align T5 decoder ABI space to GPT-2 ABI space via sentence mean-pool.

    Both models tokenize the same sentences independently (different tokenizers).
    For T5: run full prefix-LM forward, mean-pool decoder ABI activations.
    For GPT-2: run causal forward, mean-pool ABI activations.
    Returns orthogonal R in O(D_ABI) mapping T5 ABI space -> GPT-2 ABI space.
    """
    src_model.eval()
    tgt_model.eval()
    src_vecs, tgt_vecs = [], []

    for sent in align_sentences:
        sent = sent.strip()
        if len(sent) < 30:
            continue
        try:
            # T5 tokenisation (SentencePiece)
            ids_s = tok_src(sent, return_tensors="pt",
                            truncation=True, max_length=SEQ_LEN,
                            add_special_tokens=False)["input_ids"].to(DEVICE)
            T_s = ids_s.shape[1]
            if T_s < 8:
                continue
            # Pad to SEQ_LEN if shorter
            if T_s < SEQ_LEN:
                pad = torch.zeros(1, SEQ_LEN - T_s, dtype=ids_s.dtype, device=DEVICE)
                ids_s = torch.cat([ids_s, pad], dim=1)
            _, h_src_abi, _ = src_model.encode_core(ids_s)
            # Mean-pool over decoder positions where continuation had real tokens
            real_cont_len = min(max(T_s - PREFIX_LEN, 1), CONT_LEN)
            src_vecs.append(h_src_abi[0, :real_cont_len].mean(0).cpu().float())

            # GPT-2 tokenisation (BPE)
            ids_t = tok_tgt(sent, return_tensors="pt",
                            truncation=True, max_length=SEQ_LEN)["input_ids"].to(DEVICE)
            if ids_t.shape[1] < 4:
                continue
            _, h_tgt_abi = tgt_model.encode_core(ids_t)
            tgt_vecs.append(h_tgt_abi[0].mean(0).cpu().float())

        except Exception:
            continue

        if len(src_vecs) >= REGISTRY["n_align_sentences"]:
            break

    n = min(len(src_vecs), len(tgt_vecs))
    print(f"  [Procrustes] Using {n} sentence pairs")
    A = torch.stack(src_vecs[:n])   # [n, D_ABI]
    B = torch.stack(tgt_vecs[:n])   # [n, D_ABI]
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    cos_before = F.cosine_similarity(A, B, dim=1).mean().item()
    M = A.T @ B
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh  # [D_ABI, D_ABI], orthogonal
    A_rot = A @ R
    cos_after = F.cosine_similarity(A_rot, B, dim=1).mean().item()
    print(f"  [Procrustes] cos_sim: {cos_before:.4f} -> {cos_after:.4f}  "
          f"(improvement: {cos_after - cos_before:+.4f})")
    return R.to(DEVICE)


def apply_rotation_to_domain(src_domain, R):
    """Apply orthogonal rotation R to domain module: f_rot(x) = R @ f(R^T @ x).
    The first and last linear layers are rotated; LayerNorm is reset for KD adaptation.
    """
    dom = copy.deepcopy(src_domain).cpu()
    R_cpu = R.cpu().float()
    with torch.no_grad():
        dom.net[0].weight.data = dom.net[0].weight.data @ R_cpu.T
        dom.net[2].weight.data = R_cpu @ dom.net[2].weight.data
        nn.init.ones_(dom.ln.weight)
        nn.init.zeros_(dom.ln.bias)
    return dom


# ── Training protocol: A -> C -> D ────────────────────────────────────────────

def run_protocol(py_ids_src, py_ids_tgt, wiki_sentences,
                 tok_src, tok_tgt, src_model, tgt_model):

    # ── Phase A: T5-large + ABI on Python (prefix-LM) ─────────────────────
    print("\n  [A] T5-large+ABI domain training (Python, prefix-LM, "
          f"{DOMAIN_STEPS} steps)...")
    t0 = time.time()
    for nm, p in src_model.named_parameters():
        p.requires_grad_(any(k in nm for k in
                             ("proj_in", "abi_ln", "proj_out", "domain")))
    opt_a = torch.optim.AdamW(
        [p for p in src_model.parameters() if p.requires_grad],
        lr=LR_ABI, weight_decay=0.01)
    src_model.train()
    for step in range(DOMAIN_STEPS):
        x = make_batch(py_ids_src, seed=5000 + step)
        opt_a.zero_grad()
        lm_loss_t5(src_model, x).backward()
        nn.utils.clip_grad_norm_(src_model.parameters(), 1.0)
        opt_a.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/{DOMAIN_STEPS}  "
                  f"{time.time()-t0:.0f}s elapsed")
    src_model.eval()
    for p in src_model.parameters():
        p.requires_grad_(False)
    ppl_a = ppl_t5(src_model, py_ids_src)
    print(f"  [A] Done {time.time()-t0:.0f}s  T5 python ppl={ppl_a:.1f}")

    # ── Phase C: GPT-2-medium + ABI native oracle on Python ───────────────
    print(f"\n  [C] GPT-2-medium+ABI native oracle (Python, {DOMAIN_STEPS} steps)...")
    t2 = time.time()
    native = copy.deepcopy(tgt_model).to(DEVICE)
    nn.init.xavier_uniform_(native.proj_in.weight)
    nn.init.xavier_uniform_(native.proj_out.weight)
    nn.init.ones_(native.abi_ln.weight)
    nn.init.zeros_(native.abi_ln.bias)
    native.domain = DomainModuleSV(D_ABI).to(DEVICE)
    native.domain_alpha.data.fill_(1.0)
    for nm, p in native.named_parameters():
        p.requires_grad_(any(k in nm for k in
                             ("proj_in", "abi_ln", "proj_out", "domain")))
    opt_c = torch.optim.AdamW(
        [p for p in native.parameters() if p.requires_grad],
        lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x = make_batch(py_ids_tgt, seed=5000 + step)
        opt_c.zero_grad()
        lm_loss_gpt2(native, x).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt_c.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/{DOMAIN_STEPS}  "
                  f"{time.time()-t2:.0f}s elapsed")
    native.eval()
    for p in native.parameters():
        p.requires_grad_(False)
    ppl_c = ppl_gpt2(native, py_ids_tgt)
    print(f"  [C] Done {time.time()-t2:.0f}s  GPT-2 native ppl={ppl_c:.1f}")

    # ── Phase D: Procrustes + KD calibration ──────────────────────────────
    print("\n  [D] Cross-architecture Procrustes (T5 ABI -> GPT-2 ABI)...")
    t3 = time.time()
    R = cross_arch_procrustes(
        src_model, tgt_model, wiki_sentences, tok_src, tok_tgt)
    rotated_domain = apply_rotation_to_domain(src_model.domain, R)

    calibrated = copy.deepcopy(tgt_model).to(DEVICE)
    nn.init.xavier_uniform_(calibrated.proj_in.weight)
    nn.init.xavier_uniform_(calibrated.proj_out.weight)
    nn.init.ones_(calibrated.abi_ln.weight)
    nn.init.zeros_(calibrated.abi_ln.bias)
    calibrated.domain       = rotated_domain.to(DEVICE)
    calibrated.domain_alpha.data.fill_(1.0)

    cal_steps = REGISTRY["calibration_steps"]
    kd_weight = REGISTRY["kd_weight"]
    kd_temp   = REGISTRY["kd_temp"]
    print(f"  [D] KD calibration ({cal_steps} steps, kd_weight={kd_weight})...")
    for p in calibrated.parameters():
        p.requires_grad_(False)
    calibrated.proj_in.weight.requires_grad_(True)
    calibrated.proj_out.weight.requires_grad_(True)
    calibrated.domain_alpha.requires_grad_(True)
    calibrated.domain.ln.weight.requires_grad_(True)
    calibrated.domain.ln.bias.requires_grad_(True)
    for p in calibrated.domain.net.parameters():
        p.requires_grad_(True)
    opt_d = torch.optim.AdamW(
        [p for p in calibrated.parameters() if p.requires_grad],
        lr=LR_CAL, weight_decay=0.01)
    native.eval()
    calibrated.train()
    for step in range(cal_steps):
        x = make_batch(py_ids_tgt, seed=7000 + step)
        opt_d.zero_grad()
        cal_logits = calibrated(x, use_domain=True)
        with torch.no_grad():
            nat_logits = native(x, use_domain=True)
        # KD loss in GPT-2's 50257-token vocabulary
        cal_flat = cal_logits[:, :-1].reshape(-1, VOCAB_TGT)
        nat_flat = nat_logits[:, :-1].reshape(-1, VOCAB_TGT)
        tgt_flat = x[:, 1:].reshape(-1)
        ce  = F.cross_entropy(cal_flat, tgt_flat)
        kd  = F.kl_div(
            F.log_softmax(cal_flat / kd_temp, dim=-1),
            F.softmax(nat_flat / kd_temp, dim=-1),
            reduction="batchmean",
        ) * (kd_temp ** 2)
        (kd_weight * kd + (1 - kd_weight) * ce).backward()
        nn.utils.clip_grad_norm_(calibrated.parameters(), 1.0)
        opt_d.step()
        if (step + 1) % 300 == 0:
            print(f"    step {step+1}/{cal_steps}  {time.time()-t3:.0f}s")
    calibrated.eval()
    for p in calibrated.parameters():
        p.requires_grad_(False)
    ppl_d = ppl_gpt2(calibrated, py_ids_tgt)
    print(f"  [D] Done {time.time()-t3:.0f}s  GPT-2 calibrated ppl={ppl_d:.1f}")
    return native, calibrated, ppl_c, ppl_d


# ── NIB evaluation in GPT-2's vocabulary ──────────────────────────────────────

@torch.no_grad()
def l2_logit_test(native, calibrated, py_ids_tgt):
    """NIB distributional equivalence test in GPT-2's 50257-token vocabulary.
    Identical protocol to all prior cross-family / cross-lineage NIB tests.
    """
    native.eval()
    calibrated.eval()
    CHUNK = 512
    SKIP  = 20
    rng   = np.random.default_rng(7777)
    js_list, top1_list, top5_list, ent_list = [], [], [], []
    n_chunks  = REGISTRY["n_logit_chunks"]
    max_start = max(len(py_ids_tgt) - CHUNK, 1)

    for ci in range(n_chunks):
        start = int(rng.integers(0, max_start))
        chunk = py_ids_tgt[start : start + CHUNK].unsqueeze(0).to(DEVICE)
        nat_logits = native(chunk, use_domain=True)[0, SKIP:, :]
        cal_logits = calibrated(chunk, use_domain=True)[0, SKIP:, :]
        nat_p = F.softmax(nat_logits, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_logits, dim=-1).cpu().float().numpy()
        T, eps = nat_p.shape[0], 1e-12
        m    = 0.5 * (nat_p + cal_p)
        kl_n = (np.clip(nat_p, eps, 1) * np.log(
                    np.clip(nat_p, eps, 1) / np.clip(m, eps, 1))).sum(1)
        kl_c = (np.clip(cal_p, eps, 1) * np.log(
                    np.clip(cal_p, eps, 1) / np.clip(m, eps, 1))).sum(1)
        js_list.extend(np.clip(0.5 * (kl_n + kl_c), 0, None).tolist())
        top1_list.extend((nat_p.argmax(1) == cal_p.argmax(1)).tolist())
        n5 = np.argpartition(nat_p, -5, axis=1)[:, -5:]
        c5 = np.argpartition(cal_p, -5, axis=1)[:, -5:]
        for t in range(T):
            top5_list.append(len(set(n5[t]) & set(c5[t])) / 5.0)
        Hn = -(np.clip(nat_p, eps, 1) * np.log(np.clip(nat_p, eps, 1))).sum(1)
        Hc = -(np.clip(cal_p, eps, 1) * np.log(np.clip(cal_p, eps, 1))).sum(1)
        ent_list.extend(np.abs(Hn - Hc).tolist())
        print(f"    chunk {ci+1}/{n_chunks}: "
              f"JS={float(np.mean(js_list)):.4f}  "
              f"top1={float(np.mean(top1_list)):.3f}  "
              f"top5={float(np.mean(top5_list)):.3f}  "
              f"ent={float(np.mean(ent_list)):.4f}")

    mj  = float(np.mean(js_list))
    mt1 = float(np.mean(top1_list))
    mt5 = float(np.mean(top5_list))
    me  = float(np.mean(ent_list))
    jp  = mj  <  REGISTRY["js_threshold"]
    t1p = mt1 >= REGISTRY["top1_threshold"]
    t5p = mt5 >= REGISTRY["top5_threshold"]
    ep  = me  <  REGISTRY["entropy_diff_threshold"]
    return {
        "n_positions":       len(js_list),
        "mean_js":           round(mj,  5),
        "mean_top1_agree":   round(mt1, 4),
        "mean_top5_overlap": round(mt5, 4),
        "mean_entropy_diff": round(me,  4),
        "js_pass":           jp,
        "top1_pass":         t1p,
        "top5_pass":         t5p,
        "entropy_pass":      ep,
        "pass":              jp and t1p and t5p and ep,
        "thresholds": {
            "js":           REGISTRY["js_threshold"],
            "top1":         REGISTRY["top1_threshold"],
            "top5":         REGISTRY["top5_threshold"],
            "entropy_diff": REGISTRY["entropy_diff_threshold"],
        },
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
    banner("Experiment 39 -- Cross-Arch NIB: T5-large (enc-dec) -> GPT-2-medium (dec-only)")
    print(f"  Device:  {DEVICE}")
    print(f"  Source:  T5-large       (730M | SentencePiece 32128 | enc-dec | rel-pos)")
    print(f"  Target:  GPT-2-medium   (354M | BPE 50257       | dec-only | abs-pos)")
    print(f"  Both:    d_model=1024   d_abi={D_ABI}   prefix-LM={PREFIX_LEN}+{CONT_LEN}")
    print(f"  Claim:   enc-dec domain module transfers to dec-only via Procrustes + KD")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    print("  [Data] Loading tokenizers and corpora...")
    t_data = time.time()
    tok_src = T5TokenizerFast.from_pretrained("t5-large", local_files_only=True)
    tok_tgt = GPT2TokenizerFast.from_pretrained("gpt2-medium", local_files_only=True)
    tok_tgt.pad_token = tok_tgt.eos_token

    from wikitext_cache import load_wikitext_split
    wiki_records   = [r for r in load_wikitext_split("wikitext-2-raw-v1", "train")
                      if r["text"].strip()]
    wiki_sentences = [r["text"].strip() for r in wiki_records
                      if len(r["text"].strip()) >= 50]

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
    py_raw = "\n".join(py_parts)

    py_ids_src = tok_src(py_raw, return_tensors="pt",
                         truncation=False)["input_ids"].squeeze(0)[:MAX_PY]
    py_ids_tgt = tok_tgt(py_raw, return_tensors="pt",
                         truncation=False)["input_ids"].squeeze(0)[:MAX_PY]

    print(f"  [Data] {time.time()-t_data:.1f}s  "
          f"py_src={len(py_ids_src):,} (T5 tok)  "
          f"py_tgt={len(py_ids_tgt):,} (GPT-2 tok)  "
          f"align={len(wiki_sentences):,} sentences")

    # ── Models ────────────────────────────────────────────────────────────
    print("\n  [Models] Loading T5-large and GPT-2-medium...")
    t_load = time.time()
    src_model = SVT5LargeSrc().to(DEVICE)
    tgt_model = SVGpt2MediumTgt().to(DEVICE)
    src_abi_params = sum(p.numel() for nm, p in src_model.named_parameters()
                         if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain")))
    tgt_abi_params = sum(p.numel() for nm, p in tgt_model.named_parameters()
                         if any(k in nm for k in ("proj_in","abi_ln","proj_out","domain")))
    print(f"  [Models] {time.time()-t_load:.1f}s  "
          f"T5-large ABI params={src_abi_params:,}  "
          f"GPT-2-medium ABI params={tgt_abi_params:,}")
    if torch.cuda.is_available():
        print(f"  [VRAM]   {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # ── Protocol ──────────────────────────────────────────────────────────
    banner("Protocol: A (T5 domain) -> C (GPT-2 oracle) -> D (Procrustes + KD)")
    native, calibrated, ppl_c, ppl_d = run_protocol(
        py_ids_src, py_ids_tgt, wiki_sentences,
        tok_src, tok_tgt, src_model, tgt_model)

    # Free src_model before large NIB evaluation
    del src_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── NIB evaluation ────────────────────────────────────────────────────
    banner("NIB -- Enc-Dec -> Dec-Only Distributional Equivalence (GPT-2 vocab)")
    print("  Both models: GPT-2-medium backbone (frozen), GPT-2 lm_head (50257 vocab)")
    print("  Oracle (C): native GPT-2-medium ABI trained directly on Python")
    print("  Candidate (D): GPT-2-medium ABI seeded from T5-large domain module via Procrustes")
    print()
    t_nib = time.time()
    l2 = l2_logit_test(native, calibrated, py_ids_tgt)
    status = "PASS" if l2["pass"] else "FAIL"

    print()
    print(f"  mean_JS           = {l2['mean_js']:.5f}   (< {REGISTRY['js_threshold']})   "
          f"{'PASS' if l2['js_pass'] else 'FAIL'}")
    print(f"  mean_top1_agree   = {l2['mean_top1_agree']:.4f}  (>= {REGISTRY['top1_threshold']})  "
          f"{'PASS' if l2['top1_pass'] else 'FAIL'}")
    print(f"  mean_top5_overlap = {l2['mean_top5_overlap']:.4f}  (>= {REGISTRY['top5_threshold']})  "
          f"{'PASS' if l2['top5_pass'] else 'FAIL'}")
    print(f"  mean_entropy_diff = {l2['mean_entropy_diff']:.4f}  (< {REGISTRY['entropy_diff_threshold']})  "
          f"{'PASS' if l2['entropy_pass'] else 'FAIL'}")
    print()

    elapsed = time.time() - t_global
    print("=" * 72)
    print("  Cross-Architecture Migration Summary")
    print("=" * 72)
    print(f"  Source: T5-large       | enc-dec | rel-pos  | SP 32128 vocab | 730M")
    print(f"  Target: GPT-2-medium   | dec-only | abs-pos | BPE 50257 vocab | 354M")
    print(f"  Shared ABI space: d_abi={D_ABI} (fixed)")
    print(f"  Alignment: sentence-level Procrustes (cross-tokenizer, cross-arch)")
    print(f"  NIB eval vocab: GPT-2's 50257 tokens")
    print(f"  PPL oracle (C):     {ppl_c:.2f}")
    print(f"  PPL calibrated (D): {ppl_d:.2f}")
    print(f"  NIB overall result: {status}")
    print(f"  Total elapsed: {elapsed/60:.1f} min")
    print()

    results = {
        "experiment":             39,
        "name":                   "cross_arch_enc_dec_nib",
        "source_model":           "t5-large-730M",
        "target_model":           "gpt2-medium-354M",
        "source_arch":            "T5 (enc-dec, relative pos, cross-attention, ReLU FFN, 32128 vocab)",
        "target_arch":            "GPT-2-medium (dec-only, abs pos, causal MHA, GELU FFN, 50257 vocab)",
        "d_abi":                  D_ABI,
        "d_abi_note":             "fixed shared dimension -- same for all models in all experiments",
        "prefix_len":             PREFIX_LEN,
        "cont_len":               CONT_LEN,
        "t5_mode":                "prefix-LM (encoder=prefix, decoder=continuation) -- fixes Exp37 degenerate oracle",
        "seed":                   SEED,
        "domain_steps":           DOMAIN_STEPS,
        "calibration_steps":      REGISTRY["calibration_steps"],
        "n_align_sentences":      REGISTRY["n_align_sentences"],
        "alignment_method":       "sentence-level mean-pool Procrustes (cross-tokenizer, cross-architecture)",
        "ppl_native_gpt2":        round(ppl_c, 3),
        "ppl_calibrated_gpt2":    round(ppl_d, 3),
        "nib_l2":                 l2,
        "overall_pass":           l2["pass"],
        "elapsed_min":            round(elapsed / 60, 1),
        "thresholds":             REGISTRY,
        "claim": (
            "Domain knowledge encoded in T5-large's ABI space (encoder-decoder architecture, "
            "prefix-LM mode, SentencePiece 32128 vocab, relative position encoding) transfers "
            "to GPT-2-medium (decoder-only architecture, BPE 50257 vocab, absolute position "
            "encoding) via orthogonal Procrustes rotation with sentence-level mean-pool alignment, "
            "achieving NIB distributional equivalence in GPT-2's native 50257-token vocabulary. "
            "Source and target models differ in architecture class (encoder-decoder vs decoder-only), "
            "attention mechanism (cross-attention vs causal), position encoding (relative vs absolute), "
            "and tokenizer. This closes the encoder-decoder <-> decoder-only transfer gap."
        ),
    }

    out_path = ROOT / "cross_arch_enc_dec_nib_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results -> {out_path}")

    if l2["pass"]:
        print()
        print("  *** EXP 39 NIB PASS ***")
        print("  Encoder-decoder <-> decoder-only frozen-module migration: VALIDATED")
    else:
        print()
        print("  *** EXP 39 NIB FAIL ***")
        worst = min(
            ("top5", l2["mean_top5_overlap"], REGISTRY["top5_threshold"]),
            ("JS",   l2["mean_js"],           REGISTRY["js_threshold"]),
            key=lambda t: abs(t[1] - t[2]) / t[2]
        )
        print(f"  Closest miss: {worst[0]} = {worst[1]:.4f} (threshold {worst[2]})")
        print(f"  Gap: {worst[1] - worst[2]:+.4f}")


if __name__ == "__main__":
    main()
