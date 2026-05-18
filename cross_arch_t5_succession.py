#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 40 -- T5-large Backbone-Update Invariance (Enc-Dec Succession Test)
===============================================================================
Tests whether ABI domain modules trained on T5-large survive backbone fine-tuning
with the ABI stability constraint -- the encoder-decoder analog of the decoder-only
succession_test_v2.py result (GPT-2-medium, 65% efficacy at 1000 steps).

Gap being closed:
  The GPT-2 succession tests confirmed backbone-update invariance for decoder-only
  models at 354M scale (65% efficacy @ 1000 steps, signal persists at 3000 steps).
  Earlier T5 update experiments suggested catastrophic forgetting -- but those used
  standard T5 teacher-forcing, which produces a degenerate oracle (ppl~1.18, margin~0).
  This experiment uses prefix-LM mode (encoder=prefix, decoder=continuation) which
  creates genuine decoder uncertainty and is the correct T5 causal-LM analog.

Protocol:
  Phase A : Train T5-large+ABI on Python domain (prefix-LM, 500 steps)
            Encoder: 64-token prefix.  Decoder: predict 64-token continuation.
            ABI: single-tap on T5 decoder final hidden state, d_abi=256.
            proj_in is FROZEN from here on (fixes ABI coordinate frame).
  Phase B : Fine-tune T5-large backbone on WikiText-2 (1000 steps)
            Loss: LM_loss(T5_prefix-LM_on_WikiText) + alpha * MSE(h_abi_new, h_abi_pre)
            proj_in frozen (same as succession_test_v2.py insight for decoder-only).
  Eval    : Zero-shot Python PPL with domain module from Phase A.
            Efficacy = (no-domain PPL - domain PPL) / (no-domain PPL - cold-start PPL)
            Pass criterion: efficacy >= 50% (same bar as decoder-only succession test).
  Phase C : Native cold-start T5-large+ABI on Python (500 steps) -- PPL oracle.

Key design:
  The ABI stability constraint during Phase B:
    L = cross_entropy(T5_prefix_LM_output, wiki_continuation_tokens)
        + alpha * MSE(proj_in(h_dec_final), proj_in_frozen(h_dec_final_pre_update))
  This forces the T5 decoder to keep producing ABI-compatible representations even
  as the backbone updates on WikiText. proj_in is frozen throughout, so the ABI
  coordinate frame is fixed -- the domain module from Phase A can be directly reused.

Result file: cross_arch_t5_succession_results.json
Runtime:     ~45-60 min on RTX 3080 Laptop.
"""

import copy
import json
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5TokenizerFast

sys.stdout.reconfigure(line_buffering=True)

ROOT   = pathlib.Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters ────────────────────────────────────────────────────────────
D_ABI        = 256
D_MODEL      = 1024    # T5-large decoder d_model
VOCAB        = 32128   # T5 SentencePiece vocab

PREFIX_LEN   = 64
CONT_LEN     = 64
SEQ_LEN      = PREFIX_LEN + CONT_LEN  # 128 total tokens

DOMAIN_STEPS = 500     # Phase A: Python domain training
UPDATE_STEPS = 1000    # Phase B: WikiText backbone update
COLD_STEPS   = 500     # Phase C: cold-start oracle
LR_ABI       = 3e-4
LR_BACKBONE  = 5e-5
ALPHA_STAB   = 1.0     # ABI stability coefficient (same as decoder-only tests)
BATCH        = 4
SEED         = 42
MAX_PY       = 500_000
MAX_WIKI     = 600_000

# Pass criterion: zero-shot transfer efficacy >= 50%
EFFICACY_THRESHOLD = 0.50

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Domain module (identical structure across all experiments) ─────────────────

class DomainModuleSV(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ln  = nn.LayerNorm(d)
    def forward(self, h):
        return self.ln(self.net(h))


# ── T5-large + ABI (prefix-LM mode) ───────────────────────────────────────────

class T5ABIModel(nn.Module):
    """T5-large in prefix-LM mode with d_abi=256 ABI bottleneck.

    ABI module: single-tap on T5 decoder final hidden state.
      proj_in  (1024 -> 256): projects to shared ABI space
      domain   (256 -> 256):  domain-specific transformation
      proj_out (256 -> 1024): projects back to residual stream
    """
    def __init__(self, freeze_backbone=True):
        super().__init__()
        t5 = T5ForConditionalGeneration.from_pretrained(
            "t5-large", local_files_only=True)
        self.encoder   = t5.encoder
        self.decoder   = t5.decoder
        self.lm_head   = t5.lm_head
        self.d_model   = t5.config.d_model   # 1024
        self.dec_start = t5.config.decoder_start_token_id  # 0
        del t5
        if freeze_backbone:
            for p in self.encoder.parameters(): p.requires_grad_(False)
            for p in self.decoder.parameters(): p.requires_grad_(False)
            for p in self.lm_head.parameters(): p.requires_grad_(False)
        # ABI components
        self.proj_in  = nn.Linear(D_MODEL, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, D_MODEL, bias=False)
        self.domain   = DomainModuleSV(D_ABI)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def _forward_t5(self, x):
        """Run T5 in prefix-LM mode. Returns (h_dec, cont)."""
        B = x.shape[0]
        prefix = x[:, :PREFIX_LEN]
        cont   = x[:, PREFIX_LEN:SEQ_LEN]
        enc_attn = (prefix != 0).long()
        dec_start_tok = torch.full(
            (B, 1), self.dec_start, dtype=x.dtype, device=x.device)
        dec_in = torch.cat([dec_start_tok, cont[:, :-1]], dim=1)
        enc_out = self.encoder(
            input_ids=prefix,
            attention_mask=enc_attn,
        ).last_hidden_state
        dec_out = self.decoder(
            input_ids=dec_in,
            encoder_hidden_states=enc_out,
            encoder_attention_mask=enc_attn,
            use_cache=False,
        )
        return dec_out.last_hidden_state, cont  # [B, CONT_LEN, 1024], [B, CONT_LEN]

    def forward_raw(self, x):
        """Pure T5 backbone logits -- NO ABI correction at all.
        Used for LM loss in Phase B and for all no-domain PPL measurements.
        """
        h_dec, cont = self._forward_t5(x)
        h_final = h_dec * (self.d_model ** -0.5)
        return self.lm_head(h_final), cont

    def get_h_abi(self, x):
        """Return (h_dec, h_abi, cont). h_abi = abi_ln(proj_in(h_dec)).
        Used to build the Phase A reference cache and for stability loss.
        """
        h_dec, cont = self._forward_t5(x)
        h_abi = self.abi_ln(self.proj_in(h_dec))
        return h_dec, h_abi, cont

    def forward(self, x, use_domain=True):
        """Full ABI forward (domain module active). Used for ABI PPL measurements."""
        h_dec, cont = self._forward_t5(x)
        h_abi = self.abi_ln(self.proj_in(h_dec))
        if use_domain:
            h_out = h_abi + self.domain_alpha * self.domain(h_abi)
        else:
            h_out = h_abi
        correction = self.proj_out(h_out)
        h_scaled = (h_dec + correction) * (self.d_model ** -0.5)
        return self.lm_head(h_scaled), cont


# ── Utilities ──────────────────────────────────────────────────────────────────

def make_batch(tokens, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    starts = torch.randint(0, max_start, (BATCH,), generator=rng)
    return torch.stack([tokens[s : s + SEQ_LEN] for s in starts]).to(DEVICE)


def lm_loss(logits, cont):
    """T5 prefix-LM loss: logits[:,i,:] predicts cont[:,i]."""
    return F.cross_entropy(logits.reshape(-1, VOCAB), cont.reshape(-1))


@torch.no_grad()
def compute_ppl_raw(model, tokens, n_batches=30):
    """PPL using pure backbone (no ABI). Requires forward_raw."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(n_batches):
        x = make_batch(tokens, seed=20000 + i)
        logits, cont = model.forward_raw(x)
        tot += lm_loss(logits, cont).item()
        n += 1
    return math.exp(tot / n)


@torch.no_grad()
def compute_ppl_abi(model, tokens, n_batches=30):
    """PPL using full ABI (backbone + domain module active)."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(n_batches):
        x = make_batch(tokens, seed=20000 + i)
        logits, cont = model(x, use_domain=True)
        tot += lm_loss(logits, cont).item()
        n += 1
    return math.exp(tot / n)


def banner(msg):
    print()
    print("=" * 72)
    print(f"  {msg}")
    print("=" * 72)


# ── Protocol ───────────────────────────────────────────────────────────────────

def main():
    t_global = time.time()
    banner("Experiment 40 -- T5-large Backbone-Update Invariance (Enc-Dec Succession)")
    print(f"  Device:  {DEVICE}")
    print(f"  Model:   T5-large (730M, d_model={D_MODEL}, prefix-LM mode)")
    print(f"  d_abi:   {D_ABI}  (fixed)")
    print(f"  Phase A: {DOMAIN_STEPS} domain steps (Python, prefix-LM)")
    print(f"  Phase B: {UPDATE_STEPS} backbone update steps (WikiText-2, stability alpha={ALPHA_STAB})")
    print(f"  Pass:    zero-shot efficacy >= {EFFICACY_THRESHOLD*100:.0f}%")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    print("  [Data] Loading tokenizer and corpora...")
    t_data = time.time()
    tok = T5TokenizerFast.from_pretrained("t5-large", local_files_only=True)

    from wikitext_cache import load_wikitext_split
    wiki_records = [r for r in load_wikitext_split("wikitext-2-raw-v1", "train")
                    if r["text"].strip()]
    wiki_raw = "\n".join(r["text"] for r in wiki_records)

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

    py_ids   = tok(py_raw,   return_tensors="pt",
                   truncation=False)["input_ids"].squeeze(0)[:MAX_PY]
    wiki_ids = tok(wiki_raw, return_tensors="pt",
                   truncation=False)["input_ids"].squeeze(0)[:MAX_WIKI]

    print(f"  [Data] {time.time()-t_data:.1f}s  "
          f"py={len(py_ids):,}  wiki={len(wiki_ids):,} tokens (T5 vocab)")

    # ── Phase A: T5 + ABI domain training on Python ───────────────────────
    banner("Phase A -- T5-large+ABI Domain Training (Python, prefix-LM)")
    t_a = time.time()
    model = T5ABIModel(freeze_backbone=True).to(DEVICE)
    for nm, p in model.named_parameters():
        p.requires_grad_(any(k in nm for k in
                             ("proj_in", "abi_ln", "proj_out", "domain")))
    opt_a = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_ABI, weight_decay=0.01)

    # Baseline: pure T5 backbone PPL (no ABI at all)
    ppl_py_base = compute_ppl_raw(model, py_ids)
    print(f"  Baseline T5 Python ppl (raw backbone): {ppl_py_base:.1f}")

    model.train()
    for step in range(DOMAIN_STEPS):
        x = make_batch(py_ids, seed=5000 + step)
        opt_a.zero_grad()
        logits, cont = model(x, use_domain=True)
        lm_loss(logits, cont).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_a.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/{DOMAIN_STEPS}  {time.time()-t_a:.0f}s")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ppl_py_domain_a  = compute_ppl_abi(model, py_ids)
    ppl_wiki_raw_pre = compute_ppl_raw(model, wiki_ids)
    print(f"  [A] Done {time.time()-t_a:.0f}s")
    print(f"  Python ppl: raw={ppl_py_base:.1f}  domain={ppl_py_domain_a:.1f}  "
          f"(improvement: {(ppl_py_base - ppl_py_domain_a)/max(ppl_py_base,1)*100:.1f}%)")
    print(f"  WikiText raw ppl (pre-update): {ppl_wiki_raw_pre:.1f}")

    # Save the full Phase A ABI state
    domain_state   = copy.deepcopy(model.domain.state_dict())
    domain_alpha   = model.domain_alpha.data.clone()
    proj_in_state  = copy.deepcopy(model.proj_in.state_dict())
    proj_out_state = copy.deepcopy(model.proj_out.state_dict())
    abi_ln_state   = copy.deepcopy(model.abi_ln.state_dict())
    print(f"  Phase A ABI state saved.")

    # ── Pre-compute Phase A h_abi reference cache for Phase B stability ───
    banner("Pre-computing Phase A h_abi reference cache (1000 batches -> CPU)")
    print("  These are the EXACT Phase A checkpoint h_abi values for each Phase B batch.")
    print("  Phase B stability loss = MSE(h_abi_updated_backbone, h_abi_ref_cached).")
    t_cache = time.time()
    h_abi_refs = []
    with torch.no_grad():
        model.eval()
        for step in range(UPDATE_STEPS):
            x = make_batch(wiki_ids, seed=9000 + step)
            _, h_abi_ref, _ = model.get_h_abi(x)
            h_abi_refs.append(h_abi_ref.cpu())
    cache_mb = sum(t.numel() for t in h_abi_refs) * 4 / 1e6
    print(f"  Cached {UPDATE_STEPS} reference tensors ({cache_mb:.0f} MB) in {time.time()-t_cache:.1f}s")

    # ── Phase B: Backbone update on WikiText with ABI stability ───────────
    banner(f"Phase B -- Backbone Update on WikiText-2 ({UPDATE_STEPS} steps, alpha={ALPHA_STAB})")
    print("  LM loss:    forward_raw(x)  -- pure backbone, bypasses ABI entirely")
    print("  Stability:  MSE(h_abi_new, h_abi_ref_cached) -- Phase A checkpoint reference")
    print("  proj_in:    FROZEN (fixes ABI coordinate frame across backbone update)")
    t_b = time.time()

    # Unfreeze backbone only; all ABI components stay frozen
    for p in model.encoder.parameters(): p.requires_grad_(True)
    for p in model.decoder.parameters(): p.requires_grad_(True)
    for p in model.proj_in.parameters():   p.requires_grad_(False)
    for p in model.proj_out.parameters():  p.requires_grad_(False)
    for p in model.abi_ln.parameters():    p.requires_grad_(False)
    for p in model.domain.parameters():    p.requires_grad_(False)
    for p in model.lm_head.parameters():   p.requires_grad_(False)

    opt_b = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_BACKBONE, weight_decay=0.01)

    model.train()
    for step in range(UPDATE_STEPS):
        x = make_batch(wiki_ids, seed=9000 + step)
        h_abi_ref = h_abi_refs[step].to(DEVICE)  # Phase A frozen reference
        opt_b.zero_grad()
        # Single T5 forward pass -- compute both losses from the same h_dec
        h_dec, cont = model._forward_t5(x)
        # LM loss: pure backbone (T5 scaling, no ABI corruption)
        h_raw = h_dec * (model.d_model ** -0.5)
        ll = lm_loss(model.lm_head(h_raw), cont)
        # Stability: h_abi from updated backbone vs Phase A checkpoint
        h_abi_new = model.abi_ln(model.proj_in(h_dec))
        stab = F.mse_loss(h_abi_new, h_abi_ref)
        (ll + ALPHA_STAB * stab).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_b.step()
        if (step + 1) % 200 == 0:
            print(f"    step {step+1}/{UPDATE_STEPS}  "
                  f"ll={ll.item():.4f}  stab={stab.item():.6f}  "
                  f"{time.time()-t_b:.0f}s")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ppl_wiki_raw_after  = compute_ppl_raw(model, wiki_ids)
    ppl_py_raw_after    = compute_ppl_raw(model, py_ids)
    print(f"  [B] Done {time.time()-t_b:.0f}s")
    print(f"  WikiText raw ppl: {ppl_wiki_raw_pre:.1f} -> {ppl_wiki_raw_after:.1f} "
          f"({'improved' if ppl_wiki_raw_after < ppl_wiki_raw_pre else 'changed'})")
    print(f"  Python raw ppl (post-update backbone): {ppl_py_raw_after:.1f}")

    # Re-attach full Phase A ABI state and measure zero-shot transfer
    model.domain.load_state_dict(domain_state)
    model.domain_alpha.data.copy_(domain_alpha)
    model.proj_in.load_state_dict(proj_in_state)
    model.proj_out.load_state_dict(proj_out_state)
    model.abi_ln.load_state_dict(abi_ln_state)

    ppl_py_zero_shot = compute_ppl_abi(model, py_ids)
    zero_shot_gain = (ppl_py_raw_after - ppl_py_zero_shot) / max(ppl_py_raw_after, 1.0)
    print(f"\n  Zero-shot Python ppl (Phase A domain re-attached): {ppl_py_zero_shot:.1f}")
    print(f"  Zero-shot gain vs raw backbone: "
          f"{ppl_py_raw_after:.1f} -> {ppl_py_zero_shot:.1f} "
          f"({zero_shot_gain*100:.1f}%)")

    # ── Phase C: Cold-start oracle ─────────────────────────────────────────
    banner("Phase C -- Cold-Start Oracle (Fresh T5-large + Fresh ABI, Python)")
    print("  Reloads original T5-large backbone, randomly initialized ABI.")
    print("  500 domain steps on Python -- establishes the best-achievable PPL ceiling.")
    t_c = time.time()
    oracle_model = T5ABIModel(freeze_backbone=True).to(DEVICE)
    for nm, p in oracle_model.named_parameters():
        p.requires_grad_(any(k in nm for k in
                             ("proj_in", "abi_ln", "proj_out", "domain")))
    opt_c = torch.optim.AdamW(
        [p for p in oracle_model.parameters() if p.requires_grad],
        lr=LR_ABI, weight_decay=0.01)
    oracle_model.train()
    for step in range(COLD_STEPS):
        x = make_batch(py_ids, seed=5000 + step)
        opt_c.zero_grad()
        logits, cont = oracle_model(x, use_domain=True)
        lm_loss(logits, cont).backward()
        nn.utils.clip_grad_norm_(oracle_model.parameters(), 1.0)
        opt_c.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/{COLD_STEPS}  {time.time()-t_c:.0f}s")
    oracle_model.eval()
    for p in oracle_model.parameters():
        p.requires_grad_(False)
    ppl_cold_start = compute_ppl_abi(oracle_model, py_ids)
    print(f"  [C] Done {time.time()-t_c:.0f}s  cold-start ppl={ppl_cold_start:.1f}")
    del oracle_model

    # ── Efficacy calculation ───────────────────────────────────────────────
    banner("Results -- T5-large Backbone-Update Invariance")
    # Efficacy: fraction of cold-start improvement captured by zero-shot transfer
    #   numerator:   raw_ppl_after - zero_shot_ppl  (what Phase A domain buys)
    #   denominator: raw_ppl_after - cold_start_ppl  (what fresh ABI buys on orig backbone)
    denom     = max(ppl_py_raw_after - ppl_cold_start, 1e-6)
    numerator = max(ppl_py_raw_after - ppl_py_zero_shot, 0.0)
    efficacy  = numerator / denom

    elapsed = time.time() - t_global
    passed  = efficacy >= EFFICACY_THRESHOLD

    print(f"  WikiText update: {UPDATE_STEPS} steps, alpha={ALPHA_STAB}, proj_in frozen")
    print(f"  Python raw ppl (baseline, pre-update):       {ppl_py_base:.2f}")
    print(f"  Python ppl  (Phase A domain, pre-update):    {ppl_py_domain_a:.2f}")
    print(f"  Python raw ppl (post-update backbone):       {ppl_py_raw_after:.2f}")
    print(f"  Python ppl  (zero-shot, Phase A domain):     {ppl_py_zero_shot:.2f}")
    print(f"  Python ppl  (cold-start oracle, orig backbone): {ppl_cold_start:.2f}")
    print()
    print(f"  Zero-shot gain vs raw backbone: {zero_shot_gain*100:.1f}%")
    print(f"  Transfer efficacy: "
          f"({ppl_py_raw_after:.2f} - {ppl_py_zero_shot:.2f}) / "
          f"({ppl_py_raw_after:.2f} - {ppl_cold_start:.2f}) "
          f"= {efficacy*100:.1f}%  (>= {EFFICACY_THRESHOLD*100:.0f}% to PASS)")
    print()
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    results = {
        "experiment":               40,
        "name":                     "cross_arch_t5_succession",
        "model":                    "t5-large-730M",
        "architecture":             "T5 (enc-dec, prefix-LM mode)",
        "d_abi":                    D_ABI,
        "prefix_len":               PREFIX_LEN,
        "cont_len":                 CONT_LEN,
        "t5_mode":                  "prefix-LM (encoder=prefix64, decoder=cont64)",
        "seed":                     SEED,
        "domain_steps":             DOMAIN_STEPS,
        "update_steps":             UPDATE_STEPS,
        "alpha_stability":          ALPHA_STAB,
        "proj_in_frozen":           True,
        "stability_target":         "Phase A h_abi pre-computed reference (exact checkpoint)",
        "lm_loss_path":             "forward_raw -- pure T5 backbone, no ABI",
        "ppl_py_raw_base":          round(ppl_py_base,          2),
        "ppl_py_domain_phaseA":     round(ppl_py_domain_a,      2),
        "ppl_wiki_raw_before":      round(ppl_wiki_raw_pre,     2),
        "ppl_wiki_raw_after":       round(ppl_wiki_raw_after,   2),
        "ppl_py_raw_post_update":   round(ppl_py_raw_after,     2),
        "ppl_py_zero_shot":         round(ppl_py_zero_shot,     2),
        "ppl_cold_start_oracle":    round(ppl_cold_start,       2),
        "zero_shot_gain_pct":       round(zero_shot_gain * 100, 2),
        "transfer_efficacy_pct":    round(efficacy * 100,       2),
        "efficacy_threshold_pct":   EFFICACY_THRESHOLD * 100,
        "passed":                   passed,
        "elapsed_min":              round(elapsed / 60, 1),
        "claim": (
            "T5-large backbone fine-tuned on WikiText-2 for 1000 steps with ABI "
            "stability constraint (pre-computed Phase A h_abi reference, proj_in frozen, "
            "alpha=1.0, LM loss via forward_raw bypassing ABI) retains >= 50% "
            "transfer efficacy. Efficacy = (raw_ppl_post - zero_shot_ppl) / "
            "(raw_ppl_post - cold_start_ppl). "
            "Closes backbone-update invariance for encoder-decoder architectures."
        ),
    }

    out_path = ROOT / "cross_arch_t5_succession_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results -> {out_path}")

    if passed:
        print()
        print(f"  *** EXP 40 PASS: T5 backbone-update invariance VALIDATED ***")
        print(f"  Efficacy: {efficacy*100:.1f}% >= {EFFICACY_THRESHOLD*100:.0f}%")
    else:
        print()
        print(f"  *** EXP 40 FAIL: efficacy {efficacy*100:.1f}% < {EFFICACY_THRESHOLD*100:.0f}% ***")
        if efficacy > 0.30:
            print(f"  Signal present. Try alpha={ALPHA_STAB * 2} or more domain steps.")
        else:
            print(f"  Weak signal. Consider longer Phase A or larger d_abi.")


if __name__ == "__main__":
    main()
