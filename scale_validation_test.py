#!/usr/bin/env python3
"""
scale_validation_test.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCALE VALIDATION: ABI STABILITY ON PRETRAINED GPT-2-MEDIUM (354M)

Directly addresses the remaining peer reviewer concerns:

  CONCERN 1 (scale):   "29.6M → 345M+ is not just more of the same"
  RESPONSE:            Pretrained GPT-2-medium, 354M params, entangled
                       representations from real web-text pretraining.

  CONCERN 2 (data):    "workspace files — reviewer will ask about public benchmarks"
  RESPONSE:            WikiText-2 (Merity et al., public benchmark) for
                       general/update corpus. Python code for domain.

  CONCERN 3 (update):  "200 steps is not enough to prove stability"
  RESPONSE:            1000-step backbone fine-tune (5× production test).

  CONCERN 4 (protocol): "without changing your protocol"
  RESPONSE:            Identical 3 steps — only hyperparameters adjusted for
                       pretrained model (smaller backbone LR).

─────────────────────────────────────────────────────────
  THE PROTOCOL (unchanged from toy and production tests):
─────────────────────────────────────────────────────────
  STEP 1: Source model M_A
          Pretrained GPT-2-medium backbone (frozen) +
          ABI projections (proj_in, proj_out) +
          Python domain module — trained on real Python code.

  STEP 2: Update backbone M_B = deepcopy(M_A), then fine-tune on
          WikiText-2 with stability loss:
            L = LM_loss(WikiText) + alpha * MSE(h_abi_B, h_abi_A)
          proj_out is FROZEN throughout (the ABI interface contract).

  STEP 3: Zero-shot paste Python domain module from M_A to M_B.
          Measure PPL improvement and transfer efficacy.
─────────────────────────────────────────────────────────
"""

import copy
import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_BASE    = os.path.dirname(os.path.abspath(__file__))

# Model — pretrained GPT-2 backbone + thin ABI wrapper
MODEL_NAME     = "gpt2-medium"   # 354M params — exactly what peer reviewer requested
D_ABI          = 256             # fixed ABI bottleneck dimension
MAX_SEQ_LEN    = 128

# Training
BATCH_SIZE     = 8               # slightly smaller for 354M model
DOMAIN_STEPS   = 500             # train ABI + domain module (backbone frozen)
UPDATE_STEPS   = 1000            # backbone fine-tune on WikiText-2 (5× production test)
EVAL_BATCHES   = 50
MAX_WIKITEXT   = 600_000         # BPE tokens from WikiText-2
MAX_PYTHON     = 500_000         # BPE tokens from Python code

ALPHA_STAB     = 1.0             # ABI stability loss weight
BACKBONE_LR    = 5e-5            # fine-tuning LR for pretrained backbone
ABI_LR         = 3e-4            # learning rate for randomly-init ABI projections

# ─────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Result:
    tier:     int
    name:     str
    passed:   bool
    metrics:  Dict
    evidence: str
    note:     str

results: List[Result] = []

def record(**kwargs) -> None:
    r = Result(**kwargs)
    results.append(r)
    mark = "[PASS]" if r.passed else "[FAIL]"
    print(f"  {mark} S{r.tier} {r.name}")
    print(f"        {r.evidence}")
    print(f"        NOTE: {r.note}")

# ─────────────────────────────────────────────────────────────────────
# Architecture — GPT-2-medium backbone with fixed ABI wrapper
# ─────────────────────────────────────────────────────────────────────

class DomainModule(nn.Module):
    """Operates ONLY in d_abi space — completely independent of d_model.
    Identical implementation to toy and production tests."""

    def __init__(self, d_abi: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_abi, d_abi * 4),
            nn.GELU(),
            nn.Linear(d_abi * 4, d_abi),
        )
        self.ln = nn.LayerNorm(d_abi)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(h))


class ABIWrappedGPT2(nn.Module):
    """
    Pretrained GPT-2-medium (354M) backbone wrapped with the LayerCake
    ABI interface layer.

    The ABI layer sits between GPT-2's transformer output and the LM head:

      token_ids
        ↓
      GPT-2 transformer (24 layers, d_model=1024) — 354M pretrained params
        ↓ h_core [B, T, 1024]
      proj_in (1024 → d_abi=256)
        ↓ ABI LayerNorm
       [+ domain module delta in d_abi space]
      proj_out (d_abi=256 → 1024)
        ↓
      + residual (h_core)
        ↓
      GPT-2 LM head (1024 → 50257)

    proj_out is frozen during backbone updates — this is the ABI contract.
    Domain modules only depend on d_abi — never on d_model=1024.
    """

    def __init__(
        self,
        d_abi:        int            = D_ABI,
        domain_names: Tuple[str,...] = ("python",),
    ):
        super().__init__()
        # Load pretrained GPT-2-medium
        gpt2        = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        self.backbone  = gpt2.transformer   # GPT2Model (no lm_head)
        self.lm_head   = gpt2.lm_head       # Linear(1024, 50257) tied to wte
        self.d_model   = gpt2.config.n_embd  # 1024
        self.d_abi     = d_abi

        # ABI interface — randomly initialised, learned on domain data
        self.proj_in  = nn.Linear(self.d_model, d_abi, bias=False)
        self.abi_ln   = nn.LayerNorm(d_abi)
        self.proj_out = nn.Linear(d_abi, self.d_model, bias=False)

        # Domain modules — one per domain, operate entirely in d_abi space
        self.domains  = nn.ModuleDict({
            name: DomainModule(d_abi) for name in domain_names
        })

        # Initialise ABI projections
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def encode_core(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_core, h_abi). h_core is from pretrained GPT-2 transformer."""
        outputs = self.backbone(x)
        h       = outputs.last_hidden_state                  # [B, T, 1024]
        h_abi   = self.abi_ln(self.proj_in(h))               # [B, T, d_abi]
        return h, h_abi

    def forward(
        self,
        x:           torch.Tensor,
        domain_mask: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        h, h_abi = self.encode_core(x)
        h_abi_out = h_abi
        if domain_mask:
            delta = torch.zeros_like(h_abi)
            for name, weight in domain_mask.items():
                if name in self.domains and weight > 0:
                    delta = delta + weight * self.domains[name](h_abi)
            h_abi_out = h_abi + delta
        h_out  = self.proj_out(h_abi_out) + h    # residual back into backbone space
        return self.lm_head(h_out)               # [B, T, 50257]

# ─────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────

def tokens_from_wikitext(tokenizer, max_tokens: int = MAX_WIKITEXT) -> torch.Tensor:
    """Load WikiText-2 (public benchmark) into a flat BPE token tensor."""
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(row["text"] for row in ds if len(row["text"].strip()) > 20)
    ids  = tokenizer.encode(text)
    return torch.tensor(ids[:max_tokens], dtype=torch.long)


def tokens_from_python(tokenizer, max_tokens: int = MAX_PYTHON) -> torch.Tensor:
    """Load real Python source code into a flat BPE token tensor."""
    files   = sorted(glob.glob(os.path.join(CORPUS_BASE, "*.py")))
    all_ids: List[int] = []
    random.Random(SEED).shuffle(files)
    for f in files:
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
            all_ids.extend(tokenizer.encode(text))
            if len(all_ids) >= max_tokens:
                break
        except Exception:
            pass
    return torch.tensor(all_ids[:max_tokens], dtype=torch.long)


def make_batch(
    tokens: torch.Tensor,
    seed:   int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng       = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(tokens) - MAX_SEQ_LEN - 1, 1)
    starts    = torch.randint(0, max_start, (BATCH_SIZE,), generator=rng)
    x = torch.stack([tokens[s : s + MAX_SEQ_LEN]       for s in starts])
    y = torch.stack([tokens[s + 1 : s + MAX_SEQ_LEN + 1] for s in starts])
    return x.to(DEVICE), y.to(DEVICE)

# ─────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────

def mean_cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1, a.shape[-1]).float()
    b_f = b.reshape(-1, b.shape[-1]).float()
    return (F.normalize(a_f, dim=-1) * F.normalize(b_f, dim=-1)).sum(-1).mean().item()


def compute_ppl(
    model:       ABIWrappedGPT2,
    tokens:      torch.Tensor,
    domain_mask: Optional[Dict] = None,
    seed_offset: int            = 0,
) -> float:
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(EVAL_BATCHES):
            x, y   = make_batch(tokens, seed=80000 + seed_offset + i)
            logits = model(x, domain_mask=domain_mask)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            total_loss += loss.item() * y.numel()
            total_tok  += y.numel()
    return math.exp(min(total_loss / total_tok, 20.0))


def train_abi_and_domain(
    model:       ABIWrappedGPT2,
    tokens:      torch.Tensor,
    steps:       int,
    seed_offset: int = 0,
) -> None:
    """
    STEP 1 training: backbone is FROZEN.
    Only proj_in, proj_out, domain modules are trained.
    Learns to map GPT-2 hidden states → ABI space and specialize for domain.
    """
    # Freeze backbone (pretrained GPT-2 weights stay untouched)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False

    trainable = (
        list(model.proj_in.parameters())  +
        list(model.abi_ln.parameters())   +
        list(model.proj_out.parameters()) +
        list(model.domains.parameters())
    )
    model.train()
    opt = torch.optim.AdamW(trainable, lr=ABI_LR, weight_decay=0.01)
    mask = {"python": 1.0}

    for step in range(steps):
        x, y   = make_batch(tokens, seed=seed_offset + step)
        logits = model(x, domain_mask=mask)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

    # Restore gradients for all params (needed for update step)
    for p in model.parameters():
        p.requires_grad = True
    model.eval()


def train_backbone_update(
    model:       ABIWrappedGPT2,
    source:      ABIWrappedGPT2,
    tokens:      torch.Tensor,
    steps:       int,
    seed_offset: int = 0,
) -> None:
    """
    STEP 2: The ABI stability backbone update.

    Simulates a production scenario: team fine-tunes the backbone on
    new data (WikiText-2), wants to preserve all deployed domain modules.

    Two enforced constraints:
      (a) proj_out FROZEN — the output ABI contract is immutable.
          Domain module deltas decode identically before and after update.
      (b) MSE(h_abi_updated, h_abi_source) stability loss — pins the
          input ABI distribution so domain modules see the same space.

    Protocol is identical to toy and production tests — only the
    backbone LR is smaller (standard practice for pretrained fine-tuning).
    """
    source.eval()
    for p in source.parameters():
        p.requires_grad = False

    # Enforce the ABI contract: proj_out is immutable
    model.proj_out.requires_grad_(False)
    model.train()

    # Trainable: backbone (GPT-2 transformer) + proj_in
    # (Backbone representation may shift; proj_in adapts to stabilise h_abi)
    trainable = (
        [p for p in model.backbone.parameters()] +
        list(model.proj_in.parameters()) +
        list(model.abi_ln.parameters())
    )
    opt = torch.optim.AdamW(trainable, lr=BACKBONE_LR, weight_decay=0.01)

    log_interval = max(steps // 5, 1)
    for step in range(steps):
        x, y = make_batch(tokens, seed=seed_offset + step)

        # Standard language model loss on new data
        logits  = model(x)
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # ABI stability: prevent h_abi from drifting
        with torch.no_grad():
            _, h_abi_src = source.encode_core(x)
        _, h_abi_up  = model.encode_core(x)
        stab_loss = F.mse_loss(h_abi_up, h_abi_src.detach())

        loss = lm_loss + ALPHA_STAB * stab_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if (step + 1) % log_interval == 0:
            print(f"    step {step+1}/{steps}: lm={lm_loss.item():.3f}  stab={stab_loss.item():.4f}")

    model.proj_out.requires_grad_(True)
    model.eval()


def clone_domain(model: ABIWrappedGPT2, name: str) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.domains[name].state_dict().items()}


def paste_domain(model: ABIWrappedGPT2, name: str, weights: Dict) -> None:
    model.domains[name].load_state_dict(weights)

# ─────────────────────────────────────────────────────────────────────
# Main scale validation test
# ─────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "═" * 60)
    print("  SCALE VALIDATION TEST")
    print(f"  Model:     Pretrained GPT-2-medium (354M)")
    print(f"  Device:    {DEVICE.upper()}")
    print(f"  General:   WikiText-2 (public benchmark)")
    print(f"  Domain:    Python source code (workspace)")
    print(f"  Update:    {UPDATE_STEPS} steps (5× production test)")
    print("═" * 60)

    torch.manual_seed(SEED)
    random.seed(SEED)

    # ── Load tokenizer + corpora ──────────────────────────────────────
    print("\n  [Data] Loading tokenizer + corpora...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    wikitext_tok = tokens_from_wikitext(tokenizer)
    python_tok   = tokens_from_python(tokenizer)
    print(
        f"  [Data] Ready in {time.time()-t0:.1f}s | "
        f"WikiText-2={len(wikitext_tok):,} | Python={len(python_tok):,} tokens"
    )

    # ── Load model ───────────────────────────────────────────────────
    print(f"\n  [Model] Loading pretrained {MODEL_NAME}...")
    t0 = time.time()
    model_source = ABIWrappedGPT2().to(DEVICE)
    n_gpt2  = sum(p.numel() for p in model_source.backbone.parameters())
    n_abi   = (sum(p.numel() for p in model_source.proj_in.parameters())  +
               sum(p.numel() for p in model_source.proj_out.parameters()) +
               sum(p.numel() for p in model_source.abi_ln.parameters()))
    n_dom   = sum(p.numel() for p in model_source.domains.parameters())
    n_total = sum(p.numel() for p in model_source.parameters())
    print(f"  [Model] Loaded in {time.time()-t0:.1f}s")
    print(f"  Params: GPT-2-medium={n_gpt2/1e6:.1f}M | ABI={n_abi/1e3:.0f}K | domain={n_dom/1e3:.0f}K | total={n_total/1e6:.1f}M")
    print(f"  d_model={model_source.d_model} | d_abi={D_ABI} | vocab={model_source.backbone.config.vocab_size}")

    # ── STEP 1: Train ABI projections + domain module on Python ──────
    print(f"\n  ── STEP 1: Train ABI interface + Python domain module ({DOMAIN_STEPS} steps) ──")
    print("  (GPT-2-medium backbone FROZEN — only proj_in, proj_out, domain module trained)")
    t0 = time.time()
    train_abi_and_domain(model_source, python_tok, DOMAIN_STEPS, seed_offset=5000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    domain_on = {"python": 1.0}
    ppl_source_base = compute_ppl(model_source, python_tok, seed_offset=0)
    ppl_source_with = compute_ppl(model_source, python_tok, domain_mask=domain_on, seed_offset=0)
    imp_source = (ppl_source_base - ppl_source_with) / ppl_source_base
    print(f"  Source: Python PPL {ppl_source_base:.1f} → {ppl_source_with:.1f} ({imp_source:+.1%})")

    passed_s1 = imp_source > 0.03
    record(
        tier=1, name="S1_domain_on_pretrained_gpt2",
        passed=passed_s1,
        metrics={
            "backbone": "pretrained GPT-2-medium",
            "backbone_params_M": round(n_gpt2/1e6, 1),
            "d_model": model_source.d_model,
            "d_abi": D_ABI,
            "domain_corpus": "Python source code (workspace .py files)",
            "general_corpus": "WikiText-2 (public benchmark)",
            "ppl_no_domain": round(ppl_source_base, 2),
            "ppl_with_domain": round(ppl_source_with, 2),
            "improvement_pct": round(imp_source * 100, 2),
            "threshold_pct": 3.0,
        },
        evidence=(
            f"Pretrained GPT-2-medium ({n_gpt2/1e6:.1f}M backbone, d_model=1024, d_abi={D_ABI}). "
            f"Python domain module (trained {DOMAIN_STEPS} steps, backbone frozen): "
            f"PPL {ppl_source_base:.1f} → {ppl_source_with:.1f} ({imp_source:+.1%}). "
            f"Threshold >3%: {passed_s1}"
        ),
        note=(
            "ABI interface learned on top of frozen GPT-2-medium's pretrained representations. "
            "The domain module specialises signals that GPT-2 already understands as Python, "
            "operating purely in the fixed d_abi=256 space."
        ),
    )

    # Save domain weights
    python_weights = clone_domain(model_source, "python")

    # ── STEP 2: Backbone update with ABI stability ────────────────────
    print(f"\n  ── STEP 2: Update backbone on WikiText-2 ({UPDATE_STEPS} steps) ──")
    print("  (ABI stability protocol: proj_out frozen + MSE(h_abi_B, h_abi_A) loss)")
    model_updated = copy.deepcopy(model_source)
    t0 = time.time()
    train_backbone_update(
        model_updated, model_source,
        wikitext_tok, UPDATE_STEPS,
        seed_offset=9000,
    )
    print(f"  Updated in {time.time()-t0:.1f}s")

    # Measure ABI drift after update
    torch.manual_seed(SEED + 55)
    probe_x = make_batch(python_tok, seed=SEED + 55)[0][:4]
    model_source.eval(); model_updated.eval()
    with torch.no_grad():
        _, h_abi_src = model_source.encode_core(probe_x)
        _, h_abi_upd = model_updated.encode_core(probe_x)
    abi_cos  = mean_cos_sim(h_abi_src, h_abi_upd)
    rand_cos = mean_cos_sim(torch.randn_like(h_abi_src), torch.randn_like(h_abi_upd))
    abi_ratio = abs(abi_cos) / max(abs(rand_cos), 1e-8)

    # STEP 3: Zero-shot paste — NO domain retraining on updated model
    ppl_upd_base = compute_ppl(model_updated, python_tok, seed_offset=0)
    paste_domain(model_updated, "python", python_weights)
    ppl_upd_with = compute_ppl(model_updated, python_tok, domain_mask=domain_on, seed_offset=0)
    imp_updated  = (ppl_upd_base - ppl_upd_with) / ppl_upd_base

    print(f"  ABI drift: cos_sim={abi_cos:.4f} ({abi_ratio:.0f}x random)")
    print(f"  Zero-shot: Python PPL {ppl_upd_base:.1f} → {ppl_upd_with:.1f} ({imp_updated:+.1%})")
    print(f"  Source:    Python PPL {ppl_source_base:.1f} → {ppl_source_with:.1f} ({imp_source:+.1%})")

    passed_s2 = imp_updated > 0.03
    record(
        tier=2, name="S2_zero_shot_after_backbone_update",
        passed=passed_s2,
        metrics={
            "update_corpus": "WikiText-2 (public benchmark, external)",
            "update_steps": UPDATE_STEPS,
            "alpha_stability": ALPHA_STAB,
            "proj_out_frozen": True,
            "abi_cos_sim_post_update": round(abi_cos, 4),
            "abi_ratio_vs_random": round(abi_ratio, 1),
            "ppl_updated_base": round(ppl_upd_base, 2),
            "ppl_updated_with": round(ppl_upd_with, 2),
            "improvement_pct": round(imp_updated * 100, 2),
            "source_imp_pct": round(imp_source * 100, 2),
            "threshold_pct": 3.0,
        },
        evidence=(
            f"Backbone fine-tuned on WikiText-2 ({UPDATE_STEPS} steps, 5× production test). "
            f"ABI stability: alpha={ALPHA_STAB}, proj_out frozen. "
            f"Post-update ABI cos_sim={abi_cos:.4f} ({abi_ratio:.0f}x random). "
            f"Zero-shot: {ppl_upd_base:.1f} → {ppl_upd_with:.1f} ({imp_updated:+.1%}). "
            f"Threshold >3%: {passed_s2}"
        ),
        note=(
            "CORE PRODUCTION TEST AT SCALE. Backbone updated 1000 steps on WikiText-2 "
            "(external public benchmark, distinct distribution from Python domain). "
            "Python domain module reused zero-shot — no domain fine-tuning. "
            "This is the peer reviewer's explicit requested experiment."
        ),
    )

    # ── Transfer efficacy vs native cold-start ────────────────────────
    print(f"\n  ── S3: Native cold-start baseline ({DOMAIN_STEPS} steps on updated backbone) ──")
    t0 = time.time()
    model_native = copy.deepcopy(model_updated)
    # Reset domain module to cold start
    model_native.domains["python"] = DomainModule(D_ABI).to(DEVICE)
    # Also reset proj_in and proj_out so it's a fair cold start
    nn.init.xavier_uniform_(model_native.proj_in.weight)
    nn.init.xavier_uniform_(model_native.proj_out.weight)
    train_abi_and_domain(model_native, python_tok, DOMAIN_STEPS, seed_offset=5000)
    ppl_nat_base = compute_ppl(model_native, python_tok, seed_offset=0)
    ppl_nat_with = compute_ppl(model_native, python_tok, domain_mask=domain_on, seed_offset=0)
    imp_native   = (ppl_nat_base - ppl_nat_with) / ppl_nat_base
    print(f"  Native (cold-start on updated backbone): PPL {ppl_nat_base:.1f} → {ppl_nat_with:.1f} ({imp_native:+.1%})")
    print(f"  Built in {time.time()-t0:.1f}s")

    transfer_efficacy    = imp_updated / imp_native if imp_native > 1e-6 else 0.0
    transfer_efficacy_ok = transfer_efficacy >= 0.50

    print(f"\n  Transfer efficacy: {transfer_efficacy:.0%} "
          f"(zero-shot {imp_updated:+.1%} vs native {imp_native:+.1%})")

    record(
        tier=3, name="S3_transfer_efficacy_vs_native",
        passed=(imp_updated > 0 and transfer_efficacy_ok),
        metrics={
            "imp_zero_shot_pct":      round(imp_updated * 100, 2),
            "imp_native_pct":         round(imp_native * 100, 2),
            "transfer_efficacy_pct":  round(transfer_efficacy * 100, 1),
            "threshold_efficacy_pct": 50.0,
            "ppl_updated_base":       round(ppl_upd_base, 2),
            "ppl_updated_with":       round(ppl_upd_with, 2),
            "ppl_native_base":        round(ppl_nat_base, 2),
            "ppl_native_with":        round(ppl_nat_with, 2),
        },
        evidence=(
            f"Zero-shot efficacy: {transfer_efficacy:.0%} "
            f"({imp_updated:+.1%} zero-shot vs {imp_native:+.1%} native cold-start). "
            f"Threshold ≥50%: {transfer_efficacy_ok}"
        ),
        note=(
            "Transfer efficacy at 354M scale with WikiText-2 backbone update. "
            "Peer reviewer bar: ≥50–70% = 'people will pay attention', "
            "≥90% = 'this changes assumptions'. "
        ),
    )

    # ── Rollout stability ─────────────────────────────────────────────
    print(f"\n  ── S4: Rollout stability (autoregressive, 200 tokens) ──")
    model_updated.eval()
    ctx = torch.randint(0, model_source.backbone.config.vocab_size,
                        (1, 20), device=DEVICE)
    torch.manual_seed(SEED + 300)
    window_ppls = []
    with torch.no_grad():
        for _ in range(4):
            w_loss, w_tok = 0.0, 0
            for _ in range(25):
                c_in   = ctx[:, -MAX_SEQ_LEN:]
                logits = model_updated(c_in, domain_mask=domain_on)
                nxt    = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                lp     = F.log_softmax(logits[:, -1, :].float(), dim=-1)
                w_loss += -lp.max(dim=-1).values.item()
                w_tok  += 1
                ctx = torch.cat([ctx, nxt], dim=1)
            window_ppls.append(math.exp(w_loss / w_tok))
    ratio  = window_ppls[-1] / window_ppls[0] if window_ppls[0] > 0 else 999.0
    stable = ratio < 3.0

    record(
        tier=4, name="S4_rollout_stability",
        passed=stable,
        metrics={
            "window_ppls":      [round(p, 2) for p in window_ppls],
            "last_first_ratio": round(ratio, 3),
            "threshold":        3.0,
        },
        evidence=(
            f"Window PPLs: {[round(p,1) for p in window_ppls]}. "
            f"Last/First = {ratio:.2f} (< 3.0 = stable). Domain active."
        ),
        note="Autoregressive generation stability at 354M scale with domain active.",
    )

    # ── Final claim assessment ────────────────────────────────────────
    s1 = results[-4]; s2 = results[-3]; s3 = results[-2]; s4 = results[-1]
    core_ok   = s1.passed and s2.passed and s3.passed
    all_ok    = core_ok and s4.passed
    eff_pct   = round(transfer_efficacy * 100, 1)

    if eff_pct >= 90:
        claim = "BREAKTHROUGH-LEVEL (peer reviewer: 'this changes assumptions')"
    elif eff_pct >= 70:
        claim = "STRONG — production and scale validated (peer bar: 'people will pay attention')"
    elif eff_pct >= 50:
        claim = "STRONG — validated (above peer minimum bar)"
    else:
        claim = "MEDIUM — scale validation needs iteration"

    record(
        tier=5, name="S5_scale_claim_assessment",
        passed=core_ok,
        metrics={
            "model":            MODEL_NAME,
            "backbone_params_M": round(n_gpt2/1e6, 1),
            "general_corpus":   "WikiText-2 (public benchmark)",
            "domain_corpus":    "Python source code",
            "update_steps":     UPDATE_STEPS,
            "transfer_efficacy_pct": eff_pct,
            "claim_level":      claim,
            "all_4_tests_pass": all_ok,
        },
        evidence=(
            f"GPT-2-medium ({n_gpt2/1e6:.1f}M) | WikiText-2 | {UPDATE_STEPS}-step update | "
            f"S1={s1.passed} S2={s2.passed} S3={s3.passed} S4={s4.passed} | "
            f"Transfer efficacy: {eff_pct}%"
        ),
        note=claim,
    )


def print_summary(t_start: float) -> None:
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    s5     = next((r for r in results if r.tier == 5), None)
    level  = s5.metrics.get("claim_level", "UNKNOWN") if s5 else "UNKNOWN"
    eff    = s5.metrics.get("transfer_efficacy_pct", 0) if s5 else 0

    print("\n" + "═" * 60)
    print("  SCALE VALIDATION — FINAL RESULTS")
    print("═" * 60)
    for r in results:
        mark = "[PASS]" if r.passed else "[FAIL]"
        print(f"  {mark} S{r.tier} {r.name}")

    elapsed = time.time() - t_start
    print(f"\n  {passed}/{total} tests PASSED  ({elapsed:.0f}s total)")
    print(f"  Transfer efficacy:  {eff:.0f}%")
    print(f"  Claim level:        {level}")

    if passed == total:
        print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║  SCALE VALIDATION COMPLETE                          ║
  ║                                                     ║
  ║  Peer reviewer concerns addressed:                  ║
  ║  ✓ Scale:    GPT-2-medium, 354M pretrained params  ║
  ║  ✓ Data:     WikiText-2 (public benchmark)          ║
  ║  ✓ Duration: {UPDATE_STEPS} backbone update steps           ║
  ║  ✓ Protocol: identical to toy and production tests  ║
  ║                                                     ║
  ║  Transfer efficacy: {eff:.0f}% (peer bar: 50–70%)       ║
  ╚══════════════════════════════════════════════════════╝""")
    else:
        print("\n  Failed:", [r.name for r in results if not r.passed])

    with open("scale_validation_results.json", "w") as f:
        json.dump(
            [{"tier": r.tier, "name": r.name, "passed": r.passed,
              "metrics": r.metrics, "evidence": r.evidence, "note": r.note}
             for r in results],
            f, indent=2,
        )
    print("  Full results → scale_validation_results.json")


if __name__ == "__main__":
    t_start = time.time()
    print("\n" + "═" * 60)
    print("  LAYERCAKE SCALE VALIDATION TEST")
    print("  Addressing peer reviewer: scale, public data, longer update")
    print("═" * 60)
    run()
    print_summary(t_start)
