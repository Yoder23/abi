#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abi/models.py
=============
ABI architecture for T5-large: Autonomous Basis Injection.

This module defines the two model classes used in the published experiment
(Path 2C, Experiment 45AS):

  - AnchorABI         Single-tap anchor model (D_ABI=512, tap=[24]).
                      Trained first, then frozen. Its correction vector is
                      the calibration target for every subsequent candidate.

  - CandidateABI      6-tap candidate (taps=[19..24], per-tap LayerNorm,
                      D_ABI=4096). The architecture whose NIB result is the
                      published breakthrough: top-5 agreement = 0.8725, PASS.

Both classes share the same frozen T5-large backbone and implement
`forward_with_correction()`, which returns both the final logits and the
intermediate correction vector used by the corrMSE training objective.

Published architecture constants
---------------------------------
  TAP_LAYERS  = [19, 20, 21, 22, 23, 24]   # 6 consecutive final decoder layers
  D_MODEL     = 1024                         # T5-large hidden dimension
  D_IN        = 6 × 1024 = 6144             # concatenated tap input
  D_ABI       = 4096                         # ABI bottleneck
  proj_in     : Linear(6144 → 4096, bias=False)
  abi_ln      : LayerNorm(4096)
  proj_out    : Linear(4096 → 1024, bias=False)
  domain      : Linear(4096→16384) + GELU + Linear(16384→4096) + LayerNorm(4096)
  domain_alpha: scalar learnable weight
  Total ABI params: 163,627,009
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ── Architecture constants (immutable) ────────────────────────────────────────

TAP_LAYERS  = [19, 20, 21, 22, 23, 24]   # decoder layer indices to tap
D_MODEL     = 1024                         # T5-large hidden size
D_IN        = D_MODEL * len(TAP_LAYERS)   # 6144
D_ABI       = 4096                         # bottleneck width (best: null-space geometry)
D_ABI_ANCHOR = 512                         # anchor uses smaller bottleneck
DOMAIN_SCALE = 4                           # expansion ratio inside DomainModule


# ── Sub-modules ───────────────────────────────────────────────────────────────

class DomainModule(nn.Module):
    """
    Two-layer MLP with skip-free architecture:
        h → Linear(d, d*4) → GELU → Linear(d*4, d) → LayerNorm(d)

    Acts as a domain-specific residual that adds structure to the ABI
    bottleneck representation. The learnable scalar `domain_alpha`
    (held on the parent model) gates its contribution.
    """
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * DOMAIN_SCALE),
            nn.GELU(),
            nn.Linear(d * DOMAIN_SCALE, d),
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(h))


# ── Main model classes ─────────────────────────────────────────────────────────

class AnchorABI(nn.Module):
    """
    Single-tap anchor model.

    Reads the final decoder layer (layer 24) of T5-large.
    Uses D_ABI=512 — sufficient for an anchor because its role is only to
    produce a calibration *target* correction vector, not to maximise NIB.

    Architecture:
        h_24                      (T5 layer-24 hidden state)
        h_abi   = LN(proj_in(h_24))    Linear(1024 → 512)
        h_out   = h_abi + alpha * domain(h_abi)
        corr    = proj_out(h_out)       Linear(512 → 1024)
        h_final = corr + h_24
        logits  = lm_head(h_final / sqrt(1024))   [tie_word_embeddings scaling]

    Params frozen during calibration of CandidateABI.
    """
    def __init__(self, seed: int = 42):
        super().__init__()
        self.t5           = T5ForConditionalGeneration.from_pretrained("t5-large")
        self.model_dim    = self.t5.config.d_model          # 1024
        self.proj_in      = nn.Linear(self.model_dim, D_ABI_ANCHOR, bias=False)
        self.abi_ln       = nn.LayerNorm(D_ABI_ANCHOR)
        self.proj_out     = nn.Linear(D_ABI_ANCHOR, self.model_dim, bias=False)
        self.domain       = DomainModule(D_ABI_ANCHOR)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def retie_weights(self):
        self.t5.tie_weights()

    def _encode(self, enc_ids, dec_ids):
        enc_out = self.t5.encoder(input_ids=enc_ids)
        dec_out = self.t5.decoder(
            input_ids=dec_ids,
            encoder_hidden_states=enc_out.last_hidden_state,
        )
        h     = dec_out.last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        return h, h_abi

    def forward(self, enc_ids, dec_ids, use_domain: bool = True) -> torch.Tensor:
        h, h_abi = self._encode(enc_ids, dec_ids)
        h_out    = h_abi + self.domain_alpha * self.domain(h_abi) if use_domain else h_abi
        h_final  = self.proj_out(h_out) + h
        if self.t5.config.tie_word_embeddings:
            h_final = h_final * (self.model_dim ** -0.5)
        return self.t5.lm_head(h_final)

    def forward_with_correction(self, enc_ids, dec_ids, use_domain: bool = True):
        h, h_abi   = self._encode(enc_ids, dec_ids)
        h_out      = h_abi + self.domain_alpha * self.domain(h_abi) if use_domain else h_abi
        correction = self.proj_out(h_out)
        h_final    = correction + h
        if self.t5.config.tie_word_embeddings:
            h_final = h_final * (self.model_dim ** -0.5)
        return self.t5.lm_head(h_final), correction


class CandidateABI(nn.Module):
    """
    6-tap candidate model — the published breakthrough architecture.

    Per-tap LayerNorm + D_ABI=4096 over 6 consecutive final decoder layers.

    Key insight over a single-tap model:
      (1) Per-tap LN equalises the magnitude of hidden states from different
          layers before concatenation, giving the optimizer a stable input.
      (2) D_ABI=4096 opens a 3072-dimensional null space in proj_out
          (since proj_out maps 4096 → 1024). Correction residuals can
          occupy directions in this null space that are orthogonal to the
          top-5 token embedding rows, making them *logit-neutral*. This is
          why the architecture achieves top-5 agreement = 0.8725 despite
          a corrMSE floor of ~0.003047.

    Architecture:
        For each tap layer k in [19..24]:
            h_k   = decoder hidden state at layer k
            h_k'  = tap_ln[k](h_k)               per-tap LayerNorm

        h_tap   = cat([h_19', h_20', ..., h_24'], dim=-1)   [B, T, 6144]
        h_24    = last_hidden_state                           [B, T, 1024]  (residual)
        h_abi   = abi_ln(proj_in(h_tap))                     [B, T, 4096]
        h_out   = h_abi + domain_alpha * domain(h_abi)       [B, T, 4096]
        corr    = proj_out(h_out)                             [B, T, 1024]
        h_final = corr + h_24                                 [B, T, 1024]
        logits  = lm_head(h_final / sqrt(1024))

    Published result (Path 2C, Experiment 45AS):
        Official NIB (rng=7777, n=5):
            JS=0.01391  top1=0.8508  top5=0.8725  ent=0.2256  PASS=True
        Extended NIB (n=25, seeds: 7777,1111,2222,3333,4444):
            mean_top5=0.8549  95% CI=[0.8425, 0.8673]
        Training:
            corrMSE=0.003047  @step 15466  elapsed=237.4 min
    """
    def __init__(self, seed: int = 99):
        super().__init__()
        self.t5           = T5ForConditionalGeneration.from_pretrained("t5-large")
        self.model_dim    = self.t5.config.d_model          # 1024
        self.tap_lns      = nn.ModuleList(
            [nn.LayerNorm(self.model_dim) for _ in TAP_LAYERS]
        )   # 6 × LayerNorm(1024) — one per tap, equalises scale before concat
        self.proj_in      = nn.Linear(D_IN, D_ABI, bias=False)     # 6144 → 4096
        self.abi_ln       = nn.LayerNorm(D_ABI)
        self.proj_out     = nn.Linear(D_ABI, self.model_dim, bias=False)  # 4096 → 1024
        self.domain       = DomainModule(D_ABI)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def retie_weights(self):
        self.t5.tie_weights()

    def _encode(self, enc_ids, dec_ids):
        enc_out = self.t5.encoder(input_ids=enc_ids)
        dec_out = self.t5.decoder(
            input_ids=dec_ids,
            encoder_hidden_states=enc_out.last_hidden_state,
            output_hidden_states=True,          # exposes all decoder layer states
        )
        # Per-tap normalisation before concatenation
        h_tap = torch.cat(
            [self.tap_lns[i](dec_out.hidden_states[layer_idx])
             for i, layer_idx in enumerate(TAP_LAYERS)],
            dim=-1,
        )   # [B, T, 6144]
        h_24  = dec_out.last_hidden_state       # unscaled final layer for residual
        h_abi = self.abi_ln(self.proj_in(h_tap))
        return h_24, h_abi

    def forward(self, enc_ids, dec_ids, use_domain: bool = True) -> torch.Tensor:
        h_24, h_abi = self._encode(enc_ids, dec_ids)
        h_out       = h_abi + self.domain_alpha * self.domain(h_abi) if use_domain else h_abi
        h_final     = self.proj_out(h_out) + h_24
        if self.t5.config.tie_word_embeddings:
            h_final = h_final * (self.model_dim ** -0.5)
        return self.t5.lm_head(h_final)

    def forward_with_correction(self, enc_ids, dec_ids, use_domain: bool = True):
        h_24, h_abi = self._encode(enc_ids, dec_ids)
        h_out       = h_abi + self.domain_alpha * self.domain(h_abi) if use_domain else h_abi
        correction  = self.proj_out(h_out)
        h_final     = correction + h_24
        if self.t5.config.tie_word_embeddings:
            h_final = h_final * (self.model_dim ** -0.5)
        return self.t5.lm_head(h_final), correction


# ── Utility ───────────────────────────────────────────────────────────────────

def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze the T5 backbone and unfreeze only ABI-specific parameters.
    The backbone is never updated after this call during ABI training.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if any(k in name for k in ("proj_in", "abi_ln", "proj_out", "domain", "tap_lns")):
            p.requires_grad_(True)


def count_parameters(model: nn.Module) -> dict:
    """Return trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
