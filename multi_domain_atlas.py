#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Domain Atlas  --  Domain-Indexed Procrustes Routing
==========================================================
Tests the "domain atlas" hypothesis:

  Each domain occupies a different coordinate chart in ABI space.
  A single Procrustes solve (0 SGD steps) maps the shared backbone to any
  chart.  A hard router selects the correct chart at inference; a soft
  alpha-mixture interpolates between charts.

The test answers:
  1. Can each domain achieve L2 parity with its own rotation?      (diagonal)
  2. What is the cross-domain interference?                        (D x D matrix)
  3. Does off-domain PPL degrade catastrophically?                 (non-degradation bound)
  4. Is there a Pareto frontier for mixtures of two rotations?     (alpha sweep)

Protocol:
  Shared:    A -> B  (Python anchor, WikiText backbone drift -- same as NIB)
  Per-domain: C_D    (native oracle on each domain corpus, 500 steps)
             Proc_D  (Procrustes solve: 0 SGD steps, ~60s per domain)
  Eval:
    - Diagonal L2: calibrated_D vs native_D on domain D tokens
    - Interference matrix: M[eval_d][rot_d] = mean JS(native_d, calibrated with rot_d)
    - PPL matrix: ppl[applied_rot][eval_corpus]
    - Alpha sweep: proj_out = alpha * proj_out_py + (1-alpha) * proj_out_wiki

Domains:
  python   -- local .py files  (~500k tokens)
  wikitext -- wikitext-2-raw-v1 (~600k tokens)
  markdown -- local .md files   (~variable; padded with wikitext-103 if sparse)
  sql      -- synthetic SQL      (~400k tokens, no external download)

Results: multi_domain_atlas_results.json
Runtime: ~55-70 min on RTX 3080 (GPU strongly recommended)
"""

import copy
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import GPT2TokenizerFast

# -- Import shared NIB infrastructure -------------------------------------------
from non_inferiority_benchmark import (
    PROBE_BANK, ADV_VARIANTS, REGISTRY,
    SVGPT2, DomainModuleSV,
    DEVICE, D_ABI, SEQ_LEN, DOMAIN_STEPS, UPDATE_STEPS,
    LR_ABI, LR_BACKBONE, LR_CAL, ALPHA,
    MAX_PY_SV, MAX_WIKI_SV, BATCH_SV, SEED, ROOT,
    make_batch_sv, ppl_sv,
)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DOMAIN_SEED_OFFSET = {
    "python": 101,
    "wikitext": 202,
    "markdown": 303,
    "sql": 404,
}


def seed_domain_stage(dname: str, stage_offset: int = 0):
    seed = SEED + DOMAIN_SEED_OFFSET.get(dname, 900) + stage_offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

# -- Constants ------------------------------------------------------------------
N_COLLECT  = 200      # batches for Procrustes pair collection (200 x 8 x 128 = 204,800)
N_COLLECT_BY_DOMAIN = (
    {"sql": int(os.environ["ATLAS_SQL_N_COLLECT"])}
    if os.environ.get("ATLAS_SQL_N_COLLECT") else {}
)
LSTSQ_RCOND_BY_DOMAIN = (
    {"sql": float(os.environ["ATLAS_SQL_LSTSQ_RCOND"])}
    if os.environ.get("ATLAS_SQL_LSTSQ_RCOND") else {}
)
N_L2_CHUNKS = int(os.environ.get("ATLAS_N_L2_CHUNKS", "5"))  # x 512 positions
L2_SKIP     = 20      # skip first N positions (low-context bias, matching NIB)
MAX_MD_TOKENS = 400_000
MAX_SQL_TOKENS = 400_000
SQL_CORPUS_MODE = os.environ.get("ATLAS_SQL_CORPUS_MODE", "classic").strip().lower()
REFINE_STEPS = 100    # post-Procrustes KD chart refinement; 0 disables
REFINE_PROJ_IN = True # include input geometry in each routed chart
REFINE_PROJ_IN_BY_DOMAIN = (
    {"sql": os.environ.get("ATLAS_SQL_REFINE_PROJ_IN", "1").lower() in {"1", "true", "yes"}}
    if os.environ.get("ATLAS_SQL_REFINE_PROJ_IN") is not None else {}
)
REFINE_STEPS_BY_DOMAIN = {
    # Procrustes already gives WikiText a clean diagonal pass; do not spend
    # calibration where it only adds drift. Spend budget on rank-limited charts.
    "python": 100,
    "wikitext": 0,
    "markdown": 400,
    "sql": int(os.environ.get("ATLAS_SQL_REFINE_STEPS", "5600")),
}
REFINE_L2_CHECKPOINTS_BY_DOMAIN = {
    "sql": [800, 1600, 2400, 3200, 4000, 4800, 5600, 6400, 7200],
}
SELECT_CHECKPOINT_BY_DOMAIN = (
    {"sql": True}
    if os.environ.get("ATLAS_SQL_SELECT_CHECKPOINT", "").lower() in {"1", "true", "yes"}
    else {}
)
SELECT_CHECKPOINT_SEED_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_SELECT_CHECKPOINT_SEED", "17777")),
}
SWA_BY_DOMAIN = (
    {"sql": True}
    if os.environ.get("ATLAS_SQL_SWA", "").lower() in {"1", "true", "yes"}
    else {}
)
SWA_START_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_SWA_START", "4800")),
}
SWA_EVERY_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_SWA_EVERY", "800")),
}
KD_WEIGHT_BY_DOMAIN = {
    "sql": 1.0,
}
KD_TEMP_BY_DOMAIN = {
    "sql": 8.0,
}
RANK_TOPK_K = 5
RANK_TOPK_MARGIN = float(os.environ.get("ATLAS_SQL_RANK_TOPK_MARGIN", "0.05"))
RANK_TOPK_WEIGHT_BY_DOMAIN = (
    {"sql": float(os.environ["ATLAS_SQL_RANK_TOPK_WEIGHT"])}
    if os.environ.get("ATLAS_SQL_RANK_TOPK_WEIGHT") else {}
)
RANK_TOPK_START_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_RANK_TOPK_START", "2400")),
}
TOPK_SET_K = 5
TOPK_SET_WEIGHT_BY_DOMAIN = {}
TOPK_CE_K = 5
TOPK_CE_WEIGHT_BY_DOMAIN = (
    {"sql": float(os.environ["ATLAS_SQL_TOPK_CE_WEIGHT"])}
    if os.environ.get("ATLAS_SQL_TOPK_CE_WEIGHT") else {}
)
TOPK_CE_TEMP_BY_DOMAIN = {
    "sql": float(os.environ.get("ATLAS_SQL_TOPK_CE_TEMP", "1.0")),
}
TOPK_CE_START_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_TOPK_CE_START", "3200")),
}
LOCAL_TOPK_KL_K = 10
LOCAL_TOPK_KL_WEIGHT_BY_DOMAIN = (
    {"sql": float(os.environ["ATLAS_SQL_LOCAL_TOPK_KL_WEIGHT"])}
    if os.environ.get("ATLAS_SQL_LOCAL_TOPK_KL_WEIGHT") else {}
)
LOCAL_TOPK_KL_TEMP_BY_DOMAIN = {
    "sql": float(os.environ.get("ATLAS_SQL_LOCAL_TOPK_KL_TEMP", "1.0")),
}
LOCAL_TOPK_KL_START_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_LOCAL_TOPK_KL_START", "2400")),
}
UNION_TOPK_NATIVE_K = 5
UNION_TOPK_CANDIDATE_K = 10
UNION_TOPK_KL_WEIGHT_BY_DOMAIN = (
    {"sql": float(os.environ["ATLAS_SQL_UNION_TOPK_KL_WEIGHT"])}
    if os.environ.get("ATLAS_SQL_UNION_TOPK_KL_WEIGHT") else {}
)
UNION_TOPK_KL_TEMP_BY_DOMAIN = {
    "sql": float(os.environ.get("ATLAS_SQL_UNION_TOPK_KL_TEMP", "1.0")),
}
UNION_TOPK_KL_START_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_UNION_TOPK_KL_START", "3200")),
}
BOUNDARY_TOPK_POS = 5
BOUNDARY_TOPK_NEG = 10
BOUNDARY_TOPK_MARGIN = float(os.environ.get("ATLAS_SQL_BOUNDARY_TOPK_MARGIN", "0.02"))
BOUNDARY_TOPK_WEIGHT_BY_DOMAIN = (
    {"sql": float(os.environ["ATLAS_SQL_BOUNDARY_TOPK_WEIGHT"])}
    if os.environ.get("ATLAS_SQL_BOUNDARY_TOPK_WEIGHT") else {}
)
BOUNDARY_TOPK_START_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_BOUNDARY_TOPK_START", "3200")),
}
INTRUDER_TOPK_POS = 5
INTRUDER_TOPK_NEG = 10
INTRUDER_TOPK_MARGIN = float(os.environ.get("ATLAS_SQL_INTRUDER_TOPK_MARGIN", "0.02"))
INTRUDER_TOPK_WEIGHT_BY_DOMAIN = (
    {"sql": float(os.environ["ATLAS_SQL_INTRUDER_TOPK_WEIGHT"])}
    if os.environ.get("ATLAS_SQL_INTRUDER_TOPK_WEIGHT") else {}
)
INTRUDER_TOPK_START_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_INTRUDER_TOPK_START", "3200")),
}
REFINE_LOGIT_BIAS_BY_DOMAIN = {}
LOGIT_BIAS_LR_MULT_BY_DOMAIN = {
    "sql": 5.0,
}
LOGIT_BIAS_L2_BY_DOMAIN = {
    "sql": 1e-4,
}
REFINE_LOGIT_SCALE_BY_DOMAIN = (
    {"sql": True}
    if os.environ.get("ATLAS_SQL_REFINE_LOGIT_SCALE", "").lower() in {"1", "true", "yes"}
    else {}
)
LOGIT_SCALE_LR_MULT_BY_DOMAIN = {
    "sql": float(os.environ.get("ATLAS_SQL_LOGIT_SCALE_LR_MULT", "2.0")),
}
LOGIT_SCALE_L2_BY_DOMAIN = {
    "sql": float(os.environ.get("ATLAS_SQL_LOGIT_SCALE_L2", "1e-4")),
}
REFINE_ABI_RESIDUAL_BY_DOMAIN = (
    {"sql": True}
    if os.environ.get("ATLAS_SQL_REFINE_ABI_RESIDUAL", "").lower() in {"1", "true", "yes"}
    else {}
)
ABI_RESIDUAL_RANK_BY_DOMAIN = {
    "sql": int(os.environ.get("ATLAS_SQL_ABI_RESIDUAL_RANK", "8")),
}
ABI_RESIDUAL_SCALE_BY_DOMAIN = {
    "sql": float(os.environ.get("ATLAS_SQL_ABI_RESIDUAL_SCALE", "1.0")),
}
ABI_RESIDUAL_LR_MULT_BY_DOMAIN = {
    "sql": float(os.environ.get("ATLAS_SQL_ABI_RESIDUAL_LR_MULT", "1.0")),
}
ABI_RESIDUAL_L2_BY_DOMAIN = {
    "sql": float(os.environ.get("ATLAS_SQL_ABI_RESIDUAL_L2", "1e-4")),
}
VOCAB_SIZE = 50257

# ===============================================================================
# Utilities
# ===============================================================================

def banner(msg: str):
    w = 72
    print(f"\n{'='*w}\n  {msg}\n{'='*w}", flush=True)

def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def l2_selection_score(res: dict) -> tuple:
    """Validation-only ordering for checkpoint choice; final eval stays separate."""
    return (
        int(bool(res["pass"])),
        float(res["top5"]),
        float(res["top1"]),
        -float(res["mean_js"]),
        -float(res["entropy"]),
    )


def get_nested_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def atlas_forward(model: SVGPT2, x: torch.Tensor, use_domain: bool = True) -> torch.Tensor:
    abi_resid_down = getattr(model, "abi_resid_down", None)
    if abi_resid_down is None:
        logits = model(x, use_domain=use_domain)
    else:
        h, h_abi = model.encode_core(x)
        if use_domain:
            h_out = h_abi + model.domain_alpha * model.domain(h_abi)
        else:
            h_out = h_abi
        logits = model.lm_head(model.proj_out(h_out) + h)
        resid = model.abi_resid_up(F.gelu(abi_resid_down(h_out)))
        logits = logits + float(getattr(model, "abi_resid_scale", 1.0)) * resid
    scale = getattr(model, "logit_scale", None)
    if scale is not None:
        scale = scale.to(logits.device, dtype=logits.dtype).clamp(-0.5, 0.5)
        logits = logits * torch.exp(scale).view(1, 1, -1)
    bias = getattr(model, "logit_bias", None)
    if bias is not None:
        logits = logits + bias.to(logits.device, dtype=logits.dtype).view(1, 1, -1)
    return logits


def load_wikitext_split(config: str, split: str):
    """
    Prefer the local Hugging Face Arrow cache.  On this Windows machine,
    load_dataset(..., offline) can hang while resolving the cached builder,
    but Dataset.from_file() loads the same cached split immediately.
    """
    cache_root = (pathlib.Path.home() / ".cache" / "huggingface" /
                  "datasets" / "wikitext" / config)
    candidates = sorted(
        cache_root.glob(f"**/wikitext-{split}.arrow"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        flush(f"  [data] using cached {config}/{split}: {candidates[0]}")
        return Dataset.from_file(str(candidates[0]))
    return load_dataset("wikitext", config, split=split)


# ===============================================================================
# Data Loading
# ===============================================================================

def load_py_ids(tok) -> torch.Tensor:
    texts = []
    for f in sorted(ROOT.glob("*.py")):
        try:
            texts.append(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    ids = tok("\n".join(texts))["input_ids"][:MAX_PY_SV]
    return torch.tensor(ids, dtype=torch.long)


def load_wiki_ids(tok) -> torch.Tensor:
    ds = load_wikitext_split("wikitext-2-raw-v1", "train")
    text = "\n".join(r for r in ds["text"] if r.strip())
    ids = tok(text)["input_ids"][:MAX_WIKI_SV]
    return torch.tensor(ids, dtype=torch.long)


def load_md_ids(tok) -> torch.Tensor:
    """Load .md files from workspace; fall back to wikitext-103 if < 100k tokens."""
    texts = []
    for pat in (ROOT.glob("*.md"), ROOT.parent.glob("*.md")):
        for f in sorted(pat):
            try:
                texts.append(f.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
    raw_ids = tok("\n\n".join(texts))["input_ids"]
    flush(f"  [md] {len(raw_ids):,} tokens from local .md files")
    if len(raw_ids) < 100_000:
        flush("  [md] padding with wikitext-103-raw-v1 train split...")
        ds = load_wikitext_split("wikitext-103-raw-v1", "train")
        extra_text = "\n".join(r for r in ds["text"] if r.strip())
        extra_ids = tok(extra_text)["input_ids"]
        raw_ids = raw_ids + extra_ids
    ids = raw_ids[:MAX_MD_TOKENS]
    return torch.tensor(ids, dtype=torch.long)


def _make_sql_text() -> str:
    if SQL_CORPUS_MODE == "rich":
        return _make_sql_text_rich()
    if SQL_CORPUS_MODE not in {"classic", ""}:
        raise ValueError(f"Unknown ATLAS_SQL_CORPUS_MODE={SQL_CORPUS_MODE!r}")
    tables = ["users", "orders", "products", "events", "sessions",
              "logs", "items", "accounts", "metrics", "tasks"]
    cols   = ["id", "user_id", "name", "status", "created_at",
              "updated_at", "value", "count", "score", "label"]
    aggs   = ["SUM", "COUNT", "AVG", "MAX", "MIN"]
    rng    = np.random.default_rng(SEED)
    lines  = []
    for i in range(90_000):
        t  = tables[i % len(tables)]
        c1 = cols[i % len(cols)]
        c2 = cols[(i + 3) % len(cols)]
        v  = int(rng.integers(1, 10_000))
        op = i % 6
        if   op == 0: lines.append(f"SELECT {c1}, {c2} FROM {t} WHERE {c1} = {v};")
        elif op == 1:
            agg = aggs[i % len(aggs)]
            lines.append(f"SELECT {agg}({c1}) AS result FROM {t} GROUP BY {c2};")
        elif op == 2: lines.append(f"INSERT INTO {t} ({c1}, {c2}) VALUES ({v}, {v+1});")
        elif op == 3: lines.append(f"UPDATE {t} SET {c1} = {v} WHERE id = {i % 1000};")
        elif op == 4: lines.append(f"DELETE FROM {t} WHERE {c1} < {v};")
        else:         lines.append(f"CREATE INDEX idx_{t}_{c1} ON {t} ({c1});")
    return "\n".join(lines)


def _make_sql_text_rich() -> str:
    tables = ["users", "orders", "products", "events", "sessions", "logs",
              "items", "accounts", "metrics", "tasks", "payments", "invoices",
              "tickets", "messages", "features", "experiments"]
    cols = ["id", "user_id", "account_id", "product_id", "session_id",
            "event_id", "status", "category", "region", "channel",
            "created_at", "updated_at", "deleted_at", "amount", "price",
            "quantity", "score", "rank", "label", "payload", "is_active",
            "priority", "owner_id", "team_id", "started_at", "ended_at"]
    aggs = ["SUM", "COUNT", "AVG", "MAX", "MIN"]
    funcs = ["LOWER", "UPPER", "COALESCE", "DATE_TRUNC"]
    rng = np.random.default_rng(SEED + 17)
    lines = []
    for i in range(95_000):
        t1 = tables[i % len(tables)]
        t2 = tables[(i + 5) % len(tables)]
        t3 = tables[(i + 9) % len(tables)]
        c1 = cols[i % len(cols)]
        c2 = cols[(i + 4) % len(cols)]
        c3 = cols[(i + 11) % len(cols)]
        c4 = cols[(i + 17) % len(cols)]
        agg = aggs[i % len(aggs)]
        fn = funcs[i % len(funcs)]
        v1 = int(rng.integers(1, 10_000))
        v2 = int(rng.integers(10_000, 50_000))
        op = i % 16
        if op == 0:
            lines.append(
                f"SELECT {c1}, {c2}, {c3} FROM {t1} "
                f"WHERE {c1} = {v1} AND {c2} <> {v2} ORDER BY {c3} DESC LIMIT 50;"
            )
        elif op == 1:
            lines.append(
                f"SELECT {agg}({c1}) AS total_{c1}, {c2} FROM {t1} "
                f"WHERE {c3} IS NOT NULL GROUP BY {c2} HAVING {agg}({c1}) > {v1};"
            )
        elif op == 2:
            lines.append(
                f"SELECT a.{c1}, b.{c2} FROM {t1} AS a "
                f"JOIN {t2} AS b ON a.id = b.{c1} WHERE b.{c3} BETWEEN {v1} AND {v2};"
            )
        elif op == 3:
            lines.append(
                f"WITH recent AS (SELECT id, {c1}, {c2} FROM {t1} "
                f"WHERE created_at >= CURRENT_DATE - INTERVAL '30 days') "
                f"SELECT {c1}, COUNT(*) FROM recent GROUP BY {c1};"
            )
        elif op == 4:
            lines.append(
                f"SELECT {c1}, CASE WHEN {c2} > {v1} THEN 'high' ELSE 'low' END AS bucket "
                f"FROM {t1} WHERE {c3} IN (SELECT {c3} FROM {t2} WHERE {c4} = {v2});"
            )
        elif op == 5:
            lines.append(
                f"SELECT {c1}, ROW_NUMBER() OVER (PARTITION BY {c2} ORDER BY {c3} DESC) AS rn "
                f"FROM {t1} WHERE {c4} IS NOT NULL;"
            )
        elif op == 6:
            lines.append(
                f"INSERT INTO {t1} ({c1}, {c2}, {c3}) VALUES ({v1}, {v2}, '{t2}_{i % 97}') "
                f"ON CONFLICT ({c1}) DO UPDATE SET {c2} = EXCLUDED.{c2};"
            )
        elif op == 7:
            lines.append(
                f"UPDATE {t1} SET {c1} = {v1}, {c2} = {c2} + 1 "
                f"WHERE id IN (SELECT id FROM {t2} WHERE {c3} < {v2});"
            )
        elif op == 8:
            lines.append(
                f"DELETE FROM {t1} WHERE {c1} < {v1} AND NOT EXISTS "
                f"(SELECT 1 FROM {t2} WHERE {t2}.{c2} = {t1}.{c2});"
            )
        elif op == 9:
            lines.append(
                f"CREATE TABLE IF NOT EXISTS audit_{t1}_{i % 13} "
                f"(id BIGINT PRIMARY KEY, {c1} TEXT, {c2} INTEGER, created_at TIMESTAMP);"
            )
        elif op == 10:
            lines.append(f"CREATE INDEX idx_{t1}_{c1}_{c2} ON {t1} ({c1}, {c2});")
        elif op == 11:
            lines.append(f"ALTER TABLE {t1} ADD COLUMN IF NOT EXISTS {c4}_flag BOOLEAN DEFAULT FALSE;")
        elif op == 12:
            lines.append(
                f"SELECT {fn}({c1}) AS normalized_{c1}, {c2} FROM {t1} "
                f"UNION ALL SELECT {fn}({c3}), {c4} FROM {t2};"
            )
        elif op == 13:
            lines.append(
                f"SELECT DISTINCT {c1}, {c2} FROM {t1} LEFT JOIN {t2} "
                f"ON {t1}.{c3} = {t2}.{c3} WHERE {t2}.id IS NULL;"
            )
        elif op == 14:
            lines.append(
                f"BEGIN; UPDATE {t1} SET {c2} = {v1} WHERE {c1} = {v2}; "
                f"INSERT INTO {t3} ({c1}, {c2}) VALUES ({v2}, {v1}); COMMIT;"
            )
        else:
            lines.append(
                f"SELECT {c1}, {agg}({c2}) FILTER (WHERE {c3} > {v1}) AS filtered_{c2} "
                f"FROM {t1} GROUP BY {c1} ORDER BY filtered_{c2} NULLS LAST;"
            )
    return "\n".join(lines)


def load_sql_ids(tok) -> torch.Tensor:
    flush(f"  [sql] corpus_mode={SQL_CORPUS_MODE}")
    ids = tok(_make_sql_text())["input_ids"][:MAX_SQL_TOKENS]
    return torch.tensor(ids, dtype=torch.long)


# ===============================================================================
# Batching & PPL
# ===============================================================================

def make_batch(ids: torch.Tensor, seed: int,
               batch: int = BATCH_SV, seq: int = SEQ_LEN):
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(ids) - seq - 1, 1)
    starts = torch.randint(0, max_start, (batch,), generator=rng)
    x = torch.stack([ids[s:s+seq]   for s in starts]).to(DEVICE)
    y = torch.stack([ids[s+1:s+seq+1] for s in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def model_ppl(model: SVGPT2, ids: torch.Tensor,
              n_batches: int = 50, seed: int = 80001) -> float:
    model.eval()
    total = 0.0
    for i in range(n_batches):
        x, y = make_batch(ids, seed=seed + i)
        logits = atlas_forward(model, x)
        total += F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)).item()
    return math.exp(total / n_batches)


# ===============================================================================
# Per-domain Step C: native oracle
# ===============================================================================

def train_native_oracle(transferred_state: dict,
                        domain_ids: torch.Tensor,
                        dname: str) -> SVGPT2:
    """
    Spawn a fresh ABI on the shared transferred backbone and train 500 steps
    on domain_ids.  Returns eval-mode native oracle (no gradients).
    """
    seed_domain_stage(dname, 1000)
    flush(f"  [C_{dname}] training native oracle ({DOMAIN_STEPS} steps)...")
    t0 = time.time()
    native = SVGPT2().to(DEVICE)
    native.load_state_dict(transferred_state)
    nn.init.xavier_uniform_(native.proj_in.weight)
    nn.init.xavier_uniform_(native.proj_out.weight)
    nn.init.ones_(native.abi_ln.weight)
    nn.init.zeros_(native.abi_ln.bias)
    native.domain = DomainModuleSV(D_ABI).to(DEVICE)
    native.domain_alpha.data.fill_(1.0)
    for p in native.parameters(): p.requires_grad_(False)
    for nm, p in native.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad],
                             lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(domain_ids, seed=5000 + step)
        opt.zero_grad()
        F.cross_entropy(native(x).reshape(-1, 50257), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt.step()
    native.eval()
    for p in native.parameters(): p.requires_grad_(False)
    ppl = model_ppl(native, domain_ids)
    flush(f"  [C_{dname}] {time.time()-t0:.0f}s  ppl_nat={ppl:.2f}")
    return native


# ===============================================================================
# Procrustes solve (0 SGD steps)
# ===============================================================================

def procrustes_solve(transferred_state: dict,
                     native: SVGPT2,
                     domain_ids: torch.Tensor,
                     dname: str,
                     n_collect: int = N_COLLECT) -> tuple:
    """
    Collect N_COLLECT batches of (h_full_cal, h_full_nat) ABI vectors.
    Solve A* = lstsq(H_cal, H_nat).
    Bake: proj_out_new.weight = proj_out_nat.weight @ A*.T
    Returns (calibrated_model, A_star_cpu, r_squared, cond_number, ppl_cal).
    """
    flush(f"  [Proc_{dname}] collecting {n_collect} batches...")
    t0 = time.time()

    # Temporary "pre-calibration" model (transferred state)
    pre = SVGPT2().to(DEVICE)
    pre.load_state_dict(transferred_state)
    pre.eval()
    for p in pre.parameters(): p.requires_grad_(False)

    H_cal_list, H_nat_list = [], []
    with torch.no_grad():
        for i in range(n_collect):
            x, _ = make_batch(domain_ids, seed=2000 + i)
            _, h_abi_cal = pre.encode_core(x)     # (B, T, D_ABI)
            _, h_abi_nat = native.encode_core(x)

            # Match procrustes_full_nib.py: Step D solves on the full
            # domain-modulated ABI state, not raw encode_core() output.
            h_full_cal = h_abi_cal + pre.domain_alpha * pre.domain(h_abi_cal)
            h_full_nat = h_abi_nat + native.domain_alpha * native.domain(h_abi_nat)

            H_cal_list.append(h_full_cal.reshape(-1, D_ABI).cpu().float())
            H_nat_list.append(h_full_nat.reshape(-1, D_ABI).cpu().float())
    del pre; torch.cuda.empty_cache()

    H_cal = torch.cat(H_cal_list, dim=0)   # (N, D_ABI)
    H_nat = torch.cat(H_nat_list, dim=0)
    del H_cal_list, H_nat_list

    rcond = LSTSQ_RCOND_BY_DOMAIN.get(dname)
    A_star, _, _, _ = torch.linalg.lstsq(H_cal, H_nat, rcond=rcond)  # (D_ABI, D_ABI)

    # Diagnostics
    H_nat_pred = H_cal @ A_star
    ss_res = float(((H_nat - H_nat_pred) ** 2).sum())
    ss_tot = float(((H_nat - H_nat.mean(0)) ** 2).sum())
    r_sq   = round(1.0 - ss_res / ss_tot, 5) if ss_tot > 0 else 0.0
    sv     = torch.linalg.svdvals(H_cal)
    cond   = float(sv.max() / sv.min().clamp(min=1e-12))
    del H_cal, H_nat, H_nat_pred

    # Build calibrated model
    calibrated = SVGPT2().to(DEVICE)
    calibrated.load_state_dict(transferred_state)
    new_w = native.proj_out.weight.cpu().float() @ A_star.T
    calibrated.proj_out.weight.data.copy_(
        new_w.to(DEVICE).to(calibrated.proj_out.weight.dtype))
    calibrated.domain_alpha.data.copy_(native.domain_alpha.data)
    calibrated.domain.ln.weight.data.copy_(native.domain.ln.weight.data)
    calibrated.domain.ln.bias.data.copy_(native.domain.ln.bias.data)
    for p in calibrated.parameters(): p.requires_grad_(False)
    calibrated.eval()

    ppl_cal = model_ppl(calibrated, domain_ids)
    flush(f"  [Proc_{dname}] R²={r_sq:.5f}  cond={cond:.1f}  "
          f"ppl_cal={ppl_cal:.2f}  ({time.time()-t0:.0f}s)")
    return calibrated, A_star.cpu(), r_sq, cond, ppl_cal


def refine_chart_kd(calibrated: SVGPT2,
                    native: SVGPT2,
                    domain_ids: torch.Tensor,
                    dname: str,
                    steps: int = REFINE_STEPS,
                    eval_checkpoints=None) -> tuple:
    """
    Small repo-native Step-D refinement after Procrustes.

    This is the same calibration mechanism that fixed NIB top-k/entropy
    alignment, but scoped to the domain chart already produced by Procrustes.
    It leaves the learned domain.net untouched and calibrates only chart
    parameters that _extract_proj_weights() can route per domain.
    """
    checkpoint_set = set(eval_checkpoints or [])
    checkpoint_history = []
    select_checkpoint = bool(SELECT_CHECKPOINT_BY_DOMAIN.get(dname, False))
    select_seed = SELECT_CHECKPOINT_SEED_BY_DOMAIN.get(dname, 17777)
    if select_checkpoint:
        checkpoint_set.add(steps)
    use_swa = bool(SWA_BY_DOMAIN.get(dname, False))
    swa_start = SWA_START_BY_DOMAIN.get(dname, 4800)
    swa_every = max(1, SWA_EVERY_BY_DOMAIN.get(dname, 800))
    if use_swa:
        checkpoint_set.update(range(swa_start, steps + 1, swa_every))
        checkpoint_set.add(steps)
    swa_state = {}
    swa_count = 0
    swa_info = {
        "enabled": use_swa,
        "start": swa_start if use_swa else None,
        "every": swa_every if use_swa else None,
        "count": 0,
        "steps": [],
    }
    best_checkpoint = None
    best_checkpoint_state = None
    selected_info = {
        "enabled": select_checkpoint,
        "seed": select_seed if select_checkpoint else None,
        "step": None,
        "score": None,
        "l2": None,
    }
    if steps <= 0:
        return calibrated, model_ppl(calibrated, domain_ids), checkpoint_history, selected_info

    seed_domain_stage(dname, 2000)
    t0 = time.time()
    kd_weight = KD_WEIGHT_BY_DOMAIN.get(dname, REGISTRY["kd_weight"])
    kd_temp   = KD_TEMP_BY_DOMAIN.get(dname, REGISTRY["kd_temp"])
    ce_weight = 1.0 - kd_weight
    refine_proj_in = REFINE_PROJ_IN_BY_DOMAIN.get(dname, REFINE_PROJ_IN)
    rank_topk_weight = RANK_TOPK_WEIGHT_BY_DOMAIN.get(dname, 0.0)
    rank_topk_start = RANK_TOPK_START_BY_DOMAIN.get(dname, 0)
    topk_set_weight = TOPK_SET_WEIGHT_BY_DOMAIN.get(dname, 0.0)
    topk_ce_weight = TOPK_CE_WEIGHT_BY_DOMAIN.get(dname, 0.0)
    topk_ce_temp = TOPK_CE_TEMP_BY_DOMAIN.get(dname, 1.0)
    topk_ce_start = TOPK_CE_START_BY_DOMAIN.get(dname, 0)
    local_topk_kl_weight = LOCAL_TOPK_KL_WEIGHT_BY_DOMAIN.get(dname, 0.0)
    local_topk_kl_temp = LOCAL_TOPK_KL_TEMP_BY_DOMAIN.get(dname, 1.0)
    local_topk_kl_start = LOCAL_TOPK_KL_START_BY_DOMAIN.get(dname, 0)
    union_topk_kl_weight = UNION_TOPK_KL_WEIGHT_BY_DOMAIN.get(dname, 0.0)
    union_topk_kl_temp = UNION_TOPK_KL_TEMP_BY_DOMAIN.get(dname, 1.0)
    union_topk_kl_start = UNION_TOPK_KL_START_BY_DOMAIN.get(dname, 0)
    boundary_weight = BOUNDARY_TOPK_WEIGHT_BY_DOMAIN.get(dname, 0.0)
    boundary_start = BOUNDARY_TOPK_START_BY_DOMAIN.get(dname, 0)
    intruder_weight = INTRUDER_TOPK_WEIGHT_BY_DOMAIN.get(dname, 0.0)
    intruder_start = INTRUDER_TOPK_START_BY_DOMAIN.get(dname, 0)
    use_logit_bias = bool(REFINE_LOGIT_BIAS_BY_DOMAIN.get(dname, False))
    logit_bias_l2 = LOGIT_BIAS_L2_BY_DOMAIN.get(dname, 0.0)
    logit_bias_lr_mult = LOGIT_BIAS_LR_MULT_BY_DOMAIN.get(dname, 1.0)
    use_logit_scale = bool(REFINE_LOGIT_SCALE_BY_DOMAIN.get(dname, False))
    logit_scale_l2 = LOGIT_SCALE_L2_BY_DOMAIN.get(dname, 0.0)
    logit_scale_lr_mult = LOGIT_SCALE_LR_MULT_BY_DOMAIN.get(dname, 1.0)
    use_abi_residual = bool(REFINE_ABI_RESIDUAL_BY_DOMAIN.get(dname, False))
    abi_residual_rank = ABI_RESIDUAL_RANK_BY_DOMAIN.get(dname, 8)
    abi_residual_scale = ABI_RESIDUAL_SCALE_BY_DOMAIN.get(dname, 1.0)
    abi_residual_lr_mult = ABI_RESIDUAL_LR_MULT_BY_DOMAIN.get(dname, 1.0)
    abi_residual_l2 = ABI_RESIDUAL_L2_BY_DOMAIN.get(dname, 0.0)
    flush(f"  [KD_{dname}] refining chart ({steps} steps, "
          f"kd_weight={kd_weight}, T={kd_temp}, "
          f"refine_proj_in={refine_proj_in})...")
    if rank_topk_weight:
        flush(f"  [KD_{dname}] rank_topk k={RANK_TOPK_K}  "
              f"margin={RANK_TOPK_MARGIN}  weight={rank_topk_weight}  "
              f"start={rank_topk_start}")
    if topk_set_weight:
        flush(f"  [KD_{dname}] topk_set_mass k={TOPK_SET_K}  "
              f"weight={topk_set_weight}")
    if topk_ce_weight:
        flush(f"  [KD_{dname}] topk_ce k={TOPK_CE_K}  "
              f"weight={topk_ce_weight}  T={topk_ce_temp}  start={topk_ce_start}")
    if local_topk_kl_weight:
        flush(f"  [KD_{dname}] local_topk_kl k={LOCAL_TOPK_KL_K}  "
              f"weight={local_topk_kl_weight}  T={local_topk_kl_temp}  "
              f"start={local_topk_kl_start}")
    if union_topk_kl_weight:
        flush(f"  [KD_{dname}] union_topk_kl native_k={UNION_TOPK_NATIVE_K}  "
              f"candidate_k={UNION_TOPK_CANDIDATE_K}  "
              f"weight={union_topk_kl_weight}  T={union_topk_kl_temp}  "
              f"start={union_topk_kl_start}")
    if boundary_weight:
        flush(f"  [KD_{dname}] boundary_topk pos={BOUNDARY_TOPK_POS}  "
              f"neg={BOUNDARY_TOPK_NEG}  margin={BOUNDARY_TOPK_MARGIN}  "
              f"weight={boundary_weight}  start={boundary_start}")
    if intruder_weight:
        flush(f"  [KD_{dname}] intruder_topk pos={INTRUDER_TOPK_POS}  "
              f"neg={INTRUDER_TOPK_NEG}  margin={INTRUDER_TOPK_MARGIN}  "
              f"weight={intruder_weight}  start={intruder_start}")
    if use_logit_bias:
        calibrated.logit_bias = nn.Parameter(torch.zeros(VOCAB_SIZE, device=DEVICE))
        flush(f"  [KD_{dname}] logit_bias enabled  lr_mult={logit_bias_lr_mult}  "
              f"l2={logit_bias_l2}")
    if use_logit_scale:
        calibrated.logit_scale = nn.Parameter(torch.zeros(VOCAB_SIZE, device=DEVICE))
        flush(f"  [KD_{dname}] logit_scale enabled  lr_mult={logit_scale_lr_mult}  "
              f"l2={logit_scale_l2}")
    if use_abi_residual:
        calibrated.abi_resid_down = nn.Linear(D_ABI, abi_residual_rank, bias=False).to(DEVICE)
        calibrated.abi_resid_up = nn.Linear(abi_residual_rank, VOCAB_SIZE, bias=False).to(DEVICE)
        calibrated.abi_resid_scale = abi_residual_scale
        nn.init.xavier_uniform_(calibrated.abi_resid_down.weight)
        nn.init.zeros_(calibrated.abi_resid_up.weight)
        flush(f"  [KD_{dname}] abi_residual rank={abi_residual_rank}  "
              f"scale={abi_residual_scale}  lr_mult={abi_residual_lr_mult}  "
              f"l2={abi_residual_l2}")
    if select_checkpoint:
        flush(f"  [KD_{dname}] validation checkpoint selection enabled  seed={select_seed}")
    if use_swa:
        flush(f"  [KD_{dname}] SWA enabled  start={swa_start}  every={swa_every}")

    for p in calibrated.parameters():
        p.requires_grad_(False)

    params = [
        calibrated.domain_alpha,
        calibrated.domain.ln.weight,
        calibrated.domain.ln.bias,
        calibrated.proj_out.weight,
    ]
    if refine_proj_in:
        params.insert(0, calibrated.proj_in.weight)
    bias_params = []
    if use_logit_bias:
        calibrated.logit_bias.requires_grad_(True)
        bias_params.append(calibrated.logit_bias)
    scale_params = []
    if use_logit_scale:
        calibrated.logit_scale.requires_grad_(True)
        scale_params.append(calibrated.logit_scale)
    residual_params = []
    if use_abi_residual:
        for p in list(calibrated.abi_resid_down.parameters()) + list(calibrated.abi_resid_up.parameters()):
            p.requires_grad_(True)
            residual_params.append(p)
    for p in params:
        p.requires_grad_(True)

    opt_groups = [{"params": params, "lr": LR_CAL, "weight_decay": 0.01}]
    if bias_params:
        opt_groups.append({
            "params": bias_params,
            "lr": LR_CAL * logit_bias_lr_mult,
            "weight_decay": 0.0,
        })
    if scale_params:
        opt_groups.append({
            "params": scale_params,
            "lr": LR_CAL * logit_scale_lr_mult,
            "weight_decay": 0.0,
        })
    if residual_params:
        opt_groups.append({
            "params": residual_params,
            "lr": LR_CAL * abi_residual_lr_mult,
            "weight_decay": 0.0,
        })
    opt = torch.optim.AdamW(opt_groups)
    native.eval()
    calibrated.train()
    swa_param_names = ["domain_alpha", "domain.ln.weight", "domain.ln.bias", "proj_out.weight"]
    if refine_proj_in:
        swa_param_names.insert(0, "proj_in.weight")
    if use_logit_bias:
        swa_param_names.append("logit_bias")
    if use_logit_scale:
        swa_param_names.append("logit_scale")
    if use_abi_residual:
        swa_param_names.extend(["abi_resid_down.weight", "abi_resid_up.weight"])

    for step in range(steps):
        x, y = make_batch(domain_ids, seed=31000 + step)
        opt.zero_grad()
        cal_logits = atlas_forward(calibrated, x, use_domain=True)
        with torch.no_grad():
            nat_logits = atlas_forward(native, x, use_domain=True)
        V = cal_logits.shape[-1]
        kd_loss = F.kl_div(
            F.log_softmax(cal_logits.reshape(-1, V) / kd_temp, dim=-1),
            F.softmax(nat_logits.reshape(-1, V) / kd_temp, dim=-1),
            reduction="batchmean",
        ) * (kd_temp ** 2)
        ce_loss = F.cross_entropy(cal_logits.reshape(-1, V), y.reshape(-1))
        loss = kd_weight * kd_loss + ce_weight * ce_loss
        if rank_topk_weight and (step + 1) >= rank_topk_start:
            with torch.no_grad():
                nat_topk = nat_logits.topk(RANK_TOPK_K, dim=-1).indices
                cal_cutoff = cal_logits.detach().topk(
                    RANK_TOPK_K, dim=-1).values[..., -1:].contiguous()
            cal_native_topk = cal_logits.gather(-1, nat_topk)
            rank_loss = F.relu(
                RANK_TOPK_MARGIN + cal_cutoff - cal_native_topk).mean()
            loss = loss + rank_topk_weight * rank_loss
        if topk_set_weight:
            with torch.no_grad():
                nat_topk = nat_logits.topk(TOPK_SET_K, dim=-1).indices
            cal_logp = F.log_softmax(cal_logits, dim=-1)
            topk_log_mass = torch.logsumexp(cal_logp.gather(-1, nat_topk), dim=-1)
            topk_set_loss = -topk_log_mass.mean()
            loss = loss + topk_set_weight * topk_set_loss
        if topk_ce_weight and (step + 1) >= topk_ce_start:
            with torch.no_grad():
                nat_topk = nat_logits.topk(TOPK_CE_K, dim=-1).indices
                nat_topk_logits = nat_logits.gather(-1, nat_topk)
                topk_target = F.softmax(nat_topk_logits / topk_ce_temp, dim=-1)
            cal_logp = F.log_softmax(cal_logits / topk_ce_temp, dim=-1)
            cal_topk_logp = cal_logp.gather(-1, nat_topk)
            topk_ce_loss = -(topk_target * cal_topk_logp).sum(-1).mean() * (topk_ce_temp ** 2)
            loss = loss + topk_ce_weight * topk_ce_loss
        if local_topk_kl_weight and (step + 1) >= local_topk_kl_start:
            with torch.no_grad():
                nat_topk = nat_logits.topk(LOCAL_TOPK_KL_K, dim=-1).indices
                nat_local_logits = nat_logits.gather(-1, nat_topk)
                nat_local_target = F.softmax(
                    nat_local_logits.reshape(-1, LOCAL_TOPK_KL_K) / local_topk_kl_temp,
                    dim=-1,
                )
            cal_local_logits = cal_logits.gather(-1, nat_topk)
            local_topk_kl_loss = F.kl_div(
                F.log_softmax(
                    cal_local_logits.reshape(-1, LOCAL_TOPK_KL_K) / local_topk_kl_temp,
                    dim=-1,
                ),
                nat_local_target,
                reduction="batchmean",
            ) * (local_topk_kl_temp ** 2)
            loss = loss + local_topk_kl_weight * local_topk_kl_loss
        if union_topk_kl_weight and (step + 1) >= union_topk_kl_start:
            with torch.no_grad():
                nat_topk = nat_logits.topk(UNION_TOPK_NATIVE_K, dim=-1).indices
                cal_topk = cal_logits.detach().topk(
                    UNION_TOPK_CANDIDATE_K, dim=-1).indices
                union_idx = torch.cat([nat_topk, cal_topk], dim=-1)
                nat_union_logits = nat_logits.gather(-1, union_idx)
                union_target = F.softmax(
                    nat_union_logits.reshape(-1, union_idx.shape[-1]) / union_topk_kl_temp,
                    dim=-1,
                )
            cal_union_logits = cal_logits.gather(-1, union_idx)
            union_topk_kl_loss = F.kl_div(
                F.log_softmax(
                    cal_union_logits.reshape(-1, union_idx.shape[-1]) / union_topk_kl_temp,
                    dim=-1,
                ),
                union_target,
                reduction="batchmean",
            ) * (union_topk_kl_temp ** 2)
            loss = loss + union_topk_kl_weight * union_topk_kl_loss
        if boundary_weight and (step + 1) >= boundary_start:
            with torch.no_grad():
                pos_idx = nat_logits.topk(BOUNDARY_TOPK_POS, dim=-1).indices
                neg_idx = cal_logits.detach().topk(BOUNDARY_TOPK_NEG, dim=-1).indices
                same = pos_idx.unsqueeze(-1).eq(neg_idx.unsqueeze(-2))
            pos_logits = cal_logits.gather(-1, pos_idx).unsqueeze(-1)
            neg_logits = cal_logits.gather(-1, neg_idx).unsqueeze(-2)
            pair_loss = F.softplus(neg_logits - pos_logits + BOUNDARY_TOPK_MARGIN)
            pair_loss = pair_loss.masked_fill(same, 0.0)
            denom = (~same).sum().clamp_min(1)
            boundary_loss = pair_loss.sum() / denom
            loss = loss + boundary_weight * boundary_loss
        if intruder_weight and (step + 1) >= intruder_start:
            with torch.no_grad():
                pos_idx = nat_logits.topk(INTRUDER_TOPK_POS, dim=-1).indices
                cand_idx = cal_logits.detach().topk(INTRUDER_TOPK_NEG, dim=-1).indices
                is_native = cand_idx.unsqueeze(-2).eq(pos_idx.unsqueeze(-1)).any(dim=-2)
            pos_logits = cal_logits.gather(-1, pos_idx)
            cutoff = pos_logits.min(dim=-1, keepdim=True).values
            intruder_logits = cal_logits.gather(-1, cand_idx)
            intruder_loss = F.relu(INTRUDER_TOPK_MARGIN + intruder_logits - cutoff)
            intruder_loss = intruder_loss.masked_fill(is_native, 0.0)
            denom = (~is_native).sum().clamp_min(1)
            loss = loss + intruder_weight * (intruder_loss.sum() / denom)
        if use_logit_bias and logit_bias_l2:
            loss = loss + logit_bias_l2 * calibrated.logit_bias.float().pow(2).mean()
        if use_logit_scale and logit_scale_l2:
            loss = loss + logit_scale_l2 * calibrated.logit_scale.float().pow(2).mean()
        if use_abi_residual and abi_residual_l2:
            resid_l2 = calibrated.abi_resid_down.weight.float().pow(2).mean()
            resid_l2 = resid_l2 + calibrated.abi_resid_up.weight.float().pow(2).mean()
            loss = loss + abi_residual_l2 * resid_l2
        loss.backward()
        nn.utils.clip_grad_norm_(params + bias_params + scale_params + residual_params, 1.0)
        opt.step()

        done = step + 1
        if done in checkpoint_set:
            calibrated.eval()
            eval_seed = select_seed if select_checkpoint else 7777
            res = l2_eval(native, calibrated, domain_ids, seed=eval_seed)
            ppl_ckpt = model_ppl(calibrated, domain_ids)
            score = l2_selection_score(res)
            checkpoint_history.append({
                "step": done,
                "eval_seed": eval_seed,
                "ppl": round(ppl_ckpt, 2),
                "l2": res,
                "selection_score": list(score),
            })
            if select_checkpoint and (best_checkpoint is None or score > best_checkpoint["score_tuple"]):
                best_checkpoint = {
                    "step": done,
                    "score_tuple": score,
                    "score": list(score),
                    "ppl": round(ppl_ckpt, 2),
                    "l2": res,
                }
                best_checkpoint_state = {
                    k: v.detach().cpu().clone()
                    for k, v in calibrated.state_dict().items()
                }
            if use_swa and done >= swa_start and ((done - swa_start) % swa_every == 0 or done == steps):
                for name in swa_param_names:
                    tensor = get_nested_attr(calibrated, name).detach().cpu().float()
                    if name not in swa_state:
                        swa_state[name] = tensor.clone()
                    else:
                        swa_state[name].mul_(swa_count / (swa_count + 1.0)).add_(
                            tensor / (swa_count + 1.0))
                swa_count += 1
                swa_info["count"] = swa_count
                swa_info["steps"].append(done)
            flush(f"  [KD_{dname}] step={done}  ppl={ppl_ckpt:.2f}  "
                  f"JS={res['mean_js']:.5f}  top1={res['top1']:.4f}  "
                  f"top5={res['top5']:.4f}  pass={res['pass']}")
            calibrated.train()

    calibrated.eval()
    if use_swa and swa_state:
        with torch.no_grad():
            for name, tensor in swa_state.items():
                param = get_nested_attr(calibrated, name)
                param.data.copy_(tensor.to(param.device, dtype=param.dtype))
        flush(f"  [KD_{dname}] loaded SWA chart from steps={swa_info['steps']}")
    if select_checkpoint and best_checkpoint_state is not None:
        calibrated.load_state_dict({
            k: v.to(DEVICE) for k, v in best_checkpoint_state.items()
        }, strict=False)
        selected_info = {
            "enabled": True,
            "seed": select_seed,
            "step": best_checkpoint["step"],
            "score": best_checkpoint["score"],
            "ppl": best_checkpoint["ppl"],
            "l2": best_checkpoint["l2"],
        }
        flush(f"  [KD_{dname}] selected checkpoint step={selected_info['step']}  "
              f"val_top5={selected_info['l2']['top5']:.4f}  "
              f"val_pass={selected_info['l2']['pass']}")
    selected_info["swa"] = swa_info
    for p in calibrated.parameters():
        p.requires_grad_(False)

    ppl_ref = model_ppl(calibrated, domain_ids)
    flush(f"  [KD_{dname}] ppl_ref={ppl_ref:.2f}  ({time.time()-t0:.0f}s)")
    return calibrated, ppl_ref, checkpoint_history, selected_info


# ===============================================================================
# L2 evaluation (matches NIB l2_logit_test exactly)
# ===============================================================================

@torch.no_grad()
def l2_eval(native: SVGPT2, candidate: SVGPT2,
            domain_ids: torch.Tensor,
            n_chunks: int = N_L2_CHUNKS,
            seed: int = 7777) -> dict:
    """
    Evaluates L2 parity between native and candidate on domain_ids.
    Matches NIB l2_logit_test: JS divergence, top-1 frac, top-5 overlap fraction,
    entropy diff, with SKIP=20 leading positions.
    """
    native.eval(); candidate.eval()
    CHUNK = 512
    SKIP  = L2_SKIP
    rng   = np.random.default_rng(seed)
    js_list, top1_list, top5_list, top10_list = [], [], [], []
    nat5_in_cal10_list, cal_mass_nat5_list, nat_mass_cal5_list, ent_list = [], [], [], []
    nat_margin_5_6_list, cal_margin_5_6_list, missed_nat5_gap_list = [], [], []

    max_start = max(len(domain_ids) - CHUNK, 1)
    for ci in range(n_chunks):
        start = int(rng.integers(0, max_start))
        chunk = domain_ids[start:start+CHUNK].unsqueeze(0).to(DEVICE)

        nat_logits = atlas_forward(native, chunk, use_domain=True)[0, SKIP:, :]     # (T-S, V)
        cal_logits = atlas_forward(candidate, chunk, use_domain=True)[0, SKIP:, :]

        nat_p = F.softmax(nat_logits, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_logits, dim=-1).cpu().float().numpy()
        T = nat_p.shape[0]
        eps = 1e-12

        # JS divergence. np.where evaluates both branches, so clip before
        # logging to avoid harmless divide-by-zero warnings from zero probs.
        m     = 0.5 * (nat_p + cal_p)
        nat_safe = np.clip(nat_p, eps, 1.0)
        cal_safe = np.clip(cal_p, eps, 1.0)
        m_safe   = np.clip(m, eps, 1.0)
        kl_nm = (nat_p * (np.log(nat_safe) - np.log(m_safe))).sum(1)
        kl_cm = (cal_p * (np.log(cal_safe) - np.log(m_safe))).sum(1)
        js    = np.clip(0.5 * (kl_nm + kl_cm), 0, None)
        js_list.extend(js.tolist())

        # Top-1 agreement
        top1_list.extend((nat_p.argmax(1) == cal_p.argmax(1)).tolist())

        # Top-5 overlap fraction (inter/5)
        nat5 = np.argpartition(nat_p, -5, axis=1)[:, -5:]
        cal5 = np.argpartition(cal_p, -5, axis=1)[:, -5:]
        nat10 = np.argpartition(nat_p, -10, axis=1)[:, -10:]
        cal10 = np.argpartition(cal_p, -10, axis=1)[:, -10:]
        nat_sorted10 = np.sort(np.partition(nat_p, -10, axis=1)[:, -10:], axis=1)[:, ::-1]
        cal_sorted10 = np.sort(np.partition(cal_p, -10, axis=1)[:, -10:], axis=1)[:, ::-1]
        nat_margin_5_6_list.extend((nat_sorted10[:, 4] - nat_sorted10[:, 5]).tolist())
        cal_margin_5_6_list.extend((cal_sorted10[:, 4] - cal_sorted10[:, 5]).tolist())
        for t in range(T):
            nat5_set = set(nat5[t])
            cal5_set = set(cal5[t])
            cal10_set = set(cal10[t])
            inter = len(nat5_set & cal5_set)
            top5_list.append(inter / 5.0)
            top10_list.append(len(set(nat10[t]) & cal10_set) / 10.0)
            nat5_in_cal10_list.append(len(nat5_set & cal10_set) / 5.0)
            cal_mass_nat5_list.append(float(cal_p[t, nat5[t]].sum()))
            nat_mass_cal5_list.append(float(nat_p[t, cal5[t]].sum()))
            missed = list(nat5_set - cal5_set)
            if missed:
                cutoff = np.partition(cal_p[t], -5)[-5]
                missed_nat5_gap_list.extend((cutoff - cal_p[t, missed]).tolist())

        # Entropy difference
        H_nat = -(nat_p * np.log(nat_p + eps)).sum(1)
        H_cal = -(cal_p * np.log(cal_p + eps)).sum(1)
        ent_list.extend(np.abs(H_nat - H_cal).tolist())

    mean_js  = float(np.mean(js_list))
    mean_top1 = float(np.mean(top1_list))
    mean_top5 = float(np.mean(top5_list))
    mean_top10 = float(np.mean(top10_list))
    mean_nat5_in_cal10 = float(np.mean(nat5_in_cal10_list))
    mean_cal_mass_nat5 = float(np.mean(cal_mass_nat5_list))
    mean_nat_mass_cal5 = float(np.mean(nat_mass_cal5_list))
    mean_ent  = float(np.mean(ent_list))
    mean_nat_margin_5_6 = float(np.mean(nat_margin_5_6_list))
    mean_cal_margin_5_6 = float(np.mean(cal_margin_5_6_list))
    mean_missed_nat5_gap = float(np.mean(missed_nat5_gap_list)) if missed_nat5_gap_list else 0.0

    js_pass   = mean_js   <  REGISTRY["js_threshold"]
    top1_pass = mean_top1 >= REGISTRY["top1_threshold"]
    top5_pass = mean_top5 >= REGISTRY["top5_threshold"]
    ent_pass  = mean_ent  <  REGISTRY["entropy_diff_threshold"]
    passes = js_pass and top1_pass and top5_pass and ent_pass

    return {
        "mean_js": round(mean_js, 5),
        "top1":    round(mean_top1, 4),
        "top5":    round(mean_top5, 4),
        "top10":   round(mean_top10, 4),
        "native_top5_in_candidate_top10": round(mean_nat5_in_cal10, 4),
        "candidate_mass_on_native_top5": round(mean_cal_mass_nat5, 4),
        "native_mass_on_candidate_top5": round(mean_nat_mass_cal5, 4),
        "native_margin_5_6": round(mean_nat_margin_5_6, 8),
        "candidate_margin_5_6": round(mean_cal_margin_5_6, 8),
        "missed_native_top5_gap": round(mean_missed_nat5_gap, 8),
        "entropy": round(mean_ent, 4),
        "js_pass": bool(js_pass),
        "top1_pass": bool(top1_pass),
        "top5_pass": bool(top5_pass),
        "entropy_pass": bool(ent_pass),
        "pass":    passes,
    }


# ===============================================================================
# Calibrated-model builder from saved per-domain projection weights
# ===============================================================================

def _extract_proj_weights(cal: SVGPT2) -> dict:
    weights = {
        "proj_in_w":     cal.proj_in.weight.data.cpu().clone(),
        "proj_out_w":    cal.proj_out.weight.data.cpu().clone(),
        "domain_alpha":  cal.domain_alpha.data.cpu().clone(),
        "domain_ln_w":   cal.domain.ln.weight.data.cpu().clone(),
        "domain_ln_b":   cal.domain.ln.bias.data.cpu().clone(),
    }
    bias = getattr(cal, "logit_bias", None)
    if bias is not None:
        weights["logit_bias"] = bias.data.cpu().clone()
    scale = getattr(cal, "logit_scale", None)
    if scale is not None:
        weights["logit_scale"] = scale.data.cpu().clone()
    abi_resid_down = getattr(cal, "abi_resid_down", None)
    abi_resid_up = getattr(cal, "abi_resid_up", None)
    if abi_resid_down is not None and abi_resid_up is not None:
        weights["abi_resid_down_w"] = abi_resid_down.weight.data.cpu().clone()
        weights["abi_resid_up_w"] = abi_resid_up.weight.data.cpu().clone()
        weights["abi_resid_scale"] = float(getattr(cal, "abi_resid_scale", 1.0))
    return weights


def _build_from_weights(transferred_state: dict, w: dict) -> SVGPT2:
    m = SVGPT2().to(DEVICE)
    m.load_state_dict(transferred_state)
    if "proj_in_w" in w:
        m.proj_in.weight.data.copy_(w["proj_in_w"].to(DEVICE))
    m.proj_out.weight.data.copy_(w["proj_out_w"].to(DEVICE))
    m.domain_alpha.data.copy_(w["domain_alpha"].to(DEVICE))
    m.domain.ln.weight.data.copy_(w["domain_ln_w"].to(DEVICE))
    m.domain.ln.bias.data.copy_(w["domain_ln_b"].to(DEVICE))
    if "logit_bias" in w:
        m.logit_bias = nn.Parameter(w["logit_bias"].to(DEVICE), requires_grad=False)
    if "logit_scale" in w:
        m.logit_scale = nn.Parameter(w["logit_scale"].to(DEVICE), requires_grad=False)
    if "abi_resid_down_w" in w and "abi_resid_up_w" in w:
        rank = int(w["abi_resid_down_w"].shape[0])
        m.abi_resid_down = nn.Linear(D_ABI, rank, bias=False).to(DEVICE)
        m.abi_resid_up = nn.Linear(rank, VOCAB_SIZE, bias=False).to(DEVICE)
        m.abi_resid_down.weight.data.copy_(w["abi_resid_down_w"].to(DEVICE))
        m.abi_resid_up.weight.data.copy_(w["abi_resid_up_w"].to(DEVICE))
        m.abi_resid_scale = float(w.get("abi_resid_scale", 1.0))
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


# ===============================================================================
# Soft mixture
# ===============================================================================

def build_mixture(transferred_state: dict,
                  w_a: dict, w_b: dict,
                  alpha: float) -> SVGPT2:
    """
    proj_out = alpha * proj_out_a  +  (1-alpha) * proj_out_b
    Other ABI params follow proj_a (arbitrary; the test is about proj_out rotation).
    """
    mix = SVGPT2().to(DEVICE)
    mix.load_state_dict(transferred_state)
    if "proj_in_w" in w_a and "proj_in_w" in w_b:
        pia = w_a["proj_in_w"].float().to(DEVICE)
        pib = w_b["proj_in_w"].float().to(DEVICE)
        mix.proj_in.weight.data.copy_((alpha * pia + (1.0 - alpha) * pib)
                                       .to(mix.proj_in.weight.dtype))
    wa = w_a["proj_out_w"].float().to(DEVICE)
    wb = w_b["proj_out_w"].float().to(DEVICE)
    mix.proj_out.weight.data.copy_((alpha * wa + (1.0 - alpha) * wb)
                                    .to(mix.proj_out.weight.dtype))
    mix.domain_alpha.data.copy_(w_a["domain_alpha"].to(DEVICE))
    mix.domain.ln.weight.data.copy_(w_a["domain_ln_w"].to(DEVICE))
    mix.domain.ln.bias.data.copy_(w_a["domain_ln_b"].to(DEVICE))
    if "logit_bias" in w_a and "logit_bias" in w_b:
        ba = w_a["logit_bias"].float().to(DEVICE)
        bb = w_b["logit_bias"].float().to(DEVICE)
        mix.logit_bias = nn.Parameter((alpha * ba + (1.0 - alpha) * bb), requires_grad=False)
    elif "logit_bias" in w_a:
        mix.logit_bias = nn.Parameter(w_a["logit_bias"].to(DEVICE), requires_grad=False)
    elif "logit_bias" in w_b:
        mix.logit_bias = nn.Parameter(w_b["logit_bias"].to(DEVICE), requires_grad=False)
    if "logit_scale" in w_a and "logit_scale" in w_b:
        sa = w_a["logit_scale"].float().to(DEVICE)
        sb = w_b["logit_scale"].float().to(DEVICE)
        mix.logit_scale = nn.Parameter((alpha * sa + (1.0 - alpha) * sb), requires_grad=False)
    elif "logit_scale" in w_a:
        mix.logit_scale = nn.Parameter(w_a["logit_scale"].to(DEVICE), requires_grad=False)
    elif "logit_scale" in w_b:
        mix.logit_scale = nn.Parameter(w_b["logit_scale"].to(DEVICE), requires_grad=False)
    if "abi_resid_down_w" in w_a and "abi_resid_up_w" in w_a:
        rank = int(w_a["abi_resid_down_w"].shape[0])
        mix.abi_resid_down = nn.Linear(D_ABI, rank, bias=False).to(DEVICE)
        mix.abi_resid_up = nn.Linear(rank, VOCAB_SIZE, bias=False).to(DEVICE)
        mix.abi_resid_down.weight.data.copy_(w_a["abi_resid_down_w"].to(DEVICE))
        mix.abi_resid_up.weight.data.copy_(w_a["abi_resid_up_w"].to(DEVICE))
        mix.abi_resid_scale = float(w_a.get("abi_resid_scale", 1.0))
    mix.eval()
    for p in mix.parameters(): p.requires_grad_(False)
    return mix


# ===============================================================================
# Main
# ===============================================================================

def main():
    t_global = time.time()
    banner("MULTI-DOMAIN ATLAS  --  Domain-Indexed Procrustes Routing")
    flush(f"  device: {DEVICE}  |  D_ABI={D_ABI}")

    # ── Load tokenizer ─────────────────────────────────────────────────────
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    # We tokenize whole corpora, then sample 128-token windows manually.
    # Avoid the misleading tokenizer max-length warning on corpus tokenization.
    tok.model_max_length = sys.maxsize

    # ── Load corpora ───────────────────────────────────────────────────────
    banner("Loading Corpora")
    flush("  [data] Python...")
    py_ids   = load_py_ids(tok)
    flush(f"  py_ids   = {len(py_ids):,} tokens")

    flush("  [data] WikiText...")
    wiki_ids = load_wiki_ids(tok)
    flush(f"  wiki_ids = {len(wiki_ids):,} tokens")

    flush("  [data] Markdown...")
    md_ids   = load_md_ids(tok)
    flush(f"  md_ids   = {len(md_ids):,} tokens")

    flush("  [data] SQL (synthetic)...")
    sql_ids  = load_sql_ids(tok)
    flush(f"  sql_ids  = {len(sql_ids):,} tokens")

    DOMAIN_IDS = {
        "python":   py_ids,
        "wikitext": wiki_ids,
        "markdown": md_ids,
        "sql":      sql_ids,
    }
    domain_list = list(DOMAIN_IDS.keys())
    domain_filter = os.environ.get("ATLAS_DOMAIN_FILTER", "").strip()
    if domain_filter:
        requested = [d.strip() for d in domain_filter.split(",") if d.strip()]
        unknown = sorted(set(requested) - set(DOMAIN_IDS))
        if unknown:
            raise ValueError(f"Unknown ATLAS_DOMAIN_FILTER domain(s): {unknown}")
        DOMAIN_IDS = {d: DOMAIN_IDS[d] for d in requested}
        domain_list = list(DOMAIN_IDS.keys())
        flush(f"  [filter] ATLAS_DOMAIN_FILTER={domain_filter} -> {domain_list}")

    # ── Step A: anchor (Python, ABI only) ──────────────────────────────────
    banner("Step A  --  Anchor (Python, ABI-only, 500 steps)")
    t1 = time.time()
    anchor = SVGPT2().to(DEVICE)
    for p in anchor.parameters(): p.requires_grad_(False)
    for nm, p in anchor.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_a = torch.optim.AdamW([p for p in anchor.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    anchor.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(py_ids, seed=1000 + step)
        opt_a.zero_grad()
        F.cross_entropy(anchor(x).reshape(-1, 50257), y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(anchor.parameters(), 1.0)
        opt_a.step()
    anchor.eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    flush(f"  [A] {time.time()-t1:.0f}s  ppl={model_ppl(anchor, py_ids):.2f}")

    # ── Step B: backbone drift (WikiText) ──────────────────────────────────
    banner("Step B  --  Backbone drift (WikiText, 1000 steps)")
    t2 = time.time()
    transferred = copy.deepcopy(anchor).to(DEVICE)
    saved_dom   = copy.deepcopy(transferred.domain.state_dict())
    for p in transferred.parameters(): p.requires_grad_(False)
    for nm, p in transferred.named_parameters():
        if "backbone" in nm or "proj_in" in nm or "abi_ln" in nm:
            p.requires_grad_(True)
    opt_b = torch.optim.AdamW([p for p in transferred.parameters() if p.requires_grad],
                               lr=LR_BACKBONE, weight_decay=0.01)
    transferred.train()
    for step in range(UPDATE_STEPS):
        x, y = make_batch(wiki_ids, seed=9000 + step)
        opt_b.zero_grad()
        h, h_abi = transferred.encode_core(x)
        logits = transferred.lm_head(transferred.proj_out(h_abi) + h)
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
    transferred_state = copy.deepcopy(transferred.state_dict())
    del anchor; torch.cuda.empty_cache()
    flush(f"  [B] {time.time()-t2:.0f}s")

    # ── Baseline PPL of transferred model on all domains ───────────────────
    banner("Transferred Baseline PPL")
    base_ppl = {}
    for dname, dids in DOMAIN_IDS.items():
        base_ppl[dname] = round(model_ppl(transferred, dids), 2)
    flush(f"  base_ppl = {base_ppl}")

    # ── Per-domain C (native oracle) + Procrustes solve ────────────────────
    banner("Per-Domain  C (native oracle)  +  Procrustes solve")

    # We store only the projection weights to keep GPU memory low.
    # Each calibrated model is rebuilt on demand during evaluation.
    cal_weights  = {}   # dname -> {proj_out_w, domain_alpha, domain_ln_w, domain_ln_b}
    nat_states   = {}   # dname -> full state_dict (CPU tensors)
    r_sq_results = {}
    cond_results = {}
    ppl_procrustes_results = {}
    ppl_cal_results = {}
    refine_diag_history = {}
    refine_selection = {}
    refine_steps_actual = {d: REFINE_STEPS_BY_DOMAIN.get(d, REFINE_STEPS)
                           for d in domain_list}
    flush(f"  refine_steps_by_domain = {refine_steps_actual}")

    for dname, dids in DOMAIN_IDS.items():
        flush(f"\n  === DOMAIN: {dname.upper()} ===")
        native = train_native_oracle(transferred_state, dids, dname)
        cal, A_star, r_sq, cond, ppl_cal = procrustes_solve(
            transferred_state, native, dids, dname,
            n_collect=N_COLLECT_BY_DOMAIN.get(dname, N_COLLECT))
        ppl_procrustes_results[dname] = round(ppl_cal, 2)
        cal, ppl_cal, refine_history, selection_info = refine_chart_kd(
            cal,
            native,
            dids,
            dname,
            steps=refine_steps_actual[dname],
            eval_checkpoints=REFINE_L2_CHECKPOINTS_BY_DOMAIN.get(dname),
        )

        # Save native state_dict to CPU, delete GPU copy
        nat_states[dname]   = {k: v.cpu() for k, v in native.state_dict().items()}
        cal_weights[dname]  = _extract_proj_weights(cal)
        r_sq_results[dname] = r_sq
        cond_results[dname] = cond
        ppl_cal_results[dname] = round(ppl_cal, 2)
        refine_diag_history[dname] = refine_history
        refine_selection[dname] = selection_info
        del native, cal, A_star; torch.cuda.empty_cache()

    # ── PPL matrix: [applied_rotation x eval_corpus] ───────────────────────
    banner("PPL Matrix  (applied_rotation x eval_corpus)")
    flush(f"  {'':12}", end="")
    for d in domain_list: flush(f"  {d[:8]:>8}", end="")
    flush("")

    ppl_matrix = {}
    for rot_d in domain_list:
        cal = _build_from_weights(transferred_state, cal_weights[rot_d])
        ppl_matrix[rot_d] = {}
        row = f"  rot={rot_d:<8}"
        for eval_d, eval_ids in DOMAIN_IDS.items():
            p = round(model_ppl(cal, eval_ids), 2)
            ppl_matrix[rot_d][eval_d] = p
            row += f"  {p:>8.2f}"
        flush(row)
        del cal; torch.cuda.empty_cache()

    # ── Diagonal L2 + Interference matrix ─────────────────────────────────
    banner("Diagonal L2  +  Interference Matrix  (mean JS)")
    _hdr = "eval\\rot"
    flush(f"\n  {_hdr:12}", end="")
    for d in domain_list: flush(f"  {d[:8]:>8}", end="")
    flush("")

    interference = {ed: {} for ed in domain_list}
    diag_l2      = {}

    for eval_d in domain_list:
        flush(f"\n  evaluating on: {eval_d}...")
        # Load native oracle for eval_d onto GPU
        nat = SVGPT2().to(DEVICE)
        nat.load_state_dict({k: v.to(DEVICE) for k, v in nat_states[eval_d].items()})
        nat.eval()
        for p in nat.parameters(): p.requires_grad_(False)

        eval_ids = DOMAIN_IDS[eval_d]
        row = f"  {eval_d:<12}"
        for rot_d in domain_list:
            cal = _build_from_weights(transferred_state, cal_weights[rot_d])
            res = l2_eval(nat, cal, eval_ids)
            interference[eval_d][rot_d] = res
            if eval_d == rot_d:
                diag_l2[eval_d] = res
            row += f"  {res['mean_js']:.4f}"
            if eval_d == rot_d:
                row += "*"       # mark diagonal
            else:
                row += " "
            del cal; torch.cuda.empty_cache()
        flush(row)
        del nat; torch.cuda.empty_cache()

    flush("\n  (* = diagonal: correct rotation for this domain)")

    # ── Soft mixture alpha sweep: Python + WikiText ────────────────────────
    mixture_results = []
    if "python" in domain_list and "wikitext" in domain_list:
        banner("Soft Mixture Sweep  (alpha_py * proj_py  +  (1-alpha_py) * proj_wiki)")
        flush(f"  {'alpha_py':>8}  {'JS_py':>8}  {'py_ok':>6}  {'JS_wiki':>8}  {'wiki_ok':>7}")

        ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]

        nat_py   = SVGPT2().to(DEVICE)
        nat_py.load_state_dict({k: v.to(DEVICE) for k, v in nat_states["python"].items()})
        nat_py.eval()
        for p in nat_py.parameters(): p.requires_grad_(False)

        nat_wiki = SVGPT2().to(DEVICE)
        nat_wiki.load_state_dict({k: v.to(DEVICE) for k, v in nat_states["wikitext"].items()})
        nat_wiki.eval()
        for p in nat_wiki.parameters(): p.requires_grad_(False)

        for alpha in ALPHAS:
            mix = build_mixture(transferred_state,
                                cal_weights["python"],
                                cal_weights["wikitext"], alpha)
            r_py   = l2_eval(nat_py,   mix, py_ids)
            r_wiki = l2_eval(nat_wiki, mix, wiki_ids)
            ppl_py   = round(model_ppl(mix, py_ids),   2)
            ppl_wiki = round(model_ppl(mix, wiki_ids), 2)
            del mix; torch.cuda.empty_cache()

            ok_py   = "ok" if r_py["pass"]   else "--"
            ok_wiki = "ok" if r_wiki["pass"] else "--"
            flush(f"  {alpha:>8.2f}  {r_py['mean_js']:>8.5f}  {ok_py:>6}"
                  f"  {r_wiki['mean_js']:>8.5f}  {ok_wiki:>7}")
            mixture_results.append({
                "alpha_python":  alpha,
                "js_python":     round(r_py["mean_js"], 5),
                "top5_python":   round(r_py["top5"], 4),
                "js_wikitext":   round(r_wiki["mean_js"], 5),
                "top5_wikitext": round(r_wiki["top5"], 4),
                "ppl_python":    ppl_py,
                "ppl_wikitext":  ppl_wiki,
                "pass_python":   r_py["pass"],
                "pass_wikitext": r_wiki["pass"],
            })

        del nat_py, nat_wiki; torch.cuda.empty_cache()
    else:
        banner("Soft Mixture Sweep skipped")
        flush("  Requires both python and wikitext domains in ATLAS_DOMAIN_FILTER.")

    # ── Non-degradation analysis ────────────────────────────────────────────
    DEGRAD_FACTOR = 2.0   # PPL must stay within 2x of transferred baseline
    non_degrad = {}
    for rot_d in domain_list:
        non_degrad[rot_d] = {}
        for eval_d in domain_list:
            base = base_ppl[eval_d]
            val  = ppl_matrix[rot_d][eval_d]
            ratio = val / base
            non_degrad[rot_d][eval_d] = {
                "ppl": val, "ratio": round(ratio, 3),
                "pass": bool(ratio <= DEGRAD_FACTOR)
            }

    # ── VERDICT ────────────────────────────────────────────────────────────
    banner("RESULTS SUMMARY")
    n_diag_pass = sum(1 for v in diag_l2.values() if v["pass"])
    flush(f"\n  Diagonal L2 (self-parity, correct rotation per domain):  "
          f"{n_diag_pass}/{len(domain_list)} PASS")
    for dname in domain_list:
        r = diag_l2[dname]
        sym = "+" if r["pass"] else "x"
        flush(f"    [{sym}] {dname:<10}  JS={r['mean_js']:.5f}  "
              f"top5={r['top5']:.4f}  R²={r_sq_results[dname]:.5f}")

    flush(f"\n  Interference matrix (mean JS,  * = diagonal = own rotation):")
    _hdr2 = "eval\\rot"
    flush(f"  {_hdr2:12}", end="")
    for d in domain_list: flush(f"  {d[:8]:>8}", end="")
    flush("")
    for eval_d in domain_list:
        flush(f"  {eval_d:<12}", end="")
        for rot_d in domain_list:
            js = interference[eval_d][rot_d]["mean_js"]
            marker = "*" if eval_d == rot_d else " "
            flush(f"  {js:.4f}{marker}", end="")
        flush("")

    flush(f"\n  Soft mixture Pareto frontier (Python <-> WikiText):")
    if mixture_results:
        flush(f"  {'alpha_py':>8}  {'JS_py':>8}  {'py_ok':>6}  {'JS_wiki':>8}  {'wiki_ok':>7}")
        for r in mixture_results:
            ok_py   = "ok" if r["pass_python"]   else "--"
            ok_wiki = "ok" if r["pass_wikitext"] else "--"
            flush(f"  {r['alpha_python']:>8.2f}  {r['js_python']:>8.5f}  {ok_py:>6}"
                  f"  {r['js_wikitext']:>8.5f}  {ok_wiki:>7}")
    else:
        flush("  skipped (requires python and wikitext domains)")

    off_diag_js = [
        interference[ed][rd]["mean_js"]
        for ed in domain_list for rd in domain_list if ed != rd
    ]
    diag_js = [interference[d][d]["mean_js"] for d in domain_list]
    locality_ratio = (float(np.mean(off_diag_js)) / float(np.mean(diag_js))
                      if off_diag_js else None)
    if locality_ratio is None:
        flush(f"\n  Locality ratio  (mean off-diag JS / mean diag JS):  skipped")
    else:
        flush(f"\n  Locality ratio  (mean off-diag JS / mean diag JS):  {locality_ratio:.2f}x")
    flush(f"  (higher = more local; each domain's rotation is domain-specific)")

    # Check if any mixture alpha achieves dual parity
    dual_parity = [r for r in mixture_results
                   if r["pass_python"] and r["pass_wikitext"]]
    if not mixture_results:
        flush(f"\n  Dual-parity mixture check skipped with current domain filter.")
    elif dual_parity:
        flush(f"\n  DUAL PARITY ACHIEVED by soft mixture at alpha(s): "
              f"{[r['alpha_python'] for r in dual_parity]}")
    else:
        flush(f"\n  No single alpha achieves simultaneous L2 parity on both domains.")
        flush(f"  -> Confirms: domain charts are distinct;  "
              f"hard routing, not blending, is the right architecture.")

    runtime = (time.time() - t_global) / 60
    flush(f"\n  Total runtime: {runtime:.1f} min")

    # ── Save results ────────────────────────────────────────────────────────
    output = {
        "diagonal_l2":          diag_l2,
        "r_squared":            r_sq_results,
        "cond_numbers":         cond_results,
        "n_collect":            N_COLLECT,
        "n_collect_by_domain":  N_COLLECT_BY_DOMAIN,
        "lstsq_rcond_by_domain": LSTSQ_RCOND_BY_DOMAIN,
        "n_l2_chunks":          N_L2_CHUNKS,
        "sql_corpus_mode":      SQL_CORPUS_MODE,
        "refine_steps":         REFINE_STEPS,
        "refine_steps_by_domain": refine_steps_actual,
        "refine_l2_checkpoints_by_domain": REFINE_L2_CHECKPOINTS_BY_DOMAIN,
        "refine_diag_history":  refine_diag_history,
        "select_checkpoint_by_domain": SELECT_CHECKPOINT_BY_DOMAIN,
        "select_checkpoint_seed_by_domain": SELECT_CHECKPOINT_SEED_BY_DOMAIN,
        "swa_by_domain":       SWA_BY_DOMAIN,
        "swa_start_by_domain": SWA_START_BY_DOMAIN,
        "swa_every_by_domain": SWA_EVERY_BY_DOMAIN,
        "refine_selection":     refine_selection,
        "refine_proj_in":       REFINE_PROJ_IN,
        "refine_proj_in_by_domain": REFINE_PROJ_IN_BY_DOMAIN,
        "kd_weight_by_domain":  KD_WEIGHT_BY_DOMAIN,
        "kd_temp_by_domain":    KD_TEMP_BY_DOMAIN,
        "rank_topk_k":          RANK_TOPK_K,
        "rank_topk_margin":     RANK_TOPK_MARGIN,
        "rank_topk_weight_by_domain": RANK_TOPK_WEIGHT_BY_DOMAIN,
        "rank_topk_start_by_domain": RANK_TOPK_START_BY_DOMAIN,
        "topk_set_k":           TOPK_SET_K,
        "topk_set_weight_by_domain": TOPK_SET_WEIGHT_BY_DOMAIN,
        "topk_ce_k":            TOPK_CE_K,
        "topk_ce_weight_by_domain": TOPK_CE_WEIGHT_BY_DOMAIN,
        "topk_ce_temp_by_domain": TOPK_CE_TEMP_BY_DOMAIN,
        "topk_ce_start_by_domain": TOPK_CE_START_BY_DOMAIN,
        "local_topk_kl_k":      LOCAL_TOPK_KL_K,
        "local_topk_kl_weight_by_domain": LOCAL_TOPK_KL_WEIGHT_BY_DOMAIN,
        "local_topk_kl_temp_by_domain": LOCAL_TOPK_KL_TEMP_BY_DOMAIN,
        "local_topk_kl_start_by_domain": LOCAL_TOPK_KL_START_BY_DOMAIN,
        "union_topk_native_k": UNION_TOPK_NATIVE_K,
        "union_topk_candidate_k": UNION_TOPK_CANDIDATE_K,
        "union_topk_kl_weight_by_domain": UNION_TOPK_KL_WEIGHT_BY_DOMAIN,
        "union_topk_kl_temp_by_domain": UNION_TOPK_KL_TEMP_BY_DOMAIN,
        "union_topk_kl_start_by_domain": UNION_TOPK_KL_START_BY_DOMAIN,
        "boundary_topk_pos": BOUNDARY_TOPK_POS,
        "boundary_topk_neg": BOUNDARY_TOPK_NEG,
        "boundary_topk_margin": BOUNDARY_TOPK_MARGIN,
        "boundary_topk_weight_by_domain": BOUNDARY_TOPK_WEIGHT_BY_DOMAIN,
        "boundary_topk_start_by_domain": BOUNDARY_TOPK_START_BY_DOMAIN,
        "intruder_topk_pos": INTRUDER_TOPK_POS,
        "intruder_topk_neg": INTRUDER_TOPK_NEG,
        "intruder_topk_margin": INTRUDER_TOPK_MARGIN,
        "intruder_topk_weight_by_domain": INTRUDER_TOPK_WEIGHT_BY_DOMAIN,
        "intruder_topk_start_by_domain": INTRUDER_TOPK_START_BY_DOMAIN,
        "refine_logit_bias_by_domain": REFINE_LOGIT_BIAS_BY_DOMAIN,
        "logit_bias_lr_mult_by_domain": LOGIT_BIAS_LR_MULT_BY_DOMAIN,
        "logit_bias_l2_by_domain": LOGIT_BIAS_L2_BY_DOMAIN,
        "refine_logit_scale_by_domain": REFINE_LOGIT_SCALE_BY_DOMAIN,
        "logit_scale_lr_mult_by_domain": LOGIT_SCALE_LR_MULT_BY_DOMAIN,
        "logit_scale_l2_by_domain": LOGIT_SCALE_L2_BY_DOMAIN,
        "refine_abi_residual_by_domain": REFINE_ABI_RESIDUAL_BY_DOMAIN,
        "abi_residual_rank_by_domain": ABI_RESIDUAL_RANK_BY_DOMAIN,
        "abi_residual_scale_by_domain": ABI_RESIDUAL_SCALE_BY_DOMAIN,
        "abi_residual_lr_mult_by_domain": ABI_RESIDUAL_LR_MULT_BY_DOMAIN,
        "abi_residual_l2_by_domain": ABI_RESIDUAL_L2_BY_DOMAIN,
        "ppl_procrustes":       ppl_procrustes_results,
        "ppl_calibrated":       ppl_cal_results,
        "base_ppl":             base_ppl,
        "ppl_matrix":           ppl_matrix,
        "interference_matrix":  interference,
        "mixture_results":      mixture_results,
        "non_degradation":      non_degrad,
        "locality_ratio":       round(locality_ratio, 3) if locality_ratio is not None else None,
        "dual_parity_alphas":   [r["alpha_python"] for r in dual_parity],
        "n_diag_pass":          n_diag_pass,
        "n_domains":            len(domain_list),
        "runtime_min":          round(runtime, 1),
        "registry":             REGISTRY,
    }
    out_path = ROOT / "multi_domain_atlas_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    flush(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
