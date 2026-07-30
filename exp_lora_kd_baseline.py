#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched LoRA/KD baseline for ABI adoption gating.

This runner is intentionally a baseline, not an ABI transfer experiment:

1. Train the same native target ABI oracle used by the generic ABI/NIB runner.
2. Freeze a clean target backbone and inject LoRA into selected linear layers.
3. Train only LoRA parameters against the native target ABI oracle using the
   same KD/rank/top-set losses available to ABI calibration.
4. Evaluate the LoRA candidate with the same NIB L2 logit certificate.

The result answers a reviewer-facing question: at roughly the same target-side
trainable-parameter budget, does ordinary target-side PEFT match the ABI
certificate, or is ABI doing something materially different?
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


TARGET_MODEL_ID = os.environ.get(
    "ABI_BASELINE_TARGET_MODEL_ID",
    os.environ.get("ABI_TARGET_MODEL_ID", "Qwen/Qwen2.5-0.5B"),
)
TARGET_TOKENIZER_ID = os.environ.get(
    "ABI_BASELINE_TARGET_TOKENIZER_ID",
    os.environ.get("ABI_TARGET_TOKENIZER_ID", TARGET_MODEL_ID),
)
TARGET_LABEL = os.environ.get(
    "ABI_BASELINE_TARGET_LABEL",
    os.environ.get("ABI_TARGET_LABEL", slug(TARGET_MODEL_ID)),
)
TEACHER_D_ABI = int(
    os.environ.get("ABI_BASELINE_D_ABI", os.environ.get("ABI_D_ABI", "512"))
)
DOMAIN_STEPS = int(
    os.environ.get("ABI_BASELINE_DOMAIN_STEPS", os.environ.get("ABI_DOMAIN_STEPS", "500"))
)
CALIBRATION_STEPS = int(
    os.environ.get("ABI_BASELINE_CAL_STEPS", os.environ.get("ABI_CAL_STEPS", "1200"))
)
BATCH = int(os.environ.get("ABI_BASELINE_BATCH", os.environ.get("ABI_BATCH", "2")))
PPL_BATCHES = int(
    os.environ.get("ABI_BASELINE_PPL_BATCHES", os.environ.get("ABI_PPL_BATCHES", "50"))
)
EXPERIMENT_SEED = int(
    os.environ.get("ABI_BASELINE_SEED", os.environ.get("ABI_SEED", "42"))
)
SEED_OFFSET = int(os.environ.get("ABI_BASELINE_SEED_OFFSET", os.environ.get("ABI_SEED_OFFSET", "0")))
NATIVE_DOMAIN_SEED_BASE = int(
    os.environ.get("ABI_BASELINE_NATIVE_DOMAIN_SEED_BASE", str(5000 + SEED_OFFSET))
)
CAL_SEED_BASE = int(
    os.environ.get("ABI_BASELINE_CAL_SEED_BASE", str(7000 + SEED_OFFSET))
)
PPL_SEED_BASE = int(
    os.environ.get("ABI_BASELINE_PPL_SEED_BASE", str(80000 + SEED_OFFSET))
)
NIB_SEED = int(os.environ.get("ABI_BASELINE_NIB_SEED", str(7777 + SEED_OFFSET)))
DOMAIN_CORPUS = os.environ.get(
    "ABI_BASELINE_DOMAIN_CORPUS",
    os.environ.get("ABI_DOMAIN_CORPUS", "wikitext"),
).strip().lower()
WIKITEXT_DOMAIN_SPLIT = os.environ.get(
    "ABI_BASELINE_WIKITEXT_DOMAIN_SPLIT",
    os.environ.get("ABI_WIKITEXT_DOMAIN_SPLIT", "validation"),
).strip().lower()
WIKITEXT_POSTHOC_SPLIT = os.environ.get(
    "ABI_BASELINE_WIKITEXT_POSTHOC_SPLIT",
    os.environ.get("ABI_WIKITEXT_POSTHOC_SPLIT", WIKITEXT_DOMAIN_SPLIT),
).strip().lower()
WIKITEXT_EVAL_SPLIT = os.environ.get(
    "ABI_BASELINE_WIKITEXT_EVAL_SPLIT",
    os.environ.get("ABI_WIKITEXT_EVAL_SPLIT", WIKITEXT_DOMAIN_SPLIT),
).strip().lower()
LORA_RANK = int(os.environ.get("ABI_BASELINE_LORA_RANK", "8"))
LORA_ALPHA = float(os.environ.get("ABI_BASELINE_LORA_ALPHA", str(2 * LORA_RANK)))
LORA_TARGETS = os.environ.get("ABI_BASELINE_LORA_TARGETS", "attn").strip().lower()
LORA_DROPOUT = float(os.environ.get("ABI_BASELINE_LORA_DROPOUT", "0.0"))
LORA_DTYPE = os.environ.get("ABI_BASELINE_LORA_DTYPE", "float32").strip().lower()
KD_WEIGHT = float(os.environ.get("ABI_BASELINE_KD_WEIGHT", "0.90"))
KD_TEMP = float(os.environ.get("ABI_BASELINE_KD_TEMP", "2.0"))
LR = float(os.environ.get("ABI_BASELINE_LR", os.environ.get("ABI_LR_CAL", "1e-4")))
WEIGHT_DECAY = float(os.environ.get("ABI_BASELINE_WEIGHT_DECAY", "0.01"))
GRAD_CLIP = float(os.environ.get("ABI_BASELINE_GRAD_CLIP", "1.0"))
TRAIN_CE = env_bool("ABI_BASELINE_TRAIN_CE", True)
MAX_TRAIN_SECONDS_RAW = os.environ.get("ABI_BASELINE_MAX_TRAIN_SECONDS")
MAX_TRAIN_SECONDS = (
    float(MAX_TRAIN_SECONDS_RAW) if MAX_TRAIN_SECONDS_RAW not in {None, ""} else None
)

TOPK_KD_WEIGHT = float(os.environ.get("ABI_TOPK_KD_WEIGHT", "0.0"))
TOPK = int(os.environ.get("ABI_TOPK", "32"))
RANK_MARGIN_WEIGHT = float(os.environ.get("ABI_RANK_MARGIN_WEIGHT", "0.0"))
RANK_TOP_POS = int(os.environ.get("ABI_RANK_TOP_POS", "5"))
RANK_NEG_K = int(os.environ.get("ABI_RANK_NEG_K", str(max(32, RANK_TOP_POS))))
RANK_MARGIN = float(os.environ.get("ABI_RANK_MARGIN", "0.25"))
TOP1_GAP_WEIGHT = float(os.environ.get("ABI_TOP1_GAP_WEIGHT", "0.0"))
TOP1_GAP_K = int(os.environ.get("ABI_TOP1_GAP_K", "5"))
TOP1_GAP_MARGIN = float(os.environ.get("ABI_TOP1_GAP_MARGIN", "0.05"))
TOP1_CE_WEIGHT = float(os.environ.get("ABI_TOP1_CE_WEIGHT", "0.0"))
HARD_NEG_WEIGHT = float(os.environ.get("ABI_HARD_NEG_WEIGHT", "0.0"))
HARD_NEG_K = int(os.environ.get("ABI_HARD_NEG_K", "32"))
TOP1_HARD_NEG_WEIGHT = float(os.environ.get("ABI_TOP1_HARD_NEG_WEIGHT", "0.0"))
TOP1_HARD_NEG_K = int(os.environ.get("ABI_TOP1_HARD_NEG_K", str(HARD_NEG_K)))
TOPSET_WEIGHT = float(os.environ.get("ABI_TOPSET_WEIGHT", "0.0"))
TOPSET_K = int(os.environ.get("ABI_TOPSET_K", "5"))
TOPSET_TEMP = float(os.environ.get("ABI_TOPSET_TEMP", "1.0"))
ENTROPY_WEIGHT = float(os.environ.get("ABI_ENTROPY_WEIGHT", "0.0"))

if DOMAIN_CORPUS not in {"python", "wikitext"}:
    raise ValueError("ABI_BASELINE_DOMAIN_CORPUS must be one of: python, wikitext")

for split_name, split_value in {
    "ABI_BASELINE_WIKITEXT_DOMAIN_SPLIT": WIKITEXT_DOMAIN_SPLIT,
    "ABI_BASELINE_WIKITEXT_POSTHOC_SPLIT": WIKITEXT_POSTHOC_SPLIT,
    "ABI_BASELINE_WIKITEXT_EVAL_SPLIT": WIKITEXT_EVAL_SPLIT,
}.items():
    if split_value not in {"train", "validation", "test"}:
        raise ValueError(f"{split_name} must be one of: train, validation, test")

if LORA_RANK < 1:
    raise ValueError("ABI_BASELINE_LORA_RANK must be >= 1")

if LORA_TARGETS not in {"all", "attn", "qv", "mlp"}:
    raise ValueError("ABI_BASELINE_LORA_TARGETS must be one of: all, attn, qv, mlp")

if LORA_DTYPE not in {"base", "float32"}:
    raise ValueError("ABI_BASELINE_LORA_DTYPE must be one of: base, float32")

# Keep exp_generic_causal_nib_v2 configured to the same target/teacher settings
# before importing it. The import is used only for shared ABI/NIB utilities.
os.environ.setdefault("ABI_TARGET_MODEL_ID", TARGET_MODEL_ID)
os.environ.setdefault("ABI_TARGET_TOKENIZER_ID", TARGET_TOKENIZER_ID)
os.environ.setdefault("ABI_TARGET_LABEL", TARGET_LABEL)
os.environ.setdefault("ABI_D_ABI", str(TEACHER_D_ABI))
os.environ.setdefault("ABI_DOMAIN_STEPS", str(DOMAIN_STEPS))
os.environ.setdefault("ABI_CAL_STEPS", str(CALIBRATION_STEPS))
os.environ.setdefault("ABI_BATCH", str(BATCH))
os.environ.setdefault("ABI_PPL_BATCHES", str(PPL_BATCHES))
os.environ.setdefault("ABI_SEED", str(EXPERIMENT_SEED))
os.environ.setdefault("ABI_SEED_OFFSET", str(SEED_OFFSET))
os.environ.setdefault("ABI_NATIVE_DOMAIN_SEED_BASE", str(NATIVE_DOMAIN_SEED_BASE))
os.environ.setdefault("ABI_CAL_SEED_BASE", str(CAL_SEED_BASE))
os.environ.setdefault("ABI_PPL_SEED_BASE", str(PPL_SEED_BASE))
os.environ.setdefault("ABI_NIB_SEED", str(NIB_SEED))
os.environ.setdefault("ABI_DOMAIN_CORPUS", DOMAIN_CORPUS)
os.environ.setdefault("ABI_WIKITEXT_DOMAIN_SPLIT", WIKITEXT_DOMAIN_SPLIT)
os.environ.setdefault("ABI_WIKITEXT_POSTHOC_SPLIT", WIKITEXT_POSTHOC_SPLIT)
os.environ.setdefault("ABI_WIKITEXT_EVAL_SPLIT", WIKITEXT_EVAL_SPLIT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import exp_generic_causal_nib_v2 as generic

base = generic.base

sys.stdout.reconfigure(line_buffering=True)
torch.manual_seed(EXPERIMENT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(EXPERIMENT_SEED)


def banner(msg: str) -> None:
    print()
    print("=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def trainable_count(params) -> int:
    return int(sum(p.numel() for p in params))


class LoraLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoraLinear can only wrap nn.Linear modules")
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        device = self.base.weight.device
        dtype = torch.float32 if LORA_DTYPE == "float32" else self.base.weight.dtype
        self.lora_a = nn.Linear(
            self.base.in_features,
            self.rank,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lora_b = nn.Linear(
            self.rank,
            self.base.out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        base_out = self.base(x)
        lora_in = self.dropout(x).to(self.lora_a.weight.dtype)
        lora_out = self.lora_b(self.lora_a(lora_in)) * self.scale
        return base_out + lora_out.to(base_out.dtype)


def should_lora(name: str) -> bool:
    if name == "lm_head" or name.endswith(".lm_head") or ".lm_head." in name:
        return False
    if LORA_TARGETS == "all":
        return True
    is_attn = "self_attn" in name or ".attn." in name or "attn.attention" in name
    is_qv = name.endswith("q_proj") or name.endswith("v_proj")
    is_attn_proj = is_qv or name.endswith("k_proj") or name.endswith("o_proj") or name.endswith("out_proj")
    is_mlp = ".mlp." in name or name.endswith("c_fc") or name.endswith("c_proj")
    if LORA_TARGETS == "attn":
        return is_attn and is_attn_proj
    if LORA_TARGETS == "qv":
        return is_qv
    return is_mlp


def replace_module(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = module


def inject_lora(model: nn.Module) -> list[dict]:
    replacements = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and should_lora(name)
    ]
    injected = []
    for name, mod in replacements:
        wrapped = LoraLinear(mod, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
        replace_module(model, name, wrapped)
        injected.append(
            {
                "name": name,
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "trainable_params": trainable_count(wrapped.parameters()) - trainable_count(mod.parameters()),
            }
        )
    return injected


class LoraCausalLM(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        kwargs = {"local_files_only": True}
        if generic.TORCH_DTYPE is not None:
            kwargs["torch_dtype"] = generic.TORCH_DTYPE
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.config.use_cache = False
        self.vocab_size = generic.config_int(self.model.config, "vocab_size")
        self.model_type = getattr(self.model.config, "model_type", "unknown")
        self.target_param_count = trainable_count(self.model.parameters())
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.injected_lora = inject_lora(self.model)
        self.logit_scale_value = 1.0

    def forward(self, x, use_domain=True):
        del use_domain
        out = self.model(input_ids=x, use_cache=False, return_dict=True)
        logits = out.logits
        scale = float(getattr(self, "logit_scale_value", 1.0))
        if scale != 1.0:
            logits = logits * scale
        return logits


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params = []
    for module in model.modules():
        if isinstance(module, LoraLinear):
            params.extend([module.lora_a.weight, module.lora_b.weight])
    return params


def make_batch(tokens, seed: int):
    return generic.make_batch(tokens, seed=seed, batch=BATCH)


def load_split_tokens(tokenizer):
    py_text, py_meta = base.load_local_python_text(
        base.ROOT,
        base.MAX_PY,
        exclude_globs=generic.V2_CORPUS_EXCLUDES,
    )
    wiki_cache: dict[str, tuple[str, list[str], dict]] = {}

    def load_wiki(split):
        if split not in wiki_cache:
            wiki_cache[split] = base.load_wikitext_text_and_sentences(
                split=split,
                min_chars=20,
            )
        return wiki_cache[split]

    if DOMAIN_CORPUS == "python":
        domain_text = py_text
        posthoc_text = py_text
        eval_text = py_text
        domain_detail = (
            f"local_python_files={py_meta['files']} skipped={py_meta['skipped']}"
        )
        posthoc_detail = "same_as_domain"
        eval_detail = "same_as_domain"
        wiki_meta = {"records": 0, "split": None}
    else:
        domain_text, _domain_sentences, domain_meta = load_wiki(WIKITEXT_DOMAIN_SPLIT)
        posthoc_text, _posthoc_sentences, posthoc_meta = load_wiki(WIKITEXT_POSTHOC_SPLIT)
        eval_text, _eval_sentences, eval_meta = load_wiki(WIKITEXT_EVAL_SPLIT)
        domain_detail = (
            f"wikitext_split={domain_meta['split']} records={domain_meta['records']}"
        )
        posthoc_detail = (
            f"wikitext_split={posthoc_meta['split']} records={posthoc_meta['records']}"
        )
        eval_detail = (
            f"wikitext_split={eval_meta['split']} records={eval_meta['records']}"
        )
        wiki_meta = {
            "domain_records": domain_meta["records"],
            "posthoc_records": posthoc_meta["records"],
            "eval_records": eval_meta["records"],
        }
    domain_ids = tokenizer(
        domain_text, return_tensors="pt", truncation=False
    )["input_ids"].squeeze(0)[: base.MAX_PY]
    posthoc_ids = tokenizer(
        posthoc_text, return_tensors="pt", truncation=False
    )["input_ids"].squeeze(0)[: base.MAX_PY]
    eval_ids = tokenizer(
        eval_text, return_tensors="pt", truncation=False
    )["input_ids"].squeeze(0)[: base.MAX_PY]
    details = {
        "domain": domain_detail,
        "posthoc": posthoc_detail,
        "eval": eval_detail,
    }
    return domain_ids, posthoc_ids, eval_ids, details, py_meta, wiki_meta


def set_native_trainable(native: nn.Module) -> list[nn.Parameter]:
    for p in native.parameters():
        p.requires_grad_(False)
    params = []
    for name, p in native.named_parameters():
        if any(key in name for key in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
            params.append(p)
    return params


def add_calibration_losses(loss, student_logits, teacher_logits, vocab_size):
    student_flat = student_logits.reshape(-1, vocab_size).float()
    teacher_flat = teacher_logits.reshape(-1, vocab_size).float()
    if ENTROPY_WEIGHT > 0:
        student_logp = F.log_softmax(student_flat, dim=-1)
        student_p = student_logp.exp()
        with torch.no_grad():
            teacher_logp = F.log_softmax(teacher_flat, dim=-1)
            teacher_entropy = -(teacher_logp.exp() * teacher_logp).sum(dim=-1)
        student_entropy = -(student_p * student_logp).sum(dim=-1)
        loss = loss + ENTROPY_WEIGHT * F.mse_loss(student_entropy, teacher_entropy)
    if TOPK_KD_WEIGHT > 0:
        with torch.no_grad():
            top_idx = teacher_flat.topk(min(TOPK, vocab_size), dim=-1).indices
            top_teacher = teacher_flat.gather(1, top_idx)
        top_student = student_flat.gather(1, top_idx)
        topk_kd = F.kl_div(
            F.log_softmax(top_student / KD_TEMP, dim=-1),
            F.softmax(top_teacher / KD_TEMP, dim=-1),
            reduction="batchmean",
        ) * (KD_TEMP ** 2)
        loss = loss + TOPK_KD_WEIGHT * topk_kd
    if RANK_MARGIN_WEIGHT > 0:
        with torch.no_grad():
            rank_k = min(max(RANK_NEG_K, RANK_TOP_POS + 1), vocab_size)
            rank_idx = teacher_flat.topk(rank_k, dim=-1).indices
        rank_student = student_flat.gather(1, rank_idx)
        pos = rank_student[:, :RANK_TOP_POS]
        neg = rank_student[:, RANK_TOP_POS:]
        rank_loss = F.softplus(neg.unsqueeze(1) - pos.unsqueeze(2) + RANK_MARGIN).mean()
        loss = loss + RANK_MARGIN_WEIGHT * rank_loss
    if TOP1_GAP_WEIGHT > 0:
        with torch.no_grad():
            gap_k = min(max(TOP1_GAP_K, 2), vocab_size)
            gap_idx = teacher_flat.topk(gap_k, dim=-1).indices
        gap_student = student_flat.gather(1, gap_idx)
        top1 = gap_student[:, :1]
        next_choices = gap_student[:, 1:]
        top1_gap_loss = F.softplus(next_choices - top1 + TOP1_GAP_MARGIN).mean()
        loss = loss + TOP1_GAP_WEIGHT * top1_gap_loss
    if TOP1_CE_WEIGHT > 0:
        with torch.no_grad():
            teacher_argmax = teacher_flat.argmax(dim=-1)
        top1_ce = F.cross_entropy(student_flat, teacher_argmax)
        loss = loss + TOP1_CE_WEIGHT * top1_ce
    if HARD_NEG_WEIGHT > 0:
        with torch.no_grad():
            pos_idx = teacher_flat.topk(min(RANK_TOP_POS, vocab_size), dim=-1).indices
            cand_k = min(HARD_NEG_K + pos_idx.shape[1], vocab_size)
            cand_idx = student_flat.topk(cand_k, dim=-1).indices
            is_teacher_pos = cand_idx.unsqueeze(-1).eq(pos_idx.unsqueeze(1)).any(-1)
        pos = student_flat.gather(1, pos_idx)
        cand_logits = student_flat.gather(1, cand_idx).masked_fill(is_teacher_pos, -1.0e4)
        hard_neg = cand_logits.topk(min(HARD_NEG_K, cand_logits.shape[1]), dim=-1).values
        hard_neg_loss = F.softplus(hard_neg.unsqueeze(1) - pos.unsqueeze(2) + RANK_MARGIN).mean()
        loss = loss + HARD_NEG_WEIGHT * hard_neg_loss
    if TOP1_HARD_NEG_WEIGHT > 0:
        with torch.no_grad():
            top1_idx = teacher_flat.argmax(dim=-1, keepdim=True)
            cand_k = min(TOP1_HARD_NEG_K + 1, vocab_size)
            cand_idx = student_flat.topk(cand_k, dim=-1).indices
            is_teacher_top1 = cand_idx.eq(top1_idx)
        top1_pos = student_flat.gather(1, top1_idx)
        cand_logits = student_flat.gather(1, cand_idx).masked_fill(is_teacher_top1, -1.0e4)
        top1_hard_neg = cand_logits.topk(
            min(TOP1_HARD_NEG_K, cand_logits.shape[1]), dim=-1
        ).values
        top1_hard_neg_loss = F.softplus(top1_hard_neg - top1_pos + RANK_MARGIN).mean()
        loss = loss + TOP1_HARD_NEG_WEIGHT * top1_hard_neg_loss
    if TOPSET_WEIGHT > 0:
        with torch.no_grad():
            topset_k = min(TOPSET_K, vocab_size)
            topset_idx = teacher_flat.topk(topset_k, dim=-1).indices
            topset_teacher = teacher_flat.gather(1, topset_idx)
            topset_target = F.softmax(topset_teacher / TOPSET_TEMP, dim=-1)
        topset_logp = F.log_softmax(student_flat / TOPSET_TEMP, dim=-1).gather(1, topset_idx)
        topset_loss = -(topset_target * topset_logp).sum(dim=-1).mean()
        loss = loss + TOPSET_WEIGHT * (TOPSET_TEMP ** 2) * topset_loss
    return loss


def main() -> None:
    t_global = time.time()
    banner(f"LoRA/KD baseline: {TARGET_MODEL_ID}")
    print(f"  Device:          {base.DEVICE}")
    print(f"  Target:          {TARGET_MODEL_ID}")
    print(f"  Tokenizer:       {TARGET_TOKENIZER_ID}")
    print(f"  Native D_ABI:    {TEACHER_D_ABI}")
    print(f"  Domain corpus:   {DOMAIN_CORPUS}")
    if DOMAIN_CORPUS == "wikitext":
        print(
            f"  WikiText splits: train={WIKITEXT_DOMAIN_SPLIT}  "
            f"posthoc={WIKITEXT_POSTHOC_SPLIT}  eval={WIKITEXT_EVAL_SPLIT}"
        )
    print(f"  Domain steps:    {DOMAIN_STEPS}")
    print(f"  LoRA steps:      {CALIBRATION_STEPS}")
    if MAX_TRAIN_SECONDS is not None:
        print(f"  LoRA time cap:   {MAX_TRAIN_SECONDS:.0f}s")
    print(f"  LoRA rank/alpha: {LORA_RANK}/{LORA_ALPHA}")
    print(f"  LoRA targets:    {LORA_TARGETS}")
    print(f"  LoRA dtype:      {LORA_DTYPE}")
    print(f"  Batch:           {BATCH}")
    print(f"  KD weight/temp:  {KD_WEIGHT}/{KD_TEMP}")
    print(f"  Top-k/rank/topset: {TOPK_KD_WEIGHT}/{RANK_MARGIN_WEIGHT}/{TOPSET_WEIGHT}")
    print(f"  Hard/top1-hard:  {HARD_NEG_WEIGHT}/{TOP1_HARD_NEG_WEIGHT}")

    banner("Data loading")
    tok = AutoTokenizer.from_pretrained(generic.HF_TARGET_TOKENIZER, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    domain_ids, posthoc_ids, eval_ids, details, py_meta, wiki_meta = load_split_tokens(tok)
    print(
        f"  train_tokens={len(domain_ids):,}  "
        f"posthoc_tokens={len(posthoc_ids):,}  eval_tokens={len(eval_ids):,}"
    )
    print(
        f"  splits: train=({details['domain']})  "
        f"posthoc=({details['posthoc']})  eval=({details['eval']})"
    )

    banner(f"Phase C - Native {TARGET_LABEL} ABI oracle")
    t_c = time.time()
    native = generic.GenericCausalABI(generic.HF_TARGET).to(base.DEVICE)
    native_params = set_native_trainable(native)
    opt_native = torch.optim.AdamW(native_params, lr=base.LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(domain_ids, NATIVE_DOMAIN_SEED_BASE + step)
        opt_native.zero_grad()
        logits = native(x)
        loss = F.cross_entropy(logits.reshape(-1, native.vocab_size), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(native_params, 1.0)
        opt_native.step()
        if (step + 1) % 100 == 0:
            print(f"  C step {step + 1}/{DOMAIN_STEPS}  {time.time() - t_c:.0f}s")
    native.eval()
    for p in native.parameters():
        p.requires_grad_(False)
    ppl_native = generic.ppl(native, eval_ids)
    print(f"  Phase C complete: {time.time() - t_c:.0f}s  native ppl={ppl_native:.1f}")

    banner("Phase B - Target-side LoRA KD baseline")
    t_b = time.time()
    candidate = LoraCausalLM(generic.HF_TARGET).to(base.DEVICE)
    lora_params = lora_parameters(candidate)
    trainable = trainable_count(lora_params)
    print(f"  Injected LoRA modules: {len(candidate.injected_lora)}")
    print(f"  Target params before LoRA: {candidate.target_param_count:,}")
    print(f"  LoRA trainable params: {trainable:,}")
    opt = torch.optim.AdamW(lora_params, lr=LR, weight_decay=WEIGHT_DECAY)
    candidate.train()
    stopped_early = False
    nan_step = None
    completed_steps = 0
    stop_reason = None
    for step in range(CALIBRATION_STEPS):
        if MAX_TRAIN_SECONDS is not None and (time.time() - t_b) >= MAX_TRAIN_SECONDS:
            stopped_early = True
            stop_reason = "max_train_seconds"
            print(
                f"  Wall-time cap reached before step {step + 1}; "
                "stopping baseline calibration"
            )
            break
        x, y = make_batch(domain_ids, CAL_SEED_BASE + step)
        opt.zero_grad()
        student_logits = candidate(x)
        with torch.no_grad():
            teacher_logits = native(x)
        student_flat = student_logits.reshape(-1, candidate.vocab_size).float()
        teacher_flat = teacher_logits.reshape(-1, candidate.vocab_size).float()
        ce = F.cross_entropy(student_flat, y.reshape(-1))
        kd = F.kl_div(
            F.log_softmax(student_flat / KD_TEMP, dim=-1),
            F.softmax(teacher_flat / KD_TEMP, dim=-1),
            reduction="batchmean",
        ) * (KD_TEMP ** 2)
        loss = KD_WEIGHT * kd + ((1 - KD_WEIGHT) * ce if TRAIN_CE else 0.0)
        loss = add_calibration_losses(loss, student_logits, teacher_logits, candidate.vocab_size)
        if not torch.isfinite(loss):
            stopped_early = True
            nan_step = step + 1
            stop_reason = "non_finite_loss"
            print(f"  Non-finite LoRA loss at step {nan_step}; stopping baseline calibration")
            break
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
        if not torch.isfinite(grad_norm):
            stopped_early = True
            nan_step = step + 1
            stop_reason = "non_finite_grad_norm"
            print(f"  Non-finite LoRA grad norm at step {nan_step}; stopping baseline calibration")
            break
        opt.step()
        completed_steps = step + 1
        if (step + 1) % 300 == 0:
            print(f"  B step {step + 1}/{CALIBRATION_STEPS}  {time.time() - t_b:.0f}s")
    candidate.eval()
    for p in lora_params:
        p.requires_grad_(False)

    posthoc_logit_scale = generic.calibrate_posthoc_logit_scale(native, candidate, posthoc_ids)
    if posthoc_logit_scale["applied"]:
        selection = posthoc_logit_scale["selection"]
        print(
            f"  Posthoc logit scale={posthoc_logit_scale['scale']:.4f}  "
            f"calib_JS={selection['mean_js']:.5f}  "
            f"calib_entropy_diff={selection['mean_entropy_diff']:.4f}"
        )
    ppl_candidate = generic.ppl(candidate, eval_ids)
    print(f"  Phase B complete: {time.time() - t_b:.0f}s  LoRA ppl={ppl_candidate:.1f}")

    banner("NIB L2 evaluation")
    t_nib = time.time()
    l2 = generic.l2_logit_test(native, candidate, eval_ids)
    overall = l2["pass"]
    status = "PASS" if overall else "FAIL"
    print()
    print(f"  mean_JS          = {l2['mean_js']:.5f}   {'PASS' if l2['js_pass'] else 'FAIL'}")
    print(f"  mean_top1_agree  = {l2['mean_top1_agree']:.4f}  {'PASS' if l2['top1_pass'] else 'FAIL'}")
    print(f"  mean_top5_overlap= {l2['mean_top5_overlap']:.4f}  {'PASS' if l2['top5_pass'] else 'FAIL'}")
    print(f"  mean_entropy_diff= {l2['mean_entropy_diff']:.4f}  {'PASS' if l2['entropy_pass'] else 'FAIL'}")
    print(f"  NIB overall: {status}  ({time.time() - t_nib:.1f}s)")

    elapsed = time.time() - t_global
    tag = os.environ.get(
        "ABI_BASELINE_EXPERIMENT_TAG",
        f"{TARGET_LABEL}_{LORA_TARGETS}_r{LORA_RANK}_{DOMAIN_CORPUS}_cal{CALIBRATION_STEPS}_seed{EXPERIMENT_SEED}",
    )
    results = {
        "experiment": "lora_kd_baseline",
        "name": f"exp_lora_kd_baseline_{tag}",
        "variant_type": "matched target-side LoRA/KD comparator",
        "baseline_type": "target_side_lora_kd",
        "target_model": TARGET_MODEL_ID,
        "target_tokenizer": TARGET_TOKENIZER_ID,
        "target_label": TARGET_LABEL,
        "target_model_type": candidate.model_type,
        "target_param_count": candidate.target_param_count,
        "teacher_d_abi": TEACHER_D_ABI,
        "domain_corpus": DOMAIN_CORPUS,
        "wikitext_domain_split": (
            WIKITEXT_DOMAIN_SPLIT if DOMAIN_CORPUS == "wikitext" else None
        ),
        "wikitext_posthoc_split": (
            WIKITEXT_POSTHOC_SPLIT if DOMAIN_CORPUS == "wikitext" else None
        ),
        "wikitext_eval_split": (
            WIKITEXT_EVAL_SPLIT if DOMAIN_CORPUS == "wikitext" else None
        ),
        "withheld_nib_eval": bool(
            DOMAIN_CORPUS == "wikitext"
            and (
                WIKITEXT_EVAL_SPLIT != WIKITEXT_DOMAIN_SPLIT
                or WIKITEXT_EVAL_SPLIT != WIKITEXT_POSTHOC_SPLIT
            )
        ),
        "domain_train_tokens_target": int(len(domain_ids)),
        "posthoc_tokens_target": int(len(posthoc_ids)),
        "nib_eval_tokens_target": int(len(eval_ids)),
        "domain_steps_native_oracle": DOMAIN_STEPS,
        "calibration_steps": CALIBRATION_STEPS,
        "completed_calibration_steps": completed_steps,
        "requested_calibration_steps": CALIBRATION_STEPS,
        "max_train_seconds": MAX_TRAIN_SECONDS,
        "seed": EXPERIMENT_SEED,
        "seed_offset": SEED_OFFSET,
        "native_domain_seed_base": NATIVE_DOMAIN_SEED_BASE,
        "calibration_seed_base": CAL_SEED_BASE,
        "ppl_seed_base": PPL_SEED_BASE,
        "nib_seed": NIB_SEED,
        "batch": BATCH,
        "ppl_batches": PPL_BATCHES,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_dtype": LORA_DTYPE,
        "lora_targets": LORA_TARGETS,
        "lora_injected_modules": len(candidate.injected_lora),
        "lora_injected_module_details": candidate.injected_lora,
        "calibration_trainable_groups": [{"name": "lora", "params": trainable}],
        "calibration_trainable_params": trainable,
        "trainable_fraction_of_target": trainable / candidate.target_param_count,
        "kd_weight": KD_WEIGHT,
        "kd_temp": KD_TEMP,
        "train_ce": TRAIN_CE,
        "grad_clip": GRAD_CLIP,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "nan_step": nan_step,
        "topk_kd_weight": TOPK_KD_WEIGHT,
        "topk": TOPK,
        "rank_margin_weight": RANK_MARGIN_WEIGHT,
        "rank_top_pos": RANK_TOP_POS,
        "rank_neg_k": RANK_NEG_K,
        "rank_margin": RANK_MARGIN,
        "top1_gap_weight": TOP1_GAP_WEIGHT,
        "top1_gap_k": TOP1_GAP_K,
        "top1_gap_margin": TOP1_GAP_MARGIN,
        "top1_ce_weight": TOP1_CE_WEIGHT,
        "hard_neg_weight": HARD_NEG_WEIGHT,
        "hard_neg_k": HARD_NEG_K,
        "top1_hard_neg_weight": TOP1_HARD_NEG_WEIGHT,
        "top1_hard_neg_k": TOP1_HARD_NEG_K,
        "topset_weight": TOPSET_WEIGHT,
        "topset_k": TOPSET_K,
        "topset_temp": TOPSET_TEMP,
        "entropy_weight": ENTROPY_WEIGHT,
        "posthoc_logit_scale": posthoc_logit_scale,
        "torch_dtype": generic.TORCH_DTYPE_LABEL,
        "corpus_exclude_globs": list(generic.V2_CORPUS_EXCLUDES),
        "corpus_skipped_files": py_meta["skipped"],
        "wikitext_records": wiki_meta,
        "ppl_native_target": round(ppl_native, 3),
        "ppl_lora_target": round(ppl_candidate, 3),
        "nib_l2": l2,
        "overall_pass": overall,
        "elapsed_min": round(elapsed / 60, 1),
        "thresholds": base.REGISTRY,
        "interpretation": (
            "Matched target-side LoRA/KD control. It adapts the target model directly "
            "and does not copy a frozen source domain core, so it is a comparator for "
            "PEFT efficiency rather than evidence of ABI portability."
        ),
    }
    out_path = generic.result_path("exp_lora_kd_baseline", tag)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results -> {out_path}")
    banner(f"Done - {status} - {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
