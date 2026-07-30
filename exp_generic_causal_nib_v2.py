#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic decoder-only ABI/NIB v2 runner for cached Hugging Face targets.

Default target:
  EleutherAI/pythia-410m

Environment knobs:
  ABI_SOURCE_MODEL_ID   default gpt2-medium
  ABI_SOURCE_TOKENIZER_ID optional tokenizer override
  ABI_SOURCE_LABEL      short label for result filenames
  ABI_TARGET_MODEL_ID   e.g. EleutherAI/pythia-410m
  ABI_TARGET_TOKENIZER_ID optional tokenizer override
  ABI_TARGET_LABEL      short label for result filenames
  ABI_D_ABI             default 256
  ABI_DOMAIN_STEPS      default 500
  ABI_CAL_STEPS         default 1200
  ABI_CAL_LR_DECAY_STEP default 0; step after which target calibration LR decays
  ABI_CAL_LR_DECAY_FACTOR default 1.0
  ABI_CAL_LR_SCHEDULE   step | cosine_after; default step
  ABI_CAL_LR_FINAL_FACTOR default ABI_CAL_LR_DECAY_FACTOR for cosine_after
  ABI_CAL_LR            default base.LR_CAL; target calibration LR override
  ABI_CAL_ACCUM_STEPS default 1; gradient accumulation micro-batches per update
  ABI_CAL_ACCUM_SEED_STRIDE default 100000; seed stride between micro-batches
  ABI_CAL_MODE          freeze_domain_net | train_domain | freeze_all_domain
  ABI_CAL_INIT          xavier | zero_out | native; initialize target calibration
                      interface from scratch, with a zero residual passthrough,
                      or from the native Phase C target ABI oracle
  ABI_TARGET_INTERFACE_CACHE none | save | load | auto; optionally save/load
                      a reusable target-base ABI interface prefit for Phase C
  ABI_TARGET_INTERFACE_CACHE_PATH optional explicit .pt path
  ABI_ORACLE_MODE       full_native_target_oracle | target_base_interface |
                      base_target_reference. The default trains the native
                      target domain oracle. target_base_interface trains only
                      the target ABI interface with domain disabled.
                      base_target_reference skips target ABI training and uses
                      pure base-model logits as the reference teacher.
  ABI_CAL_SELECT        none | validation_top1; restore best validation checkpoint
  ABI_CAL_SELECT_EVERY  default 600
  ABI_CAL_SELECT_CHUNKS default 2
  ABI_CAL_SELECT_REPEATS default 1; independent validation samples per checkpoint
  ABI_CAL_SELECT_REDUCTION mean_score | min_score
  ABI_CAL_SELECT_TOP1_WEIGHT default 1.0
  ABI_CAL_SELECT_TOP5_WEIGHT default 0.0
  ABI_CAL_SELECT_JS_WEIGHT default 0.25
  ABI_CAL_SELECT_ENTROPY_WEIGHT default 0.05
  ABI_CAL_SELECT_AVG_TOP_N default 1; average top-N validation checkpoints
  ABI_CAL_SELECT_SOUP_WEIGHTS optional comma weights for top-N checkpoints
  ABI_CAL_SELECT_SOUP_WEIGHT_GRID optional comma first-checkpoint weights for
                      validation-selected top-two checkpoint soup
  ABI_CAL_SELECT_SOUP_GRID_REPEATS default 1; repeated validation samples for grid
  ABI_CAL_SELECT_SOUP_GRID_REDUCTION mean_score | min_score
  ABI_CAL_SELECT_AUDIT_NIB default false; diagnostic only, evaluates validation
                      checkpoint states on withheld NIB without using it to select
  ABI_CAL_SELECT_AUDIT_MAX default 0; max ranked checkpoints to audit, 0 means all
  ABI_CAL_SELECT_SOUP_AUDIT_WEIGHTS optional comma first-checkpoint weights for
                      diagnostic held-out NIB audit of top-two checkpoint soups
  ABI_CAL_EMA_DECAY    default 0.0 disabled; EMA decay for trainable cal params
  ABI_CAL_EMA_START_STEP default 1
  ABI_CAL_EMA_EVERY    default 1
  ABI_CAL_EMA_AS_CANDIDATE default true; add EMA state to validation candidates
  ABI_CAL_EMA_RESTORE  default false; force final EMA state after selection
  ABI_CAL_FINAL_SELECT none | validation; validation-only final selector over
                      selected checkpoint soup, EMA, final, and best checkpoint
  ABI_CAL_FINAL_SELECT_CANDIDATES default selected,ema
  ABI_CAL_FINAL_SELECT_CHUNKS default ABI_CAL_SELECT_CHUNKS
  ABI_CAL_FINAL_SELECT_REPEATS default 1
  ABI_CAL_FINAL_SELECT_TOP1_WEIGHT default 1.0
  ABI_CAL_FINAL_SELECT_TOP5_WEIGHT default 0.25
  ABI_CAL_FINAL_SELECT_JS_WEIGHT default 0.25
  ABI_CAL_FINAL_SELECT_ENTROPY_WEIGHT default 0.05
  ABI_CAL_FINAL_AUDIT_NIB default false; diagnostic only, evaluates requested
                      final candidates on withheld NIB without using it to select
  ABI_CAL_FINAL_AUDIT_CANDIDATES default ABI_CAL_FINAL_SELECT_CANDIDATES
  ABI_CAL_FINAL_SOUP_AUDIT_CANDIDATES default selected,ema
  ABI_CAL_FINAL_SOUP_AUDIT_WEIGHTS optional comma first-candidate weights for
                      diagnostic held-out NIB audit of final-candidate soups
  ABI_CAL_TEMPORAL_AVG_START_STEP default 0 disabled; uniformly average saved
                      validation checkpoint states at/after this step
  ABI_CAL_TEMPORAL_AVG_RESTORE default false; force final temporal average
  ABI_KD_WEIGHT         default base registry value; native-teacher KD mix
  ABI_KD_TEMP           default base registry value; native-teacher KD temperature
  ABI_TOPK_KD_WEIGHT    default 0.0; extra interface-only top-k KD term
  ABI_TOPK              default 32
  ABI_UNION_TOPK_KD_WEIGHT default 0.0; KD over concatenated teacher top-k and
                      current student top-k candidates
  ABI_UNION_TOPK        default 64
  ABI_UNION_TOPK_TEMP   default ABI_KD_TEMP
  ABI_RANK_MARGIN_WEIGHT default 0.0; pairwise teacher-top5 rank loss
  ABI_RANK_TOP_POS      default 5
  ABI_RANK_NEG_K        default 32
  ABI_RANK_MARGIN       default 0.25
  ABI_HARD_NEG_WEIGHT   default 0.0; suppress student top candidates outside teacher top5
  ABI_HARD_NEG_K        default 32
  ABI_TOP1_HARD_NEG_WEIGHT default 0.0; suppress student top candidates above teacher argmax
  ABI_TOP1_HARD_NEG_K   default ABI_HARD_NEG_K
  ABI_STABLE_TOP1_CE_WEIGHT default 0.0; teacher-argmax CE only on tokens where
                      the native target's domain-on and domain-off top-1 agree
  ABI_STABLE_TOP1_HARD_NEG_WEIGHT default 0.0; hard-negative top-1 suppression
                      only on native target domain-stable tokens
  ABI_STABLE_TOP1_HARD_NEG_K default ABI_TOP1_HARD_NEG_K
  ABI_STABLE_TOP1_REQUIRE_BASE_AGREE default true
  ABI_STABLE_TOP1_MIN_MARGIN default 0.0; optional native top-1/top-2 margin floor
  ABI_TOPSET_WEIGHT     default 0.0; full-vocab listwise teacher top-set loss
  ABI_TOPSET_K          default 5
  ABI_TOPSET_TEMP       default 1.0
  ABI_TOP_LOGIT_MSE_WEIGHT default 0.0; centered teacher top-logit matching
  ABI_TOP_LOGIT_MSE_K   default 64
  ABI_DOMAIN_DELTA_LOGIT_MSE_WEIGHT default 0.0; match native/calibrated
                      domain-on minus domain-off logit deltas on largest native
                      delta coordinates
  ABI_DOMAIN_DELTA_LOGIT_MSE_K default 64
  ABI_DOMAIN_DELTA_LOGIT_MSE_CENTER default true
  ABI_ENTROPY_WEIGHT    default 0.0; match teacher/student token entropy
  ABI_ABI_PRE_MSE_WEIGHT default 0.0; match native/calibrated pre-domain ABI state
  ABI_ABI_STATE_MSE_WEIGHT default 0.0; match native/calibrated post-domain ABI state
  ABI_CONF_WEIGHT_MODE  none | teacher_margin | teacher_low_margin;
                      confidence-weight rank/listwise auxiliary losses by
                      teacher top-1/top-2 logit margin
  ABI_CONF_WEIGHT_CENTER default 0.75
  ABI_CONF_WEIGHT_TEMP  default 0.25
  ABI_CONF_WEIGHT_MIN   default 0.25
  ABI_CONF_WEIGHT_MAX   default 2.0
  ABI_POSTHOC_LOGIT_SCALE none | entropy_grid; tune global calibrated-logit scale
  ABI_POSTHOC_BIAS        none | global; train validation-only vocab bias after scale
  ABI_POSTHOC_BIAS_STEPS  default 0
  ABI_POSTHOC_BIAS_LR     default 0.05
  ABI_POSTHOC_BIAS_L2     default 0.001
  ABI_POSTHOC_BIAS_TOP1_CE_WEIGHT default 0.25
  ABI_POSTHOC_SCALE_MIN default 0.25
  ABI_POSTHOC_SCALE_MAX default 4.0
  ABI_POSTHOC_SCALE_STEPS default 25
  ABI_POSTHOC_SCALE_CHUNKS default 4
  ABI_POSTHOC_SCALE_REPEATS default 1; independent held-out chunk seeds
  ABI_POSTHOC_SELECTION mean_entropy | minimax_entropy | balanced_entropy
  ABI_POSTHOC_SIGNED_ENTROPY_WEIGHT default 1.0
  ABI_POSTHOC_SELECTIVE_WEIGHT default 0.0; when selective eval is enabled,
                      add off-domain no-leakage validation to post-hoc scale
                      selection instead of selecting scale from the target
                      domain alone
  ABI_DOMAIN_BRIDGE     none | linear; train target-side bridge around frozen domain core
  ABI_DOMAIN_RESIDUAL_RANK default 0; ABI-space residual around frozen domain core
  ABI_DOMAIN_RESIDUAL_SCALE default 1.0
  ABI_TARGET_RESIDUAL   none | hidden | logit_abi; optional target residual for hybrid benchmarks
  ABI_TARGET_RESIDUAL_RANK default 64; bottleneck rank for optional target residual
  ABI_TARGET_RESIDUAL_SCALE default 1.0
  ABI_TORCH_DTYPE       auto | float32 | float16 | bfloat16
  ABI_BATCH             default 4
  ABI_PPL_BATCHES       default 50
  ABI_SEED              default 42
  ABI_SEED_OFFSET       default 0; shifts batch/PPL/NIB seeds for repeat runs
  ABI_N_ALIGN_SENTENCES default base registry value; Procrustes sentence pairs
  ABI_ALIGN_MIN_CHARS   default 20; minimum sentence length for Procrustes pairs
  ABI_ALIGN_SELECT      none | procrustes_trim; optional geometry-aware pair trim
  ABI_ALIGN_POOL_SENTENCES default ABI_N_ALIGN_SENTENCES; pool size for trim
  ABI_ALIGN_FIT_NORMALIZE center | zscore; normalization for Procrustes fit
  ABI_ALIGN_MAP        procrustes | linear | linear_blend; default procrustes
  ABI_ALIGN_RIDGE      default 0.01; ridge for linear alignment map
  ABI_ALIGN_LINEAR_BLEND default 0.25; blend weight for linear map when
                      ABI_ALIGN_MAP=linear_blend
  ABI_ROTATION_ENSEMBLE_SIZE default 1; number of independently fitted
                      Procrustes rotations of the same copied source domain core
  ABI_ROTATION_ENSEMBLE_STRIDE default ABI_N_ALIGN_SENTENCES; sentence-window
                      offset between ensemble rotations
  ABI_ROTATION_ENSEMBLE_WEIGHTS optional comma weights over ensemble members
  ABI_ROTATION_ENSEMBLE_TRAIN_WEIGHTS default false; train a softmax over
                      copied rotated domain cores during validation-calibrated D
  ABI_RELEASE_SOURCE_BEFORE_TARGET default false; cache source alignment/domain then free source
  ABI_SOURCE_PRESERVATION_EVAL default false; cache source next-token text
                      probes before optional source release and compare them to
                      the final calibrated target
  ABI_SOURCE_PRESERVATION_PROMPTS default 32
  ABI_SOURCE_PRESERVATION_TOPK default 5
  ABI_SOURCE_PRESERVATION_MAX_LENGTH default 128
  ABI_SOURCE_PRESERVATION_PREFIX_CHARS default 256
  ABI_SOURCE_PRESERVATION_COMPLETION_EVAL default true; score source top-k
                      decoded continuations under the target tokenizer/model
                      so cross-tokenizer multi-token equivalents are audited
  ABI_SOURCE_COMPLETION_LOSS_WEIGHT default 0.0; optional Phase D CE loss
                      that ranks the source model's decoded top-1 continuation
                      above its other decoded top-k continuations under the
                      calibrated target tokenizer/model
  ABI_SOURCE_COMPLETION_LOSS_EVERY default 8; apply every N Phase D steps
  ABI_SOURCE_COMPLETION_LOSS_BATCH default 1; source prompts per application
  ABI_SOURCE_COMPLETION_LOSS_PROMPTS default 0; 0 uses all collected prompts
  ABI_SOURCE_COMPLETION_LOSS_CANDIDATES default ABI_SOURCE_PRESERVATION_TOPK
  ABI_SOURCE_COMPLETION_LOSS_TEMP default 1.0
  ABI_SOURCE_COMPLETION_LOSS_START_STEP default 0; delay source-completion loss
                      until this target calibration step
  ABI_SOURCE_COMPLETION_MARGIN_WEIGHT default 0.0; optional pairwise margin
                      term inside the source-completion loss that pushes the
                      source top-1 decoded continuation above source top-k
                      alternatives under the calibrated target
  ABI_SOURCE_COMPLETION_MARGIN default 0.10
  ABI_SOURCE_COMPLETION_NLL_WEIGHT default 0.0; optional absolute NLL term
                      inside the source-completion loss that pushes the source
                      top-1 decoded continuation against the full target vocab
  ABI_SOURCE_COMPLETION_NLL_CAP default 0.0; if >0, the NLL term becomes a
                      hinge above this cap instead of pushing probability to 1
  ABI_DOMAIN_CORPUS     python | wikitext; domain train/eval corpus
  ABI_WIKITEXT_DOMAIN_SPLIT default validation; wikitext split for A/C/D training
  ABI_WIKITEXT_ALIGN_SPLIT default validation; wikitext split for Procrustes pairs
  ABI_WIKITEXT_POSTHOC_SPLIT default eval split; split for post-hoc scale selection
  ABI_WIKITEXT_EVAL_SPLIT default domain split; split for final PPL/NIB evaluation
  ABI_SELECTIVE_TRANSFER_EVAL default false; additionally audit an off-domain
                      corpus against the frozen target base/reference to check
                      whether the selected-domain payload leaks into unrelated
                      behavior
  ABI_SELECTIVE_OFF_DOMAIN_CORPUS auto | python | wikitext; default auto picks
                      the opposite of ABI_DOMAIN_CORPUS
  ABI_SELECTIVE_OFF_DOMAIN_WIKITEXT_SPLIT default ABI_WIKITEXT_EVAL_SPLIT
  ABI_SELECTIVE_OFF_DOMAIN_REFERENCE base | target_abi_no_domain; default base

This runner uses the strict v2 copy/paste protocol: rotate the source domain MLP
into the target ABI basis, freeze the rotated domain MLP core, and calibrate only
the target interface plus target-side normalization/scale by default.
"""

from __future__ import annotations

import json
import math
import os
import re
import hashlib
import time
import copy
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from abi.artifacts import (
    build_abi_artifact,
    build_compatibility_certificate,
    build_cost_ledger,
    module_param_count as artifact_module_param_count,
    module_state_sha256 as artifact_module_state_sha256,
)
import exp_deepseek_1p3b_nib as base


def env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def result_path(prefix, tag):
    name = f"{prefix}_{tag}_results.json"
    path = base.ROOT / name
    if len(str(path)) < 240:
        return path
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:10]
    short_tag = f"{tag[:110].rstrip('_')}_{digest}"
    return base.ROOT / f"{prefix}_{short_tag}_results.json"


def default_target_interface_cache_path():
    tag = (
        f"{slug(TARGET_MODEL_ID)}_d{D_ABI}_{ORACLE_MODE}_"
        f"{DOMAIN_CORPUS}_steps{DOMAIN_STEPS}_"
        f"nativebase{NATIVE_DOMAIN_SEED_BASE}"
    )
    return base.ROOT / "target_interface_cache" / f"{tag}.pt"


def target_interface_cache_path():
    if TARGET_INTERFACE_CACHE_PATH_RAW:
        return Path(TARGET_INTERFACE_CACHE_PATH_RAW)
    return default_target_interface_cache_path()


def interface_state_dict(model):
    return {
        "proj_in": model.proj_in.state_dict(),
        "abi_ln": model.abi_ln.state_dict(),
        "proj_out": model.proj_out.state_dict(),
    }


def save_target_interface_cache(model, path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "abi-target-interface-cache-v1",
        "summary": summary,
        "state": interface_state_dict(model),
    }
    torch.save(payload, path)


def load_target_interface_cache(model, path):
    payload = torch.load(path, map_location=base.DEVICE, weights_only=False)
    summary = payload.get("summary", {})
    expected = {
        "target_model": TARGET_MODEL_ID,
        "d_abi": D_ABI,
        "oracle_mode": ORACLE_MODE,
        "target_reference_uses_domain": TARGET_REFERENCE_USES_DOMAIN,
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Target interface cache metadata mismatch for {path}: {mismatches}"
        )
    state = payload["state"]
    model.proj_in.load_state_dict(state["proj_in"])
    model.abi_ln.load_state_dict(state["abi_ln"])
    model.proj_out.load_state_dict(state["proj_out"])
    return summary


SOURCE_MODEL_ID = os.environ.get("ABI_SOURCE_MODEL_ID", "gpt2-medium")
SOURCE_TOKENIZER_ID = os.environ.get("ABI_SOURCE_TOKENIZER_ID", SOURCE_MODEL_ID)
SOURCE_LABEL = os.environ.get("ABI_SOURCE_LABEL", slug(SOURCE_MODEL_ID))
TARGET_MODEL_ID = os.environ.get("ABI_TARGET_MODEL_ID", "EleutherAI/pythia-410m")
TARGET_TOKENIZER_ID = os.environ.get("ABI_TARGET_TOKENIZER_ID", TARGET_MODEL_ID)
TARGET_LABEL = os.environ.get("ABI_TARGET_LABEL", slug(TARGET_MODEL_ID))
HF_SOURCE = base.hf_local_path(SOURCE_MODEL_ID)
HF_SOURCE_TOKENIZER = base.hf_local_path(SOURCE_TOKENIZER_ID)
HF_TARGET = base.hf_local_path(TARGET_MODEL_ID)
HF_TARGET_TOKENIZER = base.hf_local_path(TARGET_TOKENIZER_ID)

D_ABI = int(os.environ.get("ABI_D_ABI", "256"))
DOMAIN_STEPS = int(os.environ.get("ABI_DOMAIN_STEPS", str(base.DOMAIN_STEPS)))
CALIBRATION_STEPS = int(os.environ.get("ABI_CAL_STEPS", "1200"))
CAL_LR = float(
    os.environ.get("ABI_CAL_LR", os.environ.get("ABI_LR_CAL", str(base.LR_CAL)))
)
CAL_LR_DECAY_STEP = int(os.environ.get("ABI_CAL_LR_DECAY_STEP", "0"))
CAL_LR_DECAY_FACTOR = float(os.environ.get("ABI_CAL_LR_DECAY_FACTOR", "1.0"))
CAL_LR_SCHEDULE = os.environ.get("ABI_CAL_LR_SCHEDULE", "step").strip().lower()
CAL_LR_FINAL_FACTOR = float(
    os.environ.get("ABI_CAL_LR_FINAL_FACTOR", str(CAL_LR_DECAY_FACTOR))
)
CAL_ACCUM_STEPS = int(os.environ.get("ABI_CAL_ACCUM_STEPS", "1"))
CAL_ACCUM_SEED_STRIDE = int(os.environ.get("ABI_CAL_ACCUM_SEED_STRIDE", "100000"))
CAL_MODE = os.environ.get("ABI_CAL_MODE", "freeze_domain_net").strip().lower()
CAL_INIT = os.environ.get("ABI_CAL_INIT", "xavier").strip().lower()
TARGET_INTERFACE_CACHE_MODE = os.environ.get(
    "ABI_TARGET_INTERFACE_CACHE", "none"
).strip().lower()
TARGET_INTERFACE_CACHE_PATH_RAW = os.environ.get(
    "ABI_TARGET_INTERFACE_CACHE_PATH", ""
).strip()
ORACLE_MODE = os.environ.get(
    "ABI_ORACLE_MODE", "full_native_target_oracle"
).strip().lower()
CAL_SELECT = os.environ.get("ABI_CAL_SELECT", "none").strip().lower()
CAL_SELECT_EVERY = int(os.environ.get("ABI_CAL_SELECT_EVERY", "600"))
CAL_SELECT_CHUNKS = int(os.environ.get("ABI_CAL_SELECT_CHUNKS", "2"))
CAL_SELECT_REPEATS = int(os.environ.get("ABI_CAL_SELECT_REPEATS", "1"))
CAL_SELECT_REDUCTION = os.environ.get(
    "ABI_CAL_SELECT_REDUCTION", "mean_score"
).strip().lower()
CAL_SELECT_TOP1_WEIGHT = float(os.environ.get("ABI_CAL_SELECT_TOP1_WEIGHT", "1.0"))
CAL_SELECT_TOP5_WEIGHT = float(os.environ.get("ABI_CAL_SELECT_TOP5_WEIGHT", "0.0"))
CAL_SELECT_JS_WEIGHT = float(os.environ.get("ABI_CAL_SELECT_JS_WEIGHT", "0.25"))
CAL_SELECT_ENTROPY_WEIGHT = float(
    os.environ.get("ABI_CAL_SELECT_ENTROPY_WEIGHT", "0.05")
)
CAL_SELECT_AVG_TOP_N = int(os.environ.get("ABI_CAL_SELECT_AVG_TOP_N", "1"))
CAL_SELECT_SOUP_WEIGHTS_RAW = os.environ.get("ABI_CAL_SELECT_SOUP_WEIGHTS", "").strip()
CAL_SELECT_SOUP_WEIGHT_GRID_RAW = os.environ.get(
    "ABI_CAL_SELECT_SOUP_WEIGHT_GRID", ""
).strip()
CAL_SELECT_SOUP_GRID_REPEATS = int(
    os.environ.get("ABI_CAL_SELECT_SOUP_GRID_REPEATS", "1")
)
CAL_SELECT_SOUP_GRID_REDUCTION = os.environ.get(
    "ABI_CAL_SELECT_SOUP_GRID_REDUCTION", "mean_score"
).strip().lower()
CAL_SELECT_AUDIT_NIB = env_bool("ABI_CAL_SELECT_AUDIT_NIB", False)
CAL_SELECT_AUDIT_MAX = int(os.environ.get("ABI_CAL_SELECT_AUDIT_MAX", "0"))
CAL_SELECT_SOUP_AUDIT_WEIGHTS_RAW = os.environ.get(
    "ABI_CAL_SELECT_SOUP_AUDIT_WEIGHTS", ""
).strip()
CAL_EMA_DECAY = float(os.environ.get("ABI_CAL_EMA_DECAY", "0.0"))
CAL_EMA_START_STEP = int(os.environ.get("ABI_CAL_EMA_START_STEP", "1"))
CAL_EMA_EVERY = int(os.environ.get("ABI_CAL_EMA_EVERY", "1"))
CAL_EMA_AS_CANDIDATE = env_bool("ABI_CAL_EMA_AS_CANDIDATE", True)
CAL_EMA_RESTORE = env_bool("ABI_CAL_EMA_RESTORE", False)
CAL_FINAL_SELECT = os.environ.get("ABI_CAL_FINAL_SELECT", "none").strip().lower()
CAL_FINAL_SELECT_CANDIDATES_RAW = os.environ.get(
    "ABI_CAL_FINAL_SELECT_CANDIDATES", "selected,ema"
).strip()
CAL_FINAL_SELECT_CHUNKS = int(
    os.environ.get("ABI_CAL_FINAL_SELECT_CHUNKS", str(CAL_SELECT_CHUNKS))
)
CAL_FINAL_SELECT_REPEATS = int(os.environ.get("ABI_CAL_FINAL_SELECT_REPEATS", "1"))
CAL_FINAL_SELECT_TOP1_WEIGHT = float(
    os.environ.get("ABI_CAL_FINAL_SELECT_TOP1_WEIGHT", "1.0")
)
CAL_FINAL_SELECT_TOP5_WEIGHT = float(
    os.environ.get("ABI_CAL_FINAL_SELECT_TOP5_WEIGHT", "0.25")
)
CAL_FINAL_SELECT_JS_WEIGHT = float(
    os.environ.get("ABI_CAL_FINAL_SELECT_JS_WEIGHT", "0.25")
)
CAL_FINAL_SELECT_ENTROPY_WEIGHT = float(
    os.environ.get("ABI_CAL_FINAL_SELECT_ENTROPY_WEIGHT", "0.05")
)
CAL_FINAL_AUDIT_NIB = env_bool("ABI_CAL_FINAL_AUDIT_NIB", False)
CAL_FINAL_AUDIT_CANDIDATES_RAW = os.environ.get(
    "ABI_CAL_FINAL_AUDIT_CANDIDATES", CAL_FINAL_SELECT_CANDIDATES_RAW
).strip()
CAL_FINAL_SOUP_AUDIT_CANDIDATES_RAW = os.environ.get(
    "ABI_CAL_FINAL_SOUP_AUDIT_CANDIDATES", "selected,ema"
).strip()
CAL_FINAL_SOUP_AUDIT_WEIGHTS_RAW = os.environ.get(
    "ABI_CAL_FINAL_SOUP_AUDIT_WEIGHTS", ""
).strip()
CAL_TEMPORAL_AVG_START_STEP = int(os.environ.get("ABI_CAL_TEMPORAL_AVG_START_STEP", "0"))
CAL_TEMPORAL_AVG_RESTORE = env_bool("ABI_CAL_TEMPORAL_AVG_RESTORE", False)
TRAIN_DOMAIN_ALPHA = env_bool("ABI_TRAIN_DOMAIN_ALPHA", True)
KD_WEIGHT = float(os.environ.get("ABI_KD_WEIGHT", str(base.REGISTRY["kd_weight"])))
KD_TEMP = float(os.environ.get("ABI_KD_TEMP", str(base.REGISTRY["kd_temp"])))
TOPK_KD_WEIGHT = float(os.environ.get("ABI_TOPK_KD_WEIGHT", "0.0"))
TOPK = int(os.environ.get("ABI_TOPK", "32"))
UNION_TOPK_KD_WEIGHT = float(os.environ.get("ABI_UNION_TOPK_KD_WEIGHT", "0.0"))
UNION_TOPK = int(os.environ.get("ABI_UNION_TOPK", "64"))
UNION_TOPK_TEMP = float(os.environ.get("ABI_UNION_TOPK_TEMP", str(KD_TEMP)))
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
STABLE_TOP1_CE_WEIGHT = float(os.environ.get("ABI_STABLE_TOP1_CE_WEIGHT", "0.0"))
STABLE_TOP1_HARD_NEG_WEIGHT = float(
    os.environ.get("ABI_STABLE_TOP1_HARD_NEG_WEIGHT", "0.0")
)
STABLE_TOP1_HARD_NEG_K = int(
    os.environ.get("ABI_STABLE_TOP1_HARD_NEG_K", str(TOP1_HARD_NEG_K))
)
STABLE_TOP1_REQUIRE_BASE_AGREE = env_bool(
    "ABI_STABLE_TOP1_REQUIRE_BASE_AGREE", True
)
STABLE_TOP1_MIN_MARGIN = float(os.environ.get("ABI_STABLE_TOP1_MIN_MARGIN", "0.0"))
TOPSET_WEIGHT = float(os.environ.get("ABI_TOPSET_WEIGHT", "0.0"))
TOPSET_K = int(os.environ.get("ABI_TOPSET_K", "5"))
TOPSET_TEMP = float(os.environ.get("ABI_TOPSET_TEMP", "1.0"))
TOP_LOGIT_MSE_WEIGHT = float(os.environ.get("ABI_TOP_LOGIT_MSE_WEIGHT", "0.0"))
TOP_LOGIT_MSE_K = int(os.environ.get("ABI_TOP_LOGIT_MSE_K", "64"))
DOMAIN_DELTA_LOGIT_MSE_WEIGHT = float(
    os.environ.get("ABI_DOMAIN_DELTA_LOGIT_MSE_WEIGHT", "0.0")
)
DOMAIN_DELTA_LOGIT_MSE_K = int(
    os.environ.get("ABI_DOMAIN_DELTA_LOGIT_MSE_K", "64")
)
DOMAIN_DELTA_LOGIT_MSE_CENTER = env_bool(
    "ABI_DOMAIN_DELTA_LOGIT_MSE_CENTER", True
)
ENTROPY_WEIGHT = float(os.environ.get("ABI_ENTROPY_WEIGHT", "0.0"))
ABI_PRE_MSE_WEIGHT = float(os.environ.get("ABI_ABI_PRE_MSE_WEIGHT", "0.0"))
ABI_STATE_MSE_WEIGHT = float(os.environ.get("ABI_ABI_STATE_MSE_WEIGHT", "0.0"))
CONF_WEIGHT_MODE = os.environ.get("ABI_CONF_WEIGHT_MODE", "none").strip().lower()
CONF_WEIGHT_CENTER = float(os.environ.get("ABI_CONF_WEIGHT_CENTER", "0.75"))
CONF_WEIGHT_TEMP = float(os.environ.get("ABI_CONF_WEIGHT_TEMP", "0.25"))
CONF_WEIGHT_MIN = float(os.environ.get("ABI_CONF_WEIGHT_MIN", "0.25"))
CONF_WEIGHT_MAX = float(os.environ.get("ABI_CONF_WEIGHT_MAX", "2.0"))
POSTHOC_LOGIT_SCALE = os.environ.get("ABI_POSTHOC_LOGIT_SCALE", "none").strip().lower()
POSTHOC_BIAS = os.environ.get("ABI_POSTHOC_BIAS", "none").strip().lower()
POSTHOC_BIAS_STEPS = int(os.environ.get("ABI_POSTHOC_BIAS_STEPS", "0"))
POSTHOC_BIAS_LR = float(os.environ.get("ABI_POSTHOC_BIAS_LR", "0.05"))
POSTHOC_BIAS_L2 = float(os.environ.get("ABI_POSTHOC_BIAS_L2", "0.001"))
POSTHOC_BIAS_TOP1_CE_WEIGHT = float(
    os.environ.get("ABI_POSTHOC_BIAS_TOP1_CE_WEIGHT", "0.25")
)
POSTHOC_SCALE_MIN = float(os.environ.get("ABI_POSTHOC_SCALE_MIN", "0.25"))
POSTHOC_SCALE_MAX = float(os.environ.get("ABI_POSTHOC_SCALE_MAX", "4.0"))
POSTHOC_SCALE_STEPS = int(os.environ.get("ABI_POSTHOC_SCALE_STEPS", "25"))
POSTHOC_SCALE_CHUNKS = int(os.environ.get("ABI_POSTHOC_SCALE_CHUNKS", "4"))
POSTHOC_SCALE_REPEATS = int(os.environ.get("ABI_POSTHOC_SCALE_REPEATS", "1"))
POSTHOC_SELECTION = os.environ.get("ABI_POSTHOC_SELECTION", "mean_entropy").strip().lower()
POSTHOC_SIGNED_ENTROPY_WEIGHT = float(
    os.environ.get("ABI_POSTHOC_SIGNED_ENTROPY_WEIGHT", "1.0")
)
POSTHOC_SELECTIVE_WEIGHT = float(
    os.environ.get("ABI_POSTHOC_SELECTIVE_WEIGHT", "0.0")
)
DOMAIN_BRIDGE = os.environ.get("ABI_DOMAIN_BRIDGE", "none").strip().lower()
DOMAIN_RESIDUAL_RANK = int(os.environ.get("ABI_DOMAIN_RESIDUAL_RANK", "0"))
DOMAIN_RESIDUAL_SCALE = float(os.environ.get("ABI_DOMAIN_RESIDUAL_SCALE", "1.0"))
TARGET_RESIDUAL = os.environ.get("ABI_TARGET_RESIDUAL", "none").strip().lower()
TARGET_RESIDUAL_RANK = int(os.environ.get("ABI_TARGET_RESIDUAL_RANK", "64"))
TARGET_RESIDUAL_SCALE = float(os.environ.get("ABI_TARGET_RESIDUAL_SCALE", "1.0"))
BATCH = int(os.environ.get("ABI_BATCH", str(base.BATCH)))
PPL_BATCHES = int(os.environ.get("ABI_PPL_BATCHES", "50"))
EXPERIMENT_SEED = int(os.environ.get("ABI_SEED", str(base.SEED)))
SEED_OFFSET = int(os.environ.get("ABI_SEED_OFFSET", "0"))
SOURCE_DOMAIN_SEED_BASE = int(
    os.environ.get("ABI_SOURCE_DOMAIN_SEED_BASE", str(5000 + SEED_OFFSET))
)
NATIVE_DOMAIN_SEED_BASE = int(
    os.environ.get("ABI_NATIVE_DOMAIN_SEED_BASE", str(5000 + SEED_OFFSET))
)
CAL_SEED_BASE = int(os.environ.get("ABI_CAL_SEED_BASE", str(7000 + SEED_OFFSET)))
PPL_SEED_BASE = int(os.environ.get("ABI_PPL_SEED_BASE", str(80000 + SEED_OFFSET)))
NIB_SEED = int(os.environ.get("ABI_NIB_SEED", str(7777 + SEED_OFFSET)))
N_ALIGN_SENTENCES = int(
    os.environ.get("ABI_N_ALIGN_SENTENCES", str(base.REGISTRY["n_align_sentences"]))
)
ALIGN_MIN_CHARS = int(os.environ.get("ABI_ALIGN_MIN_CHARS", "20"))
ALIGN_SELECT = os.environ.get("ABI_ALIGN_SELECT", "none").strip().lower()
ALIGN_POOL_SENTENCES = int(
    os.environ.get("ABI_ALIGN_POOL_SENTENCES", str(N_ALIGN_SENTENCES))
)
ALIGN_FIT_NORMALIZE = os.environ.get("ABI_ALIGN_FIT_NORMALIZE", "center").strip().lower()
ALIGN_MAP = os.environ.get("ABI_ALIGN_MAP", "procrustes").strip().lower()
ALIGN_RIDGE = float(os.environ.get("ABI_ALIGN_RIDGE", "0.01"))
ALIGN_LINEAR_BLEND = float(os.environ.get("ABI_ALIGN_LINEAR_BLEND", "0.25"))
ROTATION_ENSEMBLE_SIZE = int(os.environ.get("ABI_ROTATION_ENSEMBLE_SIZE", "1"))
ROTATION_ENSEMBLE_STRIDE = int(
    os.environ.get("ABI_ROTATION_ENSEMBLE_STRIDE", str(N_ALIGN_SENTENCES))
)
ROTATION_ENSEMBLE_WEIGHTS_RAW = os.environ.get(
    "ABI_ROTATION_ENSEMBLE_WEIGHTS", ""
).strip()
ROTATION_ENSEMBLE_TRAIN_WEIGHTS = env_bool(
    "ABI_ROTATION_ENSEMBLE_TRAIN_WEIGHTS", False
)
RELEASE_SOURCE_BEFORE_TARGET = env_bool("ABI_RELEASE_SOURCE_BEFORE_TARGET", False)
SOURCE_PRESERVATION_EVAL = env_bool("ABI_SOURCE_PRESERVATION_EVAL", False)
SOURCE_PRESERVATION_PROMPTS = int(
    os.environ.get("ABI_SOURCE_PRESERVATION_PROMPTS", "32")
)
SOURCE_PRESERVATION_TOPK = int(os.environ.get("ABI_SOURCE_PRESERVATION_TOPK", "5"))
SOURCE_PRESERVATION_MAX_LENGTH = int(
    os.environ.get("ABI_SOURCE_PRESERVATION_MAX_LENGTH", "128")
)
SOURCE_PRESERVATION_PREFIX_CHARS = int(
    os.environ.get("ABI_SOURCE_PRESERVATION_PREFIX_CHARS", "256")
)
SOURCE_PRESERVATION_COMPLETION_EVAL = env_bool(
    "ABI_SOURCE_PRESERVATION_COMPLETION_EVAL", True
)
SOURCE_COMPLETION_LOSS_WEIGHT = float(
    os.environ.get("ABI_SOURCE_COMPLETION_LOSS_WEIGHT", "0.0")
)
SOURCE_COMPLETION_LOSS_EVERY = int(
    os.environ.get("ABI_SOURCE_COMPLETION_LOSS_EVERY", "8")
)
SOURCE_COMPLETION_LOSS_BATCH = int(
    os.environ.get("ABI_SOURCE_COMPLETION_LOSS_BATCH", "1")
)
SOURCE_COMPLETION_LOSS_PROMPTS = int(
    os.environ.get("ABI_SOURCE_COMPLETION_LOSS_PROMPTS", "0")
)
SOURCE_COMPLETION_LOSS_CANDIDATES = int(
    os.environ.get(
        "ABI_SOURCE_COMPLETION_LOSS_CANDIDATES",
        str(SOURCE_PRESERVATION_TOPK),
    )
)
SOURCE_COMPLETION_LOSS_TEMP = float(
    os.environ.get("ABI_SOURCE_COMPLETION_LOSS_TEMP", "1.0")
)
SOURCE_COMPLETION_LOSS_START_STEP = int(
    os.environ.get("ABI_SOURCE_COMPLETION_LOSS_START_STEP", "0")
)
SOURCE_COMPLETION_MARGIN_WEIGHT = float(
    os.environ.get("ABI_SOURCE_COMPLETION_MARGIN_WEIGHT", "0.0")
)
SOURCE_COMPLETION_MARGIN = float(
    os.environ.get("ABI_SOURCE_COMPLETION_MARGIN", "0.10")
)
SOURCE_COMPLETION_NLL_WEIGHT = float(
    os.environ.get("ABI_SOURCE_COMPLETION_NLL_WEIGHT", "0.0")
)
SOURCE_COMPLETION_NLL_CAP = float(
    os.environ.get("ABI_SOURCE_COMPLETION_NLL_CAP", "0.0")
)
DOMAIN_CORPUS = os.environ.get("ABI_DOMAIN_CORPUS", "python").strip().lower()
WIKITEXT_DOMAIN_SPLIT = os.environ.get(
    "ABI_WIKITEXT_DOMAIN_SPLIT", "validation"
).strip().lower()
WIKITEXT_ALIGN_SPLIT = os.environ.get(
    "ABI_WIKITEXT_ALIGN_SPLIT", "validation"
).strip().lower()
WIKITEXT_EVAL_SPLIT = os.environ.get(
    "ABI_WIKITEXT_EVAL_SPLIT", WIKITEXT_DOMAIN_SPLIT
).strip().lower()
WIKITEXT_POSTHOC_SPLIT = os.environ.get(
    "ABI_WIKITEXT_POSTHOC_SPLIT", WIKITEXT_EVAL_SPLIT
).strip().lower()
SELECTIVE_TRANSFER_EVAL = env_bool("ABI_SELECTIVE_TRANSFER_EVAL", False)
SELECTIVE_OFF_DOMAIN_CORPUS_RAW = os.environ.get(
    "ABI_SELECTIVE_OFF_DOMAIN_CORPUS", "auto"
).strip().lower()
SELECTIVE_OFF_DOMAIN_WIKITEXT_SPLIT = os.environ.get(
    "ABI_SELECTIVE_OFF_DOMAIN_WIKITEXT_SPLIT", WIKITEXT_EVAL_SPLIT
).strip().lower()
SELECTIVE_OFF_DOMAIN_REFERENCE = os.environ.get(
    "ABI_SELECTIVE_OFF_DOMAIN_REFERENCE", "base"
).strip().lower()
V2_CORPUS_EXCLUDES = (
    "exp_*_nib_v2.py",
    "exp_generic_causal_nib_v2.py",
    "tests/test_model_agnostic_followups.py",
)

if CAL_MODE not in {"freeze_domain_net", "train_domain", "freeze_all_domain"}:
    raise ValueError(
        "ABI_CAL_MODE must be one of: freeze_domain_net, train_domain, freeze_all_domain"
    )

if CAL_INIT not in {"xavier", "zero_out", "native"}:
    raise ValueError("ABI_CAL_INIT must be one of: xavier, zero_out, native")

if TARGET_INTERFACE_CACHE_MODE not in {"none", "save", "load", "auto"}:
    raise ValueError(
        "ABI_TARGET_INTERFACE_CACHE must be one of: none, save, load, auto"
    )

if ORACLE_MODE not in {
    "full_native_target_oracle",
    "target_base_interface",
    "base_target_reference",
}:
    raise ValueError(
        "ABI_ORACLE_MODE must be one of: full_native_target_oracle, "
        "target_base_interface, base_target_reference"
    )

TARGET_REFERENCE_USES_DOMAIN = ORACLE_MODE == "full_native_target_oracle"
TARGET_REFERENCE_BYPASS_ABI = ORACLE_MODE == "base_target_reference"
TARGET_REFERENCE_FORWARD_MODE = (
    "base" if TARGET_REFERENCE_BYPASS_ABI else TARGET_REFERENCE_USES_DOMAIN
)
TARGET_NATIVE_ORACLE_REQUIRED = ORACLE_MODE == "full_native_target_oracle"
PHASE_C_TRAINS_TARGET_INTERFACE = ORACLE_MODE in {
    "full_native_target_oracle",
    "target_base_interface",
}
PHASE_C_TRAINS_TARGET_DOMAIN = ORACLE_MODE == "full_native_target_oracle"

if ORACLE_MODE == "base_target_reference" and CAL_INIT == "native":
    raise ValueError(
        "ABI_CAL_INIT=native requires Phase C interface training; use "
        "ABI_ORACLE_MODE=full_native_target_oracle or target_base_interface, "
        "or set ABI_CAL_INIT=xavier/zero_out."
    )

if CAL_SELECT not in {"none", "validation_top1"}:
    raise ValueError("ABI_CAL_SELECT must be one of: none, validation_top1")

if CAL_SELECT_EVERY < 1:
    raise ValueError("ABI_CAL_SELECT_EVERY must be >= 1")

if CAL_SELECT_CHUNKS < 1:
    raise ValueError("ABI_CAL_SELECT_CHUNKS must be >= 1")

if CAL_SELECT_REPEATS < 1:
    raise ValueError("ABI_CAL_SELECT_REPEATS must be >= 1")

if CAL_SELECT_REDUCTION not in {"mean_score", "min_score"}:
    raise ValueError(
        "ABI_CAL_SELECT_REDUCTION must be one of: mean_score, min_score"
    )

if CAL_SELECT_AVG_TOP_N < 1:
    raise ValueError("ABI_CAL_SELECT_AVG_TOP_N must be >= 1")

if CAL_LR_DECAY_STEP < 0:
    raise ValueError("ABI_CAL_LR_DECAY_STEP must be >= 0")

if CAL_LR <= 0:
    raise ValueError("ABI_CAL_LR must be > 0")

if CAL_LR_DECAY_FACTOR <= 0:
    raise ValueError("ABI_CAL_LR_DECAY_FACTOR must be > 0")

if CAL_LR_SCHEDULE not in {"step", "cosine_after"}:
    raise ValueError("ABI_CAL_LR_SCHEDULE must be one of: step, cosine_after")

if CAL_LR_FINAL_FACTOR < 0:
    raise ValueError("ABI_CAL_LR_FINAL_FACTOR must be >= 0")

if CAL_ACCUM_STEPS < 1:
    raise ValueError("ABI_CAL_ACCUM_STEPS must be >= 1")

if CAL_ACCUM_SEED_STRIDE < 1:
    raise ValueError("ABI_CAL_ACCUM_SEED_STRIDE must be >= 1")

if ALIGN_MIN_CHARS < 1:
    raise ValueError("ABI_ALIGN_MIN_CHARS must be >= 1")

if ALIGN_SELECT not in {"none", "procrustes_trim"}:
    raise ValueError("ABI_ALIGN_SELECT must be one of: none, procrustes_trim")

if ALIGN_FIT_NORMALIZE not in {"center", "zscore"}:
    raise ValueError("ABI_ALIGN_FIT_NORMALIZE must be one of: center, zscore")

if ALIGN_MAP not in {"procrustes", "linear", "linear_blend"}:
    raise ValueError(
        "ABI_ALIGN_MAP must be one of: procrustes, linear, linear_blend"
    )

if ALIGN_RIDGE < 0:
    raise ValueError("ABI_ALIGN_RIDGE must be >= 0")

if ALIGN_LINEAR_BLEND < 0 or ALIGN_LINEAR_BLEND > 1:
    raise ValueError("ABI_ALIGN_LINEAR_BLEND must be in [0, 1]")

if ALIGN_POOL_SENTENCES < 1:
    raise ValueError("ABI_ALIGN_POOL_SENTENCES must be >= 1")

if ALIGN_SELECT == "procrustes_trim" and ALIGN_POOL_SENTENCES < N_ALIGN_SENTENCES:
    raise ValueError(
        "ABI_ALIGN_POOL_SENTENCES must be >= ABI_N_ALIGN_SENTENCES when "
        "ABI_ALIGN_SELECT=procrustes_trim"
    )

if ROTATION_ENSEMBLE_SIZE < 1:
    raise ValueError("ABI_ROTATION_ENSEMBLE_SIZE must be >= 1")

if SOURCE_PRESERVATION_PROMPTS < 1:
    raise ValueError("ABI_SOURCE_PRESERVATION_PROMPTS must be >= 1")

if SOURCE_PRESERVATION_TOPK < 1:
    raise ValueError("ABI_SOURCE_PRESERVATION_TOPK must be >= 1")

if SOURCE_PRESERVATION_MAX_LENGTH < 4:
    raise ValueError("ABI_SOURCE_PRESERVATION_MAX_LENGTH must be >= 4")

if SOURCE_PRESERVATION_PREFIX_CHARS < 20:
    raise ValueError("ABI_SOURCE_PRESERVATION_PREFIX_CHARS must be >= 20")

if SOURCE_COMPLETION_LOSS_WEIGHT < 0:
    raise ValueError("ABI_SOURCE_COMPLETION_LOSS_WEIGHT must be >= 0")

if SOURCE_COMPLETION_LOSS_EVERY < 1:
    raise ValueError("ABI_SOURCE_COMPLETION_LOSS_EVERY must be >= 1")

if SOURCE_COMPLETION_LOSS_BATCH < 1:
    raise ValueError("ABI_SOURCE_COMPLETION_LOSS_BATCH must be >= 1")

if SOURCE_COMPLETION_LOSS_PROMPTS < 0:
    raise ValueError("ABI_SOURCE_COMPLETION_LOSS_PROMPTS must be >= 0")

if SOURCE_COMPLETION_LOSS_CANDIDATES < 2:
    raise ValueError("ABI_SOURCE_COMPLETION_LOSS_CANDIDATES must be >= 2")

if SOURCE_COMPLETION_LOSS_TEMP <= 0:
    raise ValueError("ABI_SOURCE_COMPLETION_LOSS_TEMP must be > 0")

if SOURCE_COMPLETION_LOSS_START_STEP < 0:
    raise ValueError("ABI_SOURCE_COMPLETION_LOSS_START_STEP must be >= 0")

if SOURCE_COMPLETION_MARGIN_WEIGHT < 0:
    raise ValueError("ABI_SOURCE_COMPLETION_MARGIN_WEIGHT must be >= 0")

if SOURCE_COMPLETION_MARGIN < 0:
    raise ValueError("ABI_SOURCE_COMPLETION_MARGIN must be >= 0")

if SOURCE_COMPLETION_NLL_WEIGHT < 0:
    raise ValueError("ABI_SOURCE_COMPLETION_NLL_WEIGHT must be >= 0")

if SOURCE_COMPLETION_NLL_CAP < 0:
    raise ValueError("ABI_SOURCE_COMPLETION_NLL_CAP must be >= 0")

if ROTATION_ENSEMBLE_STRIDE < 1:
    raise ValueError("ABI_ROTATION_ENSEMBLE_STRIDE must be >= 1")

if CAL_SELECT_SOUP_GRID_REPEATS < 1:
    raise ValueError("ABI_CAL_SELECT_SOUP_GRID_REPEATS must be >= 1")

if CAL_SELECT_SOUP_GRID_REDUCTION not in {"mean_score", "min_score"}:
    raise ValueError(
        "ABI_CAL_SELECT_SOUP_GRID_REDUCTION must be one of: mean_score, min_score"
    )

if CAL_SELECT_AUDIT_MAX < 0:
    raise ValueError("ABI_CAL_SELECT_AUDIT_MAX must be >= 0")

if CAL_EMA_DECAY < 0 or CAL_EMA_DECAY >= 1:
    raise ValueError("ABI_CAL_EMA_DECAY must be in [0, 1)")

if CAL_EMA_START_STEP < 1:
    raise ValueError("ABI_CAL_EMA_START_STEP must be >= 1")

if CAL_EMA_EVERY < 1:
    raise ValueError("ABI_CAL_EMA_EVERY must be >= 1")

if CAL_FINAL_SELECT not in {"none", "validation"}:
    raise ValueError("ABI_CAL_FINAL_SELECT must be one of: none, validation")

if CAL_FINAL_SELECT_CHUNKS < 1:
    raise ValueError("ABI_CAL_FINAL_SELECT_CHUNKS must be >= 1")

if CAL_FINAL_SELECT_REPEATS < 1:
    raise ValueError("ABI_CAL_FINAL_SELECT_REPEATS must be >= 1")


def parse_soup_weights(raw):
    if not raw:
        return []
    weights = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not weights:
        return []
    if any(weight < 0 for weight in weights):
        raise ValueError("ABI_CAL_SELECT_SOUP_WEIGHTS must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("ABI_CAL_SELECT_SOUP_WEIGHTS must sum to a positive value")
    return [weight / total for weight in weights]


CAL_SELECT_SOUP_WEIGHTS = parse_soup_weights(CAL_SELECT_SOUP_WEIGHTS_RAW)


def parse_rotation_ensemble_weights(raw):
    if not raw:
        return []
    weights = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not weights:
        return []
    if any(weight < 0 for weight in weights):
        raise ValueError("ABI_ROTATION_ENSEMBLE_WEIGHTS must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("ABI_ROTATION_ENSEMBLE_WEIGHTS must sum to a positive value")
    return [weight / total for weight in weights]


ROTATION_ENSEMBLE_WEIGHTS = parse_rotation_ensemble_weights(
    ROTATION_ENSEMBLE_WEIGHTS_RAW
)

if ROTATION_ENSEMBLE_WEIGHTS and len(ROTATION_ENSEMBLE_WEIGHTS) != ROTATION_ENSEMBLE_SIZE:
    raise ValueError(
        "ABI_ROTATION_ENSEMBLE_WEIGHTS length must equal "
        "ABI_ROTATION_ENSEMBLE_SIZE"
    )


def parse_soup_weight_grid(raw):
    if not raw:
        return []
    weights = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not weights:
        return []
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("ABI_CAL_SELECT_SOUP_WEIGHT_GRID weights must be in [0, 1]")
    return weights


CAL_SELECT_SOUP_WEIGHT_GRID = parse_soup_weight_grid(
    CAL_SELECT_SOUP_WEIGHT_GRID_RAW
)
CAL_SELECT_SOUP_AUDIT_WEIGHTS = parse_soup_weight_grid(
    CAL_SELECT_SOUP_AUDIT_WEIGHTS_RAW
)


def parse_final_select_candidates(raw):
    candidates = [part.strip().lower() for part in raw.split(",") if part.strip()]
    valid = {"selected", "ema", "final", "best"}
    if any(candidate not in valid for candidate in candidates):
        raise ValueError(
            "ABI_CAL_FINAL_SELECT_CANDIDATES entries must be from: "
            "selected, ema, final, best"
        )
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


CAL_FINAL_SELECT_CANDIDATES = parse_final_select_candidates(
    CAL_FINAL_SELECT_CANDIDATES_RAW
)
CAL_FINAL_AUDIT_CANDIDATES = parse_final_select_candidates(
    CAL_FINAL_AUDIT_CANDIDATES_RAW
)
CAL_FINAL_SOUP_AUDIT_CANDIDATES = parse_final_select_candidates(
    CAL_FINAL_SOUP_AUDIT_CANDIDATES_RAW
)
CAL_FINAL_SOUP_AUDIT_WEIGHTS = parse_soup_weight_grid(
    CAL_FINAL_SOUP_AUDIT_WEIGHTS_RAW
)

if CAL_FINAL_SOUP_AUDIT_WEIGHTS and len(CAL_FINAL_SOUP_AUDIT_CANDIDATES) != 2:
    raise ValueError("ABI_CAL_FINAL_SOUP_AUDIT_CANDIDATES must contain exactly two candidates")

if CAL_TEMPORAL_AVG_START_STEP < 0:
    raise ValueError("ABI_CAL_TEMPORAL_AVG_START_STEP must be >= 0")

if CONF_WEIGHT_MODE not in {"none", "teacher_margin", "teacher_low_margin"}:
    raise ValueError(
        "ABI_CONF_WEIGHT_MODE must be one of: none, teacher_margin, teacher_low_margin"
    )

if STABLE_TOP1_CE_WEIGHT < 0 or STABLE_TOP1_HARD_NEG_WEIGHT < 0:
    raise ValueError("ABI stable top-1 objective weights must be non-negative")

if STABLE_TOP1_HARD_NEG_K < 1:
    raise ValueError("ABI_STABLE_TOP1_HARD_NEG_K must be >= 1")

if STABLE_TOP1_MIN_MARGIN < 0:
    raise ValueError("ABI_STABLE_TOP1_MIN_MARGIN must be >= 0")

if DOMAIN_DELTA_LOGIT_MSE_WEIGHT < 0:
    raise ValueError("ABI_DOMAIN_DELTA_LOGIT_MSE_WEIGHT must be non-negative")

if DOMAIN_DELTA_LOGIT_MSE_K < 1:
    raise ValueError("ABI_DOMAIN_DELTA_LOGIT_MSE_K must be >= 1")

if UNION_TOPK_KD_WEIGHT < 0:
    raise ValueError("ABI_UNION_TOPK_KD_WEIGHT must be non-negative")

if UNION_TOPK < 1:
    raise ValueError("ABI_UNION_TOPK must be >= 1")

if UNION_TOPK_TEMP <= 0:
    raise ValueError("ABI_UNION_TOPK_TEMP must be > 0")

if ABI_PRE_MSE_WEIGHT < 0 or ABI_STATE_MSE_WEIGHT < 0:
    raise ValueError("ABI ABI-state MSE weights must be non-negative")

if CONF_WEIGHT_TEMP <= 0:
    raise ValueError("ABI_CONF_WEIGHT_TEMP must be > 0")

if CONF_WEIGHT_MIN < 0 or CONF_WEIGHT_MAX < CONF_WEIGHT_MIN:
    raise ValueError("ABI_CONF_WEIGHT_MIN/MAX must satisfy 0 <= min <= max")

if CAL_SELECT_SOUP_WEIGHTS and CAL_SELECT_SOUP_WEIGHT_GRID:
    raise ValueError(
        "Set either ABI_CAL_SELECT_SOUP_WEIGHTS or "
        "ABI_CAL_SELECT_SOUP_WEIGHT_GRID, not both"
    )

if CAL_SELECT_SOUP_WEIGHT_GRID and CAL_SELECT_AVG_TOP_N != 2:
    raise ValueError(
        "ABI_CAL_SELECT_SOUP_WEIGHT_GRID currently requires "
        "ABI_CAL_SELECT_AVG_TOP_N=2"
    )

if CAL_MODE == "freeze_all_domain":
    TRAIN_DOMAIN_ALPHA = False

if DOMAIN_BRIDGE not in {"none", "linear"}:
    raise ValueError("ABI_DOMAIN_BRIDGE must be one of: none, linear")

if DOMAIN_RESIDUAL_RANK < 0:
    raise ValueError("ABI_DOMAIN_RESIDUAL_RANK must be >= 0")

if TARGET_RESIDUAL not in {"none", "hidden", "logit_abi"}:
    raise ValueError(
        "ABI_TARGET_RESIDUAL must be one of: none, hidden, logit_abi"
    )

if TARGET_RESIDUAL != "none" and TARGET_RESIDUAL_RANK < 1:
    raise ValueError("ABI_TARGET_RESIDUAL_RANK must be >= 1 when enabled")

if DOMAIN_CORPUS not in {"python", "wikitext"}:
    raise ValueError("ABI_DOMAIN_CORPUS must be one of: python, wikitext")

if SELECTIVE_OFF_DOMAIN_CORPUS_RAW not in {"auto", "python", "wikitext"}:
    raise ValueError(
        "ABI_SELECTIVE_OFF_DOMAIN_CORPUS must be one of: auto, python, wikitext"
    )

for split_name, split_value in {
    "ABI_WIKITEXT_DOMAIN_SPLIT": WIKITEXT_DOMAIN_SPLIT,
    "ABI_WIKITEXT_ALIGN_SPLIT": WIKITEXT_ALIGN_SPLIT,
    "ABI_WIKITEXT_POSTHOC_SPLIT": WIKITEXT_POSTHOC_SPLIT,
    "ABI_WIKITEXT_EVAL_SPLIT": WIKITEXT_EVAL_SPLIT,
    "ABI_SELECTIVE_OFF_DOMAIN_WIKITEXT_SPLIT": SELECTIVE_OFF_DOMAIN_WIKITEXT_SPLIT,
}.items():
    if split_value not in {"train", "validation", "test"}:
        raise ValueError(f"{split_name} must be one of: train, validation, test")

if SELECTIVE_OFF_DOMAIN_REFERENCE not in {"base", "target_abi_no_domain"}:
    raise ValueError(
        "ABI_SELECTIVE_OFF_DOMAIN_REFERENCE must be one of: "
        "base, target_abi_no_domain"
    )

SELECTIVE_OFF_DOMAIN_CORPUS = (
    ("wikitext" if DOMAIN_CORPUS == "python" else "python")
    if SELECTIVE_OFF_DOMAIN_CORPUS_RAW == "auto"
    else SELECTIVE_OFF_DOMAIN_CORPUS_RAW
)
if SELECTIVE_TRANSFER_EVAL and SELECTIVE_OFF_DOMAIN_CORPUS == DOMAIN_CORPUS:
    raise ValueError(
        "ABI_SELECTIVE_OFF_DOMAIN_CORPUS must differ from ABI_DOMAIN_CORPUS "
        "when ABI_SELECTIVE_TRANSFER_EVAL=true"
    )


def selective_reference_forward_mode():
    if SELECTIVE_OFF_DOMAIN_REFERENCE == "base":
        return "base"
    return False

if POSTHOC_LOGIT_SCALE not in {"none", "entropy_grid"}:
    raise ValueError("ABI_POSTHOC_LOGIT_SCALE must be one of: none, entropy_grid")

if POSTHOC_BIAS not in {"none", "global"}:
    raise ValueError("ABI_POSTHOC_BIAS must be one of: none, global")

if POSTHOC_BIAS_STEPS < 0:
    raise ValueError("ABI_POSTHOC_BIAS_STEPS must be >= 0")

if POSTHOC_SCALE_MIN <= 0 or POSTHOC_SCALE_MAX <= 0:
    raise ValueError("ABI_POSTHOC_SCALE_MIN/MAX must be positive")

if POSTHOC_SCALE_MIN > POSTHOC_SCALE_MAX:
    raise ValueError("ABI_POSTHOC_SCALE_MIN must be <= ABI_POSTHOC_SCALE_MAX")

if POSTHOC_SCALE_STEPS < 1:
    raise ValueError("ABI_POSTHOC_SCALE_STEPS must be >= 1")

if POSTHOC_SCALE_CHUNKS < 1:
    raise ValueError("ABI_POSTHOC_SCALE_CHUNKS must be >= 1")

if POSTHOC_SCALE_REPEATS < 1:
    raise ValueError("ABI_POSTHOC_SCALE_REPEATS must be >= 1")

if POSTHOC_SELECTIVE_WEIGHT < 0:
    raise ValueError("ABI_POSTHOC_SELECTIVE_WEIGHT must be >= 0")

if POSTHOC_SELECTION not in {"mean_entropy", "minimax_entropy", "balanced_entropy"}:
    raise ValueError(
        "ABI_POSTHOC_SELECTION must be one of: "
        "mean_entropy, minimax_entropy, balanced_entropy"
    )

TAG = os.environ.get(
    "ABI_EXPERIMENT_TAG",
    f"{SOURCE_LABEL}_to_{TARGET_LABEL}_d{D_ABI}_cal{CALIBRATION_STEPS}_{CAL_MODE}"
    + ("_alpha" if TRAIN_DOMAIN_ALPHA else "_frozen_alpha"),
)

base.D_ABI = D_ABI
base.REGISTRY = dict(base.REGISTRY)
base.REGISTRY["calibration_steps"] = CALIBRATION_STEPS
torch.manual_seed(EXPERIMENT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(EXPERIMENT_SEED)


def resolve_torch_dtype():
    raw = os.environ.get("ABI_TORCH_DTYPE", "auto").strip().lower()
    if raw in {"", "auto", "default", "none"}:
        return None, raw or "auto"
    if raw in {"float32", "fp32"}:
        return torch.float32, "float32"
    if raw in {"float16", "fp16", "half"}:
        return torch.float16, "float16"
    if raw in {"bfloat16", "bf16"}:
        return torch.bfloat16, "bfloat16"
    raise ValueError(
        "ABI_TORCH_DTYPE must be one of: auto, float32, float16, bfloat16"
    )


TORCH_DTYPE, TORCH_DTYPE_LABEL = resolve_torch_dtype()


def banner(msg):
    print()
    print("=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def trainable_count(params):
    return int(sum(p.numel() for p in params))


def module_param_count(module):
    return int(sum(p.numel() for p in module.parameters()))


def add_group(name, params, cal_params, groups):
    params = list(params)
    for p in params:
        p.requires_grad_(True)
    cal_params.extend(params)
    groups.append({"name": name, "params": trainable_count(params)})


def add_module_group(name, module, cal_params, groups):
    add_group(name, module.parameters(), cal_params, groups)


def identity_linear(dim):
    layer = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(dim))
    return layer


def zero_init_linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
    return layer


def config_int(config, *names):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"None of the config fields exists: {names}")


def is_legacy_gpt2_medium_source():
    return SOURCE_MODEL_ID.lower() in {"gpt2-medium", "openai-community/gpt2-medium"}


def make_batch(tokens, seed, batch=BATCH):
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(tokens) - base.SEQ_LEN - 1, 1)
    starts = torch.randint(0, max_start, (batch,), generator=rng)
    x = torch.stack([tokens[s : s + base.SEQ_LEN] for s in starts]).to(base.DEVICE)
    y = torch.stack([tokens[s + 1 : s + base.SEQ_LEN + 1] for s in starts]).to(
        base.DEVICE
    )
    return x, y


@torch.no_grad()
def ppl(model, tokens, use_domain=True, n_batches=PPL_BATCHES, seed_offset=0):
    model.eval()
    tot, n = 0.0, 0
    max_start = max(len(tokens) - base.SEQ_LEN - 1, 1)
    rng = torch.Generator()
    for i in range(n_batches):
        rng.manual_seed(PPL_SEED_BASE + seed_offset + i)
        starts = torch.randint(0, max_start, (BATCH,), generator=rng)
        x = torch.stack([tokens[s : s + base.SEQ_LEN] for s in starts]).to(base.DEVICE)
        y = torch.stack([tokens[s + 1 : s + base.SEQ_LEN + 1] for s in starts]).to(
            base.DEVICE
        )
        logits = model(x, use_domain=use_domain)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += 1
    return math.exp(tot / n)


class GenericCausalABI(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        kwargs = {"local_files_only": True}
        if TORCH_DTYPE is not None:
            kwargs["torch_dtype"] = TORCH_DTYPE
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model.config.use_cache = False
        self.model = model
        self.lm_head = model.get_output_embeddings()
        self.d_model = config_int(model.config, "hidden_size", "n_embd", "d_model")
        self.vocab_size = config_int(model.config, "vocab_size")
        self.model_type = getattr(model.config, "model_type", "unknown")
        self.proj_in = nn.Linear(self.d_model, D_ABI, bias=False)
        self.abi_ln = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, self.d_model, bias=False)
        self.domain = base.DomainModule(D_ABI)
        self.domain_alpha = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def encode_core(self, x):
        out = self.model(
            input_ids=x,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        h = out.hidden_states[-1]
        h_abi = self.abi_ln(self.proj_in(h.to(self.proj_in.weight.dtype)))
        return h, h_abi

    def domain_delta_from_abi(self, h_abi):
        domain_in = (
            self.domain_bridge_in(h_abi)
            if hasattr(self, "domain_bridge_in")
            else h_abi
        )
        if hasattr(self, "domain_ensemble"):
            domain_deltas = torch.stack(
                [domain(domain_in) for domain in self.domain_ensemble],
                dim=0,
            )
            if hasattr(self, "domain_ensemble_logits"):
                weights = F.softmax(self.domain_ensemble_logits.float(), dim=0)
            elif hasattr(self, "domain_ensemble_weights"):
                weights = self.domain_ensemble_weights.float()
            else:
                weights = torch.full(
                    (len(self.domain_ensemble),),
                    1.0 / len(self.domain_ensemble),
                    device=domain_deltas.device,
                    dtype=torch.float32,
                )
            weight_shape = (weights.numel(),) + (1,) * (domain_deltas.dim() - 1)
            weights = weights.to(
                device=domain_deltas.device,
                dtype=domain_deltas.dtype,
            ).view(weight_shape)
            domain_delta = (domain_deltas * weights).sum(dim=0)
        else:
            domain_delta = self.domain(domain_in)
        if hasattr(self, "domain_bridge_out"):
            domain_delta = self.domain_bridge_out(domain_delta)
        if hasattr(self, "domain_residual_down"):
            residual_in = h_abi.to(self.domain_residual_down.weight.dtype)
            domain_residual = self.domain_residual_up(
                F.gelu(self.domain_residual_down(residual_in))
            )
            domain_delta = domain_delta + (
                self.domain_residual_scale * domain_residual
            ).to(domain_delta.dtype)
        return domain_delta

    def logits_from_abi_state(self, h, h_out):
        logit_h = self.proj_out(h_out) + h.to(self.proj_out.weight.dtype)
        if hasattr(self, "target_residual_down"):
            residual_in = logit_h.to(self.target_residual_down.weight.dtype)
            residual = self.target_residual_up(
                F.gelu(self.target_residual_down(residual_in))
            )
            logit_h = logit_h + (
                self.target_residual_scale * residual
            ).to(logit_h.dtype)
        logits = self.lm_head(logit_h.to(self.lm_head.weight.dtype))
        if hasattr(self, "logit_residual_down"):
            residual_in = h_out.to(self.logit_residual_down.weight.dtype)
            logit_delta = self.logit_residual_up(
                F.gelu(self.logit_residual_down(residual_in))
            )
            logits = logits + (
                self.target_residual_scale * logit_delta
            ).to(logits.dtype)
        logit_scale = float(getattr(self, "logit_scale_value", 1.0))
        if logit_scale != 1.0:
            logits = logits * logit_scale
        logit_bias = getattr(self, "logit_bias_value", None)
        if logit_bias is not None:
            logits = logits + logit_bias.to(device=logits.device, dtype=logits.dtype)
        return logits

    def forward_with_abi_states(self, x, use_domain=True):
        if use_domain == "base":
            out = self.model(
                input_ids=x,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            h = out.hidden_states[-1]
            logits = self.lm_head(h.to(self.lm_head.weight.dtype))
            return logits, None, None
        h, h_abi = self.encode_core(x)
        if use_domain:
            h_out = h_abi + self.domain_alpha * self.domain_delta_from_abi(h_abi)
        else:
            h_out = h_abi
        return self.logits_from_abi_state(h, h_out), h_abi, h_out

    def forward(self, x, use_domain=True):
        logits, _, _ = self.forward_with_abi_states(x, use_domain=use_domain)
        return logits


@torch.no_grad()
def distribution_stats_for_scale(
    native,
    calibrated,
    tokens,
    scale,
    n_chunks,
    seed,
    reference_forward_mode=None,
):
    native.eval()
    calibrated.eval()
    old_scale = float(getattr(calibrated, "logit_scale_value", 1.0))
    calibrated.logit_scale_value = float(scale)
    if reference_forward_mode is None:
        reference_forward_mode = TARGET_REFERENCE_FORWARD_MODE
    chunk = 512
    skip = 20
    rng = base.np.random.default_rng(seed)
    max_start = max(len(tokens) - chunk, 1)
    js_list, ent_list, nat_ent_list, cal_ent_list = [], [], [], []
    try:
        for _ in range(n_chunks):
            start = int(rng.integers(0, max_start))
            x = tokens[start : start + chunk].unsqueeze(0).to(base.DEVICE)
            nat_logits = native(x, use_domain=reference_forward_mode)[0, skip:, :]
            cal_logits = calibrated(x, use_domain=True)[0, skip:, :]
            nat_p = F.softmax(nat_logits, dim=-1).cpu().float().numpy()
            cal_p = F.softmax(cal_logits, dim=-1).cpu().float().numpy()
            eps = 1e-12
            m = 0.5 * (nat_p + cal_p)
            kl_n = (
                base.np.clip(nat_p, eps, 1)
                * base.np.log(
                    base.np.clip(nat_p / base.np.clip(m, eps, 1), eps, None)
                )
            ).sum(1)
            kl_c = (
                base.np.clip(cal_p, eps, 1)
                * base.np.log(
                    base.np.clip(cal_p / base.np.clip(m, eps, 1), eps, None)
                )
            ).sum(1)
            js_list.extend(base.np.clip(0.5 * (kl_n + kl_c), 0, None).tolist())
            Hn = -(
                base.np.clip(nat_p, eps, 1) * base.np.log(base.np.clip(nat_p, eps, 1))
            ).sum(1)
            Hc = -(
                base.np.clip(cal_p, eps, 1) * base.np.log(base.np.clip(cal_p, eps, 1))
            ).sum(1)
            ent_list.extend(base.np.abs(Hn - Hc).tolist())
            nat_ent_list.extend(Hn.tolist())
            cal_ent_list.extend(Hc.tolist())
    finally:
        calibrated.logit_scale_value = old_scale

    return {
        "scale": float(scale),
        "mean_js": float(base.np.mean(js_list)),
        "mean_entropy_diff": float(base.np.mean(ent_list)),
        "mean_native_entropy": float(base.np.mean(nat_ent_list)),
        "mean_calibrated_entropy": float(base.np.mean(cal_ent_list)),
    }


@torch.no_grad()
def aggregate_distribution_stats_for_scale(
    native,
    calibrated,
    tokens,
    scale,
    reference_forward_mode=None,
):
    repeats = [
        distribution_stats_for_scale(
            native,
            calibrated,
            tokens,
            scale=scale,
            n_chunks=POSTHOC_SCALE_CHUNKS,
            seed=CAL_SEED_BASE + 123456 + 100003 * repeat,
            reference_forward_mode=reference_forward_mode,
        )
        for repeat in range(POSTHOC_SCALE_REPEATS)
    ]
    return {
        "scale": float(scale),
        "mean_js": float(base.np.mean([row["mean_js"] for row in repeats])),
        "max_js": float(base.np.max([row["mean_js"] for row in repeats])),
        "mean_entropy_diff": float(
            base.np.mean([row["mean_entropy_diff"] for row in repeats])
        ),
        "max_entropy_diff": float(
            base.np.max([row["mean_entropy_diff"] for row in repeats])
        ),
        "mean_native_entropy": float(
            base.np.mean([row["mean_native_entropy"] for row in repeats])
        ),
        "mean_calibrated_entropy": float(
            base.np.mean([row["mean_calibrated_entropy"] for row in repeats])
        ),
    }


def posthoc_selection_score(row):
    if POSTHOC_SELECTION == "mean_entropy":
        return row["mean_entropy_diff"]
    if POSTHOC_SELECTION == "minimax_entropy":
        return row["max_entropy_diff"]
    signed_gap = abs(row["mean_calibrated_entropy"] - row["mean_native_entropy"])
    return row["max_entropy_diff"] + POSTHOC_SIGNED_ENTROPY_WEIGHT * signed_gap


def rounded_posthoc_selection(row):
    signed_gap = abs(row["mean_calibrated_entropy"] - row["mean_native_entropy"])
    return {
        "mean_js": round(row["mean_js"], 5),
        "max_js": round(row["max_js"], 5),
        "mean_entropy_diff": round(row["mean_entropy_diff"], 4),
        "max_entropy_diff": round(row["max_entropy_diff"], 4),
        "signed_entropy_gap": round(signed_gap, 4),
        "selection_score": round(posthoc_selection_score(row), 4),
        "mean_native_entropy": round(row["mean_native_entropy"], 4),
        "mean_calibrated_entropy": round(row["mean_calibrated_entropy"], 4),
    }


def selective_posthoc_score(row):
    score = posthoc_selection_score(row)
    off_domain = row.get("off_domain")
    if off_domain is not None:
        score += POSTHOC_SELECTIVE_WEIGHT * posthoc_selection_score(off_domain)
    return score


@torch.no_grad()
def calibrate_posthoc_logit_scale(native, calibrated, tokens, off_domain_tokens=None):
    if POSTHOC_LOGIT_SCALE == "none":
        calibrated.logit_scale_value = 1.0
        return {
            "mode": "none",
            "applied": False,
            "scale": 1.0,
        }

    log_min = math.log(POSTHOC_SCALE_MIN)
    log_max = math.log(POSTHOC_SCALE_MAX)
    candidates = base.np.exp(
        base.np.linspace(log_min, log_max, POSTHOC_SCALE_STEPS)
    ).tolist()
    rows = [
        aggregate_distribution_stats_for_scale(native, calibrated, tokens, scale=scale)
        for scale in candidates
    ]
    use_selective_objective = (
        SELECTIVE_TRANSFER_EVAL
        and POSTHOC_SELECTIVE_WEIGHT > 0
        and off_domain_tokens is not None
    )
    if use_selective_objective:
        off_reference_mode = selective_reference_forward_mode()
        for row in rows:
            row["off_domain"] = aggregate_distribution_stats_for_scale(
                native,
                calibrated,
                off_domain_tokens,
                scale=row["scale"],
                reference_forward_mode=off_reference_mode,
            )
    viable = [
        row
        for row in rows
        if row["mean_js"] < base.REGISTRY["js_threshold"]
        and (
            row.get("off_domain") is None
            or row["off_domain"]["mean_js"] < base.REGISTRY["js_threshold"]
        )
    ]
    pool = viable if viable else rows
    best = min(pool, key=selective_posthoc_score)
    calibrated.logit_scale_value = best["scale"]
    return {
        "mode": POSTHOC_LOGIT_SCALE,
        "applied": True,
        "scale": best["scale"],
        "candidate_count": len(rows),
        "calibration_chunks": POSTHOC_SCALE_CHUNKS,
        "calibration_repeats": POSTHOC_SCALE_REPEATS,
        "selection_rule": POSTHOC_SELECTION,
        "signed_entropy_weight": POSTHOC_SIGNED_ENTROPY_WEIGHT,
        "selective_weight": POSTHOC_SELECTIVE_WEIGHT,
        "selective_objective_applied": use_selective_objective,
        "scale_min": POSTHOC_SCALE_MIN,
        "scale_max": POSTHOC_SCALE_MAX,
        "selection_score": round(selective_posthoc_score(best), 4),
        "selection": rounded_posthoc_selection(best),
        "off_domain_selection": (
            rounded_posthoc_selection(best["off_domain"])
            if best.get("off_domain") is not None
            else None
        ),
    }


def calibrate_posthoc_logit_bias(native, calibrated, tokens):
    if POSTHOC_BIAS == "none" or POSTHOC_BIAS_STEPS == 0:
        calibrated.logit_bias_value = None
        return {
            "mode": POSTHOC_BIAS,
            "applied": False,
            "steps": POSTHOC_BIAS_STEPS,
        }

    native.eval()
    calibrated.eval()
    calibrated.logit_bias_value = None
    vocab_size = calibrated.vocab_size
    bias = torch.zeros(
        vocab_size,
        device=base.DEVICE,
        dtype=torch.float32,
        requires_grad=True,
    )
    opt = torch.optim.AdamW([bias], lr=POSTHOC_BIAS_LR, weight_decay=0.0)
    last_loss = None
    for step in range(POSTHOC_BIAS_STEPS):
        x, _ = make_batch(tokens, seed=CAL_SEED_BASE + 223000 + step)
        opt.zero_grad()
        with torch.no_grad():
            teacher_flat = native(
                x, use_domain=TARGET_REFERENCE_FORWARD_MODE
            ).reshape(-1, vocab_size).float()
            student_flat = calibrated(x, use_domain=True).reshape(-1, vocab_size).float()
            teacher_argmax = teacher_flat.argmax(dim=-1)
            teacher_probs = F.softmax(teacher_flat / KD_TEMP, dim=-1)
        biased = student_flat + bias.unsqueeze(0)
        kd = F.kl_div(
            F.log_softmax(biased / KD_TEMP, dim=-1),
            teacher_probs,
            reduction="batchmean",
        ) * (KD_TEMP ** 2)
        top1_ce = F.cross_entropy(biased, teacher_argmax)
        bias_l2 = bias.pow(2).mean()
        loss = kd + POSTHOC_BIAS_TOP1_CE_WEIGHT * top1_ce + POSTHOC_BIAS_L2 * bias_l2
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if (step + 1) % 100 == 0 or step + 1 == POSTHOC_BIAS_STEPS:
            print(
                f"  posthoc bias step {step + 1}/{POSTHOC_BIAS_STEPS}  "
                f"loss={last_loss:.4f}"
            )

    with torch.no_grad():
        bias_detached = bias.detach()
        calibrated.logit_bias_value = bias_detached
        return {
            "mode": POSTHOC_BIAS,
            "applied": True,
            "steps": POSTHOC_BIAS_STEPS,
            "lr": POSTHOC_BIAS_LR,
            "l2": POSTHOC_BIAS_L2,
            "top1_ce_weight": POSTHOC_BIAS_TOP1_CE_WEIGHT,
            "final_loss": round(last_loss, 5) if last_loss is not None else None,
            "bias_l2": round(float(bias_detached.pow(2).mean().cpu()), 6),
            "bias_max_abs": round(float(bias_detached.abs().max().cpu()), 6),
        }


@torch.no_grad()
def fit_alignment_map(src_vecs, tgt_vecs):
    A = src_vecs - src_vecs.mean(0)
    B = tgt_vecs - tgt_vecs.mean(0)
    if ALIGN_FIT_NORMALIZE == "zscore":
        A = A / A.std(0).clamp_min(1.0e-6)
        B = B / B.std(0).clamp_min(1.0e-6)
    U, _, Vh = torch.linalg.svd(A.T @ B)
    procrustes_map = U @ Vh
    if ALIGN_MAP in {"linear", "linear_blend"}:
        gram = A.T @ A
        if ALIGN_RIDGE > 0:
            gram = gram + ALIGN_RIDGE * torch.eye(
                gram.shape[0],
                device=gram.device,
                dtype=gram.dtype,
            )
        linear_map = torch.linalg.solve(gram, A.T @ B)
        if ALIGN_MAP == "linear":
            alignment_map = linear_map
        else:
            alignment_map = (
                (1.0 - ALIGN_LINEAR_BLEND) * procrustes_map
                + ALIGN_LINEAR_BLEND * linear_map
            )
    else:
        linear_map = None
        alignment_map = procrustes_map
    before = F.cosine_similarity(A, B, dim=1)
    after = F.cosine_similarity(A @ alignment_map, B, dim=1)
    map_info = {
        "align_map": ALIGN_MAP,
        "align_ridge": ALIGN_RIDGE,
        "align_linear_blend": ALIGN_LINEAR_BLEND,
    }
    if linear_map is not None:
        map_info.update(
            {
                "linear_map_fro_norm": round(float(linear_map.norm().item()), 4),
                "procrustes_map_fro_norm": round(
                    float(procrustes_map.norm().item()), 4
                ),
                "selected_map_fro_norm": round(
                    float(alignment_map.norm().item()), 4
                ),
            }
        )
    return alignment_map, before, after, map_info


def alignment_collection_limit():
    if ALIGN_SELECT == "procrustes_trim":
        return ALIGN_POOL_SENTENCES
    return N_ALIGN_SENTENCES


def rotation_alignment_collection_limit():
    return alignment_collection_limit() + (
        ROTATION_ENSEMBLE_SIZE - 1
    ) * ROTATION_ENSEMBLE_STRIDE


@torch.no_grad()
def fit_selected_procrustes(src_vecs, tgt_vecs, label):
    n = min(len(src_vecs), len(tgt_vecs))
    if n < 1:
        raise RuntimeError("No valid Procrustes alignment pairs were collected")
    pool_src = torch.stack(src_vecs[:n])
    pool_tgt = torch.stack(tgt_vecs[:n])
    keep_n = min(N_ALIGN_SENTENCES, n)
    info = {
        "mode": ALIGN_SELECT,
        "label": label,
        "pool_pairs": int(n),
        "final_pairs": int(keep_n),
        "requested_final_pairs": int(N_ALIGN_SENTENCES),
        "pool_requested": int(alignment_collection_limit()),
        "min_chars": int(ALIGN_MIN_CHARS),
        "fit_normalize": ALIGN_FIT_NORMALIZE,
        "align_map": ALIGN_MAP,
        "align_ridge": ALIGN_RIDGE,
        "align_linear_blend": ALIGN_LINEAR_BLEND,
    }
    if ALIGN_SELECT == "procrustes_trim" and n > keep_n:
        _R0, before0, after0, _map_info0 = fit_alignment_map(pool_src, pool_tgt)
        keep_idx = torch.topk(after0, k=keep_n).indices
        info.update(
            {
                "initial_before_mean": round(float(before0.mean().item()), 4),
                "initial_after_mean": round(float(after0.mean().item()), 4),
                "trim_kept_after_mean": round(
                    float(after0.index_select(0, keep_idx).mean().item()), 4
                ),
                "trim_dropped_after_mean": round(
                    float(after0.index_select(0, torch.topk(after0, k=n - keep_n, largest=False).indices).mean().item()),
                    4,
                ),
            }
        )
        fit_src = pool_src.index_select(0, keep_idx)
        fit_tgt = pool_tgt.index_select(0, keep_idx)
    else:
        fit_src = pool_src[:keep_n]
        fit_tgt = pool_tgt[:keep_n]
    R, before, after, map_info = fit_alignment_map(fit_src, fit_tgt)
    info.update(map_info)
    info.update(
        {
            "final_before_mean": round(float(before.mean().item()), 4),
            "final_after_mean": round(float(after.mean().item()), 4),
            "final_after_min": round(float(after.min().item()), 4),
        }
    )
    print(
        f"  [Procrustes] Using {keep_n}/{n} {label} pairs "
        f"(mode={ALIGN_SELECT}, fit={ALIGN_FIT_NORMALIZE}, map={ALIGN_MAP})"
    )
    if "initial_after_mean" in info:
        print(
            f"  [Procrustes] initial cos: "
            f"{info['initial_before_mean']:.4f} -> {info['initial_after_mean']:.4f}; "
            f"kept mean={info['trim_kept_after_mean']:.4f}"
        )
    print(
        f"  [Procrustes] final cos: "
        f"{info['final_before_mean']:.4f} -> {info['final_after_mean']:.4f}"
    )
    return R.to(base.DEVICE), info


@torch.no_grad()
def procrustes(src_model, tgt_model, align_sentences, src_tok, tgt_tok):
    src_model.eval()
    tgt_model.eval()
    src_vecs, tgt_vecs = [], []
    collect_n = alignment_collection_limit()
    for sent in align_sentences:
        sent = sent.strip()
        if len(sent) < ALIGN_MIN_CHARS:
            continue
        try:
            ids_src = src_tok(
                sent, return_tensors="pt", truncation=True, max_length=128
            )["input_ids"].to(base.DEVICE)
            ids_tgt = tgt_tok(
                sent, return_tensors="pt", truncation=True, max_length=128
            )["input_ids"].to(base.DEVICE)
            if ids_src.shape[1] < 4 or ids_tgt.shape[1] < 4:
                continue
            _, h_src = src_model.encode_core(ids_src)
            _, h_tgt = tgt_model.encode_core(ids_tgt)
            src_vecs.append(h_src[0].mean(0).cpu().float())
            tgt_vecs.append(h_tgt[0].mean(0).cpu().float())
        except Exception:
            continue
        if len(src_vecs) >= collect_n:
            break

    return fit_selected_procrustes(src_vecs, tgt_vecs, "sentence")


@torch.no_grad()
def collect_source_alignment_pairs(src_model, align_sentences, src_tok):
    src_model.eval()
    pairs = []
    for sent in align_sentences:
        sent = sent.strip()
        if len(sent) < ALIGN_MIN_CHARS:
            continue
        try:
            ids_src = src_tok(
                sent, return_tensors="pt", truncation=True, max_length=128
            )["input_ids"].to(base.DEVICE)
            if ids_src.shape[1] < 4:
                continue
            _, h_src = src_model.encode_core(ids_src)
            pairs.append((sent, h_src[0].mean(0).cpu().float()))
        except Exception:
            continue
        if len(pairs) >= rotation_alignment_collection_limit():
            break
    print(f"  [Procrustes] Cached {len(pairs)} source sentence vectors")
    return pairs


@torch.no_grad()
def procrustes_from_cached_source(source_pairs, tgt_model, tgt_tok):
    tgt_model.eval()
    src_vecs, tgt_vecs = [], []
    for sent, src_vec in source_pairs:
        try:
            ids_tgt = tgt_tok(
                sent, return_tensors="pt", truncation=True, max_length=128
            )["input_ids"].to(base.DEVICE)
            if ids_tgt.shape[1] < 4:
                continue
            _, h_tgt = tgt_model.encode_core(ids_tgt)
            src_vecs.append(src_vec)
            tgt_vecs.append(h_tgt[0].mean(0).cpu().float())
        except Exception:
            continue

    return fit_selected_procrustes(src_vecs, tgt_vecs, "cached source/target")


@torch.no_grad()
def l2_logit_test(
    native,
    calibrated,
    py_ids_tgt,
    reference_forward_mode=None,
    calibrated_forward_mode=True,
    seed=None,
    label="NIB",
):
    native.eval()
    calibrated.eval()
    if reference_forward_mode is None:
        reference_forward_mode = TARGET_REFERENCE_FORWARD_MODE
    if seed is None:
        seed = NIB_SEED
    chunk = 512
    skip = 20
    rng = base.np.random.default_rng(seed)
    n_chunks = base.REGISTRY["n_logit_chunks"]
    max_start = max(len(py_ids_tgt) - chunk, 1)
    js_list, top1_list, top5_list, ent_list = [], [], [], []

    for ci in range(n_chunks):
        start = int(rng.integers(0, max_start))
        x = py_ids_tgt[start : start + chunk].unsqueeze(0).to(base.DEVICE)
        nat_logits = native(x, use_domain=reference_forward_mode)[0, skip:, :]
        cal_logits = calibrated(x, use_domain=calibrated_forward_mode)[0, skip:, :]
        nat_p = F.softmax(nat_logits, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_logits, dim=-1).cpu().float().numpy()
        eps = 1e-12
        m = 0.5 * (nat_p + cal_p)
        kl_n = (
            base.np.clip(nat_p, eps, 1)
            * base.np.log(base.np.clip(nat_p / base.np.clip(m, eps, 1), eps, None))
        ).sum(1)
        kl_c = (
            base.np.clip(cal_p, eps, 1)
            * base.np.log(base.np.clip(cal_p / base.np.clip(m, eps, 1), eps, None))
        ).sum(1)
        js_list.extend(base.np.clip(0.5 * (kl_n + kl_c), 0, None).tolist())
        top1_list.extend((nat_p.argmax(1) == cal_p.argmax(1)).tolist())
        n5 = base.np.argpartition(nat_p, -5, axis=1)[:, -5:]
        c5 = base.np.argpartition(cal_p, -5, axis=1)[:, -5:]
        for t in range(nat_p.shape[0]):
            top5_list.append(len(set(n5[t]) & set(c5[t])) / 5.0)
        Hn = -(base.np.clip(nat_p, eps, 1) * base.np.log(base.np.clip(nat_p, eps, 1))).sum(1)
        Hc = -(base.np.clip(cal_p, eps, 1) * base.np.log(base.np.clip(cal_p, eps, 1))).sum(1)
        ent_list.extend(base.np.abs(Hn - Hc).tolist())
        print(
            f"    {label} chunk {ci + 1}/{n_chunks}: "
            f"JS={float(base.np.mean(js_list)):.4f} "
            f"top1={float(base.np.mean(top1_list)):.3f} "
            f"top5={float(base.np.mean(top5_list)):.3f}"
        )

    mj = float(base.np.mean(js_list))
    mt1 = float(base.np.mean(top1_list))
    mt5 = float(base.np.mean(top5_list))
    me = float(base.np.mean(ent_list))
    return {
        "reference_mode": ORACLE_MODE,
        "reference_use_domain": reference_forward_mode is True,
        "reference_bypass_abi": reference_forward_mode == "base",
        "reference_forward_mode": reference_forward_mode,
        "calibrated_forward_mode": calibrated_forward_mode,
        "target_native_oracle_required": TARGET_NATIVE_ORACLE_REQUIRED,
        "n_positions": len(js_list),
        "mean_js": round(mj, 5),
        "mean_top1_agree": round(mt1, 4),
        "mean_top5_overlap": round(mt5, 4),
        "mean_entropy_diff": round(me, 4),
        "js_pass": mj < base.REGISTRY["js_threshold"],
        "top1_pass": mt1 >= base.REGISTRY["top1_threshold"],
        "top5_pass": mt5 >= base.REGISTRY["top5_threshold"],
        "entropy_pass": me < base.REGISTRY["entropy_diff_threshold"],
        "pass": (
            mj < base.REGISTRY["js_threshold"]
            and mt1 >= base.REGISTRY["top1_threshold"]
            and mt5 >= base.REGISTRY["top5_threshold"]
            and me < base.REGISTRY["entropy_diff_threshold"]
        ),
    }


@torch.no_grad()
def validation_l2_stats(native, calibrated, tokens, n_chunks, seed):
    native.eval()
    calibrated.eval()
    chunk = 512
    skip = 20
    rng = base.np.random.default_rng(seed)
    max_start = max(len(tokens) - chunk, 1)
    js_list, top1_list, top5_list, ent_list = [], [], [], []
    for _ in range(n_chunks):
        start = int(rng.integers(0, max_start))
        x = tokens[start : start + chunk].unsqueeze(0).to(base.DEVICE)
        nat_logits = native(x, use_domain=TARGET_REFERENCE_FORWARD_MODE)[0, skip:, :]
        cal_logits = calibrated(x, use_domain=True)[0, skip:, :]
        nat_p = F.softmax(nat_logits, dim=-1).cpu().float().numpy()
        cal_p = F.softmax(cal_logits, dim=-1).cpu().float().numpy()
        eps = 1e-12
        m = 0.5 * (nat_p + cal_p)
        kl_n = (
            base.np.clip(nat_p, eps, 1)
            * base.np.log(base.np.clip(nat_p / base.np.clip(m, eps, 1), eps, None))
        ).sum(1)
        kl_c = (
            base.np.clip(cal_p, eps, 1)
            * base.np.log(base.np.clip(cal_p / base.np.clip(m, eps, 1), eps, None))
        ).sum(1)
        js_list.extend(base.np.clip(0.5 * (kl_n + kl_c), 0, None).tolist())
        top1_list.extend((nat_p.argmax(1) == cal_p.argmax(1)).tolist())
        n5 = base.np.argpartition(nat_p, -5, axis=1)[:, -5:]
        c5 = base.np.argpartition(cal_p, -5, axis=1)[:, -5:]
        for t in range(nat_p.shape[0]):
            top5_list.append(len(set(n5[t]) & set(c5[t])) / 5.0)
        Hn = -(
            base.np.clip(nat_p, eps, 1) * base.np.log(base.np.clip(nat_p, eps, 1))
        ).sum(1)
        Hc = -(
            base.np.clip(cal_p, eps, 1) * base.np.log(base.np.clip(cal_p, eps, 1))
        ).sum(1)
        ent_list.extend(base.np.abs(Hn - Hc).tolist())
    return {
        "mean_js": float(base.np.mean(js_list)),
        "mean_top1_agree": float(base.np.mean(top1_list)),
        "mean_top5_overlap": float(base.np.mean(top5_list)),
        "mean_entropy_diff": float(base.np.mean(ent_list)),
    }


def calibration_selection_score(stats):
    return (
        CAL_SELECT_TOP1_WEIGHT * stats["mean_top1_agree"]
        + CAL_SELECT_TOP5_WEIGHT * stats["mean_top5_overlap"]
        - CAL_SELECT_JS_WEIGHT * stats["mean_js"]
        - CAL_SELECT_ENTROPY_WEIGHT * stats["mean_entropy_diff"]
    )


def final_selection_score(stats):
    return (
        CAL_FINAL_SELECT_TOP1_WEIGHT * stats["mean_top1_agree"]
        + CAL_FINAL_SELECT_TOP5_WEIGHT * stats["mean_top5_overlap"]
        - CAL_FINAL_SELECT_JS_WEIGHT * stats["mean_js"]
        - CAL_FINAL_SELECT_ENTROPY_WEIGHT * stats["mean_entropy_diff"]
    )


def round_validation_row(stats, score, extra=None):
    row = {
        "score": round(float(score), 6),
        "mean_js": round(stats["mean_js"], 5),
        "mean_top1_agree": round(stats["mean_top1_agree"], 4),
        "mean_top5_overlap": round(stats["mean_top5_overlap"], 4),
        "mean_entropy_diff": round(stats["mean_entropy_diff"], 4),
    }
    if extra:
        row.update(extra)
    return row


def average_validation_stats(rows):
    keys = ("mean_js", "mean_top1_agree", "mean_top5_overlap", "mean_entropy_diff")
    return {key: float(base.np.mean([row[key] for row in rows])) for key in keys}


def min_validation_stats(rows):
    keys = ("mean_top1_agree", "mean_top5_overlap")
    inverse_keys = ("mean_js", "mean_entropy_diff")
    stats = {key: float(min(row[key] for row in rows)) for key in keys}
    stats.update({key: float(max(row[key] for row in rows)) for key in inverse_keys})
    return stats


def reduce_validation_scores(scores):
    if CAL_SELECT_SOUP_GRID_REDUCTION == "min_score":
        return float(min(scores))
    return float(base.np.mean(scores))


def reduce_checkpoint_validation_scores(scores):
    if CAL_SELECT_REDUCTION == "min_score":
        return float(min(scores))
    return float(base.np.mean(scores))


@torch.no_grad()
def evaluate_selection_checkpoint(native, calibrated, tokens, step):
    repeat_rows = []
    repeat_scores = []
    seed_base = CAL_SEED_BASE + 331000 + int(step)
    for repeat_idx in range(CAL_SELECT_REPEATS):
        stats = validation_l2_stats(
            native,
            calibrated,
            tokens,
            n_chunks=CAL_SELECT_CHUNKS,
            seed=seed_base + 1009 * repeat_idx,
        )
        score = calibration_selection_score(stats)
        repeat_rows.append(round_validation_row(stats, score))
        repeat_scores.append(score)
    stats = average_validation_stats(repeat_rows)
    score = reduce_checkpoint_validation_scores(repeat_scores)
    return stats, score, repeat_rows


def normalized_soup_weights(n_avg):
    if CAL_SELECT_SOUP_WEIGHTS:
        weights = CAL_SELECT_SOUP_WEIGHTS[:n_avg]
        if len(weights) < n_avg:
            weights = weights + [0.0] * (n_avg - len(weights))
        total_weight = sum(weights)
        if total_weight <= 0:
            return [1.0 / n_avg] * n_avg
        return [weight / total_weight for weight in weights]
    return [1.0 / n_avg] * n_avg


def clone_calibration_state(cal_params):
    return [p.detach().cpu().clone() for p in cal_params]


@torch.no_grad()
def restore_calibration_state(cal_params, state):
    for param, saved in zip(cal_params, state):
        param.copy_(saved.to(device=param.device, dtype=param.dtype))


@torch.no_grad()
def restore_weighted_calibration_state(cal_params, ranked_states, weights):
    for idx, param in enumerate(cal_params):
        avg = sum(
            weight * item["state"][idx].float()
            for weight, item in zip(weights, ranked_states)
        )
        param.copy_(avg.to(device=param.device, dtype=param.dtype))


def evaluate_final_candidate(native, calibrated, cal_params, state, tokens, seed_base):
    current_state = clone_calibration_state(cal_params)
    restore_calibration_state(cal_params, state)
    calibrated.eval()
    repeat_rows = []
    repeat_scores = []
    for repeat_idx in range(CAL_FINAL_SELECT_REPEATS):
        stats = validation_l2_stats(
            native,
            calibrated,
            tokens,
            n_chunks=CAL_FINAL_SELECT_CHUNKS,
            seed=seed_base + 1009 * repeat_idx,
        )
        score = final_selection_score(stats)
        repeat_rows.append(round_validation_row(stats, score))
        repeat_scores.append(score)
    stats = average_validation_stats(repeat_rows)
    score = float(base.np.mean(repeat_scores))
    restore_calibration_state(cal_params, current_state)
    calibrated.train()
    return stats, score, repeat_rows


def normalize_token_surface(text):
    text = text.replace("\u0120", " ").replace("\u010a", " ")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def build_source_preservation_prompts(eval_text, eval_sentences):
    prompts = []
    seen = set()
    candidates = [sent.strip() for sent in eval_sentences if sent.strip()]
    if not candidates:
        candidates = [line.strip() for line in eval_text.splitlines() if line.strip()]
    for candidate in candidates:
        if len(candidate) < SOURCE_PRESERVATION_PREFIX_CHARS:
            prompt = candidate
        else:
            prompt = candidate[:SOURCE_PRESERVATION_PREFIX_CHARS].rsplit(" ", 1)[0]
        prompt = prompt.strip()
        if len(prompt) < 20:
            continue
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        prompts.append(prompt)
        if len(prompts) >= SOURCE_PRESERVATION_PROMPTS:
            break
    return prompts


@torch.no_grad()
def decoded_next_token_surfaces(model, tokenizer, prompt, topk):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=SOURCE_PRESERVATION_MAX_LENGTH,
    )
    ids = encoded["input_ids"].to(base.DEVICE)
    if ids.shape[1] < 2:
        return None
    logits = model(ids, use_domain=True)[0, -1, :].float()
    k = min(topk, logits.shape[-1])
    top = logits.topk(k)
    rows = []
    seen = set()
    for token_id, logit in zip(top.indices.tolist(), top.values.tolist()):
        decoded = tokenizer.decode(
            [int(token_id)],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
        surface = normalize_token_surface(decoded)
        if not surface or surface in seen:
            continue
        seen.add(surface)
        rows.append(
            {
                "rank": len(rows) + 1,
                "token_id": int(token_id),
                "decoded_text": decoded,
                "surface": surface,
                "logit": round(float(logit), 6),
            }
        )
    return rows


@torch.no_grad()
def target_completion_score(model, tokenizer, prompt, continuation):
    if not continuation:
        return None
    prompt_encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=SOURCE_PRESERVATION_MAX_LENGTH,
    )
    continuation_encoded = tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_ids = prompt_encoded["input_ids"].to(base.DEVICE)
    continuation_ids = continuation_encoded["input_ids"].to(base.DEVICE)
    prompt_len = int(prompt_ids.shape[1])
    continuation_len = int(continuation_ids.shape[1])
    if prompt_len < 1 or continuation_len < 1:
        return None
    combined_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    logits = model(combined_ids, use_domain=True).float()
    continuation_logits = logits[
        :, prompt_len - 1 : prompt_len + continuation_len - 1, :
    ]
    logp = F.log_softmax(continuation_logits, dim=-1)
    token_logp = logp.gather(-1, continuation_ids.unsqueeze(-1)).squeeze(-1)
    token_count = int(continuation_ids.numel())
    total = float(token_logp.sum().item())
    return {
        "token_count": token_count,
        "sum_logprob": round(total, 6),
        "mean_logprob": round(total / token_count, 6),
    }


def target_completion_mean_logprob_tensor(model, tokenizer, prompt, continuation):
    if not continuation:
        return None
    prompt_encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=SOURCE_PRESERVATION_MAX_LENGTH,
    )
    continuation_encoded = tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_ids = prompt_encoded["input_ids"].to(base.DEVICE)
    continuation_ids = continuation_encoded["input_ids"].to(base.DEVICE)
    prompt_len = int(prompt_ids.shape[1])
    continuation_len = int(continuation_ids.shape[1])
    if prompt_len < 1 or continuation_len < 1:
        return None
    combined_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    logits = model(combined_ids, use_domain=True).float()
    continuation_logits = logits[
        :, prompt_len - 1 : prompt_len + continuation_len - 1, :
    ]
    logp = F.log_softmax(continuation_logits, dim=-1)
    token_logp = logp.gather(-1, continuation_ids.unsqueeze(-1)).squeeze(-1)
    return token_logp.mean()


def rank_source_continuations_under_target(source_top, model, tokenizer, prompt):
    rows = []
    seen = set()
    for source_row in source_top:
        continuation = source_row.get("decoded_text") or source_row.get("surface")
        surface = source_row.get("surface")
        if not continuation or not surface or surface in seen:
            continue
        seen.add(surface)
        score = target_completion_score(model, tokenizer, prompt, continuation)
        if score is None:
            continue
        rows.append(
            {
                "source_rank": source_row["rank"],
                "source_token_id": source_row["token_id"],
                "surface": surface,
                "decoded_text": continuation,
                **score,
            }
        )
    rows.sort(key=lambda row: row["mean_logprob"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["target_completion_rank"] = idx
    return rows


def prepare_source_completion_loss_records(source_records):
    if SOURCE_COMPLETION_LOSS_WEIGHT <= 0:
        return []
    prepared = []
    for source_record in source_records:
        prompt = source_record.get("_prompt")
        if not prompt:
            continue
        seen = set()
        candidates = []
        for source_row in source_record.get("source_top", []):
            continuation = source_row.get("decoded_text") or source_row.get("surface")
            surface = source_row.get("surface")
            if not continuation or not surface or surface in seen:
                continue
            seen.add(surface)
            candidates.append(
                {
                    "source_rank": int(source_row["rank"]),
                    "surface": surface,
                    "decoded_text": continuation,
                }
            )
            if len(candidates) >= SOURCE_COMPLETION_LOSS_CANDIDATES:
                break
        if len(candidates) < 2 or candidates[0]["source_rank"] != 1:
            continue
        prepared.append(
            {
                "_prompt": prompt,
                "prompt_sha1": source_record["prompt_sha1"],
                "candidates": candidates,
            }
        )
    if SOURCE_COMPLETION_LOSS_PROMPTS > 0:
        prepared = prepared[:SOURCE_COMPLETION_LOSS_PROMPTS]
    return prepared


def collect_source_preservation_source(model, tokenizer, prompts):
    records = []
    model.eval()
    for prompt in prompts:
        top = decoded_next_token_surfaces(
            model, tokenizer, prompt, SOURCE_PRESERVATION_TOPK
        )
        if not top:
            continue
        records.append(
            {
                "_prompt": prompt,
                "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
                "prompt_chars": len(prompt),
                "source_top": top,
            }
        )
    return records


def evaluate_source_preservation_target(source_records, model, tokenizer):
    if not source_records:
        return {
            "enabled": SOURCE_PRESERVATION_EVAL,
            "measured": False,
            "reason": "no_source_records",
        }
    records = []
    top1_matches = 0
    source_top1_in_target_topk = 0
    overlap_scores = []
    completion_records = 0
    source_top1_completion_wins = 0
    source_top1_completion_ranks = []
    source_top1_completion_margins = []
    completion_candidate_counts = []
    model.eval()
    for source_record in source_records:
        prompt_digest = source_record["prompt_sha1"]
        # The prompt text itself is intentionally not stored in the final JSON.
        # Re-evaluation uses only records collected within the same run.
        prompt = source_record.get("_prompt")
        if prompt is None:
            continue
        target_top = decoded_next_token_surfaces(
            model, tokenizer, prompt, SOURCE_PRESERVATION_TOPK
        )
        if not target_top:
            continue
        source_surfaces = [row["surface"] for row in source_record["source_top"]]
        target_surfaces = [row["surface"] for row in target_top]
        source_set = set(source_surfaces)
        target_set = set(target_surfaces)
        top1_match = bool(
            source_surfaces
            and target_surfaces
            and source_surfaces[0] == target_surfaces[0]
        )
        top1_in_topk = bool(source_surfaces and source_surfaces[0] in target_set)
        overlap = len(source_set & target_set) / max(len(source_set), 1)
        top1_matches += int(top1_match)
        source_top1_in_target_topk += int(top1_in_topk)
        overlap_scores.append(overlap)
        completion_ranking = []
        source_top1_completion_rank = None
        source_top1_completion_best = None
        source_top1_completion_margin = None
        if SOURCE_PRESERVATION_COMPLETION_EVAL:
            completion_ranking = rank_source_continuations_under_target(
                source_record["source_top"], model, tokenizer, prompt
            )
            if completion_ranking:
                completion_records += 1
                completion_candidate_counts.append(len(completion_ranking))
                best_completion = completion_ranking[0]
                source_top1_completion_best = (
                    best_completion.get("source_rank") == 1
                )
                source_top1_completion_wins += int(source_top1_completion_best)
                source_top1_row = next(
                    (
                        row
                        for row in completion_ranking
                        if row.get("source_rank") == 1
                    ),
                    None,
                )
                if source_top1_row is not None:
                    source_top1_completion_rank = source_top1_row[
                        "target_completion_rank"
                    ]
                    source_top1_completion_ranks.append(
                        source_top1_completion_rank
                    )
                    source_top1_completion_margin = (
                        source_top1_row["mean_logprob"]
                        - best_completion["mean_logprob"]
                    )
                    source_top1_completion_margins.append(
                        source_top1_completion_margin
                    )
        records.append(
            {
                "prompt_sha1": prompt_digest,
                "source_top": source_record["source_top"],
                "target_top": target_top,
                "top1_surface_match": top1_match,
                "source_top1_in_target_topk": top1_in_topk,
                "topk_surface_overlap": round(float(overlap), 6),
                "target_source_completion_ranking": completion_ranking,
                "source_top1_completion_best": source_top1_completion_best,
                "source_top1_completion_rank": source_top1_completion_rank,
                "source_top1_completion_margin_vs_best": (
                    round(float(source_top1_completion_margin), 6)
                    if source_top1_completion_margin is not None
                    else None
                ),
            }
        )
    n = len(records)
    return {
        "enabled": SOURCE_PRESERVATION_EVAL,
        "measured": n > 0,
        "probe_type": "decoded_next_token_surface_overlap",
        "prompt_count_requested": SOURCE_PRESERVATION_PROMPTS,
        "prompt_count_measured": n,
        "topk": SOURCE_PRESERVATION_TOPK,
        "max_length": SOURCE_PRESERVATION_MAX_LENGTH,
        "prefix_chars": SOURCE_PRESERVATION_PREFIX_CHARS,
        "cross_tokenizer_completion_eval": SOURCE_PRESERVATION_COMPLETION_EVAL,
        "top1_surface_agree": round(top1_matches / n, 6) if n else None,
        "source_top1_in_target_topk": (
            round(source_top1_in_target_topk / n, 6) if n else None
        ),
        "mean_topk_surface_overlap": (
            round(float(base.np.mean(overlap_scores)), 6) if overlap_scores else None
        ),
        "completion_prompt_count_measured": completion_records,
        "source_top1_completion_preferred": (
            round(source_top1_completion_wins / completion_records, 6)
            if completion_records
            else None
        ),
        "mean_source_top1_completion_rank": (
            round(float(base.np.mean(source_top1_completion_ranks)), 6)
            if source_top1_completion_ranks
            else None
        ),
        "mean_source_top1_completion_margin_vs_best": (
            round(float(base.np.mean(source_top1_completion_margins)), 6)
            if source_top1_completion_margins
            else None
        ),
        "mean_completion_candidate_count": (
            round(float(base.np.mean(completion_candidate_counts)), 6)
            if completion_candidate_counts
            else None
        ),
        "records": records,
    }


def main():
    t_global = time.time()
    banner(f"Generic causal ABI/NIB v2: {SOURCE_MODEL_ID} -> {TARGET_MODEL_ID}")
    print(f"  Device:          {base.DEVICE}")
    print(f"  Source:          {SOURCE_MODEL_ID}")
    print(f"  Target:          {TARGET_MODEL_ID}")
    print(f"  Source tokenizer:{SOURCE_TOKENIZER_ID}")
    print(f"  Target tokenizer:{TARGET_TOKENIZER_ID}")
    print(f"  D_ABI:           {D_ABI}")
    print(f"  Domain steps:    {DOMAIN_STEPS}")
    print(f"  Calibration:     {CALIBRATION_STEPS} steps")
    print(
        f"  Cal LR:          {CAL_LR}  decay step={CAL_LR_DECAY_STEP}  "
        f"factor={CAL_LR_DECAY_FACTOR}  schedule={CAL_LR_SCHEDULE}  "
        f"final_factor={CAL_LR_FINAL_FACTOR}"
    )
    print(
        f"  Cal accum:       steps={CAL_ACCUM_STEPS}  "
        f"seed_stride={CAL_ACCUM_SEED_STRIDE}"
    )
    print(f"  Cal mode:        {CAL_MODE}")
    print(f"  Cal init:        {CAL_INIT}")
    print(
        f"  Oracle mode:     {ORACLE_MODE}  "
        f"reference_use_domain={TARGET_REFERENCE_USES_DOMAIN}  "
        f"reference_bypass_abi={TARGET_REFERENCE_BYPASS_ABI}"
    )
    print(
        f"  Target cache:    mode={TARGET_INTERFACE_CACHE_MODE}  "
        f"path={TARGET_INTERFACE_CACHE_PATH_RAW or 'default'}"
    )
    print(
        f"  Cal selection:   {CAL_SELECT}  every={CAL_SELECT_EVERY}  "
        f"chunks={CAL_SELECT_CHUNKS}  repeats={CAL_SELECT_REPEATS}  "
        f"reduction={CAL_SELECT_REDUCTION}  avg_top_n={CAL_SELECT_AVG_TOP_N}  "
        f"soup_weights={CAL_SELECT_SOUP_WEIGHTS or 'equal'}  "
        f"soup_weight_grid={CAL_SELECT_SOUP_WEIGHT_GRID or 'none'}  "
        f"grid_repeats={CAL_SELECT_SOUP_GRID_REPEATS}  "
        f"grid_reduction={CAL_SELECT_SOUP_GRID_REDUCTION}"
    )
    print(
        f"  Cal sel audit:   nib={CAL_SELECT_AUDIT_NIB}  "
        f"max_ranked={CAL_SELECT_AUDIT_MAX or 'all'}"
    )
    print(
        f"  Cal soup audit:  weights={CAL_SELECT_SOUP_AUDIT_WEIGHTS or 'none'}"
    )
    print(
        f"  Cal EMA:         decay={CAL_EMA_DECAY}  "
        f"start={CAL_EMA_START_STEP}  every={CAL_EMA_EVERY}  "
        f"as_candidate={CAL_EMA_AS_CANDIDATE}  restore={CAL_EMA_RESTORE}"
    )
    print(
        f"  Final selector:  {CAL_FINAL_SELECT}  "
        f"candidates={CAL_FINAL_SELECT_CANDIDATES or 'none'}  "
        f"chunks={CAL_FINAL_SELECT_CHUNKS}  repeats={CAL_FINAL_SELECT_REPEATS}  "
        f"weights=(top1={CAL_FINAL_SELECT_TOP1_WEIGHT}, "
        f"top5={CAL_FINAL_SELECT_TOP5_WEIGHT}, js={CAL_FINAL_SELECT_JS_WEIGHT}, "
        f"ent={CAL_FINAL_SELECT_ENTROPY_WEIGHT})"
    )
    print(
        f"  Final audit:     nib={CAL_FINAL_AUDIT_NIB}  "
        f"candidates={CAL_FINAL_AUDIT_CANDIDATES or 'none'}"
    )
    print(
        f"  Final soup audit:candidates={CAL_FINAL_SOUP_AUDIT_CANDIDATES or 'none'}  "
        f"weights={CAL_FINAL_SOUP_AUDIT_WEIGHTS or 'none'}"
    )
    print(
        f"  Temporal avg:    start={CAL_TEMPORAL_AVG_START_STEP}  "
        f"restore={CAL_TEMPORAL_AVG_RESTORE}"
    )
    print(f"  KD weight/temp:  {KD_WEIGHT}/{KD_TEMP}")
    print(f"  Top-k KD:        weight={TOPK_KD_WEIGHT}  k={TOPK}")
    print(
        f"  Union top-k KD:  weight={UNION_TOPK_KD_WEIGHT}  "
        f"k={UNION_TOPK}  temp={UNION_TOPK_TEMP}"
    )
    print(
        f"  Rank margin:     weight={RANK_MARGIN_WEIGHT}  "
        f"top_pos={RANK_TOP_POS}  neg_k={RANK_NEG_K}  margin={RANK_MARGIN}"
    )
    print(f"  Hard negatives:  weight={HARD_NEG_WEIGHT}  k={HARD_NEG_K}")
    print(f"  Top1 hard neg:   weight={TOP1_HARD_NEG_WEIGHT}  k={TOP1_HARD_NEG_K}")
    print(
        f"  Stable top1:     ce={STABLE_TOP1_CE_WEIGHT}  "
        f"hard_neg={STABLE_TOP1_HARD_NEG_WEIGHT}  "
        f"k={STABLE_TOP1_HARD_NEG_K}  "
        f"base_agree={STABLE_TOP1_REQUIRE_BASE_AGREE}  "
        f"min_margin={STABLE_TOP1_MIN_MARGIN}"
    )
    print(f"  Top-set loss:    weight={TOPSET_WEIGHT}  k={TOPSET_K}  temp={TOPSET_TEMP}")
    print(f"  Top-logit MSE:   weight={TOP_LOGIT_MSE_WEIGHT}  k={TOP_LOGIT_MSE_K}")
    print(
        f"  Domain-delta MSE:weight={DOMAIN_DELTA_LOGIT_MSE_WEIGHT}  "
        f"k={DOMAIN_DELTA_LOGIT_MSE_K}  center={DOMAIN_DELTA_LOGIT_MSE_CENTER}"
    )
    print(f"  Entropy match:   weight={ENTROPY_WEIGHT}")
    print(
        f"  ABI-state MSE:   pre={ABI_PRE_MSE_WEIGHT}  "
        f"post={ABI_STATE_MSE_WEIGHT}"
    )
    print(
        f"  Conf weighting:  mode={CONF_WEIGHT_MODE}  "
        f"center={CONF_WEIGHT_CENTER}  temp={CONF_WEIGHT_TEMP}  "
        f"range=[{CONF_WEIGHT_MIN}, {CONF_WEIGHT_MAX}]"
    )
    print(
        f"  Posthoc scale:   {POSTHOC_LOGIT_SCALE}  "
        f"range=[{POSTHOC_SCALE_MIN}, {POSTHOC_SCALE_MAX}]  "
        f"steps={POSTHOC_SCALE_STEPS}  chunks={POSTHOC_SCALE_CHUNKS}  "
        f"repeats={POSTHOC_SCALE_REPEATS}  selection={POSTHOC_SELECTION}  "
        f"signed_w={POSTHOC_SIGNED_ENTROPY_WEIGHT}"
    )
    print(
        f"  Posthoc bias:    {POSTHOC_BIAS}  steps={POSTHOC_BIAS_STEPS}  "
        f"lr={POSTHOC_BIAS_LR}  l2={POSTHOC_BIAS_L2}"
    )
    print(f"  Domain bridge:   {DOMAIN_BRIDGE}")
    print(
        f"  Domain residual: rank={DOMAIN_RESIDUAL_RANK}  "
        f"scale={DOMAIN_RESIDUAL_SCALE}"
    )
    print(
        f"  Target residual: {TARGET_RESIDUAL}  rank={TARGET_RESIDUAL_RANK}  "
        f"scale={TARGET_RESIDUAL_SCALE}"
    )
    print(f"  Torch dtype:     {TORCH_DTYPE_LABEL}")
    print(f"  Batch:           {BATCH}")
    print(f"  PPL batches:     {PPL_BATCHES}")
    print(
        f"  Align sentences: {N_ALIGN_SENTENCES}  min_chars={ALIGN_MIN_CHARS}  "
        f"select={ALIGN_SELECT}  pool={alignment_collection_limit()}  "
        f"fit={ALIGN_FIT_NORMALIZE}  map={ALIGN_MAP}  "
        f"ridge={ALIGN_RIDGE}  blend={ALIGN_LINEAR_BLEND}"
    )
    print(
        f"  Rotation ensemble:size={ROTATION_ENSEMBLE_SIZE}  "
        f"stride={ROTATION_ENSEMBLE_STRIDE}  "
        f"source_pool={rotation_alignment_collection_limit()}  "
        f"weights={ROTATION_ENSEMBLE_WEIGHTS or 'uniform'}  "
        f"train_weights={ROTATION_ENSEMBLE_TRAIN_WEIGHTS}"
    )
    print(f"  Release source:  {RELEASE_SOURCE_BEFORE_TARGET}")
    print(
        f"  Source preserve: enabled={SOURCE_PRESERVATION_EVAL}  "
        f"prompts={SOURCE_PRESERVATION_PROMPTS}  topk={SOURCE_PRESERVATION_TOPK}"
    )
    print(
        f"  Source completion loss: weight={SOURCE_COMPLETION_LOSS_WEIGHT}  "
        f"every={SOURCE_COMPLETION_LOSS_EVERY}  "
        f"batch={SOURCE_COMPLETION_LOSS_BATCH}  "
        f"candidates={SOURCE_COMPLETION_LOSS_CANDIDATES}  "
        f"start={SOURCE_COMPLETION_LOSS_START_STEP}  "
        f"nll_w={SOURCE_COMPLETION_NLL_WEIGHT}  "
        f"nll_cap={SOURCE_COMPLETION_NLL_CAP}"
    )
    print(f"  Domain corpus:   {DOMAIN_CORPUS}")
    if DOMAIN_CORPUS == "wikitext":
        print(
            f"  WikiText splits: train={WIKITEXT_DOMAIN_SPLIT}  "
            f"align={WIKITEXT_ALIGN_SPLIT}  posthoc={WIKITEXT_POSTHOC_SPLIT}  "
            f"eval={WIKITEXT_EVAL_SPLIT}"
        )
    print(
        f"  Selective eval:  enabled={SELECTIVE_TRANSFER_EVAL}  "
        f"off_domain={SELECTIVE_OFF_DOMAIN_CORPUS}  "
        f"reference={SELECTIVE_OFF_DOMAIN_REFERENCE}"
    )
    print(f"  Seed:            {EXPERIMENT_SEED}  offset={SEED_OFFSET}")
    print(
        f"  Seed bases:      source={SOURCE_DOMAIN_SEED_BASE}  "
        f"native={NATIVE_DOMAIN_SEED_BASE}  cal={CAL_SEED_BASE}  nib={NIB_SEED}"
    )
    print(f"  Train alpha:     {TRAIN_DOMAIN_ALPHA}")

    banner("Data loading")
    t_data = time.time()
    if is_legacy_gpt2_medium_source():
        tok_src = base.GPT2TokenizerFast.from_pretrained(
            base.HF_GPT2_MEDIUM, local_files_only=True
        )
        tok_src.model_max_length = base.sys.maxsize
    else:
        tok_src = AutoTokenizer.from_pretrained(HF_SOURCE_TOKENIZER, local_files_only=True)
    if tok_src.pad_token is None:
        tok_src.pad_token = tok_src.eos_token
    tok_tgt = AutoTokenizer.from_pretrained(HF_TARGET_TOKENIZER, local_files_only=True)
    if tok_tgt.pad_token is None:
        tok_tgt.pad_token = tok_tgt.eos_token

    py_text, py_meta = base.load_local_python_text(
        base.ROOT, base.MAX_PY, exclude_globs=V2_CORPUS_EXCLUDES
    )
    wiki_cache: dict[str, tuple[str, list[str], dict]] = {}

    def load_wiki(split):
        if split not in wiki_cache:
            wiki_cache[split] = base.load_wikitext_text_and_sentences(
                split=split, min_chars=20
            )
        return wiki_cache[split]

    wiki_domain_text, wiki_domain_sentences, wiki_domain_meta = load_wiki(
        WIKITEXT_DOMAIN_SPLIT
    )
    _wiki_align_text, wiki_align_sentences, wiki_align_meta = load_wiki(
        WIKITEXT_ALIGN_SPLIT
    )
    wiki_posthoc_text, _wiki_posthoc_sentences, wiki_posthoc_meta = load_wiki(
        WIKITEXT_POSTHOC_SPLIT
    )
    wiki_eval_text, wiki_eval_sentences, wiki_eval_meta = load_wiki(
        WIKITEXT_EVAL_SPLIT
    )
    if DOMAIN_CORPUS == "python":
        domain_text = py_text
        posthoc_text = py_text
        eval_text = py_text
        domain_detail = (
            f"local_python_files={py_meta['files']} skipped={py_meta['skipped']}"
        )
        posthoc_detail = "same_as_domain"
        eval_detail = "same_as_domain"
    else:
        domain_text = wiki_domain_text
        posthoc_text = wiki_posthoc_text
        eval_text = wiki_eval_text
        domain_detail = (
            f"wikitext_split={wiki_domain_meta['split']} "
            f"records={wiki_domain_meta['records']}"
        )
        posthoc_detail = (
            f"wikitext_split={wiki_posthoc_meta['split']} "
            f"records={wiki_posthoc_meta['records']}"
        )
        eval_detail = (
            f"wikitext_split={wiki_eval_meta['split']} "
            f"records={wiki_eval_meta['records']}"
        )
    selective_off_domain_ids_tgt = None
    selective_off_domain_detail = None
    if SELECTIVE_TRANSFER_EVAL:
        if SELECTIVE_OFF_DOMAIN_CORPUS == "python":
            selective_off_domain_text = py_text
            selective_off_domain_detail = (
                f"local_python_files={py_meta['files']} skipped={py_meta['skipped']}"
            )
        else:
            (
                selective_off_domain_text,
                _selective_off_domain_sentences,
                selective_off_domain_meta,
            ) = load_wiki(SELECTIVE_OFF_DOMAIN_WIKITEXT_SPLIT)
            selective_off_domain_detail = (
                f"wikitext_split={selective_off_domain_meta['split']} "
                f"records={selective_off_domain_meta['records']}"
            )
        selective_off_domain_ids_tgt = tok_tgt(
            selective_off_domain_text,
            return_tensors="pt",
            truncation=False,
        )["input_ids"].squeeze(0)[: base.MAX_PY]
    source_preservation_prompts = (
        build_source_preservation_prompts(
            eval_text,
            wiki_eval_sentences if DOMAIN_CORPUS == "wikitext" else [],
        )
        if SOURCE_PRESERVATION_EVAL or SOURCE_COMPLETION_LOSS_WEIGHT > 0
        else []
    )

    domain_ids_src = tok_src(
        domain_text, return_tensors="pt", truncation=False
    )["input_ids"].squeeze(0)[: base.MAX_PY]
    domain_ids_tgt = tok_tgt(
        domain_text, return_tensors="pt", truncation=False
    )["input_ids"].squeeze(0)[: base.MAX_PY]
    posthoc_ids_tgt = tok_tgt(
        posthoc_text, return_tensors="pt", truncation=False
    )["input_ids"].squeeze(0)[: base.MAX_PY]
    eval_ids_tgt = tok_tgt(
        eval_text, return_tensors="pt", truncation=False
    )["input_ids"].squeeze(0)[: base.MAX_PY]

    print(
        f"  {time.time() - t_data:.1f}s  "
        f"domain_src={len(domain_ids_src):,}  domain_tgt={len(domain_ids_tgt):,}  "
        f"posthoc_tgt={len(posthoc_ids_tgt):,}  eval_tgt={len(eval_ids_tgt):,}  "
        f"align_sentences={len(wiki_align_sentences):,}"
    )
    print(
        f"  corpus: domain={DOMAIN_CORPUS}  train=({domain_detail})  "
        f"posthoc=({posthoc_detail})  eval=({eval_detail})  "
        f"alignment_wikitext_split={wiki_align_meta['split']} "
        f"records={wiki_align_meta['records']}"
    )
    if SELECTIVE_TRANSFER_EVAL:
        print(
            f"  selective off-domain: corpus={SELECTIVE_OFF_DOMAIN_CORPUS}  "
            f"detail=({selective_off_domain_detail})  "
            f"tokens_tgt={len(selective_off_domain_ids_tgt):,}"
        )
    if SOURCE_PRESERVATION_EVAL or SOURCE_COMPLETION_LOSS_WEIGHT > 0:
        print(
            f"  Source-preservation prompts: "
            f"{len(source_preservation_prompts)}/{SOURCE_PRESERVATION_PROMPTS}"
        )

    banner(f"Phase A - {SOURCE_LABEL} ABI source domain training")
    t_a = time.time()
    if is_legacy_gpt2_medium_source():
        src_model = base.GPT2MedABI().to(base.DEVICE)
        source_vocab = base.VOCAB_SRC
        source_d_model = base.D_MODEL_SRC
        source_model_type = "gpt2"
    else:
        src_model = GenericCausalABI(HF_SOURCE).to(base.DEVICE)
        source_vocab = src_model.vocab_size
        source_d_model = src_model.d_model
        source_model_type = src_model.model_type
    for p in src_model.parameters():
        p.requires_grad_(False)
    for nm, p in src_model.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_a = torch.optim.AdamW(
        [p for p in src_model.parameters() if p.requires_grad],
        lr=base.LR_ABI,
        weight_decay=0.01,
    )
    src_model.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch(domain_ids_src, seed=SOURCE_DOMAIN_SEED_BASE + step)
        opt_a.zero_grad()
        loss = F.cross_entropy(src_model(x).reshape(-1, source_vocab), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(src_model.parameters(), 1.0)
        opt_a.step()
        if (step + 1) % 100 == 0:
            print(f"  A step {step + 1}/{DOMAIN_STEPS}  {time.time() - t_a:.0f}s")
    src_model.eval()
    for p in src_model.parameters():
        p.requires_grad_(False)
    ppl_a = ppl(src_model, domain_ids_src)
    phase_a_sec = time.time() - t_a
    print(f"  Phase A complete: {phase_a_sec:.0f}s  {SOURCE_LABEL} ppl={ppl_a:.1f}")
    source_alignment_pairs = None
    source_domain_for_rotation = src_model.domain
    source_domain_alpha = src_model.domain_alpha.detach().clone()
    source_domain_core_sha256 = artifact_module_state_sha256(
        source_domain_for_rotation.net
    )
    source_domain_full_sha256 = artifact_module_state_sha256(
        source_domain_for_rotation
    )
    source_preservation_source_records = []
    if SOURCE_PRESERVATION_EVAL or SOURCE_COMPLETION_LOSS_WEIGHT > 0:
        source_preservation_source_records = collect_source_preservation_source(
            src_model, tok_src, source_preservation_prompts
        )
        print(
            f"  Source-preservation source records: "
            f"{len(source_preservation_source_records)}"
        )
    if RELEASE_SOURCE_BEFORE_TARGET:
        source_alignment_pairs = collect_source_alignment_pairs(
            src_model, wiki_align_sentences, tok_src
        )
        source_domain_for_rotation = copy.deepcopy(src_model.domain).cpu()
        source_domain_alpha = source_domain_alpha.cpu()
        src_model.to("cpu")
        del src_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Released source backbone before loading target")

    phase_c_label = {
        "full_native_target_oracle": f"Native {TARGET_LABEL} ABI oracle",
        "target_base_interface": f"{TARGET_LABEL} base ABI interface reference",
        "base_target_reference": f"{TARGET_LABEL} frozen base reference",
    }[ORACLE_MODE]
    banner(f"Phase C - {phase_c_label}")
    t_c = time.time()
    native = GenericCausalABI(HF_TARGET).to(base.DEVICE)
    d_model_tgt = native.d_model
    vocab_tgt = native.vocab_size
    model_type = native.model_type
    for p in native.parameters():
        p.requires_grad_(False)
    phase_c_trainable_names = []
    if PHASE_C_TRAINS_TARGET_DOMAIN:
        phase_c_trainable_names = ["proj_in", "abi_ln", "proj_out", "domain"]
    elif PHASE_C_TRAINS_TARGET_INTERFACE:
        phase_c_trainable_names = ["proj_in", "abi_ln", "proj_out"]
    for nm, p in native.named_parameters():
        if any(k in nm for k in phase_c_trainable_names):
            p.requires_grad_(True)
    phase_c_trainable_params = trainable_count(
        [p for p in native.parameters() if p.requires_grad]
    )
    target_interface_cache = {
        "mode": TARGET_INTERFACE_CACHE_MODE,
        "eligible": bool(
            PHASE_C_TRAINS_TARGET_INTERFACE and not PHASE_C_TRAINS_TARGET_DOMAIN
        ),
        "path": None,
        "loaded": False,
        "saved": False,
        "skipped_phase_c_training": False,
    }
    phase_c_loaded_from_cache = False
    if target_interface_cache["eligible"] and TARGET_INTERFACE_CACHE_MODE != "none":
        cache_path = target_interface_cache_path()
        target_interface_cache["path"] = str(cache_path)
        if TARGET_INTERFACE_CACHE_MODE in {"load", "auto"} and cache_path.exists():
            loaded_summary = load_target_interface_cache(native, cache_path)
            target_interface_cache["loaded"] = True
            target_interface_cache["loaded_summary"] = loaded_summary
            target_interface_cache["skipped_phase_c_training"] = True
            phase_c_loaded_from_cache = True
            print(f"  Loaded target ABI interface cache: {cache_path}")
        elif TARGET_INTERFACE_CACHE_MODE == "load":
            raise FileNotFoundError(f"Missing target interface cache: {cache_path}")
    elif TARGET_INTERFACE_CACHE_MODE != "none":
        target_interface_cache["reason"] = (
            "cache is only supported for target_base_interface Phase C"
        )
    if PHASE_C_TRAINS_TARGET_INTERFACE:
        print(
            f"  Phase C reference mode={ORACLE_MODE}  "
            f"use_domain={TARGET_REFERENCE_FORWARD_MODE}  "
            f"trainable={phase_c_trainable_params:,}"
        )
        if phase_c_loaded_from_cache:
            print("  Phase C interface training skipped due to cache load")
        else:
            opt_c = torch.optim.AdamW(
                [p for p in native.parameters() if p.requires_grad],
                lr=base.LR_ABI,
                weight_decay=0.01,
            )
            native.train()
            for step in range(DOMAIN_STEPS):
                x, y = make_batch(domain_ids_tgt, seed=NATIVE_DOMAIN_SEED_BASE + step)
                opt_c.zero_grad()
                loss = F.cross_entropy(
                    native(x, use_domain=TARGET_REFERENCE_USES_DOMAIN).reshape(
                        -1, vocab_tgt
                    ),
                    y.reshape(-1),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(native.parameters(), 1.0)
                opt_c.step()
                if (step + 1) % 100 == 0:
                    print(
                        f"  C step {step + 1}/{DOMAIN_STEPS}  "
                        f"{time.time() - t_c:.0f}s"
                    )
            if (
                target_interface_cache["eligible"]
                and TARGET_INTERFACE_CACHE_MODE in {"save", "auto"}
            ):
                cache_path = target_interface_cache_path()
                summary = {
                    "target_model": TARGET_MODEL_ID,
                    "target_tokenizer": TARGET_TOKENIZER_ID,
                    "target_label": TARGET_LABEL,
                    "d_abi": D_ABI,
                    "oracle_mode": ORACLE_MODE,
                    "target_reference_uses_domain": TARGET_REFERENCE_USES_DOMAIN,
                    "target_reference_bypass_abi": TARGET_REFERENCE_BYPASS_ABI,
                    "domain_corpus": DOMAIN_CORPUS,
                    "domain_steps": DOMAIN_STEPS,
                    "native_domain_seed_base": NATIVE_DOMAIN_SEED_BASE,
                    "phase_c_trainable_params": phase_c_trainable_params,
                }
                save_target_interface_cache(native, cache_path, summary)
                target_interface_cache["path"] = str(cache_path)
                target_interface_cache["saved"] = True
                target_interface_cache["saved_summary"] = summary
                print(f"  Saved target ABI interface cache: {cache_path}")
    else:
        print(
            f"  Phase C skipped target ABI training; "
            f"using frozen base reference with use_domain={TARGET_REFERENCE_FORWARD_MODE}"
        )
    native.eval()
    for p in native.parameters():
        p.requires_grad_(False)
    ppl_nat = ppl(native, eval_ids_tgt, use_domain=TARGET_REFERENCE_FORWARD_MODE)
    phase_c_sec = time.time() - t_c
    print(
        f"  Phase C complete: {phase_c_sec:.0f}s  "
        f"{TARGET_LABEL} reference ppl={ppl_nat:.1f}"
    )

    banner("Phase D - Procrustes + KD calibration")
    t_d = time.time()
    alignment_members = []
    rotated_domains = []
    for rotation_idx in range(ROTATION_ENSEMBLE_SIZE):
        window_offset = rotation_idx * ROTATION_ENSEMBLE_STRIDE
        print(
            f"  [Procrustes ensemble] member "
            f"{rotation_idx + 1}/{ROTATION_ENSEMBLE_SIZE}  "
            f"offset={window_offset}"
        )
        if source_alignment_pairs is None:
            window_sentences = wiki_align_sentences[window_offset:]
            if not window_sentences:
                raise RuntimeError(
                    "Rotation ensemble offset exceeded available alignment sentences"
                )
            R, member_info = procrustes(
                src_model, native, window_sentences, tok_src, tok_tgt
            )
        else:
            window_pairs = source_alignment_pairs[window_offset:]
            if not window_pairs:
                raise RuntimeError(
                    "Rotation ensemble offset exceeded cached source alignments"
                )
            R, member_info = procrustes_from_cached_source(
                window_pairs, native, tok_tgt
            )
        member_info = dict(member_info)
        member_info["ensemble_index"] = int(rotation_idx)
        member_info["window_offset"] = int(window_offset)
        alignment_members.append(member_info)
        rotated_domains.append(base.apply_rotation_to_domain(source_domain_for_rotation, R))

    rotated_core_initial_sha256 = [
        artifact_module_state_sha256(domain.net) for domain in rotated_domains
    ]
    rotated_full_initial_sha256 = [
        artifact_module_state_sha256(domain) for domain in rotated_domains
    ]
    copied_payload_core_params = sum(
        artifact_module_param_count(domain.net) for domain in rotated_domains
    )
    copied_payload_full_params = sum(
        artifact_module_param_count(domain) for domain in rotated_domains
    )

    if ROTATION_ENSEMBLE_SIZE == 1:
        alignment_info = alignment_members[0]
    else:
        alignment_info = {
            "mode": "rotation_ensemble",
            "ensemble_size": int(ROTATION_ENSEMBLE_SIZE),
            "stride": int(ROTATION_ENSEMBLE_STRIDE),
            "final_pairs": int(
                sum(member["final_pairs"] for member in alignment_members)
            ),
            "requested_final_pairs": int(
                N_ALIGN_SENTENCES * ROTATION_ENSEMBLE_SIZE
            ),
            "pool_requested_per_member": int(alignment_collection_limit()),
            "fit_normalize": ALIGN_FIT_NORMALIZE,
            "members": alignment_members,
            "final_after_mean": round(
                float(
                    base.np.mean(
                        [member["final_after_mean"] for member in alignment_members]
                    )
                ),
                4,
            ),
        }

    calibrated = GenericCausalABI.__new__(GenericCausalABI)
    nn.Module.__init__(calibrated)
    calibrated.model = native.model
    calibrated.lm_head = native.lm_head
    calibrated.d_model = d_model_tgt
    calibrated.vocab_size = vocab_tgt
    calibrated.model_type = model_type
    calibrated.proj_in = nn.Linear(d_model_tgt, D_ABI, bias=False).to(base.DEVICE)
    calibrated.abi_ln = nn.LayerNorm(D_ABI).to(base.DEVICE)
    calibrated.proj_out = nn.Linear(D_ABI, d_model_tgt, bias=False).to(base.DEVICE)
    if len(rotated_domains) == 1:
        calibrated.domain = rotated_domains[0].to(base.DEVICE)
    else:
        calibrated.domain_ensemble = nn.ModuleList(
            [domain.to(base.DEVICE) for domain in rotated_domains]
        )
        initial_weights = ROTATION_ENSEMBLE_WEIGHTS or [
            1.0 / len(rotated_domains)
        ] * len(rotated_domains)
        initial_weights_tensor = torch.tensor(
            initial_weights,
            device=base.DEVICE,
            dtype=torch.float32,
        )
        if ROTATION_ENSEMBLE_TRAIN_WEIGHTS:
            calibrated.domain_ensemble_logits = nn.Parameter(
                torch.log(initial_weights_tensor.clamp_min(1.0e-8))
            )
        else:
            calibrated.register_buffer(
                "domain_ensemble_weights",
                initial_weights_tensor,
            )
    calibrated.domain_alpha = nn.Parameter(
        source_domain_alpha.detach().clone().to(base.DEVICE)
    )
    if DOMAIN_BRIDGE == "linear":
        calibrated.domain_bridge_in = identity_linear(D_ABI).to(base.DEVICE)
        calibrated.domain_bridge_out = identity_linear(D_ABI).to(base.DEVICE)
    if DOMAIN_RESIDUAL_RANK > 0:
        calibrated.domain_residual_down = nn.Linear(
            D_ABI, DOMAIN_RESIDUAL_RANK, bias=False
        ).to(base.DEVICE)
        calibrated.domain_residual_up = zero_init_linear(
            DOMAIN_RESIDUAL_RANK, D_ABI
        ).to(base.DEVICE)
        calibrated.domain_residual_scale = DOMAIN_RESIDUAL_SCALE
    if TARGET_RESIDUAL == "hidden":
        calibrated.target_residual_down = nn.Linear(
            d_model_tgt, TARGET_RESIDUAL_RANK, bias=False
        ).to(base.DEVICE)
        calibrated.target_residual_up = zero_init_linear(
            TARGET_RESIDUAL_RANK, d_model_tgt
        ).to(base.DEVICE)
        calibrated.target_residual_scale = TARGET_RESIDUAL_SCALE
    elif TARGET_RESIDUAL == "logit_abi":
        calibrated.logit_residual_down = nn.Linear(
            D_ABI, TARGET_RESIDUAL_RANK, bias=False
        ).to(base.DEVICE)
        calibrated.logit_residual_up = zero_init_linear(
            TARGET_RESIDUAL_RANK, vocab_tgt
        ).to(base.DEVICE)
        calibrated.target_residual_scale = TARGET_RESIDUAL_SCALE
    if CAL_INIT == "native":
        calibrated.proj_in.load_state_dict(native.proj_in.state_dict())
        calibrated.abi_ln.load_state_dict(native.abi_ln.state_dict())
        calibrated.proj_out.load_state_dict(native.proj_out.state_dict())
        print(
            f"  Initialized target ABI interface from Phase C reference "
            f"(mode={ORACLE_MODE})"
        )
    elif CAL_INIT == "zero_out":
        nn.init.xavier_uniform_(calibrated.proj_in.weight)
        nn.init.zeros_(calibrated.proj_out.weight)
        print(
            "  Initialized target ABI interface as zero-residual passthrough "
            "(proj_out=0)"
        )
    else:
        nn.init.xavier_uniform_(calibrated.proj_in.weight)
        nn.init.xavier_uniform_(calibrated.proj_out.weight)

    for p in calibrated.parameters():
        p.requires_grad_(False)
    cal_params = []
    trainable_groups = []
    add_group("proj_in", calibrated.proj_in.parameters(), cal_params, trainable_groups)
    add_group("abi_ln", calibrated.abi_ln.parameters(), cal_params, trainable_groups)
    add_group("proj_out", calibrated.proj_out.parameters(), cal_params, trainable_groups)
    if hasattr(calibrated, "domain_ensemble"):
        for domain_idx, domain in enumerate(calibrated.domain_ensemble):
            if CAL_MODE == "train_domain":
                add_group(
                    f"domain_ensemble_{domain_idx}_net",
                    domain.net.parameters(),
                    cal_params,
                    trainable_groups,
                )
                add_group(
                    f"domain_ensemble_{domain_idx}_ln",
                    domain.ln.parameters(),
                    cal_params,
                    trainable_groups,
                )
            elif CAL_MODE == "freeze_domain_net":
                add_group(
                    f"domain_ensemble_{domain_idx}_ln",
                    domain.ln.parameters(),
                    cal_params,
                    trainable_groups,
                )
        if hasattr(calibrated, "domain_ensemble_logits"):
            add_group(
                "domain_ensemble_logits",
                [calibrated.domain_ensemble_logits],
                cal_params,
                trainable_groups,
            )
    else:
        if CAL_MODE == "train_domain":
            add_group(
                "domain_net",
                calibrated.domain.net.parameters(),
                cal_params,
                trainable_groups,
            )
            add_group(
                "domain_ln",
                calibrated.domain.ln.parameters(),
                cal_params,
                trainable_groups,
            )
        elif CAL_MODE == "freeze_domain_net":
            add_group(
                "domain_ln",
                calibrated.domain.ln.parameters(),
                cal_params,
                trainable_groups,
            )
    if DOMAIN_BRIDGE == "linear":
        add_group(
            "domain_bridge_in",
            calibrated.domain_bridge_in.parameters(),
            cal_params,
            trainable_groups,
        )
        add_group(
            "domain_bridge_out",
            calibrated.domain_bridge_out.parameters(),
            cal_params,
            trainable_groups,
        )
    if TRAIN_DOMAIN_ALPHA:
        add_group("domain_alpha", [calibrated.domain_alpha], cal_params, trainable_groups)
    if DOMAIN_RESIDUAL_RANK > 0:
        add_module_group(
            "domain_residual_down",
            calibrated.domain_residual_down,
            cal_params,
            trainable_groups,
        )
        add_module_group(
            "domain_residual_up",
            calibrated.domain_residual_up,
            cal_params,
            trainable_groups,
        )
    if TARGET_RESIDUAL == "hidden":
        add_module_group(
            "target_residual_hidden_down",
            calibrated.target_residual_down,
            cal_params,
            trainable_groups,
        )
        add_module_group(
            "target_residual_hidden_up",
            calibrated.target_residual_up,
            cal_params,
            trainable_groups,
        )
    elif TARGET_RESIDUAL == "logit_abi":
        add_module_group(
            "target_residual_logit_down",
            calibrated.logit_residual_down,
            cal_params,
            trainable_groups,
        )
        add_module_group(
            "target_residual_logit_up",
            calibrated.logit_residual_up,
            cal_params,
            trainable_groups,
        )

    print("  Trainable groups:")
    for group in trainable_groups:
        print(f"    {group['name']}: {group['params']:,}")
    print(f"  Total trainable during D: {trainable_count(cal_params):,}")

    opt_d = torch.optim.AdamW(cal_params, lr=CAL_LR, weight_decay=0.01)
    native.eval()
    calibrated.train()
    kd_weight = KD_WEIGHT
    kd_temp = KD_TEMP
    cal_lr_decayed = False
    source_completion_loss_records = prepare_source_completion_loss_records(
        source_preservation_source_records
    )
    source_completion_loss = {
        "enabled": SOURCE_COMPLETION_LOSS_WEIGHT > 0,
        "weight": SOURCE_COMPLETION_LOSS_WEIGHT,
        "every": SOURCE_COMPLETION_LOSS_EVERY,
        "batch": SOURCE_COMPLETION_LOSS_BATCH,
        "prompt_limit": SOURCE_COMPLETION_LOSS_PROMPTS,
        "candidates": SOURCE_COMPLETION_LOSS_CANDIDATES,
        "temperature": SOURCE_COMPLETION_LOSS_TEMP,
        "start_step": SOURCE_COMPLETION_LOSS_START_STEP,
        "margin_weight": SOURCE_COMPLETION_MARGIN_WEIGHT,
        "margin": SOURCE_COMPLETION_MARGIN,
        "nll_weight": SOURCE_COMPLETION_NLL_WEIGHT,
        "nll_cap": SOURCE_COMPLETION_NLL_CAP,
        "source_record_count": len(source_preservation_source_records),
        "train_record_count": len(source_completion_loss_records),
        "updates": 0,
        "skipped_applications": 0,
        "mean_loss": None,
    }
    source_completion_loss_total = 0.0
    if SOURCE_COMPLETION_LOSS_WEIGHT > 0:
        print(
            f"  Source-completion loss records: "
            f"{len(source_completion_loss_records)}/"
            f"{len(source_preservation_source_records)}"
        )
    calibration_selection = {
        "mode": CAL_SELECT,
        "applied": False,
        "every": CAL_SELECT_EVERY,
        "chunks": CAL_SELECT_CHUNKS,
        "repeats": CAL_SELECT_REPEATS,
        "reduction": CAL_SELECT_REDUCTION,
        "top1_weight": CAL_SELECT_TOP1_WEIGHT,
        "top5_weight": CAL_SELECT_TOP5_WEIGHT,
        "js_weight": CAL_SELECT_JS_WEIGHT,
        "entropy_weight": CAL_SELECT_ENTROPY_WEIGHT,
        "avg_top_n": CAL_SELECT_AVG_TOP_N,
        "soup_weights_requested": CAL_SELECT_SOUP_WEIGHTS,
        "soup_weight_grid_requested": CAL_SELECT_SOUP_WEIGHT_GRID,
        "soup_weight_grid_repeats": CAL_SELECT_SOUP_GRID_REPEATS,
        "soup_weight_grid_reduction": CAL_SELECT_SOUP_GRID_REDUCTION,
        "nib_audit_enabled": CAL_SELECT_AUDIT_NIB,
        "nib_audit_max": CAL_SELECT_AUDIT_MAX,
        "soup_nib_audit_weights_requested": CAL_SELECT_SOUP_AUDIT_WEIGHTS,
        "soup_weight_grid_selection": None,
        "averaged": False,
        "candidates": [],
        "nib_audit": [],
        "soup_nib_audit": [],
    }
    calibration_ema = {
        "decay": CAL_EMA_DECAY,
        "start_step": CAL_EMA_START_STEP,
        "every": CAL_EMA_EVERY,
        "as_candidate": CAL_EMA_AS_CANDIDATE,
        "restore": CAL_EMA_RESTORE,
        "enabled": CAL_EMA_DECAY > 0,
        "updates": 0,
        "candidate": None,
        "restored": False,
    }
    calibration_final_selection = {
        "mode": CAL_FINAL_SELECT,
        "applied": False,
        "candidates_requested": CAL_FINAL_SELECT_CANDIDATES,
        "chunks": CAL_FINAL_SELECT_CHUNKS,
        "repeats": CAL_FINAL_SELECT_REPEATS,
        "weights": {
            "top1": CAL_FINAL_SELECT_TOP1_WEIGHT,
            "top5": CAL_FINAL_SELECT_TOP5_WEIGHT,
            "js": CAL_FINAL_SELECT_JS_WEIGHT,
            "entropy": CAL_FINAL_SELECT_ENTROPY_WEIGHT,
        },
        "candidates": [],
        "selected": None,
    }
    calibration_final_audit = {
        "nib_enabled": CAL_FINAL_AUDIT_NIB,
        "candidates_requested": CAL_FINAL_AUDIT_CANDIDATES,
        "candidates": [],
        "selection_used_for_restore": False,
    }
    calibration_final_soup_audit = {
        "candidates_requested": CAL_FINAL_SOUP_AUDIT_CANDIDATES,
        "weights_requested": CAL_FINAL_SOUP_AUDIT_WEIGHTS,
        "candidates": [],
    }
    calibration_temporal_average = {
        "enabled": CAL_TEMPORAL_AVG_START_STEP > 0,
        "start_step": CAL_TEMPORAL_AVG_START_STEP,
        "restore": CAL_TEMPORAL_AVG_RESTORE,
        "applied": False,
        "steps": [],
        "weights": [],
        "restored": False,
    }
    best_selection_score = None
    best_selection_state = None
    best_selection_row = None
    selection_states = []
    ema_state = None
    ema_cpu_state = None
    ema_updates = 0
    stable_top1_stats = {
        "enabled": bool(
            STABLE_TOP1_CE_WEIGHT > 0 or STABLE_TOP1_HARD_NEG_WEIGHT > 0
        ),
        "tokens_seen": 0,
        "tokens_selected": 0,
    }

    def teacher_confidence_weights(nat_logits):
        if CONF_WEIGHT_MODE == "none":
            return None
        teacher_flat = nat_logits.reshape(-1, vocab_tgt).float()
        top2 = teacher_flat.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        if CONF_WEIGHT_MODE == "teacher_low_margin":
            raw = torch.sigmoid((CONF_WEIGHT_CENTER - margin) / CONF_WEIGHT_TEMP)
        else:
            raw = torch.sigmoid((margin - CONF_WEIGHT_CENTER) / CONF_WEIGHT_TEMP)
        weights = CONF_WEIGHT_MIN + (CONF_WEIGHT_MAX - CONF_WEIGHT_MIN) * raw
        weights = weights / weights.mean().clamp_min(1.0e-6)
        return weights.detach()

    def stable_top1_weights(nat_logits, nat_base_logits):
        if not stable_top1_stats["enabled"]:
            return None
        teacher_flat = nat_logits.reshape(-1, vocab_tgt).float()
        stable = torch.ones(teacher_flat.shape[0], device=teacher_flat.device).bool()
        if STABLE_TOP1_REQUIRE_BASE_AGREE:
            base_flat = nat_base_logits.reshape(-1, vocab_tgt).float()
            stable = stable & (teacher_flat.argmax(dim=-1) == base_flat.argmax(dim=-1))
        if STABLE_TOP1_MIN_MARGIN > 0:
            top2 = teacher_flat.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            stable = stable & (margin >= STABLE_TOP1_MIN_MARGIN)
        stable_top1_stats["tokens_seen"] += int(stable.numel())
        stable_top1_stats["tokens_selected"] += int(stable.sum().item())
        weights = stable.float()
        mean_weight = weights.mean()
        if float(mean_weight.item()) > 0:
            weights = weights / mean_weight
        return weights.detach()

    def combine_token_weights(*weights):
        active = [weight for weight in weights if weight is not None]
        if not active:
            return None
        combined = active[0]
        for weight in active[1:]:
            combined = combined * weight.to(
                device=combined.device, dtype=combined.dtype
            )
        mean_weight = combined.mean()
        if float(mean_weight.item()) > 0:
            combined = combined / mean_weight
        return combined.detach()

    def maybe_weighted_mean(values, weights):
        if weights is None:
            return values.mean()
        return (values * weights.to(device=values.device, dtype=values.dtype)).mean()

    def calibration_batch_loss(x, y):
        use_abi_state_loss = ABI_PRE_MSE_WEIGHT > 0 or ABI_STATE_MSE_WEIGHT > 0
        use_stable_top1 = stable_top1_stats["enabled"]
        use_domain_delta_logit = DOMAIN_DELTA_LOGIT_MSE_WEIGHT > 0
        nat_base_logits = None
        cal_base_logits = None
        if use_abi_state_loss:
            cal_logits, cal_h_abi, cal_h_out = calibrated.forward_with_abi_states(x)
            if use_domain_delta_logit:
                cal_base_logits = calibrated(x, use_domain=False)
            with torch.no_grad():
                nat_logits, nat_h_abi, nat_h_out = native.forward_with_abi_states(
                    x, use_domain=TARGET_REFERENCE_FORWARD_MODE
                )
                if use_stable_top1 or use_domain_delta_logit:
                    nat_base_logits = native(x, use_domain=False)
        else:
            cal_logits = calibrated(x)
            if use_domain_delta_logit:
                cal_base_logits = calibrated(x, use_domain=False)
            with torch.no_grad():
                nat_logits = native(x, use_domain=TARGET_REFERENCE_FORWARD_MODE)
                if use_stable_top1 or use_domain_delta_logit:
                    nat_base_logits = native(x, use_domain=False)
        conf_weights = teacher_confidence_weights(nat_logits)
        stable_weights = stable_top1_weights(nat_logits, nat_base_logits)
        ce = F.cross_entropy(cal_logits.reshape(-1, vocab_tgt), y.reshape(-1))
        kd = F.kl_div(
            F.log_softmax(cal_logits.reshape(-1, vocab_tgt) / kd_temp, dim=-1),
            F.softmax(nat_logits.reshape(-1, vocab_tgt) / kd_temp, dim=-1),
            reduction="batchmean",
        ) * (kd_temp ** 2)
        loss = kd_weight * kd + (1 - kd_weight) * ce
        if ABI_PRE_MSE_WEIGHT > 0:
            pre_mse = F.mse_loss(cal_h_abi.float(), nat_h_abi.float())
            loss = loss + ABI_PRE_MSE_WEIGHT * pre_mse
        if ABI_STATE_MSE_WEIGHT > 0:
            state_mse = F.mse_loss(cal_h_out.float(), nat_h_out.float())
            loss = loss + ABI_STATE_MSE_WEIGHT * state_mse
        if ENTROPY_WEIGHT > 0:
            student_logp = F.log_softmax(
                cal_logits.reshape(-1, vocab_tgt).float(), dim=-1
            )
            student_p = student_logp.exp()
            with torch.no_grad():
                teacher_logp = F.log_softmax(
                    nat_logits.reshape(-1, vocab_tgt).float(), dim=-1
                )
                teacher_entropy = -(teacher_logp.exp() * teacher_logp).sum(dim=-1)
            student_entropy = -(student_p * student_logp).sum(dim=-1)
            entropy_loss = F.mse_loss(student_entropy, teacher_entropy)
            loss = loss + ENTROPY_WEIGHT * entropy_loss
        if TOPK_KD_WEIGHT > 0:
            with torch.no_grad():
                teacher_flat = nat_logits.reshape(-1, vocab_tgt)
                top_idx = teacher_flat.topk(min(TOPK, vocab_tgt), dim=-1).indices
                top_teacher = teacher_flat.gather(1, top_idx)
            top_student = cal_logits.reshape(-1, vocab_tgt).gather(1, top_idx)
            topk_token_kd = F.kl_div(
                F.log_softmax(top_student / kd_temp, dim=-1),
                F.softmax(top_teacher / kd_temp, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            topk_kd = maybe_weighted_mean(
                topk_token_kd,
                conf_weights,
            ) * (kd_temp ** 2)
            loss = loss + TOPK_KD_WEIGHT * topk_kd
        if UNION_TOPK_KD_WEIGHT > 0:
            with torch.no_grad():
                teacher_flat = nat_logits.reshape(-1, vocab_tgt)
                student_flat_for_idx = cal_logits.reshape(-1, vocab_tgt)
                union_k = min(UNION_TOPK, vocab_tgt)
                teacher_idx = teacher_flat.topk(union_k, dim=-1).indices
                student_idx = student_flat_for_idx.topk(union_k, dim=-1).indices
                union_idx = torch.cat([teacher_idx, student_idx], dim=-1)
                union_teacher = teacher_flat.gather(1, union_idx)
            student_flat = cal_logits.reshape(-1, vocab_tgt)
            union_student = student_flat.gather(1, union_idx)
            union_token_kd = F.kl_div(
                F.log_softmax(union_student / UNION_TOPK_TEMP, dim=-1),
                F.softmax(union_teacher / UNION_TOPK_TEMP, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            union_kd = maybe_weighted_mean(
                union_token_kd, conf_weights
            ) * (UNION_TOPK_TEMP ** 2)
            loss = loss + UNION_TOPK_KD_WEIGHT * union_kd
        if RANK_MARGIN_WEIGHT > 0:
            with torch.no_grad():
                teacher_flat = nat_logits.reshape(-1, vocab_tgt)
                rank_k = min(max(RANK_NEG_K, RANK_TOP_POS + 1), vocab_tgt)
                rank_idx = teacher_flat.topk(rank_k, dim=-1).indices
            rank_student = cal_logits.reshape(-1, vocab_tgt).gather(1, rank_idx)
            pos = rank_student[:, :RANK_TOP_POS]
            neg = rank_student[:, RANK_TOP_POS:]
            rank_token_loss = F.softplus(
                neg.unsqueeze(1) - pos.unsqueeze(2) + RANK_MARGIN
            ).mean(dim=(1, 2))
            rank_loss = maybe_weighted_mean(rank_token_loss, conf_weights)
            loss = loss + RANK_MARGIN_WEIGHT * rank_loss
        if TOP1_GAP_WEIGHT > 0:
            with torch.no_grad():
                teacher_flat = nat_logits.reshape(-1, vocab_tgt)
                gap_k = min(max(TOP1_GAP_K, 2), vocab_tgt)
                gap_idx = teacher_flat.topk(gap_k, dim=-1).indices
            gap_student = cal_logits.reshape(-1, vocab_tgt).gather(1, gap_idx)
            top1 = gap_student[:, :1]
            next_choices = gap_student[:, 1:]
            top1_gap_token_loss = F.softplus(
                next_choices - top1 + TOP1_GAP_MARGIN
            ).mean(dim=1)
            top1_gap_loss = maybe_weighted_mean(top1_gap_token_loss, conf_weights)
            loss = loss + TOP1_GAP_WEIGHT * top1_gap_loss
        if TOP1_CE_WEIGHT > 0:
            with torch.no_grad():
                teacher_argmax = nat_logits.reshape(-1, vocab_tgt).argmax(dim=-1)
            student_flat = cal_logits.reshape(-1, vocab_tgt).float()
            top1_ce_token = F.cross_entropy(
                student_flat, teacher_argmax, reduction="none"
            )
            top1_ce = maybe_weighted_mean(top1_ce_token, conf_weights)
            loss = loss + TOP1_CE_WEIGHT * top1_ce
        if STABLE_TOP1_CE_WEIGHT > 0:
            with torch.no_grad():
                teacher_argmax = nat_logits.reshape(-1, vocab_tgt).argmax(dim=-1)
            student_flat = cal_logits.reshape(-1, vocab_tgt).float()
            stable_top1_ce_token = F.cross_entropy(
                student_flat, teacher_argmax, reduction="none"
            )
            stable_top1_ce = maybe_weighted_mean(
                stable_top1_ce_token,
                combine_token_weights(conf_weights, stable_weights),
            )
            loss = loss + STABLE_TOP1_CE_WEIGHT * stable_top1_ce
        if HARD_NEG_WEIGHT > 0:
            with torch.no_grad():
                teacher_flat = nat_logits.reshape(-1, vocab_tgt)
                student_flat = cal_logits.reshape(-1, vocab_tgt)
                pos_idx = teacher_flat.topk(min(RANK_TOP_POS, vocab_tgt), dim=-1).indices
                cand_k = min(HARD_NEG_K + pos_idx.shape[1], vocab_tgt)
                cand_idx = student_flat.topk(cand_k, dim=-1).indices
                is_teacher_pos = cand_idx.unsqueeze(-1).eq(pos_idx.unsqueeze(1)).any(-1)
            student_flat = cal_logits.reshape(-1, vocab_tgt)
            pos = student_flat.gather(1, pos_idx)
            cand_logits = student_flat.gather(1, cand_idx)
            cand_logits = cand_logits.masked_fill(is_teacher_pos, -1.0e4)
            hard_k = min(HARD_NEG_K, cand_logits.shape[1])
            hard_neg = cand_logits.topk(hard_k, dim=-1).values
            hard_neg_token_loss = F.softplus(
                hard_neg.unsqueeze(1) - pos.unsqueeze(2) + RANK_MARGIN
            ).mean(dim=(1, 2))
            hard_neg_loss = maybe_weighted_mean(hard_neg_token_loss, conf_weights)
            loss = loss + HARD_NEG_WEIGHT * hard_neg_loss
        if TOP1_HARD_NEG_WEIGHT > 0:
            with torch.no_grad():
                teacher_flat = nat_logits.reshape(-1, vocab_tgt)
                student_flat = cal_logits.reshape(-1, vocab_tgt)
                top1_idx = teacher_flat.argmax(dim=-1, keepdim=True)
                cand_k = min(TOP1_HARD_NEG_K + 1, vocab_tgt)
                cand_idx = student_flat.topk(cand_k, dim=-1).indices
                is_teacher_top1 = cand_idx.eq(top1_idx)
            student_flat = cal_logits.reshape(-1, vocab_tgt)
            top1_pos = student_flat.gather(1, top1_idx)
            cand_logits = student_flat.gather(1, cand_idx)
            cand_logits = cand_logits.masked_fill(is_teacher_top1, -1.0e4)
            top1_hard_neg = cand_logits.topk(
                min(TOP1_HARD_NEG_K, cand_logits.shape[1]), dim=-1
            ).values
            top1_hard_neg_token_loss = F.softplus(
                top1_hard_neg - top1_pos + RANK_MARGIN
            ).mean(dim=1)
            top1_hard_neg_loss = maybe_weighted_mean(
                top1_hard_neg_token_loss, conf_weights
            )
            loss = loss + TOP1_HARD_NEG_WEIGHT * top1_hard_neg_loss
        if STABLE_TOP1_HARD_NEG_WEIGHT > 0:
            with torch.no_grad():
                teacher_flat = nat_logits.reshape(-1, vocab_tgt)
                student_flat = cal_logits.reshape(-1, vocab_tgt)
                stable_top1_idx = teacher_flat.argmax(dim=-1, keepdim=True)
                cand_k = min(STABLE_TOP1_HARD_NEG_K + 1, vocab_tgt)
                cand_idx = student_flat.topk(cand_k, dim=-1).indices
                is_teacher_top1 = cand_idx.eq(stable_top1_idx)
            student_flat = cal_logits.reshape(-1, vocab_tgt)
            stable_top1_pos = student_flat.gather(1, stable_top1_idx)
            cand_logits = student_flat.gather(1, cand_idx)
            cand_logits = cand_logits.masked_fill(is_teacher_top1, -1.0e4)
            stable_top1_hard_neg = cand_logits.topk(
                min(STABLE_TOP1_HARD_NEG_K, cand_logits.shape[1]), dim=-1
            ).values
            stable_top1_hard_neg_token_loss = F.softplus(
                stable_top1_hard_neg - stable_top1_pos + RANK_MARGIN
            ).mean(dim=1)
            stable_top1_hard_neg_loss = maybe_weighted_mean(
                stable_top1_hard_neg_token_loss,
                combine_token_weights(conf_weights, stable_weights),
            )
            loss = loss + STABLE_TOP1_HARD_NEG_WEIGHT * stable_top1_hard_neg_loss
        if TOPSET_WEIGHT > 0:
            with torch.no_grad():
                teacher_flat = nat_logits.reshape(-1, vocab_tgt)
                topset_k = min(TOPSET_K, vocab_tgt)
                topset_idx = teacher_flat.topk(topset_k, dim=-1).indices
                topset_teacher = teacher_flat.gather(1, topset_idx)
                topset_target = F.softmax(topset_teacher / TOPSET_TEMP, dim=-1)
            student_flat = cal_logits.reshape(-1, vocab_tgt)
            topset_logp = F.log_softmax(student_flat / TOPSET_TEMP, dim=-1).gather(
                1, topset_idx
            )
            topset_token_loss = -(topset_target * topset_logp).sum(dim=-1)
            topset_loss = maybe_weighted_mean(topset_token_loss, conf_weights)
            loss = loss + TOPSET_WEIGHT * (TOPSET_TEMP ** 2) * topset_loss
        if TOP_LOGIT_MSE_WEIGHT > 0:
            with torch.no_grad():
                teacher_flat = nat_logits.reshape(-1, vocab_tgt)
                top_logit_k = min(TOP_LOGIT_MSE_K, vocab_tgt)
                top_logit_idx = teacher_flat.topk(top_logit_k, dim=-1).indices
                top_teacher = teacher_flat.gather(1, top_logit_idx)
                top_teacher = top_teacher - top_teacher.mean(dim=-1, keepdim=True)
            student_flat = cal_logits.reshape(-1, vocab_tgt)
            top_student = student_flat.gather(1, top_logit_idx)
            top_student = top_student - top_student.mean(dim=-1, keepdim=True)
            top_logit_token_mse = F.mse_loss(
                top_student, top_teacher, reduction="none"
            ).mean(dim=-1)
            top_logit_mse = maybe_weighted_mean(top_logit_token_mse, conf_weights)
            loss = loss + TOP_LOGIT_MSE_WEIGHT * top_logit_mse
        if DOMAIN_DELTA_LOGIT_MSE_WEIGHT > 0:
            with torch.no_grad():
                teacher_delta = (
                    nat_logits.reshape(-1, vocab_tgt).float()
                    - nat_base_logits.reshape(-1, vocab_tgt).float()
                )
                delta_k = min(DOMAIN_DELTA_LOGIT_MSE_K, vocab_tgt)
                delta_idx = teacher_delta.abs().topk(delta_k, dim=-1).indices
                top_teacher_delta = teacher_delta.gather(1, delta_idx)
                if DOMAIN_DELTA_LOGIT_MSE_CENTER:
                    top_teacher_delta = top_teacher_delta - top_teacher_delta.mean(
                        dim=-1, keepdim=True
                    )
            student_delta = (
                cal_logits.reshape(-1, vocab_tgt).float()
                - cal_base_logits.reshape(-1, vocab_tgt).float()
            )
            top_student_delta = student_delta.gather(1, delta_idx)
            if DOMAIN_DELTA_LOGIT_MSE_CENTER:
                top_student_delta = top_student_delta - top_student_delta.mean(
                    dim=-1, keepdim=True
                )
            domain_delta_token_mse = F.mse_loss(
                top_student_delta, top_teacher_delta, reduction="none"
            ).mean(dim=-1)
            domain_delta_mse = maybe_weighted_mean(
                domain_delta_token_mse, conf_weights
            )
            loss = loss + DOMAIN_DELTA_LOGIT_MSE_WEIGHT * domain_delta_mse
        return loss

    def source_completion_batch_loss(step):
        if SOURCE_COMPLETION_LOSS_WEIGHT <= 0:
            return None
        if step < SOURCE_COMPLETION_LOSS_START_STEP:
            return None
        if step % SOURCE_COMPLETION_LOSS_EVERY != 0:
            return None
        if not source_completion_loss_records:
            source_completion_loss["skipped_applications"] += 1
            return None
        losses = []
        n_records = len(source_completion_loss_records)
        for batch_idx in range(SOURCE_COMPLETION_LOSS_BATCH):
            record_idx = (step * SOURCE_COMPLETION_LOSS_BATCH + batch_idx) % n_records
            record = source_completion_loss_records[record_idx]
            scores = []
            for candidate in record["candidates"]:
                score = target_completion_mean_logprob_tensor(
                    calibrated,
                    tok_tgt,
                    record["_prompt"],
                    candidate["decoded_text"],
                )
                if score is not None:
                    scores.append(score)
            if len(scores) < 2:
                continue
            scores = torch.stack(scores)
            logits = scores.unsqueeze(0) / SOURCE_COMPLETION_LOSS_TEMP
            target = torch.zeros(1, dtype=torch.long, device=logits.device)
            source_loss = F.cross_entropy(logits, target)
            if SOURCE_COMPLETION_MARGIN_WEIGHT > 0:
                positive = scores[0]
                alternatives = scores[1:]
                margin_loss = F.softplus(
                    alternatives - positive + SOURCE_COMPLETION_MARGIN
                ).mean()
                source_loss = (
                    source_loss
                    + SOURCE_COMPLETION_MARGIN_WEIGHT * margin_loss
                )
            if SOURCE_COMPLETION_NLL_WEIGHT > 0:
                source_nll = -scores[0]
                if SOURCE_COMPLETION_NLL_CAP > 0:
                    source_nll = torch.relu(source_nll - SOURCE_COMPLETION_NLL_CAP)
                source_loss = (
                    source_loss
                    + SOURCE_COMPLETION_NLL_WEIGHT * source_nll
                )
            losses.append(source_loss)
        if not losses:
            source_completion_loss["skipped_applications"] += 1
            return None
        return torch.stack(losses).mean()

    def lr_factor_for_step(step):
        if CAL_LR_DECAY_STEP <= 0 or step < CAL_LR_DECAY_STEP:
            return 1.0
        if CAL_LR_SCHEDULE == "step":
            return CAL_LR_DECAY_FACTOR
        decay_span = max(CALIBRATION_STEPS - CAL_LR_DECAY_STEP, 1)
        progress = min(max((step - CAL_LR_DECAY_STEP + 1) / decay_span, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return CAL_LR_FINAL_FACTOR + (
            CAL_LR_DECAY_FACTOR - CAL_LR_FINAL_FACTOR
        ) * cosine

    for step in range(CALIBRATION_STEPS):
        if (
            CAL_LR_DECAY_STEP > 0
            and step >= CAL_LR_DECAY_STEP
            and (CAL_LR_SCHEDULE == "cosine_after" or not cal_lr_decayed)
        ):
            lr_factor = lr_factor_for_step(step)
            decayed_lr = CAL_LR * lr_factor
            for group in opt_d.param_groups:
                group["lr"] = decayed_lr
            if not cal_lr_decayed:
                cal_lr_decayed = True
                print(
                    f"  D LR decay at step {step}: "
                    f"{CAL_LR:.2e} -> {decayed_lr:.2e} "
                    f"(schedule={CAL_LR_SCHEDULE})"
                )
        opt_d.zero_grad()
        for accum_idx in range(CAL_ACCUM_STEPS):
            micro_seed = CAL_SEED_BASE + step + accum_idx * CAL_ACCUM_SEED_STRIDE
            x, y = make_batch(domain_ids_tgt, seed=micro_seed)
            loss = calibration_batch_loss(x, y)
            (loss / CAL_ACCUM_STEPS).backward()
        source_loss = source_completion_batch_loss(step)
        if source_loss is not None:
            (SOURCE_COMPLETION_LOSS_WEIGHT * source_loss).backward()
            source_completion_loss["updates"] += 1
            source_completion_loss_total += float(source_loss.detach().item())
            source_completion_loss["mean_loss"] = round(
                source_completion_loss_total / source_completion_loss["updates"],
                6,
            )
        nn.utils.clip_grad_norm_(cal_params, 1.0)
        opt_d.step()
        if (
            CAL_EMA_DECAY > 0
            and (step + 1) >= CAL_EMA_START_STEP
            and ((step + 1 - CAL_EMA_START_STEP) % CAL_EMA_EVERY == 0)
        ):
            with torch.no_grad():
                if ema_state is None:
                    ema_state = [p.detach().float().clone() for p in cal_params]
                else:
                    for avg, param in zip(ema_state, cal_params):
                        avg.mul_(CAL_EMA_DECAY).add_(
                            param.detach().float(), alpha=1.0 - CAL_EMA_DECAY
                        )
                ema_updates += 1
        if (step + 1) % 300 == 0:
            print(f"  D step {step + 1}/{CALIBRATION_STEPS}  {time.time() - t_d:.0f}s")
        should_select = (
            CAL_SELECT != "none"
            and (
                (step + 1) % CAL_SELECT_EVERY == 0
                or (step + 1) == CALIBRATION_STEPS
            )
        )
        if should_select:
            stats, score, repeat_rows = evaluate_selection_checkpoint(
                native,
                calibrated,
                posthoc_ids_tgt,
                step + 1,
            )
            row = {
                "step": step + 1,
                "repeats": CAL_SELECT_REPEATS,
                "reduction": CAL_SELECT_REDUCTION,
                "repeat_scores": [
                    repeat_row["score"] for repeat_row in repeat_rows
                ],
                "repeat_rows": repeat_rows,
            }
            row.update(round_validation_row(stats, score))
            calibration_selection["candidates"].append(row)
            print(
                f"  validation select step {step + 1}: "
                f"top1={row['mean_top1_agree']:.4f} "
                f"top5={row['mean_top5_overlap']:.4f} "
                f"JS={row['mean_js']:.5f} score={row['score']:.6f}"
            )
            checkpoint_state = [
                p.detach().cpu().clone() for p in cal_params
            ]
            selection_states.append(
                {
                    "score": float(score),
                    "row": row,
                    "state": checkpoint_state,
                }
            )
            if best_selection_score is None or score > best_selection_score:
                best_selection_score = float(score)
                best_selection_row = row
                best_selection_state = checkpoint_state
            calibrated.train()
    final_training_state = clone_calibration_state(cal_params)
    if ema_state is not None:
        ema_cpu_state = [avg.detach().cpu().clone() for avg in ema_state]
        calibration_ema["updates"] = ema_updates
    if (
        ema_cpu_state is not None
        and CAL_SELECT != "none"
        and CAL_EMA_AS_CANDIDATE
    ):
        current_state = clone_calibration_state(cal_params)
        restore_calibration_state(cal_params, ema_cpu_state)
        calibrated.eval()
        repeat_rows = []
        repeat_scores = []
        for repeat_idx in range(CAL_SELECT_REPEATS):
            stats = validation_l2_stats(
                native,
                calibrated,
                posthoc_ids_tgt,
                n_chunks=CAL_SELECT_CHUNKS,
                seed=CAL_SEED_BASE + 551000 + 1009 * repeat_idx,
            )
            repeat_score = calibration_selection_score(stats)
            repeat_rows.append(round_validation_row(stats, repeat_score))
            repeat_scores.append(repeat_score)
        stats = average_validation_stats(repeat_rows)
        score = reduce_checkpoint_validation_scores(repeat_scores)
        row = {
            "step": f"ema_{CALIBRATION_STEPS}",
            "ema": True,
            "updates": ema_updates,
            "repeats": CAL_SELECT_REPEATS,
            "reduction": CAL_SELECT_REDUCTION,
            "repeat_scores": [
                repeat_row["score"] for repeat_row in repeat_rows
            ],
            "repeat_rows": repeat_rows,
        }
        row.update(round_validation_row(stats, score))
        calibration_selection["candidates"].append(row)
        calibration_ema["candidate"] = row
        selection_states.append(
            {
                "score": float(score),
                "row": row,
                "state": ema_cpu_state,
            }
        )
        if best_selection_score is None or score > best_selection_score:
            best_selection_score = float(score)
            best_selection_row = row
            best_selection_state = ema_cpu_state
        print(
            f"  validation select EMA: "
            f"top1={row['mean_top1_agree']:.4f} "
            f"top5={row['mean_top5_overlap']:.4f} "
            f"JS={row['mean_js']:.5f} score={row['score']:.6f}"
        )
        restore_calibration_state(cal_params, current_state)
        calibrated.train()
    if selection_states:
        ranked_states = sorted(
            selection_states, key=lambda item: item["score"], reverse=True
        )
        n_avg = min(CAL_SELECT_AVG_TOP_N, len(ranked_states))
        if n_avg == 1:
            restore_calibration_state(cal_params, best_selection_state)
            weights = [1.0]
        else:
            if CAL_SELECT_SOUP_WEIGHT_GRID:
                grid_seed = CAL_SEED_BASE + 441000
                grid_candidates = []
                best_grid_score = None
                best_grid_row = None
                best_grid_weights = None
                grid_states = ranked_states[:2]
                grid_steps = [item["row"]["step"] for item in grid_states]
                for first_weight in CAL_SELECT_SOUP_WEIGHT_GRID:
                    candidate_weights = [first_weight, 1.0 - first_weight]
                    restore_weighted_calibration_state(
                        cal_params, grid_states, candidate_weights
                    )
                    calibrated.eval()
                    repeat_stats = []
                    repeat_scores = []
                    for repeat_idx in range(CAL_SELECT_SOUP_GRID_REPEATS):
                        stats = validation_l2_stats(
                            native,
                            calibrated,
                            posthoc_ids_tgt,
                            n_chunks=CAL_SELECT_CHUNKS,
                            seed=grid_seed + 1009 * repeat_idx,
                        )
                        repeat_stats.append(stats)
                        repeat_scores.append(calibration_selection_score(stats))
                    stats = average_validation_stats(repeat_stats)
                    score = reduce_validation_scores(repeat_scores)
                    row = {
                        "steps": grid_steps,
                        "weights": [round(weight, 6) for weight in candidate_weights],
                        "seed": grid_seed,
                        "repeats": CAL_SELECT_SOUP_GRID_REPEATS,
                        "reduction": CAL_SELECT_SOUP_GRID_REDUCTION,
                        "repeat_scores": [
                            round(float(repeat_score), 6)
                            for repeat_score in repeat_scores
                        ],
                    }
                    row.update(round_validation_row(stats, score))
                    grid_candidates.append(row)
                    print(
                        f"  soup grid weights "
                        f"[{candidate_weights[0]:.3f}, {candidate_weights[1]:.3f}]: "
                        f"top1={row['mean_top1_agree']:.4f} "
                        f"top5={row['mean_top5_overlap']:.4f} "
                        f"JS={row['mean_js']:.5f} "
                        f"{CAL_SELECT_SOUP_GRID_REDUCTION}={row['score']:.6f}"
                    )
                    if best_grid_score is None or score > best_grid_score:
                        best_grid_score = float(score)
                        best_grid_row = row
                        best_grid_weights = candidate_weights
                weights = best_grid_weights
                calibration_selection["soup_weight_grid_selection"] = {
                    "seed": grid_seed,
                    "candidates": grid_candidates,
                    "best": best_grid_row,
                }
            else:
                weights = normalized_soup_weights(n_avg)
            restore_weighted_calibration_state(
                cal_params, ranked_states[:n_avg], weights
            )
        calibration_selection["applied"] = True
        calibration_selection["best"] = best_selection_row
        calibration_selection["averaged"] = n_avg > 1
        calibration_selection["averaged_top_n"] = n_avg
        calibration_selection["averaged_steps"] = [
            item["row"]["step"] for item in ranked_states[:n_avg]
        ]
        calibration_selection["soup_weights_applied"] = (
            weights if n_avg > 1 else [1.0]
        )
        if n_avg == 1:
            print(
                f"  Restored validation-selected calibration step "
                f"{best_selection_row['step']} "
                f"(top1={best_selection_row['mean_top1_agree']:.4f}, "
                f"JS={best_selection_row['mean_js']:.5f})"
            )
        else:
            steps = ", ".join(str(item["row"]["step"]) for item in ranked_states[:n_avg])
            weight_text = ", ".join(f"{weight:.3f}" for weight in weights)
            print(
                f"  Restored validation checkpoint soup over top {n_avg} "
                f"steps [{steps}] weights [{weight_text}] "
                f"(best top1={best_selection_row['mean_top1_agree']:.4f}, "
                f"JS={best_selection_row['mean_js']:.5f})"
            )
    selected_state = clone_calibration_state(cal_params)
    final_candidate_states = {
        "selected": selected_state,
        "final": final_training_state,
    }
    if ema_cpu_state is not None:
        final_candidate_states["ema"] = ema_cpu_state
    if best_selection_state is not None:
        final_candidate_states["best"] = best_selection_state
    temporal_avg_state = None
    if CAL_TEMPORAL_AVG_START_STEP > 0:
        temporal_states = [
            item
            for item in selection_states
            if isinstance(item["row"].get("step"), int)
            and item["row"]["step"] >= CAL_TEMPORAL_AVG_START_STEP
        ]
        if temporal_states:
            temporal_weights = [1.0 / len(temporal_states)] * len(temporal_states)
            current_state = clone_calibration_state(cal_params)
            restore_weighted_calibration_state(
                cal_params, temporal_states, temporal_weights
            )
            temporal_avg_state = clone_calibration_state(cal_params)
            restore_calibration_state(cal_params, current_state)
            temporal_steps = [item["row"]["step"] for item in temporal_states]
            calibration_temporal_average.update(
                {
                    "applied": True,
                    "steps": temporal_steps,
                    "weights": temporal_weights,
                }
            )
            final_candidate_states["temporal_avg"] = temporal_avg_state
            print(
                f"  Built temporal checkpoint average from steps {temporal_steps} "
                f"start={CAL_TEMPORAL_AVG_START_STEP}"
            )
    if ema_cpu_state is not None and CAL_EMA_RESTORE:
        restore_calibration_state(cal_params, ema_cpu_state)
        calibration_ema["restored"] = True
        print(
            f"  Restored EMA calibration state "
            f"(decay={CAL_EMA_DECAY}, updates={ema_updates})"
        )
    if temporal_avg_state is not None and CAL_TEMPORAL_AVG_RESTORE:
        restore_calibration_state(cal_params, temporal_avg_state)
        calibration_temporal_average["restored"] = True
        print(
            f"  Restored temporal checkpoint average "
            f"(start={CAL_TEMPORAL_AVG_START_STEP})"
        )
    if CAL_FINAL_SELECT == "validation":
        candidate_rows = []
        best_final_row = None
        best_final_state = None
        select_seed = CAL_SEED_BASE + 661000
        for name in CAL_FINAL_SELECT_CANDIDATES:
            state = final_candidate_states.get(name)
            if state is None:
                continue
            stats, score, repeats = evaluate_final_candidate(
                native,
                calibrated,
                cal_params,
                state,
                posthoc_ids_tgt,
                select_seed + 100003 * len(candidate_rows),
            )
            row = {"name": name, "repeat_rows": repeats}
            row.update(round_validation_row(stats, score))
            candidate_rows.append(row)
            print(
                f"  final selector {name}: "
                f"top1={row['mean_top1_agree']:.4f} "
                f"top5={row['mean_top5_overlap']:.4f} "
                f"JS={row['mean_js']:.5f} score={row['score']:.6f}"
            )
            if best_final_row is None or score > best_final_row["score"]:
                best_final_row = row
                best_final_state = state
        if best_final_state is not None:
            restore_calibration_state(cal_params, best_final_state)
            calibration_final_selection["applied"] = True
            calibration_final_selection["candidates"] = candidate_rows
            calibration_final_selection["selected"] = best_final_row
            print(
                f"  Final selector restored {best_final_row['name']} "
                f"(score={best_final_row['score']:.6f})"
            )
    calibrated.eval()
    for p in cal_params:
        p.requires_grad_(False)
    posthoc_logit_scale = calibrate_posthoc_logit_scale(
        native, calibrated, posthoc_ids_tgt
    )
    if posthoc_logit_scale["applied"]:
        selection = posthoc_logit_scale["selection"]
        print(
            f"  Posthoc logit scale={posthoc_logit_scale['scale']:.4f}  "
            f"calib_JS={selection['mean_js']:.5f}  "
            f"calib_entropy_diff={selection['mean_entropy_diff']:.4f}"
        )
    posthoc_logit_bias = calibrate_posthoc_logit_bias(
        native, calibrated, posthoc_ids_tgt
    )
    if posthoc_logit_bias["applied"]:
        print(
            f"  Posthoc logit bias applied: "
            f"l2={posthoc_logit_bias['bias_l2']:.6f}  "
            f"max_abs={posthoc_logit_bias['bias_max_abs']:.6f}"
        )
    if CAL_SELECT_SOUP_AUDIT_WEIGHTS and len(selection_states) >= 2:
        restored_state = clone_calibration_state(cal_params)
        audit_rows = []
        ranked_soup_states = sorted(
            selection_states, key=lambda item: item["score"], reverse=True
        )[:2]
        audit_steps = [item["row"]["step"] for item in ranked_soup_states]
        for first_weight in CAL_SELECT_SOUP_AUDIT_WEIGHTS:
            audit_weights = [float(first_weight), float(1.0 - first_weight)]
            print(
                f"  Selection soup NIB audit weights "
                f"[{audit_weights[0]:.3f}, {audit_weights[1]:.3f}] "
                f"steps={audit_steps}"
            )
            restore_weighted_calibration_state(
                cal_params, ranked_soup_states, audit_weights
            )
            calibrated.eval()
            audit_rows.append(
                {
                    "steps": audit_steps,
                    "weights": audit_weights,
                    "nib_l2": l2_logit_test(native, calibrated, eval_ids_tgt),
                }
            )
        restore_calibration_state(cal_params, restored_state)
        calibrated.eval()
        calibration_selection["soup_nib_audit"] = audit_rows
    if CAL_SELECT_AUDIT_NIB and selection_states:
        restored_state = clone_calibration_state(cal_params)
        audit_rows = []
        ranked_audit_states = sorted(
            selection_states, key=lambda item: item["score"], reverse=True
        )
        if CAL_SELECT_AUDIT_MAX > 0:
            ranked_audit_states = ranked_audit_states[:CAL_SELECT_AUDIT_MAX]
        for audit_rank, item in enumerate(ranked_audit_states, start=1):
            row = item["row"]
            print(
                f"  Selection NIB audit rank {audit_rank}: "
                f"step={row['step']} validation_top1={row['mean_top1_agree']:.4f}"
            )
            restore_calibration_state(cal_params, item["state"])
            calibrated.eval()
            audit_rows.append(
                {
                    "rank": audit_rank,
                    "step": row["step"],
                    "validation": row,
                    "nib_l2": l2_logit_test(native, calibrated, eval_ids_tgt),
                }
            )
        restore_calibration_state(cal_params, restored_state)
        calibrated.eval()
        calibration_selection["nib_audit"] = audit_rows
    if CAL_FINAL_SOUP_AUDIT_WEIGHTS and len(CAL_FINAL_SOUP_AUDIT_CANDIDATES) == 2:
        first_name, second_name = CAL_FINAL_SOUP_AUDIT_CANDIDATES
        first_state = final_candidate_states.get(first_name)
        second_state = final_candidate_states.get(second_name)
        if first_state is not None and second_state is not None:
            restored_state = clone_calibration_state(cal_params)
            audit_rows = []
            audit_items = [{"state": first_state}, {"state": second_state}]
            for first_weight in CAL_FINAL_SOUP_AUDIT_WEIGHTS:
                audit_weights = [float(first_weight), float(1.0 - first_weight)]
                print(
                    f"  Final soup NIB audit weights "
                    f"[{audit_weights[0]:.3f}, {audit_weights[1]:.3f}] "
                    f"candidates=[{first_name}, {second_name}]"
                )
                restore_weighted_calibration_state(
                    cal_params, audit_items, audit_weights
                )
                calibrated.eval()
                audit_rows.append(
                    {
                        "candidates": [first_name, second_name],
                        "weights": audit_weights,
                        "nib_l2": l2_logit_test(native, calibrated, eval_ids_tgt),
                    }
                )
            restore_calibration_state(cal_params, restored_state)
            calibrated.eval()
            calibration_final_soup_audit["candidates"] = audit_rows
    if CAL_FINAL_AUDIT_NIB:
        restored_state = clone_calibration_state(cal_params)
        audit_rows = []
        for name in CAL_FINAL_AUDIT_CANDIDATES:
            state = final_candidate_states.get(name)
            if state is None:
                continue
            print(f"  Final NIB audit candidate: {name}")
            restore_calibration_state(cal_params, state)
            calibrated.eval()
            row = {"name": name, "nib_l2": l2_logit_test(native, calibrated, eval_ids_tgt)}
            audit_rows.append(row)
        restore_calibration_state(cal_params, restored_state)
        calibrated.eval()
        calibration_final_audit["candidates"] = audit_rows
        calibration_final_audit["selection_used_for_restore"] = False
    ppl_cal = ppl(calibrated, eval_ids_tgt)
    phase_d_sec = time.time() - t_d
    print(f"  Phase D complete: {phase_d_sec:.0f}s  {TARGET_LABEL} calibrated ppl={ppl_cal:.1f}")

    banner("NIB L2 evaluation")
    t_nib = time.time()
    l2 = l2_logit_test(native, calibrated, eval_ids_tgt)
    overall = l2["pass"]
    status_str = "PASS" if overall else "FAIL"
    print()
    print(
        f"  mean_JS          = {l2['mean_js']:.5f}   "
        f"(thr < {base.REGISTRY['js_threshold']})   {'PASS' if l2['js_pass'] else 'FAIL'}"
    )
    print(
        f"  mean_top1_agree  = {l2['mean_top1_agree']:.4f}  "
        f"(thr >= {base.REGISTRY['top1_threshold']})  {'PASS' if l2['top1_pass'] else 'FAIL'}"
    )
    print(
        f"  mean_top5_overlap= {l2['mean_top5_overlap']:.4f}  "
        f"(thr >= {base.REGISTRY['top5_threshold']})  {'PASS' if l2['top5_pass'] else 'FAIL'}"
    )
    print(
        f"  mean_entropy_diff= {l2['mean_entropy_diff']:.4f}  "
        f"(thr < {base.REGISTRY['entropy_diff_threshold']})  "
        f"{'PASS' if l2['entropy_pass'] else 'FAIL'}"
    )
    phase_nib_sec = time.time() - t_nib
    print(f"  NIB overall: {status_str}  ({phase_nib_sec:.1f}s)")

    selective_transfer = {
        "enabled": SELECTIVE_TRANSFER_EVAL,
        "target_domain_corpus": DOMAIN_CORPUS,
        "target_domain_nib_pass": bool(overall),
        "off_domain_corpus": SELECTIVE_OFF_DOMAIN_CORPUS
        if SELECTIVE_TRANSFER_EVAL
        else None,
        "off_domain_detail": selective_off_domain_detail,
        "off_domain_reference": SELECTIVE_OFF_DOMAIN_REFERENCE,
        "off_domain_reference_forward_mode": (
            selective_reference_forward_mode() if SELECTIVE_TRANSFER_EVAL else None
        ),
        "off_domain_nib_l2": None,
        "off_domain_no_leakage_pass": None,
        "selective_transfer_pass": False,
        "claim_boundary": (
            "This opt-in probe checks whether the selected-domain transfer still "
            "matches the target base/reference on an unrelated corpus. It is an "
            "off-domain leakage screen, not a standalone proof of lossless "
            "task-level selective migration."
        ),
    }
    phase_selective_sec = 0.0
    if SELECTIVE_TRANSFER_EVAL:
        banner("Selective transfer off-domain audit")
        t_selective = time.time()
        off_reference_mode = selective_reference_forward_mode()
        off_l2 = l2_logit_test(
            native,
            calibrated,
            selective_off_domain_ids_tgt,
            reference_forward_mode=off_reference_mode,
            calibrated_forward_mode=True,
            seed=NIB_SEED + 4242,
            label="off-domain",
        )
        off_ppl_ref = ppl(
            native,
            selective_off_domain_ids_tgt,
            use_domain=off_reference_mode,
        )
        off_ppl_cal = ppl(
            calibrated,
            selective_off_domain_ids_tgt,
            use_domain=True,
        )
        phase_selective_sec = time.time() - t_selective
        selective_transfer.update(
            {
                "off_domain_nib_l2": off_l2,
                "off_domain_no_leakage_pass": bool(off_l2["pass"]),
                "off_domain_tokens_target": int(len(selective_off_domain_ids_tgt)),
                "ppl_off_domain_reference": round(off_ppl_ref, 3),
                "ppl_off_domain_calibrated": round(off_ppl_cal, 3),
                "ppl_off_domain_relative_overhead": round(
                    (off_ppl_cal / off_ppl_ref) - 1.0,
                    6,
                )
                if off_ppl_ref
                else None,
                "selective_transfer_pass": bool(overall and off_l2["pass"]),
            }
        )
        print(
            f"  Selective off-domain: corpus={SELECTIVE_OFF_DOMAIN_CORPUS}  "
            f"pass={off_l2['pass']}  top5={off_l2['mean_top5_overlap']:.4f}  "
            f"JS={off_l2['mean_js']:.5f}  entropy={off_l2['mean_entropy_diff']:.4f}  "
            f"ppl_ref={off_ppl_ref:.3f}  ppl_cal={off_ppl_cal:.3f}"
        )

    elapsed = time.time() - t_global
    thresholds = dict(base.REGISTRY)
    thresholds["kd_weight"] = KD_WEIGHT
    thresholds["kd_temp"] = KD_TEMP
    thresholds["n_align_sentences"] = N_ALIGN_SENTENCES
    stable_top1_summary = {
        "enabled": stable_top1_stats["enabled"],
        "ce_weight": STABLE_TOP1_CE_WEIGHT,
        "hard_neg_weight": STABLE_TOP1_HARD_NEG_WEIGHT,
        "hard_neg_k": STABLE_TOP1_HARD_NEG_K,
        "require_base_agree": STABLE_TOP1_REQUIRE_BASE_AGREE,
        "min_margin": STABLE_TOP1_MIN_MARGIN,
        "tokens_seen": stable_top1_stats["tokens_seen"],
        "tokens_selected": stable_top1_stats["tokens_selected"],
        "selected_fraction": (
            round(
                stable_top1_stats["tokens_selected"]
                / stable_top1_stats["tokens_seen"],
                6,
            )
            if stable_top1_stats["tokens_seen"]
            else None
        ),
    }
    if hasattr(calibrated, "domain_ensemble_logits"):
        final_ensemble_weights = (
            F.softmax(calibrated.domain_ensemble_logits.detach().float(), dim=0)
            .cpu()
            .tolist()
        )
        rotation_ensemble_weight_summary = {
            "mode": "trainable_softmax",
            "initial_weights": ROTATION_ENSEMBLE_WEIGHTS
            or [1.0 / ROTATION_ENSEMBLE_SIZE] * ROTATION_ENSEMBLE_SIZE,
            "final_weights": [round(float(weight), 6) for weight in final_ensemble_weights],
        }
    elif hasattr(calibrated, "domain_ensemble_weights"):
        final_ensemble_weights = calibrated.domain_ensemble_weights.detach().cpu().tolist()
        rotation_ensemble_weight_summary = {
            "mode": "fixed",
            "initial_weights": [round(float(weight), 6) for weight in final_ensemble_weights],
            "final_weights": [round(float(weight), 6) for weight in final_ensemble_weights],
        }
    else:
        rotation_ensemble_weight_summary = {
            "mode": "single",
            "initial_weights": [1.0],
            "final_weights": [1.0],
        }
    active_domains = (
        list(calibrated.domain_ensemble)
        if hasattr(calibrated, "domain_ensemble")
        else [calibrated.domain]
    )
    rotated_core_final_sha256 = [
        artifact_module_state_sha256(domain.net) for domain in active_domains
    ]
    rotated_full_final_sha256 = [
        artifact_module_state_sha256(domain) for domain in active_domains
    ]
    if SOURCE_PRESERVATION_EVAL:
        source_preservation = evaluate_source_preservation_target(
            source_preservation_source_records, calibrated, tok_tgt
        )
        if source_preservation.get("measured"):
            print(
                f"  Source preservation: "
                f"top1={source_preservation['top1_surface_agree']:.4f}  "
                f"top1_in_topk="
                f"{source_preservation['source_top1_in_target_topk']:.4f}  "
                f"topk_overlap="
                f"{source_preservation['mean_topk_surface_overlap']:.4f}"
            )
            if source_preservation.get("source_top1_completion_preferred") is not None:
                print(
                    "  Cross-tokenizer completion: "
                    f"src_top1_preferred="
                    f"{source_preservation['source_top1_completion_preferred']:.4f}  "
                    f"mean_rank="
                    f"{source_preservation['mean_source_top1_completion_rank']:.4f}"
                )
        else:
            print(
                "  Source preservation: not measured "
                f"({source_preservation.get('reason', 'no_records')})"
            )
    else:
        source_preservation = {
            "enabled": False,
            "measured": False,
            "reason": "disabled",
        }
    target_side_components = [group["name"] for group in trainable_groups]
    abi_artifact = build_abi_artifact(
        source_model=SOURCE_MODEL_ID,
        target_model=TARGET_MODEL_ID,
        d_abi=D_ABI,
        domain_corpus=DOMAIN_CORPUS,
        calibration_mode=CAL_MODE,
        calibration_init=CAL_INIT,
        oracle_mode=ORACLE_MODE,
        source_domain_core_sha256=source_domain_core_sha256,
        source_domain_full_sha256=source_domain_full_sha256,
        rotated_core_initial_sha256=rotated_core_initial_sha256,
        rotated_core_final_sha256=rotated_core_final_sha256,
        rotated_full_initial_sha256=rotated_full_initial_sha256,
        rotated_full_final_sha256=rotated_full_final_sha256,
        copied_payload_core_params=copied_payload_core_params,
        copied_payload_full_params=copied_payload_full_params,
        trainable_groups=trainable_groups,
        alignment=alignment_info,
        target_side_components=target_side_components,
    )
    compatibility_certificate = build_compatibility_certificate(
        artifact=abi_artifact,
        alignment=alignment_info,
        nib_l2=l2,
        posthoc_logit_scale=posthoc_logit_scale,
        posthoc_logit_bias=posthoc_logit_bias,
        target_native_oracle_required=TARGET_NATIVE_ORACLE_REQUIRED,
        source_preservation=(
            source_preservation if source_preservation.get("measured") else None
        ),
        selective_transfer=(
            selective_transfer if selective_transfer.get("enabled") else None
        ),
        oracle_mode=ORACLE_MODE,
    )
    cost_ledger = build_cost_ledger(
        phase_seconds={
            "phase_a_source_domain": phase_a_sec,
            "phase_c_target_reference": phase_c_sec,
            "phase_d_target_calibration": phase_d_sec,
            "nib_eval": phase_nib_sec,
            "selective_off_domain_eval": phase_selective_sec,
        },
        phase_steps={
            "source_domain_steps": DOMAIN_STEPS,
            "target_reference_steps": DOMAIN_STEPS
            if PHASE_C_TRAINS_TARGET_INTERFACE
            else 0,
            "target_calibration_steps": CALIBRATION_STEPS,
            "posthoc_bias_steps": POSTHOC_BIAS_STEPS,
        },
        token_counts={
            "domain_train_tokens_source": len(domain_ids_src),
            "domain_train_tokens_target": len(domain_ids_tgt),
            "posthoc_tokens_target": len(posthoc_ids_tgt),
            "nib_eval_tokens_target": len(eval_ids_tgt),
            "selective_off_domain_tokens_target": len(selective_off_domain_ids_tgt)
            if selective_off_domain_ids_tgt is not None
            else 0,
        },
        param_counts={
            "source_domain_core": artifact_module_param_count(
                source_domain_for_rotation.net
            ),
            "copied_payload_core": copied_payload_core_params,
            "copied_payload_full": copied_payload_full_params,
            "calibration_trainable": trainable_count(cal_params),
            "target_model_module": artifact_module_param_count(native.model),
            "target_lm_head_module": artifact_module_param_count(native.lm_head),
            "calibrated_wrapper": artifact_module_param_count(calibrated),
        },
    )
    results = {
        "experiment": "generic_causal_v2",
        "name": f"exp_generic_causal_nib_v2_{TAG}",
        "variant_type": "strict frozen-domain-core model-family follow-up",
        "source_model": SOURCE_MODEL_ID,
        "source_tokenizer": SOURCE_TOKENIZER_ID,
        "source_label": SOURCE_LABEL,
        "source_model_type": source_model_type,
        "target_model": TARGET_MODEL_ID,
        "target_tokenizer": TARGET_TOKENIZER_ID,
        "target_label": TARGET_LABEL,
        "target_model_type": model_type,
        "source_vocab": source_vocab,
        "target_vocab": vocab_tgt,
        "source_d_model": source_d_model,
        "target_d_model": d_model_tgt,
        "d_abi": D_ABI,
        "seed": EXPERIMENT_SEED,
        "seed_offset": SEED_OFFSET,
        "source_domain_seed_base": SOURCE_DOMAIN_SEED_BASE,
        "native_domain_seed_base": NATIVE_DOMAIN_SEED_BASE,
        "calibration_seed_base": CAL_SEED_BASE,
        "ppl_seed_base": PPL_SEED_BASE,
        "nib_seed": NIB_SEED,
        "domain_corpus": DOMAIN_CORPUS,
        "selective_transfer_eval": SELECTIVE_TRANSFER_EVAL,
        "selective_off_domain_corpus": SELECTIVE_OFF_DOMAIN_CORPUS,
        "selective_off_domain_reference": SELECTIVE_OFF_DOMAIN_REFERENCE,
        "wikitext_domain_split": (
            WIKITEXT_DOMAIN_SPLIT if DOMAIN_CORPUS == "wikitext" else None
        ),
        "wikitext_align_split": (
            WIKITEXT_ALIGN_SPLIT if DOMAIN_CORPUS == "wikitext" else None
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
        "domain_train_tokens_source": int(len(domain_ids_src)),
        "domain_train_tokens_target": int(len(domain_ids_tgt)),
        "posthoc_tokens_target": int(len(posthoc_ids_tgt)),
        "nib_eval_tokens_target": int(len(eval_ids_tgt)),
        "selective_off_domain_tokens_target": int(len(selective_off_domain_ids_tgt))
        if selective_off_domain_ids_tgt is not None
        else 0,
        "domain_steps": DOMAIN_STEPS,
        "calibration_steps": CALIBRATION_STEPS,
        "cal_accum_steps": CAL_ACCUM_STEPS,
        "cal_accum_seed_stride": CAL_ACCUM_SEED_STRIDE,
        "cal_lr": CAL_LR,
        "cal_lr_decay_step": CAL_LR_DECAY_STEP,
        "cal_lr_decay_factor": CAL_LR_DECAY_FACTOR,
        "cal_lr_schedule": CAL_LR_SCHEDULE,
        "cal_lr_final_factor": CAL_LR_FINAL_FACTOR,
        "cal_lr_decayed": cal_lr_decayed,
        "calibration_mode": CAL_MODE,
        "calibration_init": CAL_INIT,
        "oracle_mode": ORACLE_MODE,
        "target_reference_use_domain": TARGET_REFERENCE_USES_DOMAIN,
        "target_reference_bypass_abi": TARGET_REFERENCE_BYPASS_ABI,
        "target_reference_forward_mode": TARGET_REFERENCE_FORWARD_MODE,
        "target_native_oracle_required": TARGET_NATIVE_ORACLE_REQUIRED,
        "phase_c_trains_target_interface": PHASE_C_TRAINS_TARGET_INTERFACE,
        "phase_c_trains_target_domain": PHASE_C_TRAINS_TARGET_DOMAIN,
        "phase_c_trainable_params": phase_c_trainable_params,
        "phase_c_trainable_names": phase_c_trainable_names,
        "target_interface_cache": target_interface_cache,
        "source_completion_loss": source_completion_loss,
        "calibration_selection": calibration_selection,
        "calibration_ema": calibration_ema,
        "calibration_final_selection": calibration_final_selection,
        "calibration_final_audit": calibration_final_audit,
        "calibration_final_soup_audit": calibration_final_soup_audit,
        "calibration_temporal_average": calibration_temporal_average,
        "kd_weight": KD_WEIGHT,
        "kd_temp": KD_TEMP,
        "topk_kd_weight": TOPK_KD_WEIGHT,
        "topk": TOPK,
        "union_topk_kd_weight": UNION_TOPK_KD_WEIGHT,
        "union_topk": UNION_TOPK,
        "union_topk_temp": UNION_TOPK_TEMP,
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
        "stable_top1": stable_top1_summary,
        "topset_weight": TOPSET_WEIGHT,
        "topset_k": TOPSET_K,
        "topset_temp": TOPSET_TEMP,
        "top_logit_mse_weight": TOP_LOGIT_MSE_WEIGHT,
        "top_logit_mse_k": TOP_LOGIT_MSE_K,
        "domain_delta_logit_mse_weight": DOMAIN_DELTA_LOGIT_MSE_WEIGHT,
        "domain_delta_logit_mse_k": DOMAIN_DELTA_LOGIT_MSE_K,
        "domain_delta_logit_mse_center": DOMAIN_DELTA_LOGIT_MSE_CENTER,
        "entropy_weight": ENTROPY_WEIGHT,
        "abi_pre_mse_weight": ABI_PRE_MSE_WEIGHT,
        "abi_state_mse_weight": ABI_STATE_MSE_WEIGHT,
        "confidence_weighting": {
            "mode": CONF_WEIGHT_MODE,
            "center": CONF_WEIGHT_CENTER,
            "temp": CONF_WEIGHT_TEMP,
            "min": CONF_WEIGHT_MIN,
            "max": CONF_WEIGHT_MAX,
        },
        "posthoc_logit_scale": posthoc_logit_scale,
        "posthoc_logit_bias": posthoc_logit_bias,
        "domain_bridge": DOMAIN_BRIDGE,
        "domain_residual_rank": DOMAIN_RESIDUAL_RANK,
        "domain_residual_scale": DOMAIN_RESIDUAL_SCALE,
        "target_residual": TARGET_RESIDUAL,
        "target_residual_rank": TARGET_RESIDUAL_RANK,
        "target_residual_scale": TARGET_RESIDUAL_SCALE,
        "torch_dtype": TORCH_DTYPE_LABEL,
        "batch": BATCH,
        "ppl_batches": PPL_BATCHES,
        "n_align_sentences_requested": N_ALIGN_SENTENCES,
        "align_min_chars": ALIGN_MIN_CHARS,
        "align_select": ALIGN_SELECT,
        "align_pool_sentences": ALIGN_POOL_SENTENCES,
        "align_fit_normalize": ALIGN_FIT_NORMALIZE,
        "align_map": ALIGN_MAP,
        "align_ridge": ALIGN_RIDGE,
        "align_linear_blend": ALIGN_LINEAR_BLEND,
        "rotation_ensemble_size": ROTATION_ENSEMBLE_SIZE,
        "rotation_ensemble_stride": ROTATION_ENSEMBLE_STRIDE,
        "rotation_alignment_pool_sentences": rotation_alignment_collection_limit(),
        "rotation_ensemble_train_weights": ROTATION_ENSEMBLE_TRAIN_WEIGHTS,
        "rotation_ensemble_weights": rotation_ensemble_weight_summary,
        "release_source_before_target": RELEASE_SOURCE_BEFORE_TARGET,
        "train_domain_alpha": TRAIN_DOMAIN_ALPHA,
        "corpus_exclude_globs": list(V2_CORPUS_EXCLUDES),
        "corpus_skipped_files": py_meta["skipped"],
        "calibration_trainable_groups": trainable_groups,
        "calibration_trainable_params": trainable_count(cal_params),
        "n_align_sentences": alignment_info["final_pairs"],
        "alignment_method": "sentence-level mean-pool Procrustes (cross-tokenizer)",
        "alignment_selection": alignment_info,
        "ppl_source": round(ppl_a, 3),
        "ppl_native_target": round(ppl_nat, 3),
        "ppl_target_reference": round(ppl_nat, 3),
        "ppl_calibrated_target": round(ppl_cal, 3),
        "nib_l2": l2,
        "source_preservation": source_preservation,
        "selective_transfer": selective_transfer,
        "abi_artifact": abi_artifact,
        "compatibility_certificate": compatibility_certificate,
        "cost_ledger": cost_ledger,
        "overall_pass": overall,
        "elapsed_min": round(elapsed / 60, 1),
        "thresholds": thresholds,
        "interpretation": (
            "Generic decoder-only target follow-up using the strict v2 protocol. "
            "The rotated domain MLP core is frozen during target calibration unless "
            "ABI_CAL_MODE=train_domain is explicitly selected."
        ),
    }
    out_path = result_path("exp_generic_causal_nib_v2", TAG)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results -> {out_path}")
    banner(f"Done - {status_str} - {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
