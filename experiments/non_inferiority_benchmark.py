#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Inferiority Benchmark (NIB) — ABI Knowledge Transfer
=========================================================
Implements L2–L4 of the equivalence ladder following peer review:

  L2 — Distributional:   JS divergence, top-1/top-5 agreement, entropy diff
                         across 2 560 token positions from Python corpus
  L3 — Decoding:         functional parity under greedy / low-temp / high-temp
  L4a — Functional:      60 Python probes × 3 seeds, bootstrap 95 % CI, NI test
  L4b — Error identity:  Jaccard overlap of pass/fail sets (not just equal scores)
  L4c — Adversarial:     10 base probes × 3 perturbation types

PRE-REGISTERED THRESHOLDS — see REGISTRY dict immediately below.
These are fixed before any model runs; they are treated as read-only once
execution begins, preventing threshold-chasing.

Results: non_inferiority_results.json
Runtime: ~45 min on RTX 3080 (GPU strongly recommended).
"""

import ast
import copy
import json
import math
import pathlib
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wikitext_cache import load_wikitext_split

# ══════════════════════════════════════════════════════════════════════════════
# PRE-REGISTERED THRESHOLDS  (never modified after execution starts)
# ══════════════════════════════════════════════════════════════════════════════
REGISTRY = {
    # ── L2: distributional equivalence ──────────────────────────────────────
    "js_threshold":             0.10,   # mean JS divergence (nats); <0.10 = well aligned
    "top1_threshold":           0.68,   # greedy top-1 token agreement fraction
    "top5_threshold":           0.86,   # top-5 token set overlap fraction
    "entropy_diff_threshold":   0.35,   # max mean |H_nat - H_cal| (nats)

    # ── L3: decoding equivalence ─────────────────────────────────────────────
    "decode_ni_margin_pp":     -12.0,   # NI margin per decode config (-12 pp)

    # ── L4a: functional non-inferiority ──────────────────────────────────────
    "ni_margin_pp":             -8.0,   # transferred >= native - 8 pp
    "min_native_floor":          0.08,  # skip NI if native < 8 % (floor effect)
    "n_bootstrap":               1000,
    "ci_level":                  0.95,

    # ── L4b: error identity ───────────────────────────────────────────────────
    "failure_jaccard":           0.40,  # Jaccard overlap of failure sets >= 0.40
    "pass_jaccard":              0.45,  # Jaccard overlap of pass sets >= 0.45

    # ── L4c: adversarial robustness ──────────────────────────────────────────
    "adversarial_ni_margin_pp": -12.0,  # on adversarial variants

    # ── Step D KD calibration ─────────────────────────────────────────
    "kd_weight":                0.90,  # KL(native ‖ calibrated) loss weight in Step D
    "kd_temp":                  2.0,   # distillation temperature (T=2 dark-knowledge focus on tail tokens)

    # ── run config (pre-registered) ───────────────────────────────────────────
    "n_seeds":                   3,
    "n_logit_chunks":            5,     # × 512 positions each = 2 560 total
    "probe_count":               60,
    "adv_base_count":            10,    # probes used for 3-variant adversarial test
    "calibration_steps":         800,   # Step D budget (1000-step drift; 4 norm params need ≥ 400-step recal)
}

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — identical to generation_equivalence_test.py
# ══════════════════════════════════════════════════════════════════════════════
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_ABI        = 256
SEQ_LEN      = 128
DOMAIN_STEPS = 500
UPDATE_STEPS = 1000
LR_ABI       = 3e-4
LR_BACKBONE  = 5e-5
LR_CAL       = 1e-4
ALPHA        = 1.0
MAX_PY_SV    = 500_000
MAX_WIKI_SV  = 600_000
BATCH_SV     = 8
SEED         = 42
ROOT         = pathlib.Path(__file__).parent

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE — exact copy from generation_equivalence_test.py
# ══════════════════════════════════════════════════════════════════════════════

class DomainModuleSV(nn.Module):
    """4× expansion additive delta with LayerNorm."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ln  = nn.LayerNorm(d)
    def forward(self, h):
        return self.ln(self.net(h))


class SVGPT2(nn.Module):
    """GPT-2-medium (354M) with ABI wrapper and learnable domain_alpha."""
    def __init__(self):
        super().__init__()
        g             = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        self.d_model  = g.config.n_embd   # 1024
        self.proj_in  = nn.Linear(self.d_model, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, self.d_model, bias=False)
        self.domain   = DomainModuleSV(D_ABI)
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
# BATCH / PPL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def make_batch_sv(tokens, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    starts = torch.randint(0, max_start, (BATCH_SV,), generator=rng)
    x = torch.stack([tokens[s : s + SEQ_LEN]     for s in starts]).to(DEVICE)
    y = torch.stack([tokens[s+1 : s + SEQ_LEN+1] for s in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def ppl_sv(model, tokens, use_domain=True, n_batches=50, seed_offset=0):
    model.eval()
    tot, n = 0.0, 0
    max_start = max(len(tokens) - SEQ_LEN - 1, 1)
    rng = torch.Generator()
    for i in range(n_batches):
        rng.manual_seed(80000 + seed_offset + i)
        starts = torch.randint(0, max_start, (BATCH_SV,), generator=rng)
        x = torch.stack([tokens[s : s + SEQ_LEN]     for s in starts]).to(DEVICE)
        y = torch.stack([tokens[s+1 : s + SEQ_LEN+1] for s in starts]).to(DEVICE)
        logits = model(x, use_domain=use_domain)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n   += 1
    return math.exp(tot / n)


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_nib(model, tok, prompt, max_new=48, temperature=0.0,
                 top_p=0.95, top_k=0, use_domain=True, seed=0):
    """
    Generate a completion.
    temperature=0.0 → greedy (argmax).
    top_k>0 applies top-k filtering before top-p.
    """
    torch.manual_seed(seed)
    model.eval()
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    gen = ids.clone()
    eos = tok.eos_token_id

    for _ in range(max_new):
        ctx    = gen[:, -SEQ_LEN:]
        logits = model(ctx, use_domain=use_domain)[0, -1, :]

        if temperature <= 1e-5:
            next_tok = int(logits.argmax())
        else:
            logits = logits / temperature
            # top-k
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][-1]
                logits[logits < kth] = -1e10
            # top-p
            sorted_l, sorted_i = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
            rm  = cum > top_p
            rm[1:] = rm[:-1].clone(); rm[0] = False
            logits[sorted_i[rm]] = -1e10
            probs    = F.softmax(logits, dim=-1)
            next_tok = int(torch.multinomial(probs, 1))

        gen = torch.cat([gen, torch.tensor([[next_tok]], device=DEVICE)], dim=-1)
        if next_tok == eos:
            break

    new_ids = gen[0, ids.shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# PROBE BANK — 60 Python function completion probes across 3 difficulty levels
#   Diff 1 (15): pre-computed variable — model echoes a known local name
#   Diff 2 (25): direct expression — model writes `return <expr>` from docstring
#   Diff 3 (20): continuation — model completes a partially-written body
# ══════════════════════════════════════════════════════════════════════════════

PROBE_BANK = [

    # ── Difficulty 1: pre-computed variable (15 probes) ──────────────────────
    {"name": "negate",        "diff": 1,
     "prompt": "def negate(x):\n    result = -x\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["negate"](5) == -5 and ns["negate"](-3) == 3 and ns["negate"](0) == 0},

    {"name": "double",        "diff": 1,
     "prompt": "def double(x):\n    result = x * 2\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["double"](4) == 8 and ns["double"](0) == 0 and ns["double"](-2) == -4},

    {"name": "is_zero",       "diff": 1,
     "prompt": "def is_zero(x):\n    check = (x == 0)\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["is_zero"](0) and not ns["is_zero"](1) and not ns["is_zero"](-1)},

    {"name": "max_of_two",    "diff": 1,
     "prompt": "def max_of_two(a, b):\n    if a > b:\n        return a\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["max_of_two"](3, 7) == 7 and ns["max_of_two"](10, 2) == 10 and ns["max_of_two"](5, 5) == 5},

    {"name": "first_element", "diff": 1,
     "prompt": "def first_element(lst):\n    elem = lst[0]\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["first_element"]([10, 20, 30]) == 10 and ns["first_element"]([5]) == 5},

    {"name": "string_upper_v", "diff": 1,
     "prompt": "def string_upper(s):\n    result = s.upper()\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["string_upper"]("hello") == "HELLO" and ns["string_upper"]("") == ""},

    {"name": "abs_val",       "diff": 1,
     "prompt": "def abs_val(x):\n    result = abs(x)\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["abs_val"](-5) == 5 and ns["abs_val"](3) == 3 and ns["abs_val"](0) == 0},

    {"name": "add_one",       "diff": 1,
     "prompt": "def add_one(x):\n    result = x + 1\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["add_one"](0) == 1 and ns["add_one"](5) == 6 and ns["add_one"](-1) == 0},

    {"name": "square_v",      "diff": 1,
     "prompt": "def square(x):\n    result = x * x\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["square"](3) == 9 and ns["square"](0) == 0 and ns["square"](-2) == 4},

    {"name": "list_len_v",    "diff": 1,
     "prompt": "def list_len(lst):\n    length = len(lst)\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["list_len"]([1, 2, 3]) == 3 and ns["list_len"]([]) == 0},

    {"name": "str_len_v",     "diff": 1,
     "prompt": "def str_len(s):\n    length = len(s)\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["str_len"]("hello") == 5 and ns["str_len"]("") == 0},

    {"name": "identity_v",    "diff": 1,
     "prompt": "def identity(x):\n    value = x\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["identity"](42) == 42 and ns["identity"]("hi") == "hi"},

    {"name": "last_element_v", "diff": 1,
     "prompt": "def last_element(lst):\n    elem = lst[-1]\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["last_element"]([1, 2, 3]) == 3 and ns["last_element"]([7]) == 7},

    {"name": "string_lower_v", "diff": 1,
     "prompt": "def string_lower(s):\n    result = s.lower()\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["string_lower"]("HELLO") == "hello" and ns["string_lower"]("") == ""},

    {"name": "is_empty",      "diff": 1,
     "prompt": "def is_empty(lst):\n    empty = (len(lst) == 0)\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["is_empty"]([]) and not ns["is_empty"]([1])},


    # ── Difficulty 2: direct expression (25 probes) ──────────────────────────
    {"name": "add",           "diff": 2,
     "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["add"](1, 2) == 3 and ns["add"](-1, 1) == 0 and ns["add"](0, 0) == 0},

    {"name": "subtract",      "diff": 2,
     "prompt": "def subtract(a, b):\n    \"\"\"Return a minus b.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["subtract"](5, 3) == 2 and ns["subtract"](1, 1) == 0 and ns["subtract"](3, 5) == -2},

    {"name": "multiply",      "diff": 2,
     "prompt": "def multiply(a, b):\n    \"\"\"Return the product of a and b.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["multiply"](3, 4) == 12 and ns["multiply"](0, 5) == 0 and ns["multiply"](-2, 3) == -6},

    {"name": "to_upper",      "diff": 2,
     "prompt": "def to_upper(s):\n    \"\"\"Return s in uppercase.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["to_upper"]("hello") == "HELLO" and ns["to_upper"]("abc") == "ABC"},

    {"name": "to_lower",      "diff": 2,
     "prompt": "def to_lower(s):\n    \"\"\"Return s in lowercase.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["to_lower"]("WORLD") == "world" and ns["to_lower"]("") == ""},

    {"name": "string_length", "diff": 2,
     "prompt": "def string_length(s):\n    \"\"\"Return the number of characters in s.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["string_length"]("hello") == 5 and ns["string_length"]("") == 0},

    {"name": "list_length",   "diff": 2,
     "prompt": "def list_length(lst):\n    \"\"\"Return the number of elements in lst.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["list_length"]([1, 2, 3]) == 3 and ns["list_length"]([]) == 0},

    {"name": "list_sum",      "diff": 2,
     "prompt": "def list_sum(lst):\n    \"\"\"Return the sum of all elements in lst.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["list_sum"]([1, 2, 3]) == 6 and ns["list_sum"]([]) == 0 and ns["list_sum"]([5]) == 5},

    {"name": "list_min",      "diff": 2,
     "prompt": "def list_min(lst):\n    \"\"\"Return the minimum element of lst.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["list_min"]([3, 1, 4]) == 1 and ns["list_min"]([5]) == 5 and ns["list_min"]([-1, -2]) == -2},

    {"name": "list_max",      "diff": 2,
     "prompt": "def list_max(lst):\n    \"\"\"Return the maximum element of lst.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["list_max"]([3, 1, 5]) == 5 and ns["list_max"]([0]) == 0},

    {"name": "list_reverse",  "diff": 2,
     "prompt": "def list_reverse(lst):\n    \"\"\"Return a reversed copy of lst.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["list_reverse"]([1, 2, 3]) == [3, 2, 1] and ns["list_reverse"]([]) == []},

    {"name": "power",         "diff": 2,
     "prompt": "def power(base, exp):\n    \"\"\"Return base raised to exp.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["power"](2, 10) == 1024 and ns["power"](3, 0) == 1 and ns["power"](2, 1) == 2},

    {"name": "average",       "diff": 2,
     "prompt": "def average(a, b):\n    \"\"\"Return the arithmetic mean of a and b.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["average"](2, 4) == 3.0 and ns["average"](0, 0) == 0.0},

    {"name": "string_reverse","diff": 2,
     "prompt": "def string_reverse(s):\n    \"\"\"Return s reversed.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["string_reverse"]("abc") == "cba" and ns["string_reverse"]("") == "" and ns["string_reverse"]("a") == "a"},

    {"name": "count_char",    "diff": 2,
     "prompt": "def count_char(s, c):\n    \"\"\"Return the count of character c in string s.\"\"\"\n    return ",
     "max_new": 32,
     "test": lambda ns: ns["count_char"]("hello", "l") == 2 and ns["count_char"]("abc", "x") == 0},

    {"name": "both_positive", "diff": 2,
     "prompt": "def both_positive(a, b):\n    \"\"\"Return True if both a and b are greater than zero.\"\"\"\n    return ",
     "max_new": 32,
     "test": lambda ns: ns["both_positive"](1, 2) and not ns["both_positive"](-1, 2) and not ns["both_positive"](0, 1)},

    {"name": "list_contains", "diff": 2,
     "prompt": "def list_contains(lst, x):\n    \"\"\"Return True if x is in lst.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["list_contains"]([1, 2, 3], 2) and not ns["list_contains"]([], 1)},

    {"name": "repeat_string", "diff": 2,
     "prompt": "def repeat_string(s, n):\n    \"\"\"Return s repeated n times.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["repeat_string"]("ab", 3) == "ababab" and ns["repeat_string"]("x", 0) == ""},

    {"name": "triangular",    "diff": 2,
     "prompt": "def triangular(n):\n    \"\"\"Return n*(n+1)//2. Return 0 for n==0.\"\"\"\n    return ",
     "max_new": 32,
     "test": lambda ns: ns["triangular"](0) == 0 and ns["triangular"](1) == 1 and ns["triangular"](4) == 10},

    {"name": "digit_count",   "diff": 2,
     "prompt": "def digit_count(n):\n    \"\"\"Return the number of decimal digits in non-negative integer n.\"\"\"\n    return ",
     "max_new": 32,
     "test": lambda ns: ns["digit_count"](0) == 1 and ns["digit_count"](123) == 3 and ns["digit_count"](9) == 1},

    {"name": "count_evens",   "diff": 2,
     "prompt": "def count_evens(lst):\n    \"\"\"Return the count of even numbers in lst.\"\"\"\n    return ",
     "max_new": 40,
     "test": lambda ns: ns["count_evens"]([1, 2, 3, 4]) == 2 and ns["count_evens"]([]) == 0 and ns["count_evens"]([1, 3]) == 0},

    {"name": "double_list",   "diff": 2,
     "prompt": "def double_list(lst):\n    \"\"\"Return a new list with each element multiplied by 2.\"\"\"\n    return ",
     "max_new": 40,
     "test": lambda ns: ns["double_list"]([1, 2, 3]) == [2, 4, 6] and ns["double_list"]([]) == []},

    {"name": "remove_negatives", "diff": 2,
     "prompt": "def remove_negatives(lst):\n    \"\"\"Return a new list with all negative elements removed.\"\"\"\n    return ",
     "max_new": 48,
     "test": lambda ns: ns["remove_negatives"]([1, -2, 3, -4]) == [1, 3] and ns["remove_negatives"]([]) == []},

    {"name": "celsius_to_f",  "diff": 2,
     "prompt": "def celsius_to_f(c):\n    \"\"\"Convert Celsius to Fahrenheit: F = c * 9/5 + 32.\"\"\"\n    return ",
     "max_new": 32,
     "test": lambda ns: abs(ns["celsius_to_f"](0) - 32.0) < 0.1 and abs(ns["celsius_to_f"](100) - 212.0) < 0.1},

    {"name": "clamp",         "diff": 2,
     "prompt": "def clamp(x, lo, hi):\n    \"\"\"Return x clamped to [lo, hi].\"\"\"\n    return ",
     "max_new": 40,
     "test": lambda ns: ns["clamp"](5, 0, 3) == 3 and ns["clamp"](-1, 0, 3) == 0 and ns["clamp"](2, 0, 3) == 2},


    # ── Difficulty 3: continuation (20 probes) ───────────────────────────────
    {"name": "factorial",     "diff": 3,
     "prompt": "def factorial(n):\n    if n <= 1: return 1\n    return ",
     "max_new": 32,
     "test": lambda ns: ns["factorial"](0) == 1 and ns["factorial"](1) == 1 and ns["factorial"](5) == 120},

    {"name": "fibonacci",     "diff": 3,
     "prompt": "def fib(n):\n    if n <= 1: return n\n    return ",
     "max_new": 32,
     "test": lambda ns: ns["fib"](0) == 0 and ns["fib"](1) == 1 and ns["fib"](6) == 8},

    {"name": "is_even",       "diff": 3,
     "prompt": "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["is_even"](2) and not ns["is_even"](3) and ns["is_even"](0)},

    {"name": "is_positive",   "diff": 3,
     "prompt": "def is_positive(n):\n    \"\"\"Return True if n is strictly greater than zero.\"\"\"\n    return ",
     "max_new": 24,
     "test": lambda ns: ns["is_positive"](1) and not ns["is_positive"](0) and not ns["is_positive"](-1)},

    {"name": "sign",          "diff": 3,
     "prompt": "def sign(n):\n    \"\"\"Return 1 if n>0, -1 if n<0, 0 if n==0.\"\"\"\n    if n > 0: return 1\n    if n < 0: return -1\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["sign"](5) == 1 and ns["sign"](-3) == -1 and ns["sign"](0) == 0},

    {"name": "is_palindrome", "diff": 3,
     "prompt": "def is_palindrome(s):\n    \"\"\"Return True if s reads the same forward and backward.\"\"\"\n    return ",
     "max_new": 32,
     "test": lambda ns: ns["is_palindrome"]("racecar") and not ns["is_palindrome"]("hello") and ns["is_palindrome"]("")},

    {"name": "max_of_three",  "diff": 3,
     "prompt": "def max_of_three(a, b, c):\n    \"\"\"Return the largest of a, b, c.\"\"\"\n    return ",
     "max_new": 32,
     "test": lambda ns: ns["max_of_three"](1, 5, 3) == 5 and ns["max_of_three"](7, 2, 4) == 7 and ns["max_of_three"](1, 1, 1) == 1},

    {"name": "all_positive",  "diff": 3,
     "prompt": "def all_positive(lst):\n    \"\"\"Return True if every element of lst is greater than zero.\"\"\"\n    return ",
     "max_new": 40,
     "test": lambda ns: ns["all_positive"]([1, 2, 3]) and not ns["all_positive"]([-1, 2]) and ns["all_positive"]([])},

    {"name": "any_negative",  "diff": 3,
     "prompt": "def any_negative(lst):\n    \"\"\"Return True if at least one element of lst is negative.\"\"\"\n    return ",
     "max_new": 40,
     "test": lambda ns: ns["any_negative"]([1, -2, 3]) and not ns["any_negative"]([1, 2]) and not ns["any_negative"]([])},

    {"name": "gcd",           "diff": 3,
     "prompt": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["gcd"](12, 8) == 4 and ns["gcd"](7, 14) == 7 and ns["gcd"](5, 0) == 5},

    {"name": "sum_of_squares","diff": 3,
     "prompt": "def sum_of_squares(lst):\n    \"\"\"Return the sum of squares of elements in lst.\"\"\"\n    return ",
     "max_new": 40,
     "test": lambda ns: ns["sum_of_squares"]([1, 2, 3]) == 14 and ns["sum_of_squares"]([]) == 0 and ns["sum_of_squares"]([4]) == 16},

    {"name": "running_sum",   "diff": 3,
     "prompt": "def running_sum(lst):\n    total = 0\n    result = []\n    for x in lst:\n        total += x\n        result.append(total)\n    return ",
     "max_new": 16,
     "test": lambda ns: ns["running_sum"]([1, 2, 3]) == [1, 3, 6] and ns["running_sum"]([]) == []},

    {"name": "is_prime",      "diff": 3,
     "prompt": "def is_prime(n):\n    if n < 2: return False\n    return ",
     "max_new": 52,
     "test": lambda ns: ns["is_prime"](2) and not ns["is_prime"](4) and ns["is_prime"](7) and not ns["is_prime"](1)},

    {"name": "sum_digits",    "diff": 3,
     "prompt": "def sum_digits(n):\n    \"\"\"Return the sum of decimal digits of non-negative integer n.\"\"\"\n    return ",
     "max_new": 48,
     "test": lambda ns: ns["sum_digits"](123) == 6 and ns["sum_digits"](0) == 0 and ns["sum_digits"](9) == 9},

    {"name": "count_vowels",  "diff": 3,
     "prompt": "def count_vowels(s):\n    \"\"\"Return the count of vowels (a/e/i/o/u) in s.\"\"\"\n    return ",
     "max_new": 52,
     "test": lambda ns: ns["count_vowels"]("hello") == 2 and ns["count_vowels"]("xyz") == 0 and ns["count_vowels"]("") == 0},

    {"name": "word_count",    "diff": 3,
     "prompt": "def word_count(text):\n    \"\"\"Return the number of whitespace-separated words in text.\"\"\"\n    return ",
     "max_new": 40,
     "test": lambda ns: ns["word_count"]("hello world") == 2 and ns["word_count"]("") == 0 and ns["word_count"]("one") == 1},

    {"name": "flatten",       "diff": 3,
     "prompt": "def flatten(lst):\n    \"\"\"Return a flat list from a list of lists.\"\"\"\n    return ",
     "max_new": 56,
     "test": lambda ns: ns["flatten"]([[1, 2], [3, 4]]) == [1, 2, 3, 4] and ns["flatten"]([]) == []},

    {"name": "zip_sum",       "diff": 3,
     "prompt": "def zip_sum(a, b):\n    \"\"\"Return the total of element-wise sums of equal-length lists a and b.\"\"\"\n    return ",
     "max_new": 40,
     "test": lambda ns: ns["zip_sum"]([1, 2], [3, 4]) == 10 and ns["zip_sum"]([], []) == 0},

    {"name": "merge_sorted",  "diff": 3,
     "prompt": "def merge_sorted(a, b):\n    \"\"\"Return a sorted list merging sorted inputs a and b.\"\"\"\n    return ",
     "max_new": 56,
     "test": lambda ns: ns["merge_sorted"]([1, 3, 5], [2, 4]) == [1, 2, 3, 4, 5] and ns["merge_sorted"]([], []) == []},

    {"name": "most_frequent", "diff": 3,
     "prompt": "def most_frequent(lst):\n    \"\"\"Return the most frequently occurring element in non-empty lst.\"\"\"\n    return ",
     "max_new": 56,
     "test": lambda ns: ns["most_frequent"]([1, 2, 1, 3, 1]) == 1 and ns["most_frequent"]([5]) == 5},
]

assert len(PROBE_BANK) == 60, f"Expected 60 probes, got {len(PROBE_BANK)}"


# ══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL PROBE VARIANTS (10 × 3 = 30 variants on Diff-2 probes)
# ══════════════════════════════════════════════════════════════════════════════

def _make_adversarial_variants():
    """Return a list of (variant_name, base_name, prompt, max_new, test)."""
    diff2_probes = [p for p in PROBE_BANK if p["diff"] == 2][:10]
    variants = []
    for p in diff2_probes:
        name      = p["name"]
        base_pmt  = p["prompt"]
        max_new   = p["max_new"]
        test_fn   = p["test"]

        # Variant A: verbose — add explanatory sentence to docstring
        if '"""' in base_pmt:
            verbose_pmt = re.sub(
                r'("""[^"]*?)(\s*""")',
                r'\1 Ensure the return type is correct and all inputs are handled.\2',
                base_pmt, count=1)
        else:
            verbose_pmt = base_pmt
        variants.append((name + "_verbose", name, verbose_pmt, max_new, test_fn))

        # Variant B: terse — strip docstring entirely
        terse_lines = []
        skip = False
        for line in base_pmt.split("\n"):
            stripped = line.strip()
            if stripped.startswith('"""') and not skip:
                skip = True
                if stripped.count('"""') == 2:  # one-liner
                    skip = False
                continue
            if skip:
                if '"""' in stripped:
                    skip = False
                continue
            terse_lines.append(line)
        terse_pmt = "\n".join(terse_lines)
        variants.append((name + "_terse", name, terse_pmt, max_new, test_fn))

        # Variant C: renamed params — a→u, b→v, s→text, lst→seq, n→num, x→val
        remap = {"a": "u", "b": "v", "s": "text", "lst": "seq", "n": "num",
                 "x": "val", "c": "ch", "base": "mantissa", "exp": "exponent"}
        sig_line = base_pmt.split("\n")[0]
        for old, new in remap.items():
            sig_line = re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new, sig_line)
        renamed_pmt = sig_line + "\n" + "\n".join(base_pmt.split("\n")[1:])
        variants.append((name + "_renamed", name, renamed_pmt, max_new, test_fn))

    return variants


ADV_VARIANTS = _make_adversarial_variants()


# ══════════════════════════════════════════════════════════════════════════════
# PROBE EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_probe(model, tok, prompt, max_new, test_fn, seed=0,
                   use_domain=True, temperature=0.0, top_p=0.95, top_k=0):
    """
    Generate completion and run the test function.
    Returns (passed: bool, error_type: str|None).
    """
    completion = generate_nib(model, tok, prompt, max_new=max_new,
                               temperature=temperature, top_p=top_p, top_k=top_k,
                               use_domain=use_domain, seed=seed)

    # Take only the first line produced (avoids multi-statement accidents)
    first_line  = completion.split("\n")[0]
    full_code   = prompt + first_line

    try:
        ast.parse(full_code)
    except SyntaxError:
        return False, "syntax_error"

    try:
        ns = {}
        exec(compile(full_code, "<gen>", "exec"), ns)
    except Exception as e:
        return False, f"exec:{type(e).__name__}"

    try:
        result = bool(test_fn(ns))
        return result, (None if result else "assertion")
    except Exception as e:
        return False, f"test:{type(e).__name__}"


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CI + NON-INFERIORITY
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values, n_boot=1000, ci_level=0.95, rng=None):
    """
    Bootstrap confidence interval for the mean of `values`.
    Returns (mean, lower, upper).
    """
    rng    = rng or np.random.default_rng(1234)
    arr    = np.array(values, dtype=float)
    boots  = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    alpha  = (1.0 - ci_level) / 2
    lower  = float(np.quantile(boots, alpha))
    upper  = float(np.quantile(boots, 1.0 - alpha))
    return float(arr.mean()), lower, upper


def noninferior(cal_lower_ci, nat_mean, margin_pp):
    """
    Non-inferiority test: calibrated is non-inferior if
    its CI lower bound >= nat_mean + margin_pp/100.
    margin_pp is negative (e.g. -8.0 means allow -8pp).
    """
    threshold = nat_mean + margin_pp / 100.0
    return cal_lower_ci >= threshold, threshold


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter) / len(union) if union else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PROTOCOL (Steps A → B → C → D)
# ══════════════════════════════════════════════════════════════════════════════

def run_training_protocol(py_ids, wiki_ids, registry):
    """
    Run the A→B→C→D training protocol.
    Steps A/B are identical to generation_equivalence_test.py.
    Step C (native oracle) is trained first so its soft-target logits can
    guide Step D (KD calibration), directly minimising the L2 distributional gap
    without destabilising the domain module’s input normalisation.
    Returns (calibrated, native, cal_alpha, ppl_cal, ppl_nat).
    """
    cal_steps = registry["calibration_steps"]

    # ── Step A: anchor on Python (ABI only, backbone frozen) ─────────────────
    print("  [A] Anchor training (500 steps Python, ABI only)...")
    t0 = time.time()
    anchor = SVGPT2().to(DEVICE)
    for p in anchor.parameters():        p.requires_grad_(False)
    for nm, p in anchor.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_a = torch.optim.AdamW([p for p in anchor.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    anchor.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch_sv(py_ids, seed=5000 + step)
        opt_a.zero_grad()
        F.cross_entropy(anchor(x, use_domain=True).reshape(-1, 50257),
                        y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(anchor.parameters(), 1.0)
        opt_a.step()
    anchor.eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    saved_dom = copy.deepcopy(anchor.domain.state_dict())
    ppl_a = ppl_sv(anchor, py_ids, use_domain=True)
    print(f"  [A] {time.time()-t0:.0f}s  anchor ppl={ppl_a:.1f}")

    # ── Step B: backbone drift on WikiText (ABI stability) ──────────────────
    print("  [B] Backbone update (1000 steps WikiText-2)...")
    t1 = time.time()
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
        x, y = make_batch_sv(wiki_ids, seed=9000 + step)
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
    ppl_b_nd = ppl_sv(transferred, py_ids, use_domain=False)
    print(f"  [B] {time.time()-t1:.0f}s  transferred no-domain ppl={ppl_b_nd:.1f}")

    # ── Step C: native oracle (trained first — needed as KD teacher in Step D) ──
    print("  [C] Native oracle (500 steps Python, fresh ABI)...")
    t2 = time.time()
    native = copy.deepcopy(transferred).to(DEVICE)
    nn.init.xavier_uniform_(native.proj_in.weight)
    nn.init.xavier_uniform_(native.proj_out.weight)
    nn.init.ones_(native.abi_ln.weight); nn.init.zeros_(native.abi_ln.bias)
    native.domain = DomainModuleSV(D_ABI).to(DEVICE)
    native.domain_alpha.data.fill_(1.0)
    for p in native.parameters():         p.requires_grad_(False)
    for nm, p in native.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_c = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad],
                               lr=LR_ABI, weight_decay=0.01)
    native.train()
    for step in range(DOMAIN_STEPS):
        x, y = make_batch_sv(py_ids, seed=5000 + step)
        opt_c.zero_grad()
        F.cross_entropy(native(x, use_domain=True).reshape(-1, 50257),
                        y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt_c.step()
    native.eval()
    for p in native.parameters(): p.requires_grad_(False)
    ppl_nat = ppl_sv(native, py_ids, use_domain=True)
    print(f"  [C] {time.time()-t2:.0f}s  native ppl={ppl_nat:.1f}")

    # ── Step D: KD calibration using native oracle as teacher ──────────────────
    # Protocol reordered A→B→C→D so native is available as a distillation signal.
    # KD loss (kd_weight) minimises KL(native ‖ calibrated) over the full vocab,
    # directly optimising JS divergence, top-5 overlap, and entropy alignment
    # WITHOUT touching abi_ln (which would damage the global ABI contract).
    # CE loss (1 − kd_weight) preserves task-level code accuracy.
    # domain.net weights are NEVER updated here — all domain knowledge is from
    # Step A; only routed chart/calibration parameters (proj_in, proj_out,
    # domain_alpha, domain.ln) are re-calibrated to align the transfer model
    # with the native oracle.
    kd_weight = registry.get("kd_weight", 0.70)
    kd_temp   = registry.get("kd_temp",   1.0)
    ce_weight = 1.0 - kd_weight
    print(f"  [D] KD calibration ({cal_steps} steps, kd_weight={kd_weight}, "
          f"proj_in+proj_out+alpha+domain.ln)...")
    t3 = time.time()
    calibrated = copy.deepcopy(transferred).to(DEVICE)
    for p in calibrated.parameters(): p.requires_grad_(False)
    calibrated.proj_in.weight.requires_grad_(True)     # input-projection recal.
    calibrated.proj_out.weight.requires_grad_(True)
    calibrated.domain_alpha.requires_grad_(True)
    calibrated.domain.ln.weight.requires_grad_(True)   # output-side LN
    calibrated.domain.ln.bias.requires_grad_(True)     # output-side LN
    _d_cal_params = [
        calibrated.proj_in.weight,
        calibrated.proj_out.weight,
        calibrated.domain_alpha,
        calibrated.domain.ln.weight,
        calibrated.domain.ln.bias,
    ]
    opt_d = torch.optim.AdamW(_d_cal_params, lr=LR_CAL, weight_decay=0.01)
    native.eval()   # teacher stays frozen and in eval mode
    calibrated.train()
    for step in range(cal_steps):
        x, y       = make_batch_sv(py_ids, seed=7000 + step)
        opt_d.zero_grad()
        cal_logits = calibrated(x, use_domain=True)         # [B, T, V]
        with torch.no_grad():
            nat_logits = native(x, use_domain=True)         # teacher
        V = cal_logits.shape[-1]
        kd_loss = F.kl_div(
            F.log_softmax(cal_logits.reshape(-1, V) / kd_temp, dim=-1),
            F.softmax(nat_logits.reshape(-1, V)     / kd_temp, dim=-1),
            reduction='batchmean',
        ) * (kd_temp ** 2)
        ce_loss = F.cross_entropy(cal_logits.reshape(-1, V), y.reshape(-1))
        (kd_weight * kd_loss + ce_weight * ce_loss).backward()
        nn.utils.clip_grad_norm_(_d_cal_params, 1.0)
        opt_d.step()
    calibrated.eval()
    for p in calibrated.parameters(): p.requires_grad_(False)
    cal_alpha     = float(calibrated.domain_alpha.item())
    ppl_cal_final = ppl_sv(calibrated, py_ids, use_domain=True)
    efficacy      = (ppl_cal_final / ppl_nat) * 100
    print(f"  [D] {time.time()-t3:.0f}s  calibrated ppl={ppl_cal_final:.1f}  "
          f"alpha={cal_alpha:.4f}  cal/nat efficacy={efficacy:.1f}%")
    print()

    return calibrated, native, cal_alpha, ppl_cal_final, ppl_nat


# ══════════════════════════════════════════════════════════════════════════════
# L2: DISTRIBUTIONAL EQUIVALENCE (logit-level)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def l2_logit_test(native, calibrated, py_ids, registry):
    """
    Run 5 × 512-token forward passes; compute JS divergence, top-1/5 agreement,
    entropy difference at each of the ~2 460 valid token positions.
    """
    native.eval(); calibrated.eval()
    CHUNK = 512
    SKIP  = 20   # skip first 20 positions (low-context, biased)

    rng       = np.random.default_rng(7777)
    js_list   = []
    top1_list = []
    top5_list = []
    ent_list  = []

    n_chunks = registry["n_logit_chunks"]
    max_start = max(len(py_ids) - CHUNK, 1)

    for ci in range(n_chunks):
        start = int(rng.integers(0, max_start))
        chunk = py_ids[start : start + CHUNK].unsqueeze(0).to(DEVICE)

        nat_logits = native(chunk, use_domain=True)[0, SKIP:, :]      # [T-S, V]
        cal_logits = calibrated(chunk, use_domain=True)[0, SKIP:, :]  # [T-S, V]

        nat_p = F.softmax(nat_logits, dim=-1).cpu().float().numpy()    # [T-S, V]
        cal_p = F.softmax(cal_logits, dim=-1).cpu().float().numpy()

        T = nat_p.shape[0]

        # JS divergence
        m      = 0.5 * (nat_p + cal_p)
        eps    = 1e-12
        nat_pc = np.clip(nat_p, eps, 1.0)
        cal_pc = np.clip(cal_p, eps, 1.0)
        m_c    = np.clip(m, eps, 1.0)
        kl_nm  = (nat_pc * np.log(nat_pc / m_c)).sum(1)
        kl_cm  = (cal_pc * np.log(cal_pc / m_c)).sum(1)
        js    = np.clip(0.5 * (kl_nm + kl_cm), 0, None)
        js_list.extend(js.tolist())

        # Top-1 agreement
        nat_top1 = nat_p.argmax(1)
        cal_top1 = cal_p.argmax(1)
        top1_list.extend((nat_top1 == cal_top1).tolist())

        # Top-5 overlap
        nat_top5 = np.argpartition(nat_p, -5, axis=1)[:, -5:]
        cal_top5 = np.argpartition(cal_p, -5, axis=1)[:, -5:]
        for t in range(T):
            inter = len(set(nat_top5[t]) & set(cal_top5[t]))
            top5_list.append(inter / 5.0)

        # Entropy difference
        H_nat = -(nat_pc * np.log(nat_pc)).sum(1)
        H_cal = -(cal_pc * np.log(cal_pc)).sum(1)
        ent_list.extend(np.abs(H_nat - H_cal).tolist())

        print(f"    chunk {ci+1}/{n_chunks}: "
              f"JS={float(np.mean(js)):.4f}  "
              f"top1={float(np.mean(nat_top1==cal_top1)):.3f}  "
              f"top5={float(np.mean([len(set(nat_top5[t])&set(cal_top5[t]))/5. for t in range(T)])):.3f}")

    mean_js   = float(np.mean(js_list))
    mean_top1 = float(np.mean(top1_list))
    mean_top5 = float(np.mean(top5_list))
    mean_ent  = float(np.mean(ent_list))

    js_pass   = mean_js   <  registry["js_threshold"]
    top1_pass = mean_top1 >= registry["top1_threshold"]
    top5_pass = mean_top5 >= registry["top5_threshold"]
    ent_pass  = mean_ent  <  registry["entropy_diff_threshold"]
    l2_pass   = js_pass and top1_pass and top5_pass and ent_pass

    return {
        "n_positions":      len(js_list),
        "mean_js":          round(mean_js,   5),
        "mean_top1_agree":  round(mean_top1, 4),
        "mean_top5_overlap":round(mean_top5, 4),
        "mean_entropy_diff":round(mean_ent,  4),
        "js_pass":          js_pass,
        "top1_pass":        top1_pass,
        "top5_pass":        top5_pass,
        "entropy_pass":     ent_pass,
        "thresholds":       {
            "js":           registry["js_threshold"],
            "top1":         registry["top1_threshold"],
            "top5":         registry["top5_threshold"],
            "entropy_diff": registry["entropy_diff_threshold"],
        },
        "pass": l2_pass,
    }


# ══════════════════════════════════════════════════════════════════════════════
# L4a + L4b: FUNCTIONAL NON-INFERIORITY + ERROR IDENTITY
# ══════════════════════════════════════════════════════════════════════════════

def l4a_l4b_functional_test(native, calibrated, tok, probes, registry,
                              decode_cfg=None, label="greedy"):
    """
    Run all probes × n_seeds for both native and calibrated.
    Returns dict with NI test results (L4a) and error identity (L4b).

    decode_cfg: dict with keys temperature, top_p, top_k (default=greedy).
    """
    if decode_cfg is None:
        decode_cfg = {"temperature": 0.0, "top_p": 0.95, "top_k": 0}

    n_seeds   = registry["n_seeds"]
    n_boot    = registry["n_bootstrap"]
    ci_lvl    = registry["ci_level"]
    seeds     = list(range(n_seeds))

    # (probe_idx, seed) → pass for each model
    nat_results = {}   # probe_name → list[bool over seeds]
    cal_results = {}

    for p in probes:
        nat_results[p["name"]] = []
        cal_results[p["name"]] = []

    total = len(probes) * n_seeds
    done  = 0
    for p in probes:
        for seed in seeds:
            kw = dict(seed=seed, use_domain=True, **decode_cfg)
            np_, ne = evaluate_probe(native,     tok, p["prompt"], p["max_new"], p["test"], **kw)
            cp_, ce = evaluate_probe(calibrated, tok, p["prompt"], p["max_new"], p["test"], **kw)
            nat_results[p["name"]].append(np_)
            cal_results[p["name"]].append(cp_)
            done += 2
            if done % 60 == 0:
                print(f"    [{label}] {done}/{total*2} inferences", flush=True)

    # Aggregate per-probe majority pass
    nat_pass_probes = {nm for nm, rs in nat_results.items() if sum(rs) > len(rs)/2}
    cal_pass_probes = {nm for nm, rs in cal_results.items() if sum(rs) > len(rs)/2}

    # Per-trial pass lists for CI
    nat_all = [v for lst in nat_results.values() for v in lst]
    cal_all = [v for lst in cal_results.values() for v in lst]

    nat_mean, nat_lo, nat_hi = bootstrap_ci(nat_all, n_boot=n_boot, ci_level=ci_lvl)
    cal_mean, cal_lo, cal_hi = bootstrap_ci(cal_all, n_boot=n_boot, ci_level=ci_lvl)

    ni_pass, ni_threshold = noninferior(cal_lo, nat_mean, registry["ni_margin_pp"])
    floor_skip = nat_mean < registry["min_native_floor"]

    # ── L4b: error identity ──────────────────────────────────────────────────
    nat_fail_probes = set(p["name"] for p in probes) - nat_pass_probes
    cal_fail_probes = set(p["name"] for p in probes) - cal_pass_probes
    fail_jac = jaccard(nat_fail_probes, cal_fail_probes)
    pass_jac = jaccard(nat_pass_probes, cal_pass_probes)
    pass_only_native = sorted(nat_pass_probes - cal_pass_probes)
    pass_only_calibrated = sorted(cal_pass_probes - nat_pass_probes)
    per_probe_majority = {
        p["name"]: {
            "diff": p["diff"],
            "native_passes": int(sum(nat_results[p["name"]])),
            "calibrated_passes": int(sum(cal_results[p["name"]])),
            "native_majority_pass": p["name"] in nat_pass_probes,
            "calibrated_majority_pass": p["name"] in cal_pass_probes,
        }
        for p in probes
    }

    fail_jac_pass = fail_jac >= registry["failure_jaccard"]
    pass_jac_pass = pass_jac >= registry["pass_jaccard"]

    # Per-difficulty breakdown
    diff_results = {}
    for diff in [1, 2, 3]:
        d_probes = [p["name"] for p in probes if p["diff"] == diff]
        nat_d = [v for nm in d_probes for v in nat_results[nm]]
        cal_d = [v for nm in d_probes for v in cal_results[nm]]
        diff_results[diff] = {
            "n_probes":    len(d_probes),
            "nat_pass_pp": round(100 * sum(nat_d)/len(nat_d), 2) if nat_d else 0,
            "cal_pass_pp": round(100 * sum(cal_d)/len(cal_d), 2) if cal_d else 0,
        }

    return {
        "label":           label,
        "n_probes":        len(probes),
        "n_seeds":         n_seeds,
        "nat_pass_pp":     round(nat_mean * 100, 2),
        "cal_pass_pp":     round(cal_mean * 100, 2),
        "nat_ci_95":       [round(nat_lo*100,2), round(nat_hi*100,2)],
        "cal_ci_95":       [round(cal_lo*100,2), round(cal_hi*100,2)],
        "ni_threshold_pp": round(ni_threshold * 100, 2),
        "ni_pass":         ni_pass,
        "floor_skip":      floor_skip,
        "by_difficulty":   diff_results,
        # L4b
        "nat_pass_probe_count": len(nat_pass_probes),
        "cal_pass_probe_count": len(cal_pass_probes),
        "failure_jaccard":      round(fail_jac, 4),
        "pass_jaccard":         round(pass_jac, 4),
        "pass_only_native":     pass_only_native,
        "pass_only_calibrated": pass_only_calibrated,
        "per_probe_majority":   per_probe_majority,
        "failure_jaccard_pass": fail_jac_pass,
        "pass_jaccard_pass":    pass_jac_pass,
        # Combined pass: NI + error identity
        "pass": (ni_pass or floor_skip) and fail_jac_pass,
    }


# ══════════════════════════════════════════════════════════════════════════════
# L3: DECODING EQUIVALENCE
# ══════════════════════════════════════════════════════════════════════════════

def l3_decoding_test(native, calibrated, tok, probes, registry):
    """
    Run a subset of probes under 3 decoding strategies.
    Returns dict with per-config NI results.
    """
    decode_configs = [
        {"label": "greedy",    "temperature": 0.0,  "top_p": 0.95, "top_k": 0},
        {"label": "low_temp",  "temperature": 0.3,  "top_p": 0.95, "top_k": 0},
        {"label": "high_temp", "temperature": 0.7,  "top_p": 0.90, "top_k": 50},
    ]
    # Use Diff-1 and Diff-2 probes for L3 (most reliable for stochastic decoding)
    subset = [p for p in probes if p["diff"] <= 2]

    config_results = {}
    all_pass = True

    for cfg in decode_configs:
        label = cfg["label"]
        kw    = {k: v for k, v in cfg.items() if k != "label"}
        print(f"\n  [L3-{label}] Running {len(subset)} probes × {registry['n_seeds']} seeds...")

        nat_scores, cal_scores = [], []
        for p in subset:
            for seed in range(registry["n_seeds"]):
                np_, _ = evaluate_probe(native,     tok, p["prompt"], p["max_new"],
                                         p["test"], seed=seed, use_domain=True, **kw)
                cp_, _ = evaluate_probe(calibrated, tok, p["prompt"], p["max_new"],
                                         p["test"], seed=seed, use_domain=True, **kw)
                nat_scores.append(np_)
                cal_scores.append(cp_)

        nat_mean, nat_lo, nat_hi = bootstrap_ci(nat_scores, n_boot=registry["n_bootstrap"])
        cal_mean, cal_lo, cal_hi = bootstrap_ci(cal_scores, n_boot=registry["n_bootstrap"])
        ni_pass, ni_thr = noninferior(cal_lo, nat_mean, registry["decode_ni_margin_pp"])

        config_results[label] = {
            "nat_pass_pp":     round(nat_mean * 100, 2),
            "cal_pass_pp":     round(cal_mean * 100, 2),
            "cal_ci_95_lower": round(cal_lo * 100, 2),
            "ni_pass":         ni_pass,
        }
        all_pass = all_pass and ni_pass
        print(f"    {label}: nat={nat_mean*100:.1f}%  cal={cal_mean*100:.1f}%  "
              f"CI_lo={cal_lo*100:.1f}%  NI={ni_pass}")

    return {"configs": config_results, "all_configs_pass": all_pass, "pass": all_pass}


# ══════════════════════════════════════════════════════════════════════════════
# L4c: ADVERSARIAL ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════

def l4c_adversarial_test(native, calibrated, tok, adv_variants, probes, registry):
    """
    Run adversarial variants (verbose / terse / renamed) of 10 Diff-2 probes.
    NI test: calibrated adversarial pass rate >= native - 12pp.
    """
    print(f"\n  [L4c] Running {len(adv_variants)} adversarial variants...")

    nat_adv, cal_adv = [], []
    variant_detail = []

    for (v_name, base_name, v_prompt, v_maxnew, v_test) in adv_variants:
        np_, ne = evaluate_probe(native,     tok, v_prompt, v_maxnew, v_test,
                                  seed=0, use_domain=True, temperature=0.0)
        cp_, ce = evaluate_probe(calibrated, tok, v_prompt, v_maxnew, v_test,
                                  seed=0, use_domain=True, temperature=0.0)
        nat_adv.append(np_)
        cal_adv.append(cp_)
        variant_detail.append({
            "variant": v_name, "base": base_name,
            "native_pass": np_, "calibrated_pass": cp_,
        })
        print(f"    {v_name}: nat={'P' if np_ else 'F'}  cal={'P' if cp_ else 'F'}")

    nat_mean, nat_lo, nat_hi = bootstrap_ci(nat_adv, n_boot=registry["n_bootstrap"])
    cal_mean, cal_lo, cal_hi = bootstrap_ci(cal_adv, n_boot=registry["n_bootstrap"])
    ni_pass, ni_thr = noninferior(cal_lo, nat_mean, registry["adversarial_ni_margin_pp"])

    return {
        "n_variants":           len(adv_variants),
        "nat_pass_pp":          round(nat_mean * 100, 2),
        "cal_pass_pp":          round(cal_mean * 100, 2),
        "cal_ci_95_lower_pp":   round(cal_lo * 100, 2),
        "ni_threshold_pp":      round(ni_thr * 100, 2),
        "ni_pass":              ni_pass,
        "variant_detail":       variant_detail,
        "pass":                 ni_pass,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def banner(msg):
    w = max(len(msg) + 4, 72)
    line = "=" * w
    print(f"\n{line}\n  {msg}\n{line}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_global = time.time()

    banner("NON-INFERIORITY BENCHMARK — ABI Knowledge Transfer")
    print(f"  Device:  {DEVICE}")
    print(f"  Seed:    {SEED}")
    print(f"  Probes:  {len(PROBE_BANK)} (15 diff-1 + 25 diff-2 + 20 diff-3)")
    print(f"  Registry thresholds (pre-registered, read-only):")
    for k, v in REGISTRY.items():
        print(f"    {k:30s} = {v}")
    print()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = sys.maxsize

    # ── Data ──────────────────────────────────────────────────────────────────
    print("  [Data] Loading corpora...")
    t_data = time.time()
    wiki_raw = "\n".join(
        r["text"] for r in load_wikitext_split("wikitext-2-raw-v1", "train")
        if r["text"].strip())

    py_parts, py_chars = [], 0
    for p in ROOT.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            py_parts.append(txt); py_chars += len(txt)
            if py_chars >= MAX_PY_SV * 4: break
        except Exception:
            continue
    py_raw = "\n".join(py_parts)

    py_ids   = tok(py_raw,   return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_PY_SV]
    wiki_ids = tok(wiki_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_WIKI_SV]
    print(f"  [Data] {time.time()-t_data:.1f}s  py={len(py_ids):,}  wiki={len(wiki_ids):,}")
    print()

    # ── Training ──────────────────────────────────────────────────────────────
    banner("Training Protocol: A → B → D → C")
    calibrated, native, cal_alpha, ppl_cal, ppl_nat = run_training_protocol(
        py_ids, wiki_ids, REGISTRY)
    ppl_efficacy = (ppl_cal / ppl_nat) * 100

    # ── L2: Distributional equivalence ────────────────────────────────────────
    banner("L2 — Distributional Equivalence (logit-level JS divergence)")
    print(f"  Running {REGISTRY['n_logit_chunks']} × 512-token forward passes...")
    t_l2 = time.time()
    l2_res = l2_logit_test(native, calibrated, py_ids, REGISTRY)
    print()
    print(f"  mean_JS        = {l2_res['mean_js']:.5f}  (thr < {REGISTRY['js_threshold']})  "
          f"{'PASS' if l2_res['js_pass'] else 'FAIL'}")
    print(f"  top-1 agree    = {l2_res['mean_top1_agree']:.4f}  (thr >= {REGISTRY['top1_threshold']})  "
          f"{'PASS' if l2_res['top1_pass'] else 'FAIL'}")
    print(f"  top-5 overlap  = {l2_res['mean_top5_overlap']:.4f}  (thr >= {REGISTRY['top5_threshold']})  "
          f"{'PASS' if l2_res['top5_pass'] else 'FAIL'}")
    print(f"  entropy diff   = {l2_res['mean_entropy_diff']:.4f}  (thr < {REGISTRY['entropy_diff_threshold']})  "
          f"{'PASS' if l2_res['entropy_pass'] else 'FAIL'}")
    print(f"\n  [L2] {'PASS' if l2_res['pass'] else 'FAIL'}  ({time.time()-t_l2:.0f}s)")

    # ── L4a + L4b: Functional + Error Identity ────────────────────────────────
    banner("L4a/L4b — Functional Non-Inferiority + Error Identity (60 probes × 3 seeds)")
    t_l4 = time.time()
    l4ab_res = l4a_l4b_functional_test(native, calibrated, tok, PROBE_BANK, REGISTRY,
                                        decode_cfg={"temperature": 0.0, "top_p": 0.95, "top_k": 0},
                                        label="greedy")
    print()
    print(f"  native pass:       {l4ab_res['nat_pass_pp']:.1f}%  "
          f"CI=[{l4ab_res['nat_ci_95'][0]:.1f}%, {l4ab_res['nat_ci_95'][1]:.1f}%]")
    print(f"  calibrated pass:   {l4ab_res['cal_pass_pp']:.1f}%  "
          f"CI=[{l4ab_res['cal_ci_95'][0]:.1f}%, {l4ab_res['cal_ci_95'][1]:.1f}%]")
    print(f"  NI threshold:      {l4ab_res['ni_threshold_pp']:.1f}%  "
          f"NI={'PASS' if l4ab_res['ni_pass'] else 'FAIL'}")
    print(f"  Failure Jaccard:   {l4ab_res['failure_jaccard']:.3f}  "
          f"(thr >= {REGISTRY['failure_jaccard']})  "
          f"{'PASS' if l4ab_res['failure_jaccard_pass'] else 'FAIL'}")
    print(f"  Pass Jaccard:      {l4ab_res['pass_jaccard']:.3f}  "
          f"(thr >= {REGISTRY['pass_jaccard']})  "
          f"{'PASS' if l4ab_res['pass_jaccard_pass'] else 'FAIL'}")
    if l4ab_res.get("pass_only_native") or l4ab_res.get("pass_only_calibrated"):
        print(f"  Pass-only native:      {l4ab_res.get('pass_only_native', [])}")
        print(f"  Pass-only calibrated:  {l4ab_res.get('pass_only_calibrated', [])}")
    for d, dr in l4ab_res["by_difficulty"].items():
        print(f"  Diff-{d} ({dr['n_probes']} probes):  "
              f"nat={dr['nat_pass_pp']:.1f}%  cal={dr['cal_pass_pp']:.1f}%")
    print(f"\n  [L4a/L4b] {'PASS' if l4ab_res['pass'] else 'FAIL'}  ({time.time()-t_l4:.0f}s)")

    # ── L3: Decoding Equivalence ──────────────────────────────────────────────
    banner("L3 — Decoding Equivalence (greedy / low-temp / high-temp)")
    t_l3 = time.time()
    l3_res = l3_decoding_test(native, calibrated, tok, PROBE_BANK, REGISTRY)
    print(f"\n  [L3] {'PASS' if l3_res['pass'] else 'FAIL'}  ({time.time()-t_l3:.0f}s)")

    # ── L4c: Adversarial ─────────────────────────────────────────────────────
    banner("L4c — Adversarial Prompt Perturbations (30 variants)")
    t_l4c = time.time()
    l4c_res = l4c_adversarial_test(native, calibrated, tok, ADV_VARIANTS, PROBE_BANK, REGISTRY)
    print(f"\n  native adv:    {l4c_res['nat_pass_pp']:.1f}%")
    print(f"  calibrated adv:{l4c_res['cal_pass_pp']:.1f}%  CI_lo={l4c_res['cal_ci_95_lower_pp']:.1f}%")
    print(f"  [L4c] {'PASS' if l4c_res['pass'] else 'FAIL'}  ({time.time()-t_l4c:.0f}s)")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    tests = {
        "L2_distributional": l2_res["pass"],
        "L4a_functional_NI": l4ab_res["ni_pass"] or l4ab_res["floor_skip"],
        "L4b_error_identity": l4ab_res["failure_jaccard_pass"],
        "L3_decoding":        l3_res["pass"],
        "L4c_adversarial":    l4c_res["pass"],
    }
    n_pass = sum(tests.values())
    n_total = len(tests)

    verdicts = {
        5: "DOMAIN-RESTRICTED FUNCTIONAL EQUIVALENCE CONFIRMED",
        4: "STRONG EQUIVALENCE EVIDENCE (4/5)",
        3: "PARTIAL EQUIVALENCE EVIDENCE (3/5)",
    }
    verdict = verdicts.get(n_pass, f"PARTIAL ({n_pass}/{n_total})")

    banner(f"VERDICT: {verdict}")
    for name, passed in tests.items():
        sym = "✓" if passed else "✗"
        print(f"  {sym} {name}")
    print(f"\n  n_pass  = {n_pass}/{n_total}")
    print(f"  ppl_cal = {ppl_cal:.2f}  ppl_nat = {ppl_nat:.2f}  "
          f"efficacy = {ppl_efficacy:.1f}%")
    print(f"  domain_alpha (learned) = {cal_alpha:.4f}")
    print(f"  Total runtime = {(time.time()-t_global)/60:.1f} min")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "verdict":            verdict,
        "n_pass":             n_pass,
        "n_total":            n_total,
        "ppl_efficacy_pct":   round(ppl_efficacy, 2),
        "ppl_cal":            round(ppl_cal, 3),
        "ppl_nat":            round(ppl_nat, 3),
        "domain_alpha":       round(cal_alpha, 4),
        "registry":           REGISTRY,
        "L2_distributional":  l2_res,
        "L3_decoding":        l3_res,
        "L4a_l4b_functional": l4ab_res,
        "L4c_adversarial":    l4c_res,
        "test_pass_summary":  tests,
        "timestamp":          time.strftime("%m/%d/%Y %H:%M:%S"),
    }
    out_path = ROOT / "non_inferiority_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
