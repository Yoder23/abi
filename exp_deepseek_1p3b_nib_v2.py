#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek follow-up variants for the cross-lineage ABI/NIB test.

This runner intentionally does not overwrite exp_deepseek_1p3b_nib.py or its
failed D_ABI=256 result. It imports the baseline implementation and changes only
explicitly labeled follow-up knobs:

  ABI_D_ABI       default 512
  ABI_CAL_STEPS   default 2400
  ABI_CAL_MODE    freeze_domain_net | train_domain | freeze_all_domain

The default mode, freeze_domain_net, freezes the rotated domain MLP core during
target calibration. It trains the target interface plus the domain LayerNorm and
scalar gate, which makes the run stricter than the legacy scripts while still
allowing target-side normalization/scale calibration.
"""

import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import exp_deepseek_1p3b_nib as base


def env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


D_ABI = int(os.environ.get("ABI_D_ABI", "512"))
CALIBRATION_STEPS = int(os.environ.get("ABI_CAL_STEPS", "2400"))
CAL_MODE = os.environ.get("ABI_CAL_MODE", "freeze_domain_net").strip().lower()
TRAIN_DOMAIN_ALPHA = env_bool("ABI_TRAIN_DOMAIN_ALPHA", True)
V2_CORPUS_EXCLUDES = (
    "exp_*_nib_v2.py",
    "tests/test_model_agnostic_followups.py",
)

if CAL_MODE not in {"freeze_domain_net", "train_domain", "freeze_all_domain"}:
    raise ValueError(
        "ABI_CAL_MODE must be one of: freeze_domain_net, train_domain, freeze_all_domain"
    )

if CAL_MODE == "freeze_all_domain":
    TRAIN_DOMAIN_ALPHA = False

TAG = os.environ.get(
    "ABI_EXPERIMENT_TAG",
    f"d{D_ABI}_cal{CALIBRATION_STEPS}_{CAL_MODE}"
    + ("_alpha" if TRAIN_DOMAIN_ALPHA else "_frozen_alpha"),
)

# Baseline module functions/classes read these globals at runtime.
base.D_ABI = D_ABI
base.REGISTRY = dict(base.REGISTRY)
base.REGISTRY["calibration_steps"] = CALIBRATION_STEPS


def banner(msg):
    print()
    print("=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def trainable_count(params):
    return int(sum(p.numel() for p in params))


def add_group(name, params, cal_params, groups):
    params = list(params)
    for p in params:
        p.requires_grad_(True)
    cal_params.extend(params)
    groups.append({"name": name, "params": trainable_count(params)})


def main():
    t_global = time.time()
    banner(f"DeepSeek ABI/NIB v2 follow-up: {TAG}")
    print(f"  Device:          {base.DEVICE}")
    print(f"  D_ABI:           {D_ABI}")
    print(f"  Calibration:     {CALIBRATION_STEPS} steps")
    print(f"  Cal mode:        {CAL_MODE}")
    print(f"  Train alpha:     {TRAIN_DOMAIN_ALPHA}")
    print("  Baseline record: exp_deepseek_1p3b_nib_results.json (D_ABI=256 FAIL)")

    banner("Data loading")
    t_data = time.time()
    tok_src = base.GPT2TokenizerFast.from_pretrained(
        base.HF_GPT2_MEDIUM, local_files_only=True
    )
    tok_src.pad_token = tok_src.eos_token
    tok_src.model_max_length = base.sys.maxsize

    tok_tgt = base.AutoTokenizer.from_pretrained(
        base.HF_DEEPSEEK, local_files_only=True
    )
    tok_tgt.pad_token = tok_tgt.eos_token

    py_text, py_meta = base.load_local_python_text(
        base.ROOT, base.MAX_PY, exclude_globs=V2_CORPUS_EXCLUDES
    )
    py_ids_src = tok_src(
        py_text, return_tensors="pt", truncation=False
    )["input_ids"].squeeze(0)[: base.MAX_PY]
    py_ids_tgt = tok_tgt(
        py_text, return_tensors="pt", truncation=False
    )["input_ids"].squeeze(0)[: base.MAX_PY]

    _, wiki_sentences, wiki_meta = base.load_wikitext_text_and_sentences(
        split="validation", min_chars=20
    )

    print(
        f"  {time.time() - t_data:.1f}s  "
        f"py_src={len(py_ids_src):,}  py_tgt={len(py_ids_tgt):,}  "
        f"sentences={len(wiki_sentences):,}"
    )
    print(
        f"  corpus: local_python_files={py_meta['files']}  "
        f"skipped={py_meta['skipped']}  "
        f"wikitext_split={wiki_meta['split']} records={wiki_meta['records']}"
    )

    kd_weight = base.REGISTRY["kd_weight"]
    kd_temp = base.REGISTRY["kd_temp"]

    banner("Phase A - GPT-2-medium ABI source domain training")
    t_a = time.time()
    src_model = base.GPT2MedABI().to(base.DEVICE)
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
    for step in range(base.DOMAIN_STEPS):
        x, y = base.make_batch(py_ids_src, seed=5000 + step)
        opt_a.zero_grad()
        loss = F.cross_entropy(src_model(x).reshape(-1, base.VOCAB_SRC), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(src_model.parameters(), 1.0)
        opt_a.step()
        if (step + 1) % 100 == 0:
            print(f"  A step {step + 1}/{base.DOMAIN_STEPS}  {time.time() - t_a:.0f}s")
    src_model.eval()
    for p in src_model.parameters():
        p.requires_grad_(False)
    ppl_a = base.ppl(src_model, py_ids_src)
    print(f"  Phase A complete: {time.time() - t_a:.0f}s  GPT-2-med ppl={ppl_a:.1f}")

    banner("Phase C - Native DeepSeek ABI oracle")
    t_c = time.time()
    native = base.DeepSeekCoder1p3BABI().to(base.DEVICE)
    for p in native.parameters():
        p.requires_grad_(False)
    for nm, p in native.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "proj_out", "domain")):
            p.requires_grad_(True)
    opt_c = torch.optim.AdamW(
        [p for p in native.parameters() if p.requires_grad],
        lr=base.LR_ABI,
        weight_decay=0.01,
    )
    native.train()
    for step in range(base.DOMAIN_STEPS):
        x, y = base.make_batch(py_ids_tgt, seed=5000 + step)
        opt_c.zero_grad()
        loss = F.cross_entropy(native(x).reshape(-1, base.VOCAB_TGT), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(native.parameters(), 1.0)
        opt_c.step()
        if (step + 1) % 100 == 0:
            print(f"  C step {step + 1}/{base.DOMAIN_STEPS}  {time.time() - t_c:.0f}s")
    native.eval()
    for p in native.parameters():
        p.requires_grad_(False)
    ppl_nat = base.ppl(native, py_ids_tgt)
    print(f"  Phase C complete: {time.time() - t_c:.0f}s  DeepSeek native ppl={ppl_nat:.1f}")

    banner("Phase D - Procrustes + KD calibration")
    t_d = time.time()
    R = base.cross_lineage_procrustes(src_model, native, wiki_sentences, tok_src, tok_tgt)
    rotated_domain = base.apply_rotation_to_domain(src_model.domain, R)

    calibrated = base.DeepSeekCoder1p3BABI.__new__(base.DeepSeekCoder1p3BABI)
    nn.Module.__init__(calibrated)
    calibrated.backbone = native.backbone
    calibrated.lm_head = native.lm_head
    calibrated.proj_in = nn.Linear(base.D_MODEL_TGT, D_ABI, bias=False).to(base.DEVICE)
    calibrated.abi_ln = nn.LayerNorm(D_ABI).to(base.DEVICE)
    calibrated.proj_out = nn.Linear(D_ABI, base.D_MODEL_TGT, bias=False).to(base.DEVICE)
    calibrated.domain = rotated_domain.to(base.DEVICE)
    calibrated.domain_alpha = nn.Parameter(
        src_model.domain_alpha.detach().clone().to(base.DEVICE)
    )
    nn.init.xavier_uniform_(calibrated.proj_in.weight)
    nn.init.xavier_uniform_(calibrated.proj_out.weight)
    calibrated.encode_core = native.encode_core.__func__.__get__(
        calibrated, base.DeepSeekCoder1p3BABI
    )
    calibrated.forward = native.forward.__func__.__get__(
        calibrated, base.DeepSeekCoder1p3BABI
    )

    for p in calibrated.parameters():
        p.requires_grad_(False)

    cal_params = []
    trainable_groups = []
    add_group("proj_in", calibrated.proj_in.parameters(), cal_params, trainable_groups)
    add_group("abi_ln", calibrated.abi_ln.parameters(), cal_params, trainable_groups)
    add_group("proj_out", calibrated.proj_out.parameters(), cal_params, trainable_groups)

    if CAL_MODE == "train_domain":
        add_group("domain_net", calibrated.domain.net.parameters(), cal_params, trainable_groups)
        add_group("domain_ln", calibrated.domain.ln.parameters(), cal_params, trainable_groups)
    elif CAL_MODE == "freeze_domain_net":
        add_group("domain_ln", calibrated.domain.ln.parameters(), cal_params, trainable_groups)

    if TRAIN_DOMAIN_ALPHA:
        add_group("domain_alpha", [calibrated.domain_alpha], cal_params, trainable_groups)

    print("  Trainable groups:")
    for group in trainable_groups:
        print(f"    {group['name']}: {group['params']:,}")
    print(f"  Total trainable during D: {trainable_count(cal_params):,}")

    opt_d = torch.optim.AdamW(cal_params, lr=base.LR_CAL, weight_decay=0.01)
    native.eval()
    calibrated.train()
    for step in range(CALIBRATION_STEPS):
        x, y = base.make_batch(py_ids_tgt, seed=7000 + step)
        opt_d.zero_grad()
        cal_logits = calibrated(x)
        with torch.no_grad():
            nat_logits = native(x)
        ce = F.cross_entropy(cal_logits.reshape(-1, base.VOCAB_TGT), y.reshape(-1))
        kd = F.kl_div(
            F.log_softmax(cal_logits.reshape(-1, base.VOCAB_TGT) / kd_temp, dim=-1),
            F.softmax(nat_logits.reshape(-1, base.VOCAB_TGT) / kd_temp, dim=-1),
            reduction="batchmean",
        ) * (kd_temp ** 2)
        (kd_weight * kd + (1 - kd_weight) * ce).backward()
        nn.utils.clip_grad_norm_(cal_params, 1.0)
        opt_d.step()
        if (step + 1) % 300 == 0:
            print(f"  D step {step + 1}/{CALIBRATION_STEPS}  {time.time() - t_d:.0f}s")

    calibrated.eval()
    for p in cal_params:
        p.requires_grad_(False)
    ppl_cal = base.ppl(calibrated, py_ids_tgt)
    print(f"  Phase D complete: {time.time() - t_d:.0f}s  DeepSeek calibrated ppl={ppl_cal:.1f}")

    banner("NIB L2 evaluation")
    t_nib = time.time()
    l2 = base.l2_logit_test(native, calibrated, py_ids_tgt)
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
    print(f"  NIB overall: {status_str}  ({time.time() - t_nib:.1f}s)")

    elapsed = time.time() - t_global
    results = {
        "experiment": "cross_lineage_deepseek_coder_1p3b_v2",
        "name": f"exp_deepseek_1p3b_nib_v2_{TAG}",
        "baseline_result": "exp_deepseek_1p3b_nib_results.json",
        "variant_type": "post-baseline follow-up",
        "corpus_exclude_globs": list(V2_CORPUS_EXCLUDES),
        "corpus_skipped_files": py_meta["skipped"],
        "source_model": "gpt2-medium-354M",
        "target_model": "deepseek-coder-1.3b-base-1300M",
        "source_vocab": base.VOCAB_SRC,
        "target_vocab": base.VOCAB_TGT,
        "source_d_model": base.D_MODEL_SRC,
        "target_d_model": base.D_MODEL_TGT,
        "source_arch": "GPT-2 (abs-pos, MHA, GELU, LayerNorm)",
        "target_arch": "Llama (RoPE, GQA, SwiGLU, RMSNorm)",
        "d_abi": D_ABI,
        "d_abi_note": "follow-up capacity variant; baseline D_ABI=256 failed top-5",
        "seed": base.SEED,
        "domain_steps": base.DOMAIN_STEPS,
        "calibration_steps": CALIBRATION_STEPS,
        "calibration_mode": CAL_MODE,
        "train_domain_alpha": TRAIN_DOMAIN_ALPHA,
        "calibration_trainable_groups": trainable_groups,
        "calibration_trainable_params": trainable_count(cal_params),
        "n_align_sentences": base.REGISTRY["n_align_sentences"],
        "alignment_method": "sentence-level mean-pool Procrustes (cross-tokenizer)",
        "ppl_source_gpt2_medium": round(ppl_a, 3),
        "ppl_native_deepseek": round(ppl_nat, 3),
        "ppl_calibrated_deepseek": round(ppl_cal, 3),
        "nib_l2": l2,
        "overall_pass": overall,
        "elapsed_min": round(elapsed / 60, 1),
        "thresholds": base.REGISTRY,
        "interpretation": (
            "This is a labeled follow-up after the D_ABI=256 DeepSeek baseline failed "
            "the top-5 NIB threshold. It tests whether Llama-family transfer requires "
            "a wider ABI and whether the rotated domain MLP core can remain frozen "
            "during target calibration."
        ),
    }
    out_path = base.ROOT / f"exp_deepseek_1p3b_nib_v2_{TAG}_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results -> {out_path}")
    banner(f"Done - {status_str} - {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
