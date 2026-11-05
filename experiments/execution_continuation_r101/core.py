"""Frozen R101 selection, evidence validation and diagnostic statistics."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

from experiments.acquisition_capacity_r100 import core as old
from experiments.grounded_english_r99 import fit_audit

ROOT, HOST, PARENT = old.ROOT, old.HOST, old.PARENT
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/execution_continuation_r101"
ARMS = ("full_trunk", "lora", "distillation", "parent")
CUTS = (0, 25, 50)
SEED, PARITY_TOKENS, CONTEXT, MAX_NEW = 101001, 8, 1024, 640
sha, read, rows, write_once, append = old.sha, old.read, old.rows, old.write_once, old.append


def catalog():
    return read(fit_audit.OUT / "training_selection.json") + read(old.OUT / "development.json")


def parity_ids(population):
    return [
        min(r["id"] for r in population if r["task"] == task and r["split"] == split)
        for split in ("train", "development") for task in old.TASKS
    ]


def binding():
    frozen = read(OUT / "freeze.json")
    if not frozen.get("files"):
        raise RuntimeError("Missing R101 bindings")
    for relative, digest in frozen["files"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f"Changed R101 input: {relative}")
    return frozen


def literal_present(literal, text):
    return bool(re.search(r"\b" + re.escape(literal) + r"\b", text, re.I))


def eligible_literals(row, target, offsets, cut):
    boundary = offsets[cut - 1][1] if cut else 0
    eligible = []
    for literal in row["required"]:
        matches = list(re.finditer(r"\b" + re.escape(literal) + r"\b", target, re.I))
        if matches and min(m.start() for m in matches) >= boundary:
            eligible.append(literal)
    return eligible


def paired_interval(deltas):
    if not deltas:
        return {"rows": 0, "difference": None, "ci95": None}
    rng = random.Random(SEED)
    boot = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(10000))
    return {"rows": len(deltas), "difference": sum(deltas) / len(deltas),
            "ci95": [boot[249], boot[9749]]}


def validate_generation(record, ref, target, tokenizer, cut_percent):
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    cut = len(encoded.input_ids) * cut_percent // 100
    prefix = encoded.input_ids[:cut]
    generated = record["generated_ids"]
    if not isinstance(generated, list) or any(
        type(t) is not int or not 0 <= t < len(tokenizer) for t in generated
    ):
        raise RuntimeError("Invalid generated token IDs")
    if tokenizer.eos_token_id in generated:
        raise RuntimeError("EOS must be a stop, not a visible output token")
    output = tokenizer.decode(prefix + generated, skip_special_tokens=True)
    tail = tokenizer.decode(generated, skip_special_tokens=True)
    prompt = tokenizer.encode(old.model_prompt(ref, "full_trunk"), add_special_tokens=False)
    budget = min(MAX_NEW, CONTEXT - len(prompt) - 1) - cut
    if budget < 0 or len(generated) > budget:
        raise RuntimeError("Generation budget violation")
    stop = record["stop_reason"]
    if stop not in ("eos", "budget") or (stop == "budget" and len(generated) != budget):
        raise RuntimeError("Invalid stop evidence")
    expected = {
        "id": ref["id"], "task": ref["task"], "split": ref["split"],
        "cut_percent": cut_percent, "supplied_ids": prefix, "output": output, "tail": tail,
        "eligible_literals": eligible_literals(ref, target, encoded.offset_mapping, cut),
        "screen": old.screen(ref, output),
    }
    if any(record[k] != v for k, v in expected.items()):
        raise RuntimeError("Unrecomputable generation record")
    return expected


def validate_forced(record, ref, target, tokenizer):
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    suffix = encoded.input_ids + [tokenizer.eos_token_id]
    masks = fit_audit.token_masks(ref, target, encoded.offset_mapping, "full_trunk")
    if (record["id"] != ref["id"] or record["task"] != ref["task"]
            or record["split"] != ref["split"] or record["target_ids"] != suffix
            or (record["response_mask"], record["fact_mask"]) != masks):
        raise RuntimeError("Changed forced targets/masks")
    if len(record["argmax_ids"]) != len(suffix) or len(record["nll"]) != len(suffix):
        raise RuntimeError("Missing forced token observations")
    if any(type(t) is not int or not 0 <= t < len(tokenizer) for t in record["argmax_ids"]):
        raise RuntimeError("Invalid forced argmax IDs")
    if any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in record["nll"]):
        raise RuntimeError("Invalid forced NLL")


def validate_coverage(forced, generations, population):
    if ([r["id"] for r in forced] != [r["id"] for r in population]
            or [(r["id"], r["cut_percent"]) for r in generations]
            != [(r["id"], cut) for r in population for cut in CUTS]):
        raise RuntimeError("Missing, duplicate or reordered raw rows")


def parity_metrics(tensors, ids, counts, vocab):
    import torch

    expected = {f"{key}.{mode}" for key in ids for mode in ("full", "padded", "cached")}
    if set(tensors) != expected:
        raise RuntimeError("Missing or extra parity tensors")
    result = {}
    for mode in ("padded", "cached"):
        peak, violations, disagreements, elements = 0.0, 0, 0, 0
        for key in ids:
            reference, other = tensors[f"{key}.full"], tensors[f"{key}.{mode}"]
            if (reference.shape != (counts[key], vocab) or other.shape != reference.shape
                    or reference.dtype != torch.float32 or other.dtype != torch.float32
                    or not torch.isfinite(reference).all() or not torch.isfinite(other).all()):
                raise RuntimeError("Invalid parity tensors")
            error = (other - reference).abs()
            peak = max(peak, error.max().item())
            violations += int((error > 1e-3 + 1e-4 * reference.abs()).sum())
            disagreements += int((other.argmax(-1) != reference.argmax(-1)).sum())
            elements += reference.numel()
        result[mode] = {"max_abs_error": peak, "tolerance_violations": violations,
                        "argmax_disagreements": disagreements, "elements": elements}
    return result


def summarize(forced, generations):
    result = {}
    for split in ("train", "development"):
        selected = [r for r in generations if r["split"] == split]
        zero = {r["id"]: r for r in selected if r["cut_percent"] == 0}
        metrics = {}
        for field in ("response_mask", "fact_mask"):
            pairs, nlls = [], []
            for r in forced:
                if r["split"] != split:
                    continue
                for wanted, gold, argmax, nll in zip(r[field], r["target_ids"], r["argmax_ids"], r["nll"]):
                    if wanted:
                        pairs.append(gold == argmax)
                        nlls.append(nll)
            if not pairs:
                raise RuntimeError("No conditional observations")
            metrics[field] = {"tokens": len(pairs), "correct": sum(pairs),
                              "accuracy": sum(pairs) / len(pairs), "mean_nll": sum(nlls) / len(nlls)}
        interventions = {}
        for cut in CUTS:
            chosen = [r for r in selected if r["cut_percent"] == cut]
            deltas, supplied, baseline, assisted = [], 0, 0, 0
            for r in chosen:
                supplied += len(r["supplied_ids"])
                if not r["eligible_literals"]:
                    continue
                before = all(literal_present(t, zero[r["id"]]["output"]) for t in r["eligible_literals"])
                after = all(literal_present(t, r["tail"]) for t in r["eligible_literals"])
                baseline += before
                assisted += after
                deltas.append(int(after) - int(before))
            interventions[str(cut)] = {
                "joined_response_screen": old.aggregate(chosen),
                "oracle_supplied_tokens": supplied, "eligible_rows": len(deltas),
                "excluded_no_eligible_literal": len(chosen) - len(deltas),
                "unassisted_remaining_coverage": baseline, "tail_remaining_coverage": assisted,
                "paired_remaining_coverage": paired_interval(deltas),
            }
        result[split] = {"conditional": metrics, "interventions": interventions}
    return result
