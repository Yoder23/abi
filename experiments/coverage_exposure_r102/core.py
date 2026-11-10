"""Locked corpus transformations, schedules and paired diagnostic statistics."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from pathlib import Path

from experiments.acquisition_capacity_r100 import core as previous

ROOT, HOST, PARENT = previous.ROOT, previous.HOST, previous.PARENT
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/coverage_exposure_r102"
TASKS = previous.TASKS
METHODS, POLICIES, ENDPOINTS = ("lora", "distillation"), ("repeat", "augment"), (384, 1536)
SEED, BATCH, CONTEXT, MAX_NEW = 99001, 2, 1024, 640
STRATA = ("same_shape", "shifted_shape")
SYSTEM = "Follow the user's English-language request. Use only the supplied fictional information. Return only the requested response, without commentary about the task."
sha, read, rows, write_once, append = previous.sha, previous.read, previous.rows, previous.write_once, previous.append
model_prompt, screen, aggregate = previous.model_prompt, previous.screen, previous.aggregate


def replace_literals(text, mapping):
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")\b"
    return re.sub(pattern, lambda match: mapping[match.group()], text)


def transform(row, mapping):
    result = copy.deepcopy(row)
    for field in ("prompt", "code"):
        result[field] = replace_literals(row[field], mapping)
    result["required"] = [replace_literals(x, mapping) for x in row["required"]]
    result["plan"]["facts"] = [replace_literals(x, mapping) for x in row["plan"]["facts"]]
    result["plan"]["request"] = replace_literals(row["plan"]["request"], mapping)
    return result


def entities(row):
    return sorted(set(re.findall(r"\b(?:NOTE|Person)[A-Za-z0-9]+\b", row["prompt"])))


def development_catalog():
    result = []
    for stratum in STRATA:
        for index, original in enumerate(previous.development_catalog()):
            number = 7000 + index
            mapping = {}
            for literal in entities(original):
                kind = "NOTE" if literal.startswith("NOTE") else "Person"
                stem = f"T{number}" if stratum == "same_shape" else f"N102V{number}"
                mapping[literal] = kind + stem + (literal[-1] if kind == "Person" else "")
            row = transform(original, mapping)
            row["id"] = f"r102-{stratum}-{index:03d}-{row['task']}"
            row["stratum"] = stratum
            result.append(row)
    random.Random(102002).shuffle(result)
    return result


def alias_mapping(row, rng):
    code_tag = rng.choice("ABCDEFHJKLMNPQRSUVWXYZ")
    person_tag = rng.choice("ABCDEFHJKLMNPQRSUVWXYZ")
    code_number, person_number = rng.randrange(1000, 7000), rng.randrange(1000, 7000)
    return {
        text: (f"NOTE{code_tag}{code_number}" if text.startswith("NOTE")
               else f"Person{person_tag}{person_number}{text[-1]}")
        for text in entities(row)
    }


def construct_alias(row, answer, alias, tokenizer, occupied):
    seed = int.from_bytes(hashlib.sha256(f"r102:{row['id']}:{alias}".encode()).digest()[:8], "big")
    rng = random.Random(seed)
    prompt_count = len(tokenizer.encode(model_prompt(row, "distillation"), add_special_tokens=False))
    target_count = len(tokenizer.encode(answer, add_special_tokens=False))
    original_screen = screen(row, answer)
    for attempt in range(4096):
        mapping = alias_mapping(row, rng)
        if len(set(mapping.values())) != len(mapping) or set(mapping.values()) & occupied:
            continue
        derived = transform(row, mapping)
        target = replace_literals(answer, mapping)
        if (len(tokenizer.encode(model_prompt(derived, "distillation"), add_special_tokens=False)) != prompt_count
                or len(tokenizer.encode(target, add_special_tokens=False)) != target_count
                or screen(derived, target) != original_screen):
            continue
        inverse = {v: k for k, v in mapping.items()}
        if transform(derived, inverse) != row or replace_literals(target, inverse) != answer:
            raise RuntimeError("Alias is not exactly invertible")
        occupied.update(mapping.values())
        return {"original_id": row["id"], "alias": alias, "mapping": mapping,
                "attempt": attempt, "row": derived, "target": target,
                "prompt_tokens": prompt_count, "target_tokens": target_count}
    raise RuntimeError(f"No token-matched alias within bound: {row['id']} / {alias}")


def training_schedule():
    rng = random.Random(SEED)
    order = list(range(192))
    schedule = []
    for step in range(ENDPOINTS[-1]):
        if step * BATCH % len(order) == 0:
            rng.shuffle(order)
        schedule.append({"step": step + 1, "epoch": step * BATCH // len(order),
                         "indexes": [order[(step * BATCH + j) % len(order)] for j in range(BATCH)]})
    return schedule


def trajectories():
    return [f"{method}_{policy}" for method in METHODS for policy in POLICIES]


def endpoint_names():
    return [f"{trajectory}_s{step}" for trajectory in trajectories() for step in ENDPOINTS]


def source_gate(observations):
    summary = aggregate(observations)
    return (len(observations) == 288 and summary["passing"] >= 260 and summary["collapses"] == 0
            and all(r["rows"] == 36 and r["passing"] >= 27 for r in summary["per_task"].values())
            and not any(r["truncated"] for r in observations))


def quality_screen(summary, comparison):
    return (summary["rows"] == 48 and summary["passing"] >= 44 and summary["collapses"] == 0
            and set(summary["per_task"]) == set(TASKS)
            and sum(r["rows"] for r in summary["per_task"].values()) == summary["rows"]
            and sum(r["passing"] for r in summary["per_task"].values()) == summary["passing"]
            and all(r["rows"] == 6 and r["passing"] >= 5 for r in summary["per_task"].values())
            and comparison["ci95"][0] >= -.10)


def paired(a, b):
    if not a or [r["id"] for r in a] != [r["id"] for r in b]:
        raise RuntimeError("Unpaired comparison")
    deltas = [int(x["screen"]["mechanical_pass"]) - int(y["screen"]["mechanical_pass"]) for x, y in zip(a, b)]
    clusters = {}
    for row, delta in zip(a, deltas):
        match = re.fullmatch(r"r102-(?:same_shape|shifted_shape)-(\d{3})-.+", row["id"])
        key = match.group(1) if match else row["id"]
        clusters.setdefault(key, []).append(delta)
    if len({len(values) for values in clusters.values()}) != 1:
        raise RuntimeError("Unbalanced paired clusters")
    cluster_deltas = [sum(values)/len(values) for values in clusters.values()]
    rng = random.Random(102001)
    boot = sorted(sum(rng.choice(cluster_deltas) for _ in cluster_deltas) / len(cluster_deltas) for _ in range(10000))
    return {"rows": len(deltas), "clusters": len(clusters), "deltas": deltas, "difference": sum(deltas)/len(deltas),
            "ci95": [boot[249], boot[9749]], "ci97_5": [boot[124], boot[9874]]}


def validate_generation(observed, catalog, tokenizer):
    if [r["id"] for r in observed] != [r["id"] for r in catalog]:
        raise RuntimeError("Missing/duplicate/reordered generation rows")
    vocab = len(tokenizer)
    for row, reference in zip(observed, catalog):
        tokens = row["token_ids"]
        if not isinstance(tokens, list) or any(type(t) is not int or not 0 <= t < vocab for t in tokens):
            raise RuntimeError("Invalid generated tokens")
        output = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        budget = min(MAX_NEW, CONTEXT - len(tokenizer.encode(model_prompt(reference, "distillation"), add_special_tokens=False)) - 1)
        if len(tokens) > budget or tokenizer.eos_token_id in tokens:
            raise RuntimeError("Invalid generation budget/EOS")
        if (row["task"] != reference["task"] or row["output"] != output
                or row["screen"] != screen(reference, output)):
            raise RuntimeError("Unrecomputable generation screen")
        if row["stop_reason"] not in ("eos", "budget") or (row["stop_reason"] == "budget" and len(tokens) != budget):
            raise RuntimeError("Invalid stopping evidence")


def binding():
    frozen = read(OUT / "freeze.json")
    if not frozen.get("files"):
        raise RuntimeError("Missing R102 freeze")
    for relative, digest in frozen["files"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f"Changed or missing R102 input: {relative}")
    return frozen


def finite_log(log):
    if [r["step"] for r in log] != list(range(1, ENDPOINTS[-1] + 1)):
        raise RuntimeError("Incomplete optimization")
    if any(not math.isfinite(r[k]) for r in log for k in ("loss", "gradient_norm")):
        raise RuntimeError("Nonfinite optimizer evidence")
