"""Frozen inputs, literal screens and paired interface contrasts; no learning."""

from __future__ import annotations

import copy
import math
import random
import re
from pathlib import Path

from experiments.plan_english_r98 import core as base

ROOT, HOST, PARENT = base.ROOT, base.HOST, base.PARENT
SOURCE, REVISION = base.SOURCE, base.REVISION
HOST_CHECKOUT = "865a2b9e59e6221caa1185f9b64479d2077953f3"
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/evaluator_control_r104"
ARMS = ("parent", "full_model", "lora", "distillation")
INTERFACES = ("native", "benchmark")
TASKS = base.TASKS
ROUTE_ORDER = ("grammar", "coherence", "prompt_grounding", "instruction_following", "conversation", "summarization",
               "rewriting", "email_drafting", "tone_control", "format_control", "clarification", "abstention",
               "domain_independent_reasoning", "cake_output_realization")
MAX_NEW, CONTEXT, SEED = 640, 1024, 104001
SYSTEM = "Follow the user's English-language request. Use only the supplied fictional information. Return only the requested response, without commentary about the task."
sha, read, rows, write_once, append = base.sha, base.read, base.rows, base.write_once, base.append
aggregate = base.aggregate


def approval_evidence(answer):
    """Narrow clause-local polarity/conditional audit, not semantic entailment.

    Only listed syntactic forms are exempted. A conditional cannot suppress an
    assertion after a comma, conjunction, sentence boundary or newline.
    """
    assertions, exempt = [], []
    pattern = re.compile(r"\b(?:is|are|was|were|has been|have been) (?:fully )?(?:approved|confirmed)\b", re.I)
    for match in pattern.finditer(answer):
        prefix = re.split(r"[.!?;,\n]|\b(?:and|but|however|although)\b", answer[:match.start()], flags=re.I)[-1]
        conditional = re.search(r"\b(?:if|when|once|until|before|after|as soon as)\s+(?:(?:the|our|these|those)\s+)?(?:details|arrangements|plans|project|approval|it|they)\s*$", prefix, re.I)
        (exempt if conditional else assertions).append(match.group())
    # The main pattern does not match 'is not approved'. Include other common
    # explicit positive claims and guaranteed future approval, without ignoring
    # an entire answer merely because it contains 'pending' or a conditional.
    other = re.findall(r"\b(?:approval (?:has arrived|was granted|is granted)|(?:fully|definitely|already) approved|(?:will|shall) (?:definitely )?be (?:approved|confirmed)|(?:approved|confirmed)\s*:\s*yes)\b", answer, re.I)
    assertions.extend(other)
    return {"assertions": assertions, "conditional_exemptions": exempt}


def screen(row, answer):
    legacy = base.screen(row, answer)
    words = re.findall(r"\b[\w']+\b", answer.lower())
    evidence = approval_evidence(answer)
    result = dict(legacy)
    result["explicit_approval_contradiction"] = bool(evidence["assertions"])
    result["approval_evidence"] = evidence
    result["nonempty_bounded"] = 3 <= len(words) <= 120
    days = set(re.findall(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", answer, re.I))
    supplied_days = set(re.findall(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", row["prompt"], re.I))
    result["unsupported_days"] = sorted({d.lower() for d in days} - {d.lower() for d in supplied_days})
    # Explicitly named sentence-count proxy: no claim to general parsing.
    sentences = [s for s in re.split(r"[.!?]+(?:\s+|$)", answer.strip()) if s.strip()]
    result["requested_sentence_count"] = row["task"] not in ("summarization", "tone_control") or 2 <= len(sentences) <= 3
    result["mechanical_pass"] = bool(
        result["required_literal_coverage"] and result["surface_form"]
        and result["nonempty_bounded"] and result["requested_sentence_count"]
        and not result["unknown_identifiers"] and not result["extra_numbers"]
        and not result["unsupported_days"] and not result["collapse"]
        and not result["explicit_approval_contradiction"])
    return result


def checkpoint(arm):
    paths = {"parent": PARENT, "full_model": ROOT / "results/acquisition_capacity_r100/full_trunk",
             **{a: ROOT / "results/grounded_english_r99_fp32" / a for a in ("lora", "distillation")}}
    if arm not in paths:
        raise ValueError("Unregistered recipient")
    return paths[arm]


def catalog(split):
    if split not in ("qualification", "comparison"):
        raise ValueError("Unregistered split")
    rng = random.Random(104002 if split == "qualification" else 104003)
    result = []
    for index, original in enumerate(base.catalog("development")):
        row = copy.deepcopy(original)
        mapping = {v: v.replace("NOTEV", f"NOTEN104{split[0].upper()}").replace("PersonV", f"PersonN104{split[0].upper()}")
                   for v in set(re.findall(r"\b(?:NOTE|Person)[A-Za-z0-9]+\b", row["prompt"]))}
        old_day = next(day for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday") if day in row["prompt"])
        mapping[old_day] = rng.choice([d for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday") if d != old_day])

        def replace(text):
            return re.sub(r"\b(?:" + "|".join(re.escape(v) for v in mapping) + r")\b", lambda m: mapping[m.group()], text)

        row["id"] = f"r104-{split}-{index:03d}-{row['task']}"
        row["split"] = split
        row["code"] = replace(row["code"])
        row["required"] = [replace(v) for v in row["required"]]
        row["plan"]["facts"] = [replace(v) for v in row["plan"]["facts"]]
        rng.shuffle(row["plan"]["facts"])
        row["plan"]["request"] = replace(row["plan"]["request"]) + (
            " Keep the entire response within 120 words. Preserve requested identifiers exactly, including every letter suffix."
            " Include every requested field. Put each bullet on its own line when bullets are requested."
            " Do not add a subject line or commentary about the task.")
        row["prompt"] = "Use only the following fictional supplied notes.\n" + "\n".join("- " + v for v in row["plan"]["facts"]) + "\n\n" + row["plan"]["request"]
        result.append(row)
    rng.shuffle(result)
    return result


def render(row, arm, interface, tokenizer):
    if arm not in (*ARMS, "control") or interface not in INTERFACES:
        raise ValueError("Unregistered cell")
    if interface == "benchmark":
        return base.model_prompt(row, "distillation")
    if arm == "control":
        return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}],
                                             tokenize=False, add_generation_prompt=True)
    return row["prompt"] + "\n"


def qualified(observed):
    value = aggregate(observed)
    return (value["rows"] == 48 and value["passing"] >= 44 and value["collapses"] == 0
            and set(value["per_task"]) == set(TASKS)
            and len({r["id"] for r in observed}) == 48
            and all(r["rows"] == 6 and r["passing"] >= 5 for r in value["per_task"].values())
            and not any(r["stop_reason"] == "budget" for r in observed))


def paired(first, second):
    import numpy as np

    if not first or [r["id"] for r in first] != [r["id"] for r in second] or len({r["id"] for r in first}) != len(first):
        raise RuntimeError("Incomplete or unpaired contrast")
    delta = np.array([int(a["screen"]["mechanical_pass"]) - int(b["screen"]["mechanical_pass"]) for a, b in zip(first, second)])
    rng = np.random.default_rng(SEED)
    draws = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {"rows": len(delta), "difference": float(delta.mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "ci97_5": np.quantile(draws, [.0125, .9875]).tolist()}


def binding():
    frozen = read(OUT / "freeze.json")
    if not frozen.get("files"):
        raise RuntimeError("Missing frozen inputs")
    for relative, digest in frozen["files"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f"Missing or changed frozen input: {relative}")
    return frozen


def validate(observed, population, arm, interfaces, tokenizer):
    expected = [(r, interface) for r in population for interface in interfaces]
    vocab_size = len(tokenizer)
    if len(observed) != len(expected):
        raise RuntimeError("Incomplete generation matrix")
    for actual, (ref, interface) in zip(observed, expected):
        prompt = render(ref, arm, interface, tokenizer)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = actual["token_ids"]
        if (actual["id"] != ref["id"] or actual["task"] != ref["task"] or actual["arm"] != arm
                or actual["interface"] != interface or actual["prompt"] != prompt or actual["prompt_ids"] != prompt_ids
                or actual["route"] != ref["task"] or not tokens or len(tokens) > MAX_NEW
                or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens)
                or len(prompt_ids) + MAX_NEW > CONTEXT):
            raise RuntimeError("Wrong generation lineage or token matrix")
        eos = tokenizer.eos_token_id
        stop = "eos" if tokens[-1] == eos else "budget"
        if eos in tokens[:-1] or actual["stop_reason"] != stop or (stop == "budget" and len(tokens) != MAX_NEW):
            raise RuntimeError("Invalid termination")
        output = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if (actual["output"] != output or actual["screen"] != screen(ref, output)
                or actual["legacy_screen"] != base.screen(ref, output)
                or actual["output_utf8_bytes"] != len(output.encode()) or actual["output_characters"] != len(output)
                or not math.isfinite(actual["wall_seconds"]) or actual["wall_seconds"] < 0):
            raise RuntimeError("Unrecomputable text, screen or accounting")
        if arm != "control":
            route = ROUTE_ORDER.index(ref["task"])
            expected_trace = {"cakes": [route], "adapters": [[route]] * 6}
            if actual["execution_routes"] != [expected_trace] * len(tokens):
                raise RuntimeError("Incomplete or wrong execution-route observations")


def summarize(observed):
    return {"summary": aggregate(observed), "mechanical_screen_pass": qualified(observed),
            "legacy_summary_descriptive_only": aggregate(observed, "legacy_screen"),
            "input_tokens": sum(len(r["prompt_ids"]) for r in observed),
            "output_tokens": sum(len(r["token_ids"]) for r in observed),
            "output_utf8_bytes": sum(r["output_utf8_bytes"] for r in observed),
            "unique_prompt_utf8_bytes": sum(len(p.encode()) for p in {r["prompt"] for r in observed}),
            "unique_output_utf8_bytes": sum(len(p.encode()) for p in {r["output"] for r in observed}),
            "generation_wall_seconds": sum(r["wall_seconds"] for r in observed),
            "truncations": sum(r["stop_reason"] == "budget" for r in observed),
            "overlapping_failure_counts": {
                **{k: sum(not r["screen"][k] for r in observed) for k in ("required_literal_coverage", "surface_form", "nonempty_bounded")},
                **{k: sum(bool(r["screen"][k]) for r in observed) for k in ("unknown_identifiers", "extra_numbers", "explicit_approval_contradiction", "collapse")},
            }}
