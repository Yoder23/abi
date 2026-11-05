"""Frozen scope intervention and fresh evaluation for one full-model control."""

import copy
import random
import re
from pathlib import Path

from experiments.grounded_english_r99 import core as r99
from experiments.grounded_english_r99_fp32 import core as baseline

ROOT, PARENT, HOST = r99.ROOT, r99.PARENT, r99.HOST
PARENT_SHA, HOST_COMMIT = r99.PARENT_SHA, r99.HOST_COMMIT
SOURCE, REVISION = r99.SOURCE, r99.REVISION
TASKS, SEED = r99.TASKS, r99.SEED
STEPS, BATCH, CONTEXT, MAX_NEW = r99.STEPS, r99.BATCH, r99.CONTEXT, r99.MAX_NEW
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/acquisition_capacity_r100"
ARMS = ("full_trunk", "distillation", "lora", "abi_plan", "parent")
previous = r99.previous
sha, canonical, write_once = r99.sha, r99.canonical, r99.write_once
read, rows, append = r99.read, r99.rows, r99.append
model_prompt, target = r99.model_prompt, r99.target
visible_response, screen = r99.visible_response, r99.screen
aggregate, source_gate, normalize = r99.aggregate, r99.source_gate, r99.normalize


def development_catalog():
    rng = random.Random(100002)
    records = []
    days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    for original in r99.development_catalog():
        row = copy.deepcopy(original)
        replacements = {
            text: text.replace("N99V", "N100V")
            for text in set(re.findall(r"\b(?:NOTE|Person)[A-Za-z0-9]+\b", row["prompt"]))
        }
        day = next(day for day in days if day in row["prompt"])
        replacements[day] = rng.choice([d for d in days if d != day])

        def replace(text):
            for before, after in replacements.items():
                text = text.replace(before, after)
            return text

        row["id"] = row["id"].replace("r99-", "r100-")
        row["plan"]["facts"] = [replace(f) for f in row["plan"]["facts"]]
        rng.shuffle(row["plan"]["facts"])
        row["plan"]["request"] = replace(row["plan"]["request"])
        row["prompt"] = (
            "Use only the following fictional supplied notes.\n"
            + "\n".join("- " + fact for fact in row["plan"]["facts"])
            + "\n\n"
            + row["plan"]["request"]
        )
        row["required"] = [replace(value) for value in row["required"]]
        row["code"] = replace(row["code"])
        records.append(row)
    rng.shuffle(records)
    return records


def checkpoint(arm):
    if arm not in ARMS:
        raise ValueError("Unregistered control")
    if arm == "parent":
        return PARENT
    return (OUT if arm == "full_trunk" else baseline.OUT) / arm


def verify_binding():
    manifest = read(OUT / "freeze.json")
    if not manifest["files"]:
        raise RuntimeError("Missing capacity-control freeze")
    for relative, digest in manifest["files"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f"Capacity-control input changed: {relative}")
    baseline.verify_binding()
    return manifest
