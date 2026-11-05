"""Additive configuration; the frozen R98 files are never edited."""

from pathlib import Path

from experiments.plan_english_r98 import core as original

ROOT, PARENT, HOST = original.ROOT, original.PARENT, original.HOST
PARENT_SHA, HOST_COMMIT = original.PARENT_SHA, original.HOST_COMMIT
SOURCE, REVISION = original.SOURCE, original.REVISION
TASKS, ARMS, SEED = original.TASKS, original.ARMS, original.SEED
STEPS, BATCH = original.STEPS, original.BATCH
CONTEXT, MAX_NEW, SOURCE_MAX_NEW = 1024, 640, 384
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/plan_english_r98_completion"
ORIGINAL_ROWS_SHA = "39776ce074f118a0bdf60fb70d4b0a42dc2cd0bb32f07bb356356041ee25188c"
ORIGINAL_FREEZE_SHA = "196d771f8533951fff93722784107d562cb4485218627472f7f0367b3de1ea35"

sha, canonical, write_once = original.sha, original.canonical, original.write_once
read, rows, append = original.read, original.rows, original.append
model_prompt, target = original.model_prompt, original.target
visible_response, screen = original.visible_response, original.screen
aggregate, source_gate = original.aggregate, original.source_gate


def original_evidence():
    if sha(original.OUT / "freeze.json") != ORIGINAL_FREEZE_SHA:
        raise RuntimeError("Historical freeze changed")
    original.verify_binding()
    path = original.OUT / "source/rows.jsonl"
    if sha(path) != ORIGINAL_ROWS_SHA:
        raise RuntimeError("Historical raw evidence changed")
    observed = rows(path)
    catalog = read(original.OUT / "train.json") + read(original.OUT / "development.json")
    if len(observed) != 240 or [r["id"] for r in observed] != [r["id"] for r in catalog]:
        raise RuntimeError("Historical row coverage mismatch")
    for row, reference in zip(observed, catalog):
        if any(row[k] != reference[k] for k in ("id", "task", "split", "prompt")):
            raise RuntimeError("Historical source/catalog mismatch")
        if row["screen"] != screen(reference, row["teacher_output"]):
            raise RuntimeError("Historical screen mismatch")
        tokens = row["token_ids"]
        if not tokens or not all(type(t) is int and t >= 0 for t in tokens):
            raise RuntimeError("Historical tokens missing or invalid")
        # The pinned source's eos_token_id is bound by its tokenizer inventory.
        if row["truncated"] != (len(tokens) == 144 and tokens[-1] != 151645):
            raise RuntimeError("Historical truncation mismatch")
    result = read(original.OUT / "source/result.json")
    if result["rows_sha256"] != ORIGINAL_ROWS_SHA or result["summary"] != aggregate(observed):
        raise RuntimeError("Historical summary mismatch")
    if result["source_gate"] != source_gate(observed):
        raise RuntimeError("Historical gate mismatch")
    return catalog, observed


def verify_binding():
    manifest = read(OUT / "freeze.json")
    if not manifest.get("files"):
        raise RuntimeError("Missing frozen inputs")
    for relative, digest in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"R98-C1 frozen input changed or missing: {relative}")
    original_evidence()
    return manifest


def merge_completions(catalog, old, completed):
    expected = [r["id"] for r in old if r["truncated"]]
    if [r["id"] for r in completed] != expected:
        raise RuntimeError("Completion coverage mismatch; no selection permitted")
    replacement = {r["id"]: r for r in completed}
    merged = []
    for reference, previous in zip(catalog, old):
        row = replacement.get(previous["id"], previous)
        if row is not previous:
            if any(row[k] != reference[k] for k in ("id", "split", "task", "prompt")):
                raise RuntimeError("Completion prompt mismatch")
            if row["token_ids"][: len(previous["token_ids"])] != previous["token_ids"]:
                raise RuntimeError(
                    "Completion token prefix differs; not a length-only intervention"
                )
            tokens = row["token_ids"]
            if (
                not tokens
                or len(tokens) > SOURCE_MAX_NEW
                or not all(type(t) is int for t in tokens)
            ):
                raise RuntimeError("Completion tokens invalid")
            if row["truncated"] != (len(tokens) == SOURCE_MAX_NEW and tokens[-1] != 151645):
                raise RuntimeError("Completion truncation mismatch")
            if row["screen"] != screen(reference, row["teacher_output"]):
                raise RuntimeError("Completion screen mismatch")
        merged.append(row)
    return merged
