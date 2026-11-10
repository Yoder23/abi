"""Preserve and diagnose the first R102 corpus-freeze failure; no GPU or retry."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.coverage_exposure_r102 import core as c  # noqa: E402


def main():
    from transformers import AutoTokenizer

    output = ROOT / "results/coverage_exposure_r102_preflight_v1"
    if output.exists() or c.OUT.exists():
        raise RuntimeError("Preserve existing preflight evidence")
    tokenizer = AutoTokenizer.from_pretrained(c.PARENT, local_files_only=True)
    training = c.read(c.previous.OUT / "train.json")
    development = c.development_catalog()
    old_dev = c.read(c.previous.OUT / "development.json") + c.read(c.previous.baseline.OUT / "development.json")
    targets = {r["id"]: r["teacher_output"] for r in c.rows(c.previous.OUT / "source/rows.jsonl")}
    occupied = {name for row in training + development + old_dev for name in c.entities(row)}
    completed = 0
    for row in training:
        for alias in range(4):
            try:
                c.construct_alias(row, targets[row["id"]], alias, tokenizer, occupied)
                completed += 1
                continue
            except RuntimeError as error:
                observed_error = str(error)
            seed = int.from_bytes(hashlib.sha256(f"r102:{row['id']}:{alias}".encode()).digest()[:8], "big")
            rng = random.Random(seed)
            target = targets[row["id"]]
            pc = len(tokenizer.encode(c.model_prompt(row, "distillation"), add_special_tokens=False))
            tc = len(tokenizer.encode(target, add_special_tokens=False))
            records, counts = [], Counter()
            for attempt in range(4096):
                mapping = c.alias_mapping(row, rng)
                derived = c.transform(row, mapping)
                text = c.replace_literals(target, mapping)
                dp = len(tokenizer.encode(c.model_prompt(derived, "distillation"), add_special_tokens=False)) - pc
                dt = len(tokenizer.encode(text, add_special_tokens=False)) - tc
                collision = bool(set(mapping.values()) & occupied) or len(set(mapping.values())) != len(mapping)
                same_screen = c.screen(derived, text) == c.screen(row, target)
                reason = "collision" if collision else "length" if dp or dt else "screen" if not same_screen else "eligible"
                counts[reason] += 1
                records.append({"attempt": attempt, "mapping": mapping, "prompt_token_delta": dp,
                                "target_token_delta": dt, "reason": reason})
            output.mkdir()
            with (output / "candidates.jsonl").open("x", encoding="utf-8") as stream:
                for record in records:
                    c.append(stream, record)
            summary = {"format": "r102-preflight-failure/1", "client_date": "2026-11-10",
                       "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "error": observed_error, "completed_aliases_before_failure": completed,
                       "row": row, "target": target, "alias": alias, "attempts": len(records),
                       "rejection_counts": dict(counts), "candidate_sha256": c.sha(output / "candidates.jsonl"),
                       "source_queries": 0, "optimizer_steps": 0, "successful_freeze_created": False}
            c.write_once(output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return
    raise RuntimeError("Original preflight failure did not reproduce")


if __name__ == "__main__":
    main()
