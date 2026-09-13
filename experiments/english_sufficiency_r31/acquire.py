"""Acquire validator-selected, fully accounted R31 teacher responses."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.english_substrate_r30 import package_v7 as v7
from experiments.english_substrate_r30.protocol import INSTRUCTIONS, TASKS, _details
from experiments.factual_semantic_r16.public_qualification import _generate, _load_source, _render_chat
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


MODEL_ID = "Qwen/Qwen2-7B-Instruct"
REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
QUOTAS = {"train": 48, "evaluation": 12}
SYSTEMS = (
    "Perform the requested operation using only SUPPLIED MATERIAL. Preserve every identifier and relevant number. Follow the requested format. Never add facts. Return only the answer.",
    "Answer the request exactly from SUPPLIED MATERIAL. Silently check identifiers, numbers, logic, format, and unsupported claims before returning only the final answer.",
    "Produce a concise grounded answer. Copy every required identifier and number exactly, obey all format constraints, and explicitly abstain when support is absent. Return no commentary.",
)


def _candidate(task: str, split: str, ordinal: int) -> dict[str, Any]:
    task_index = TASKS.index(task)
    offset = 400_000 if split == "train" else 800_000
    index = offset + task_index * 10_000 + ordinal
    bank = list(INSTRUCTIONS[task])
    random.Random(31_000 + task_index + (10_000 if split == "evaluation" else 0)).shuffle(bank)
    instruction = bank[ordinal % len(bank)]
    details = _details(task, index)
    prompt = "\n".join((f"INSTRUCTION: {instruction}", "SUPPLIED MATERIAL:", *details))
    digest = hashlib.sha256(f"r31|{split}|{task}|{ordinal}|{prompt}".encode()).hexdigest()[:20]
    return {"record_id": f"r31-{split}-{digest}", "split": split, "oracle_task": task, "instruction": instruction, "details": details, "nonce": f"Virelon{index:05d}", "prompt": prompt}


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R31 acquisition exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R31 acquisition requires CUDA")
    output.mkdir(parents=True)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(MODEL_ID, REVISION)
    load_seconds = time.perf_counter() - started
    accepted, attempts, quarantined = [], [], []
    generated_tokens = rendered_tokens = 0
    for split, quota in QUOTAS.items():
        for task in TASKS:
            task_accepted = 0
            ordinal = 0
            while task_accepted < quota and ordinal < quota * 3:
                row = _candidate(task, split, ordinal)
                ordinal += 1
                chosen = None
                row_attempts = []
                for attempt_index, system in enumerate(SYSTEMS):
                    rendered = _render_chat(tokenizer, system, row["prompt"])
                    answer, count = _generate(tokenizer, model, rendered, 192)
                    answer = answer.strip().replace("\r\n", "\n")
                    scored = v7._score(row, answer, answer)
                    item = {"record_id": row["record_id"], "attempt": attempt_index + 1, "system_sha256": hashlib.sha256(system.encode()).hexdigest(), "output": answer, "output_sha256": hashlib.sha256(answer.encode()).hexdigest(), "generated_tokens": count, "score": scored}
                    attempts.append(item)
                    row_attempts.append(item)
                    generated_tokens += count
                    rendered_tokens += len(tokenizer.encode(rendered, add_special_tokens=False))
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    if scored["functional"]:
                        chosen = item
                        break
                if chosen is None:
                    quarantined.append({**row, "attempt_ids": [item["attempt"] for item in row_attempts]})
                else:
                    accepted.append({**row, "teacher_output": chosen["output"], "teacher_output_sha256": chosen["output_sha256"], "teacher_output_tokens": chosen["generated_tokens"], "accepted_attempt": chosen["attempt"]})
                    task_accepted += 1
                if len(attempts) == 1 or len(attempts) % 50 == 0:
                    print(json.dumps({"split": split, "task": task, "accepted": task_accepted, "quota": quota, "teacher_calls": len(attempts), "quarantined": len(quarantined), "seconds": time.perf_counter() - started}), flush=True)
            if task_accepted != quota:
                break
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    accepted_path = output / "accepted_rows.jsonl"
    attempts_path = output / "attempts.jsonl"
    quarantine_path = output / "quarantine.jsonl"
    write_jsonl_once(accepted_path, accepted)
    write_jsonl_once(attempts_path, attempts)
    write_jsonl_once(quarantine_path, quarantined)
    counts = {split: {task: sum(row["split"] == split and row["oracle_task"] == task for row in accepted) for task in TASKS} for split in QUOTAS}
    passed = all(counts[split][task] == quota for split, quota in QUOTAS.items() for task in TASKS)
    result = {
        "format": "abi-r31-validator-selected-source/1",
        "verdict": "PASS_NORMALIZED_SOURCE" if passed else "FAIL_NORMALIZED_SOURCE",
        "source": {"model_id": MODEL_ID, "revision": REVISION, "snapshot": str(snapshot), "teacher_training_steps": 0},
        "quotas": QUOTAS,
        "metrics": {"accepted_rows": len(accepted), "teacher_calls": len(attempts), "quarantined_candidates": len(quarantined), "accepted_by_split_task": counts, "accepted_by_attempt": {str(index): sum(row["accepted_attempt"] == index for row in accepted) for index in range(1, 4)}},
        "information_accounting": {"raw_source_prompts": len(attempts), "unique_candidate_prompts": len(accepted) + len(quarantined), "unique_candidate_prompt_bytes": sum(len(row["prompt"].encode()) for row in accepted + quarantined), "teacher_output_bytes_all_attempts": sum(len(row["output"].encode()) for row in attempts), "teacher_generated_tokens": generated_tokens, "rendered_prompt_tokens": rendered_tokens, "logits_stored": 0, "hidden_activations_stored": 0, "source_parameters_copied": 0, "source_load_seconds": load_seconds, "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss), "external_hardware_used": False},
        "artifacts": {name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for name, path in (("accepted_rows", accepted_path), ("attempts", attempts_path), ("quarantine", quarantine_path))},
        "claim_ceiling": "VALIDATOR_SELECTED_DISCLOSED_SUPPLIED_CONTENT_NOT_GENERAL_ENGLISH",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
