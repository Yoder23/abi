"""Acquire an exactly counterbalanced R32 teacher corpus."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.english_substrate_r30 import package_v7 as v7
from experiments.english_substrate_r30.protocol import INSTRUCTIONS, TASKS, _details
from experiments.english_sufficiency_r31.acquire import MODEL_ID, REVISION, SYSTEMS
from experiments.factual_semantic_r16.public_qualification import _generate, _load_source, _render_chat
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


ROWS_PER_CELL = 2
MAX_CANDIDATES_PER_CELL = 6
DETAIL_OFFSET = 1_500_000


def _detail_index(task_index: int, instruction_index: int, detail_cycle: int, candidate: int) -> int:
    value = DETAIL_OFFSET + task_index * 100_000 + instruction_index * 10_000 + detail_cycle * 1_000 + candidate * 7
    while value % 3 != detail_cycle:
        value += 1
    return value


def _candidate(task: str, instruction_index: int, detail_cycle: int, candidate: int) -> dict[str, Any]:
    task_index = TASKS.index(task)
    index = _detail_index(task_index, instruction_index, detail_cycle, candidate)
    instruction = INSTRUCTIONS[task][instruction_index]
    details = _details(task, index)
    prompt = "\n".join((f"INSTRUCTION: {instruction}", "SUPPLIED MATERIAL:", *details))
    digest = hashlib.sha256(
        f"r32|{task}|{instruction_index}|{detail_cycle}|{candidate}|{prompt}".encode()
    ).hexdigest()[:20]
    return {
        "record_id": f"r32-train-{digest}",
        "split": "train",
        "oracle_task": task,
        "instruction": instruction,
        "instruction_index": instruction_index,
        "detail_cycle": detail_cycle,
        "details": details,
        "nonce": f"Virelon{index:05d}",
        "prompt": prompt,
    }


def run(
    output: Path,
    *,
    candidate_factory=_candidate,
    format_version: str = "abi-r32-counterbalanced-source/1",
    index_strategy: str = "high-offset-v1",
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R32 acquisition exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R32 acquisition requires CUDA")
    output.mkdir(parents=True)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(MODEL_ID, REVISION)
    load_seconds = time.perf_counter() - started
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    generated_tokens = 0
    rendered_tokens = 0
    failed = False

    for task in TASKS:
        for instruction_index in range(3):
            for detail_cycle in range(3):
                cell_accepted = 0
                for candidate_index in range(MAX_CANDIDATES_PER_CELL):
                    if cell_accepted == ROWS_PER_CELL:
                        break
                    row = candidate_factory(task, instruction_index, detail_cycle, candidate_index)
                    chosen = None
                    row_attempts = []
                    for attempt_index, system in enumerate(SYSTEMS, 1):
                        rendered = _render_chat(tokenizer, system, row["prompt"])
                        answer, count = _generate(tokenizer, model, rendered, 192)
                        answer = answer.strip().replace("\r\n", "\n")
                        scored = v7._score(row, answer, answer)
                        item = {
                            "record_id": row["record_id"],
                            "attempt": attempt_index,
                            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                            "output": answer,
                            "output_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                            "generated_tokens": count,
                            "score": scored,
                        }
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
                        accepted.append(
                            {
                                **row,
                                "teacher_output": chosen["output"],
                                "teacher_output_sha256": chosen["output_sha256"],
                                "teacher_output_tokens": chosen["generated_tokens"],
                                "accepted_attempt": chosen["attempt"],
                            }
                        )
                        cell_accepted += 1
                if cell_accepted != ROWS_PER_CELL:
                    failed = True
                    break
            if failed:
                break
        print(
            json.dumps(
                {
                    "task": task,
                    "accepted": sum(row["oracle_task"] == task for row in accepted),
                    "teacher_calls": len(attempts),
                    "quarantined": len(quarantined),
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        if failed:
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
    cells = {
        task: {
            f"instruction-{instruction_index}/detail-{detail_cycle}": sum(
                row["oracle_task"] == task
                and row["instruction_index"] == instruction_index
                and row["detail_cycle"] == detail_cycle
                for row in accepted
            )
            for instruction_index in range(3)
            for detail_cycle in range(3)
        }
        for task in TASKS
    }
    passed = len(accepted) == 216 and all(count == ROWS_PER_CELL for task in cells.values() for count in task.values())
    result = {
        "format": format_version,
        "verdict": "PASS_COUNTERBALANCED_SOURCE" if passed else "FAIL_COUNTERBALANCED_SOURCE",
        "source": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "snapshot": str(snapshot),
            "teacher_training_steps": 0,
        },
        "design": {
            "instruction_wordings": 3,
            "detail_cycles": 3,
            "rows_per_cell": ROWS_PER_CELL,
            "max_candidates_per_cell": MAX_CANDIDATES_PER_CELL,
            "index_strategy": index_strategy,
            "accepted_by_cell": cells,
        },
        "metrics": {
            "accepted_rows": len(accepted),
            "teacher_calls": len(attempts),
            "quarantined_candidates": len(quarantined),
            "accepted_by_task": {task: sum(row["oracle_task"] == task for row in accepted) for task in TASKS},
            "accepted_by_attempt": {str(index): sum(row["accepted_attempt"] == index for row in accepted) for index in range(1, 4)},
        },
        "information_accounting": {
            "raw_source_prompts": len(attempts),
            "unique_candidate_prompts": len(accepted) + len(quarantined),
            "unique_candidate_prompt_bytes": sum(len(row["prompt"].encode()) for row in accepted + quarantined),
            "teacher_output_bytes_all_attempts": sum(len(row["output"].encode()) for row in attempts),
            "teacher_generated_tokens": generated_tokens,
            "rendered_prompt_tokens": rendered_tokens,
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "source_parameters_copied": 0,
            "source_load_seconds": load_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cpu_rss_bytes": int(peak_rss),
            "external_hardware_used": False,
        },
        "artifacts": {
            name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            for name, path in (("accepted_rows", accepted_path), ("attempts", attempts_path), ("quarantine", quarantine_path))
        },
        "claim_ceiling": "COUNTERBALANCED_SUPPLIED_CONTENT_SOURCE_NOT_GENERAL_ENGLISH",
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
