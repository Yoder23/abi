"""Evaluate the frozen LayerCake English substrate separately from ABI transfer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once
from . import package_v4 as base
from . import package_v7 as v7
from .protocol import TASKS


CHECKPOINT_SHA256 = "9e0e6b9add32b4c460f7b570a32584f380e59bf6d631e313ff813069d24e09e1"
TOKENIZER_SHA256 = "1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e"
TASK_ROUTES = {
    "prose": 9,
    "summary": 6,
    "rewrite": 9,
    "email": 4,
    "bullets": 4,
    "tone": 4,
    "clarification": 7,
    "conversation": 9,
    "planning": 2,
    "comparison": 3,
    "reasoning": 5,
    "abstention": 7,
}


def _imports(layercake_root: Path):
    value = str(layercake_root)
    if value not in sys.path:
        sys.path.insert(0, value)
    from layercake.training.phase2_shallow_sparse import load_student
    from layercake.phase2_campaign import _deterministic_token
    return load_student, _deterministic_token


@torch.inference_mode()
def _generate(model, tokenizer, prompt: str, route: int, select, device: torch.device) -> tuple[str, int]:
    prompt_ids = tokenizer.encode(prompt.rstrip() + "\n")
    if not prompt_ids or len(prompt_ids) >= model.config.max_tokens - 192:
        raise RuntimeError("R30 v9 prompt is empty or exceeds the frozen host context")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    task_route = torch.tensor([route], dtype=torch.long, device=device)
    result = model(
        ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long, device=device),
        task_routes=task_route,
        use_cache=True,
    )
    state = {
        "past_key_values": result["past_key_values"],
        "task_routes": task_route,
        "next_logits": result["logits"][:, -1],
        "generated_ids": ids[:, :0],
    }
    generated: list[int] = []
    for _ in range(192):
        selected = select(
            state["next_logits"],
            generated,
            penalty=1.10,
            repeat_last_n=64,
            no_repeat_ngram_size=4,
        )
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        _, state = model.decode_step(state, next_token=selected)
    return tokenizer.decode(generated), len(generated)


def run(source: Path, diagnosis: Path, layercake_root: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R30 v9 output exists: {output}")
    rows_path = source / "source_rows.jsonl"
    diagnosis_path = diagnosis / "result.json"
    diagnosis_doc = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if diagnosis_doc.get("verdict") != "PASS_PUBLIC_NEURAL_DIAGNOSIS":
        raise RuntimeError("R30 diagnosis prerequisite changed")
    if sha256_file(checkpoint / "model.safetensors") != CHECKPOINT_SHA256:
        raise RuntimeError("frozen LayerCake checkpoint changed")
    if sha256_file(checkpoint / "tokenizer.json") != TOKENIZER_SHA256:
        raise RuntimeError("frozen LayerCake tokenizer changed")
    load_student, select = _imports(layercake_root)
    device = torch.device("cuda")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, metadata = load_student(checkpoint, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen LayerCake parameter became trainable")
    eval_rows = [row for row in base._jsonl(rows_path) if row["split"] == "evaluation"]
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    observations = []
    for index, row in enumerate(eval_rows, 1):
        before = time.perf_counter()
        text, tokens = _generate(model, tokenizer, v7._normalized_prompt(row), TASK_ROUTES[row["oracle_task"]], select, device)
        torch.cuda.synchronize()
        scored = v7._score(row, text, row["teacher_output"])
        observations.append({
            "record_id": row["record_id"],
            "oracle_task": row["oracle_task"],
            "task_route": TASK_ROUTES[row["oracle_task"]],
            "output": text,
            "generated_tokens": tokens,
            "latency_seconds": time.perf_counter() - before,
            "score": scored,
        })
        peak_rss = max(peak_rss, process.memory_info().rss)
        if index == 1 or index % 24 == 0:
            print(json.dumps({"evaluated": index, "functional": sum(item["score"]["functional"] for item in observations)}), flush=True)
    rows_out = output / "evaluation.jsonl"
    output.mkdir(parents=True)
    write_jsonl_once(rows_out, observations)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    passed = functional >= 132 and all(item["functional"] >= 10 for item in by_task.values()) and by_task["abstention"]["functional"] == 12
    result = {
        "format": "abi-r30-frozen-layercake-host-baseline/9",
        "verdict": "PASS_FROZEN_HOST_BASELINE" if passed else "FAIL_FROZEN_HOST_BASELINE",
        "scientific_role": "RECIPIENT_SUBSTRATE_DIAGNOSTIC_NOT_ABI_TRANSFER",
        "source_rows_sha256": sha256_file(rows_path),
        "diagnosis_sha256": sha256_file(diagnosis_path),
        "layercake": {
            "repository_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=layercake_root, text=True).strip(),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "parameters_total": int(metadata["parameters"]["total"]),
            "parameters_active": int(metadata["parameters"]["active"]),
            "parameter_updates": 0,
        },
        "metrics": {"evaluation_rows": len(observations), "functional": functional, "by_task": by_task},
        "information_accounting": {
            "teacher_present": False,
            "abi_package_present": False,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cpu_rss_bytes": int(peak_rss),
        },
        "artifacts": {"evaluation": {"path": rows_out.name, "sha256": sha256_file(rows_out), "bytes": rows_out.stat().st_size}},
        "claim_ceiling": "FROZEN_LAYERCAKE_HOST_DIAGNOSTIC_NOT_ABI_EXTRACTION_OR_GENERAL_ENGLISH",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.layercake_root.resolve(), args.checkpoint.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
