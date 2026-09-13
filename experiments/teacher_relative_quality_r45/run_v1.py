"""Run the frozen R44 package outputs against their live source teacher."""

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

from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31.acquire import (
    MODEL_ID,
    REVISION,
    SYSTEMS,
)
from experiments.english_sufficiency_r31.cascade_v3 import _runtime_score
from experiments.factual_semantic_r16.public_qualification import (
    _generate,
    _load_source,
    _render_chat,
)
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)

EXPECTED_R44 = "41f8625fba5b73538a572bb5bc9fe059da83f605b7933201909d76141176eec8"
EXPECTED_FIXTURE = "9409797e98fa771a3bbd62715bf33477f5db0db4e0c55b822b89805cb2ddf5c2"
EXPECTED_EVALUATION = "85314b258b3f46edd42da24c717f1104ac79b525dba70aff1412d6184e3a3845"
EXPECTED_PACKAGE = "6ccb115de59e71b1c70efcfe2b5a1673b88d8f1478fb0059d8577ef5e2b6e6a4"
EXPECTED_SYSTEM = "6d5f78c347dd548eb592b2d70c8b65a2066d75c9f379e77ca45472c87b682e08"
MAXIMUM_NEW_TOKENS = 192


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(
    r44_result: Path,
    fixture_path: Path,
    evaluation_path: Path,
    candidate_package: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R45 output exists: {output}")
    for path, digest in (
        (r44_result, EXPECTED_R44),
        (fixture_path, EXPECTED_FIXTURE),
        (evaluation_path, EXPECTED_EVALUATION),
        (candidate_package, EXPECTED_PACKAGE),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"R45 frozen input changed: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("R45 requires CUDA")
    system = SYSTEMS[0]
    if hashlib.sha256(system.encode()).hexdigest() != EXPECTED_SYSTEM:
        raise RuntimeError("R45 system prompt changed")
    fixture = _jsonl(fixture_path)
    evaluation = _jsonl(evaluation_path)
    candidates = {
        row["record_id"]: row
        for row in evaluation
        if row.get("device") == "cpu" and row.get("condition") == "candidate"
    }
    if (
        len(fixture) != 144
        or len(candidates) != 144
        or {row["record_id"] for row in fixture} != set(candidates)
    ):
        raise RuntimeError("R45 paired candidate matrix changed")
    output.mkdir(parents=True)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(MODEL_ID, REVISION)
    load_seconds = time.perf_counter() - started
    source_parameters = sum(parameter.numel() for parameter in model.parameters())
    rows = []
    total_input_tokens = 0
    total_output_tokens = 0
    for index, source in enumerate(fixture, 1):
        rendered = _render_chat(tokenizer, system, source["prompt"])
        input_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        before = time.perf_counter()
        teacher, output_tokens = _generate(tokenizer, model, rendered, MAXIMUM_NEW_TOKENS)
        latency = time.perf_counter() - before
        teacher = teacher.strip().replace("\r\n", "\n")
        teacher.encode("utf-8", errors="strict")
        candidate = candidates[source["record_id"]]["output"]
        _, teacher_score = _runtime_score(source["prompt"], teacher)
        _, candidate_score = _runtime_score(source["prompt"], candidate)
        rows.append(
            {
                "record_id": source["record_id"],
                "oracle_task": source["oracle_task"],
                "prompt_sha256": hashlib.sha256(source["prompt"].encode()).hexdigest(),
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "input_tokens": input_tokens,
                "teacher_output": teacher,
                "teacher_output_sha256": hashlib.sha256(teacher.encode()).hexdigest(),
                "teacher_output_tokens": output_tokens,
                "teacher_latency_seconds": latency,
                "teacher_score": teacher_score,
                "candidate_output": candidate,
                "candidate_output_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                "candidate_score": candidate_score,
                "candidate_improvement": bool(
                    candidate_score["functional"] and not teacher_score["functional"]
                ),
                "candidate_regression": bool(
                    teacher_score["functional"] and not candidate_score["functional"]
                ),
            }
        )
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        peak_rss = max(peak_rss, process.memory_info().rss)
        if index == 1 or index % 12 == 0:
            print(
                json.dumps(
                    {
                        "rows": index,
                        "teacher_functional": sum(
                            row["teacher_score"]["functional"] for row in rows
                        ),
                        "candidate_functional": sum(
                            row["candidate_score"]["functional"] for row in rows
                        ),
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    rows_path = output / "paired_quality.jsonl"
    write_jsonl_once(rows_path, rows)
    by_task = {
        task: {
            "rows": sum(row["oracle_task"] == task for row in rows),
            "teacher_functional": sum(
                row["oracle_task"] == task and row["teacher_score"]["functional"] for row in rows
            ),
            "candidate_functional": sum(
                row["oracle_task"] == task and row["candidate_score"]["functional"] for row in rows
            ),
        }
        for task in TASKS
    }
    metrics = {
        "paired_rows": len(rows),
        "teacher_functional": sum(row["teacher_score"]["functional"] for row in rows),
        "candidate_functional": sum(row["candidate_score"]["functional"] for row in rows),
        "candidate_improvements": sum(row["candidate_improvement"] for row in rows),
        "candidate_regressions": sum(row["candidate_regression"] for row in rows),
        "teacher_noncollapsed": sum(row["teacher_score"]["noncollapsed"] for row in rows),
        "candidate_noncollapsed": sum(row["candidate_score"]["noncollapsed"] for row in rows),
        "by_task": by_task,
    }
    passed = (
        metrics["paired_rows"] == 144
        and metrics["candidate_functional"] == 141
        and metrics["candidate_functional"] >= metrics["teacher_functional"]
        and metrics["candidate_improvements"] >= metrics["candidate_regressions"]
        and metrics["candidate_regressions"] <= 3
        and all(row["candidate_functional"] >= 10 for row in by_task.values())
        and by_task["abstention"]["candidate_functional"] == 12
        and metrics["teacher_noncollapsed"] == 144
        and metrics["candidate_noncollapsed"] == 144
    )
    result = {
        "format": "abi-r45-teacher-relative-quality/1",
        "verdict": "PASS_R45_BOUNDED_TEACHER_RELATIVE_QUALITY"
        if passed
        else "FAIL_R45_BOUNDED_TEACHER_RELATIVE_QUALITY",
        "inputs": {
            "r44_result_sha256": EXPECTED_R44,
            "fixture_sha256": EXPECTED_FIXTURE,
            "evaluation_sha256": EXPECTED_EVALUATION,
            "candidate_package_sha256": EXPECTED_PACKAGE,
            "system_sha256": EXPECTED_SYSTEM,
        },
        "source": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "snapshot": str(snapshot),
            "parameters": source_parameters,
            "generation": "greedy",
            "attempts_per_prompt": 1,
            "maximum_new_tokens": MAXIMUM_NEW_TOKENS,
        },
        "metrics": metrics,
        "information_accounting": {
            "teacher_calls": len(rows),
            "rendered_input_tokens": total_input_tokens,
            "teacher_generated_tokens": total_output_tokens,
            "raw_prompt_bytes": sum(len(row["prompt"].encode()) for row in fixture),
            "teacher_output_bytes": sum(len(row["teacher_output"].encode()) for row in rows),
            "candidate_output_bytes": sum(len(row["candidate_output"].encode()) for row in rows),
            "source_load_seconds": load_seconds,
            "source_generation_seconds": sum(row["teacher_latency_seconds"] for row in rows),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cpu_rss_bytes": peak_rss,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "source_model_deleted_after_capture": True,
            "cuda_allocated_after_cleanup": int(torch.cuda.memory_allocated()),
            "external_hardware_used": False,
            "candidate_regenerated": False,
            "candidate_training_steps": 0,
        },
        "artifacts": {
            "paired_quality": {
                "path": rows_path.name,
                "sha256": sha256_file(rows_path),
                "bytes": rows_path.stat().st_size,
            }
        },
        "claim_ceiling": "SAME_PROMPT_BOUNDED_FUNCTIONAL_TEACHER_COMPARISON_NOT_HUMAN_OR_UNRESTRICTED_ENGLISH_PARITY",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r44-result", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.r44_result.resolve(),
        args.fixture.resolve(),
        args.evaluation.resolve(),
        args.candidate_package.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
