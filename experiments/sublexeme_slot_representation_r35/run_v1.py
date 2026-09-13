"""Execute the R35 typed sublexeme-slot representation gate."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import psutil
import torch
from safetensors.torch import save_file

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30 import package_v7 as v7
from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31.cascade_v3 import _runtime_score
from experiments.english_sufficiency_r31.hidden_v4 import TASK_TO_CLUSTER
from experiments.field_slot_representation_r34 import run_v1 as r34
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once


EXPECTED_R34_FAILURE_SHA256 = "72ccfc0358ad16a4f530847e1980c683737afdf95e8ddaf307b52e14e298c17c"
FORMAT = "abi-r35-lossless-sublexeme-field-slot/1"
BOUNDARY = re.compile(rb"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


class SublexemeSlotTokenizer(r34.FieldSlotTokenizer):
    @staticmethod
    def split(value: bytes | str) -> list[bytes]:
        pieces = base._layercake(Path(__file__).resolve().parents[2])["LosslessLexemePointerTokenizer"].split(value)
        result: list[bytes] = []
        for piece in pieces:
            result.extend(part for part in BOUNDARY.split(piece) if part)
        return result

    def document(self) -> dict[str, Any]:
        document = super().document()
        document["format"] = FORMAT
        document["sublexeme_boundary"] = "ASCII letter-digit and digit-letter"
        return document


def _prepare(rows: list[dict[str, Any]]) -> tuple[SublexemeSlotTokenizer, list[tuple[list[int], list[int]]]]:
    prepared = [{"prompt": v7._normalized_prompt(row), "response": row["teacher_output"]} for row in rows]
    tokenizer = SublexemeSlotTokenizer.build(prepared)
    encoded = []
    for row in prepared:
        source, source_lexemes = tokenizer.encode_source(row["prompt"])
        target = tokenizer.encode_target(row["response"], source_lexemes)
        if len(source) > 128 or len(target) > r34.MAXIMUM_ACTIONS:
            raise RuntimeError("R35 source or target exceeds frozen boundary")
        if tokenizer.decode_actions(target, source_lexemes).decode("utf-8", errors="strict") != row["response"]:
            raise RuntimeError("R35 training representation is not lossless")
        encoded.append((source, target))
    return tokenizer, encoded


def run(r31_rows: Path, r31_failure: Path, r33_failure: Path, r34_failure: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R35 output exists: {output}")
    prerequisites = (
        (r31_rows, r34.EXPECTED_R31_ROWS_SHA256),
        (r31_failure, r34.EXPECTED_R31_FAILURE_SHA256),
        (r33_failure, r34.EXPECTED_R33_FAILURE_SHA256),
        (r34_failure, EXPECTED_R34_FAILURE_SHA256),
    )
    for path, digest in prerequisites:
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"R35 prerequisite missing or changed: {path}")

    training_rows = [row for row in base._jsonl(r31_rows) if row["split"] == "train"]
    fixture_path = r31_failure.parent / "fixture.jsonl"
    fixture = base._jsonl(fixture_path)
    if len(training_rows) != 576 or len(fixture) != 144:
        raise RuntimeError("R35 source or fixture cardinality changed")

    output.mkdir(parents=True)
    (output / "models").mkdir()
    api = base._layercake(Path(__file__).resolve().parents[2])
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    observations: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    representation_failures = 0

    for position, task in enumerate(TASKS, 1):
        task_rows = [row for row in training_rows if row["oracle_task"] == task]
        tokenizer, encoded = _prepare(task_rows)
        model, training = r34._train(api, encoded, tokenizer, 35_000 + position)
        state_path = output / "models" / f"{task}.safetensors"
        save_file({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, state_path)
        tokenizer_path = output / "models" / f"{task}.tokenizer.json"
        write_json_once(tokenizer_path, tokenizer.document())
        models.append({
            "task": task,
            "cluster": TASK_TO_CLUSTER[task],
            "state": {"path": state_path.relative_to(output).as_posix(), "sha256": sha256_file(state_path), "bytes": state_path.stat().st_size},
            "tokenizer": {"path": tokenizer_path.relative_to(output).as_posix(), "sha256": sha256_file(tokenizer_path), "bytes": tokenizer_path.stat().st_size, "semantic_sha256": tokenizer.hash()},
            "training": training,
        })
        for row in [item for item in fixture if item["oracle_task"] == task]:
            prompt = v7._normalized_prompt({"prompt": row["prompt"], "nonce": row["nonce"]})
            try:
                source, source_lexemes = tokenizer.encode_source(prompt)
                source_tensor = torch.tensor([source], dtype=torch.long, device="cuda")
                actions = model.generate_actions(source_tensor, maximum_actions=r34.MAXIMUM_ACTIONS)[0]
                generated = tokenizer.decode_actions(actions, source_lexemes).decode("utf-8", errors="strict")
            except ValueError as exc:
                representation_failures += 1
                actions, generated, error = [], "", str(exc)
            else:
                error = None
            inferred, score = _runtime_score(row["prompt"], generated)
            observations.append({
                "record_id": row["record_id"],
                "oracle_task": row["oracle_task"],
                "cluster": TASK_TO_CLUSTER[inferred],
                "route_exact": TASK_TO_CLUSTER[inferred] == TASK_TO_CLUSTER[row["oracle_task"]],
                "inferred_task": inferred,
                "contract_exact": inferred == row["oracle_task"],
                "actions": actions,
                "output": generated,
                "representation_error": error,
                "score": score,
            })
            peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps({"trained_models": position, "task": task, "functional": sum(item["score"]["functional"] for item in observations), "seconds": time.perf_counter() - started}), flush=True)
        del model
        torch.cuda.empty_cache()

    observations.sort(key=lambda row: (TASKS.index(row["oracle_task"]), row["record_id"]))
    rows_path = output / "evaluation.jsonl"
    write_jsonl_once(rows_path, observations)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    route_exact = sum(row["route_exact"] for row in observations)
    contract_exact = sum(row["contract_exact"] for row in observations)
    passed = functional >= 132 and all(item["functional"] >= 11 for item in by_task.values()) and by_task["abstention"]["functional"] == 12 and route_exact == 144 and contract_exact == 144 and representation_failures == 0
    result = {
        "format": "abi-r35-sublexeme-field-slot-feasibility/1",
        "verdict": "PASS_SUBLEXEME_SLOT_DEVELOPMENT" if passed else "FAIL_SUBLEXEME_SLOT_DEVELOPMENT",
        "inputs": {"r31_rows_sha256": sha256_file(r31_rows), "r31_failure_sha256": sha256_file(r31_failure), "r33_failure_sha256": sha256_file(r33_failure), "r34_failure_sha256": sha256_file(r34_failure), "fixture_sha256": sha256_file(fixture_path)},
        "metrics": {"evaluation_rows": len(observations), "functional": functional, "by_task": by_task, "route_exact": route_exact, "contract_exact": contract_exact, "representation_failures": representation_failures},
        "models": models,
        "information_accounting": {"unique_teacher_outputs": len(training_rows), "teacher_output_tokens": sum(row["teacher_output_tokens"] for row in training_rows), "teacher_output_bytes": sum(len(row["teacher_output"].encode()) for row in training_rows), "new_teacher_calls": 0, "new_teacher_tokens": 0, "source_parameters_copied": 0, "maximum_active_parameters": max(item["training"]["parameters"] for item in models), "deployed_state_bytes": sum(item["state"]["bytes"] + item["tokenizer"]["bytes"] for item in models), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)},
        "evaluation": {"path": rows_path.name, "sha256": sha256_file(rows_path), "bytes": rows_path.stat().st_size},
        "claim_ceiling": "ABI_REPRESENTATION_DEVELOPMENT_ONLY_NOT_LAYERCAKE_PACKAGE_OR_HIDDEN_REPLICATION",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r31-rows", type=Path, required=True)
    parser.add_argument("--r31-failure", type=Path, required=True)
    parser.add_argument("--r33-failure", type=Path, required=True)
    parser.add_argument("--r34-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.r31_rows.resolve(), args.r31_failure.resolve(), args.r33_failure.resolve(), args.r34_failure.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
