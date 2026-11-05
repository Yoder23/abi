"""R98-C1 orchestration. No edit, overwrite, or promotion of historical R98."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

from experiments.plan_english_r98 import run as engine

from . import core as c


def audit_r98():
    _, observed = c.original_evidence()
    result = {
        "format": "abi-r98-source-audit/1",
        "raw_sha256": c.ORIGINAL_ROWS_SHA,
        "summary": c.aggregate(observed),
        "source_gate_recomputed": c.source_gate(observed),
        "truncated_rows": sum(r["truncated"] for r in observed),
        "truncation_by_task": {
            task: sum(r["task"] == task and r["truncated"] for r in observed) for task in c.TASKS
        },
        "training_artifacts_present": [arm for arm in c.ARMS if (c.original.OUT / arm).exists()],
        "attribution": "ABI_SOURCE_PREREQUISITE_FAILED; LayerCake generation untested",
        "full_moonshot": "NOT_PROVEN",
    }
    path = c.ROOT / "results/plan_english_r98_review/source_audit.json"
    if path.exists():
        if c.read(path) != result:
            raise RuntimeError("Historical audit is stale")
    else:
        c.write_once(path, result)
    print(json.dumps(result, indent=2), flush=True)


def freeze():
    if c.OUT.exists():
        raise RuntimeError("C1 output already exists; preserve it")
    catalog, old = c.original_evidence()
    if len([r for r in old if r["truncated"]]) != 31:
        raise RuntimeError("Unregistered completion population")
    if engine.git(c.HOST, "diff", "--", "layercake"):
        raise RuntimeError("Dirty host source")
    for split in ("train", "development"):
        c.write_once(c.OUT / f"{split}.json", [r for r in catalog if r["split"] == split])
    files = dict(c.read(c.original.OUT / "freeze.json")["files"])
    paths = list(c.HERE.glob("*.py")) + [c.HERE / "PROTOCOL.md"]
    paths += [c.ROOT / "tests/test_plan_english_r98_completion.py"]
    paths += [c.OUT / "train.json", c.OUT / "development.json"]
    paths += [
        c.original.OUT / "freeze.json",
        c.original.OUT / "source/rows.jsonl",
        c.original.OUT / "source/result.json",
        c.original.OUT / "stop.json",
    ]
    files.update({os.path.relpath(p, c.ROOT).replace("\\", "/"): c.sha(p) for p in paths})
    value = {
        "format": "abi-r98-completion-freeze/1",
        "date": "2026-11-05",
        "files": files,
        "abi_commit": engine.git(c.ROOT, "rev-parse", "HEAD"),
        "host_source_commit": c.HOST_COMMIT,
        "parent_sha256": c.PARENT_SHA,
        "source": {"model": c.SOURCE, "revision": c.REVISION},
        "completion_ids": [r["id"] for r in old if r["truncated"]],
        "source_max_new_tokens": c.SOURCE_MAX_NEW,
        "context": c.CONTEXT,
        "generation_max_new_tokens": c.MAX_NEW,
        "seed": c.SEED,
        "steps": c.STEPS,
        "batch": c.BATCH,
        "completion_outputs_observed": 0,
        "development_is_hidden": False,
        "full_moonshot": "NOT_PROVEN",
    }
    c.write_once(c.OUT / "freeze.json", value)
    print(json.dumps({"freeze_sha256": c.sha(c.OUT / "freeze.json"), "files": len(files)}))


def acquire():
    import psutil
    import torch

    from experiments.factual_semantic_r16.public_qualification import _load_source, _render_chat

    c.verify_binding()
    catalog, old = c.original_evidence()
    old_result = c.read(c.original.OUT / "source/result.json")
    destination = c.OUT / "source"
    destination.mkdir()
    start = time.perf_counter()
    process = psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model, snapshot = _load_source(c.SOURCE, c.REVISION)
    inventory = {
        str(p.relative_to(snapshot)): {"sha256": c.sha(p), "bytes": p.stat().st_size}
        for p in snapshot.rglob("*")
        if p.is_file()
    }
    if inventory != old_result["accounting"]["source_files"]:
        raise RuntimeError("Source snapshot differs from R98")
    if tokenizer.eos_token_id != 151645:
        raise RuntimeError("Unexpected pinned source EOS")
    peak_rss = process.memory_info().rss
    system = "Follow the user's English-language request. Use only the supplied fictional information. Return only the requested response, without commentary about the task."
    completed = []
    with (destination / "completions.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
        for reference, previous in zip(catalog, old):
            if not previous["truncated"]:
                continue
            rendered = _render_chat(tokenizer, system, reference["prompt"])
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
            before = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=c.SOURCE_MAX_NEW,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            tokens = generated[0, encoded.input_ids.shape[1] :].tolist()
            answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            row = {
                **{key: reference[key] for key in ("id", "split", "task", "prompt")},
                "teacher_output": answer,
                "token_ids": tokens,
                "input_tokens": int(encoded.input_ids.numel()),
                "wall_seconds": time.perf_counter() - before,
                "truncated": len(tokens) == c.SOURCE_MAX_NEW
                and tokens[-1] != tokenizer.eos_token_id,
                "screen": c.screen(reference, answer),
            }
            c.append(journal, row)
            completed.append(row)
            peak_rss = max(peak_rss, process.memory_info().rss)
            if tokens[: len(previous["token_ids"])] != previous["token_ids"]:
                raise RuntimeError("Completion prefix disagreement; attempt preserved")
            print(
                json.dumps(
                    {
                        "stage": "complete_source",
                        "rows": len(completed),
                        "tokens": len(tokens),
                        "passes": row["screen"]["mechanical_pass"],
                    }
                ),
                flush=True,
            )
    merged = c.merge_completions(catalog, old, completed)
    with (destination / "rows.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
        for row in merged:
            c.append(journal, row)
    costs = {
        "wall_seconds": time.perf_counter() - start,
        "cpu_process_seconds": sum(process.cpu_times()[:2]) - cpu_start,
        "source_inference_seconds": sum(r["wall_seconds"] for r in completed),
        "teacher_input_tokens": sum(r["input_tokens"] for r in completed),
        "teacher_generated_tokens": sum(len(r["token_ids"]) for r in completed),
        "peak_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "external_hardware": False,
    }
    value = {
        "format": "abi-r98-completion-source/1",
        "original_rows_sha256": c.ORIGINAL_ROWS_SHA,
        "rows_sha256": c.sha(destination / "rows.jsonl"),
        "completions_sha256": c.sha(destination / "completions.jsonl"),
        "summary": c.aggregate(merged),
        "source_gate": c.source_gate(merged),
        "accounting": {
            "original_acquisition": old_result["accounting"],
            "completion_acquisition": costs,
            "total_teacher_generated_tokens": old_result["accounting"]["teacher_generated_tokens"]
            + costs["teacher_generated_tokens"],
            "total_teacher_input_tokens": old_result["accounting"]["teacher_input_tokens"]
            + costs["teacher_input_tokens"],
            "raw_unique_prompts": len(merged),
            "total_source_requests": len(old) + len(completed),
            "unique_prompt_utf8_bytes": sum(len(s.encode()) for s in {r["prompt"] for r in merged}),
            "imported_completed_output_tokens": sum(len(r["token_ids"]) for r in merged),
            "imported_completed_output_utf8_bytes": sum(
                len(r["teacher_output"].encode()) for r in merged
            ),
            "stored_logits": 0,
            "stored_activations": 0,
            "source_parameters_copied": 0,
        },
    }
    c.write_once(destination / "result.json", value)
    print(
        json.dumps(
            {
                "stage": "source_complete",
                "summary": value["summary"],
                "source_gate": value["source_gate"],
            }
        ),
        flush=True,
    )


def verify_source():
    from transformers import AutoTokenizer

    c.verify_binding()
    catalog, old = c.original_evidence()
    destination = c.OUT / "source"
    result = c.read(destination / "result.json")
    for file, key in (("rows.jsonl", "rows_sha256"), ("completions.jsonl", "completions_sha256")):
        if c.sha(destination / file) != result[key]:
            raise RuntimeError("Source evidence hash mismatch")
    completed = c.rows(destination / "completions.jsonl")
    merged = c.merge_completions(catalog, old, completed)
    if c.rows(destination / "rows.jsonl") != merged:
        raise RuntimeError("Merged evidence not reconstructible")
    tokenizer = AutoTokenizer.from_pretrained(c.SOURCE, revision=c.REVISION, local_files_only=True)
    for row in merged:
        if (
            tokenizer.decode(row["token_ids"], skip_special_tokens=True).strip()
            != row["teacher_output"]
        ):
            raise RuntimeError("Source token/text mismatch")
    if result["summary"] != c.aggregate(merged) or result["source_gate"] != c.source_gate(merged):
        raise RuntimeError("Unrecomputable source gate")
    print(
        json.dumps(
            {
                "verified_raw_source": len(merged),
                "source_gate": c.source_gate(merged),
                "summary": c.aggregate(merged),
                "full_moonshot": "NOT_PROVEN",
            }
        ),
        flush=True,
    )
    return merged


def pilot():
    c.verify_binding()
    start = time.perf_counter()
    order = list(c.ARMS)
    random.Random(c.SEED).shuffle(order)
    stages = [("preflight",), ("source",), ("verify-source",)]
    stages += [("train", "--arm", arm) for arm in order]
    stages += [("evaluate", "--arm", arm) for arm in ("parent", *order)] + [("verify",)]
    c.write_once(
        c.OUT / "launch.json",
        {
            "command": sys.argv,
            "pid": os.getpid(),
            "stages": stages,
            "start_unix": time.time(),
            "wall_limit_seconds": 21600,
        },
    )
    for args in stages:
        remaining = 21600 - (time.perf_counter() - start)
        try:
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, 21600)
            completed = subprocess.run(
                [sys.executable, "-u", "-m", "experiments.plan_english_r98_completion.run", *args],
                cwd=c.ROOT,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            c.write_once(
                c.OUT / "stop.json",
                {
                    "stage": args,
                    "reason": "EXPERIMENT_BUDGET_STOP",
                    "hard_resource_limit_proven": False,
                },
            )
            return
        if completed.returncode:
            c.write_once(
                c.OUT / "stop.json",
                {
                    "stage": args,
                    "returncode": completed.returncode,
                    "reason": "FAILED_PREREQUISITE_OR_OPERATIONAL_ERROR",
                    "layercake_failure_proven": False,
                },
            )
            return
        if sum(p.stat().st_size for p in c.OUT.rglob("*") if p.is_file()) > 20 * 1024**3:
            c.write_once(
                c.OUT / "stop.json", {"stage": args, "reason": "EXPERIMENT_DISK_BUDGET_STOP"}
            )
            return
        if args[0] == "verify-source":
            observed = c.rows(c.OUT / "source/rows.jsonl")
            if not c.source_gate(observed):
                c.write_once(
                    c.OUT / "stop.json",
                    {
                        "stage": "source",
                        "reason": "ABI_SOURCE_PREREQUISITE_FAILED",
                        "summary": c.aggregate(observed),
                        "student_training_performed": False,
                        "layercake_quality_tested": False,
                        "rows_sha256": c.sha(c.OUT / "source/rows.jsonl"),
                    },
                )
                print("C1 source prerequisite failed; no student training authorized", flush=True)
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "audit-r98",
            "freeze",
            "pilot",
            "preflight",
            "source",
            "verify-source",
            "train",
            "evaluate",
            "verify",
        ),
    )
    parser.add_argument("--arm", choices=(*c.ARMS, "parent"))
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Explicit process-local configuration of a hash-bound immutable engine.
    # No historical Python module, checkpoint, or output file is modified.
    engine.c = c
    if args.command in {"train", "evaluate", "verify"}:
        if not c.source_gate(verify_source()):
            raise RuntimeError("ABI_SOURCE_PREREQUISITE_FAILED")
    if args.command == "train":
        if args.arm not in c.ARMS:
            parser.error("Training requires a registered arm")
        engine.student(args.arm)
    elif args.command == "evaluate":
        if args.arm is None:
            parser.error("Evaluation requires an arm")
        engine.evaluate(args.arm)
    else:
        {
            "audit-r98": audit_r98,
            "freeze": freeze,
            "pilot": pilot,
            "preflight": engine.preflight,
            "source": acquire,
            "verify-source": verify_source,
            "verify": engine.verify,
        }[args.command]()


if __name__ == "__main__":
    main()
