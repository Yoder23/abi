"""Once-only R99 source normalization and matched learner comparison."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from experiments.plan_english_r98 import run as engine

from . import core as c


def freeze():
    if c.OUT.exists():
        raise RuntimeError("R99 output already exists; preserve it")
    c.previous.verify_binding()
    train = c.read(c.previous.OUT / "train.json")
    dev = c.development_catalog()
    if {r["prompt"] for r in train} & {r["prompt"] for r in dev}:
        raise RuntimeError("Training/development overlap")
    c.write_once(c.OUT / "train.json", train)
    c.write_once(c.OUT / "development.json", dev)
    files = dict(c.read(c.previous.OUT / "freeze.json")["files"])
    paths = list(c.HERE.glob("*.py")) + [
        c.HERE / "PROTOCOL.md",
        c.ROOT / "tests/test_grounded_english_r99.py",
    ]
    paths += [
        c.previous.OUT / "source/rows.jsonl",
        c.previous.OUT / "source/result.json",
        c.previous.OUT / "freeze.json",
    ]
    paths += [c.OUT / "train.json", c.OUT / "development.json"]
    files.update({os.path.relpath(p, c.ROOT).replace("\\", "/"): c.sha(p) for p in paths})
    c.write_once(
        c.OUT / "freeze.json",
        {
            "format": "abi-r99-freeze/1",
            "date": "2026-11-05",
            "files": files,
            "abi_commit": engine.git(c.ROOT, "rev-parse", "HEAD"),
            "parent_sha256": c.PARENT_SHA,
            "host_source_commit": c.HOST_COMMIT,
            "source": {"model": c.SOURCE, "revision": c.REVISION},
            "seed": c.SEED,
            "training_rows": 192,
            "prospective_development_rows": 48,
            "new_development_outputs_observed": 0,
            "student_training_performed": False,
            "full_moonshot": "NOT_PROVEN",
        },
    )
    print(json.dumps({"freeze_sha256": c.sha(c.OUT / "freeze.json"), "files": len(files)}))


def tokenizer_from_snapshot():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(
        snapshot_download(c.SOURCE, revision=c.REVISION, local_files_only=True)
    ).resolve()
    inventory = c.read(c.previous.OUT / "source/result.json")["accounting"]["original_acquisition"][
        "source_files"
    ]
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "config.json",
    ):
        if c.sha(snapshot / name) != inventory[name]["sha256"]:
            raise RuntimeError("Pinned tokenizer changed")
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def acquire():
    import psutil
    import torch

    from experiments.factual_semantic_r16.public_qualification import _load_source, _render_chat

    c.verify_binding()
    destination = c.OUT / "source"
    destination.mkdir()
    train, dev = c.read(c.OUT / "train.json"), c.read(c.OUT / "development.json")
    prior = {r["id"]: r for r in c.rows(c.previous.OUT / "source/rows.jsonl")}
    prior_costs = c.read(c.previous.OUT / "source/result.json")["accounting"]
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
    if inventory != prior_costs["original_acquisition"]["source_files"]:
        raise RuntimeError("Source snapshot changed")
    peak_rss = process.memory_info().rss
    raw, normalized = [], []
    system = "Follow the user's English-language request. Use only the supplied fictional information. Return only the requested response, without commentary about the task."
    with (
        (destination / "raw.jsonl").open("x", encoding="utf-8", newline="\n") as raw_log,
        (destination / "rows.jsonl").open("x", encoding="utf-8", newline="\n") as normalized_log,
    ):
        for reference in train + dev:
            if reference["split"] == "train":
                observed = prior[reference["id"]]
            else:
                rendered = _render_chat(tokenizer, system, reference["prompt"])
                encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(
                    "cuda"
                )
                before = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=384,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                tokens = generated[0, encoded.input_ids.shape[1] :].tolist()
                answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                observed = {
                    **{k: reference[k] for k in ("id", "split", "task", "prompt")},
                    "teacher_output": answer,
                    "token_ids": tokens,
                    "input_tokens": int(encoded.input_ids.numel()),
                    "wall_seconds": time.perf_counter() - before,
                    "truncated": len(tokens) == 384 and tokens[-1] != tokenizer.eos_token_id,
                    "screen": c.screen(reference, answer),
                }
                peak_rss = max(peak_rss, process.memory_info().rss)
                print(
                    json.dumps(
                        {"stage": "new_development_source", "rows": len(raw) - len(train) + 1}
                    ),
                    flush=True,
                )
            c.append(raw_log, observed)
            selected = c.normalize(reference, observed["teacher_output"])
            row = {
                **observed,
                "teacher_output": selected["text"],
                "token_ids": tokenizer.encode(selected["text"], add_special_tokens=False),
                "screen": c.screen(reference, selected["text"]),
                "normalization": selected,
                "raw_source_token_count": len(observed["token_ids"]),
            }
            c.append(normalized_log, row)
            raw.append(observed)
            normalized.append(row)
    new = [r for r in raw if r["split"] == "development"]
    c.write_once(
        destination / "result.json",
        {
            "format": "abi-r99-normalized-source/1",
            "rows_sha256": c.sha(destination / "rows.jsonl"),
            "raw_sha256": c.sha(destination / "raw.jsonl"),
            "summary": c.aggregate(normalized),
            "raw_summary": c.aggregate(raw),
            "source_gate": c.source_gate(normalized),
            "accounting": {
                "prior_r98_c1_costs": prior_costs,
                "new_raw_prompts": len(new),
                "total_source_requests_including_prior": prior_costs["total_source_requests"]
                + len(new),
                "new_source_input_tokens": sum(r["input_tokens"] for r in new),
                "new_source_generated_tokens": sum(len(r["token_ids"]) for r in new),
                "total_source_generated_tokens_including_prior": prior_costs[
                    "total_teacher_generated_tokens"
                ]
                + sum(len(r["token_ids"]) for r in new),
                "source_inference_seconds_new": sum(r["wall_seconds"] for r in new),
                "new_unique_prompt_utf8_bytes": sum(len(r["prompt"].encode()) for r in new),
                "raw_target_utf8_bytes": sum(len(r["teacher_output"].encode()) for r in raw),
                "normalized_target_utf8_bytes": sum(
                    len(r["teacher_output"].encode()) for r in normalized
                ),
                "normalized_target_tokens": sum(len(r["token_ids"]) for r in normalized),
                "wall_seconds": time.perf_counter() - start,
                "cpu_process_seconds": sum(process.cpu_times()[:2]) - cpu_start,
                "peak_rss_bytes": peak_rss,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "stored_logits": 0,
                "stored_activations": 0,
                "source_parameters_copied": 0,
                "external_hardware": False,
            },
        },
    )
    print(
        json.dumps(
            {
                "stage": "normalized_source_complete",
                "summary": c.aggregate(normalized),
                "gate": c.source_gate(normalized),
            }
        ),
        flush=True,
    )


def verify_source():
    c.verify_binding()
    destination = c.OUT / "source"
    result = c.read(destination / "result.json")
    for name, key in (("rows.jsonl", "rows_sha256"), ("raw.jsonl", "raw_sha256")):
        if c.sha(destination / name) != result[key]:
            raise RuntimeError("R99 source hash mismatch")
    catalog = c.read(c.OUT / "train.json") + c.read(c.OUT / "development.json")
    raw, normalized = c.rows(destination / "raw.jsonl"), c.rows(destination / "rows.jsonl")
    if len(raw) != 240 or len(normalized) != 240:
        raise RuntimeError("Missing or duplicate source rows")
    tokenizer = tokenizer_from_snapshot()
    prior = {r["id"]: r for r in c.rows(c.previous.OUT / "source/rows.jsonl")}
    for reference, observed, selected in zip(catalog, raw, normalized):
        for key in ("id", "split", "task", "prompt"):
            if observed[key] != reference[key] or selected[key] != reference[key]:
                raise RuntimeError("R99 source coverage mismatch")
        if reference["split"] == "train" and observed != prior[reference["id"]]:
            raise RuntimeError("Reused training source changed")
        if reference["split"] == "development":
            tokens = observed["token_ids"]
            if (
                not tokens
                or len(tokens) > 384
                or not all(type(t) is int and t >= 0 for t in tokens)
            ):
                raise RuntimeError("Invalid raw source tokens")
            if observed["truncated"] != (
                len(tokens) == 384 and tokens[-1] != tokenizer.eos_token_id
            ):
                raise RuntimeError("Unrecomputable raw truncation")
        if (
            tokenizer.decode(observed["token_ids"], skip_special_tokens=True).strip()
            != observed["teacher_output"]
        ):
            raise RuntimeError("Raw token/text mismatch")
        expected = c.normalize(reference, observed["teacher_output"])
        if selected["normalization"] != expected or selected["teacher_output"] != expected["text"]:
            raise RuntimeError("Non-source or unrecomputable normalization")
        if selected["token_ids"] != tokenizer.encode(expected["text"], add_special_tokens=False):
            raise RuntimeError("Normalized token count mismatch")
        if selected["screen"] != c.screen(reference, expected["text"]):
            raise RuntimeError("Unrecomputable normalized screen")
        if observed["screen"] != c.screen(reference, observed["teacher_output"]):
            raise RuntimeError("Unrecomputable raw screen")
        if selected["truncated"] != observed["truncated"]:
            raise RuntimeError("Source truncation hidden")
        if selected["raw_source_token_count"] != len(observed["token_ids"]):
            raise RuntimeError("Raw token accounting mismatch")
    if (
        result["summary"] != c.aggregate(normalized)
        or result["raw_summary"] != c.aggregate(raw)
        or result["source_gate"] != c.source_gate(normalized)
    ):
        raise RuntimeError("Unrecomputable source gate")
    new = [r for r in raw if r["split"] == "development"]
    prior_costs = c.read(c.previous.OUT / "source/result.json")["accounting"]
    expected_accounting = {
        "prior_r98_c1_costs": prior_costs,
        "new_raw_prompts": len(new),
        "total_source_requests_including_prior": prior_costs["total_source_requests"] + len(new),
        "new_source_input_tokens": sum(r["input_tokens"] for r in new),
        "new_source_generated_tokens": sum(len(r["token_ids"]) for r in new),
        "total_source_generated_tokens_including_prior": prior_costs[
            "total_teacher_generated_tokens"
        ]
        + sum(len(r["token_ids"]) for r in new),
        "source_inference_seconds_new": sum(r["wall_seconds"] for r in new),
        "new_unique_prompt_utf8_bytes": sum(len(r["prompt"].encode()) for r in new),
        "raw_target_utf8_bytes": sum(len(r["teacher_output"].encode()) for r in raw),
        "normalized_target_utf8_bytes": sum(len(r["teacher_output"].encode()) for r in normalized),
        "normalized_target_tokens": sum(len(r["token_ids"]) for r in normalized),
    }
    if any(result["accounting"][key] != value for key, value in expected_accounting.items()):
        raise RuntimeError("Unrecomputable source information budget")
    print(
        json.dumps({"verified_normalized_rows": 240, "source_gate": c.source_gate(normalized)}),
        flush=True,
    )
    return normalized


def verify():
    engine.verify()
    raw_teacher = [r for r in c.rows(c.OUT / "source/raw.jsonl") if r["split"] == "development"]
    candidate = c.rows(c.OUT / "abi_plan/development.jsonl")
    if [r["id"] for r in raw_teacher] != [r["id"] for r in candidate]:
        raise RuntimeError("Unpaired raw-teacher comparison")
    delta = [
        int(a["screen"]["mechanical_pass"]) - int(b["screen"]["mechanical_pass"])
        for a, b in zip(candidate, raw_teacher)
    ]
    rng = random.Random(c.SEED)
    boot = sorted(sum(rng.choice(delta) for _ in delta) / len(delta) for _ in range(10000))
    result = {
        "raw_teacher_development": c.aggregate(raw_teacher),
        "candidate_minus_raw_teacher": {
            "difference": statistics.mean(delta),
            "ci95": [boot[249], boot[9749]],
        },
        "interpretation": "Compare raw and normalized teacher separately; no fluency/superiority certification",
        "accounting_note": "Engine unique_teacher_tokens_train counts normalized target tokens; full raw acquisition costs are in source/result.json",
    }
    path = c.OUT / "raw_teacher_reference.json"
    if path.exists():
        if c.read(path) != result:
            raise RuntimeError("Stale raw teacher comparison")
    else:
        c.write_once(path, result)


def pilot():
    c.verify_binding()
    order = list(c.ARMS)
    random.Random(c.SEED).shuffle(order)
    stages = [("preflight",), ("source",), ("verify-source",)]
    stages += [("train", "--arm", arm) for arm in order]
    stages += [("evaluate", "--arm", arm) for arm in ("parent", *order)] + [("verify",)]
    start = time.perf_counter()
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
            done = subprocess.run(
                [sys.executable, "-u", "-m", "experiments.grounded_english_r99.run", *args],
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
        if done.returncode:
            c.write_once(
                c.OUT / "stop.json",
                {
                    "stage": args,
                    "reason": "FAILED_PREREQUISITE_OR_OPERATIONAL_ERROR",
                    "returncode": done.returncode,
                    "layercake_failure_proven": False,
                },
            )
            return
        if sum(p.stat().st_size for p in c.OUT.rglob("*") if p.is_file()) > 20 * 1024**3:
            c.write_once(
                c.OUT / "stop.json", {"stage": args, "reason": "EXPERIMENT_DISK_BUDGET_STOP"}
            )
            return
        if args[0] == "verify-source" and not c.source_gate(c.rows(c.OUT / "source/rows.jsonl")):
            c.write_once(
                c.OUT / "stop.json",
                {
                    "stage": "source",
                    "reason": "ABI_NORMALIZATION_PREREQUISITE_FAILED",
                    "student_training_performed": False,
                    "layercake_quality_tested": False,
                },
            )
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
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
    engine.c = c  # process-local explicit configuration, not a historical code edit
    if args.command in {"train", "evaluate", "verify"} and not c.source_gate(verify_source()):
        raise RuntimeError("ABI_NORMALIZATION_PREREQUISITE_FAILED")
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
            "freeze": freeze,
            "pilot": pilot,
            "preflight": engine.preflight,
            "source": acquire,
            "verify-source": verify_source,
            "verify": verify,
        }[args.command]()


if __name__ == "__main__":
    main()
