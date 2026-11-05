"""One full-model control; retained controls and all failures stay immutable."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time

from experiments.grounded_english_r99 import fit_audit
from experiments.grounded_english_r99 import run as source_engine
from experiments.grounded_english_r99_fp32 import train as trainer
from experiments.plan_english_r98 import run as engine

from . import core as c


def freeze():
    fit_audit.binding()
    if c.OUT.exists():
        raise RuntimeError("Preserve prior capacity control")
    if shutil.disk_usage(c.ROOT).free < 30 * 1024**3:
        raise RuntimeError("Insufficient initial disk")
    if not (fit_audit.OUT / "summary.json").is_file():
        raise RuntimeError("Complete fit audit first")
    train, dev = c.read(c.baseline.OUT / "train.json"), c.development_catalog()
    old_dev = c.read(c.baseline.OUT / "development.json")
    if {r["prompt"] for r in dev} & {r["prompt"] for r in train + old_dev}:
        raise RuntimeError("Prospective population overlaps")
    c.write_once(c.OUT / "train.json", train)
    c.write_once(c.OUT / "development.json", dev)
    files = dict(c.read(fit_audit.OUT / "freeze.json")["files"])
    paths = list(c.HERE.glob("*.py")) + [
        c.HERE / "PROTOCOL.md",
        c.ROOT / "tests/test_acquisition_capacity_r100.py",
        c.OUT / "train.json",
        c.OUT / "development.json",
        fit_audit.OUT / "summary.json",
        c.baseline.OUT / "source/result.json",
    ]
    for arm in c.ARMS:
        if arm == "full_trunk":
            continue
        folder = c.checkpoint(arm)
        paths += [
            folder / name for name in ("model.safetensors", "metadata.json", "tokenizer.json")
        ]
        paths += [
            fit_audit.OUT / arm / name for name in ("training_generation.jsonl", "receipt.json")
        ]
        if arm != "parent":
            paths += [folder / name for name in ("training.jsonl", "training_result.json")]
    files.update({os.path.relpath(p, c.ROOT).replace("\\", "/"): c.sha(p) for p in paths})
    c.write_once(
        c.OUT / "freeze.json",
        {
            "format": "r100-capacity-freeze/1",
            "files": files,
            "abi_commit": engine.git(c.ROOT, "rev-parse", "HEAD"),
            "seed": c.SEED,
            "intervention": "all_model_parameters_optimizer_enrollment",
            "new_teacher_queries_allowed": 48,
            "new_training_steps_allowed": c.STEPS,
            "wall_budget_seconds": 7200,
        },
    )
    print(json.dumps({"freeze_sha256": c.sha(c.OUT / "freeze.json"), "inputs": len(files)}))


def train():
    def all_parameters(model, arm):
        if arm != "full_trunk":
            raise RuntimeError("Only one full-model control is registered")
        model.requires_grad_(True)
        return list(model.parameters()), []

    engine.prepare_trainable = all_parameters
    trainer.student("full_trunk")


def evaluate(arm):
    import torch

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core

    c.verify_binding()
    destination = c.OUT / arm
    destination.mkdir(exist_ok=arm == "full_trunk")
    torch.set_num_threads(4)
    torch.manual_seed(c.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, _ = load_layercake_core(
        c.checkpoint(arm), layercake_root=c.HOST, device="cuda"
    )
    model.eval()
    populations = {"development": c.read(c.OUT / "development.json")}
    if arm == "full_trunk":
        populations["training_generation"] = c.read(fit_audit.OUT / "training_selection.json")
    for name, catalog in populations.items():
        path = destination / (name + ".jsonl")
        start = time.perf_counter()
        with path.open("x", encoding="utf-8") as stream, torch.inference_mode():
            for index, row in enumerate(catalog):
                ids = torch.tensor(
                    [tokenizer.encode(c.model_prompt(row, arm), add_special_tokens=False)],
                    device="cuda",
                )
                route = torch.tensor([CAPABILITY_CAKE_ORDER.index(row["task"])], device="cuda")
                before = time.perf_counter()
                tokens = fit_audit.generate(model, tokenizer, ids, route)
                torch.cuda.synchronize()
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                output, valid = c.visible_response(raw, arm)
                c.append(
                    stream,
                    {
                        "id": row["id"],
                        "task": row["task"],
                        "arm": arm,
                        "token_ids": tokens,
                        "raw_output": raw,
                        "output": output,
                        "plan_valid": valid,
                        "screen": c.screen(row, output),
                        "diagnostic_gpu_wall_seconds": time.perf_counter() - before,
                    },
                )
                if (index + 1) % 8 == 0:
                    print(
                        json.dumps({"arm": arm, "population": name, "rows": index + 1}), flush=True
                    )
        c.write_once(
            destination / (name + "_receipt.json"),
            {
                "rows_sha256": c.sha(path),
                "checkpoint_sha256": c.sha(c.checkpoint(arm) / "model.safetensors"),
                "wall_seconds": time.perf_counter() - start,
                "precision": "float32",
                "rows": len(catalog),
            },
        )


def paired(a, b):
    if not a or [r["id"] for r in a] != [r["id"] for r in b]:
        raise RuntimeError("Unpaired observations")
    delta = [
        int(x["screen"]["mechanical_pass"]) - int(y["screen"]["mechanical_pass"])
        for x, y in zip(a, b)
    ]
    rng = random.Random(c.SEED)
    boot = sorted(sum(rng.choice(delta) for _ in delta) / len(delta) for _ in range(10000))
    return {"difference": statistics.mean(delta), "ci95": [boot[249], boot[9749]]}


def validate_generation(rows, catalog, arm, tokenizer):
    if [r["id"] for r in rows] != [r["id"] for r in catalog]:
        raise RuntimeError("Missing, repeated or unpaired generation rows")
    for row, reference in zip(rows, catalog):
        if not all(
            type(token) is int and 0 <= token < len(tokenizer) for token in row["token_ids"]
        ):
            raise RuntimeError("Invalid output tokens")
        raw = tokenizer.decode(row["token_ids"], skip_special_tokens=True)
        output, valid = c.visible_response(raw, arm)
        if (
            row["task"] != reference["task"]
            or row["raw_output"] != raw
            or row["output"] != output
            or row["plan_valid"] != valid
            or row["screen"] != c.screen(reference, output)
        ):
            raise RuntimeError("Unrecomputable generation evidence")


def verify():
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    c.verify_binding()
    source = source_engine.verify_source()
    if not c.source_gate(source):
        raise RuntimeError("Source prerequisite failed")
    tokenizer = AutoTokenizer.from_pretrained(c.PARENT, local_files_only=True)
    catalog = c.read(c.OUT / "development.json")
    observations = {}
    for arm in c.ARMS:
        path = c.OUT / arm / "development.jsonl"
        receipt = c.read(path.parent / "development_receipt.json")
        if (
            c.sha(path) != receipt["rows_sha256"]
            or c.sha(c.checkpoint(arm) / "model.safetensors") != receipt["checkpoint_sha256"]
        ):
            raise RuntimeError("Changed evaluation or checkpoint")
        observations[arm] = c.rows(path)
        validate_generation(observations[arm], catalog, arm, tokenizer)
    folder = c.OUT / "full_trunk"
    training = c.read(folder / "training_result.json")
    log = c.rows(folder / "training.jsonl")
    if (
        c.sha(folder / "model.safetensors") != training["checkpoint_sha256"]
        or c.sha(folder / "training.jsonl") != training["training_log_sha256"]
    ):
        raise RuntimeError("Changed trained payload/log")
    if [r["step"] for r in log] != list(range(1, c.STEPS + 1)) or any(
        not math.isfinite(r["loss"]) or not math.isfinite(r["gradient_norm"]) for r in log
    ):
        raise RuntimeError("Incomplete/nonfinite optimization")
    if (
        training["loss_tokens"] != sum(r["loss_tokens"] for r in log)
        or training["trainable_parameters"] != 87710428
        or training["precision"] != "float32"
    ):
        raise RuntimeError("Wrong full-scope training accounting")
    for arm in ("distillation", "lora", "abi_plan"):
        old_log = c.rows(c.baseline.OUT / arm / "training.jsonl")
        if [r["sample_ids"] for r in old_log] != [r["sample_ids"] for r in log]:
            raise RuntimeError("Unmatched training exposure")
    baseline_training = c.read(c.baseline.OUT / "distillation/training_result.json")
    if (
        training["loss_tokens"] != baseline_training["loss_tokens"]
        or training["padded_token_positions"] != baseline_training["padded_token_positions"]
    ):
        raise RuntimeError("Different response-only workload")
    changed, changed_parameters = [], 0
    with (
        safe_open(folder / "model.safetensors", framework="pt", device="cpu") as after,
        safe_open(c.PARENT / "model.safetensors", framework="pt", device="cpu") as before,
    ):
        if after.keys() != before.keys():
            raise RuntimeError("Unexpected graph change")
        for name in after.keys():
            value = after.get_tensor(name)
            if not torch.isfinite(value).all():
                raise RuntimeError("Nonfinite full-scope payload")
            if not torch.equal(value, before.get_tensor(name)):
                changed.append(name)
                changed_parameters += value.numel()
    if not any(name.startswith("transformer.h.") for name in changed):
        raise RuntimeError("Full-trunk intervention did not execute")
    selection = c.read(fit_audit.OUT / "training_selection.json")
    fitting = c.rows(folder / "training_generation.jsonl")
    validate_generation(fitting, selection, "full_trunk", tokenizer)
    fit_receipt = c.read(folder / "training_generation_receipt.json")
    if c.sha(folder / "training_generation.jsonl") != fit_receipt["rows_sha256"]:
        raise RuntimeError("Changed fit observations")
    fit_comparisons = {}
    for arm in ("distillation", "lora", "abi_plan", "parent"):
        previous = c.rows(fit_audit.OUT / arm / "training_generation.jsonl")
        validate_generation(previous, selection, arm, tokenizer)
        fit_comparisons[arm] = paired(fitting, previous)
    normalized = [r for r in source if r["split"] == "development"]
    raw = [r for r in c.rows(c.OUT / "source/raw.jsonl") if r["split"] == "development"]
    full = observations["full_trunk"]
    comparisons = {
        arm: paired(full, rows) for arm, rows in observations.items() if arm != "full_trunk"
    }
    comparisons.update(
        {"normalized_teacher": paired(full, normalized), "raw_teacher": paired(full, raw)}
    )
    summary, fit_summary = c.aggregate(full), c.aggregate(fitting)
    useful = (
        summary["passing"] >= 44
        and summary["collapses"] == 0
        and all(r["passing"] >= 5 for r in summary["per_task"].values())
        and comparisons["normalized_teacher"]["ci95"][0] >= -0.10
    )
    value = {
        "format": "r100-capacity-decision/1",
        "verdict": "CAPACITY_CONTROL_COMPLETED",
        "useful_quality_screen": useful,
        "training_fit_screen": fit_summary["passing"] >= 22 and fit_summary["collapses"] == 0,
        "scope_sensitivity_supported": all(
            fit_comparisons[a]["ci95"][0] > 0 for a in ("distillation", "lora")
        ),
        "development": {arm: c.aggregate(rows) for arm, rows in observations.items()},
        "raw_teacher": c.aggregate(raw),
        "normalized_teacher": c.aggregate(normalized),
        "training_fit": fit_summary,
        "development_comparisons": comparisons,
        "training_fit_comparisons": fit_comparisons,
        "changed_tensors": changed,
        "changed_parameter_elements": changed_parameters,
        "new_teacher_queries": 48,
        "prior_r99_development_queries_separately_incurred": 48,
        "source_accounting": c.read(c.OUT / "source/result.json")["accounting"],
        "training_accounting": training,
        "full_moonshot": "NOT_PROVEN",
        "human_fluency": "UNASSESSED",
        "interpretation": "Conventional full-model control only; no ABI product or LayerCake runtime regression certification",
    }
    path = c.OUT / "decision.json"
    if path.exists():
        if c.read(path) != value:
            raise RuntimeError("Stale capacity decision")
    else:
        c.write_once(path, value)
    print(
        json.dumps(
            {
                key: value[key]
                for key in (
                    "verdict",
                    "useful_quality_screen",
                    "training_fit_screen",
                    "scope_sensitivity_supported",
                    "development",
                    "training_fit",
                )
            }
        )
    )


def pilot():
    c.verify_binding()
    c.write_once(
        c.OUT / "launch.json",
        {"pid": os.getpid(), "start_unix": time.time(), "wall_limit_seconds": 7200},
    )
    start = time.monotonic()
    stages = [("source",), ("verify-source",), ("train",)]
    stages += [("evaluate", "--arm", arm) for arm in c.ARMS] + [("verify",)]
    for args in stages:
        try:
            remaining = 7200 - (time.monotonic() - start)
            if args[0] == "train":
                remaining = min(remaining, 1800)
            if remaining <= 0:
                raise RuntimeError("Capacity control wall budget")
            done = subprocess.run(
                [sys.executable, "-u", "-m", "experiments.acquisition_capacity_r100.run", *args],
                cwd=c.ROOT,
                timeout=remaining,
            )
            if done.returncode:
                raise RuntimeError(f"Prerequisite/operation failed: {done.returncode}")
            if sum(p.stat().st_size for p in c.OUT.rglob("*") if p.is_file()) > 8 * 1024**3:
                raise RuntimeError("Capacity control disk budget")
        except Exception as error:
            c.write_once(
                c.OUT / "stop.json",
                {"stage": args, "error": repr(error), "scientific_promotion": False},
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "pilot", "source", "verify-source", "train", "evaluate", "verify"),
    )
    parser.add_argument("--arm", choices=c.ARMS)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    engine.c = source_engine.c = trainer.c = c
    if args.command in {"verify-source", "train", "evaluate", "verify"}:
        if not c.source_gate(source_engine.verify_source()):
            raise RuntimeError("ABI_SOURCE_PREREQUISITE_FAILED")
    if args.command == "evaluate":
        if not args.arm:
            parser.error("evaluate requires --arm")
        evaluate(args.arm)
    else:
        {
            "freeze": freeze,
            "pilot": pilot,
            "source": source_engine.acquire,
            "verify-source": lambda: None,
            "train": train,
            "verify": verify,
        }[args.command]()
