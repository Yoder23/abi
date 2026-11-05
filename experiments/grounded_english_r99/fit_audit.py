"""Read-only checkpoint diagnostic with additive, write-once evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.grounded_english_r99_fp32 import core as c
from experiments.plan_english_r98 import run as engine

OUT = c.ROOT / "results/grounded_english_r99_fit_audit"
ARMS = (*c.ARMS, "parent")


def select_training(catalog):
    chosen = []
    for task in c.TASKS:
        rows = [r for r in catalog if r["task"] == task]
        rows.sort(key=lambda r: hashlib.sha256(("r99-fit-v1:" + r["id"]).encode()).hexdigest())
        chosen.extend(rows[:3])
    if len(chosen) != 24:
        raise RuntimeError("Incomplete diagnostic population")
    return chosen


def token_masks(row, target, offsets, arm):
    import re

    start = target.index("\nResponse:\n") + len("\nResponse:\n") if arm == "abi_plan" else 0
    spans = []
    for literal in row["required"]:
        spans.extend(
            (start + m.start(), start + m.end())
            for m in re.finditer(r"\b" + re.escape(literal) + r"\b", target[start:], re.I)
        )
    response = [end > start for begin, end in offsets]
    fact = [any(begin < right and end > left for left, right in spans) for begin, end in offsets]
    return response + [False], fact + [False]


def freeze():
    c.verify_binding()
    if OUT.exists():
        raise RuntimeError("Preserve previous fit diagnostic")
    selected = select_training(c.read(c.OUT / "train.json"))
    c.write_once(OUT / "training_selection.json", selected)
    paths = [
        Path(__file__),
        Path(__file__).with_name("FIT_AUDIT.md"),
        c.ROOT / "tests/test_r99_fit_audit.py",
        OUT / "training_selection.json",
        c.OUT / "freeze.json",
        c.OUT / "decision.json",
    ]
    for arm in ARMS:
        folder = c.PARENT if arm == "parent" else c.OUT / arm
        paths += [
            folder / name for name in ("model.safetensors", "metadata.json", "tokenizer.json")
        ]
        paths += [c.OUT / arm / name for name in ("development.jsonl", "development_result.json")]
    files = dict(c.read(c.OUT / "freeze.json")["files"])
    files.update({os.path.relpath(p, c.ROOT).replace("\\", "/"): c.sha(p) for p in paths})
    c.write_once(
        OUT / "freeze.json",
        {
            "format": "r99-fit-freeze/1",
            "files": files,
            "training_rows": 24,
            "development_rows": 48,
            "new_optimizer_steps": 0,
            "new_teacher_queries": 0,
        },
    )
    print(json.dumps({"freeze_sha256": c.sha(OUT / "freeze.json"), "inputs": len(files)}))


def binding():
    manifest = c.read(OUT / "freeze.json")
    if not manifest["files"]:
        raise RuntimeError("Missing fit bindings")
    for relative, digest in manifest["files"].items():
        if c.sha(c.ROOT / relative) != digest:
            raise RuntimeError(f"Fit diagnostic input changed: {relative}")
    c.verify_binding()


def generate(model, tokenizer, ids, route):
    generated = []
    first = model(ids, task_routes=route, use_cache=True)
    state = {
        "past_key_values": first["past_key_values"],
        "task_routes": route,
        "next_logits": first["logits"][:, -1],
        "generated_ids": ids[:, :0],
    }
    for _ in range(min(c.MAX_NEW, model.config.max_tokens - ids.shape[1] - 1)):
        selected = state["next_logits"].argmax(-1)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        _, state = model.decode_step(state, next_token=selected)
    return generated


def audit(arm):
    import psutil
    import torch
    import torch.nn.functional as F

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core

    binding()
    folder = OUT / arm
    folder.mkdir()
    torch.manual_seed(c.SEED)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = c.PARENT if arm == "parent" else c.OUT / arm
    model, tokenizer, _ = load_layercake_core(checkpoint, layercake_root=c.HOST, device="cuda")
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Expected FP32 checkpoint")
    selection = c.read(OUT / "training_selection.json")
    catalog = selection + c.read(c.OUT / "development.json")
    source = {r["id"]: r for r in c.rows(c.OUT / "source/rows.jsonl")}
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (
        (folder / "forced.jsonl").open("x", encoding="utf-8") as forced,
        (folder / "training_generation.jsonl").open("x", encoding="utf-8") as autonomous,
        torch.inference_mode(),
    ):
        for i, row in enumerate(catalog):
            if time.perf_counter() - start > 1800:
                raise RuntimeError("Diagnostic arm time limit")
            prefix = tokenizer.encode(c.model_prompt(row, arm), add_special_tokens=False)
            target = c.target(row, source[row["id"]]["teacher_output"], arm)
            encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
            suffix = encoded.input_ids + [tokenizer.eos_token_id]
            response_mask, fact_mask = token_masks(row, target, encoded.offset_mapping, arm)
            if len(prefix) + len(suffix) > c.CONTEXT:
                raise RuntimeError("Diagnostic truncation forbidden")
            ids = torch.tensor([prefix + suffix], device="cuda")
            route = torch.tensor([CAPABILITY_CAKE_ORDER.index(row["task"])], device="cuda")
            logits = model(
                ids,
                prompt_lengths=torch.tensor([len(prefix)], device="cuda"),
                task_routes=route,
                use_cache=False,
            )["logits"][0, len(prefix) - 1 : -1]
            gold = torch.tensor(suffix, device="cuda")
            nll = F.cross_entropy(logits, gold, reduction="none")
            if not torch.isfinite(nll).all():
                raise RuntimeError("Non-finite diagnostic logits")
            c.append(
                forced,
                {
                    "id": row["id"],
                    "task": row["task"],
                    "split": row["split"],
                    "target_ids": suffix,
                    "argmax_ids": logits.argmax(-1).tolist(),
                    "nll": nll.tolist(),
                    "response_mask": response_mask,
                    "fact_mask": fact_mask,
                },
            )
            if row["split"] == "train":
                tokens = generate(model, tokenizer, ids[:, : len(prefix)], route)
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                output, valid = c.visible_response(raw, arm)
                c.append(
                    autonomous,
                    {
                        "id": row["id"],
                        "task": row["task"],
                        "token_ids": tokens,
                        "raw_output": raw,
                        "output": output,
                        "plan_valid": valid,
                        "screen": c.screen(row, output),
                    },
                )
            if (i + 1) % 8 == 0:
                print(json.dumps({"arm": arm, "rows": i + 1}), flush=True)
    c.write_once(
        folder / "receipt.json",
        {
            "arm": arm,
            "checkpoint_sha256": c.sha(checkpoint / "model.safetensors"),
            "forced_sha256": c.sha(folder / "forced.jsonl"),
            "generation_sha256": c.sha(folder / "training_generation.jsonl"),
            "wall_seconds": time.perf_counter() - start,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "rss_bytes": psutil.Process().memory_info().rss,
            "device": torch.cuda.get_device_name(),
            "precision": "float32",
            "teacher_queries": 0,
            "optimizer_steps": 0,
        },
    )


def summarize(forced, autonomous):
    result = {"autonomous": c.aggregate(autonomous)}
    for split in ("train", "development"):
        result[split] = {}
        for mask_name in ("response_mask", "fact_mask"):
            pairs, nlls = [], []
            for row in forced:
                if row["split"] != split:
                    continue
                size = len(row["target_ids"])
                if any(
                    len(row[key]) != size
                    for key in ("argmax_ids", "nll", "response_mask", "fact_mask")
                ):
                    raise RuntimeError("Incomplete diagnostic token rows")
                if any(not math.isfinite(value) or value < 0 for value in row["nll"]):
                    raise RuntimeError("Invalid token NLL")
                for wanted, gold, predicted, nll in zip(
                    row[mask_name], row["target_ids"], row["argmax_ids"], row["nll"]
                ):
                    if wanted:
                        pairs.append(gold == predicted)
                        nlls.append(nll)
            if not pairs:
                raise RuntimeError("Missing response/fact observations")
            result[split][mask_name] = {
                "tokens": len(pairs),
                "correct": sum(pairs),
                "accuracy": sum(pairs) / len(pairs),
                "mean_nll": sum(nlls) / len(nlls),
            }
    return result


def verify():
    from transformers import AutoTokenizer

    binding()
    tokenizer = AutoTokenizer.from_pretrained(c.PARENT, local_files_only=True)
    catalog = c.read(OUT / "training_selection.json") + c.read(c.OUT / "development.json")
    source = {r["id"]: r for r in c.rows(c.OUT / "source/rows.jsonl")}
    results = {}
    for arm in ARMS:
        folder = OUT / arm
        receipt = c.read(folder / "receipt.json")
        for name, key in (
            ("forced.jsonl", "forced_sha256"),
            ("training_generation.jsonl", "generation_sha256"),
        ):
            if c.sha(folder / name) != receipt[key]:
                raise RuntimeError("Diagnostic raw rows changed")
        checkpoint = c.PARENT if arm == "parent" else c.OUT / arm
        if c.sha(checkpoint / "model.safetensors") != receipt["checkpoint_sha256"]:
            raise RuntimeError("Diagnostic checkpoint changed")
        forced = c.rows(folder / "forced.jsonl")
        free = c.rows(folder / "training_generation.jsonl")
        if [r["id"] for r in forced] != [r["id"] for r in catalog] or [r["id"] for r in free] != [
            r["id"] for r in catalog[:24]
        ]:
            raise RuntimeError("Incomplete diagnostic coverage")
        for row, ref in zip(forced, catalog):
            target = c.target(ref, source[ref["id"]]["teacher_output"], arm)
            encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
            masks = token_masks(ref, target, encoded.offset_mapping, arm)
            if (
                row["target_ids"] != encoded.input_ids + [tokenizer.eos_token_id]
                or (row["response_mask"], row["fact_mask"]) != masks
                or row["split"] != ref["split"]
            ):
                raise RuntimeError("Unrecomputable diagnostic masks/targets")
        for row, ref in zip(free, catalog[:24]):
            raw = tokenizer.decode(row["token_ids"], skip_special_tokens=True)
            output, valid = c.visible_response(raw, arm)
            if (
                row["raw_output"] != raw
                or row["output"] != output
                or row["plan_valid"] != valid
                or row["screen"] != c.screen(ref, output)
            ):
                raise RuntimeError("Unrecomputable autonomous diagnostic")
        results[arm] = summarize(forced, free)
        results[arm]["original_development"] = c.aggregate(
            c.rows(c.OUT / arm / "development.jsonl")
        )
    value = {
        "format": "r99-fit-audit/1",
        "results": results,
        "full_moonshot": "NOT_PROVEN",
        "diagnostic_only": True,
        "warning": "Teacher-forced accuracy is not autonomous quality or causal proof",
    }
    path = OUT / "summary.json"
    if path.exists():
        if c.read(path) != value:
            raise RuntimeError("Stale diagnostic summary")
    else:
        c.write_once(path, value)
    print(json.dumps(value, indent=2))


def pilot():
    binding()
    c.write_once(
        OUT / "launch.json", {"pid": os.getpid(), "start_unix": time.time(), "budget_seconds": 3600}
    )
    start = time.monotonic()
    for args in [*(("audit", "--arm", a) for a in ARMS), ("verify",)]:
        try:
            result = subprocess.run(
                [sys.executable, "-u", "-m", "experiments.grounded_english_r99.fit_audit", *args],
                cwd=c.ROOT,
                timeout=max(1, 3600 - (time.monotonic() - start)),
            )
            if result.returncode:
                raise RuntimeError(f"Subprocess failed: {result.returncode}")
            if sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) > 1024**3:
                raise RuntimeError("Diagnostic disk budget")
        except Exception as error:
            c.write_once(
                OUT / "stop.json",
                {"stage": args, "error": repr(error), "scientific_promotion": False},
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "audit", "verify", "pilot"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    engine.c = c
    if args.command == "audit":
        if not args.arm:
            parser.error("audit requires --arm")
        audit(args.arm)
    else:
        {"freeze": freeze, "verify": verify, "pilot": pilot}[args.command]()
