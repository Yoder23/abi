"""Preregister, execute and strictly recompute the bounded R102 factorial."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time

from experiments.execution_continuation_r101 import core as prior_diagnostic

from . import core as c
from . import source, train


def freeze():
    from transformers import AutoTokenizer

    prior_diagnostic.binding()
    if c.OUT.exists():
        raise RuntimeError("Preserve prior R102 evidence")
    if shutil.disk_usage(c.ROOT).free < 30 * 1024**3:
        raise RuntimeError("Insufficient initial disk")
    tokenizer = AutoTokenizer.from_pretrained(c.PARENT, local_files_only=True)
    training, development = c.read(c.previous.OUT / "train.json"), c.development_catalog()
    source_rows = {r["id"]: r for r in c.rows(c.previous.OUT / "source/rows.jsonl")}
    previous_dev = c.read(c.previous.OUT / "development.json") + c.read(c.previous.baseline.OUT / "development.json")
    original_ids = {name for r in training + previous_dev for name in c.entities(r)}
    dev_ids = {name for r in development for name in c.entities(r)}
    if dev_ids & original_ids:
        raise RuntimeError("Development identifiers overlap prior populations")
    occupied = original_ids | dev_ids
    aliases = [c.construct_profiled_alias(row, source_rows[row["id"]]["teacher_output"], alias, tokenizer, occupied)
               for row in training for alias in range(4)]
    if len({r["row"]["prompt"] for r in aliases}) != 768:
        raise RuntimeError("Aliased population has duplicates")
    if ({r["prompt"] for r in development} & {r["prompt"] for r in training + previous_dev}
            or {r["prompt"] for r in development} & {r["row"]["prompt"] for r in aliases}):
        raise RuntimeError("Development prompt overlap")
    for name, value in (("train.json", training), ("development.json", development),
                        ("aliases.json", aliases), ("schedule.json", c.training_schedule())):
        c.write_once(c.OUT / name, value)
    order = c.trajectories()
    random.Random(102003).shuffle(order)
    files = dict(c.read(prior_diagnostic.OUT / "freeze.json")["files"])
    paths = list(c.HERE.glob("*.py")) + [c.HERE / "PROTOCOL.md", c.HERE / "PREFLIGHT_A1.md",
        c.ROOT / "tests/test_coverage_exposure_r102.py", prior_diagnostic.OUT / "freeze.json", prior_diagnostic.OUT / "decision.json",
        c.previous.OUT / "source/raw.jsonl", c.previous.OUT / "source/rows.jsonl", c.previous.OUT / "source/result.json",
        c.previous.r99.OUT / "source/result.json"] + list(c.OUT.glob("*.json"))
    paths += list((c.ROOT / "results/coverage_exposure_r102_preflight_v1").glob("*"))
    files.update({os.path.relpath(p, c.ROOT).replace("\\", "/"): c.sha(p) for p in paths})
    c.write_once(c.OUT / "freeze.json", {
        "format": "r102-coverage-exposure-freeze/2", "client_date": "2026-11-10", "files": files,
        "construction_amendment": "PREFLIGHT_A1",
        "abi_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.ROOT, text=True).strip(),
        "trajectory_order": order, "endpoints": c.ENDPOINTS, "teacher_queries": 96,
        "training_steps": 6144, "budget_seconds": 10800,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "tokenizers", "safetensors", "numpy", "psutil")},
        "original_teacher_training_rows": 192, "synthetic_rows": len(aliases),
        "synthetic_unique_prompt_utf8_bytes": sum(len(p.encode()) for p in {r["row"]["prompt"] for r in aliases}),
        "synthetic_unique_target_utf8_bytes": sum(len(p.encode()) for p in {r["target"] for r in aliases}),
    })
    print(json.dumps({"freeze_sha256": c.sha(c.OUT / "freeze.json"), "inputs": len(files), "synthetic_rows": len(aliases)}))


def checkpoint(name):
    if name == "parent":
        return c.PARENT
    if name == "retained_full":
        return c.previous.checkpoint("full_trunk")
    if name not in c.endpoint_names():
        raise ValueError("Unregistered endpoint")
    return c.OUT / "checkpoints" / name


def generate(model, tokenizer, prompt, route):
    from experiments.execution_continuation_r101.run import state_from_prefill

    state = state_from_prefill(model, prompt, route)
    tokens = []
    for _ in range(min(c.MAX_NEW, model.config.max_tokens - prompt.shape[1] - 1)):
        selected = state["next_logits"].argmax(-1)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            return tokens, "eos"
        tokens.append(token)
        model.decode_step(state, next_token=selected)
    return tokens, "budget"


def evaluate(name):
    import psutil
    import torch

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core

    c.binding()
    folder = c.OUT / "evaluation" / name
    folder.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(c.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    model, tokenizer, _ = load_layercake_core(checkpoint(name), layercake_root=c.HOST, device="cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    populations = {"development": c.read(c.OUT / "development.json")}
    if name in c.endpoint_names():
        populations["training_fit"] = c.read(c.previous.ROOT / "results/grounded_english_r99_fit_audit/training_selection.json")
    hashes = {}
    peak_rss = psutil.Process().memory_info().rss
    with torch.inference_mode():
        for split, population in populations.items():
            path = folder / f"{split}.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                for index, ref in enumerate(population):
                    if time.perf_counter()-started > 900:
                        raise RuntimeError("R102 endpoint evaluation time budget")
                    ids = torch.tensor([tokenizer.encode(c.model_prompt(ref, "distillation"), add_special_tokens=False)], device="cuda")
                    route = torch.tensor([CAPABILITY_CAKE_ORDER.index(ref["task"])], device="cuda")
                    tokens, stop = generate(model, tokenizer, ids, route)
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    c.append(stream, {"id": ref["id"], "task": ref["task"], "endpoint": name,
                                      "token_ids": tokens, "output": text, "stop_reason": stop,
                                      "screen": c.screen(ref, text)})
                    peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                    if (index+1) % 24 == 0:
                        print(json.dumps({"endpoint": name, "population": split, "rows": index+1}), flush=True)
            hashes[path.name] = c.sha(path)
    c.write_once(folder / "receipt.json", {"checkpoint_sha256": c.sha(checkpoint(name) / "model.safetensors"),
                 "files": hashes, "wall_seconds": time.perf_counter()-started,
                 "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "sampled_peak_rss_bytes": peak_rss,
                 "device": torch.cuda.get_device_name(), "precision": "float32"})


def verify_aliases(tokenizer):
    training = c.read(c.OUT / "train.json")
    development = c.read(c.OUT / "development.json")
    if development != c.development_catalog() or training != c.read(c.previous.OUT / "train.json"):
        raise RuntimeError("Catalog changed")
    aliases = c.read(c.OUT / "aliases.json")
    source_rows = {r["id"]: r for r in c.rows(c.previous.OUT / "source/rows.jsonl")}
    old_dev = c.read(c.previous.OUT / "development.json") + c.read(c.previous.baseline.OUT / "development.json")
    occupied = {name for row in training + development + old_dev for name in c.entities(row)}
    expected = [c.construct_profiled_alias(row, source_rows[row["id"]]["teacher_output"], alias, tokenizer, occupied)
                for row in training for alias in range(4)]
    if aliases != expected:
        raise RuntimeError("Alias provenance, length matching or deterministic selection changed")
    if c.read(c.OUT / "schedule.json") != c.training_schedule():
        raise RuntimeError("Schedule changed")
    return training, aliases


def verify():
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER

    torch.set_num_threads(4)
    c.binding()
    normalized = source.verify()
    if not c.source_gate(normalized):
        raise RuntimeError("SOURCE_PREREQUISITE_FAILED")
    tokenizer = AutoTokenizer.from_pretrained(c.PARENT, local_files_only=True)
    training, _ = verify_aliases(tokenizer)
    source_targets = {r["id"]: r["teacher_output"] for r in normalized if r["split"] == "train"}
    prefix_counts = [len(tokenizer.encode(c.model_prompt(r, "distillation"), add_special_tokens=False)) for r in training]
    target_counts = [len(tokenizer.encode(source_targets[r["id"]], add_special_tokens=False)) + 1 for r in training]
    schedule = c.training_schedule()
    logs = {}
    for trajectory in c.trajectories():
        path = c.OUT / "training" / trajectory / "training.jsonl"
        receipt = c.read(path.parent / "receipt.json")
        if c.sha(path) != receipt["rows_sha256"] or receipt["steps"] != 1536:
            raise RuntimeError("Training log changed")
        log = c.rows(path)
        c.finite_log(log)
        for row, expected in zip(log, schedule):
            indexes = expected["indexes"]
            if (row["sample_ids"] != [training[i]["id"] for i in indexes]
                    or row["prompt_tokens"] != [prefix_counts[i] for i in indexes]
                    or row["loss_tokens"] != sum(target_counts[i] for i in indexes)
                    or row["epoch"] != expected["epoch"]
                    or row["alias"] != (expected["epoch"] % 4 if trajectory.endswith("augment") else None)):
                raise RuntimeError("Unmatched data exposure")
        logs[trajectory] = log
    with safe_open(c.PARENT / "model.safetensors", framework="pt", device="cpu") as parent:
        parent_shapes = {key: tuple(parent.get_slice(key).get_shape()) for key in parent.keys()}
    selected_routes = {CAPABILITY_CAKE_ORDER.index(task) for task in c.TASKS}
    adapter_prefixes = tuple([f"task_cakes.{route}." for route in selected_routes]
                            + [f"capability_layer_adapters.{layer}.{route}." for layer in range(6) for route in selected_routes])
    expected_trainable = {
        "lora": sum(32*(shape[0]+shape[1]) for key, shape in parent_shapes.items()
                    if key.startswith("transformer.h.") and key.endswith(("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"))),
        "distillation": sum(math.prod(shape) for key, shape in parent_shapes.items() if key.startswith(adapter_prefixes)),
    }
    for name in c.endpoint_names():
        trajectory, step_text = name.rsplit("_s", 1)
        step = int(step_text)
        folder = checkpoint(name)
        receipt = c.read(folder / "training_receipt.json")
        if (receipt["checkpoint_sha256"] != c.sha(folder / "model.safetensors")
                or receipt["step"] != step or receipt["trajectory"] != trajectory
                or receipt["loss_tokens"] != sum(r["loss_tokens"] for r in logs[trajectory][:step])
                or receipt["padded_positions"] != step*c.BATCH*c.CONTEXT or receipt["precision"] != "float32"
                or receipt["seed"] != c.SEED or receipt["inherited_parameters"] != 87710428
                or receipt["trainable_parameters"] != expected_trainable[trajectory.split("_")[0]]):
            raise RuntimeError("Unrecomputable endpoint training receipt")
        with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as state:
            if {key: tuple(state.get_slice(key).get_shape()) for key in state.keys()} != parent_shapes:
                raise RuntimeError("Unexpected deployed topology")
            if any(not torch.isfinite(state.get_tensor(key)).all() for key in state.keys()):
                raise RuntimeError("Nonfinite deployed tensors")
    dev = c.read(c.OUT / "development.json")
    fit = c.read(c.previous.ROOT / "results/grounded_english_r99_fit_audit/training_selection.json")
    observations, summaries, fits, quality = {}, {}, {}, {}
    teacher = [r for r in normalized if r["split"] == "development"]
    raw_teacher = [r for r in c.rows(c.OUT / "source/raw.jsonl") if r["split"] == "development"]
    for name in (*c.endpoint_names(), "parent", "retained_full"):
        folder = c.OUT / "evaluation" / name
        receipt = c.read(folder / "receipt.json")
        expected_files = {"development.jsonl", "training_fit.jsonl"} if name in c.endpoint_names() else {"development.jsonl"}
        if set(receipt["files"]) != expected_files or receipt["checkpoint_sha256"] != c.sha(checkpoint(name) / "model.safetensors"):
            raise RuntimeError("Incomplete evaluation receipt")
        for file_name, digest in receipt["files"].items():
            if c.sha(folder / file_name) != digest:
                raise RuntimeError("Evaluation rows changed")
        observed = c.rows(folder / "development.jsonl")
        c.validate_generation(observed, dev, tokenizer)
        if any(row["endpoint"] != name for row in observed):
            raise RuntimeError("Wrong endpoint lineage")
        observations[name] = observed
        summaries[name] = {"overall": c.aggregate(observed), "strata": {}}
        quality[name] = True
        for stratum in c.STRATA:
            selected = [r for r, ref in zip(observed, dev) if ref["stratum"] == stratum]
            comparator = [r for r, ref in zip(teacher, dev) if ref["stratum"] == stratum]
            summary = c.aggregate(selected)
            comparison = c.paired(selected, comparator)
            summaries[name]["strata"][stratum] = {"summary": summary, "normalized_teacher_comparison": comparison}
            quality[name] &= c.quality_screen(summary, comparison)
        summaries[name]["raw_teacher_comparison"] = c.paired(observed, raw_teacher)
        if name in c.endpoint_names():
            fitted = c.rows(folder / "training_fit.jsonl")
            c.validate_generation(fitted, fit, tokenizer)
            if any(r["endpoint"] != name for r in fitted):
                raise RuntimeError("Wrong training-fit lineage")
            fits[name] = c.aggregate(fitted)
    comparisons = {"entity_coverage": {}, "optimization_exposure": {}, "interaction": {}}
    for method in c.METHODS:
        comparisons["entity_coverage"][method] = c.paired(observations[f"{method}_augment_s1536"], observations[f"{method}_repeat_s1536"])
        comparisons["optimization_exposure"][method] = c.paired(observations[f"{method}_repeat_s1536"], observations[f"{method}_repeat_s384"])
        # Interaction is descriptive: difference of the two paired budget effects.
        aug = c.paired(observations[f"{method}_augment_s1536"], observations[f"{method}_augment_s384"])
        comparisons["interaction"][method] = {"augment_budget_effect": aug,
            "difference_in_differences": aug["difference"] - comparisons["optimization_exposure"][method]["difference"]}
    effects = {key: all(v["ci97_5"][0] > 0 for v in comparisons[key].values())
               for key in ("entity_coverage", "optimization_exposure")}
    value = {"format": "r102-factorial-decision/1", "verdict": "BOUNDED_FACTORIAL_COMPLETED_NO_PRODUCT_PROMOTION",
             "source": c.aggregate(normalized), "raw_teacher": c.aggregate(raw_teacher), "normalized_teacher": c.aggregate(teacher),
             "summaries": summaries, "training_fit": fits, "comparisons": comparisons,
             "supported_diagnostic_factors": effects, "autonomous_mechanical_screen": quality,
             "training_fit_screen": {name: v["passing"] >= 22 and v["collapses"] == 0 for name, v in fits.items()},
             "full_moonshot": "NOT_PROVEN", "native_layercake_defect": "NOT_ESTABLISHED"}
    path = c.OUT / "decision.json"
    if path.exists():
        if c.read(path) != value:
            raise RuntimeError("Stale R102 decision")
    else:
        c.write_once(path, value)
    print(json.dumps({"verdict": value["verdict"], "supported_factors": effects, "quality_screens": quality,
                      "decision_sha256": c.sha(path)}), flush=True)


def pilot():
    frozen = c.binding()
    stages = [("source",)] + [("train", "--trajectory", name) for name in frozen["trajectory_order"]]
    stages += [("evaluate", "--endpoint", name) for name in (*c.endpoint_names(), "parent", "retained_full")] + [("verify",)]
    c.write_once(c.OUT / "launch.json", {"pid": os.getpid(), "start_unix": time.time(), "budget_seconds": 10800, "stages": stages})
    started = time.monotonic()
    for index, args in enumerate(stages):
        print(json.dumps({"starting_stage": args}), flush=True)
        try:
            remaining = 10800 - (time.monotonic()-started)
            if remaining <= 0:
                raise RuntimeError("R102 total time budget")
            done = subprocess.run([sys.executable, "-u", "-m", "experiments.coverage_exposure_r102.run", *args],
                                  cwd=c.ROOT, timeout=min(2100, remaining), capture_output=True, text=True)
            c.write_once(c.OUT / "operations" / f"stage-{index:02d}.json",
                         {"args": args, "returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr})
            print(done.stdout[-2500:], flush=True)
            if done.returncode:
                raise RuntimeError(f"Subprocess failed: {args}, {done.returncode}")
            if sum(p.stat().st_size for p in c.OUT.rglob("*") if p.is_file()) > 6*1024**3:
                raise RuntimeError("R102 disk budget")
        except Exception as error:
            c.write_once(c.OUT / "stop.json", {"stage": args, "error": repr(error),
                         "native_layercake_defect": "NOT_ESTABLISHED", "scientific_promotion": False})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "source", "train", "evaluate", "verify", "pilot"))
    parser.add_argument("--trajectory", choices=c.trajectories())
    parser.add_argument("--endpoint", choices=(*c.endpoint_names(), "parent", "retained_full"))
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.command == "source":
        source.acquire()
        if not c.source_gate(source.verify()):
            raise RuntimeError("SOURCE_PREREQUISITE_FAILED")
    elif args.command == "train":
        if not args.trajectory or not c.source_gate(source.verify()):
            raise RuntimeError("Training requires source gate and registered trajectory")
        train.train(args.trajectory)
    elif args.command == "evaluate":
        if not args.endpoint:
            parser.error("evaluate requires --endpoint")
        evaluate(args.endpoint)
    else:
        {"freeze": freeze, "verify": verify, "pilot": pilot}[args.command]()
