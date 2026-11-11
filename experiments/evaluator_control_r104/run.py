"""Prospective, write-once R104 acquisition/interface diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import core as c
from .evaluator_cases import audit, require_audit


def snapshot():
    from huggingface_hub import snapshot_download

    result = Path(snapshot_download(c.SOURCE, revision=c.REVISION, local_files_only=True)).resolve()
    if result.name != c.REVISION:
        raise RuntimeError("Wrong control snapshot")
    return result


def inventory(folder):
    return {p.relative_to(folder).as_posix(): {"sha256": c.sha(p), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*")) if p.is_file()}


def tokenizer(arm):
    from transformers import AutoTokenizer

    folder = snapshot() if arm == "control" else c.checkpoint(arm)
    if arm == "control":
        frozen = c.read(c.OUT / "freeze.json")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            if c.sha(folder / name) != frozen["control_files"][name]["sha256"]:
                raise RuntimeError("Changed control tokenizer")
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def freeze():
    from transformers import AutoTokenizer

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER

    if c.OUT.exists():
        raise RuntimeError("Preserve prior R104 directory")
    evaluator = audit()
    if evaluator["mismatches"]:
        raise RuntimeError("Evaluator fails prerequisite fixtures")
    if shutil.disk_usage(c.ROOT).free < 20 * 1024**3:
        raise RuntimeError("Initial free disk below 20 GiB")
    if tuple(CAPABILITY_CAKE_ORDER) != c.ROUTE_ORDER:
        raise RuntimeError("Changed canonical routes")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.HOST, text=True).strip() != c.HOST_CHECKOUT:
        raise RuntimeError("Wrong host checkout")
    if subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=c.HOST, text=True).strip():
        raise RuntimeError("Dirty host checkout")
    control_files = inventory(snapshot())
    if control_files != c.read(c.ROOT / "results/coverage_exposure_r102/source/receipt.json")["source_inventory"]:
        raise RuntimeError("Pinned control differs from prior source identity")
    populations = {s: c.catalog(s) for s in ("qualification", "comparison")}
    all_rows = [r for records in populations.values() for r in records]
    prior_paths = [c.ROOT / f"results/{experiment}/{name}.json" for experiment, names in (
        ("coverage_exposure_r102", ("train", "development", "aliases")),
        ("acquisition_capacity_r100", ("development",)),
        ("grounded_english_r99_fp32", ("development",)),
        ("recipient_interface_r103", ("qualification", "comparison")),
    ) for name in names]
    old = c.base.catalog("train") + c.base.catalog("development")
    for path in prior_paths:
        values = c.read(path)
        old += [r["row"] for r in values] if path.name == "aliases.json" else values
    def extract(records):
        return {s for r in records for s in re.findall(r"\b(?:NOTE|Person)[A-Za-z0-9]+\b", r["prompt"])}
    if (len({r["id"] for r in all_rows}) != 96 or len({r["prompt"] for r in all_rows}) != 96
            or extract(all_rows) & extract(old)
            or extract(populations["qualification"]) & extract(populations["comparison"])):
        raise RuntimeError("Duplicate or overlapping prospective inputs")
    for arm in (*c.ARMS, "control"):
        tok = AutoTokenizer.from_pretrained(snapshot() if arm == "control" else c.checkpoint(arm), local_files_only=True, trust_remote_code=False)
        for ref in all_rows:
            for interface in c.INTERFACES:
                if len(tok.encode(c.render(ref, arm, interface, tok), add_special_tokens=False)) + c.MAX_NEW > c.CONTEXT:
                    raise RuntimeError("Registered generation budget exceeds context")
    for split, population in populations.items():
        c.write_once(c.OUT / f"{split}.json", population)
    c.write_once(c.OUT / "evaluator_audit.json", evaluator)
    tracked = subprocess.check_output(["git", "ls-files", "abi/*.py"], cwd=c.ROOT, text=True).splitlines()
    files = {name: c.sha(c.ROOT / name) for name in tracked}
    prior = c.read(c.ROOT / "results/execution_continuation_r101/freeze.json")["files"]
    files.update({name: digest for name, digest in prior.items() if name.startswith("../layercake_release/")})
    paths = [*c.HERE.glob("*.py"), c.HERE / "PROTOCOL.md", c.ROOT / "tests/test_evaluator_control_r104.py",
             c.ROOT / "experiments/plan_english_r98/core.py", *prior_paths,
             c.ROOT / "experiments/conditional_choice_host_r74/screen_candidate_v1.py",
             c.ROOT / "experiments/role_invariant_choice_r76/PROTOCOL.md",
             c.ROOT / "experiments/neural_host_reconstruction_r47/PROTOCOL.md",
             c.ROOT / "results/coverage_exposure_r102/decision.json",
             c.ROOT / "results/coverage_exposure_r102/source/receipt.json", *c.OUT.glob("*.json")]
    paths += [p for p in (c.ROOT / "experiments/__init__.py", c.ROOT / "experiments/plan_english_r98/__init__.py") if p.is_file()]
    for arm in c.ARMS:
        paths += [c.checkpoint(arm) / name for name in ("model.safetensors", "metadata.json", "tokenizer.json",
                                                     "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")]
    files.update({os.path.relpath(p, c.ROOT).replace("\\", "/"): c.sha(p) for p in paths})
    c.write_once(c.OUT / "freeze.json", {"format": "r104-recipient-interface-freeze/1", "files": files,
        "control_model": c.SOURCE, "control_revision": c.REVISION, "control_files": control_files,
        "host_checkout": c.HOST_CHECKOUT, "abi_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.ROOT, text=True).strip(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "psutil")},
        "budget_seconds": 5400, "maximum_control_calls": 144, "maximum_recipient_calls": 384,
        "new_optimizer_steps": 0, "new_copied_source_parameters": 0, "new_trained_bridge_parameters": 0,
        "new_imported_logits_or_activations": 0, "qualification_rows": 48, "comparison_rows": 48})
    c.binding()
    print(json.dumps({"freeze_sha256": c.sha(c.OUT / "freeze.json"), "inputs": len(files)}), flush=True)


def route_trace(model, route):
    trace = {"cakes": list(model.last_cake_calls),
             "adapters": [list(getattr(block, "_abi_last_deep_adapter_routes", ())) for block in model.transformer.h]}
    if trace != {"cakes": [route], "adapters": [[route]] * 6}:
        raise RuntimeError("Unlocalized integration finding: unexpected physical route")
    return trace


def recipient_generate(model, ids, route):
    import torch

    routes = torch.tensor([route], device="cuda")
    first = model(ids, task_routes=routes, use_cache=True)
    state = {"past_key_values": first["past_key_values"], "task_routes": routes,
             "next_logits": first["logits"][:, -1], "generated_ids": ids[:, :0]}
    tokens, traces = [], []
    for index in range(c.MAX_NEW):
        traces.append(route_trace(model, route))
        if not torch.isfinite(state["next_logits"]).all():
            raise RuntimeError("Nonfinite recipient logits")
        selected = state["next_logits"].argmax(-1)
        token = int(selected.item())
        tokens.append(token)
        if token == model._r104_eos or index + 1 == c.MAX_NEW:
            break
        model.decode_step(state, next_token=selected)
    return tokens, traces


def execute(arm, split):
    import psutil
    import torch
    from transformers import AutoModelForCausalLM

    from abi.layercake_core_loader import load_layercake_core

    frozen = c.binding()
    require_audit()
    if split == "comparison" and not c.qualified(verify_cell("control", "qualification")):
        raise RuntimeError("Control qualification required")
    if split == "qualification" and arm != "control":
        raise ValueError("Qualification is control-only")
    folder = c.OUT / split / arm
    folder.mkdir(parents=True)
    started, process = time.perf_counter(), psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    torch.set_num_threads(4)
    torch.manual_seed(c.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tok = tokenizer(arm)
    if arm == "control":
        if inventory(snapshot()) != frozen["control_files"]:
            raise RuntimeError("Control weights changed")
        model = AutoModelForCausalLM.from_pretrained(snapshot(), local_files_only=True, trust_remote_code=False,
                                                    dtype=torch.float16).to("cuda")
    else:
        model, loaded_tok, _ = load_layercake_core(c.checkpoint(arm), layercake_root=c.HOST, device="cuda")
        if tok.get_vocab() != loaded_tok.get_vocab() or any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError("Recipient tokenizer or precision changed")
        model._r104_eos = tok.eos_token_id
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    count = sum(p.numel() for p in model.parameters())
    if arm != "control" and count != 87710428:
        raise RuntimeError("Recipient parameter identity changed")
    population = c.read(c.OUT / f"{split}.json")
    interfaces = ("native",) if split == "qualification" else c.INTERFACES
    peak_rss = process.memory_info().rss
    path = folder / "raw.jsonl"
    with torch.inference_mode(), path.open("x", encoding="utf-8") as stream:
        for index, ref in enumerate(population):
            for interface in interfaces:
                if time.perf_counter() - started > 1500:
                    raise RuntimeError("Per-process time budget exceeded")
                prompt = c.render(ref, arm, interface, tok)
                input_ids = tok.encode(prompt, add_special_tokens=False)
                if len(input_ids) + c.MAX_NEW > c.CONTEXT:
                    raise RuntimeError("Generation context changed")
                ids = torch.tensor([input_ids], device="cuda")
                torch.cuda.synchronize()
                before = time.perf_counter()
                if arm == "control":
                    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                        max_new_tokens=c.MAX_NEW, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                    tokens, traces = generated[0, len(input_ids):].tolist(), None
                else:
                    tokens, traces = recipient_generate(model, ids, c.ROUTE_ORDER.index(ref["task"]))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                output = tok.decode(tokens, skip_special_tokens=True).strip()
                c.append(stream, {"id": ref["id"], "task": ref["task"], "arm": arm, "interface": interface,
                    "route": ref["task"], "prompt": prompt, "prompt_ids": input_ids, "token_ids": tokens,
                    "output": output, "screen": c.screen(ref, output), "legacy_screen": c.base.screen(ref, output), "output_utf8_bytes": len(output.encode()),
                    "output_characters": len(output), "stop_reason": "eos" if tokens[-1] == tok.eos_token_id else "budget",
                    "wall_seconds": elapsed, "execution_routes": traces})
                peak_rss = max(peak_rss, process.memory_info().rss)
            if (index + 1) % 8 == 0:
                print(json.dumps({"arm": arm, "split": split, "prompts": index + 1}), flush=True)
    c.binding()
    if arm == "control" and inventory(snapshot()) != frozen["control_files"]:
        raise RuntimeError("Control files changed during execution")
    observed = c.rows(path)
    c.validate(observed, population, arm, interfaces, tok)
    c.write_once(folder / "receipt.json", {"raw_sha256": c.sha(path), "arm": arm, "split": split,
        "checkpoint_sha256": None if arm == "control" else c.sha(c.checkpoint(arm) / "model.safetensors"),
        "control_inventory_sha256": hashlib.sha256(c.base.canonical(frozen["control_files"])).hexdigest() if arm == "control" else None,
        "parameter_count": count, "precision": "float16" if arm == "control" else "float32",
        "calls": len(observed), "wall_seconds": time.perf_counter()-started,
        "cpu_process_seconds": sum(process.cpu_times()[:2])-cpu_start, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "sampled_peak_rss_bytes": peak_rss, "device": torch.cuda.get_device_name(), "optimizer_steps": 0})


def verify_cell(arm, split):
    folder = c.OUT / split / arm
    receipt = c.read(folder / "receipt.json")
    path = folder / "raw.jsonl"
    expected_hash = None if arm == "control" else c.sha(c.checkpoint(arm) / "model.safetensors")
    if (receipt["raw_sha256"] != c.sha(path) or receipt["checkpoint_sha256"] != expected_hash
            or receipt["arm"] != arm or receipt["split"] != split or receipt["optimizer_steps"] != 0
            or receipt["precision"] != ("float16" if arm == "control" else "float32")):
        raise RuntimeError("Missing or stale execution receipt")
    observed = c.rows(path)
    if receipt["calls"] != len(observed) or (arm != "control" and receipt["parameter_count"] != 87710428):
        raise RuntimeError("Wrong execution coverage")
    if arm == "control" and receipt["control_inventory_sha256"] != hashlib.sha256(c.base.canonical(c.read(c.OUT / "freeze.json")["control_files"])).hexdigest():
        raise RuntimeError("Wrong control weights inventory")
    for key in ("wall_seconds", "cpu_process_seconds", "peak_cuda_allocated_bytes", "sampled_peak_rss_bytes", "parameter_count"):
        if not math.isfinite(receipt[key]) or receipt[key] <= 0:
            raise RuntimeError("Invalid measured resource receipt")
    if sum(r["wall_seconds"] for r in observed) > receipt["wall_seconds"]:
        raise RuntimeError("Generation time exceeds complete process wall")
    c.validate(observed, c.read(c.OUT / f"{split}.json"), arm,
               ("native",) if split == "qualification" else c.INTERFACES, tokenizer(arm))
    return observed


def verify():
    c.binding()
    evaluator = require_audit()
    if (c.OUT / "stop.json").exists():
        raise RuntimeError("Interrupted experiment cannot be certified complete")
    if any(c.read(c.OUT / f"{s}.json") != c.catalog(s) for s in ("qualification", "comparison")):
        raise RuntimeError("Changed prospective population")
    qualification = verify_cell("control", "qualification")
    value = {"format": "r104-decision/1", "qualification": c.summarize(qualification),
             "evaluator_cases": len(evaluator["cases"]), "evaluator_mismatches": evaluator["mismatches"],
             "full_moonshot": "NOT_PROVEN", "native_layercake_defect": "NOT_ESTABLISHED", "product_promotion": "NONE"}
    if not c.qualified(qualification):
        if (c.OUT / "comparison").exists():
            raise RuntimeError("Comparisons ran without a qualified control")
        value["verdict"] = "CONTROL_QUALIFICATION_FAILED"
    else:
        observations = {arm: verify_cell(arm, "comparison") for arm in ("control", *c.ARMS)}
        cells = {f"{arm}_{interface}": [r for r in observed if r["interface"] == interface]
                 for arm, observed in observations.items() for interface in c.INTERFACES}
        reference = cells["control_native"]
        comparisons = {arm: c.paired(cells[f"{arm}_native"], cells[f"{arm}_benchmark"]) for arm in observations}
        relative = {name: c.paired(observed, reference) for name, observed in cells.items()}
        value.update({"verdict": "BOUNDED_RECIPIENT_INTERFACE_DIAGNOSTIC_COMPLETED",
            "cells": {name: c.summarize(observed) for name, observed in cells.items()}, "interface_contrasts": comparisons,
            "control_relative": relative, "comparison_control_qualified": c.qualified(reference),
            "primary_interface_sensitivity": {arm: comparisons[arm]["ci97_5"][0] > 0 for arm in ("parent", "full_model")},
            "recipient_mechanical_gates": {name: c.qualified(reference) and c.qualified(observed) and relative[name]["ci95"][0] >= -.10
                                            for name, observed in cells.items() if not name.startswith("control_")}})
    path = c.OUT / "decision.json"
    if path.exists():
        if c.read(path) != value:
            raise RuntimeError("Stale scientific decision")
    else:
        c.write_once(path, value)
    print(json.dumps({"verdict": value["verdict"], "qualification": value["qualification"]["summary"],
                      "decision_sha256": c.sha(path)}), flush=True)
    return value


def pilot():
    c.binding()
    c.write_once(c.OUT / "launch.json", {"pid": os.getpid(), "start_unix": time.time(), "budget_seconds": 5400})
    started = time.monotonic()
    stages = [("control", "qualification"), ("control", "comparison"), *[(a, "comparison") for a in c.ARMS]]
    for index, (arm, split) in enumerate(stages):
        try:
            if index == 1 and not c.qualified(verify_cell("control", "qualification")):
                verify()
                return
            remaining = 5400 - (time.monotonic()-started)
            if remaining <= 0:
                raise RuntimeError("Total resource time budget exceeded")
            print(json.dumps({"starting": [arm, split]}), flush=True)
            done = subprocess.run([sys.executable, "-u", "-m", "experiments.evaluator_control_r104.run", "execute", "--arm", arm, "--split", split],
                                  cwd=c.ROOT, capture_output=True, text=True, timeout=min(1500, remaining))
            c.write_once(c.OUT / "operations" / f"stage-{index:02d}.json", {"arm": arm, "split": split,
                         "returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr})
            print(done.stdout[-1800:], flush=True)
            if done.returncode:
                raise RuntimeError("Model execution failed")
            if sum(p.stat().st_size for p in c.OUT.rglob("*") if p.is_file()) > 1024**3:
                raise RuntimeError("New evidence disk budget exceeded")
        except Exception as error:
            c.write_once(c.OUT / "stop.json", {"stage": [arm, split], "error": repr(error), "native_layercake_defect": "NOT_ESTABLISHED"})
            raise
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "execute", "verify", "pilot"))
    parser.add_argument("--arm", choices=(*c.ARMS, "control"))
    parser.add_argument("--split", choices=("qualification", "comparison"))
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.command == "execute":
        if not args.arm or not args.split:
            parser.error("execute requires arm and split")
        execute(args.arm, args.split)
    else:
        {"freeze": freeze, "verify": verify, "pilot": pilot}[args.command]()
