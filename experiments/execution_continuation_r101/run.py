"""Write-once GPU diagnostic with immutable checkpoints and raw parity logits."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time

from . import core as c


def history(arm):
    train = (c.old.OUT / arm / "training_generation.jsonl" if arm == "full_trunk"
             else c.fit_audit.OUT / arm / "training_generation.jsonl")
    return c.rows(train) + c.rows(c.old.OUT / arm / "development.jsonl")


def freeze():
    c.old.verify_binding()
    if c.OUT.exists():
        raise RuntimeError("Preserve prior R101 evidence")
    if shutil.disk_usage(c.ROOT).free < 30 * 1024**3:
        raise RuntimeError("R101 initial disk budget")
    population = c.catalog()
    if len(population) != 72 or len({r["id"] for r in population}) != 72:
        raise RuntimeError("Incomplete diagnostic population")
    c.write_once(c.OUT / "catalog.json", population)
    paths = list(c.HERE.glob("*.py")) + [c.HERE / "PROTOCOL.md",
        c.ROOT / "tests/test_execution_continuation_r101.py", c.OUT / "catalog.json",
        c.old.OUT / "source/rows.jsonl", c.old.OUT / "decision.json",
        c.old.OUT / "freeze.json", c.fit_audit.OUT / "training_selection.json"]
    for arm in c.ARMS:
        paths += [c.old.checkpoint(arm) / n for n in
                  ("model.safetensors", "metadata.json", "tokenizer.json", "tokenizer_config.json",
                   "vocab.json", "merges.txt", "special_tokens_map.json")]
        paths += [c.old.OUT / arm / "development.jsonl"]
        paths += [(c.old.OUT if arm == "full_trunk" else c.fit_audit.OUT)
                  / arm / "training_generation.jsonl"]
    files = dict(c.read(c.old.OUT / "freeze.json")["files"])
    files.update({os.path.relpath(p, c.ROOT).replace("\\", "/"): c.sha(p) for p in paths})
    c.write_once(c.OUT / "freeze.json", {
        "format": "r101-execution-continuation-freeze/1", "files": files,
        "abi_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.ROOT, text=True).strip(),
        "parity_ids": c.parity_ids(population), "arms": c.ARMS, "cuts": c.CUTS,
        "optimizer_steps": 0, "new_teacher_queries": 0, "budget_seconds": 5400,
    })
    print(json.dumps({"freeze_sha256": c.sha(c.OUT / "freeze.json"), "inputs": len(files)}))


def state_from_prefill(model, ids, route):
    first = model(ids, task_routes=route, use_cache=True)
    return {"past_key_values": first["past_key_values"], "task_routes": route,
            "next_logits": first["logits"][:, -1], "generated_ids": ids[:, :0]}


def generate(model, tokenizer, prefix, gold_prefix, route):
    import torch

    ids = torch.tensor([prefix], device="cuda")
    state = state_from_prefill(model, ids, route)
    for token in gold_prefix:
        model.decode_step(state, next_token=torch.tensor([token], device="cuda"))
    budget = min(c.MAX_NEW, c.CONTEXT - len(prefix) - 1) - len(gold_prefix)
    if budget <= 0:
        raise RuntimeError("No continuation budget")
    generated = []
    for _ in range(budget):
        selected = state["next_logits"].argmax(-1)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            return generated, "eos"
        generated.append(token)
        model.decode_step(state, next_token=selected)
    return generated, "budget"


def audit(arm):
    import psutil
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core

    frozen = c.binding()
    directory = c.OUT / arm
    directory.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(c.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started, cpu_start = time.perf_counter(), sum(psutil.Process().cpu_times()[:2])
    model, tokenizer, _ = load_layercake_core(c.old.checkpoint(arm), layercake_root=c.HOST, device="cuda")
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("FP32 model required")
    torch.cuda.reset_peak_memory_stats()
    peak_rss = psutil.Process().memory_info().rss
    population = c.read(c.OUT / "catalog.json")
    source = {r["id"]: r["teacher_output"] for r in c.rows(c.old.OUT / "source/rows.jsonl")}
    retained = {r["id"]: r for r in history(arm)}
    if set(retained) != {r["id"] for r in population}:
        raise RuntimeError("Missing historical replay controls")
    encoded = []
    for row in population:
        prefix = tokenizer.encode(c.old.model_prompt(row, arm), add_special_tokens=False)
        target = source[row["id"]]
        target_encoding = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
        suffix = target_encoding.input_ids + [tokenizer.eos_token_id]
        if len(prefix) + len(suffix) > c.CONTEXT:
            raise RuntimeError("Target exceeds context")
        encoded.append((prefix, suffix, CAPABILITY_CAKE_ORDER.index(row["task"]), target_encoding))
    tensors, counts = {}, {}
    with torch.inference_mode():
        for index, row in enumerate(population):
            if row["id"] not in frozen["parity_ids"]:
                continue
            prefix, suffix, route_id, _ = encoded[index]
            route = torch.tensor([route_id], device="cuda")
            ids = torch.tensor([prefix + suffix], device="cuda")
            count = min(c.PARITY_TOKENS, len(suffix))
            counts[row["id"]] = count
            full = model(ids, prompt_lengths=torch.tensor([len(prefix)], device="cuda"),
                         task_routes=route, use_cache=False)["logits"]
            tensors[row["id"] + ".full"] = full[0, len(prefix)-1:len(prefix)-1+count].cpu().contiguous()
            del full
            batch = torch.full((2, c.CONTEXT), tokenizer.eos_token_id, device="cuda", dtype=torch.long)
            attention = torch.zeros_like(batch)
            lengths, routes = [], []
            for j, triple in enumerate((encoded[index], encoded[(index + 1) % len(encoded)])):
                other_prefix, other_suffix, other_route, _ = triple
                seq = other_prefix + other_suffix
                batch[j, :len(seq)] = torch.tensor(seq, device="cuda")
                attention[j, :len(seq)] = 1
                lengths.append(len(other_prefix))
                routes.append(other_route)
            padded = model(batch, attention_mask=attention,
                           prompt_lengths=torch.tensor(lengths, device="cuda"),
                           task_routes=torch.tensor(routes, device="cuda"), use_cache=False)["logits"]
            tensors[row["id"] + ".padded"] = padded[0, len(prefix)-1:len(prefix)-1+count].cpu().contiguous()
            del padded
            state = state_from_prefill(model, ids[:, :len(prefix)], route)
            cached = []
            for pos in range(count):
                cached.append(state["next_logits"][0].cpu())
                if pos + 1 < count:
                    model.decode_step(state, next_token=torch.tensor([suffix[pos]], device="cuda"))
            tensors[row["id"] + ".cached"] = torch.stack(cached).contiguous()
        parity_path = directory / "parity.safetensors"
        save_file(tensors, str(parity_path))
        c.write_once(directory / "parity_index.json", {"ids": frozen["parity_ids"], "counts": counts,
                     "vocab_size": len(tokenizer), "sha256": c.sha(parity_path)})
        parity = c.parity_metrics(tensors, frozen["parity_ids"], counts, len(tokenizer))
        c.write_once(directory / "parity_result.json", parity)
        if any(p["tolerance_violations"] or p["argmax_disagreements"] for p in parity.values()):
            raise RuntimeError("EXECUTION_PATH_INCONSISTENCY_REQUIRES_LOCALIZATION")
        del tensors
        print(json.dumps({"arm": arm, "parity": parity}), flush=True)
        with (directory / "forced.jsonl").open("x", encoding="utf-8") as forced, (
            directory / "generation.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(population):
                if time.perf_counter() - started > 1200:
                    raise RuntimeError("R101 arm time limit")
                prefix, suffix, route_id, target_encoding = encoded[index]
                route = torch.tensor([route_id], device="cuda")
                ids = torch.tensor([prefix + suffix], device="cuda")
                logits = model(ids, prompt_lengths=torch.tensor([len(prefix)], device="cuda"),
                               task_routes=route, use_cache=False)["logits"][0, len(prefix)-1:-1]
                nll = F.cross_entropy(logits, torch.tensor(suffix, device="cuda"), reduction="none")
                if not torch.isfinite(nll).all():
                    raise RuntimeError("Nonfinite reference NLL")
                masks = c.fit_audit.token_masks(row, source[row["id"]], target_encoding.offset_mapping, arm)
                c.append(forced, {"id": row["id"], "task": row["task"], "split": row["split"],
                    "target_ids": suffix, "argmax_ids": logits.argmax(-1).tolist(), "nll": nll.tolist(),
                    "response_mask": masks[0], "fact_mask": masks[1]})
                del logits, nll
                for percent in c.CUTS:
                    cut = (len(suffix) - 1) * percent // 100
                    supplied = suffix[:cut]
                    generated, stop = generate(model, tokenizer, prefix, supplied, route)
                    output = tokenizer.decode(supplied + generated, skip_special_tokens=True)
                    record = {"id": row["id"], "task": row["task"], "split": row["split"],
                        "cut_percent": percent, "supplied_ids": supplied, "generated_ids": generated,
                        "stop_reason": stop, "output": output,
                        "tail": tokenizer.decode(generated, skip_special_tokens=True),
                        "eligible_literals": c.eligible_literals(row, source[row["id"]],
                                                                target_encoding.offset_mapping, cut),
                        "screen": c.old.screen(row, output)}
                    c.append(stream, record)
                    if percent == 0 and generated != retained[row["id"]]["token_ids"]:
                        raise RuntimeError(f"Historical unassisted live replay differs: {row['id']}")
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                if (index + 1) % 8 == 0:
                    print(json.dumps({"arm": arm, "completed": index + 1}), flush=True)
    torch.cuda.synchronize()
    c.write_once(directory / "receipt.json", {
        "arm": arm, "checkpoint_sha256": c.sha(c.old.checkpoint(arm) / "model.safetensors"),
        "files": {p.name: c.sha(p) for p in directory.iterdir() if p.is_file()},
        "wall_seconds": time.perf_counter() - started,
        "cpu_process_seconds": sum(psutil.Process().cpu_times()[:2]) - cpu_start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "sampled_peak_rss_bytes": peak_rss,
        "device": torch.cuda.get_device_name(), "precision": "float32", "optimizer_steps": 0,
        "teacher_queries": 0, "inherited_parameters": sum(p.numel() for p in model.parameters()),
        "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "safetensors", "tokenizers")},
    })


def verify():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    frozen = c.binding()
    tokenizer = AutoTokenizer.from_pretrained(c.PARENT, local_files_only=True)
    population = c.read(c.OUT / "catalog.json")
    if population != c.catalog() or frozen["parity_ids"] != c.parity_ids(population):
        raise RuntimeError("Population changed")
    source = {r["id"]: r["teacher_output"] for r in c.rows(c.old.OUT / "source/rows.jsonl")}
    results, parity_results = {}, {}
    for arm in c.ARMS:
        directory = c.OUT / arm
        receipt = c.read(directory / "receipt.json")
        names = {"parity.safetensors", "parity_index.json", "parity_result.json", "forced.jsonl", "generation.jsonl"}
        if set(receipt["files"]) != names:
            raise RuntimeError("Missing raw artifact hashes")
        for name, digest in receipt["files"].items():
            if c.sha(directory / name) != digest:
                raise RuntimeError("Raw R101 evidence changed")
        if receipt["checkpoint_sha256"] != c.sha(c.old.checkpoint(arm) / "model.safetensors"):
            raise RuntimeError("Checkpoint changed")
        index = c.read(directory / "parity_index.json")
        counts = {key: min(c.PARITY_TOKENS, len(tokenizer.encode(source[key], add_special_tokens=False)) + 1)
                  for key in frozen["parity_ids"]}
        if index != {"ids": frozen["parity_ids"], "counts": counts,
                     "vocab_size": len(tokenizer), "sha256": c.sha(directory / "parity.safetensors")}:
            raise RuntimeError("Unrecomputable parity index")
        parity = c.parity_metrics(load_file(str(directory / "parity.safetensors")), index["ids"], counts, len(tokenizer))
        if parity != c.read(directory / "parity_result.json"):
            raise RuntimeError("Stale parity result")
        if any(p["tolerance_violations"] or p["argmax_disagreements"] for p in parity.values()):
            raise RuntimeError("Execution-path prerequisite failed")
        parity_results[arm] = parity
        forced, generations = c.rows(directory / "forced.jsonl"), c.rows(directory / "generation.jsonl")
        c.validate_coverage(forced, generations, population)
        retained = {r["id"]: r for r in history(arm)}
        for i, ref in enumerate(population):
            c.validate_forced(forced[i], ref, source[ref["id"]], tokenizer)
            for j, cut in enumerate(c.CUTS):
                observed = generations[i * len(c.CUTS) + j]
                c.validate_generation(observed, ref, source[ref["id"]], tokenizer, cut)
                if cut == 0 and observed["generated_ids"] != retained[ref["id"]]["token_ids"]:
                    raise RuntimeError("Live unassisted replay mismatch")
        results[arm] = c.summarize(forced, generations)
    primary = results["full_trunk"]["development"]["interventions"]["50"]["paired_remaining_coverage"]
    value = {"format": "r101-diagnostic-decision/1", "results": results, "parity": parity_results,
             "primary_contrast": primary,
             "primary_positive_history_sensitivity": primary["ci95"] is not None and primary["ci95"][0] > 0,
             "verdict": "BOUNDED_DIAGNOSTIC_COMPLETED_NO_PRODUCT_PROMOTION",
             "unresolved": ["data_sufficiency", "optimization_adequacy", "parent_suitability", "broad_English"],
             "full_moonshot": "NOT_PROVEN"}
    path = c.OUT / "decision.json"
    if path.exists():
        if c.read(path) != value:
            raise RuntimeError("Stale R101 decision")
    else:
        c.write_once(path, value)
    print(json.dumps({"verdict": value["verdict"], "primary_contrast": primary,
                      "decision_sha256": c.sha(path)}), flush=True)


def pilot():
    c.binding()
    c.write_once(c.OUT / "launch.json", {"pid": os.getpid(), "start_unix": time.time(),
                 "budget_seconds": 5400, "commands": [["audit", "--arm", a] for a in c.ARMS] + [["verify"]]})
    started = time.monotonic()
    for args in [*(("audit", "--arm", a) for a in c.ARMS), ("verify",)]:
        try:
            done = subprocess.run([sys.executable, "-u", "-m", "experiments.execution_continuation_r101.run", *args],
                                  cwd=c.ROOT, timeout=min(1250, max(1, 5400 - (time.monotonic() - started))))
            if done.returncode:
                raise RuntimeError(f"Diagnostic subprocess failed: {done.returncode}")
            if sum(p.stat().st_size for p in c.OUT.rglob("*") if p.is_file()) > 2 * 1024**3:
                raise RuntimeError("R101 disk budget")
        except Exception as error:
            c.write_once(c.OUT / "stop.json", {"stage": args, "error": repr(error),
                         "native_layercake_defect": "NOT_ESTABLISHED", "scientific_promotion": False})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "audit", "verify", "pilot"))
    parser.add_argument("--arm", choices=c.ARMS)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.command == "audit":
        if args.arm is None:
            parser.error("audit requires --arm")
        audit(args.arm)
    else:
        {"freeze": freeze, "verify": verify, "pilot": pilot}[args.command]()
