"""Once-only, gated R98 acquisition and matched learner pilot. No release promotion."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

from . import core as c


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def freeze():
    if c.OUT.exists():
        raise RuntimeError("R98 output already exists; preserve it")
    if git(c.HOST, "rev-parse", "HEAD") != c.HOST_COMMIT or git(c.HOST, "status", "--porcelain"):
        raise RuntimeError("Exact clean LayerCake host required")
    if c.sha(c.PARENT / "model.safetensors") != c.PARENT_SHA:
        raise RuntimeError("Parent identity failed")
    train, dev = c.catalog("train"), c.catalog("development")
    if {r["prompt"] for r in train} & {r["prompt"] for r in dev}:
        raise RuntimeError("Split overlap")
    c.write_once(c.OUT / "train.json", train)
    c.write_once(c.OUT / "development.json", dev)
    paths = list(c.HERE.glob("*.py")) + [c.HERE / "PROTOCOL.md"]
    paths += [c.ROOT / "tests/test_plan_english_r98.py"]
    paths += list((c.ROOT / "abi").glob("*.py"))
    paths += [
        c.ROOT / p
        for p in (
            "experiments/factual_semantic_r16/public_qualification.py",
            "experiments/factual_semantic_r16/facts.py",
            "experiments/foreign_capability_r14/core.py",
            "experiments/preexisting_representation_r15b/public_qualification.py",
        )
    ]
    paths += [
        c.PARENT / name
        for name in (
            "metadata.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    ]
    paths += [c.OUT / "train.json", c.OUT / "development.json"]
    paths += [
        c.HOST / p
        for p in git(c.HOST, "ls-files", "layercake/*.py", "layercake/**/*.py").splitlines()
    ]
    files = {os.path.relpath(p, c.ROOT).replace("\\", "/"): c.sha(p) for p in paths}
    value = {
        "format": "abi-r98-freeze/1",
        "date": "2026-11-04",
        "files": files,
        "abi_base_commit": git(c.ROOT, "rev-parse", "HEAD"),
        "layercake_commit": c.HOST_COMMIT,
        "parent_sha256": c.PARENT_SHA,
        "source": {"model": c.SOURCE, "revision": c.REVISION},
        "train_rows": len(train),
        "development_rows": len(dev),
        "seed": c.SEED,
        "steps": c.STEPS,
        "batch": c.BATCH,
        "context": c.CONTEXT,
        "source_outputs_observed": 0,
        "model_training_performed": False,
        "full_moonshot": "NOT_PROVEN",
        "human_fluency": "UNASSESSED",
    }
    c.write_once(c.OUT / "freeze.json", value)
    print(
        json.dumps(
            {"frozen": str(c.OUT), "sha256": c.sha(c.OUT / "freeze.json"), "files": len(files)}
        ),
        flush=True,
    )


def load_parent(device="cuda"):
    import torch

    from abi.layercake_core_loader import load_layercake_core

    torch.manual_seed(c.SEED)
    torch.set_num_threads(4)
    model, tokenizer, metadata = load_layercake_core(c.PARENT, layercake_root=c.HOST, device=device)
    # Never enable inherited symbolic/guard/answer-span paths.
    if model.config.layers != 6 or model.config.max_tokens < c.CONTEXT:
        raise RuntimeError("Unexpected parent geometry")
    return model, tokenizer, metadata


def preflight():
    import torch
    from huggingface_hub import snapshot_download

    c.verify_binding()
    if not torch.cuda.is_available():
        raise RuntimeError("R98 requires CUDA acquisition")
    free = shutil.disk_usage(c.ROOT).free
    if free < 30 * 1024**3:
        raise RuntimeError(f"Disk preflight: only {free} bytes free")
    snapshot = snapshot_download(c.SOURCE, revision=c.REVISION, local_files_only=True)
    model, tokenizer, _ = load_parent()
    ids = torch.tensor(
        [tokenizer.encode("Please describe the supplied notes briefly.")], device="cuda"
    )
    route = torch.tensor([5], device="cuda")
    with torch.inference_mode():
        first = model(ids, task_routes=route, use_cache=True)
        next_id = first["logits"][:, -1].argmax(-1)[:, None]
        second = model(
            next_id, task_routes=route, past_key_values=first["past_key_values"], use_cache=True
        )
        full = model(torch.cat((ids, next_id), 1), task_routes=route, use_cache=False)
        error = float((second["logits"][:, -1] - full["logits"][:, -1]).abs().max())
    value = {
        "format": "abi-r98-parent-mechanics/1",
        "max_cached_logit_error": error,
        "tolerance": 1e-3,
        "parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "disk_free": free,
        "source_snapshot": snapshot,
        "scope": "existing ABI loader mechanics; no English or LayerCake release claim",
    }
    c.write_once(c.OUT / "preflight.json", value)
    if not error <= 1e-3:
        raise RuntimeError("INTEGRATION_BOUNDARY_FAILURE: cache mismatch; attribution unresolved")
    print(json.dumps(value), flush=True)


def acquire():
    import psutil
    import torch

    from experiments.factual_semantic_r16.public_qualification import _load_source, _render_chat

    c.verify_binding()
    destination = c.OUT / "source"
    destination.mkdir()
    start = time.perf_counter()
    torch.set_num_threads(4)
    process = psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model, snapshot = _load_source(c.SOURCE, c.REVISION)
    loaded = time.perf_counter() - start
    source_inventory = {
        str(p.relative_to(snapshot)): {"sha256": c.sha(p), "bytes": p.stat().st_size}
        for p in snapshot.rglob("*")
        if p.is_file()
    }
    observations, input_tokens = [], 0
    peak_rss = psutil.Process().memory_info().rss
    system = "Follow the user's English-language request. Use only the supplied fictional information. Return only the requested response, without commentary about the task."
    with (destination / "rows.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
        for i, row in enumerate(c.read(c.OUT / "train.json") + c.read(c.OUT / "development.json")):
            rendered = _render_chat(tokenizer, system, row["prompt"])
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
            before = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=144,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            tokens = generated[0, encoded.input_ids.shape[1] :].tolist()
            answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            observed = {
                "id": row["id"],
                "split": row["split"],
                "task": row["task"],
                "prompt": row["prompt"],
                "teacher_output": answer,
                "token_ids": tokens,
                "input_tokens": int(encoded.input_ids.numel()),
                "wall_seconds": time.perf_counter() - before,
                "truncated": len(tokens) == 144 and tokens[-1] != tokenizer.eos_token_id,
                "screen": c.screen(row, answer),
            }
            c.append(journal, observed)
            observations.append(observed)
            input_tokens += observed["input_tokens"]
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            if i == 0 or (i + 1) % 8 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "source",
                            "rows": i + 1,
                            "screen_passes": sum(
                                r["screen"]["mechanical_pass"] for r in observations
                            ),
                            "seconds": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
    value = {
        "format": "abi-r98-source/1",
        "rows_sha256": c.sha(destination / "rows.jsonl"),
        "summary": c.aggregate(observations),
        "source_gate": c.source_gate(observations),
        "accounting": {
            "raw_prompts": len(observations),
            "unique_prompt_utf8_bytes": sum(
                len(s.encode()) for s in {r["prompt"] for r in observations}
            ),
            "teacher_output_bytes": sum(len(r["teacher_output"].encode()) for r in observations),
            "unique_teacher_output_bytes": sum(
                len(s.encode()) for s in {r["teacher_output"] for r in observations}
            ),
            "teacher_generated_tokens": sum(len(r["token_ids"]) for r in observations),
            "teacher_input_tokens": input_tokens,
            "stored_logits": 0,
            "stored_activations": 0,
            "source_parameters_copied": 0,
            "source_parameters_loaded": sum(p.numel() for p in model.parameters()),
            "source_inference_seconds": sum(r["wall_seconds"] for r in observations),
            "wall_seconds": time.perf_counter() - start,
            "load_seconds": loaded,
            "cpu_process_seconds": sum(process.cpu_times()[:2]) - cpu_start,
            "peak_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "external_hardware": False,
            "source_files": source_inventory,
        },
    }
    c.write_once(destination / "result.json", value)
    print(
        json.dumps(
            {"stage": "source_complete", "summary": value["summary"], "gate": value["source_gate"]}
        ),
        flush=True,
    )


def source_rows():
    result = c.read(c.OUT / "source/result.json")
    path = c.OUT / "source/rows.jsonl"
    if c.sha(path) != result["rows_sha256"]:
        raise RuntimeError("Source evidence changed")
    observed = c.rows(path)
    catalogs = c.read(c.OUT / "train.json") + c.read(c.OUT / "development.json")
    if len(observed) != len(catalogs) or [r["id"] for r in observed] != [r["id"] for r in catalogs]:
        raise RuntimeError("Source rows missing, duplicated, or reordered")
    for row, reference in zip(observed, catalogs):
        if row["prompt"] != reference["prompt"] or row["screen"] != c.screen(
            reference, row["teacher_output"]
        ):
            raise RuntimeError("Source screen cannot be recomputed")
    if not c.source_gate(observed):
        raise RuntimeError("ABI_SOURCE_PREREQUISITE_FAILED")
    return observed


def prepare_trainable(model, arm):
    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER
    from abi.layercake_host import LoRAConv1D

    model.requires_grad_(False)
    wrappers = []
    if arm == "lora":
        for block in model.transformer.h:
            for owner, name in (
                (block.attn, "c_attn"),
                (block.attn, "c_proj"),
                (block.mlp, "c_fc"),
                (block.mlp, "c_proj"),
            ):
                wrapper = LoRAConv1D(getattr(owner, name), rank=32, alpha=64)
                setattr(owner, name, wrapper)
                wrappers.append((owner, name, wrapper))
    else:
        selected = {CAPABILITY_CAKE_ORDER.index(task) for task in c.TASKS}
        for route in selected:
            model.task_cakes[route].requires_grad_(True)
            for layer in model.capability_layer_adapters:
                layer[route].requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad], wrappers


def student(arm):
    import hashlib

    import psutil
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER

    c.verify_binding()
    source = {row["id"]: row for row in source_rows() if row["split"] == "train"}
    destination = c.OUT / arm
    destination.mkdir()
    model, tokenizer, metadata = load_parent()
    params, wrappers = prepare_trainable(model, arm)
    model.eval()  # fixed parent dropout; gradients remain enabled
    encoded = []
    for row in c.read(c.OUT / "train.json"):
        prefix = tokenizer.encode(c.model_prompt(row, arm), add_special_tokens=False)
        suffix = tokenizer.encode(
            c.target(row, source[row["id"]]["teacher_output"], arm), add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        if len(prefix) + len(suffix) > c.CONTEXT:
            raise RuntimeError(f"ABI_REPRESENTATION_OVERFLOW: {row['id']}; no truncation permitted")
        encoded.append((row["id"], prefix, suffix, CAPABILITY_CAKE_ORDER.index(row["task"])))
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(c.SEED)
    order = list(range(len(encoded)))
    sample_digest = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    start, steps, loss_tokens = time.perf_counter(), 0, 0
    cpu_start = sum(psutil.Process().cpu_times()[:2])
    peak_rss = psutil.Process().memory_info().rss
    with (destination / "training.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
        while steps < c.STEPS:
            if time.perf_counter() - start > 3600:
                raise RuntimeError("EXPERIMENT_BUDGET_STOP: training exceeded one hour")
            if steps * c.BATCH % len(order) == 0:
                rng.shuffle(order)
            indexes = [order[(steps * c.BATCH + j) % len(order)] for j in range(c.BATCH)]
            ids = torch.full(
                (c.BATCH, c.CONTEXT), tokenizer.eos_token_id, device="cuda", dtype=torch.long
            )
            labels = torch.full_like(ids, -100)
            attention = torch.zeros_like(ids)
            lengths, routes, batch_tokens = [], [], 0
            for j, idx in enumerate(indexes):
                record_id, prefix, suffix, route = encoded[idx]
                seq = prefix + suffix
                ids[j, : len(seq)] = torch.tensor(seq, device="cuda")
                labels[j, len(prefix) : len(seq)] = torch.tensor(suffix, device="cuda")
                attention[j, : len(seq)] = 1
                lengths.append(len(prefix))
                routes.append(route)
                sample_digest.update((record_id + "\n").encode())
                batch_tokens += len(suffix)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(
                    ids,
                    attention_mask=attention,
                    prompt_lengths=torch.tensor(lengths, device="cuda"),
                    task_routes=torch.tensor(routes, device="cuda"),
                    use_cache=False,
                )
                loss = F.cross_entropy(
                    output["logits"][:, :-1].float().reshape(-1, model.config.vocab_size),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("ABI_OPTIMIZATION_FAILURE: non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError(
                    "ABI_OPTIMIZATION_FAILURE: non-finite gradients; no unpaired retries"
                )
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            loss_tokens += batch_tokens
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            c.append(
                journal,
                {
                    "step": steps,
                    "loss": float(loss.detach()),
                    "loss_tokens": batch_tokens,
                    "sample_ids": [encoded[i][0] for i in indexes],
                    "elapsed_seconds": time.perf_counter() - start,
                },
            )
            if steps == 1 or steps % 32 == 0:
                print(
                    json.dumps(
                        {
                            "stage": arm,
                            "step": steps,
                            "loss": float(loss.detach()),
                            "seconds": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
            if steps % 128 == 0:
                save_file(
                    {
                        name: p.detach().cpu().contiguous()
                        for name, p in model.named_parameters()
                        if p.requires_grad
                    },
                    str(destination / f"trainable-step{steps}.safetensors"),
                )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # Conventional weight-LoRA is merged for deployment; no wrappers at inference.
    with torch.no_grad():
        for owner, name, wrapper in wrappers:
            wrapper.base.weight.add_((wrapper.lora_a @ wrapper.lora_b) * wrapper.scale)
            setattr(owner, name, wrapper.base)
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("Non-finite checkpoint")
    checkpoint = destination / "model.safetensors"
    save_file(state, str(checkpoint))
    # Minimal loader metadata; do not carry old scientific claims into a new model.
    export_metadata = {
        "architecture": metadata["architecture"],
        "decoding": metadata["decoding"],
        "checkpoint": {"sha256": c.sha(checkpoint)},
        "r98": {
            "arm": arm,
            "parent": c.PARENT_SHA,
            "scope": "uncertified pilot",
            "teacher_present_at_inference": False,
        },
    }
    c.write_once(destination / "metadata.json", export_metadata)
    for name in (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        shutil.copyfile(c.PARENT / name, destination / name)
    c.write_once(
        destination / "training_result.json",
        {
            "arm": arm,
            "successful_steps": steps,
            "sample_order_sha256": sample_digest.hexdigest(),
            "source_rows_sha256": c.sha(c.OUT / "source/rows.jsonl"),
            "training_rows": len(encoded),
            "unique_teacher_tokens_train": sum(len(r["token_ids"]) for r in source.values()),
            "training_log_sha256": c.sha(destination / "training.jsonl"),
            "checkpoint_sha256": c.sha(checkpoint),
            "loss_tokens": loss_tokens,
            "padded_token_positions": steps * c.BATCH * c.CONTEXT,
            "trainable_parameters": sum(p.numel() for p in params),
            "deployed_parameters": sum(p.numel() for p in model.parameters()),
            "imported_parent_parameters": 87710428,
            "parent_bytes": (c.PARENT / "model.safetensors").stat().st_size,
            "wall_seconds": elapsed,
            "cpu_process_seconds": sum(psutil.Process().cpu_times()[:2]) - cpu_start,
            "trainable_parameter_seconds": elapsed * sum(p.numel() for p in params),
            "deployed_parameter_seconds": elapsed * sum(p.numel() for p in model.parameters()),
            "peak_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "teacher_parameters_copied": 0,
            "human_fluency": "UNASSESSED",
            "scientific_certification": False,
        },
    )


def evaluate(arm):
    import torch

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core

    c.verify_binding()
    source_rows()
    destination = c.OUT / arm
    if arm == "parent":
        destination.mkdir()
        model, tokenizer, _ = load_parent()
    else:
        model, tokenizer, _ = load_layercake_core(destination, layercake_root=c.HOST, device="cuda")
    torch.set_num_threads(4)
    observations = []
    with (destination / "development.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
        for row in c.read(c.OUT / "development.json"):
            ids = torch.tensor(
                [tokenizer.encode(c.model_prompt(row, arm), add_special_tokens=False)],
                device="cuda",
            )
            route = torch.tensor([CAPABILITY_CAKE_ORDER.index(row["task"])], device="cuda")
            generated = []
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                first = model(ids, task_routes=route, use_cache=True)
                state = {
                    "past_key_values": first["past_key_values"],
                    "task_routes": route,
                    "next_logits": first["logits"][:, -1],
                    "generated_ids": ids[:, :0],
                }
                first_seconds = None
                for _ in range(min(c.MAX_NEW, model.config.max_tokens - ids.shape[1] - 1)):
                    selected = state["next_logits"].argmax(-1)
                    token = int(selected.item())
                    if token == tokenizer.eos_token_id:
                        break
                    generated.append(token)
                    if first_seconds is None:
                        torch.cuda.synchronize()
                        first_seconds = time.perf_counter() - start
                    _, state = model.decode_step(state, next_token=selected)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            answer, plan_valid = c.visible_response(raw, arm)
            scored = c.screen(row, answer)
            observed = {
                "id": row["id"],
                "task": row["task"],
                "arm": arm,
                "raw_output": raw,
                "output": answer,
                "token_ids": generated,
                "plan_valid": plan_valid,
                "screen": scored,
                "internal_first_token_seconds": first_seconds,
                "wall_seconds": elapsed,
                "visible_bytes_per_second": len(answer.encode()) / elapsed,
                "teacher_loaded": False,
            }
            c.append(journal, observed)
            observations.append(observed)
            if len(observations) % 8 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "evaluate",
                            "arm": arm,
                            "rows": len(observations),
                            "passes": c.aggregate(observations)["passing"],
                        }
                    ),
                    flush=True,
                )
    c.write_once(
        destination / "development_result.json",
        {
            "arm": arm,
            "summary": c.aggregate(observations),
            "rows_sha256": c.sha(destination / "development.jsonl"),
            "human_fluency": "UNASSESSED",
            "diagnostic_gpu_median_visible_bytes_per_second": statistics.median(
                r["visible_bytes_per_second"] for r in observations
            ),
            "optimized_cpu_speed_certified": False,
            "signed_package_acceptance": False,
        },
    )


def verify():
    c.verify_binding()
    source = source_rows()
    catalogs = c.read(c.OUT / "development.json")
    summaries, outputs, orders = {}, {}, []
    for arm in (*c.ARMS, "parent"):
        path = c.OUT / arm / "development.jsonl"
        observed = c.rows(path)
        if c.sha(path) != c.read(path.parent / "development_result.json")["rows_sha256"]:
            raise RuntimeError("Evaluation evidence hash mismatch")
        if [r["id"] for r in observed] != [r["id"] for r in catalogs]:
            raise RuntimeError("Evaluation row coverage mismatch")
        for row, reference in zip(observed, catalogs):
            answer, valid = c.visible_response(row["raw_output"], arm)
            if (
                row["output"] != answer
                or row["plan_valid"] != valid
                or row["screen"] != c.screen(reference, answer)
            ):
                raise RuntimeError("Unrecomputable evaluation")
        summaries[arm] = c.aggregate(observed)
        outputs[arm] = observed
        if arm != "parent":
            training = c.read(c.OUT / arm / "training_result.json")
            if c.sha(c.OUT / arm / "model.safetensors") != training["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint hash mismatch")
            log = c.rows(c.OUT / arm / "training.jsonl")
            if len(log) != c.STEPS or [r["step"] for r in log] != list(range(1, c.STEPS + 1)):
                raise RuntimeError("Incomplete training")
            if c.sha(c.OUT / arm / "training.jsonl") != training["training_log_sha256"]:
                raise RuntimeError("Training log changed")
            orders.append([r["sample_ids"] for r in log])
    if not all(order == orders[0] for order in orders):
        raise RuntimeError("Unmatched sample exposure")
    teacher = {r["id"]: r for r in source if r["split"] == "development"}
    comparisons = {}
    rng = random.Random(c.SEED)
    candidate = outputs["abi_plan"]
    for name in ("teacher", "parent", "distillation", "lora"):
        comparison = [teacher[r["id"]] for r in candidate] if name == "teacher" else outputs[name]
        delta = [
            int(a["screen"]["mechanical_pass"]) - int(b["screen"]["mechanical_pass"])
            for a, b in zip(candidate, comparison)
        ]
        boot = sorted(sum(rng.choice(delta) for _ in delta) / len(delta) for _ in range(10000))
        comparisons[name] = {"difference": statistics.mean(delta), "ci95": [boot[249], boot[9749]]}
    summary = summaries["abi_plan"]
    passed = (
        summary["passing"] >= 44
        and all(x["passing"] >= 5 for x in summary["per_task"].values())
        and summary["collapses"] == 0
        and all(r["plan_valid"] for r in candidate)
        and comparisons["teacher"]["ci95"][0] >= -0.10
        and comparisons["parent"]["ci95"][0] > 0
        and all(comparisons[name]["difference"] >= 0 for name in ("distillation", "lora"))
    )
    result = {
        "format": "abi-r98-pilot-decision/1",
        "verdict": "PILOT_CONTINUATION_ELIGIBLE" if passed else "ABI_ACQUISITION_PILOT_FAILED",
        "summaries": summaries,
        "comparisons": comparisons,
        "human_fluency": "UNASSESSED",
        "full_moonshot": "NOT_PROVEN",
        "layercake_regression_established": False,
        "scope": "Mechanical supplied-content proxies and matched acquisition pilot; no fluent-core certificate",
    }
    if not (c.OUT / "decision.json").exists():
        c.write_once(c.OUT / "decision.json", result)
    elif c.read(c.OUT / "decision.json") != result:
        raise RuntimeError("Existing decision is stale")
    print(json.dumps(result, indent=2), flush=True)


def pilot():
    c.verify_binding()
    start = time.perf_counter()
    arm_order = list(c.ARMS)
    random.Random(c.SEED).shuffle(arm_order)
    stages = [("preflight",), ("source",)] + [("train", "--arm", arm) for arm in arm_order]
    stages += [("evaluate", "--arm", arm) for arm in ("parent", *arm_order)] + [("verify",)]
    c.write_once(
        c.OUT / "launch.json",
        {
            "command": sys.argv,
            "pid": os.getpid(),
            "arm_order": arm_order,
            "stages": stages,
            "total_wall_limit_seconds": 21600,
            "start_unix": time.time(),
        },
    )
    for args in stages:
        remaining = 21600 - (time.perf_counter() - start)
        if remaining <= 0:
            raise RuntimeError("EXPERIMENT_BUDGET_STOP")
        try:
            completed = subprocess.run(
                [sys.executable, "-u", "-m", "experiments.plan_english_r98.run", *args],
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
        disk_bytes = sum(p.stat().st_size for p in c.OUT.rglob("*") if p.is_file())
        if disk_bytes > 20 * 1024**3:
            c.write_once(
                c.OUT / "stop.json",
                {"stage": args, "reason": "EXPERIMENT_DISK_BUDGET_STOP", "bytes": disk_bytes},
            )
            return
        if args[0] == "source":
            observed = c.rows(c.OUT / "source/rows.jsonl")
            if not c.source_gate(observed):
                c.write_once(
                    c.OUT / "stop.json",
                    {
                        "stage": "source",
                        "reason": "ABI_SOURCE_PREREQUISITE_FAILED",
                        "student_training_performed": False,
                        "layercake_quality_tested": False,
                        "summary": c.aggregate(observed),
                        "rows_sha256": c.sha(c.OUT / "source/rows.jsonl"),
                    },
                )
                print(
                    "ABI_SOURCE_PREREQUISITE_FAILED: preserved; student training blocked",
                    flush=True,
                )
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "preflight", "source", "train", "evaluate", "verify", "pilot")
    )
    parser.add_argument("--arm", choices=(*c.ARMS, "parent"))
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.command == "train":
        if args.arm not in c.ARMS:
            parser.error("Training requires a registered arm")
        student(args.arm)
    elif args.command == "evaluate":
        if args.arm is None:
            parser.error("Evaluation requires an arm")
        evaluate(args.arm)
    else:
        {
            "freeze": freeze,
            "preflight": preflight,
            "source": acquire,
            "verify": verify,
            "pilot": pilot,
        }[args.command]()


if __name__ == "__main__":
    main()
