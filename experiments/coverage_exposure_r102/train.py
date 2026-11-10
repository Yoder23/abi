"""Four fixed trajectories; intermediate checkpoints never perturb optimization."""

from __future__ import annotations

import shutil
import time

from experiments.plan_english_r98 import run as engine

from . import core as c


def merged_state(model):
    """Fold LoRA into detached canonical tensors without changing the live model."""
    import torch

    with torch.no_grad():
        wrappers = {name: module for name, module in model.named_modules()
                    if hasattr(module, "lora_a") and hasattr(module, "lora_b") and hasattr(module, "base")}
        state = {}
        for name, value in model.state_dict().items():
            if any(name.startswith(prefix + ".") for prefix in wrappers):
                continue
            state[name] = value.detach().cpu().contiguous().clone()
        for prefix, module in wrappers.items():
            state[prefix + ".weight"] = (module.base.weight + (module.lora_a @ module.lora_b) * module.scale).detach().cpu().contiguous()
            if module.base.bias is not None:
                state[prefix + ".bias"] = module.base.bias.detach().cpu().contiguous().clone()
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("Nonfinite deployed checkpoint")
    return state


def train(trajectory):
    import psutil
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER

    c.binding()
    method, policy = trajectory.split("_")
    if method not in c.METHODS or policy not in c.POLICIES:
        raise RuntimeError("Unregistered trajectory")
    folder = c.OUT / "training" / trajectory
    folder.mkdir(parents=True)
    engine.c = c
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, metadata = engine.load_parent()
    inherited_parameters = sum(p.numel() for p in model.parameters())
    if inherited_parameters != 87710428:
        raise RuntimeError("Unexpected inherited model size")
    params, _ = engine.prepare_trainable(model, method)
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Expected FP32 master state")
    source = {r["id"]: r["teacher_output"] for r in c.rows(c.previous.OUT / "source/rows.jsonl")}
    aliases = {(r["original_id"], r["alias"]): r for r in c.read(c.OUT / "aliases.json")}
    train_rows = c.read(c.OUT / "train.json")
    encoded = {}
    for index, row in enumerate(train_rows):
        for alias in range(4):
            record = aliases[(row["id"], alias)] if policy == "augment" else {"row": row, "target": source[row["id"]]}
            prefix = tokenizer.encode(c.model_prompt(record["row"], method), add_special_tokens=False)
            suffix = tokenizer.encode(record["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(prefix) + len(suffix) > c.CONTEXT:
                raise RuntimeError("Training context overflow")
            encoded[index, alias] = (prefix, suffix, CAPABILITY_CAKE_ORDER.index(row["task"]))
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    process = psutil.Process()
    start, cpu_start = time.perf_counter(), sum(process.cpu_times()[:2])
    torch.cuda.reset_peak_memory_stats()
    peak_rss, loss_tokens, save_seconds = process.memory_info().rss, 0, 0.0
    with (folder / "training.jsonl").open("x", encoding="utf-8") as journal:
        for scheduled in c.read(c.OUT / "schedule.json"):
            if time.perf_counter() - start > 1800:
                raise RuntimeError("R102 training trajectory time budget")
            step, alias = scheduled["step"], scheduled["epoch"] % 4
            ids = torch.full((c.BATCH, c.CONTEXT), tokenizer.eos_token_id, device="cuda", dtype=torch.long)
            labels, attention = torch.full_like(ids, -100), torch.zeros_like(ids)
            lengths, routes, batch_tokens = [], [], 0
            for j, index in enumerate(scheduled["indexes"]):
                prefix, suffix, route = encoded[index, alias]
                seq = prefix + suffix
                ids[j, :len(seq)] = torch.tensor(seq, device="cuda")
                labels[j, len(prefix):len(seq)] = torch.tensor(suffix, device="cuda")
                attention[j, :len(seq)] = 1
                lengths.append(len(prefix))
                routes.append(route)
                batch_tokens += len(suffix)
            optimizer.zero_grad(set_to_none=True)
            output = model(ids, attention_mask=attention, prompt_lengths=torch.tensor(lengths, device="cuda"),
                           task_routes=torch.tensor(routes, device="cuda"), use_cache=False)
            loss = F.cross_entropy(output["logits"][:, :-1].float().reshape(-1, model.config.vocab_size),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite FP32 loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite FP32 gradient")
            optimizer.step()
            loss_tokens += batch_tokens
            peak_rss = max(peak_rss, process.memory_info().rss)
            observation = {"step": step, "epoch": scheduled["epoch"], "alias": alias if policy == "augment" else None,
                           "sample_ids": [train_rows[i]["id"] for i in scheduled["indexes"]],
                           "prompt_tokens": lengths, "loss_tokens": batch_tokens, "loss": float(loss.detach()),
                           "gradient_norm": float(norm), "elapsed_seconds": time.perf_counter()-start}
            c.append(journal, observation)
            if step == 1 or step % 128 == 0:
                print(c.previous.canonical({"trajectory": trajectory, **observation}).decode().strip(), flush=True)
            if step in c.ENDPOINTS:
                before_save = time.perf_counter()
                destination = c.OUT / "checkpoints" / f"{trajectory}_s{step}"
                destination.mkdir(parents=True)
                save_file(merged_state(model), str(destination / "model.safetensors"))
                c.write_once(destination / "metadata.json", {
                    "architecture": metadata["architecture"], "decoding": metadata["decoding"],
                    "checkpoint": {"sha256": c.sha(destination / "model.safetensors")},
                    "r102": {"trajectory": trajectory, "step": step, "scope": "diagnostic; not a certified English core"},
                })
                for name in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
                    shutil.copyfile(c.PARENT / name, destination / name)
                save_seconds += time.perf_counter()-before_save
                c.write_once(destination / "training_receipt.json", {
                    "trajectory": trajectory, "step": step, "seed": c.SEED,
                    "checkpoint_sha256": c.sha(destination / "model.safetensors"),
                    "loss_tokens": loss_tokens, "padded_positions": step*c.BATCH*c.CONTEXT,
                    "trainable_parameters": sum(p.numel() for p in params), "inherited_parameters": inherited_parameters,
                    "wall_seconds": time.perf_counter()-start, "checkpoint_save_seconds": save_seconds,
                    "cpu_process_seconds": sum(process.cpu_times()[:2])-cpu_start,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "sampled_peak_rss_bytes": peak_rss,
                    "precision": "float32", "teacher_parameters_copied": 0,
                })
    c.write_once(folder / "receipt.json", {"rows_sha256": c.sha(folder / "training.jsonl"), "steps": step,
                                         "wall_seconds": time.perf_counter()-start, "device": torch.cuda.get_device_name()})
