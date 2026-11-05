"""R98 training algorithm with only the registered FP32 numerical repair."""

from __future__ import annotations

import hashlib
import random
import shutil
import time

from experiments.plan_english_r98 import run as engine

from . import core as c


def student(arm):
    import psutil
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER

    c.verify_binding()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = {r["id"]: r for r in engine.source_rows() if r["split"] == "train"}
    destination = c.OUT / arm
    destination.mkdir()
    model, tokenizer, metadata = engine.load_parent()
    params, wrappers = engine.prepare_trainable(model, arm)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("FP32 master model required")
    model.eval()
    encoded = []
    for row in c.read(c.OUT / "train.json"):
        prefix = tokenizer.encode(c.model_prompt(row, arm), add_special_tokens=False)
        suffix = tokenizer.encode(
            c.target(row, source[row["id"]]["teacher_output"], arm), add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        if len(prefix) + len(suffix) > c.CONTEXT:
            raise RuntimeError(f"ABI_REPRESENTATION_OVERFLOW: {row['id']}")
        encoded.append((row["id"], prefix, suffix, CAPABILITY_CAKE_ORDER.index(row["task"])))
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
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
                raise RuntimeError("EXPERIMENT_BUDGET_STOP")
            if steps * c.BATCH % len(order) == 0:
                rng.shuffle(order)
            indexes = [order[(steps * c.BATCH + j) % len(order)] for j in range(c.BATCH)]
            ids = torch.full(
                (c.BATCH, c.CONTEXT), tokenizer.eos_token_id, device="cuda", dtype=torch.long
            )
            labels, attention = torch.full_like(ids, -100), torch.zeros_like(ids)
            lengths, routes, batch_tokens = [], [], 0
            for j, index in enumerate(indexes):
                record_id, prefix, suffix, route = encoded[index]
                sequence = prefix + suffix
                ids[j, : len(sequence)] = torch.tensor(sequence, device="cuda")
                labels[j, len(prefix) : len(sequence)] = torch.tensor(suffix, device="cuda")
                attention[j, : len(sequence)] = 1
                lengths.append(len(prefix))
                routes.append(route)
                sample_digest.update((record_id + "\n").encode())
                batch_tokens += len(suffix)
            optimizer.zero_grad(set_to_none=True)
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
                raise RuntimeError("ABI_OPTIMIZATION_FAILURE: non-finite FP32 loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("ABI_OPTIMIZATION_FAILURE: non-finite FP32 gradient")
            optimizer.step()
            steps += 1
            loss_tokens += batch_tokens
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            observation = {
                "step": steps,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "loss_tokens": batch_tokens,
                "sample_ids": [encoded[i][0] for i in indexes],
                "elapsed_seconds": time.perf_counter() - start,
            }
            c.append(journal, observation)
            if steps == 1 or steps % 32 == 0:
                print(c.canonical({"stage": arm, **observation}).decode().strip(), flush=True)
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
    with torch.no_grad():
        for owner, name, wrapper in wrappers:
            wrapper.base.weight.add_((wrapper.lora_a @ wrapper.lora_b) * wrapper.scale)
            setattr(owner, name, wrapper.base)
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("Non-finite FP32 checkpoint")
    checkpoint = destination / "model.safetensors"
    save_file(state, str(checkpoint))
    c.write_once(
        destination / "metadata.json",
        {
            "architecture": metadata["architecture"],
            "decoding": metadata["decoding"],
            "checkpoint": {"sha256": c.sha(checkpoint)},
            "r99_fp32": {
                "arm": arm,
                "parent": c.PARENT_SHA,
                "scope": "uncertified pilot",
                "teacher_present_at_inference": False,
            },
        },
    )
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
            "precision": "float32",
            "autocast": False,
            "loss_scaling": False,
            "tf32": False,
            "successful_steps": steps,
            "sample_order_sha256": sample_digest.hexdigest(),
            "source_rows_sha256": c.sha(c.OUT / "source/rows.jsonl"),
            "training_rows": len(encoded),
            "unique_normalized_target_tokens_train": sum(
                len(r["token_ids"]) for r in source.values()
            ),
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
