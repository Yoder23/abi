"""Parameter-matched depth-shared residual bridge for the frozen LayerCake host."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once
from . import package_v4 as base
from . import package_v7 as v7
from .host_adapter_v10 import LAYERCAKE_COMMIT
from .host_baseline_v9 import CHECKPOINT_SHA256, TASK_ROUTES, TOKENIZER_SHA256
from .protocol import TASKS


SEED = 30_711
STEPS = 2_400
BATCH_SIZE = 4
LEARNING_RATE = 0.001
RANK = 64
ROUTES = tuple(sorted(set(TASK_ROUTES.values())))
ROUTE_INDEX = {route: index for index, route in enumerate(ROUTES)}


def _imports(layercake_root: Path):
    value = str(layercake_root)
    if value not in sys.path:
        sys.path.insert(0, value)
    from layercake.training.phase2_shallow_sparse import load_student
    from layercake.phase2_campaign import _deterministic_token
    from layercake.portable_domain import state_dict_hash
    return load_student, _deterministic_token, state_dict_hash


class DepthSharedBridge(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, RANK * len(ROUTES), bias=False)
        self.up = nn.Linear(RANK * len(ROUTES), width, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor, route_indexes: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        normalized = self.norm(hidden)
        for index in route_indexes.unique(sorted=True):
            route = int(index.item())
            rows = torch.nonzero(route_indexes == route, as_tuple=False).flatten()
            start, stop = route * RANK, (route + 1) * RANK
            selected = normalized.index_select(0, rows)
            low = F.linear(selected, self.down.weight[start:stop])
            delta = F.linear(F.silu(low), self.up.weight[:, start:stop])
            output.index_copy_(0, rows, delta.to(hidden.dtype))
        return hidden + output


class BridgeHooks:
    def __init__(self, model, bridge: DepthSharedBridge):
        self.model = model
        self.bridge = bridge
        self.route_indexes: torch.Tensor | None = None
        self.calls = 0
        self.handles = [block.register_forward_pre_hook(self._hook, with_kwargs=True) for block in model.transformer.h]

    def _hook(self, module, args, kwargs):
        hidden = args[0]
        if self.route_indexes is None or self.route_indexes.shape[0] != hidden.shape[0]:
            raise RuntimeError("R30 v11 bridge route is absent")
        self.calls += 1
        return (self.bridge(hidden, self.route_indexes), *args[1:]), kwargs

    def set_routes(self, task_routes: torch.Tensor) -> None:
        values = [ROUTE_INDEX[int(value)] for value in task_routes.detach().cpu().tolist()]
        self.route_indexes = torch.tensor(values, dtype=torch.long, device=task_routes.device)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _frozen_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}


def _batch(tokenizer, rows: list[dict[str, Any]], device: torch.device):
    sequences, labels = [], []
    for row in rows:
        response, _ = base._validated_response(row)
        prompt = tokenizer.encode(v7._normalized_prompt(row).rstrip() + "\n")
        answer = tokenizer.encode(response) + [tokenizer.eos_token_id]
        sequence = prompt + answer
        if len(sequence) > 512:
            raise RuntimeError("R30 v11 sequence exceeds frozen bound")
        target = [-100] * len(prompt) + answer
        sequences.append(sequence)
        labels.append(target)
    width = max(map(len, sequences))
    ids = torch.full((len(rows), width), tokenizer.eos_token_id, dtype=torch.long, device=device)
    target = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    attention = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    lengths = []
    for index, (sequence, label) in enumerate(zip(sequences, labels)):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        target[index, : len(label)] = torch.tensor(label, dtype=torch.long, device=device)
        attention[index, : len(sequence)] = 1
        lengths.append(len(sequence) - sum(value >= 0 for value in label))
    return ids, target, attention, torch.tensor(lengths, dtype=torch.long, device=device)


def _train(model, tokenizer, rows: list[dict[str, Any]], bridge: DepthSharedBridge, hooks: BridgeHooks, device: torch.device):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in bridge.parameters():
        parameter.requires_grad_(True)
    by_route: dict[int, list[dict[str, Any]]] = {route: [] for route in ROUTES}
    for row in rows:
        by_route[TASK_ROUTES[row["oracle_task"]]].append(row)
    if any(not values for values in by_route.values()):
        raise RuntimeError("R30 v11 route has no training rows")
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    started = time.perf_counter()
    successful = 0
    skipped = 0
    while successful < STEPS:
        route = ROUTES[successful % len(ROUTES)]
        selected = [by_route[route][rng.randrange(len(by_route[route]))] for _ in range(BATCH_SIZE)]
        ids, targets, attention, prompt_lengths = _batch(tokenizer, selected, device)
        task_routes = torch.full((BATCH_SIZE,), route, dtype=torch.long, device=device)
        hooks.set_routes(task_routes)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=task_routes)
            loss = F.cross_entropy(result["logits"][:, :-1].flatten(0, 1), targets[:, 1:].flatten(), ignore_index=-100)
        before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before:
            skipped += 1
            continue
        successful += 1
        if successful in {1, 600, 1_200, 1_800, 2_400}:
            row = {"step": successful, "route": route, "loss": float(loss), "gradient_norm": float(gradient), "seconds": time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    return {
        "steps": successful,
        "skipped_amp_steps": skipped,
        "batch_size": BATCH_SIZE,
        "row_examples": successful * BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "rank": RANK,
        "routes": list(ROUTES),
        "parameters": sum(parameter.numel() for parameter in bridge.parameters()),
        "history": history,
        "seconds": time.perf_counter() - started,
        "physical_bridge_calls": hooks.calls,
    }


@torch.inference_mode()
def _generate(model, tokenizer, prompt: str, task_route: int, hooks: BridgeHooks | None, select, device: torch.device):
    prompt_ids = tokenizer.encode(prompt.rstrip() + "\n")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    route = torch.tensor([task_route], dtype=torch.long, device=device)
    if hooks is not None:
        hooks.set_routes(route)
    result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], device=device), task_routes=route, use_cache=True)
    state = {"past_key_values": result["past_key_values"], "task_routes": route, "next_logits": result["logits"][:, -1], "generated_ids": ids[:, :0]}
    generated = []
    for _ in range(192):
        selected = select(state["next_logits"], generated, penalty=1.10, repeat_last_n=64, no_repeat_ngram_size=4)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        if hooks is not None:
            hooks.set_routes(route)
        _, state = model.decode_step(state, next_token=selected)
    return tokenizer.decode(generated), len(generated)


def run(source: Path, v10: Path, layercake_root: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R30 v11 output exists: {output}")
    prior = json.loads(v10.read_text(encoding="utf-8"))
    if prior.get("verdict") != "FAIL_PUBLIC_SPARSE_ADAPTER_CONTROL" or prior.get("metrics", {}).get("functional") != 11:
        raise RuntimeError("R30 v10 negative prerequisite changed")
    if sha256_file(checkpoint / "model.safetensors") != CHECKPOINT_SHA256 or sha256_file(checkpoint / "tokenizer.json") != TOKENIZER_SHA256:
        raise RuntimeError("frozen LayerCake identity changed")
    rows_path = source / "source_rows.jsonl"
    load_student, select, state_hash = _imports(layercake_root)
    device = torch.device("cuda")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, metadata = load_student(checkpoint, device=device)
    frozen_before = state_hash(_frozen_state(model))
    bridge = DepthSharedBridge(model.config.width).to(device)
    hooks = BridgeHooks(model, bridge)
    all_rows = base._jsonl(rows_path)
    train_rows = [row for row in all_rows if row["split"] == "train"]
    eval_rows = [row for row in all_rows if row["split"] == "evaluation"]
    training = _train(model, tokenizer, train_rows, bridge, hooks, device)
    frozen_after = state_hash(_frozen_state(model))
    if frozen_after != frozen_before:
        raise RuntimeError("R30 v11 modified the frozen LayerCake base")
    output.mkdir(parents=True)
    bridge_path = output / "depth_shared_bridge.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in bridge.state_dict().items()}, str(bridge_path))
    hooks.remove()
    del model, bridge
    torch.cuda.empty_cache()

    fresh, fresh_tokenizer, fresh_metadata = load_student(checkpoint, device=device)
    fresh_hash = state_hash(_frozen_state(fresh))
    bridge = DepthSharedBridge(fresh.config.width).to(device)
    bridge.load_state_dict(load_file(str(bridge_path), device=str(device)), strict=True)
    bridge.eval()
    for parameter in fresh.parameters():
        parameter.requires_grad_(False)
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    hooks = BridgeHooks(fresh, bridge)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    observations = []
    calls_before = hooks.calls
    for index, row in enumerate(eval_rows, 1):
        before = time.perf_counter()
        text, tokens = _generate(fresh, fresh_tokenizer, v7._normalized_prompt(row), TASK_ROUTES[row["oracle_task"]], hooks, select, device)
        torch.cuda.synchronize()
        observations.append({
            "record_id": row["record_id"],
            "oracle_task": row["oracle_task"],
            "task_route": TASK_ROUTES[row["oracle_task"]],
            "output": text,
            "generated_tokens": tokens,
            "latency_seconds": time.perf_counter() - before,
            "score": v7._score(row, text, row["teacher_output"]),
        })
        peak_rss = max(peak_rss, process.memory_info().rss)
        if index == 1 or index % 24 == 0:
            print(json.dumps({"evaluated": index, "functional": sum(item["score"]["functional"] for item in observations)}), flush=True)
    evaluation_bridge_calls = hooks.calls - calls_before
    hooks.remove()
    rows_out = output / "evaluation.jsonl"
    write_jsonl_once(rows_out, observations)

    base_rows = []
    for task in TASKS:
        row = next(item for item in eval_rows if item["oracle_task"] == task)
        text, _ = _generate(fresh, fresh_tokenizer, v7._normalized_prompt(row), TASK_ROUTES[task], None, select, device)
        adapted = next(item["output"] for item in observations if item["record_id"] == row["record_id"])
        base_rows.append({"record_id": row["record_id"], "task": task, "artifact_changes_output": text != adapted})

    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    removal_causal = sum(row["artifact_changes_output"] for row in base_rows)
    passed = (
        functional >= 132
        and all(item["functional"] >= 10 for item in by_task.values())
        and by_task["abstention"]["functional"] == 12
        and frozen_before == frozen_after == fresh_hash
        and removal_causal == 12
        and evaluation_bridge_calls >= len(eval_rows) * 3
    )
    manifest = {
        "format": "abi-r30-depth-shared-layercake-bridge/1",
        "base_checkpoint_sha256": CHECKPOINT_SHA256,
        "base_tokenizer_sha256": TOKENIZER_SHA256,
        "source_rows_sha256": sha256_file(rows_path),
        "bridge_sha256": sha256_file(bridge_path),
        "bridge_bytes": bridge_path.stat().st_size,
        "architecture": {"placement": "before_each_frozen_transformer_block", "reuse": "same_route_tensor_all_three_blocks", "rank": RANK, "routes": list(ROUTES)},
        "task_routes": TASK_ROUTES,
        "training": training,
    }
    manifest_path = output / "artifact_manifest.json"
    write_json_once(manifest_path, manifest)
    result = {
        "format": "abi-r30-depth-shared-bridge-control/11",
        "verdict": "PASS_PUBLIC_DEPTH_SHARED_BRIDGE_CONTROL" if passed else "FAIL_PUBLIC_DEPTH_SHARED_BRIDGE_CONTROL",
        "scientific_role": "EXTERNAL_BRIDGE_CONTROL_NOT_ABI_SUPERIORITY",
        "v10_result_sha256": sha256_file(v10),
        "source_rows_sha256": sha256_file(rows_path),
        "layercake": {
            "repository_commit": LAYERCAKE_COMMIT,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "parameters_total": int(fresh_metadata["parameters"]["total"]),
            "frozen_state_sha256_before": frozen_before,
            "frozen_state_sha256_after": frozen_after,
            "fresh_frozen_state_sha256": fresh_hash,
        },
        "metrics": {
            "evaluation_rows": len(observations),
            "functional": functional,
            "by_task": by_task,
            "removal_causal": removal_causal,
            "evaluation_physical_bridge_calls": evaluation_bridge_calls,
        },
        "training": training,
        "artifact": {"path": bridge_path.name, "sha256": sha256_file(bridge_path), "bytes": bridge_path.stat().st_size, "parameters": training["parameters"], "manifest": {"path": manifest_path.name, "sha256": sha256_file(manifest_path)}},
        "information_accounting": {
            "teacher_present_during_training": False,
            "teacher_present_during_execution": False,
            "source_parameters_copied": 0,
            "teacher_logits_stored": 0,
            "teacher_hidden_activations_stored": 0,
            "teacher_output_bytes_in_training_rows": sum(len(base._validated_response(row)[0].encode("utf-8")) for row in train_rows),
            "training_rows": len(train_rows),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cpu_rss_bytes": int(peak_rss),
        },
        "artifacts": {"evaluation": {"path": rows_out.name, "sha256": sha256_file(rows_out), "bytes": rows_out.stat().st_size}, "removal": base_rows},
        "claim_ceiling": "DISCLOSED_EXTERNAL_BRIDGE_CONTROL_NOT_GENERAL_ENGLISH_OR_ABI_SUPERIORITY",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v10", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.v10.resolve(), args.layercake_root.resolve(), args.checkpoint.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
