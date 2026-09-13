"""Non-promotable full-LayerCake learnability upper bound for R30."""

from __future__ import annotations

import argparse
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

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once
from . import package_v4 as base
from . import package_v7 as v7
from .depth_bridge_v11 import _batch
from .host_adapter_v10 import LAYERCAKE_COMMIT
from .host_baseline_v9 import CHECKPOINT_SHA256, TASK_ROUTES, TOKENIZER_SHA256
from .protocol import TASKS


SEED = 30_812
STEPS = 2_400
BATCH_SIZE = 4
LEARNING_RATE = 2e-5


def _imports(layercake_root: Path):
    value = str(layercake_root)
    if value not in sys.path:
        sys.path.insert(0, value)
    from layercake.training.phase2_shallow_sparse import load_student
    from layercake.phase2_campaign import _deterministic_token
    from layercake.portable_domain import state_dict_hash
    return load_student, _deterministic_token, state_dict_hash


def _train(model, tokenizer, rows: list[dict[str, Any]], device: torch.device):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    by_route: dict[int, list[dict[str, Any]]] = {route: [] for route in sorted(set(TASK_ROUTES.values()))}
    for row in rows:
        by_route[TASK_ROUTES[row["oracle_task"]]].append(row)
    routes = sorted(by_route)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    history = []
    started = time.perf_counter()
    successful = 0
    skipped = 0
    while successful < STEPS:
        route = routes[successful % len(routes)]
        selected = [by_route[route][rng.randrange(len(by_route[route]))] for _ in range(BATCH_SIZE)]
        ids, targets, attention, prompt_lengths = _batch(tokenizer, selected, device)
        task_routes = torch.full((BATCH_SIZE,), route, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=task_routes)
            loss = F.cross_entropy(result["logits"][:, :-1].flatten(0, 1), targets[:, 1:].flatten(), ignore_index=-100)
        before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    return {"steps": successful, "skipped_amp_steps": skipped, "batch_size": BATCH_SIZE, "row_examples": successful * BATCH_SIZE, "learning_rate": LEARNING_RATE, "seed": SEED, "parameters_updated": sum(parameter.numel() for parameter in model.parameters()), "history": history, "seconds": time.perf_counter() - started}


@torch.inference_mode()
def _generate(model, tokenizer, prompt: str, route_value: int, select, device: torch.device):
    prompt_ids = tokenizer.encode(prompt.rstrip() + "\n")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    route = torch.tensor([route_value], dtype=torch.long, device=device)
    result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], device=device), task_routes=route, use_cache=True)
    state = {"past_key_values": result["past_key_values"], "task_routes": route, "next_logits": result["logits"][:, -1], "generated_ids": ids[:, :0]}
    generated = []
    for _ in range(192):
        selected = select(state["next_logits"], generated, penalty=1.10, repeat_last_n=64, no_repeat_ngram_size=4)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        _, state = model.decode_step(state, next_token=selected)
    return tokenizer.decode(generated), len(generated)


def run(source: Path, v11: Path, layercake_root: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R30 v12 output exists: {output}")
    prior = json.loads(v11.read_text(encoding="utf-8"))
    if prior.get("verdict") != "FAIL_PUBLIC_DEPTH_SHARED_BRIDGE_CONTROL" or prior.get("metrics", {}).get("functional") != 21:
        raise RuntimeError("R30 v11 negative prerequisite changed")
    if sha256_file(checkpoint / "model.safetensors") != CHECKPOINT_SHA256 or sha256_file(checkpoint / "tokenizer.json") != TOKENIZER_SHA256:
        raise RuntimeError("frozen LayerCake identity changed")
    load_student, select, state_hash = _imports(layercake_root)
    device = torch.device("cuda")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, metadata = load_student(checkpoint, device=device)
    base_hash = state_hash({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()})
    all_rows = base._jsonl(source / "source_rows.jsonl")
    train_rows = [row for row in all_rows if row["split"] == "train"]
    eval_rows = [row for row in all_rows if row["split"] == "evaluation"]
    training = _train(model, tokenizer, train_rows, device)
    output.mkdir(parents=True)
    checkpoint_out = output / "derived_full_model.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(checkpoint_out))
    derived_hash = state_hash({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()})
    del model
    torch.cuda.empty_cache()

    fresh, fresh_tokenizer, fresh_metadata = load_student(checkpoint, device=device)
    fresh.load_state_dict(load_file(str(checkpoint_out), device=str(device)), strict=True)
    fresh.eval()
    for parameter in fresh.parameters():
        parameter.requires_grad_(False)
    loaded_hash = state_hash({name: value.detach().cpu().contiguous() for name, value in fresh.state_dict().items()})
    if loaded_hash != derived_hash:
        raise RuntimeError("R30 v12 derived checkpoint did not reload exactly")
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    observations = []
    for index, row in enumerate(eval_rows, 1):
        before = time.perf_counter()
        text, tokens = _generate(fresh, fresh_tokenizer, v7._normalized_prompt(row), TASK_ROUTES[row["oracle_task"]], select, device)
        torch.cuda.synchronize()
        observations.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "task_route": TASK_ROUTES[row["oracle_task"]], "output": text, "generated_tokens": tokens, "latency_seconds": time.perf_counter() - before, "score": v7._score(row, text, row["teacher_output"])})
        peak_rss = max(peak_rss, process.memory_info().rss)
        if index == 1 or index % 24 == 0:
            print(json.dumps({"evaluated": index, "functional": sum(item["score"]["functional"] for item in observations)}), flush=True)
    rows_out = output / "evaluation.jsonl"
    write_jsonl_once(rows_out, observations)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    passed = functional >= 132 and all(item["functional"] >= 10 for item in by_task.values()) and by_task["abstention"]["functional"] == 12 and loaded_hash == derived_hash and derived_hash != base_hash
    result = {
        "format": "abi-r30-full-layercake-upper-bound/12",
        "verdict": "PASS_NONPROMOTABLE_FULL_HOST_UPPER_BOUND" if passed else "FAIL_NONPROMOTABLE_FULL_HOST_UPPER_BOUND",
        "scientific_role": "FULL_HOST_SEQUENCE_TRAINING_CONTROL_NEVER_PROMOTION_ELIGIBLE",
        "v11_result_sha256": sha256_file(v11),
        "source_rows_sha256": sha256_file(source / "source_rows.jsonl"),
        "layercake": {"repository_commit": LAYERCAKE_COMMIT, "base_checkpoint_sha256": CHECKPOINT_SHA256, "tokenizer_sha256": TOKENIZER_SHA256, "parameters_total": int(fresh_metadata["parameters"]["total"]), "base_state_sha256": base_hash, "derived_state_sha256": derived_hash, "fresh_loaded_state_sha256": loaded_hash},
        "metrics": {"evaluation_rows": len(observations), "functional": functional, "by_task": by_task},
        "training": training,
        "artifact": {"path": checkpoint_out.name, "sha256": sha256_file(checkpoint_out), "bytes": checkpoint_out.stat().st_size, "parameters": training["parameters_updated"], "publish": False},
        "information_accounting": {"teacher_present_during_training": False, "teacher_present_during_execution": False, "teacher_output_bytes_in_training_rows": sum(len(base._validated_response(row)[0].encode("utf-8")) for row in train_rows), "training_rows": len(train_rows), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)},
        "artifacts": {"evaluation": {"path": rows_out.name, "sha256": sha256_file(rows_out), "bytes": rows_out.stat().st_size}},
        "claim_ceiling": "NONPROMOTABLE_FULL_HOST_LEARNABILITY_CONTROL_NOT_ABI_TRANSFER",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v11", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.v11.resolve(), args.layercake_root.resolve(), args.checkpoint.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
