"""Train only LayerCake's sparse task cakes on cached R30 teacher records."""

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
from .host_baseline_v9 import CHECKPOINT_SHA256, TASK_ROUTES, TOKENIZER_SHA256, _generate
from .protocol import TASKS


SEED = 30_610
STEPS = 6_000
BATCH_SIZE = 128
LEARNING_RATE = 0.002
LAYERCAKE_COMMIT = "2170c78b7ae901eb4e99ec93ad59d573b2c75941"


def _imports(layercake_root: Path):
    value = str(layercake_root)
    if value not in sys.path:
        sys.path.insert(0, value)
    from layercake.training.phase2_shallow_sparse import load_student
    from layercake.phase2_campaign import _deterministic_token
    from layercake.portable_domain import state_dict_hash
    return load_student, _deterministic_token, state_dict_hash


def _frozen_state(model, trainable_routes: set[int]) -> dict[str, torch.Tensor]:
    prefixes = tuple(f"task_cakes.{route}." for route in sorted(trainable_routes))
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if not name.startswith(prefixes)
    }


@torch.inference_mode()
def _feature_pools(model, tokenizer, rows: list[dict[str, Any]], device: torch.device):
    pools: dict[int, dict[str, list[torch.Tensor]]] = {}
    target_tokens = 0
    for index, row in enumerate(rows, 1):
        response, _ = base._validated_response(row)
        prompt_ids = tokenizer.encode(v7._normalized_prompt(row).rstrip() + "\n")
        response_ids = tokenizer.encode(response) + [tokenizer.eos_token_id]
        full = (prompt_ids + response_ids)[: model.config.max_tokens]
        if len(full) <= len(prompt_ids):
            raise RuntimeError("R30 v10 response was truncated completely")
        inputs = torch.tensor([full[:-1]], dtype=torch.long, device=device)
        hidden = model.transformer(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state[0]
        first = len(prompt_ids) - 1
        selected = hidden[first:].detach().to("cpu", dtype=torch.float16).contiguous()
        targets = torch.tensor(full[first + 1 :], dtype=torch.long)
        if selected.shape[0] != targets.shape[0]:
            raise RuntimeError("R30 v10 cached feature alignment changed")
        route = TASK_ROUTES[row["oracle_task"]]
        pool = pools.setdefault(route, {"hidden": [], "targets": []})
        pool["hidden"].append(selected)
        pool["targets"].append(targets)
        target_tokens += int(targets.numel())
        if index % 72 == 0:
            print(json.dumps({"cached_rows": index, "target_tokens": target_tokens}), flush=True)
    return {
        route: {
            "hidden": torch.cat(values["hidden"]),
            "targets": torch.cat(values["targets"]),
        }
        for route, values in pools.items()
    }, target_tokens


def _train(model, pools, device: torch.device):
    routes = sorted(pools)
    parameters = [parameter for route in routes for parameter in model.task_cakes[route].parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.01)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    history = []
    started = time.perf_counter()
    output_weight = model.output_weight.detach()
    for step in range(1, STEPS + 1):
        route = routes[(step - 1) % len(routes)]
        pool = pools[route]
        indexes = torch.tensor([rng.randrange(pool["targets"].shape[0]) for _ in range(BATCH_SIZE)], dtype=torch.long)
        hidden = pool["hidden"].index_select(0, indexes).to(device=device, dtype=torch.float32)
        targets = pool["targets"].index_select(0, indexes).to(device)
        logits = F.linear(model.task_cakes[route](hidden), output_weight)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step in {1, 1_500, 3_000, 4_500, 6_000}:
            row = {
                "step": step,
                "route": route,
                "loss": float(loss),
                "accuracy": float(logits.argmax(-1).eq(targets).float().mean()),
                "gradient_norm": float(gradient),
                "seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    return {
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "token_examples": STEPS * BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "routes": routes,
        "parameters": sum(parameter.numel() for parameter in parameters),
        "history": history,
        "seconds": time.perf_counter() - started,
    }


def _adapter_state(model, routes: list[int]) -> dict[str, torch.Tensor]:
    prefixes = tuple(f"task_cakes.{route}." for route in routes)
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def run(source: Path, diagnosis: Path, v9: Path, layercake_root: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R30 v10 output exists: {output}")
    prior = json.loads(v9.read_text(encoding="utf-8"))
    if prior.get("verdict") != "FAIL_FROZEN_HOST_BASELINE" or prior.get("metrics", {}).get("functional") != 0:
        raise RuntimeError("R30 v9 baseline binding changed")
    if sha256_file(checkpoint / "model.safetensors") != CHECKPOINT_SHA256 or sha256_file(checkpoint / "tokenizer.json") != TOKENIZER_SHA256:
        raise RuntimeError("frozen LayerCake identity changed")
    rows_path = source / "source_rows.jsonl"
    diagnosis_path = diagnosis / "result.json"
    if json.loads(diagnosis_path.read_text(encoding="utf-8")).get("verdict") != "PASS_PUBLIC_NEURAL_DIAGNOSIS":
        raise RuntimeError("R30 diagnosis prerequisite changed")
    load_student, select, state_hash = _imports(layercake_root)
    device = torch.device("cuda")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, metadata = load_student(checkpoint, device=device)
    train_rows = [row for row in base._jsonl(rows_path) if row["split"] == "train"]
    eval_rows = [row for row in base._jsonl(rows_path) if row["split"] == "evaluation"]
    routes = set(TASK_ROUTES.values())
    frozen_before = state_hash(_frozen_state(model, routes))
    pools, target_tokens = _feature_pools(model, tokenizer, train_rows, device)
    cached_feature_values = sum(
        int(pool["hidden"].numel()) for pool in pools.values()
    )
    training = _train(model, pools, device)
    frozen_after = state_hash(_frozen_state(model, routes))
    if frozen_after != frozen_before:
        raise RuntimeError("R30 v10 modified frozen LayerCake state")
    output.mkdir(parents=True)
    adapter_path = output / "adapter.safetensors"
    save_file(_adapter_state(model, training["routes"]), str(adapter_path))
    artifact_manifest = {
        "format": "abi-r30-layercake-sparse-adapter/1",
        "base_checkpoint_sha256": CHECKPOINT_SHA256,
        "base_tokenizer_sha256": TOKENIZER_SHA256,
        "source_rows_sha256": sha256_file(rows_path),
        "diagnosis_sha256": sha256_file(diagnosis_path),
        "adapter_sha256": sha256_file(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "tensor_names": sorted(_adapter_state(model, training["routes"])),
        "task_routes": TASK_ROUTES,
        "training": training,
    }
    manifest_path = output / "artifact_manifest.json"
    write_json_once(manifest_path, artifact_manifest)
    del model, pools
    torch.cuda.empty_cache()

    fresh, fresh_tokenizer, fresh_metadata = load_student(checkpoint, device=device)
    frozen_fresh_before = state_hash(_frozen_state(fresh, routes))
    adapter = load_file(str(adapter_path), device=str(device))
    current = fresh.state_dict()
    with torch.no_grad():
        for name, value in adapter.items():
            if name not in current or current[name].shape != value.shape:
                raise RuntimeError("R30 v10 adapter tensor is incompatible")
            current[name].copy_(value)
    if state_hash(_frozen_state(fresh, routes)) != frozen_fresh_before:
        raise RuntimeError("R30 v10 fresh application modified frozen state")
    for parameter in fresh.parameters():
        parameter.requires_grad_(False)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    observations = []
    for index, row in enumerate(eval_rows, 1):
        before = time.perf_counter()
        text, tokens = _generate(fresh, fresh_tokenizer, v7._normalized_prompt(row), TASK_ROUTES[row["oracle_task"]], select, device)
        torch.cuda.synchronize()
        scored = v7._score(row, text, row["teacher_output"])
        observations.append({
            "record_id": row["record_id"],
            "oracle_task": row["oracle_task"],
            "task_route": TASK_ROUTES[row["oracle_task"]],
            "output": text,
            "generated_tokens": tokens,
            "latency_seconds": time.perf_counter() - before,
            "score": scored,
        })
        peak_rss = max(peak_rss, process.memory_info().rss)
        if index == 1 or index % 24 == 0:
            print(json.dumps({"evaluated": index, "functional": sum(item["score"]["functional"] for item in observations)}), flush=True)
    rows_out = output / "evaluation.jsonl"
    write_jsonl_once(rows_out, observations)

    baseline_rows = [json.loads(line) for line in (v9.parent / "evaluation.jsonl").read_text(encoding="utf-8").splitlines() if line]
    selected_ids = {next(row["record_id"] for row in eval_rows if row["oracle_task"] == task) for task in TASKS}
    baseline_selected = {row["record_id"]: row["output"] for row in baseline_rows if row["record_id"] in selected_ids}
    removed, removed_tokenizer, _ = load_student(checkpoint, device=device)
    removal = []
    for row in eval_rows:
        if row["record_id"] not in selected_ids:
            continue
        text, _, = _generate(removed, removed_tokenizer, v7._normalized_prompt(row), TASK_ROUTES[row["oracle_task"]], select, device)
        removal.append({"record_id": row["record_id"], "byte_exact_v9": text == baseline_selected[row["record_id"]]})

    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    passed = (
        functional >= 132
        and all(item["functional"] >= 10 for item in by_task.values())
        and by_task["abstention"]["functional"] == 12
        and frozen_before == frozen_after == frozen_fresh_before
        and len(removal) == 12
        and all(row["byte_exact_v9"] for row in removal)
    )
    result = {
        "format": "abi-r30-frozen-host-sparse-adapter-control/10",
        "verdict": "PASS_PUBLIC_SPARSE_ADAPTER_CONTROL" if passed else "FAIL_PUBLIC_SPARSE_ADAPTER_CONTROL",
        "scientific_role": "LAYERCAKE_ADAPTER_CONTROL_NOT_ABI_SUPERIORITY",
        "v9_result_sha256": sha256_file(v9),
        "source_rows_sha256": sha256_file(rows_path),
        "diagnosis_sha256": sha256_file(diagnosis_path),
        "layercake": {
            "repository_commit": LAYERCAKE_COMMIT,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "parameters_total": int(fresh_metadata["parameters"]["total"]),
            "parameters_active_base": int(fresh_metadata["parameters"]["active"]),
            "frozen_state_sha256_before": frozen_before,
            "frozen_state_sha256_after": frozen_after,
        },
        "metrics": {
            "evaluation_rows": len(observations),
            "functional": functional,
            "by_task": by_task,
            "removal_replay_exact": sum(row["byte_exact_v9"] for row in removal),
        },
        "training": {**training, "unique_cached_target_tokens": target_tokens},
        "artifact": {
            "path": adapter_path.name,
            "sha256": sha256_file(adapter_path),
            "bytes": adapter_path.stat().st_size,
            "parameters": training["parameters"],
            "manifest": {"path": manifest_path.name, "sha256": sha256_file(manifest_path)},
        },
        "information_accounting": {
            "teacher_present_during_training": False,
            "teacher_present_during_execution": False,
            "source_parameters_copied": 0,
            "teacher_logits_stored": 0,
            "teacher_hidden_activations_stored": 0,
            "teacher_output_bytes_in_training_rows": sum(len(base._validated_response(row)[0].encode("utf-8")) for row in train_rows),
            "cached_layercake_feature_values_volatile": cached_feature_values,
            "cached_layercake_features_persisted": False,
            "training_rows": len(train_rows),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cpu_rss_bytes": int(peak_rss),
        },
        "artifacts": {
            "evaluation": {"path": rows_out.name, "sha256": sha256_file(rows_out), "bytes": rows_out.stat().st_size},
            "removal": removal,
        },
        "claim_ceiling": "DISCLOSED_LAYERCAKE_ADAPTER_CONTROL_NOT_GENERAL_ENGLISH_OR_ABI_SUPERIORITY",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--v9", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.v9.resolve(), args.layercake_root.resolve(), args.checkpoint.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
