"""Fuse an ABI route bridge into LayerCake's existing sparse task-cake slot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import psutil
import torch
from safetensors.torch import load_file, save_file

from .layercake_host import (
    SYMBOLIC_SURFACE_STATE_KEY,
    LayerCakeHostError,
    _bridge_state_sha256,
    _canonical_json_bytes,
    _sha256_file,
    _validate_deployment_manifest,
    load_english_training_rows,
    load_host_model,
    route_for_capability,
)


FUSION_FORMAT = "abi-layercake-route-fusion/1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _batch_ids(
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [
        tokenizer.encode(
            str(row["prompt"]) + "\n" + str(row["response"])
        )[:max_tokens]
        for row in rows
    ]
    if any(not values for values in encoded):
        raise LayerCakeHostError("fusion row encoded to no tokens")
    width = max(len(values) for values in encoded)
    input_ids = torch.full(
        (len(encoded), width),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros(
        (len(encoded), width),
        dtype=torch.long,
        device=device,
    )
    for index, values in enumerate(encoded):
        input_ids[index, : len(values)] = torch.tensor(
            values, dtype=torch.long, device=device
        )
        attention[index, : len(values)] = 1
    return input_ids, attention


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def fuse_route_bridge(
    *,
    bundle_path: str | Path,
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    source_host_path: str | Path,
    output_path: str | Path,
    seed: int = 9824,
    steps: int = 800,
    batch_size: int = 4,
    learning_rate: float = 5.0e-4,
    max_tokens: int = 192,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Distill two sparse residuals into the existing one-cake runtime."""

    if steps <= 0 or batch_size <= 0 or max_tokens <= 0:
        raise LayerCakeHostError(
            "fusion steps, batch size, and token limit must be positive"
        )
    bundle_path = Path(bundle_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    parent_path = Path(parent_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    source_host_path = Path(source_host_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(
            f"fused host artifact is immutable: {output_path}"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise LayerCakeHostError("CUDA fusion was requested but is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    source_manifest_path = source_host_path / "deployment_manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    _validate_deployment_manifest(source_manifest)
    bridge_contract = source_manifest["host_delta"].get(
        "sparse_route_bridge", {}
    )
    if (
        bridge_contract.get("mode") != "post_transformer_residual"
        or int(bridge_contract.get("rank", 0)) != 64
    ):
        raise LayerCakeHostError(
            "source host lacks the locked rank-64 route bridge"
        )
    budget_index = int(
        source_manifest["imported_artifact"]["budget_index"]
    )
    rows, _budget, _bundle = load_english_training_rows(
        bundle_path, budget_index=budget_index
    )
    for row in rows:
        row["route"] = route_for_capability(str(row["capability"]))
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(int(row["route"]), []).append(row)
    routes = sorted(buckets)
    model, tokenizer, loaded_manifest, _ = load_host_model(
        layercake_root=layercake_root,
        parent_path=parent_path,
        canonical_abi_path=canonical_abi_path,
        host_path=source_host_path,
        device_name=device_name,
    )
    if loaded_manifest is None:
        raise LayerCakeHostError("source host manifest did not load")
    route_bridge = getattr(model, "_abi_sparse_route_bridge", None)
    if route_bridge is None:
        raise LayerCakeHostError("loaded source route bridge is absent")
    teacher_cakes = copy.deepcopy(model.task_cakes).eval()
    teacher_bridge = copy.deepcopy(route_bridge).eval()
    for module in (teacher_cakes, teacher_bridge):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for cake in model.task_cakes:
        for parameter in cake.parameters():
            parameter.requires_grad_(True)
        cake.train()
    trainable = [
        parameter
        for cake in model.task_cakes
        for parameter in cake.parameters()
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=0.0
    )
    rng = random.Random(seed)
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    cpu_before = process.cpu_times()
    started = time.perf_counter()
    curves = []
    tokens_seen = 0
    rows_seen: set[str] = set()
    for step in range(1, steps + 1):
        route = routes[(step - 1) % len(routes)]
        selected = [
            rng.choice(buckets[route]) for _ in range(batch_size)
        ]
        input_ids, attention = _batch_ids(
            tokenizer,
            selected,
            max_tokens=max_tokens,
            device=device,
        )
        with torch.no_grad():
            hidden = model.transformer(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            first = teacher_cakes[route](hidden)
            target = first + teacher_bridge.bridges[route](first)
        optimizer.zero_grad(set_to_none=True)
        student = model.task_cakes[route](hidden.detach())
        mask = attention[:, :, None].to(student.dtype)
        squared = (student.float() - target.float()).square()
        loss = (squared * mask).sum() / (
            mask.sum().clamp_min(1) * student.shape[-1]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.task_cakes[route].parameters(), 1.0
        )
        optimizer.step()
        tokens_seen += int(attention.sum().item())
        rows_seen.update(str(row["record_id"]) for row in selected)
        if step == 1 or step % 100 == 0:
            curve = {
                "step": step,
                "hidden_state_mse": float(loss.detach()),
                "route": route,
                "tokens_seen": tokens_seen,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    cpu_seconds = (
        cpu_after.user
        + cpu_after.system
        - cpu_before.user
        - cpu_before.system
    )
    source_delta_path = (
        source_host_path / source_manifest["host_delta"]["path"]
    )
    state = load_file(str(source_delta_path), device="cpu")
    route_keys = [
        key for key in state if key.startswith("route_bridge.")
    ]
    if not route_keys:
        raise LayerCakeHostError(
            "source host state has no route-bridge parameters"
        )
    for key in route_keys:
        del state[key]
    for name, value in model.task_cakes.state_dict().items():
        state[f"task_cakes.{name}"] = (
            value.detach().cpu().contiguous()
        )
    if SYMBOLIC_SURFACE_STATE_KEY not in state:
        raise LayerCakeHostError(
            "fusion source lost its symbolic substrate"
        )
    output_path.mkdir(parents=True, exist_ok=False)
    delta_path = output_path / "host_delta.safetensors"
    save_file(state, str(delta_path))
    delta_sha = _sha256_file(delta_path)
    manifest = copy.deepcopy(source_manifest)
    manifest["status"] = "FUSED_NOT_YET_SEMANTICALLY_CERTIFIED"
    manifest["host_delta"]["path"] = delta_path.name
    manifest["host_delta"]["sha256"] = delta_sha
    manifest["host_delta"]["bytes"] = delta_path.stat().st_size
    manifest["host_delta"][
        "logical_state_sha256_before"
    ] = source_manifest["host_delta"]["logical_state_sha256_after"]
    manifest["host_delta"][
        "logical_state_sha256_after"
    ] = _bridge_state_sha256(state)
    removed_parameters = int(
        source_manifest["host_delta"]["sparse_route_bridge"][
            "parameter_count"
        ]
    )
    manifest["host_delta"]["trained_parameter_count"] = (
        int(source_manifest["host_delta"]["trained_parameter_count"])
        - removed_parameters
    )
    manifest["host_delta"]["sparse_route_bridge"] = {
        "mode": "none",
        "rank": 0,
        "installed_routes": 0,
        "maximum_active_routes_per_sequence": 0,
        "parameter_count": 0,
        "fused_into_existing_task_cakes": True,
    }
    manifest["decoding"] = {
        "algorithm": "greedy",
        "no_repeat_ngram_size": 4,
        "prompt_identity_mixture": False,
        "repetition_penalty": 1.15,
    }
    components = []
    for component in manifest["components"]:
        if component["type"] == "abi_sparse_route_conformance_bridge":
            continue
        updated = dict(component)
        if updated["type"] == (
            "layercake_task_classifier_and_low_rank_cakes"
        ):
            updated["sha256"] = delta_sha
        components.append(updated)
    manifest["components"] = components
    manifest["training"]["runtime_fusion"] = {
        "format": FUSION_FORMAT,
        "seed": seed,
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "max_tokens": max_tokens,
        "objective": "masked_token_hidden_state_mse",
        "source_llm_loaded": False,
        "validation_rows_seen": 0,
        "final_test_rows_seen": 0,
        "selected_search_rows": len(rows),
        "unique_search_rows_sampled": len(rows_seen),
        "tokens_seen": tokens_seen,
        "wall_seconds": elapsed,
        "cpu_seconds": cpu_seconds,
        "active_parameter_seconds": (
            sum(parameter.numel() for parameter in trainable) * elapsed
        ),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": int(process.memory_info().rss),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "curves": curves,
    }
    manifest["derivation"] = {
        "kind": "route_bridge_distilled_into_existing_task_cakes",
        "source_host_manifest_sha256": source_manifest[
            "manifest_sha256"
        ],
        "source_host_manifest_file_sha256": _sha256_file(
            source_manifest_path
        ),
        "source_host_delta_sha256": _sha256_file(
            source_delta_path
        ),
        "training_bundle_sha256": _sha256_file(bundle_path),
        "source_llm_loaded": False,
        "frozen_transformer_changed": False,
        "classifier_changed": False,
        "symbolic_substrate_changed": False,
        "route_bridge_parameters_removed": removed_parameters,
        "task_cake_parameters_optimized": sum(
            parameter.numel() for parameter in trainable
        ),
    }
    manifest["claim_boundary"] = (
        "This artifact proves a search-only route-fusion training boundary "
        "and exact identity. It is not semantically or performance certified "
        "until the locked native validation and benchmark gates pass."
    )
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    _write_json(output_path / "deployment_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=9824)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = fuse_route_bridge(
        bundle_path=args.bundle,
        layercake_root=args.layercake_root,
        parent_path=args.parent,
        canonical_abi_path=args.canonical_abi,
        source_host_path=args.source_host,
        output_path=args.output,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest_sha256": result["manifest_sha256"],
                "host_delta": result["host_delta"],
                "runtime_fusion": result["training"]["runtime_fusion"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
