"""Restore sealed-parent general English while retaining fused ABI behavior."""

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
    LayerCakeHostError,
    _bridge_state_sha256,
    _canonical_json_bytes,
    _sha256_file,
    _validate_deployment_manifest,
    load_english_training_rows,
    load_host_model,
    route_for_capability,
)


PRESERVATION_FORMAT = "abi-layercake-general-preservation/1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_general_rows(
    path: Path, *, split: str
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise LayerCakeHostError(
            f"general curriculum has no {split!r} rows"
        )
    for row in selected:
        prompt = str(row["prompt"])
        response = str(row["response"])
        if hashlib.sha256(prompt.encode()).hexdigest() != row[
            "prompt_sha256"
        ]:
            raise LayerCakeHostError("general prompt hash changed")
        if hashlib.sha256(response.encode()).hexdigest() != row[
            "response_sha256"
        ]:
            raise LayerCakeHostError("general response hash changed")
    return selected


def _batch(
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = []
    prompt_lengths = []
    for row in rows:
        prompt_ids = tokenizer.encode(str(row["prompt"]) + "\n")
        response_ids = tokenizer.encode(str(row["response"]))
        sequence = (prompt_ids + response_ids)[:max_tokens]
        if not sequence:
            raise LayerCakeHostError("preservation row encoded empty")
        encoded.append(sequence)
        prompt_lengths.append(min(len(prompt_ids), len(sequence)))
    width = max(len(values) for values in encoded)
    ids = torch.full(
        (len(rows), width),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros(
        (len(rows), width), dtype=torch.long, device=device
    )
    for index, values in enumerate(encoded):
        ids[index, : len(values)] = torch.tensor(
            values, dtype=torch.long, device=device
        )
        attention[index, : len(values)] = 1
    return (
        ids,
        attention,
        torch.tensor(
            prompt_lengths, dtype=torch.long, device=device
        ),
    )


def _masked_mse(
    student: torch.Tensor,
    target: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    mask = attention[:, :, None].to(student.dtype)
    return (
        (student.float() - target.float()).square() * mask
    ).sum() / (mask.sum().clamp_min(1) * student.shape[-1])


def preserve_general_english(
    *,
    bundle_path: str | Path,
    general_curriculum_path: str | Path,
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    source_host_path: str | Path,
    output_path: str | Path,
    seed: int = 9824,
    steps: int = 2400,
    abi_batch_size: int = 2,
    general_batch_size: int = 2,
    learning_rate: float = 2.0e-4,
    max_tokens: int = 192,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Train only cakes against frozen ABI and parent English targets."""

    if (
        steps <= 0
        or abi_batch_size <= 0
        or general_batch_size <= 0
        or max_tokens <= 0
    ):
        raise LayerCakeHostError(
            "preservation sizes and steps must be positive"
        )
    bundle_path = Path(bundle_path).resolve()
    general_curriculum_path = Path(
        general_curriculum_path
    ).resolve()
    layercake_root = Path(layercake_root).resolve()
    parent_path = Path(parent_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    source_host_path = Path(source_host_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(
            f"preserved host artifact is immutable: {output_path}"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise LayerCakeHostError(
            "CUDA preservation was requested but unavailable"
        )
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
    bridge = source_manifest["host_delta"].get(
        "sparse_route_bridge", {}
    )
    if (
        bridge.get("mode") != "none"
        or bridge.get("fused_into_existing_task_cakes") is not True
    ):
        raise LayerCakeHostError(
            "general preservation requires the fused single-cake source"
        )
    budget_index = int(
        source_manifest["imported_artifact"]["budget_index"]
    )
    abi_rows, _budget, _bundle = load_english_training_rows(
        bundle_path, budget_index=budget_index
    )
    for row in abi_rows:
        row["route"] = route_for_capability(str(row["capability"]))
    general_rows = _load_general_rows(
        general_curriculum_path, split="train"
    )
    model, tokenizer, loaded, _ = load_host_model(
        layercake_root=layercake_root,
        parent_path=parent_path,
        canonical_abi_path=canonical_abi_path,
        host_path=source_host_path,
        device_name=device_name,
    )
    parent, _, parent_manifest, _ = load_host_model(
        layercake_root=layercake_root,
        parent_path=parent_path,
        canonical_abi_path=canonical_abi_path,
        host_path=None,
        device_name=device_name,
    )
    if loaded is None or parent_manifest is not None:
        raise LayerCakeHostError("preservation host boundary did not load")
    abi_teacher_cakes = copy.deepcopy(model.task_cakes).eval()
    for parameter in abi_teacher_cakes.parameters():
        parameter.requires_grad_(False)
    for module in (model, parent):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
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
    abi_tokens = 0
    general_tokens = 0
    abi_seen: set[str] = set()
    general_seen: set[str] = set()
    for step in range(1, steps + 1):
        selected_abi = [
            rng.choice(abi_rows) for _ in range(abi_batch_size)
        ]
        selected_general = [
            rng.choice(general_rows)
            for _ in range(general_batch_size)
        ]
        abi_ids, abi_attention, abi_prompt_lengths = _batch(
            tokenizer,
            selected_abi,
            max_tokens=max_tokens,
            device=device,
        )
        general_ids, general_attention, general_prompt_lengths = _batch(
            tokenizer,
            selected_general,
            max_tokens=max_tokens,
            device=device,
        )
        abi_routes = torch.tensor(
            [int(row["route"]) for row in selected_abi],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            abi_base = model.transformer(
                input_ids=abi_ids,
                attention_mask=abi_attention,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            abi_target = torch.empty_like(abi_base)
            for route, cake in enumerate(abi_teacher_cakes):
                indexes = torch.nonzero(
                    abi_routes == route, as_tuple=False
                ).flatten()
                if indexes.numel():
                    abi_target.index_copy_(
                        0,
                        indexes,
                        cake(abi_base.index_select(0, indexes)),
                    )
            general_base = model.transformer(
                input_ids=general_ids,
                attention_mask=general_attention,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            general_summary = model._prompt_summary(
                general_base,
                prompt_lengths=general_prompt_lengths,
                attention_mask=general_attention,
            )
            general_routes = model.task_classifier(
                general_summary
            ).argmax(dim=-1)
            parent_target = parent(
                general_ids,
                attention_mask=general_attention,
                prompt_lengths=general_prompt_lengths,
                use_cache=False,
            )["hidden"]
        optimizer.zero_grad(set_to_none=True)
        abi_student = model._dispatch(abi_base.detach(), abi_routes)
        general_student = model._dispatch(
            general_base.detach(), general_routes
        )
        abi_loss = _masked_mse(
            abi_student, abi_target, abi_attention
        )
        general_loss = _masked_mse(
            general_student, parent_target, general_attention
        )
        loss = abi_loss + general_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        abi_tokens += int(abi_attention.sum().item())
        general_tokens += int(general_attention.sum().item())
        abi_seen.update(str(row["record_id"]) for row in selected_abi)
        general_seen.update(str(row["id"]) for row in selected_general)
        if step == 1 or step % 200 == 0:
            curve = {
                "step": step,
                "abi_hidden_state_mse": float(abi_loss.detach()),
                "general_hidden_state_mse": float(
                    general_loss.detach()
                ),
                "total_loss": float(loss.detach()),
                "abi_tokens_seen": abi_tokens,
                "general_tokens_seen": general_tokens,
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
    for name, value in model.task_cakes.state_dict().items():
        state[f"task_cakes.{name}"] = (
            value.detach().cpu().contiguous()
        )
    if any(key.startswith("route_bridge.") for key in state):
        raise LayerCakeHostError(
            "preservation source unexpectedly retained route bridge"
        )
    output_path.mkdir(parents=True, exist_ok=False)
    delta_path = output_path / "host_delta.safetensors"
    save_file(state, str(delta_path))
    delta_sha = _sha256_file(delta_path)
    manifest = copy.deepcopy(source_manifest)
    manifest["status"] = (
        "GENERAL_PRESERVED_NOT_YET_SEMANTICALLY_CERTIFIED"
    )
    manifest["host_delta"]["path"] = delta_path.name
    manifest["host_delta"]["sha256"] = delta_sha
    manifest["host_delta"]["bytes"] = delta_path.stat().st_size
    manifest["host_delta"][
        "logical_state_sha256_before"
    ] = source_manifest["host_delta"]["logical_state_sha256_after"]
    manifest["host_delta"][
        "logical_state_sha256_after"
    ] = _bridge_state_sha256(state)
    for component in manifest["components"]:
        if component["type"] == (
            "layercake_task_classifier_and_low_rank_cakes"
        ):
            component["sha256"] = delta_sha
    manifest["training"]["general_preservation"] = {
        "format": PRESERVATION_FORMAT,
        "seed": seed,
        "device": str(device),
        "steps": steps,
        "abi_batch_size": abi_batch_size,
        "general_batch_size": general_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "max_tokens": max_tokens,
        "objective": (
            "equal_weight_abi_and_general_masked_hidden_state_mse"
        ),
        "source_llm_loaded": False,
        "general_training_rows": len(general_rows),
        "general_instruction_validation_rows_seen": 0,
        "speed_benchmark_rows_seen": 0,
        "final_test_rows_seen": 0,
        "unique_abi_search_rows_sampled": len(abi_seen),
        "unique_general_train_rows_sampled": len(general_seen),
        "abi_tokens_seen": abi_tokens,
        "general_tokens_seen": general_tokens,
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
    previous_derivation = copy.deepcopy(
        source_manifest.get("derivation")
    )
    manifest["derivation"] = {
        "kind": (
            "dual_teacher_general_preservation_in_existing_task_cakes"
        ),
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
        "general_curriculum_sha256": _sha256_file(
            general_curriculum_path
        ),
        "sealed_parent_checkpoint_sha256": source_manifest[
            "parent_layercake"
        ]["checkpoint_sha256"],
        "source_llm_loaded": False,
        "frozen_transformer_changed": False,
        "classifier_changed": False,
        "symbolic_substrate_changed": False,
        "route_bridge_reintroduced": False,
        "previous_derivation": previous_derivation,
    }
    manifest["claim_boundary"] = (
        "This artifact proves a disjoint-train dual-preservation boundary "
        "and exact identity. General and ABI validation plus native "
        "performance remain unproven until separate locked gates pass."
    )
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    _write_json(output_path / "deployment_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--general-curriculum", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=9824)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--abi-batch-size", type=int, default=2)
    parser.add_argument("--general-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preserve_general_english(
        bundle_path=args.bundle,
        general_curriculum_path=args.general_curriculum,
        layercake_root=args.layercake_root,
        parent_path=args.parent,
        canonical_abi_path=args.canonical_abi,
        source_host_path=args.source_host,
        output_path=args.output,
        seed=args.seed,
        steps=args.steps,
        abi_batch_size=args.abi_batch_size,
        general_batch_size=args.general_batch_size,
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
                "general_preservation": result["training"][
                    "general_preservation"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
