"""Balance a passing ABI host against sealed-parent general English outputs.

This repair starts from an ABI-passing host delta and reconstructs its LoRA
wrappers over the unchanged sealed LayerCake parent.  Only those existing LoRA
matrices and the existing sparse task cakes are optimized.  The source LLM,
validation rows, benchmark prompts, and final-test rows are absent.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .artifacts import module_state_sha256
from .layercake_host import (
    BRIDGE_PREFIXES,
    SYMBOLIC_SURFACE_STATE_KEY,
    LayerCakeHostError,
    LoRAConv1D,
    _autonomous_prefixes,
    _batch,
    _bridge_state_sha256,
    _canonical_json_bytes,
    _capture_and_remove_lora,
    _equal_record_prompt_overlap_ce,
    _equal_record_shifted_ce,
    _fuse_lora,
    _install_lora,
    _resolve_module,
    _sha256_file,
    _validate_deployment_manifest,
    load_english_training_rows,
    load_host_model,
)
from .layercake_host_preservation import _load_general_rows


BALANCING_FORMAT = "abi-layercake-general-output-balancing/1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _install_source_state(
    model,
    state: Mapping[str, torch.Tensor],
    manifest: Mapping[str, Any],
) -> tuple[list[str], int, float]:
    """Restore bridge tensors and trainable LoRA wrappers without fusing."""

    lora = manifest["host_delta"]["lora"]
    rank = int(lora["rank"])
    alpha = float(lora["alpha"])
    expected_targets = sorted(str(value) for value in lora["target_modules"])
    targets = _install_lora(model, rank=rank, alpha=alpha)
    if targets != expected_targets:
        raise LayerCakeHostError("source LoRA target graph changed")
    parameters = dict(model.named_parameters())
    expected_bridge = {
        name for name in parameters if name.startswith(BRIDGE_PREFIXES)
    }
    expected_lora = {
        key
        for name in targets
        for key in (f"lora.{name}.a", f"lora.{name}.b")
    }
    expected_symbolic = {SYMBOLIC_SURFACE_STATE_KEY}
    if set(state) != expected_bridge | expected_lora | expected_symbolic:
        raise LayerCakeHostError(
            "output balancing requires a fused single-cake symbolic source"
        )
    with torch.no_grad():
        for name in expected_bridge:
            if tuple(parameters[name].shape) != tuple(state[name].shape):
                raise LayerCakeHostError(
                    f"source bridge shape changed for {name}"
                )
            parameters[name].copy_(
                state[name].to(
                    parameters[name].device,
                    dtype=parameters[name].dtype,
                )
            )
        for name in targets:
            wrapper = _resolve_module(model, name)
            if not isinstance(wrapper, LoRAConv1D):
                raise LayerCakeHostError(
                    f"LoRA wrapper was not installed for {name}"
                )
            wrapper.lora_a.copy_(
                state[f"lora.{name}.a"].to(
                    wrapper.lora_a.device,
                    dtype=wrapper.lora_a.dtype,
                )
            )
            wrapper.lora_b.copy_(
                state[f"lora.{name}.b"].to(
                    wrapper.lora_b.device,
                    dtype=wrapper.lora_b.dtype,
                )
            )
    return targets, rank, alpha


@torch.inference_mode()
def _assign_parent_routes(
    parent,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    max_tokens: int,
    batch_size: int = 16,
) -> dict[str, int]:
    route_counts: dict[str, int] = {}
    parent.eval()
    for offset in range(0, len(rows), batch_size):
        selected = rows[offset : offset + batch_size]
        prepared = [
            {**row, "route": 0}
            for row in selected
        ]
        ids, _labels, attention, prompt_lengths, _routes, _ = _batch(
            tokenizer,
            prepared,
            device=device,
            max_tokens=max_tokens,
        )
        result = parent(
            ids,
            attention_mask=attention,
            prompt_lengths=prompt_lengths,
            use_cache=False,
        )
        for row, route in zip(selected, result["task_routes"].tolist()):
            row["route"] = int(route)
            key = str(int(route))
            route_counts[key] = route_counts.get(key, 0) + 1
    return route_counts


def _forward_with_input_embedding(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    task_routes: torch.Tensor,
    input_embedding: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if input_embedding is None:
        return model(
            input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            task_routes=task_routes,
            use_cache=False,
        )
    embedded = F.embedding(input_ids, input_embedding)
    result = model.transformer(
        inputs_embeds=embedded,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = result.last_hidden_state
    summary = model._prompt_summary(
        hidden,
        prompt_lengths=prompt_lengths,
        attention_mask=attention_mask,
    )
    task_logits = model.task_classifier(summary)
    adapted = model._dispatch(hidden, task_routes)
    return {
        "logits": F.linear(adapted, model.output_weight),
        "task_logits": task_logits,
        "hidden": adapted,
    }


def balance_general_outputs(
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
    abi_batch_size: int = 4,
    general_batch_size: int = 4,
    learning_rate: float = 5.0e-5,
    classifier_loss_weight: float = 0.1,
    anchor_loss_weight: float = 1.0e-4,
    abi_prompt_overlap_weight: float = 1.0,
    max_tokens: int = 192,
    recovery_start_step: int = 400,
    recovery_interval: int = 8,
    recovery_horizons: Sequence[int] = (4, 8, 16),
    global_int8_input_fake_quant: bool = False,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Jointly retain ABI and parent-response behavior in existing modules."""

    if min(steps, abi_batch_size, general_batch_size, max_tokens) <= 0:
        raise LayerCakeHostError(
            "balancing steps, batches, and token limit must be positive"
        )
    if (
        learning_rate <= 0
        or classifier_loss_weight < 0
        or anchor_loss_weight < 0
        or abi_prompt_overlap_weight < 0
    ):
        raise LayerCakeHostError("balancing loss settings are invalid")
    if recovery_start_step < 0 or recovery_interval < 0:
        raise LayerCakeHostError("recovery schedule is invalid")
    if recovery_interval and (
        not recovery_horizons
        or any(int(value) <= 0 for value in recovery_horizons)
    ):
        raise LayerCakeHostError("recovery horizons must be positive")

    bundle_path = Path(bundle_path).resolve()
    general_curriculum_path = Path(general_curriculum_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    parent_path = Path(parent_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    source_host_path = Path(source_host_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(
            f"balanced host artifact is immutable: {output_path}"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise LayerCakeHostError(
            "CUDA output balancing was requested but unavailable"
        )
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    rng = random.Random(seed)

    source_manifest_path = source_host_path / "deployment_manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    _validate_deployment_manifest(source_manifest)
    route_bridge = source_manifest["host_delta"].get(
        "sparse_route_bridge", {}
    )
    symbolic = source_manifest["host_delta"].get("symbolic_surface", {})
    if (
        source_manifest["host_delta"].get("bridge_mode")
        != "cakes_lora_fused"
        or route_bridge.get("mode") != "none"
        or route_bridge.get("fused_into_existing_task_cakes") is not True
        or symbolic.get("mode") != "learned_rules_and_schema_realizers"
    ):
        raise LayerCakeHostError(
            "balancing requires the fused single-cake symbolic host"
        )
    source_delta_path = (
        source_host_path / source_manifest["host_delta"]["path"]
    )
    if _sha256_file(source_delta_path) != source_manifest["host_delta"]["sha256"]:
        raise LayerCakeHostError("source host delta is stale or tampered")
    source_state = load_file(str(source_delta_path), device="cpu")
    if (
        _bridge_state_sha256(source_state)
        != source_manifest["host_delta"]["logical_state_sha256_after"]
    ):
        raise LayerCakeHostError("source host logical state changed")

    budget_index = int(
        source_manifest["imported_artifact"]["budget_index"]
    )
    abi_rows, _budget, _bundle = load_english_training_rows(
        bundle_path, budget_index=budget_index
    )
    general_rows = _load_general_rows(
        general_curriculum_path, split="train"
    )
    if len(general_rows) != 2100:
        raise LayerCakeHostError(
            "locked general-English train depth changed"
        )

    model, tokenizer, loaded_manifest, _ = load_host_model(
        layercake_root=layercake_root,
        parent_path=parent_path,
        canonical_abi_path=canonical_abi_path,
        host_path=None,
        device_name=device_name,
    )
    if loaded_manifest is not None:
        raise LayerCakeHostError("sealed parent unexpectedly loaded a host")
    parent_checkpoint_sha = _sha256_file(
        parent_path / "model.safetensors"
    )
    if (
        source_manifest["parent_layercake"]["checkpoint_sha256"]
        != parent_checkpoint_sha
    ):
        raise LayerCakeHostError(
            "source host belongs to a different sealed parent"
        )
    if (
        source_manifest["canonical_semantic_abi_sha256"]
        != _sha256_file(canonical_abi_path)
    ):
        raise LayerCakeHostError(
            "source host belongs to a different canonical ABI"
        )
    transformer_sha_before = module_state_sha256(model.transformer)
    if (
        transformer_sha_before
        != source_manifest["parent_layercake"][
            "transformer_state_sha256_before"
        ]
    ):
        raise LayerCakeHostError("sealed parent transformer identity changed")

    parent = copy.deepcopy(model).eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    parent_route_counts = _assign_parent_routes(
        parent,
        tokenizer,
        general_rows,
        device=device,
        max_tokens=max_tokens,
    )
    del parent
    if device.type == "cuda":
        torch.cuda.empty_cache()

    lora_targets, lora_rank, lora_alpha = _install_source_state(
        model, source_state, source_manifest
    )
    fake_quantized_input_embedding = None
    if global_int8_input_fake_quant:
        embedding = model.output_weight.detach().float()
        scale = (
            embedding.abs().max() / 127.0
        ).clamp_min(1.0e-8)
        fake_quantized_input_embedding = (
            (embedding / scale)
            .round()
            .clamp(-127, 127)
            .mul(scale)
            .to(dtype=model.output_weight.dtype)
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_named: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if name.startswith("task_cakes.") or name.endswith(
            (".lora_a", ".lora_b")
        ):
            parameter.requires_grad_(True)
            trainable_named.append((name, parameter))
    if not trainable_named:
        raise LayerCakeHostError("output balancing has no trainable parameters")
    if any(
        name.startswith("task_classifier.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ):
        raise LayerCakeHostError("output balancing changed the classifier")
    anchors = {
        name: parameter.detach().clone()
        for name, parameter in trainable_named
    }
    trainable = [parameter for _, parameter in trainable_named]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=0.0
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.float16))
        if use_amp
        else (lambda: nullcontext())
    )
    model.train()

    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    cpu_before = process.cpu_times()
    started = time.perf_counter()
    curves = []
    abi_seen: set[str] = set()
    general_seen: set[str] = set()
    abi_supervised_tokens = 0
    general_supervised_tokens = 0
    autonomous_prefix_tokens = 0
    recovery_batches = 0
    skipped_amp_steps = 0
    attempted_batches = 0
    successful_steps = 0
    horizon_counts = {
        str(int(value)): 0 for value in recovery_horizons
    }

    while successful_steps < steps:
        attempted_batches += 1
        if attempted_batches > steps + 1000:
            raise LayerCakeHostError(
                "too many non-finite balancing attempts"
            )
        selected_abi = [
            rng.choice(abi_rows) for _ in range(abi_batch_size)
        ]
        selected_general = [
            rng.choice(general_rows)
            for _ in range(general_batch_size)
        ]
        generated_abi = None
        generated_general = None
        if (
            recovery_interval > 0
            and successful_steps >= recovery_start_step
            and (
                successful_steps - recovery_start_step
            )
            % recovery_interval
            == 0
        ):
            horizon = int(
                recovery_horizons[
                    recovery_batches % len(recovery_horizons)
                ]
            )
            generated_abi = _autonomous_prefixes(
                model,
                tokenizer,
                selected_abi,
                horizon=horizon,
                device=device,
            )
            generated_general = _autonomous_prefixes(
                model,
                tokenizer,
                selected_general,
                horizon=horizon,
                device=device,
            )
            recovery_batches += 1
            horizon_counts[str(horizon)] += 1
            autonomous_prefix_tokens += sum(
                len(value)
                for value in (*generated_abi, *generated_general)
            )
        abi_batch = _batch(
            tokenizer,
            selected_abi,
            device=device,
            max_tokens=max_tokens,
            generated_prefixes=generated_abi,
        )
        general_batch = _batch(
            tokenizer,
            selected_general,
            device=device,
            max_tokens=max_tokens,
            generated_prefixes=generated_general,
        )
        (
            abi_ids,
            abi_labels,
            abi_attention,
            abi_prompt_lengths,
            abi_routes,
            observed_abi,
        ) = abi_batch
        (
            general_ids,
            general_labels,
            general_attention,
            general_prompt_lengths,
            general_routes,
            observed_general,
        ) = general_batch

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            abi_result = _forward_with_input_embedding(
                model,
                input_ids=abi_ids,
                attention_mask=abi_attention,
                prompt_lengths=abi_prompt_lengths,
                task_routes=abi_routes,
                input_embedding=fake_quantized_input_embedding,
            )
            general_result = _forward_with_input_embedding(
                model,
                input_ids=general_ids,
                attention_mask=general_attention,
                prompt_lengths=general_prompt_lengths,
                task_routes=general_routes,
                input_embedding=fake_quantized_input_embedding,
            )
            abi_loss = _equal_record_prompt_overlap_ce(
                abi_result["logits"],
                abi_labels,
                abi_ids,
                abi_prompt_lengths,
                overlap_weight=abi_prompt_overlap_weight,
            )
            general_loss = _equal_record_shifted_ce(
                general_result["logits"], general_labels
            )
            classifier_loss = 0.5 * (
                F.cross_entropy(
                    abi_result["task_logits"], abi_routes
                )
                + F.cross_entropy(
                    general_result["task_logits"], general_routes
                )
            )
            anchor_loss = torch.stack(
                [
                    (parameter - anchors[name])
                    .float()
                    .square()
                    .mean()
                    for name, parameter in trainable_named
                ]
            ).mean()
            loss = (
                abi_loss
                + general_loss
                + classifier_loss_weight * classifier_loss
                + anchor_loss_weight * anchor_loss
            )
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped_amp_steps += 1
            continue
        successful_steps += 1
        abi_supervised_tokens += observed_abi
        general_supervised_tokens += observed_general
        abi_seen.update(str(row["record_id"]) for row in selected_abi)
        general_seen.update(str(row["id"]) for row in selected_general)
        if (
            successful_steps == 1
            or successful_steps % 100 == 0
            or successful_steps == steps
        ):
            curve = {
                "step": successful_steps,
                "total_loss": float(loss.detach()),
                "abi_response_ce": float(abi_loss.detach()),
                "general_response_ce": float(general_loss.detach()),
                "route_classifier_ce": float(
                    classifier_loss.detach()
                ),
                "source_anchor_mse": float(anchor_loss.detach()),
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
    model.eval()
    lora_state = _capture_and_remove_lora(model, lora_targets)
    transformer_sha_after = module_state_sha256(model.transformer)
    if transformer_sha_after != transformer_sha_before:
        raise LayerCakeHostError(
            "sealed parent transformer changed during balancing"
        )
    state = dict(source_state)
    for name, value in model.task_cakes.state_dict().items():
        state[f"task_cakes.{name}"] = (
            value.detach().cpu().contiguous()
        )
    for name, value in lora_state.items():
        state[name] = value
    if SYMBOLIC_SURFACE_STATE_KEY not in state:
        raise LayerCakeHostError(
            "balanced host lost its symbolic substrate"
        )
    _fuse_lora(
        model,
        state,
        lora_targets,
        rank=lora_rank,
        alpha=lora_alpha,
    )
    fused_transformer_sha = module_state_sha256(model.transformer)

    output_path.mkdir(parents=True, exist_ok=False)
    delta_path = output_path / "host_delta.safetensors"
    save_file(state, str(delta_path))
    delta_sha = _sha256_file(delta_path)
    manifest = copy.deepcopy(source_manifest)
    manifest["status"] = (
        "OUTPUT_BALANCED_NOT_YET_SEMANTICALLY_CERTIFIED"
    )
    manifest["host_delta"]["path"] = delta_path.name
    manifest["host_delta"]["sha256"] = delta_sha
    manifest["host_delta"]["bytes"] = delta_path.stat().st_size
    manifest["host_delta"]["logical_state_sha256_before"] = (
        source_manifest["host_delta"]["logical_state_sha256_after"]
    )
    manifest["host_delta"]["logical_state_sha256_after"] = (
        _bridge_state_sha256(state)
    )
    manifest["parent_layercake"][
        "fused_runtime_transformer_state_sha256"
    ] = fused_transformer_sha
    for component in manifest["components"]:
        if component["type"] == (
            "layercake_task_classifier_and_low_rank_cakes"
        ):
            component["sha256"] = delta_sha
    manifest["training"]["general_output_balancing"] = {
        "format": BALANCING_FORMAT,
        "seed": seed,
        "device": str(device),
        "steps": steps,
        "successful_optimizer_steps": successful_steps,
        "skipped_amp_optimizer_steps": skipped_amp_steps,
        "attempted_batches": attempted_batches,
        "abi_batch_size": abi_batch_size,
        "general_batch_size": general_batch_size,
        "learning_rate": learning_rate,
        "classifier_loss_weight": classifier_loss_weight,
        "anchor_loss_weight": anchor_loss_weight,
        "abi_prompt_overlap_weight": abi_prompt_overlap_weight,
        "weight_decay": 0.0,
        "max_tokens": max_tokens,
        "objective": (
            "equal_weight_per_record_ABI_and_general_response_token_CE"
        ),
        "global_int8_input_embedding_fake_quantization": bool(
            global_int8_input_fake_quant
        ),
        "source_llm_loaded": False,
        "source_host_manifest_sha256": source_manifest[
            "manifest_sha256"
        ],
        "general_training_rows": len(general_rows),
        "general_parent_route_counts": parent_route_counts,
        "general_instruction_validation_rows_seen": 0,
        "abi_validation_rows_seen": 0,
        "speed_benchmark_rows_seen": 0,
        "final_test_rows_seen": 0,
        "unique_abi_search_rows_sampled": len(abi_seen),
        "unique_general_train_rows_sampled": len(general_seen),
        "abi_supervised_tokens_seen": abi_supervised_tokens,
        "general_supervised_tokens_seen": general_supervised_tokens,
        "self_generated_prefix_recovery": {
            "start_step": recovery_start_step,
            "interval": recovery_interval,
            "horizons": [
                int(value) for value in recovery_horizons
            ],
            "batches": recovery_batches,
            "horizon_batches": horizon_counts,
            "autonomous_prefix_tokens_seen": autonomous_prefix_tokens,
        },
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable
        ),
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
            "joint_ABI_and_sealed_parent_response_output_balancing"
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
        "source_llm_loaded": False,
        "frozen_transformer_changed": False,
        "classifier_changed": False,
        "symbolic_substrate_changed": False,
        "runtime_modules_added": 0,
        "previous_derivation": previous_derivation,
    }
    manifest["claim_boundary"] = (
        "This artifact proves a disjoint-train output-balancing boundary "
        "and exact component identity. ABI validation, held-out general "
        "validation, native equivalence, and performance remain unproven."
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
    parser.add_argument("--abi-batch-size", type=int, default=4)
    parser.add_argument("--general-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument(
        "--classifier-loss-weight", type=float, default=0.1
    )
    parser.add_argument(
        "--anchor-loss-weight", type=float, default=1.0e-4
    )
    parser.add_argument(
        "--abi-prompt-overlap-weight", type=float, default=1.0
    )
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--recovery-start-step", type=int, default=400)
    parser.add_argument("--recovery-interval", type=int, default=8)
    parser.add_argument("--recovery-horizons", default="4,8,16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--global-int8-input-fake-quant", action="store_true"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = balance_general_outputs(
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
        classifier_loss_weight=args.classifier_loss_weight,
        anchor_loss_weight=args.anchor_loss_weight,
        abi_prompt_overlap_weight=args.abi_prompt_overlap_weight,
        max_tokens=args.max_tokens,
        recovery_start_step=args.recovery_start_step,
        recovery_interval=args.recovery_interval,
        recovery_horizons=tuple(
            int(value)
            for value in args.recovery_horizons.split(",")
            if value.strip()
        ),
        global_int8_input_fake_quant=(
            args.global_int8_input_fake_quant
        ),
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest_sha256": result["manifest_sha256"],
                "host_delta": result["host_delta"],
                "general_output_balancing": result["training"][
                    "general_output_balancing"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
