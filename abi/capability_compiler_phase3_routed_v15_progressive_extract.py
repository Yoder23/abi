"""Preregistered fail-fast extraction of routed v15 layers 2 through 31."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as layer0
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-routed-v15-progressive-extract/2"

_PRIMARY_NAMES = {
    "attention_input_projection.weight",
    "attention_norm.weight",
    "attention_output_projection.weight",
    "o_proj.weight",
    "qkv_proj.weight",
}
_SECONDARY_NAMES = {
    "secondary_attention_input_projection.weight",
    "secondary_attention_norm.weight",
    "secondary_attention_output_projection.weight",
    "secondary_o_proj.weight",
    "secondary_qkv_proj.weight",
}


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_FAIL_FAST_ROUTED_V15_SOURCE_ALIGNED_PROGRESSIVE_EXTRACTION"
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("routed v15 progressive governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"routed v15 progressive binding changed: {name}")
    if list(protocol.get("layers", [])) != list(range(3, 32)):
        raise Phase3Error("routed v15 progressive layer schedule changed")
    return protocol, sha256_file(path)


def _instantiate(root: Path, protocol: dict[str, Any], device: torch.device):
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveCore

    _, tokenizer_type = sequential._types(root, base)
    tokenizer = sequential.field._tokenizer(base, tokenizer_type)
    architecture = dict(base["architecture"])
    architecture.update(
        residual_rank=int(protocol["architecture"]["residual_rank"]),
        sparse_width=int(protocol["architecture"]["sparse_width"]),
        route_names=list(layer0.ROUTES),
    )
    model = RoutedSparseRank768ProgressiveCore(
        fixed_vocab_size=tokenizer.vocab_size, **architecture
    ).bind_tokenizer(tokenizer)
    substrate = load_file(str(root / base["substrate"]["path"]), device="cpu")
    model.load_state_dict(substrate, strict=False)
    first_layer = int(protocol["layers"][0])
    expected_prefix = {
        name for name, _ in model.named_parameters()
        if name.startswith("router.")
        or any(name.startswith(f"layers.{index}.") for index in range(first_layer))
    }
    loaded: set[str] = set()
    state = model.state_dict()
    with torch.no_grad():
        for entry in protocol["prefix_checkpoints"]:
            checkpoint = load_file(str(root / entry["path"]), device="cpu")
            for name, value in checkpoint.items():
                if name not in expected_prefix or name not in state or state[name].shape != value.shape:
                    raise Phase3Error(f"prefix checkpoint boundary changed: {name}")
                if name in loaded:
                    raise Phase3Error(f"duplicate prefix tensor: {name}")
                state[name].copy_(value.to(state[name].dtype))
                loaded.add(name)
    if loaded != expected_prefix:
        raise Phase3Error("prefix checkpoint is incomplete")
    return model.to(device), tokenizer, base, len(substrate), len(loaded)


def _initial_cache(
    model,
    rows: list[dict[str, Any]],
    example_by_id: dict[str, dict],
    device,
    prefix_layers: int = 2,
):
    cache: dict[str, tuple[torch.Tensor, int]] = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            source_ids = torch.tensor(
                [example_by_id[str(row["record_id"])]["source_ids"]], dtype=torch.long, device=device
            )
            route_index = model._select_route(source_ids)
            positions = torch.arange(ids.shape[1], device=device)
            hidden = model.token_embedding(ids)
            for index in range(prefix_layers):
                hidden, _, _ = model.layers[index].forward_with_cache(hidden, positions, route_index)
            cache[str(row["record_id"])] = (hidden.squeeze(0).to(torch.float16).cpu(), route_index)
    return cache


def _targets(teacher, layer_index: int, rows: list[dict[str, Any]], cache, device):
    values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            hidden = cache[str(row["record_id"])][0].unsqueeze(0).to(device)
            attention, final = dual._teacher_components(teacher, layer_index, hidden)
            values[str(row["record_id"])] = (
                attention.squeeze(0).to(torch.float16).cpu(),
                final.squeeze(0).to(torch.float16).cpu(),
            )
    return values


def _fit_attention(layer, source_layer, rows, cache, targets, layer_index, protocol, device):
    training = protocol["training"]
    with torch.no_grad():
        layer.attention_output_projection.weight.zero_()
        layer.secondary_attention_output_projection.weight.zero_()
    trainable = []
    for name, parameter in layer.named_parameters():
        enabled = name in _PRIMARY_NAMES or name in _SECONDARY_NAMES
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    if {name for name, parameter in layer.named_parameters() if parameter.requires_grad} != (
        _PRIMARY_NAMES | _SECONDARY_NAMES
    ):
        raise Phase3Error("progressive attention trainable boundary changed")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["attention_steps_per_layer"])
    offset = int(training["record_offset_base"]) + layer_index * steps
    curves = []
    layer.train()
    for step in range(steps):
        row = rows[(step + offset) % len(rows)]
        record_id = str(row["record_id"])
        hidden = cache[record_id][0].unsqueeze(0).to(device)
        target_attention = targets[record_id][0].unsqueeze(0).to(device)
        target_final = targets[record_id][1].unsqueeze(0).to(device)
        positions = torch.arange(hidden.shape[1], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            attention = layer0._attention(layer, hidden, positions)
            feature = layer.post_attention_norm(attention)
            feature_target = source_layer.post_attention_layernorm(target_attention)
            final = attention + source_layer.mlp(feature)
            attention_rmse, attention_cosine = dual.base._metrics(attention, target_attention, hidden)
            final_rmse, final_cosine = dual.base._metrics(final, target_final, hidden)
            feature_rmse = torch.sqrt(
                (feature.float() - feature_target.float()).square().mean()
                / feature_target.float().square().mean().clamp_min(1e-8)
            )
            loss = (
                attention_rmse.square() + final_rmse.square() + feature_rmse.square()
                + float(training["cosine_weight"]) * (2.0 - attention_cosine - final_cosine)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
        optimizer.step()
        if step == 0 or (step + 1) % int(training["curve_interval"]) == 0:
            curves.append(
                {
                    "step": step + 1,
                    "attention_relative_rmse": float(attention_rmse.detach()),
                    "feature_relative_rmse": float(feature_rmse.detach()),
                    "final_relative_rmse": float(final_rmse.detach()),
                    "final_cosine": float(final_cosine.detach()),
                    "loss": float(loss.detach()),
                }
            )
    layer.eval()
    del optimizer
    return curves


def _derive_maps(layer, source_layer, rows, cache, targets, protocol, device):
    source_gate_up = source_layer.mlp.gate_up_proj.weight.float()
    source_down = source_layer.mlp.down_proj.weight.float()
    source_neurons = source_down.shape[1]
    importance = torch.zeros(source_neurons, device=device)
    features_cpu = []
    residuals_cpu = []
    token_routes = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            record_id = str(row["record_id"])
            hidden = cache[record_id][0].unsqueeze(0).to(device)
            route_index = cache[record_id][1]
            positions = torch.arange(hidden.shape[1], device=device)
            attention = layer0._attention(layer, hidden, positions)
            feature = layer.post_attention_norm(attention)
            gate, up = F.linear(feature.float(), source_gate_up).chunk(2, dim=-1)
            activation = F.silu(gate) * up
            importance += activation.square().sum(dim=(0, 1))
            value = feature.squeeze(0).float().cpu()
            features_cpu.append(value)
            target_final = targets[record_id][1]
            residuals_cpu.append(target_final.float() - attention.squeeze(0).float().cpu())
            token_routes.extend([route_index] * value.shape[0])
    importance *= source_down.square().sum(dim=0)
    selected = torch.argsort(importance, descending=True, stable=True)[
        : int(protocol["architecture"]["sparse_width"])
    ]
    rank = int(protocol["architecture"]["residual_rank"])
    _, covariance, observations = rank_audit.centered_covariance(
        residuals_cpu, int(protocol["architecture"]["full_width"]), device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    features = torch.cat(features_cpu).to(device)
    residuals = torch.cat(residuals_cpu).to(device)
    selected_gate = source_gate_up[:source_neurons].index_select(0, selected)
    selected_up = source_gate_up[source_neurons:].index_select(0, selected)
    sparse_features = torch.cat(
        [
            F.silu(F.linear(value.to(device), selected_gate))
            * F.linear(value.to(device), selected_up)
            for value in features_cpu
        ]
    ).float()
    selected_down = source_down.index_select(1, selected)
    selected_output = sparse_features @ selected_down.T
    remaining = residuals - selected_output
    mean = remaining.mean(dim=0)
    coefficients = (remaining - mean) @ basis
    linear_weights, linear_ridge = closed.solve_ridge(
        features, coefficients, float(protocol["training"]["relative_ridge"])
    )
    exact_sparse_coefficients = selected_down.T @ basis
    route_observations = []
    for route_index in range(len(layer0.ROUTES)):
        route_observations.append(sum(value == route_index for value in token_routes))
    with torch.no_grad():
        layer.mlp_residual_mean.copy_(mean)
        layer.mlp_output_projection.weight.copy_(basis)
        layer.linear_coefficient_projection.weight.copy_(linear_weights.T)
        layer.sparse_gate_up_projection.weight.copy_(torch.cat((selected_gate, selected_up), dim=0))
        for route_projection in layer.route_coefficient_projections:
            route_projection.weight.copy_(exact_sparse_coefficients.T)
    energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    return observations, energy, linear_ridge, dict(zip(layer0.ROUTES, route_observations))


def _validate(layer, rows, cache, targets, protocol, device):
    attention_rmses = []
    attention_cosines = []
    rmses = []
    cosines = []
    route_exact = 0
    records = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            record_id = str(row["record_id"])
            hidden = cache[record_id][0].unsqueeze(0).to(device)
            route_index = cache[record_id][1]
            positions = torch.arange(hidden.shape[1], device=device)
            teacher_attention = targets[record_id][0].unsqueeze(0).to(device)
            teacher_final = targets[record_id][1].unsqueeze(0).to(device)
            attention = layer0._attention(layer, hidden, positions)
            armse, acos = dual.base._metrics(attention, teacher_attention, hidden)
            prediction = attention + layer._mlp_delta(attention, route_index)
            rmse, cosine = dual.base._metrics(prediction, teacher_final, hidden)
            expected = layer0._route(str(row["capability"]))
            route_exact += int(route_index == expected)
            attention_rmses.append(float(armse)); attention_cosines.append(float(acos))
            rmses.append(float(rmse)); cosines.append(float(cosine))
            records.append(
                {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "expected_route": layer0.ROUTES[expected],
                    "predicted_route": layer0.ROUTES[route_index],
                    "relative_rmse": float(rmse),
                    "output_cosine": float(cosine),
                }
            )
    gate = protocol["gate"]
    if int(gate["exact_validation_routes_required"]) != len(rows):
        raise Phase3Error("exact route gate does not match validation population")
    mean_rmse = sum(rmses) / len(rmses)
    mean_cosine = sum(cosines) / len(cosines)
    passed = (
        mean_rmse <= float(gate["mean_relative_rmse_maximum"])
        and mean_cosine >= float(gate["mean_output_cosine_minimum"])
        and route_exact == len(rows)
    )
    return {
        "mean_attention_relative_rmse": sum(attention_rmses) / len(attention_rmses),
        "mean_attention_output_cosine": sum(attention_cosines) / len(attention_cosines),
        "mean_relative_rmse": mean_rmse,
        "maximum_relative_rmse": max(rmses),
        "mean_output_cosine": mean_cosine,
        "minimum_output_cosine": min(cosines),
        "exact_routes": route_exact,
        "passed": passed,
    }, records


def _advance_cache(layer, rows, cache, device):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            record_id = str(row["record_id"])
            hidden, route_index = cache[record_id]
            hidden = hidden.unsqueeze(0).to(device)
            positions = torch.arange(hidden.shape[1], device=device)
            value, _, _ = layer.forward_with_cache(hidden, positions, route_index)
            cache[record_id] = (value.squeeze(0).to(torch.float16).cpu(), route_index)


def execute(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol, protocol_sha = _load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")
    output.mkdir(parents=True)
    device = torch.device("cuda")
    set_determinism(int(protocol["training"]["seed"]))
    model, tokenizer, base, substrate_keys, prefix_keys = _instantiate(root, protocol, device)
    examples = sequential.field._examples(root, base, tokenizer)
    example_by_id = {str(row["record_id"]): row for row in examples}
    cfg = base["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    all_rows = train_rows + validation_rows
    cache = _initial_cache(
        model, all_rows, example_by_id, device, prefix_layers=int(protocol["layers"][0])
    )
    route_exact = sum(
        cache[str(row["record_id"])][1] == layer0._route(str(row["capability"]))
        for row in validation_rows
    )
    if route_exact != len(validation_rows):
        raise Phase3Error("bound source-prompt router failed before extraction")
    base_source = base["source"]
    load_started = time.perf_counter()
    teacher = AutoModelForCausalLM.from_pretrained(
        base_source["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(
        base_source["parameter_count"]
    ):
        raise Phase3Error("loaded source parameter count changed")
    teacher_load_seconds = time.perf_counter() - load_started
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    layer_results = []
    completed = []
    for layer_index in protocol["layers"]:
        layer = model.layers[layer_index]
        source_layer = teacher.model.layers[layer_index]
        targets = _targets(teacher, layer_index, all_rows, cache, device)
        curves = _fit_attention(
            layer, source_layer, train_rows, cache, targets, layer_index, protocol, device
        )
        observations, energy, ridge, route_observations = _derive_maps(
            layer, source_layer, train_rows, cache, targets, protocol, device
        )
        validation, records = _validate(
            layer, validation_rows, cache, targets, protocol, device
        )
        layer_directory = output / f"layer_{layer_index:02d}"
        layer_directory.mkdir()
        prefix = f"layers.{layer_index}."
        checkpoint = {
            name: parameter.detach().to(torch.float16).cpu().contiguous()
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        }
        checkpoint_path = layer_directory / f"routed_v15_layer_{layer_index:02d}.safetensors"
        save_file(
            checkpoint,
            str(checkpoint_path),
            metadata={"format": FORMAT, "protocol_sha256": protocol_sha, "layer": str(layer_index)},
        )
        layer_result = {
            "layer": layer_index,
            "status": "PASS" if validation["passed"] else "FAIL",
            "attention_steps": int(protocol["training"]["attention_steps_per_layer"]),
            "curves": curves,
            "train_observations": observations,
            "route_train_observations": route_observations,
            "basis_energy_explained": energy,
            "linear_effective_ridge": ridge,
            "decoder_rule": "source_aligned_exact_selected_neurons_plus_linear_remaining_residual",
            "route_maps_identical_by_construction": True,
            "validation": validation,
            "record_metrics": records,
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(output)).replace("\\", "/"),
                "sha256": sha256_file(checkpoint_path),
                "parameters": sum(value.numel() for value in checkpoint.values()),
            },
        }
        _write_immutable(
            layer_directory / "metadata.json",
            json.dumps(layer_result, indent=2, sort_keys=True).encode() + b"\n",
        )
        layer_results.append(layer_result)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if not validation["passed"]:
            break
        completed.append(layer_index)
        _advance_cache(layer, all_rows, cache, device)
        del targets
        torch.cuda.empty_cache()
    all_passed = completed == protocol["layers"]
    result = {
        "format": FORMAT,
        "status": "PASS_ROUTED_V15_LAYERS_02_31" if all_passed else "FAIL_ROUTED_V15_PROGRESSIVE_LAYER",
        "protocol_sha256": protocol_sha,
        "layers_scheduled": protocol["layers"],
        "layers_completed": completed,
        "first_failed_layer": None if all_passed else layer_results[-1]["layer"],
        "calibration_tokens": calibration_tokens,
        "calibration_train_records": len(train_rows),
        "calibration_validation_records": len(validation_rows),
        "exact_initial_routes": route_exact,
        "copied_substrate_tensor_keys": substrate_keys,
        "loaded_prefix_tensor_keys": prefix_keys,
        "teacher_load_seconds": teacher_load_seconds,
        "fit_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_process_rss_bytes": peak_rss,
        "layer_results": layer_results,
        "source_blocks_in_checkpoints": 0,
        "teacher_required_at_inference": False,
        "artifact_promoted": False,
        "training_performed": True,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Fail-fast routed v15 layers2-31 local extraction only; no assembled artifact, English quality, physical runtime, certificate, or superiority claim.",
    }
    _write_immutable(
        output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_SOURCE_ALIGNED_PROGRESSIVE_PROTOCOL_V319.json",
    )
    parser.add_argument(
        "--output", default="results/abi_capability_compiler_phase3_routed_v15/source_aligned_progressive_v320"
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
