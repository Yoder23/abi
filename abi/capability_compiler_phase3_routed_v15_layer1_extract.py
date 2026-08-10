"""Fail-fast routed v15 layer-1 extraction on the extracted layer-0 prefix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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


FORMAT = "abi-capability-compiler-phase3-routed-v15-layer1-extract/1"


def _prefix(model, row: dict, example_by_id: dict, device: torch.device):
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    source_ids = torch.tensor(
        [example_by_id[str(row["record_id"])]["source_ids"]], dtype=torch.long, device=device
    )
    route_index = model._select_route(source_ids)
    hidden = model.token_embedding(ids)
    positions = torch.arange(ids.shape[1], device=device)
    hidden, _, _ = model.layers[0].forward_with_cache(hidden, positions, route_index)
    return ids, hidden, positions, route_index


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_FAIL_FAST_ROUTED_V15_LAYER1_EXTRACTION"
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("routed v15 layer1 governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"routed v15 layer1 binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")
    output.mkdir(parents=True)
    device = torch.device("cuda")
    set_determinism(int(protocol["seed"]))
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveCore

    _, tokenizer_type = sequential._types(root, base)
    tokenizer = sequential.field._tokenizer(base, tokenizer_type)
    architecture = dict(base["architecture"])
    architecture.update(
        {
            "residual_rank": int(protocol["architecture"]["residual_rank"]),
            "sparse_width": int(protocol["architecture"]["sparse_width"]),
            "route_names": list(layer0.ROUTES),
        }
    )
    model = RoutedSparseRank768ProgressiveCore(
        fixed_vocab_size=tokenizer.vocab_size, **architecture
    ).bind_tokenizer(tokenizer).to(device)
    substrate = load_file(str(root / base["substrate"]["path"]), device="cpu")
    model.load_state_dict(substrate, strict=False)
    prefix_checkpoint = load_file(str(root / protocol["layer0_checkpoint"]["path"]), device="cpu")
    state = model.state_dict()
    with torch.no_grad():
        for name, value in prefix_checkpoint.items():
            if name in state:
                state[name].copy_(value.to(state[name].dtype))
    primary_checkpoint = load_file(str(root / protocol["primary_attention_checkpoint"]["path"]), device="cpu")
    copied_primary = []
    with torch.no_grad():
        for name, value in primary_checkpoint.items():
            if name.startswith("layers.1.") and name in state and state[name].shape == value.shape:
                state[name].copy_(value.to(state[name].dtype)); copied_primary.append(name)
    expected_primary = {
        "layers.1.attention_input_projection.weight",
        "layers.1.attention_norm.weight",
        "layers.1.attention_output_projection.weight",
        "layers.1.o_proj.weight",
        "layers.1.qkv_proj.weight",
    }
    if set(copied_primary) != expected_primary:
        raise Phase3Error("primary attention checkpoint tensor set changed")
    residual_checkpoint = load_file(str(root / protocol["residual_attention_checkpoint"]["path"]), device="cpu")
    secondary_mapping = {
        "attention_input_projection.weight": "secondary_attention_input_projection.weight",
        "attention_norm.weight": "secondary_attention_norm.weight",
        "attention_output_projection.weight": "secondary_attention_output_projection.weight",
        "o_proj.weight": "secondary_o_proj.weight",
        "qkv_proj.weight": "secondary_qkv_proj.weight",
    }
    copied_secondary = []
    layer = model.layers[1]
    secondary_state = layer.state_dict()
    with torch.no_grad():
        for source_name, target_name in secondary_mapping.items():
            value = residual_checkpoint[source_name]
            if secondary_state[target_name].shape != value.shape:
                raise Phase3Error("secondary attention checkpoint shape changed")
            secondary_state[target_name].copy_(value.to(secondary_state[target_name].dtype))
            copied_secondary.append(source_name)
    if set(copied_secondary) != set(secondary_mapping):
        raise Phase3Error("secondary attention checkpoint tensor set changed")
    model.eval()
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
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    source_layer = teacher.model.layers[1]
    gate_up_weight = source_layer.mlp.gate_up_proj.weight.float()
    down_weight = source_layer.mlp.down_proj.weight.float()
    source_neurons = down_weight.shape[1]
    importance = torch.zeros(source_neurons, device=device)
    features_cpu = []; residuals_cpu = []; token_routes = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids, hidden, positions, route_index = _prefix(model, row, example_by_id, device)
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            attention = layer0._attention(layer, hidden, positions)
            feature = layer.post_attention_norm(attention)
            gate, up = F.linear(feature.float(), gate_up_weight).chunk(2, dim=-1)
            activation = F.silu(gate) * up
            importance += activation.square().sum(dim=(0, 1))
            feature_cpu = feature.squeeze(0).float().cpu()
            features_cpu.append(feature_cpu)
            residuals_cpu.append((teacher_final - attention).squeeze(0).float().cpu())
            token_routes.extend([route_index] * feature_cpu.shape[0])
    importance *= down_weight.square().sum(dim=0)
    selected_count = int(protocol["architecture"]["sparse_width"])
    selected = torch.argsort(importance, descending=True, stable=True)[:selected_count]
    rank = int(protocol["architecture"]["residual_rank"])
    mean, covariance, observations = rank_audit.centered_covariance(
        residuals_cpu, int(protocol["architecture"]["full_width"]), device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    features = torch.cat(features_cpu).to(device)
    coefficients = (torch.cat(residuals_cpu).to(device) - mean) @ basis
    linear_weights, linear_ridge = closed.solve_ridge(features, coefficients, float(protocol["relative_ridge"]))
    correction_targets = coefficients - features @ linear_weights
    selected_gate = gate_up_weight[:source_neurons].index_select(0, selected)
    selected_up = gate_up_weight[source_neurons:].index_select(0, selected)
    sparse_features = torch.cat([
        F.silu(F.linear(feature.to(device), selected_gate)) * F.linear(feature.to(device), selected_up)
        for feature in features_cpu
    ]).float()
    route_weights = []; route_observations = []
    for route_index in range(len(layer0.ROUTES)):
        indices = torch.tensor([i for i, value in enumerate(token_routes) if value == route_index], dtype=torch.long, device=device)
        weights, _ = closed.solve_ridge(
            sparse_features.index_select(0, indices), correction_targets.index_select(0, indices),
            float(protocol["relative_ridge"])
        )
        route_weights.append(weights); route_observations.append(int(indices.numel()))
    with torch.no_grad():
        layer.mlp_residual_mean.copy_(mean)
        layer.mlp_output_projection.weight.copy_(basis)
        layer.linear_coefficient_projection.weight.copy_(linear_weights.T)
        layer.sparse_gate_up_projection.weight.copy_(torch.cat((selected_gate, selected_up), dim=0))
        for route_index, weights in enumerate(route_weights):
            layer.route_coefficient_projections[route_index].weight.copy_(weights.T)
    del covariance, features, coefficients, correction_targets, sparse_features
    attention_rmses = []; attention_cosines = []; rmses = []; cosines = []; route_exact = 0
    records = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids, hidden, positions, route_index = _prefix(model, row, example_by_id, device)
            teacher_attention, teacher_final = dual._teacher_components(teacher, 1, hidden)
            attention = layer0._attention(layer, hidden, positions)
            armse, acos = dual.base._metrics(attention, teacher_attention, hidden)
            attention_rmses.append(float(armse)); attention_cosines.append(float(acos))
            expected_route = layer0._route(str(row["capability"])); route_exact += int(route_index == expected_route)
            prediction = attention + layer._mlp_delta(attention, route_index)
            rmse, cosine = dual.base._metrics(prediction, teacher_final, hidden)
            rmses.append(float(rmse)); cosines.append(float(cosine))
            records.append({"record_id": row["record_id"], "capability": row["capability"], "expected_route": layer0.ROUTES[expected_route], "predicted_route": layer0.ROUTES[route_index], "relative_rmse": float(rmse), "output_cosine": float(cosine)})
    mean_rmse = sum(rmses) / len(rmses); mean_cosine = sum(cosines) / len(cosines)
    gate = protocol["gate"]
    if int(gate["exact_validation_routes_required"]) != len(validation_rows):
        raise Phase3Error("exact route gate does not match the bound validation population")
    passed = mean_rmse <= float(gate["mean_relative_rmse_maximum"]) and mean_cosine >= float(gate["mean_output_cosine_minimum"]) and route_exact == len(validation_rows)
    checkpoint = {
        name: parameter.detach().to(torch.float16).cpu().contiguous()
        for name, parameter in model.named_parameters()
        if name.startswith("layers.0.") or name.startswith("layers.1.") or name.startswith("router.")
    }
    checkpoint_path = output / "routed_v15_layers_00_01.safetensors"
    save_file(checkpoint, str(checkpoint_path), metadata={"format": FORMAT, "protocol_sha256": sha256_file(protocol_path)})
    result = {
        "format": FORMAT,
        "status": "PASS_ROUTED_V15_LAYER1" if passed else "FAIL_ROUTED_V15_LAYER1",
        "protocol_sha256": sha256_file(protocol_path),
        "layer": 1,
        "copied_primary_attention_tensor_keys": len(copied_primary),
        "copied_secondary_attention_tensor_keys": len(copied_secondary),
        "calibration_tokens": calibration_tokens,
        "train_observations": observations,
        "route_train_observations": dict(zip(layer0.ROUTES, route_observations)),
        "basis_rank": rank,
        "basis_energy_explained": float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12)),
        "linear_effective_ridge": linear_ridge,
        "validation": {
            "mean_attention_relative_rmse": sum(attention_rmses) / len(attention_rmses),
            "mean_attention_output_cosine": sum(attention_cosines) / len(attention_cosines),
            "mean_relative_rmse": mean_rmse,
            "maximum_relative_rmse": max(rmses),
            "mean_output_cosine": mean_cosine,
            "minimum_output_cosine": min(cosines),
            "exact_routes": route_exact,
            "passed": passed,
        },
        "record_metrics": records,
        "checkpoint": {"path": checkpoint_path.name, "sha256": sha256_file(checkpoint_path), "parameters": sum(value.numel() for value in checkpoint.values())},
        "source_blocks_in_checkpoint": 0,
        "artifact_promoted": False,
        "training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Fail-fast routed v15 layer-1 extraction only; no full artifact, English quality, physical runtime, certificate, or superiority claim.",
    }
    _write_immutable(output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_LAYER1_EXTRACT_PROTOCOL_V307.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_routed_v15/layer1_v308")
    args = parser.parse_args(); root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
