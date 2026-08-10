"""Fail-fast joint fit for a mixed-rank [768, 1044] replacement prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_existing_attention_refit as coverage
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_rank1044_stable_solver_audit as stable
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-variable-rank-layer1-joint-fit/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_FAIL_FAST_VARIABLE_RANK_LAYER1_JOINT_FIT"
        or protocol.get("rank_schedule_prefix") != [768, 1044]
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("variable-rank layer1 governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"variable-rank layer1 binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("variable-rank layer1 output exists or CUDA unavailable")

    output.mkdir(parents=True)
    set_determinism(int(protocol["training"]["seed"]))
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = root / protocol["artifact"]["directory"]
    artifact_path = artifact / "model.safetensors"
    artifact_before = sha256_file(artifact_path)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str((root / protocol["layercake_host"]).resolve()))

    from layercake.routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveLayer
    from layercake.routed_sparse_rank768_progressive_core_fp16 import (
        PrecisionConformantRoutedSparseRank768ProgressiveCore,
    )
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    model = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(artifact_path), device="cpu"), strict=True, assign=True)
    model_state = model.state_dict()
    layer0_checkpoint = load_file(str(root / protocol["layer0_checkpoint"]["path"]), device="cpu")
    if any(not name.startswith("layers.0.") for name in layer0_checkpoint):
        raise Phase3Error("variable-rank layer0 checkpoint identity changed")
    with torch.no_grad():
        for name, value in layer0_checkpoint.items():
            model_state[name].copy_(value.to(model_state[name].dtype))

    examples = sequential.field._examples(root, base, tokenizer)
    train_rows, validation_rows = coverage.expanded_split(
        examples,
        seed=int(base["training"]["seed"]),
        maximum_tokens=int(protocol["population"]["maximum_sequence_actions"]),
    )
    rows = train_rows + validation_rows
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    for value in teacher.parameters():
        value.requires_grad_(False)

    layer0 = model.layers[0].float().cuda().eval()
    terminal = int(base["source"]["terminal_token_id"])
    cache = {}
    route_exact = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row_index, row in enumerate(rows):
            host_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            route_index = model._select_route(host_ids)
            route_exact += int(route_index == routed._route(str(row["capability"])))
            candidate = model.token_embedding(host_ids).to(device)
            positions = torch.arange(candidate.shape[1], device=device)
            candidate, _, _ = layer0.forward_with_cache(candidate, positions, route_index)
            source_ids = torch.tensor(
                [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]],
                dtype=torch.long,
                device=device,
            )
            native = teacher.model.embed_tokens(source_ids)
            for source_index in range(2):
                _, native = dual._teacher_components(teacher, source_index, native)
            cache[str(row["record_id"])] = (
                candidate.squeeze(0).half().cpu(),
                native.squeeze(0).to(torch.bfloat16).cpu(),
                route_index,
            )
            peak_rss = max(peak_rss, process.memory_info().rss)
            if (row_index + 1) % 500 == 0:
                print(json.dumps({"capture_records": row_index + 1}), flush=True)
    del teacher, layer0
    torch.cuda.empty_cache()

    old_layer1 = model.layers[1].float().cuda().eval()
    train_features = []
    train_residuals = []
    token_routes = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row_index, row in enumerate(train_rows):
            hidden, target, route_index = cache[str(row["record_id"])]
            hidden = hidden.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            positions = torch.arange(hidden.shape[1], device=device)
            attention = routed._attention(old_layer1, hidden, positions)
            feature = old_layer1.post_attention_norm(attention).squeeze(0).float().cpu()
            train_features.append(feature)
            train_residuals.append((target - attention).squeeze(0).float().cpu())
            token_routes.extend([route_index] * feature.shape[0])
            if (row_index + 1) % 500 == 0:
                print(json.dumps({"initialization_records": row_index + 1}), flush=True)

    width = int(protocol["architecture"]["full_width"])
    rank = int(protocol["architecture"]["layer1_residual_rank"])
    mean, covariance, observations = rank_audit.centered_covariance(train_residuals, width, device)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    rank_energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    features = torch.cat(train_features).to(device)
    residuals = torch.cat(train_residuals).to(device)
    coefficient_targets = (residuals - mean) @ basis
    del residuals, covariance, train_features, train_residuals

    solve_diagnostics = []
    stable_solve = stable._stable_solver_factory(
        chunk_tokens=int(protocol["analytic_initialization"]["solver_chunk_tokens"]),
        diagnostics=solve_diagnostics,
    )
    relative_ridge = float(protocol["analytic_initialization"]["relative_ridge"])
    linear_weights, _ = stable_solve(features, coefficient_targets, relative_ridge)
    coefficient_remainder = coefficient_targets - features @ linear_weights
    sparse_weight = old_layer1.sparse_gate_up_projection.weight.detach()
    sparse_chunks = []
    for feature_chunk in features.split(int(protocol["analytic_initialization"]["feature_chunk_tokens"])):
        gate, up = F.linear(feature_chunk, sparse_weight).chunk(2, dim=-1)
        sparse_chunks.append(F.silu(gate) * up)
    sparse_features = torch.cat(sparse_chunks)
    route_tensor = torch.tensor(token_routes, dtype=torch.long, device=device)
    route_weights = []
    for route_index in range(len(protocol["route_names"])):
        indices = torch.nonzero(route_tensor == route_index, as_tuple=False).squeeze(1)
        weights, _ = stable_solve(
            sparse_features.index_select(0, indices),
            coefficient_remainder.index_select(0, indices),
            relative_ridge,
        )
        route_weights.append(weights)

    layer1 = RoutedSparseRank768ProgressiveLayer(
        width,
        int(protocol["architecture"]["bottleneck_width"]),
        int(protocol["architecture"]["attention_heads"]),
        int(protocol["architecture"]["intermediate_size"]),
        residual_rank=rank,
        sparse_width=int(protocol["architecture"]["sparse_width"]),
        routes=len(protocol["route_names"]),
        rms_epsilon=float(protocol["architecture"]["rms_epsilon"]),
        rope_theta=float(protocol["architecture"]["rope_theta"]),
    ).float().cuda()
    old_parameters = dict(old_layer1.named_parameters())
    with torch.no_grad():
        for name, value in layer1.named_parameters():
            if name in old_parameters and old_parameters[name].shape == value.shape:
                value.copy_(old_parameters[name])
        layer1.mlp_residual_mean.copy_(mean)
        layer1.mlp_output_projection.weight.copy_(basis)
        layer1.linear_coefficient_projection.weight.copy_(linear_weights.transpose(0, 1))
        for route_index, weights in enumerate(route_weights):
            layer1.route_coefficient_projections[route_index].weight.copy_(weights.transpose(0, 1))
    del old_layer1, features, coefficient_targets, coefficient_remainder, sparse_features
    del route_tensor, linear_weights, route_weights, basis, mean
    torch.cuda.empty_cache()

    trainable = list(layer1.parameters())
    trainable_parameters = sum(value.numel() for value in trainable)
    if trainable_parameters != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error("variable-rank layer1 trainable boundary changed")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(protocol["training"]["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    curves = []
    layer1.train()
    for step, row in enumerate(train_rows, start=1):
        hidden, target, route_index = cache[str(row["record_id"])]
        hidden = hidden.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, _, _ = layer1.forward_with_cache(
                hidden, torch.arange(hidden.shape[1], device=device), route_index
            )
            left = prediction.float()
            right = target.float()
            relative_mse = torch.mean((left - right).square()) / torch.mean(right.square()).clamp_min(1e-8)
            cosine = F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).mean()
            loss = relative_mse + float(protocol["training"]["cosine_weight"]) * (1 - cosine)
        if not torch.isfinite(loss):
            raise Phase3Error(f"variable-rank layer1 became nonfinite at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(protocol["training"]["gradient_clip_norm"])
        )
        optimizer.step()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(protocol["training"]["curve_interval"]) == 0:
            point = {
                "step": step,
                "loss": float(loss),
                "relative_rmse": float(torch.sqrt(relative_mse)),
                "cosine": float(cosine),
                "gradient_norm": float(gradient_norm),
            }
            curves.append(point)
            print(json.dumps(point), flush=True)

    def evaluate(population: list[dict]) -> dict:
        cosines = []
        rmses = []
        metrics = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in population:
                hidden, target, route_index = cache[str(row["record_id"])]
                hidden = hidden.unsqueeze(0).to(device)
                target = target.unsqueeze(0).to(device)
                prediction, _, _ = layer1.forward_with_cache(
                    hidden, torch.arange(hidden.shape[1], device=device), route_index
                )
                cosine, rmse = trajectory._metrics(prediction, target)
                cosines.append(cosine)
                rmses.append(rmse)
                metrics.append(
                    {
                        "record_id": row["record_id"],
                        "capability": row["capability"],
                        "cosine": cosine,
                        "relative_rmse": rmse,
                    }
                )
        return {
            "records": len(population),
            "mean_cosine": sum(cosines) / len(cosines),
            "minimum_cosine": min(cosines),
            "mean_relative_rmse": sum(rmses) / len(rmses),
            "maximum_relative_rmse": max(rmses),
            "record_metrics": metrics,
        }

    layer1.eval()
    training_metrics = evaluate(train_rows)
    validation_metrics = evaluate(validation_rows)
    artifact_after = sha256_file(artifact_path)
    gates = {
        "rank_energy": rank_energy >= float(protocol["gates"]["rank_energy_minimum"]),
        "validation_mean_cosine": validation_metrics["mean_cosine"]
        >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": validation_metrics["mean_relative_rmse"]
        <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "routes_exact": route_exact == len(rows),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    checkpoint = None
    if passed:
        checkpoint_path = output / "native_trajectory_mixed_rank_layers_00_01.safetensors"
        tensors = {name: value.detach().half().cpu().contiguous() for name, value in layer0_checkpoint.items()}
        tensors.update(
            {
                f"layers.1.{name}": value.detach().half().cpu().contiguous()
                for name, value in layer1.named_parameters()
            }
        )
        save_file(
            tensors,
            str(checkpoint_path),
            metadata={
                "format": FORMAT,
                "protocol_sha256": sha256_file(protocol_path),
                "rank_schedule_prefix": "768,1044",
            },
        )
        checkpoint = {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "parameters": sum(value.numel() for value in tensors.values()),
        }

    result = {
        "format": FORMAT,
        "status": "PASS_VARIABLE_RANK_LAYER1_JOINT_FIT" if passed else "FAIL_VARIABLE_RANK_LAYER1_JOINT_FIT",
        "protocol_sha256": sha256_file(protocol_path),
        "rank_schedule_prefix": [768, 1044],
        "rank1044_energy": rank_energy,
        "analytic_initialization_observations": observations,
        "analytic_solve_diagnostics": solve_diagnostics,
        "training": {
            "steps": len(train_rows),
            "trainable_parameters": trainable_parameters,
            "curves": curves,
        },
        "train": training_metrics,
        "validation": validation_metrics,
        "route_correct": route_exact,
        "gates": gates,
        "passed": passed,
        "checkpoint": checkpoint,
        "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": artifact_after,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "source_blocks_in_checkpoint": 0,
        "teacher_activations_persisted": 0,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Mixed-rank [768,1044] two-layer prefix conformance only; no complete artifact, autonomous quality, physical runtime, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_VARIABLE_RANK_LAYER1_JOINT_PROTOCOL_V404.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/variable_rank_layer1_joint_v405",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
