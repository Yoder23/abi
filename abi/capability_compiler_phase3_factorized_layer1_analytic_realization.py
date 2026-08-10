"""Analytically realize layer 1 behind the fixed factorized attention."""

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
from . import capability_compiler_phase3_factorized_attention_residual_span as span
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_rank1044_stable_solver_audit as stable
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-factorized-layer1-analytic-realization/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_FACTORIZED_LAYER1_ANALYTIC_REALIZATION"
        or protocol.get("device") != "cuda"
        or protocol.get("optimizer_training_authorized") is not False
        or protocol.get("component_write") != "PASS_ONLY"
        or protocol.get("source_block_promotion") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("factorized layer1 realization governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"factorized layer1 realization binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("factorized layer1 realization output exists or CUDA unavailable")

    output.mkdir(parents=True)
    set_determinism(int(protocol["seed"]))
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = root / protocol["artifact"]["directory"]
    artifact_path = artifact / "model.safetensors"
    artifact_before = sha256_file(artifact_path)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str((root / protocol["layercake_host"]).resolve()))

    from layercake.routed_sparse_rank768_progressive_core_fp16 import (
        PrecisionConformantRoutedSparseRank768ProgressiveCore,
    )
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    model = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(artifact_path), device="cpu"), strict=True, assign=True)
    state = model.state_dict()
    prefix = load_file(str(root / protocol["layer0_checkpoint"]["path"]), device="cpu")
    with torch.no_grad():
        for name, value in prefix.items():
            state[name].copy_(value.to(state[name].dtype))
    layer0 = model.layers[0].float().cuda().eval()
    host_layer1 = model.layers[1].float().cuda().eval()

    examples = sequential.field._examples(root, base, tokenizer)
    train_rows, validation_rows = coverage.expanded_split(
        examples,
        seed=int(base["training"]["seed"]),
        maximum_tokens=int(protocol["population"]["maximum_sequence_actions"]),
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    for value in teacher.parameters():
        value.requires_grad_(False)
    source_layer1 = teacher.model.layers[1]
    source_attention = source_layer1.self_attn
    attention_cfg = protocol["factorized_attention"]
    heads = int(attention_cfg["source_heads"])
    head_dimension = int(attention_cfg["source_head_dimension"])
    factor_ranks, factor_energy, runtime_factors = span._factors(
        source_attention,
        heads,
        head_dimension,
        float(attention_cfg["operator_energy_threshold"]),
    )
    if factor_ranks != [int(value) for value in attention_cfg["locked_rank_schedule"]]:
        raise Phase3Error("factorized layer1 attention schedule changed")

    terminal = int(base["source"]["terminal_token_id"])
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    route_exact = 0

    def factorized_attention(candidate: torch.Tensor) -> torch.Tensor:
        length = candidate.shape[1]
        position_ids = torch.arange(length, device=device)[None]
        position_embeddings = teacher.model.rotary_emb(candidate, position_ids)
        mask = dual.base._causal_mask(length, device=device, dtype=candidate.dtype)
        normalized = source_layer1.input_layernorm(candidate)
        _, weights = source_attention(
            hidden_states=normalized,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        delta = torch.zeros_like(candidate, dtype=torch.float32)
        for head, (value_factor, output_factor) in enumerate(runtime_factors):
            values = F.linear(normalized.float(), value_factor)
            delta.add_(F.linear(weights[:, head].float() @ values, output_factor))
        return candidate.float() + delta

    def capture(rows: list[dict], *, validation: bool) -> list[dict]:
        nonlocal route_exact, peak_rss
        values = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, row in enumerate(rows):
                host_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
                route_index = model._select_route(host_ids)
                route_exact += int(route_index == routed._route(str(row["capability"])))
                candidate = model.token_embedding(host_ids).to(device)
                positions = torch.arange(candidate.shape[1], device=device)
                candidate, _, _ = layer0.forward_with_cache(candidate, positions, route_index)
                attention = factorized_attention(candidate)
                feature = host_layer1.post_attention_norm(attention).squeeze(0).float().cpu()
                gate, up = F.linear(
                    feature.to(device), host_layer1.sparse_gate_up_projection.weight.float()
                ).chunk(2, dim=-1)
                sparse = (F.silu(gate) * up).cpu()
                source_ids = torch.tensor(
                    [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]],
                    dtype=torch.long,
                    device=device,
                )
                native = teacher.model.embed_tokens(source_ids)
                for source_index in range(2):
                    _, native = dual._teacher_components(teacher, source_index, native)
                item = {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "route": route_index,
                    "feature": feature,
                    "sparse": sparse,
                    "residual": (native.float() - attention).squeeze(0).cpu(),
                }
                if validation:
                    item["attention"] = attention.squeeze(0).cpu()
                    item["target"] = native.squeeze(0).float().cpu()
                values.append(item)
                peak_rss = max(peak_rss, process.memory_info().rss)
                if (index + 1) % 500 == 0:
                    print(json.dumps({"capture_records": index + 1}), flush=True)
        return values

    train_cache = capture(train_rows, validation=False)
    validation_cache = capture(validation_rows, validation=True)
    del teacher
    torch.cuda.empty_cache()

    width = int(protocol["architecture"]["full_width"])
    rank = int(protocol["architecture"]["residual_rank"])
    mean, covariance, observations = rank_audit.centered_covariance(
        [row["residual"] for row in train_cache], width, device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    rank_energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    features = torch.cat([row["feature"] for row in train_cache]).to(device)
    residuals = torch.cat([row["residual"] for row in train_cache]).to(device)
    coefficient_targets = (residuals - mean) @ basis
    sparse_features = torch.cat([row["sparse"] for row in train_cache]).to(device)
    route_tensor = torch.cat(
        [torch.full((row["feature"].shape[0],), row["route"], dtype=torch.long) for row in train_cache]
    ).to(device)
    del residuals, covariance

    diagnostics = []
    solve = stable._stable_solver_factory(
        chunk_tokens=int(protocol["analytic_fit"]["solver_chunk_tokens"]),
        diagnostics=diagnostics,
    )
    ridge = float(protocol["analytic_fit"]["relative_ridge"])
    linear_weights, _ = solve(features, coefficient_targets, ridge)
    remainder = coefficient_targets - features @ linear_weights
    route_weights = []
    for route_index in range(int(protocol["architecture"]["routes"])):
        indices = torch.nonzero(route_tensor == route_index, as_tuple=False).squeeze(1)
        weights, _ = solve(
            sparse_features.index_select(0, indices),
            remainder.index_select(0, indices),
            ridge,
        )
        route_weights.append(weights)

    train_prediction = features @ linear_weights
    for route_index, weights in enumerate(route_weights):
        indices = torch.nonzero(route_tensor == route_index, as_tuple=False).squeeze(1)
        train_prediction.index_add_(
            0,
            indices,
            sparse_features.index_select(0, indices) @ weights,
        )
    train_coefficient_rmse = float(
        torch.sqrt(
            (train_prediction - coefficient_targets).square().mean()
            / coefficient_targets.square().mean().clamp_min(1e-8)
        )
    )

    cosines = []
    rmses = []
    record_metrics = []
    with torch.inference_mode():
        for row in validation_cache:
            feature = row["feature"].to(device)
            sparse_feature = row["sparse"].to(device)
            coefficients = feature @ linear_weights + sparse_feature @ route_weights[row["route"]]
            prediction = row["attention"].to(device) + mean + coefficients @ basis.transpose(0, 1)
            cosine, rmse = trajectory._metrics(prediction, row["target"].to(device))
            cosines.append(cosine)
            rmses.append(rmse)
            record_metrics.append(
                {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "cosine": cosine,
                    "relative_rmse": rmse,
                }
            )

    artifact_after = sha256_file(artifact_path)
    mean_cosine = sum(cosines) / len(cosines)
    mean_rmse = sum(rmses) / len(rmses)
    objective_valid = len(diagnostics) == 4 and all(
        row["penalized_objective_ratio_to_zero"] <= 1.0 + float(protocol["analytic_fit"]["objective_tolerance"])
        for row in diagnostics
    )
    gates = {
        "fixed_attention_energy": factor_energy >= float(attention_cfg["operator_energy_threshold"]),
        "residual_rank_energy": rank_energy >= float(protocol["gates"]["residual_rank_energy_minimum"]),
        "numeric_objectives_valid": objective_valid,
        "validation_mean_cosine": mean_cosine >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": mean_rmse <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "routes_exact": route_exact == len(train_rows) + len(validation_rows),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    component_path = output / "layer1_component.safetensors"
    component_written = False
    component_sha = None
    if passed:
        qkv = source_attention.qkv_proj.weight.detach().float().cpu()
        total_rank = sum(factor_ranks)
        value_factors = torch.cat([item[0].cpu() for item in runtime_factors], dim=0)
        output_factors = torch.cat([item[1].cpu() for item in runtime_factors], dim=1)
        offsets = [0]
        for value in factor_ranks:
            offsets.append(offsets[-1] + value)
        tensors = {
            "source_input_norm.weight": source_layer1.input_layernorm.weight.detach().float().cpu(),
            "attention_q.weight": qkv[:heads * head_dimension],
            "attention_k.weight": qkv[heads * head_dimension:2 * heads * head_dimension],
            "attention_value_factors.weight": value_factors,
            "attention_output_factors.weight": output_factors,
            "attention_rank_offsets": torch.tensor(offsets, dtype=torch.int64),
            "residual_mean": mean.float().cpu(),
            "residual_basis": basis.float().cpu(),
            "coefficient_linear.weight": linear_weights.transpose(0, 1).float().cpu(),
            "post_attention_norm.weight": host_layer1.post_attention_norm.weight.detach().float().cpu(),
            "sparse_gate_up.weight": host_layer1.sparse_gate_up_projection.weight.detach().float().cpu(),
            "route_coefficient.weight": torch.stack([value.transpose(0, 1) for value in route_weights]).float().cpu(),
        }
        if value_factors.shape != (total_rank, width) or output_factors.shape != (width, total_rank):
            raise Phase3Error("factorized layer1 serialized V/O shape changed")
        save_file(tensors, str(component_path))
        component_written = True
        component_sha = sha256_file(component_path)

    total_rank = sum(factor_ranks)
    active_madds = (
        2 * width * head_dimension * heads
        + 2 * width * total_rank
        + width * rank
        + width * rank
        + 2 * width * int(protocol["architecture"]["sparse_width"])
        + int(protocol["architecture"]["sparse_width"]) * rank
    )
    result = {
        "format": FORMAT,
        "status": "PASS_FACTORIZED_LAYER1_ANALYTIC_REALIZATION" if passed else "FAIL_FACTORIZED_LAYER1_ANALYTIC_REALIZATION",
        "protocol_sha256": sha256_file(protocol_path),
        "factorized_attention_rank_schedule": factor_ranks,
        "factorized_attention_total_rank": total_rank,
        "factorized_attention_energy": factor_energy,
        "residual_rank": rank,
        "residual_rank_energy": rank_energy,
        "train_observations": observations,
        "train_coefficient_relative_rmse": train_coefficient_rmse,
        "solve_diagnostics": diagnostics,
        "validation": {
            "records": len(validation_rows),
            "mean_cosine": mean_cosine,
            "minimum_cosine": min(cosines),
            "mean_relative_rmse": mean_rmse,
            "maximum_relative_rmse": max(rmses),
            "record_metrics": record_metrics,
        },
        "physical_accounting": {
            "active_projection_multiply_adds_per_token": active_madds,
            "component_parameters": sum(value.numel() for value in tensors.values()) if passed else None,
            "key_cache_width": width,
            "value_cache_width": total_rank,
        },
        "route_correct": route_exact,
        "gates": gates,
        "passed": passed,
        "component_written": component_written,
        "component_path": component_path.name if component_written else None,
        "component_sha256": component_sha,
        "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": artifact_after,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "analytic_fit_performed": True,
        "optimizer_training_performed": False,
        "sweep_performed": False,
        "source_blocks_promoted": 0,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Single held-out layer-1 analytic realization behind the fixed factorized attention; a passing component is not installed in the host and has no physical runtime, autonomous, complete-model, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FACTORIZED_LAYER1_ANALYTIC_REALIZATION_PROTOCOL_V420.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/factorized_layer1_analytic_v421")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
