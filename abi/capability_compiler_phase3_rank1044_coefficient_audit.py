"""Read-only realizability audit for the derived rank-1044 layer-1 residual coefficients."""

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
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_existing_attention_refit as coverage
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-rank1044-coefficient-audit/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_RANK1044_COEFFICIENT_AUDIT"
        or protocol.get("analytic_fit_authorized") is not True
        or protocol.get("gradient_training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
        or int(protocol.get("rank", 0)) != 1044
    ):
        raise Phase3Error("rank1044 coefficient governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"rank1044 coefficient binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("rank1044 coefficient output exists or CUDA unavailable")

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
    layer1 = model.layers[1].float().cuda().eval()
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

    terminal = int(base["source"]["terminal_token_id"])
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    route_exact = 0

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
                attention = routed._attention(layer1, candidate, positions)
                feature = layer1.post_attention_norm(attention)
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
                    "feature": feature.squeeze(0).float().cpu(),
                    "residual": (native - attention).squeeze(0).float().cpu(),
                }
                if validation:
                    item["attention"] = attention.squeeze(0).float().cpu()
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

    width = int(protocol["full_width"])
    rank = int(protocol["rank"])
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
    del residuals, covariance
    relative_ridge = float(protocol["relative_ridge"])
    linear_weights, linear_ridge = closed.solve_ridge(features, coefficient_targets, relative_ridge)
    coefficient_remainder = coefficient_targets - features @ linear_weights

    sparse_weight = layer1.sparse_gate_up_projection.weight.detach()
    sparse_chunks = []
    for feature_chunk in features.split(int(protocol["feature_chunk_tokens"])):
        gate, up = F.linear(feature_chunk, sparse_weight).chunk(2, dim=-1)
        sparse_chunks.append(F.silu(gate) * up)
    sparse_features = torch.cat(sparse_chunks)
    token_routes = torch.cat(
        [
            torch.full((row["feature"].shape[0],), int(row["route"]), dtype=torch.long)
            for row in train_cache
        ]
    ).to(device)
    route_weights = []
    route_ridges = []
    route_observations = []
    for route_index in range(len(protocol["route_names"])):
        indices = torch.nonzero(token_routes == route_index, as_tuple=False).squeeze(1)
        weights, ridge = closed.solve_ridge(
            sparse_features.index_select(0, indices),
            coefficient_remainder.index_select(0, indices),
            relative_ridge,
        )
        route_weights.append(weights)
        route_ridges.append(ridge)
        route_observations.append(int(indices.numel()))

    training_squared_error = 0.0
    training_target_squared = 0.0
    offset = 0
    for row in train_cache:
        count = row["feature"].shape[0]
        predicted = features[offset : offset + count] @ linear_weights
        predicted = predicted + sparse_features[offset : offset + count] @ route_weights[int(row["route"])]
        target = coefficient_targets[offset : offset + count]
        training_squared_error += float((predicted - target).square().sum())
        training_target_squared += float(target.square().sum())
        offset += count
    training_coefficient_relative_rmse = (
        training_squared_error / max(training_target_squared, 1e-12)
    ) ** 0.5

    cosines = []
    rmses = []
    coefficient_squared_error = 0.0
    coefficient_target_squared = 0.0
    record_metrics = []
    with torch.inference_mode():
        for row in validation_cache:
            feature = row["feature"].to(device)
            gate, up = F.linear(feature, sparse_weight).chunk(2, dim=-1)
            sparse_feature = F.silu(gate) * up
            coefficient_prediction = feature @ linear_weights
            coefficient_prediction = coefficient_prediction + sparse_feature @ route_weights[int(row["route"])]
            residual = row["residual"].to(device)
            coefficient_target = (residual - mean) @ basis
            coefficient_squared_error += float((coefficient_prediction - coefficient_target).square().sum())
            coefficient_target_squared += float(coefficient_target.square().sum())
            prediction = row["attention"].to(device) + mean + coefficient_prediction @ basis.transpose(0, 1)
            cosine, rmse = trajectory._metrics(prediction, row["target"].to(device))
            cosines.append(cosine)
            rmses.append(rmse)
            record_metrics.append(
                {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "route": int(row["route"]),
                    "cosine": cosine,
                    "relative_rmse": rmse,
                }
            )

    validation_coefficient_relative_rmse = (
        coefficient_squared_error / max(coefficient_target_squared, 1e-12)
    ) ** 0.5
    artifact_after = sha256_file(artifact_path)
    mean_cosine = sum(cosines) / len(cosines)
    mean_rmse = sum(rmses) / len(rmses)
    gates = {
        "rank_energy": rank_energy >= float(protocol["gates"]["rank_energy_minimum"]),
        "validation_mean_cosine": mean_cosine
        >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": mean_rmse
        <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "routes_exact": route_exact == len(train_rows) + len(validation_rows),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    result = {
        "format": FORMAT,
        "status": "PASS_RANK1044_COEFFICIENT_REALIZATION_AUDIT" if passed else "FAIL_RANK1044_COEFFICIENT_REALIZATION_AUDIT",
        "protocol_sha256": sha256_file(protocol_path),
        "rank": rank,
        "rank_energy": rank_energy,
        "relative_ridge": relative_ridge,
        "linear_effective_ridge": linear_ridge,
        "route_effective_ridges": route_ridges,
        "train_observations": observations,
        "route_observations": route_observations,
        "training_coefficient_relative_rmse": training_coefficient_relative_rmse,
        "validation_coefficient_relative_rmse": validation_coefficient_relative_rmse,
        "validation": {
            "records": len(validation_rows),
            "mean_cosine": mean_cosine,
            "minimum_cosine": min(cosines),
            "mean_relative_rmse": mean_rmse,
            "maximum_relative_rmse": max(rmses),
            "record_metrics": record_metrics,
        },
        "route_correct": route_exact,
        "gates": gates,
        "passed": passed,
        "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": artifact_after,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "analytic_fit_performed": True,
        "gradient_training_performed": False,
        "artifact_written": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only fixed-coverage rank-1044 linear-plus-hard-route coefficient realizability audit only; no installed host, checkpoint, physical runtime, autonomous quality, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_RANK1044_COEFFICIENT_AUDIT_PROTOCOL_V399.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/rank1044_coefficient_audit_v400",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
