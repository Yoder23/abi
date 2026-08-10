"""No-artifact fixed-budget nonlinear native-coefficient audit for layer 0."""

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
from torch import nn
import torch.nn.functional as F

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-fixed-budget-nonlinear-coefficient-audit/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_ARTIFACT_FIXED_BUDGET_NONLINEAR_COEFFICIENT_AUDIT"
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("fixed-budget nonlinear governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"fixed-budget nonlinear binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("fixed-budget nonlinear output exists or CUDA unavailable")
    output.mkdir(parents=True)
    seed = int(protocol["seed"])
    set_determinism(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = (root / protocol["artifact"]["directory"]).resolve()
    artifact_path = artifact / "model.safetensors"
    artifact_before = sha256_file(artifact_path)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str((root / protocol["layercake_host"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    model = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(artifact_path), device="cpu"), strict=True, assign=True)
    layer = model.layers[0].cuda().eval()
    embedding = model.token_embedding
    examples = sequential.field._examples(root, base, tokenizer)
    calibration = base["calibration"]
    train, validation, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(calibration["train_records_per_capability"]),
        validation_per_capability=int(calibration["validation_records_per_capability"]),
        maximum_tokens=int(calibration["maximum_sequence_tokens"]),
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    terminal = int(base["source"]["terminal_token_id"])
    route_exact = 0
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()

    def capture(rows: list[dict]) -> list[dict]:
        nonlocal route_exact, peak_rss
        captured = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                host = torch.tensor([row["input_ids"]], dtype=torch.long)
                route = model._select_route(host)
                route_exact += int(route == routed._route(str(row["capability"])))
                hidden = embedding(host).to(device)
                attention = routed._attention(layer, hidden, torch.arange(hidden.shape[1], device=device))
                feature = layer.post_attention_norm(attention)
                gate, up = layer.sparse_gate_up_projection(feature).float().chunk(2, dim=-1)
                source = torch.tensor(
                    [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]],
                    device=device,
                )
                teacher_hidden = teacher.model.embed_tokens(source)
                _, target = dual._teacher_components(teacher, 0, teacher_hidden)
                captured.append({
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "route": route,
                    "attention": attention.squeeze(0).float().cpu(),
                    "feature": feature.squeeze(0).float().cpu(),
                    "sparse": (F.silu(gate) * up).squeeze(0).cpu(),
                    "residual": (target - attention).squeeze(0).float().cpu(),
                    "target": target.squeeze(0).float().cpu(),
                })
                peak_rss = max(peak_rss, process.memory_info().rss)
        return captured

    train_cache = capture(train)
    validation_cache = capture(validation)
    residual_batches = [row["residual"] for row in train_cache]
    mean, covariance, observations = rank_audit.centered_covariance(
        residual_batches, int(protocol["full_width"]), device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    rank = int(protocol["rank"])
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    del covariance, eigenvectors, eigenvalues, teacher, residual_batches
    torch.cuda.empty_cache()

    feature_tensor = torch.cat([row["feature"] for row in train_cache]).to(device)
    residual_tensor = torch.cat([row["residual"] for row in train_cache]).to(device)
    sparse_tensor = torch.cat([row["sparse"] for row in train_cache]).to(device)
    route_tensor = torch.cat([
        torch.full((row["feature"].shape[0],), int(row["route"]), dtype=torch.long)
        for row in train_cache
    ]).to(device)
    target_coefficients = (residual_tensor - mean) @ basis
    coefficient_scale = torch.sqrt(torch.mean(target_coefficients.square(), dim=0)).clamp_min(1e-6)
    normalized_targets = target_coefficients / coefficient_scale
    hidden_width = int(protocol["nonlinear_map"]["hidden_width"])
    first = nn.Linear(int(protocol["full_width"]), hidden_width, bias=False).to(device)
    second = nn.Linear(hidden_width, rank, bias=False).to(device)
    trainable = list(first.parameters()) + list(second.parameters())
    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["steps"]), eta_min=float(training["minimum_learning_rate"])
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    order = torch.randperm(observations, generator=generator)
    cursor = 0
    curves = []
    first.train(); second.train()
    for step in range(int(training["steps"])):
        batch_size = int(training["batch_tokens"])
        if cursor + batch_size > observations:
            order = torch.randperm(observations, generator=generator)
            cursor = 0
        indices = order[cursor:cursor + batch_size].to(device)
        cursor += batch_size
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = second(F.silu(first(feature_tensor.index_select(0, indices))))
            target_batch = normalized_targets.index_select(0, indices)
            loss = F.mse_loss(prediction.float(), target_batch)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
        optimizer.step(); scheduler.step()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 0 or (step + 1) % int(training["curve_interval"]) == 0:
            point = {
                "step": step + 1,
                "normalized_coefficient_mse": float(loss),
                "gradient_norm": float(gradient_norm),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
            curves.append(point)
            print(json.dumps(point), flush=True)
    first.eval(); second.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        nonlinear_train = second(F.silu(first(feature_tensor))).float() * coefficient_scale
    coefficient_residual = target_coefficients - nonlinear_train
    route_maps = []
    route_ridges = []
    route_observations = []
    for route in range(3):
        indices = torch.nonzero(route_tensor == route, as_tuple=False).squeeze(1)
        route_map, route_ridge = closed.solve_ridge(
            sparse_tensor.index_select(0, indices),
            coefficient_residual.index_select(0, indices),
            float(protocol["relative_ridge"]),
        )
        route_maps.append(route_map)
        route_ridges.append(route_ridge)
        route_observations.append(int(indices.numel()))
    mapped_train = nonlinear_train.clone()
    for route in range(3):
        indices = torch.nonzero(route_tensor == route, as_tuple=False).squeeze(1)
        mapped_train.index_add_(
            0, indices, sparse_tensor.index_select(0, indices) @ route_maps[route]
        )
    train_coefficient_relative_rmse = float(
        torch.sqrt(torch.mean((mapped_train - target_coefficients).square()))
        / torch.sqrt(torch.mean(target_coefficients.square())).clamp_min(1e-8)
    )
    cosines = []
    rmses = []
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_cache:
            feature = row["feature"].to(device)
            sparse = row["sparse"].to(device)
            coefficients = second(F.silu(first(feature))).float() * coefficient_scale
            coefficients = coefficients + sparse @ route_maps[int(row["route"])]
            prediction = row["attention"].to(device) + mean + F.linear(coefficients, basis)
            target = row["target"].to(device)
            cosine, relative_rmse = trajectory._metrics(prediction, target)
            cosines.append(cosine); rmses.append(relative_rmse)
            records.append({
                "record_id": row["record_id"], "capability": row["capability"],
                "cosine": cosine, "relative_rmse": relative_rmse,
            })
    artifact_after = sha256_file(artifact_path)
    mean_cosine = sum(cosines) / len(cosines)
    mean_rmse = sum(rmses) / len(rmses)
    nonlinear_values = sum(parameter.numel() for parameter in trainable)
    gates = {
        "rank_energy": energy >= float(protocol["gates"]["rank_energy_minimum"]),
        "parameter_budget": nonlinear_values <= int(protocol["nonlinear_map"]["maximum_values"]),
        "validation_mean_cosine": mean_cosine >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": mean_rmse <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "routes_exact": route_exact == len(train) + len(validation),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    result = {
        "format": FORMAT,
        "status": "PASS_FIXED_BUDGET_NONLINEAR_COEFFICIENT_MAPPING" if passed else "FAIL_FIXED_BUDGET_NONLINEAR_COEFFICIENT_MAPPING",
        "protocol_sha256": sha256_file(protocol_path),
        "rank": rank,
        "rank_energy": energy,
        "train_observations": observations,
        "nonlinear_map": {"hidden_width": hidden_width, "values": nonlinear_values},
        "training": {"steps": int(training["steps"]), "curves": curves},
        "train_coefficient_relative_rmse": train_coefficient_relative_rmse,
        "route_effective_ridge": dict(zip(routed.ROUTES, route_ridges)),
        "route_train_observations": dict(zip(routed.ROUTES, route_observations)),
        "validation": {
            "records": len(validation), "mean_cosine": mean_cosine,
            "minimum_cosine": min(cosines), "mean_relative_rmse": mean_rmse,
            "maximum_relative_rmse": max(rmses), "record_metrics": records,
        },
        "route_correct": route_exact,
        "gates": gates,
        "passed": passed,
        "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": artifact_after,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_performed": True,
        "artifact_written": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "No-artifact fixed-budget nonlinear coefficient capacity audit at layer 0 only; no extracted model, autonomous quality, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FIXED_BUDGET_NONLINEAR_COEFFICIENT_AUDIT_PROTOCOL_V361.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/nonlinear_coefficient_audit_v362")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
