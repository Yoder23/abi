"""No-artifact realizable native-basis coefficient-map audit for layer 0."""

from __future__ import annotations

from collections import defaultdict
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
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-native-basis-coefficient-audit/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_ARTIFACT_NATIVE_BASIS_COEFFICIENT_MAPPING_AUDIT"
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("native-basis coefficient governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"native-basis coefficient binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("native-basis coefficient output exists or CUDA unavailable")
    output.mkdir(parents=True)
    set_determinism(int(protocol["seed"]))
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
    cfg = base["calibration"]
    train, validation, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    terminal = int(base["source"]["terminal_token_id"])
    residuals = []
    features = []
    sparse_features = []
    routes = []
    route_exact = 0
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train:
            host = torch.tensor([row["input_ids"]], dtype=torch.long)
            route = model._select_route(host)
            route_exact += int(route == routed._route(str(row["capability"])))
            hidden = embedding(host).to(device)
            attention = routed._attention(layer, hidden, torch.arange(hidden.shape[1], device=device))
            feature = layer.post_attention_norm(attention)
            gate, up = layer.sparse_gate_up_projection(feature).float().chunk(2, dim=-1)
            source = torch.tensor([[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]], device=device)
            teacher_hidden = teacher.model.embed_tokens(source)
            _, target = dual._teacher_components(teacher, 0, teacher_hidden)
            residuals.append((target - attention).squeeze(0).float().cpu())
            features.append(feature.squeeze(0).float().cpu())
            sparse_features.append((F.silu(gate) * up).squeeze(0).cpu())
            routes.extend([route] * feature.shape[1])
            peak_rss = max(peak_rss, process.memory_info().rss)
    mean, covariance, observations = rank_audit.centered_covariance(
        residuals, int(protocol["full_width"]), device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    rank = int(protocol["rank"])
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    del covariance, eigenvectors, eigenvalues, teacher
    torch.cuda.empty_cache()
    feature_tensor = torch.cat(features).to(device)
    residual_tensor = torch.cat(residuals).to(device)
    target_coefficients = (residual_tensor - mean) @ basis
    linear_map, linear_ridge = closed.solve_ridge(
        feature_tensor, target_coefficients, float(protocol["relative_ridge"])
    )
    coefficient_residual = target_coefficients - feature_tensor @ linear_map
    sparse_tensor = torch.cat(sparse_features).to(device)
    route_maps = []
    route_ridges = []
    route_observations = []
    route_tensor = torch.tensor(routes, dtype=torch.long, device=device)
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
    mapped_train = feature_tensor @ linear_map
    for route in range(3):
        indices = torch.nonzero(route_tensor == route, as_tuple=False).squeeze(1)
        mapped_train.index_add_(0, indices, sparse_tensor.index_select(0, indices) @ route_maps[route])
    coefficient_relative_rmse = float(
        torch.sqrt(torch.mean((mapped_train - target_coefficients) ** 2))
        / torch.sqrt(torch.mean(target_coefficients**2)).clamp_min(1e-8)
    )
    del feature_tensor, residual_tensor, target_coefficients, coefficient_residual, sparse_tensor, route_tensor, mapped_train
    torch.cuda.empty_cache()
    cosines = []
    rmses = []
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher = AutoModelForCausalLM.from_pretrained(
            base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).cuda().eval()
        for row in validation:
            host = torch.tensor([row["input_ids"]], dtype=torch.long)
            route = model._select_route(host)
            route_exact += int(route == routed._route(str(row["capability"])))
            hidden = embedding(host).to(device)
            attention = routed._attention(layer, hidden, torch.arange(hidden.shape[1], device=device))
            feature = layer.post_attention_norm(attention).float()
            gate, up = layer.sparse_gate_up_projection(feature).chunk(2, dim=-1)
            coefficients = feature @ linear_map + (F.silu(gate) * up) @ route_maps[route]
            prediction = attention.float() + mean + F.linear(coefficients, basis)
            source = torch.tensor([[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]], device=device)
            teacher_hidden = teacher.model.embed_tokens(source)
            _, target = dual._teacher_components(teacher, 0, teacher_hidden)
            cosine, relative_rmse = trajectory._metrics(prediction, target)
            cosines.append(cosine); rmses.append(relative_rmse)
            records.append({"record_id": row["record_id"], "capability": row["capability"], "cosine": cosine, "relative_rmse": relative_rmse})
    artifact_after = sha256_file(artifact_path)
    mean_cosine = sum(cosines) / len(cosines)
    mean_rmse = sum(rmses) / len(rmses)
    gates = {
        "rank_energy": energy >= float(protocol["gates"]["rank_energy_minimum"]),
        "validation_mean_cosine": mean_cosine >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": mean_rmse <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "routes_exact": route_exact == len(train) + len(validation),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    result = {
        "format": FORMAT,
        "status": "PASS_NATIVE_BASIS_COEFFICIENT_MAPPING" if passed else "FAIL_NATIVE_BASIS_COEFFICIENT_MAPPING",
        "protocol_sha256": sha256_file(protocol_path),
        "rank": rank,
        "rank_energy": energy,
        "train_observations": observations,
        "train_coefficient_relative_rmse": coefficient_relative_rmse,
        "linear_effective_ridge": linear_ridge,
        "route_effective_ridge": dict(zip(routed.ROUTES, route_ridges)),
        "route_train_observations": dict(zip(routed.ROUTES, route_observations)),
        "validation": {"records": len(validation), "mean_cosine": mean_cosine, "minimum_cosine": min(cosines), "mean_relative_rmse": mean_rmse, "maximum_relative_rmse": max(rmses), "record_metrics": records},
        "route_correct": route_exact,
        "gates": gates,
        "passed": passed,
        "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": artifact_after,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_performed": False,
        "artifact_written": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "No-artifact realizable linear-plus-route coefficient-map audit only; no extracted model, autonomous quality, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_BASIS_COEFFICIENT_AUDIT_PROTOCOL_V358.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/coefficient_audit_v359")
    args = parser.parse_args(); root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
