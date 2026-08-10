"""Read-only native-attention interface oracle for the existing coefficient decoder."""

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
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-native-attention-interface-oracle/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_NATIVE_ATTENTION_INTERFACE_ORACLE"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("native-attention oracle governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"native-attention oracle binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("native-attention oracle output exists or CUDA unavailable")
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
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    model = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(artifact_path), device="cpu"), strict=True, assign=True)
    layer = model.layers[0].cuda().eval()
    examples = sequential.field._examples(root, base, tokenizer)
    calibration = base["calibration"]
    train, validation, calibration_tokens = dual._calibration_examples(
        examples, seed=int(base["training"]["seed"]),
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
    process = psutil.Process(); peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()

    def capture(rows: list[dict]) -> list[dict]:
        nonlocal route_exact, peak_rss
        values = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                host = torch.tensor([row["input_ids"]], dtype=torch.long)
                route = model._select_route(host)
                route_exact += int(route == routed._route(str(row["capability"])))
                source = torch.tensor(
                    [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]], device=device
                )
                teacher_hidden = teacher.model.embed_tokens(source)
                native_attention, target = dual._teacher_components(teacher, 0, teacher_hidden)
                feature = layer.post_attention_norm(native_attention).float()
                gate, up = layer.sparse_gate_up_projection(feature).chunk(2, dim=-1)
                values.append({
                    "record_id": row["record_id"], "capability": row["capability"], "route": route,
                    "attention": native_attention.squeeze(0).float().cpu(),
                    "target": target.squeeze(0).float().cpu(),
                    "feature": feature.squeeze(0).cpu(),
                    "sparse": (F.silu(gate) * up).squeeze(0).cpu(),
                    "residual": (target - native_attention).squeeze(0).float().cpu(),
                })
                peak_rss = max(peak_rss, process.memory_info().rss)
        return values

    train_cache = capture(train); validation_cache = capture(validation)
    mean, covariance, observations = rank_audit.centered_covariance(
        [row["residual"] for row in train_cache], int(protocol["full_width"]), device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    rank = int(protocol["rank"])
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    del covariance, eigenvectors, eigenvalues, teacher
    torch.cuda.empty_cache()
    features = torch.cat([row["feature"] for row in train_cache]).to(device)
    residuals = torch.cat([row["residual"] for row in train_cache]).to(device)
    sparse = torch.cat([row["sparse"] for row in train_cache]).to(device)
    routes = torch.cat([
        torch.full((row["feature"].shape[0],), int(row["route"]), dtype=torch.long)
        for row in train_cache
    ]).to(device)
    targets = (residuals - mean) @ basis
    linear_map, linear_ridge = closed.solve_ridge(features, targets, float(protocol["relative_ridge"]))
    coefficient_residual = targets - features @ linear_map
    route_maps = []; route_ridges = []; route_observations = []
    for route in range(3):
        indices = torch.nonzero(routes == route, as_tuple=False).squeeze(1)
        route_map, ridge = closed.solve_ridge(
            sparse.index_select(0, indices), coefficient_residual.index_select(0, indices),
            float(protocol["relative_ridge"]),
        )
        route_maps.append(route_map); route_ridges.append(ridge); route_observations.append(int(indices.numel()))
    mapped_train = features @ linear_map
    for route in range(3):
        indices = torch.nonzero(routes == route, as_tuple=False).squeeze(1)
        mapped_train.index_add_(0, indices, sparse.index_select(0, indices) @ route_maps[route])
    train_coefficient_relative_rmse = float(
        torch.sqrt(torch.mean((mapped_train - targets).square()))
        / torch.sqrt(torch.mean(targets.square())).clamp_min(1e-8)
    )
    cosines = []; rmses = []; records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_cache:
            feature = row["feature"].to(device); sparse_value = row["sparse"].to(device)
            coefficients = feature @ linear_map + sparse_value @ route_maps[int(row["route"])]
            prediction = row["attention"].to(device) + mean + F.linear(coefficients, basis)
            cosine, relative_rmse = trajectory._metrics(prediction, row["target"].to(device))
            cosines.append(cosine); rmses.append(relative_rmse)
            records.append({"record_id": row["record_id"], "capability": row["capability"], "cosine": cosine, "relative_rmse": relative_rmse})
    artifact_after = sha256_file(artifact_path)
    mean_cosine = sum(cosines) / len(cosines); mean_rmse = sum(rmses) / len(rmses)
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
        "status": "PASS_NATIVE_ATTENTION_INTERFACE_ORACLE" if passed else "FAIL_NATIVE_ATTENTION_INTERFACE_ORACLE",
        "protocol_sha256": sha256_file(protocol_path), "rank": rank, "rank_energy": energy,
        "train_observations": observations, "train_coefficient_relative_rmse": train_coefficient_relative_rmse,
        "linear_effective_ridge": linear_ridge,
        "route_effective_ridge": dict(zip(routed.ROUTES, route_ridges)),
        "route_train_observations": dict(zip(routed.ROUTES, route_observations)),
        "validation": {"records": len(validation), "mean_cosine": mean_cosine, "minimum_cosine": min(cosines), "mean_relative_rmse": mean_rmse, "maximum_relative_rmse": max(rmses), "record_metrics": records},
        "route_correct": route_exact, "gates": gates, "passed": passed,
        "artifact_model_sha256_before": artifact_before, "artifact_model_sha256_after": artifact_after,
        "wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_performed": False, "artifact_written": False, "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only exact-native-attention interface oracle at layer 0 only; no realizable attention, extracted model, autonomous quality, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_ATTENTION_INTERFACE_ORACLE_PROTOCOL_V371.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/native_attention_oracle_v372")
    args = parser.parse_args(); root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
