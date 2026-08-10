"""No-artifact feasibility audit for route-projection trajectory retargeting."""

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

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_routed_v15_layer0_extract as layer0
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-routed-v16-trajectory-retargeting/1"
HOST_EXTERNAL_OFFSET = 4
HOST_EOS = 2


def source_token_id(action: int, terminal: int) -> int:
    if action == HOST_EOS:
        return terminal
    if action < HOST_EXTERNAL_OFFSET:
        raise Phase3Error("unmappable host-special action in trajectory sequence")
    return action - HOST_EXTERNAL_OFFSET


def _metrics(candidate: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    left, right = candidate.float(), target.float()
    cosine = F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item()
    relative_rmse = torch.sqrt(torch.mean((left - right) ** 2)).div(
        torch.sqrt(torch.mean(right**2)).clamp_min(1e-8)
    ).item()
    return float(cosine), float(relative_rmse)


def solve_from_moments(gram: torch.Tensor, cross: torch.Tensor, relative_ridge: float):
    scale = float(torch.trace(gram) / gram.shape[0])
    ridge = float(relative_ridge) * scale
    solution = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype),
        cross,
    )
    return solution, ridge


def _load(root: Path, protocol_path: Path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_ARTIFACT_ROUTE_PROJECTION_TRAJECTORY_RETARGETING"
        or protocol.get("device") != "cuda"
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("trajectory-retargeting governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"trajectory-retargeting binding changed: {name}")
    return protocol


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = _load(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("trajectory output exists or CUDA is unavailable")
    output.mkdir(parents=True)
    set_determinism(int(protocol["seed"]))
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = (root / protocol["artifact"]["directory"]).resolve()
    artifact_path = artifact / "model.safetensors"
    artifact_before = sha256_file(artifact_path)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    sys.path.insert(0, str(layercake_root))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    candidate = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    incompatible = candidate.load_state_dict(load_file(str(artifact_path), device="cpu"), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Phase3Error("trajectory artifact strict load failed")
    candidate = candidate.cuda().eval()
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)

    examples = sequential.field._examples(root, base, tokenizer)
    cfg = base["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    if len(train_rows) != int(protocol["population"]["train_records"]) or len(validation_rows) != int(protocol["population"]["validation_records"]):
        raise Phase3Error("trajectory calibration population changed")
    rows = train_rows + validation_rows
    validation_ids = {str(row["record_id"]) for row in validation_rows}

    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(base["source"]["parameter_count"]):
        raise Phase3Error("trajectory source identity changed")
    terminal = int(base["source"]["terminal_token_id"])
    student_cache = {}
    teacher_cache = {}
    route_by_id = {}
    route_exact = 0
    with torch.inference_mode():
        for row in rows:
            record_id = str(row["record_id"])
            host_ids = list(row["input_ids"])
            source_ids = [source_token_id(value, terminal) for value in host_ids]
            host_tensor = torch.tensor([host_ids], dtype=torch.long, device=device)
            source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)
            route = candidate._select_route(host_tensor)
            route_exact += int(route == layer0._route(str(row["capability"])))
            route_by_id[record_id] = route
            student_cache[record_id] = candidate.token_embedding(host_tensor).squeeze(0).half().cpu()
            teacher_cache[record_id] = teacher.model.embed_tokens(source_tensor).squeeze(0).to(torch.bfloat16).cpu()
    if route_exact != len(rows):
        raise Phase3Error("trajectory router failed on calibration population")

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    layer_results = []
    fitted_values = 0
    for layer_index, (candidate_layer, _) in enumerate(zip(candidate.layers, teacher.model.layers)):
        sparse_width = candidate_layer.sparse_width
        residual_rank = candidate_layer.residual_rank
        grams = [torch.zeros((sparse_width, sparse_width), dtype=torch.float32, device=device) for _ in layer0.ROUTES]
        crosses = [torch.zeros((sparse_width, residual_rank), dtype=torch.float32, device=device) for _ in layer0.ROUTES]
        observations = [0 for _ in layer0.ROUTES]
        teacher_next = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                record_id = str(row["record_id"])
                student = student_cache[record_id].unsqueeze(0).to(device)
                source = teacher_cache[record_id].unsqueeze(0).to(device)
                _, source_final = dual._teacher_components(teacher, layer_index, source)
                teacher_next[record_id] = source_final.squeeze(0).to(torch.bfloat16).cpu()
                if record_id not in validation_ids:
                    positions = torch.arange(student.shape[1], device=device)
                    attention = layer0._attention(candidate_layer, student, positions)
                    feature = candidate_layer.post_attention_norm(attention)
                    gate, up = candidate_layer.sparse_gate_up_projection(feature).float().chunk(2, dim=-1)
                    sparse = (F.silu(gate) * up).reshape(-1, sparse_width)
                    linear = candidate_layer.linear_coefficient_projection(feature).float().reshape(-1, residual_rank)
                    target_residual = source_final.float() - attention.float() - candidate_layer.mlp_residual_mean.float()
                    target_coefficients = target_residual.reshape(-1, target_residual.shape[-1]) @ candidate_layer.mlp_output_projection.weight.float()
                    desired = target_coefficients - linear
                    route = route_by_id[record_id]
                    grams[route].add_(sparse.T @ sparse)
                    crosses[route].add_(sparse.T @ desired)
                    observations[route] += sparse.shape[0]
        ridges = []
        with torch.no_grad():
            for route, projection in enumerate(candidate_layer.route_coefficient_projections):
                if observations[route] < sparse_width:
                    raise Phase3Error(f"insufficient trajectory observations for route {route}")
                solution, ridge = solve_from_moments(grams[route], crosses[route], float(protocol["relative_ridge"]))
                projection.weight.copy_(solution.T.to(projection.dtype))
                ridges.append(ridge)
                fitted_values += projection.weight.numel()
        train_cosines = []
        train_rmses = []
        validation_cosines = []
        validation_rmses = []
        student_next = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                record_id = str(row["record_id"])
                student = student_cache[record_id].unsqueeze(0).to(device)
                positions = torch.arange(student.shape[1], device=device)
                value, _, _ = candidate_layer.forward_with_cache(student, positions, route_by_id[record_id])
                target = teacher_next[record_id].unsqueeze(0).to(device)
                cosine, relative_rmse = _metrics(value, target)
                if record_id in validation_ids:
                    validation_cosines.append(cosine); validation_rmses.append(relative_rmse)
                else:
                    train_cosines.append(cosine); train_rmses.append(relative_rmse)
                student_next[record_id] = value.squeeze(0).half().cpu()
        student_cache = student_next
        teacher_cache = teacher_next
        layer_result = {
            "layer": layer_index,
            "route_observations": dict(zip(layer0.ROUTES, observations)),
            "effective_ridge": dict(zip(layer0.ROUTES, ridges)),
            "train_mean_global_cosine": sum(train_cosines) / len(train_cosines),
            "train_mean_global_relative_rmse": sum(train_rmses) / len(train_rmses),
            "validation_mean_global_cosine": sum(validation_cosines) / len(validation_cosines),
            "validation_minimum_global_cosine": min(validation_cosines),
            "validation_mean_global_relative_rmse": sum(validation_rmses) / len(validation_rmses),
            "validation_maximum_global_relative_rmse": max(validation_rmses),
        }
        layer_results.append(layer_result)
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps(layer_result), flush=True)

    top1 = 0
    logit_cosines = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            record_id = str(row["record_id"])
            student = student_cache[record_id].unsqueeze(0).to(device)
            source = teacher_cache[record_id].unsqueeze(0).to(device)
            candidate_logits = candidate.lm_head(candidate.final_norm(student[:, -1]))
            teacher_logits = teacher.lm_head(teacher.model.norm(source[:, -1]))
            common = candidate_logits[:, HOST_EXTERNAL_OFFSET:]
            common_teacher = teacher_logits[:, :common.shape[-1]]
            logit_cosines.append(float(F.cosine_similarity(common.float(), common_teacher.float()).item()))
            action = int(candidate_logits.argmax(dim=-1).item())
            try:
                predicted = source_token_id(action, terminal)
            except Phase3Error:
                predicted = -1
            top1 += int(predicted == int(teacher_logits.argmax(dim=-1).item()))
    gates = {
        "all_layer_validation_global_cosine": min(value["validation_mean_global_cosine"] for value in layer_results) >= float(protocol["gates"]["every_layer_mean_global_cosine_minimum"]),
        "final_validation_global_cosine": layer_results[-1]["validation_mean_global_cosine"] >= float(protocol["gates"]["final_mean_global_cosine_minimum"]),
        "first_token_top1_agreement": top1 / len(validation_rows) >= float(protocol["gates"]["first_token_top1_agreement_minimum"]),
        "routes_exact": route_exact == len(rows),
        "artifact_unchanged": sha256_file(artifact_path) == artifact_before,
    }
    passed = all(gates.values())
    result = {
        "format": FORMAT,
        "status": "PASS_TRAJECTORY_RETARGETING_FEASIBILITY" if passed else "FAIL_TRAJECTORY_RETARGETING_FEASIBILITY",
        "protocol_sha256": sha256_file(protocol_path),
        "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": sha256_file(artifact_path),
        "calibration_train_records": len(train_rows),
        "calibration_validation_records": len(validation_rows),
        "calibration_tokens": calibration_tokens,
        "route_correct": route_exact,
        "analytically_fitted_existing_parameter_values": fitted_values,
        "new_parameter_values": 0,
        "persisted_parameter_values": 0,
        "layer_results": layer_results,
        "final_mean_global_cosine": layer_results[-1]["validation_mean_global_cosine"],
        "final_mean_common_logit_cosine": sum(logit_cosines) / len(logit_cosines),
        "first_token_top1_agreement": top1 / len(validation_rows),
        "gates": gates,
        "passed": passed,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "teacher_loaded_for_read_only_calibration": True,
        "gradient_training_performed": False,
        "artifact_written": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "No-artifact development-independent calibration feasibility only; no autonomous quality, runtime, Phase 3, minimum-information, or superiority claim."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V16_TRAJECTORY_RETARGETING_PROTOCOL_V343.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_routed_v16/trajectory_retargeting_v344")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
