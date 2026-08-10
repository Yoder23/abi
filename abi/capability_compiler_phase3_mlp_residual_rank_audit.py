"""Read-only intrinsic-rank audit for the measured source-MLP residual limit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_causal_field_core as field
from . import capability_compiler_phase3_dual_path_local_fit as dual
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-mlp-residual-rank-audit/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_SOURCE_MLP_RANK_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("MLP residual rank-audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"MLP residual rank-audit binding changed: {name}")
    return protocol, sha256_file(path)


def _tokenizer(root: Path, protocol: dict[str, Any]):
    _, tokenizer_type = dual._types(root, protocol)
    return field._tokenizer(protocol, tokenizer_type)


def _rows(root: Path, protocol: dict[str, Any], tokenizer: Any):
    examples = field._examples(root, protocol, tokenizer)
    cfg = protocol["calibration"]
    return dual._calibration_examples(
        examples,
        seed=int(protocol["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )


def centered_covariance(
    batches: list[torch.Tensor], width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    total = torch.zeros(width, dtype=torch.float32, device=device)
    second = torch.zeros(width, width, dtype=torch.float32, device=device)
    observations = 0
    for batch in batches:
        values = batch.float().to(device)
        total.add_(values.sum(dim=0))
        second.add_(values.transpose(0, 1) @ values)
        observations += values.shape[0]
    if observations < 2:
        raise Phase3Error("insufficient MLP residual observations")
    mean = total / observations
    covariance = (second - observations * torch.outer(mean, mean)) / (observations - 1)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    return mean, covariance, observations


def project_with_basis(values: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    centered = values.float() - mean
    return mean + (centered @ basis) @ basis.transpose(0, 1)


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol, protocol_sha = load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("MLP rank audit requires the preregistered CUDA device")
    device = torch.device("cuda")
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.use_deterministic_algorithms(True)
    tokenizer = _tokenizer(root, protocol)
    train_rows, validation_rows, calibration_tokens = _rows(root, protocol, tokenizer)
    substrate = load_file(str(root / protocol["substrate"]["path"]), device="cpu")
    embedding = substrate["token_embedding.weight"].to(device)

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    teacher = AutoModelForCausalLM.from_pretrained(
        protocol["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(protocol["source"]["parameter_count"]):
        raise Phase3Error("loaded source parameter count changed")
    teacher_load_seconds = time.perf_counter() - load_started

    started = time.perf_counter()
    train_deltas: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
            hidden = F.embedding(ids, embedding).unsqueeze(0)
            attention, final = dual._teacher_components(teacher, 0, hidden)
            train_deltas.append((final - attention).squeeze(0).float().cpu())
    width = int(protocol["width"])
    mean, covariance, train_observations = centered_covariance(train_deltas, width, device)
    del train_deltas
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    eigenvectors = eigenvectors.flip(1)
    total_train_energy = float(eigenvalues.sum())

    ranks = [int(value) for value in protocol["tested_ranks"]]
    maximum_rank = max(ranks)
    top_basis = eigenvectors[:, :maximum_rank].contiguous()
    accumulators = {
        rank: {
            "error_energy": 0.0,
            "target_centered_energy": 0.0,
            "relative_rmse": [],
            "final_cosine": [],
        }
        for rank in ranks
    }
    validation_observations = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
            hidden = F.embedding(ids, embedding).unsqueeze(0)
            attention, final = dual._teacher_components(teacher, 0, hidden)
            delta = (final - attention).squeeze(0).float()
            validation_observations += delta.shape[0]
            centered = delta - mean
            centered_energy = float(centered.square().sum())
            for rank in ranks:
                basis = top_basis[:, :rank]
                reconstruction = project_with_basis(delta, mean, basis)
                error = float((delta - reconstruction).square().sum())
                reconstructed_final = attention.float() + reconstruction.unsqueeze(0)
                relative_rmse, cosine = dual.base._metrics(reconstructed_final, final.float(), hidden.float())
                accumulator = accumulators[rank]
                accumulator["error_energy"] += error
                accumulator["target_centered_energy"] += centered_energy
                accumulator["relative_rmse"].append(float(relative_rmse))
                accumulator["final_cosine"].append(float(cosine))

    rank_results = []
    selected_rank = None
    for rank in ranks:
        accumulator = accumulators[rank]
        explained = 1.0 - accumulator["error_energy"] / max(
            accumulator["target_centered_energy"], 1e-12
        )
        training_explained = float(eigenvalues[:rank].sum()) / max(total_train_energy, 1e-12)
        mean_rmse = sum(accumulator["relative_rmse"]) / len(accumulator["relative_rmse"])
        mean_cosine = sum(accumulator["final_cosine"]) / len(accumulator["final_cosine"])
        passed = (
            explained >= float(protocol["selection_gate"]["validation_centered_energy_minimum"])
            and mean_cosine >= float(protocol["selection_gate"]["oracle_final_cosine_minimum"])
            and mean_rmse <= float(protocol["selection_gate"]["oracle_relative_rmse_maximum"])
            and training_explained - explained
            <= float(protocol["selection_gate"]["training_validation_energy_gap_maximum"])
        )
        rank_results.append(
            {
                "rank": rank,
                "training_centered_energy_explained": training_explained,
                "validation_centered_energy_explained": explained,
                "training_validation_energy_gap": training_explained - explained,
                "oracle_mean_final_relative_rmse": mean_rmse,
                "oracle_maximum_final_relative_rmse": max(accumulator["relative_rmse"]),
                "oracle_mean_final_cosine": mean_cosine,
                "oracle_minimum_final_cosine": min(accumulator["final_cosine"]),
                "selection_gate_passed": passed,
            }
        )
        if passed and selected_rank is None:
            selected_rank = rank

    result = {
        "format": FORMAT,
        "status": "PASS_RANK_SELECTED_NO_TRAINING_AUTHORIZED"
        if selected_rank is not None
        else "FAIL_NO_TESTED_RANK_QUALIFIES",
        "protocol_sha256": protocol_sha,
        "source": {
            "model": protocol["source"]["model"],
            "revision": protocol["source"]["revision"],
            "teacher_load_seconds": teacher_load_seconds,
        },
        "calibration": {
            "train_records": len(train_rows),
            "validation_records": len(validation_rows),
            "tokens": calibration_tokens,
            "train_mlp_residual_observations": train_observations,
            "validation_mlp_residual_observations": validation_observations,
        },
        "rank_results": rank_results,
        "selected_rank": selected_rank,
        "training_performed": False,
        "artifact_written": False,
        "stored_teacher_activations": 0,
        "final_test_accessed": False,
        "accounting": {
            "audit_wall_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": max(peak_rss, process.memory_info().rss),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "source_model_inference_hours": (time.perf_counter() - started) / 3600.0,
        },
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
        "next_gate": "Design one no-model asymmetric sparse-MLP feasibility target at the selected rank."
        if selected_rank is not None
        else "Reject low-rank asymmetric MLP expansion and diagnose mapping/optimization rather than width.",
        "claim_boundary": "Read-only source-layer-0 MLP residual rank attribution; no extraction artifact, training, English quality, inference, Phase 3 certificate, or superiority claim.",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_MLP_RESIDUAL_RANK_AUDIT_PROTOCOL_V237.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_dual_path/mlp_residual_rank_audit_v238.json",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error("MLP residual rank-audit output exists")
    result = execute(root, root / args.protocol)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
