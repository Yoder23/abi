"""Read-only closed-form coefficient-map audit for the rank-192 MLP basis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_causal_field_core as field
from . import capability_compiler_phase3_basis_aligned_local_fit as aligned
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-closed-form-coefficient-audit/1"


def solve_ridge(features: torch.Tensor, targets: torch.Tensor, relative_ridge: float) -> tuple[torch.Tensor, float]:
    gram = features.transpose(0, 1) @ features
    scale = float(torch.trace(gram) / gram.shape[0])
    ridge = relative_ridge * scale
    cross = features.transpose(0, 1) @ targets
    solution = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype),
        cross,
    )
    return solution, ridge


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_CLOSED_FORM_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("closed-form coefficient audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"closed-form coefficient binding changed: {name}")
    return protocol, sha256_file(path)


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol, protocol_sha = load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("closed-form coefficient audit requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(int(protocol["seed"]))
    _, tokenizer_type = aligned._types(root, protocol)
    tokenizer = field._tokenizer(protocol, tokenizer_type)
    examples = field._examples(root, protocol, tokenizer)
    cfg = protocol["calibration"]
    train_rows, validation_rows, tokens = dual._calibration_examples(
        examples, seed=int(protocol["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    substrate = load_file(str(root / protocol["substrate"]["path"]), device="cpu")
    embedding = substrate["token_embedding.weight"].to(device)
    teacher = AutoModelForCausalLM.from_pretrained(
        protocol["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    layer = teacher.model.layers[0]
    started = time.perf_counter()
    train_deltas, train_features = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
            hidden = F.embedding(ids, embedding).unsqueeze(0)
            attention, final = dual._teacher_components(teacher, 0, hidden)
            train_deltas.append((final - attention).squeeze(0).float().cpu())
            train_features.append(layer.post_attention_layernorm(attention).squeeze(0).float().cpu())
    mean, covariance, observations = rank_audit.centered_covariance(train_deltas, 3072, device)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors.flip(1)[:, : int(protocol["rank"])].contiguous()
    features = torch.cat(train_features).to(device)
    delta = torch.cat(train_deltas).to(device)
    targets = (delta - mean) @ basis
    del train_features, train_deltas, delta
    weights, ridge = solve_ridge(features, targets, float(protocol["relative_ridge"]))
    training_prediction = features @ weights
    training_coefficient_rmse = float(torch.sqrt((training_prediction - targets).square().mean() / targets.square().mean().clamp_min(1e-8)))
    del features, targets, training_prediction

    rmses, cosines, coefficient_rmses = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
            hidden = F.embedding(ids, embedding).unsqueeze(0)
            attention, final = dual._teacher_components(teacher, 0, hidden)
            features = layer.post_attention_layernorm(attention).float()
            coefficient_target = (final.float() - attention.float() - mean) @ basis
            coefficient_prediction = features @ weights
            reconstructed = attention.float() + mean + (coefficient_prediction @ basis.transpose(0, 1))
            rmse, cosine = dual.base._metrics(reconstructed, final.float(), hidden.float())
            coefficient_rmse = torch.sqrt((coefficient_prediction - coefficient_target).square().mean() / coefficient_target.square().mean().clamp_min(1e-8))
            rmses.append(float(rmse)); cosines.append(float(cosine)); coefficient_rmses.append(float(coefficient_rmse))
    mean_rmse = sum(rmses) / len(rmses); mean_cosine = sum(cosines) / len(cosines)
    passed = mean_rmse <= float(protocol["gate"]["mean_relative_rmse_maximum"]) and mean_cosine >= float(protocol["gate"]["mean_output_cosine_minimum"])
    return {
        "format": FORMAT,
        "status": "PASS_CLOSED_FORM_MAP_FEASIBLE_NO_HOST_AUTHORIZED" if passed else "FAIL_CLOSED_FORM_MAP",
        "protocol_sha256": protocol_sha,
        "rank": int(protocol["rank"]),
        "relative_ridge": float(protocol["relative_ridge"]),
        "effective_ridge": ridge,
        "train_observations": observations,
        "training_coefficient_relative_rmse": training_coefficient_rmse,
        "validation": {
            "records": len(validation_rows),
            "mean_coefficient_relative_rmse": sum(coefficient_rmses) / len(coefficient_rmses),
            "mean_final_relative_rmse": mean_rmse,
            "maximum_final_relative_rmse": max(rmses),
            "mean_final_cosine": mean_cosine,
            "minimum_final_cosine": min(cosines),
            "passed": passed,
        },
        "calibration_tokens": tokens,
        "audit_wall_seconds": time.perf_counter() - started,
        "matrix_parameters": weights.numel(),
        "artifact_written": False,
        "training_performed": False,
        "stored_activations": 0,
        "final_test_accessed": False,
        "next_gate": "Design one generic direct linear coefficient-map host and extraction protocol." if passed else "Reject direct linear coefficient mapping.",
        "claim_boundary": "Read-only layer-0 closed-form mapping audit; no deployable artifact, neural training, English quality, inference, Phase 3 certificate, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CLOSED_FORM_COEFFICIENT_AUDIT_PROTOCOL_V245.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_basis_aligned/closed_form_coefficient_audit_v246.json")
    args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists(): raise Phase3Error("closed-form audit output exists")
    result = execute(root, root / args.protocol); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if str(result["status"]).startswith("PASS") else 1


if __name__ == "__main__": raise SystemExit(main())
