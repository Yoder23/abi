"""Bounded nonlinear coefficient correction on the frozen combined-attention path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_combined_attention_mlp_audit as audit
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_residual_attention_fit as residual
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-residual-nonlinear-rank768-fit/1"


def _combined_coefficients(
    linear_coefficients: torch.Tensor, nonlinear_correction: torch.Tensor
) -> torch.Tensor:
    if linear_coefficients.shape != nonlinear_correction.shape:
        raise Phase3Error("residual-nonlinear coefficient shape changed")
    return linear_coefficients + nonlinear_correction


def _interface(
    prefix: Any,
    primary: Any,
    secondary: Any,
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(ids.shape[1], device=ids.device)
    hidden = dual.base._prefix_hidden(prefix, ids, 1)
    primary_attention = sequential._student_attention(primary, hidden, positions)
    secondary_attention = sequential._student_attention(secondary, hidden, positions)
    combined_attention = residual._combine_attention(
        primary_attention, secondary_attention, hidden
    )
    feature = primary.post_attention_norm(combined_attention)
    return hidden, combined_attention, feature


def execute(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_BOUNDED_RESIDUAL_NONLINEAR_RANK768_FIT"
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("residual-nonlinear governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"residual-nonlinear binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")

    output.mkdir(parents=True)
    device = torch.device("cuda")
    set_determinism(int(protocol["training"]["seed"]))
    base, prefix, tokenizer, primary, secondary = audit._load_paths(root, protocol, device)
    examples = sequential.field._examples(root, base, tokenizer)
    calibration = base["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(calibration["train_records_per_capability"]),
        validation_per_capability=int(calibration["validation_records_per_capability"]),
        maximum_tokens=int(calibration["maximum_sequence_tokens"]),
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    train_residuals: list[torch.Tensor] = []
    train_features: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden, combined_attention, feature = _interface(
                prefix, primary, secondary, ids
            )
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            train_residuals.append(
                (teacher_final - combined_attention).squeeze(0).float().cpu()
            )
            train_features.append(feature.squeeze(0).float().cpu())

    width = int(protocol["architecture"]["full_width"])
    rank = int(protocol["architecture"]["residual_rank"])
    hidden_width = int(protocol["architecture"]["nonlinear_hidden"])
    mean, covariance, observations = rank_audit.centered_covariance(
        train_residuals, width, device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    basis_energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    features = torch.cat(train_features).to(device)
    coefficients = (torch.cat(train_residuals).to(device) - mean) @ basis
    linear_weights, ridge = closed.solve_ridge(
        features, coefficients, float(protocol["training"]["relative_ridge"])
    )
    del covariance, train_features, train_residuals, features, coefficients

    gate_up = torch.nn.Linear(width, 2 * hidden_width, bias=False).to(device)
    nonlinear_output = torch.nn.Linear(hidden_width, rank, bias=False).to(device)
    with torch.no_grad():
        nonlinear_output.weight.zero_()
    zero_output_at_start = bool(torch.count_nonzero(nonlinear_output.weight).item() == 0)
    if not zero_output_at_start:
        raise Phase3Error("nonlinear correction is not exactly zero initialized")
    parameters = list(gate_up.parameters()) + list(nonlinear_output.parameters())
    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["steps"])
    offset = int(training["record_offset"])
    curves: list[dict[str, int | float]] = []
    gate_up.train()
    nonlinear_output.train()
    for step in range(steps):
        row = train_rows[(step + offset) % len(train_rows)]
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, combined_attention, feature = _interface(
                prefix, primary, secondary, ids
            )
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            coefficient_target = (
                teacher_final.float() - combined_attention.float() - mean
            ) @ basis
            linear_coefficients = feature.float() @ linear_weights
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gate, up = gate_up(feature).chunk(2, dim=-1)
            nonlinear_correction = nonlinear_output(F.silu(gate) * up)
            predicted_coefficients = _combined_coefficients(
                linear_coefficients, nonlinear_correction.float()
            )
            final = (
                combined_attention.float()
                + mean
                + predicted_coefficients @ basis.transpose(0, 1)
            )
            coefficient_rmse = torch.sqrt(
                (predicted_coefficients - coefficient_target).square().mean()
                / coefficient_target.square().mean().clamp_min(1e-8)
            )
            final_rmse, final_cosine = dual.base._metrics(
                final, teacher_final.float(), hidden.float()
            )
            loss = (
                coefficient_rmse.square()
                + final_rmse.square()
                + float(training["cosine_weight"]) * (1.0 - final_cosine)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip_norm"]))
        optimizer.step()
        if step == 0 or (step + 1) % int(training["curve_interval"]) == 0:
            curves.append(
                {
                    "step": step + 1,
                    "coefficient_relative_rmse": float(coefficient_rmse.detach()),
                    "final_relative_rmse": float(final_rmse.detach()),
                    "final_cosine": float(final_cosine.detach()),
                    "loss": float(loss.detach()),
                }
            )

    gate_up.eval()
    nonlinear_output.eval()
    baseline_rmses: list[float] = []
    baseline_cosines: list[float] = []
    nonlinear_rmses: list[float] = []
    nonlinear_cosines: list[float] = []
    coefficient_rmses: list[float] = []
    oracle_rmses: list[float] = []
    oracle_cosines: list[float] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden, combined_attention, feature = _interface(
                prefix, primary, secondary, ids
            )
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            target_residual = teacher_final.float() - combined_attention.float()
            coefficient_target = (target_residual - mean) @ basis
            linear_coefficients = feature.float() @ linear_weights
            baseline_final = (
                combined_attention.float()
                + mean
                + linear_coefficients @ basis.transpose(0, 1)
            )
            gate, up = gate_up(feature).chunk(2, dim=-1)
            nonlinear_correction = nonlinear_output(F.silu(gate) * up)
            predicted_coefficients = _combined_coefficients(
                linear_coefficients, nonlinear_correction.float()
            )
            nonlinear_final = (
                combined_attention.float()
                + mean
                + predicted_coefficients @ basis.transpose(0, 1)
            )
            oracle_residual = rank_audit.project_with_basis(
                target_residual.squeeze(0), mean, basis
            ).unsqueeze(0)
            oracle_final = combined_attention.float() + oracle_residual
            baseline_rmse, baseline_cosine = dual.base._metrics(
                baseline_final, teacher_final.float(), hidden.float()
            )
            nonlinear_rmse, nonlinear_cosine = dual.base._metrics(
                nonlinear_final, teacher_final.float(), hidden.float()
            )
            oracle_rmse, oracle_cosine = dual.base._metrics(
                oracle_final, teacher_final.float(), hidden.float()
            )
            coefficient_rmse = torch.sqrt(
                (predicted_coefficients - coefficient_target).square().mean()
                / coefficient_target.square().mean().clamp_min(1e-8)
            )
            baseline_rmses.append(float(baseline_rmse))
            baseline_cosines.append(float(baseline_cosine))
            nonlinear_rmses.append(float(nonlinear_rmse))
            nonlinear_cosines.append(float(nonlinear_cosine))
            oracle_rmses.append(float(oracle_rmse))
            oracle_cosines.append(float(oracle_cosine))
            coefficient_rmses.append(float(coefficient_rmse))

    mean_rmse = sum(nonlinear_rmses) / len(nonlinear_rmses)
    mean_cosine = sum(nonlinear_cosines) / len(nonlinear_cosines)
    gate = protocol["gate"]
    passed = (
        mean_rmse <= float(gate["mean_relative_rmse_maximum"])
        and mean_cosine >= float(gate["mean_output_cosine_minimum"])
    )
    checkpoint = {
        "mlp_residual_mean": mean.detach().to(torch.float16).cpu().contiguous(),
        "mlp_basis": basis.detach().to(torch.float16).cpu().contiguous(),
        "mlp_linear_coefficient.weight": linear_weights.transpose(0, 1).detach().to(torch.float16).cpu().contiguous(),
        "mlp_nonlinear_gate_up.weight": gate_up.weight.detach().to(torch.float16).cpu().contiguous(),
        "mlp_nonlinear_output.weight": nonlinear_output.weight.detach().to(torch.float16).cpu().contiguous(),
    }
    checkpoint_path = output / "layer1_residual_nonlinear_rank768.safetensors"
    save_file(
        checkpoint,
        str(checkpoint_path),
        metadata={"format": FORMAT, "protocol_sha256": sha256_file(protocol_path)},
    )
    result = {
        "format": FORMAT,
        "status": "PASS_RESIDUAL_NONLINEAR_RANK768_LOCAL" if passed else "FAIL_RESIDUAL_NONLINEAR_RANK768_LOCAL",
        "protocol_sha256": sha256_file(protocol_path),
        "layer": 1,
        "steps": steps,
        "record_offset": offset,
        "calibration_tokens": calibration_tokens,
        "train_observations": observations,
        "basis_rank": rank,
        "basis_energy_explained": basis_energy,
        "effective_ridge": ridge,
        "zero_output_at_start": zero_output_at_start,
        "curves": curves,
        "linear_baseline_validation": {
            "mean_relative_rmse": sum(baseline_rmses) / len(baseline_rmses),
            "mean_output_cosine": sum(baseline_cosines) / len(baseline_cosines),
        },
        "oracle_validation": {
            "mean_relative_rmse": sum(oracle_rmses) / len(oracle_rmses),
            "mean_output_cosine": sum(oracle_cosines) / len(oracle_cosines),
        },
        "nonlinear_validation": {
            "mean_coefficient_relative_rmse": sum(coefficient_rmses) / len(coefficient_rmses),
            "mean_relative_rmse": mean_rmse,
            "maximum_relative_rmse": max(nonlinear_rmses),
            "mean_output_cosine": mean_cosine,
            "minimum_output_cosine": min(nonlinear_cosines),
            "passed": passed,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "parameters": sum(value.numel() for value in checkpoint.values()),
        },
        "source_mlp_present_in_checkpoint": False,
        "artifact_promoted": False,
        "teacher_required_at_inference": False,
        "training_performed": True,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Bounded layer-1 residual-nonlinear fit only; no host, full artifact, English quality, physical runtime, certificate, or superiority claim.",
    }
    _write_immutable(output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_RESIDUAL_NONLINEAR_RANK768_FIT_PROTOCOL_V291.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_dual_attention_nonlinear/layer1_fit_v292",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
