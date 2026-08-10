"""Read-only analytic coefficient audit using 384 extracted source-MLP neurons."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_combined_attention_mlp_audit as audit
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_residual_nonlinear_rank768_fit as nonlinear
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-sparse-neuron-coefficient-audit/1"


def deployment_accounting(
    selected_neurons: int,
    *,
    full: int = 3072,
    rank: int = 768,
    layers: int = 32,
) -> dict[str, int]:
    imported_per_layer = 2 * full * selected_neurons + selected_neurons * rank
    return {
        "selected_neurons": selected_neurons,
        "imported_sparse_feature_parameters_per_layer": imported_per_layer,
        "imported_sparse_feature_parameters_all_layers": layers * imported_per_layer,
        "source_blocks": 0,
    }


def _apply_correction(
    linear_coefficients: torch.Tensor,
    sparse_features: torch.Tensor,
    correction_weights: torch.Tensor,
) -> torch.Tensor:
    if sparse_features.shape[-1] != correction_weights.shape[0]:
        raise Phase3Error("sparse-neuron correction feature shape changed")
    correction = sparse_features.float() @ correction_weights
    if correction.shape != linear_coefficients.shape:
        raise Phase3Error("sparse-neuron correction coefficient shape changed")
    return linear_coefficients + correction


def execute(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_SPARSE_NEURON_COEFFICIENT_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_authorized") is not False
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("sparse-neuron coefficient audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"sparse-neuron coefficient binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")

    output.mkdir(parents=True)
    device = torch.device("cuda")
    set_determinism(int(protocol["seed"]))
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
    teacher_layer = teacher.model.layers[1]
    gate_up_weight = teacher_layer.mlp.gate_up_proj.weight.float()
    down_weight = teacher_layer.mlp.down_proj.weight.float()
    source_neurons = down_weight.shape[1]
    width = int(protocol["width"])
    rank = int(protocol["rank"])
    if gate_up_weight.shape != (2 * source_neurons, width) or down_weight.shape[0] != width:
        raise Phase3Error("source MLP topology changed")

    train_features: list[torch.Tensor] = []
    train_residuals: list[torch.Tensor] = []
    importance = torch.zeros(source_neurons, device=device)
    importance_observations = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden, combined_attention, feature = nonlinear._interface(
                prefix, primary, secondary, ids
            )
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            flat_feature = feature.float()
            gate, up = F.linear(flat_feature, gate_up_weight).chunk(2, dim=-1)
            activation = F.silu(gate) * up
            importance += activation.square().sum(dim=(0, 1))
            importance_observations += activation.shape[0] * activation.shape[1]
            train_features.append(feature.squeeze(0).float().cpu())
            train_residuals.append(
                (teacher_final - combined_attention).squeeze(0).float().cpu()
            )
    importance *= down_weight.square().sum(dim=0)
    selected_count = int(protocol["selected_neurons"])
    selected = torch.argsort(importance, descending=True, stable=True)[:selected_count]

    mean, covariance, residual_observations = rank_audit.centered_covariance(
        train_residuals, width, device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    basis_energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    features = torch.cat(train_features).to(device)
    coefficients = (torch.cat(train_residuals).to(device) - mean) @ basis
    linear_weights, linear_ridge = closed.solve_ridge(
        features, coefficients, float(protocol["relative_ridge"])
    )
    linear_coefficients = features @ linear_weights
    correction_targets = coefficients - linear_coefficients
    del covariance, features, coefficients, linear_coefficients

    selected_gate = gate_up_weight[:source_neurons].index_select(0, selected)
    selected_up = gate_up_weight[source_neurons:].index_select(0, selected)
    sparse_batches: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for feature in train_features:
            values = feature.to(device)
            sparse_batches.append(
                (F.silu(F.linear(values, selected_gate)) * F.linear(values, selected_up)).float()
            )
    sparse_features = torch.cat(sparse_batches)
    correction_weights, correction_ridge = closed.solve_ridge(
        sparse_features, correction_targets, float(protocol["relative_ridge"])
    )
    predicted_correction = sparse_features @ correction_weights
    correction_training_rmse = float(
        torch.sqrt(
            (predicted_correction - correction_targets).square().mean()
            / correction_targets.square().mean().clamp_min(1e-8)
        )
    )
    del sparse_batches, sparse_features, correction_targets, predicted_correction

    baseline_rmses: list[float] = []
    baseline_cosines: list[float] = []
    corrected_rmses: list[float] = []
    corrected_cosines: list[float] = []
    oracle_rmses: list[float] = []
    oracle_cosines: list[float] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden, combined_attention, feature = nonlinear._interface(
                prefix, primary, secondary, ids
            )
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            target_residual = teacher_final.float() - combined_attention.float()
            linear_coefficients = feature.float() @ linear_weights
            baseline_final = (
                combined_attention.float()
                + mean
                + linear_coefficients @ basis.transpose(0, 1)
            )
            sparse_features = F.silu(F.linear(feature.float(), selected_gate)) * F.linear(
                feature.float(), selected_up
            )
            corrected_coefficients = _apply_correction(
                linear_coefficients, sparse_features, correction_weights
            )
            corrected_final = (
                combined_attention.float()
                + mean
                + corrected_coefficients @ basis.transpose(0, 1)
            )
            oracle_residual = rank_audit.project_with_basis(
                target_residual.squeeze(0), mean, basis
            ).unsqueeze(0)
            oracle_final = combined_attention.float() + oracle_residual
            baseline_rmse, baseline_cosine = dual.base._metrics(
                baseline_final, teacher_final.float(), hidden.float()
            )
            corrected_rmse, corrected_cosine = dual.base._metrics(
                corrected_final, teacher_final.float(), hidden.float()
            )
            oracle_rmse, oracle_cosine = dual.base._metrics(
                oracle_final, teacher_final.float(), hidden.float()
            )
            baseline_rmses.append(float(baseline_rmse))
            baseline_cosines.append(float(baseline_cosine))
            corrected_rmses.append(float(corrected_rmse))
            corrected_cosines.append(float(corrected_cosine))
            oracle_rmses.append(float(oracle_rmse))
            oracle_cosines.append(float(oracle_cosine))

    mean_rmse = sum(corrected_rmses) / len(corrected_rmses)
    mean_cosine = sum(corrected_cosines) / len(corrected_cosines)
    gate = protocol["gate"]
    passed = (
        mean_rmse <= float(gate["mean_relative_rmse_maximum"])
        and mean_cosine >= float(gate["mean_output_cosine_minimum"])
    )
    result = {
        "format": FORMAT,
        "status": "PASS_SPARSE_NEURON_COEFFICIENT_LOCAL" if passed else "FAIL_SPARSE_NEURON_COEFFICIENT_LOCAL",
        "protocol_sha256": sha256_file(protocol_path),
        "training_authorized": False,
        "training_performed": False,
        "layer": 1,
        "calibration_tokens": calibration_tokens,
        "source_neurons": source_neurons,
        "selected_neurons": selected_count,
        "importance_observations": importance_observations,
        "residual_observations": residual_observations,
        "basis_rank": rank,
        "basis_energy_explained": basis_energy,
        "linear_effective_ridge": linear_ridge,
        "correction_effective_ridge": correction_ridge,
        "correction_training_relative_rmse": correction_training_rmse,
        "linear_baseline_validation": {
            "mean_relative_rmse": sum(baseline_rmses) / len(baseline_rmses),
            "mean_output_cosine": sum(baseline_cosines) / len(baseline_cosines),
        },
        "oracle_validation": {
            "mean_relative_rmse": sum(oracle_rmses) / len(oracle_rmses),
            "mean_output_cosine": sum(oracle_cosines) / len(oracle_cosines),
        },
        "corrected_validation": {
            "mean_relative_rmse": mean_rmse,
            "maximum_relative_rmse": max(corrected_rmses),
            "mean_output_cosine": mean_cosine,
            "minimum_output_cosine": min(corrected_cosines),
            "passed": passed,
        },
        "deployment_accounting": deployment_accounting(selected_count),
        "copied_source_parameters_in_artifact": 0,
        "source_blocks_in_artifact": 0,
        "artifact_written": False,
        "artifact_promoted": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only layer-1 sparse-neuron coefficient audit only; no artifact, English quality, physical runtime, certificate, or superiority claim.",
    }
    _write_immutable(output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_SPARSE_NEURON_COEFFICIENT_AUDIT_PROTOCOL_V293.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_sparse_neuron_coefficient/audit_v294",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
