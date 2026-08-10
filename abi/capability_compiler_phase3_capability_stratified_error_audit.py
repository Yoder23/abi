"""Read-only capability stratification of the fixed V294 sparse correction."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_combined_attention_mlp_audit as audit
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_residual_nonlinear_rank768_fit as nonlinear
from . import capability_compiler_phase3_sparse_neuron_coefficient_audit as sparse
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-capability-stratified-error-audit/1"


def _summary(values: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    return {
        capability: {
            "records": len(rows),
            "mean_output_cosine": sum(rows) / len(rows),
            "minimum_output_cosine": min(rows),
        }
        for capability, rows in sorted(values.items())
    }


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_CAPABILITY_STRATIFIED_ERROR_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("capability-stratified audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"capability-stratified binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")

    output.mkdir(parents=True)
    device = torch.device("cuda")
    set_determinism(int(protocol["seed"]))
    base, prefix, tokenizer, primary, secondary = audit._load_paths(root, protocol, device)
    examples = sequential.field._examples(root, base, tokenizer)
    cfg = base["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
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

    importance = torch.zeros(source_neurons, device=device)
    features_cpu: list[torch.Tensor] = []
    residuals_cpu: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden, combined_attention, feature = nonlinear._interface(prefix, primary, secondary, ids)
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            gate, up = F.linear(feature.float(), gate_up_weight).chunk(2, dim=-1)
            activation = F.silu(gate) * up
            importance += activation.square().sum(dim=(0, 1))
            features_cpu.append(feature.squeeze(0).float().cpu())
            residuals_cpu.append((teacher_final - combined_attention).squeeze(0).float().cpu())
    importance *= down_weight.square().sum(dim=0)
    selected = torch.argsort(importance, descending=True, stable=True)[
        : int(protocol["selected_neurons"])
    ]

    width = int(protocol["width"])
    rank = int(protocol["rank"])
    mean, covariance, observations = rank_audit.centered_covariance(
        residuals_cpu, width, device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    features = torch.cat(features_cpu).to(device)
    coefficients = (torch.cat(residuals_cpu).to(device) - mean) @ basis
    linear_weights, _ = closed.solve_ridge(features, coefficients, float(protocol["relative_ridge"]))
    correction_targets = coefficients - features @ linear_weights
    selected_gate = gate_up_weight[:source_neurons].index_select(0, selected)
    selected_up = gate_up_weight[source_neurons:].index_select(0, selected)
    sparse_features = torch.cat(
        [
            F.silu(F.linear(feature.to(device), selected_gate))
            * F.linear(feature.to(device), selected_up)
            for feature in features_cpu
        ]
    ).float()
    correction_weights, _ = closed.solve_ridge(
        sparse_features, correction_targets, float(protocol["relative_ridge"])
    )
    del covariance, features, coefficients, correction_targets, sparse_features

    baseline_by_capability: dict[str, list[float]] = defaultdict(list)
    corrected_by_capability: dict[str, list[float]] = defaultdict(list)
    corrected_rmse_by_capability: dict[str, list[float]] = defaultdict(list)
    record_rows: list[dict] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden, combined_attention, feature = nonlinear._interface(prefix, primary, secondary, ids)
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            linear_coefficients = feature.float() @ linear_weights
            baseline_final = combined_attention.float() + mean + linear_coefficients @ basis.T
            sparse_features = F.silu(F.linear(feature.float(), selected_gate)) * F.linear(
                feature.float(), selected_up
            )
            corrected_coefficients = sparse._apply_correction(
                linear_coefficients, sparse_features, correction_weights
            )
            corrected_final = combined_attention.float() + mean + corrected_coefficients @ basis.T
            baseline_rmse, baseline_cosine = dual.base._metrics(
                baseline_final, teacher_final.float(), hidden.float()
            )
            corrected_rmse, corrected_cosine = dual.base._metrics(
                corrected_final, teacher_final.float(), hidden.float()
            )
            capability = str(row["capability"])
            baseline_by_capability[capability].append(float(baseline_cosine))
            corrected_by_capability[capability].append(float(corrected_cosine))
            corrected_rmse_by_capability[capability].append(float(corrected_rmse))
            record_rows.append(
                {
                    "record_id": row["record_id"],
                    "capability": capability,
                    "baseline_output_cosine": float(baseline_cosine),
                    "corrected_output_cosine": float(corrected_cosine),
                    "corrected_relative_rmse": float(corrected_rmse),
                }
            )

    capability_rows = []
    for capability in sorted(corrected_by_capability):
        baseline_values = baseline_by_capability[capability]
        corrected_values = corrected_by_capability[capability]
        rmse_values = corrected_rmse_by_capability[capability]
        capability_rows.append(
            {
                "capability": capability,
                "records": len(corrected_values),
                "baseline_mean_output_cosine": sum(baseline_values) / len(baseline_values),
                "corrected_mean_output_cosine": sum(corrected_values) / len(corrected_values),
                "corrected_minimum_output_cosine": min(corrected_values),
                "corrected_mean_relative_rmse": sum(rmse_values) / len(rmse_values),
                "records_below_cosine_gate": sum(
                    value < float(protocol["gate"]["mean_output_cosine_minimum"])
                    for value in corrected_values
                ),
            }
        )
    failing_capabilities = [
        row["capability"]
        for row in capability_rows
        if row["corrected_mean_output_cosine"]
        < float(protocol["gate"]["mean_output_cosine_minimum"])
    ]
    result = {
        "format": FORMAT,
        "status": "PASS_CAPABILITY_ERROR_STRATIFIED",
        "protocol_sha256": sha256_file(protocol_path),
        "training_performed": False,
        "artifact_written": False,
        "calibration_tokens": calibration_tokens,
        "train_observations": observations,
        "validation_records": len(validation_rows),
        "capabilities": capability_rows,
        "failing_capabilities": failing_capabilities,
        "failing_capability_count": len(failing_capabilities),
        "record_metrics": record_rows,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only capability-stratified replay of the fixed V294 method; no architecture, artifact, quality, runtime, certificate, or superiority claim.",
    }
    _write_immutable(output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_CAPABILITY_STRATIFIED_ERROR_AUDIT_PROTOCOL_V295.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_sparse_neuron_coefficient/stratified_v296",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
