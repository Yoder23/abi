"""Read-only rank-192 MLP-map audit on the frozen combined-attention interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_residual_attention_fit as residual
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-combined-attention-mlp-audit/1"


def _reconstruct(
    feature: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if feature.shape[-1] != weights.shape[0] or weights.shape[1] != basis.shape[1]:
        raise Phase3Error("combined-attention MLP map shape changed")
    if mean.shape[0] != basis.shape[0]:
        raise Phase3Error("combined-attention MLP basis shape changed")
    return mean + (feature.float() @ weights) @ basis.transpose(0, 1)


def _load_paths(root: Path, protocol: dict[str, Any], device: torch.device):
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    prefix, tokenizer, _, attention_keys, _ = sequential._model(root, base, device)
    state = prefix.state_dict()
    for layer_index in (0, 1):
        checkpoint = load_file(
            str(root / protocol["base_checkpoints"][str(layer_index)]["path"]),
            device="cpu",
        )
        for name, value in checkpoint.items():
            if name in state:
                state[name].copy_(value.to(state[name].dtype))
    qualified = load_file(str(root / protocol["qualified_primary_checkpoint"]["path"]), device="cpu")
    expected_primary = {name for name in attention_keys if name.startswith("layers.1.")}
    if set(qualified) != expected_primary:
        raise Phase3Error("qualified primary attention tensor boundary changed")
    with torch.no_grad():
        for name, value in qualified.items():
            state[name].copy_(value.to(state[name].dtype))
    for parameter in prefix.parameters():
        parameter.requires_grad_(False)
    prefix.eval()
    primary = prefix.layers[1]

    residual_protocol = json.loads((root / protocol["residual_protocol"]).read_text(encoding="utf-8"))
    secondary = residual._secondary_layer(device, primary, residual_protocol)
    secondary_checkpoint = load_file(
        str(root / protocol["residual_attention_checkpoint"]["path"]), device="cpu"
    )
    if set(secondary_checkpoint) != set(dict(secondary.named_parameters())):
        raise Phase3Error("residual attention tensor boundary changed")
    secondary.load_state_dict(secondary_checkpoint, strict=True)
    for parameter in secondary.parameters():
        parameter.requires_grad_(False)
    secondary.eval()
    return base, prefix, tokenizer, primary, secondary


def execute(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_COMBINED_ATTENTION_MLP_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("combined-attention MLP audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"combined-attention MLP audit binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")

    output.mkdir(parents=True)
    device = torch.device("cuda")
    set_determinism(int(protocol["seed"]))
    base, prefix, tokenizer, primary, secondary = _load_paths(root, protocol, device)
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
            positions = torch.arange(ids.shape[1], device=device)
            hidden = dual.base._prefix_hidden(prefix, ids, 1)
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            primary_attention = sequential._student_attention(primary, hidden, positions)
            secondary_attention = sequential._student_attention(secondary, hidden, positions)
            combined_attention = residual._combine_attention(
                primary_attention, secondary_attention, hidden
            )
            train_residuals.append((teacher_final - combined_attention).squeeze(0).float().cpu())
            train_features.append(
                primary.post_attention_norm(combined_attention).squeeze(0).float().cpu()
            )

    width = int(protocol["width"])
    rank = int(protocol["rank"])
    mean, covariance, observations = rank_audit.centered_covariance(
        train_residuals, width, device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    features = torch.cat(train_features).to(device)
    coefficients = (torch.cat(train_residuals).to(device) - mean) @ basis
    weights, ridge = closed.solve_ridge(
        features, coefficients, float(protocol["relative_ridge"])
    )
    predicted_coefficients = features @ weights
    training_coefficient_rmse = float(
        torch.sqrt(
            (predicted_coefficients - coefficients).square().mean()
            / coefficients.square().mean().clamp_min(1e-8)
        )
    )
    del features, coefficients, predicted_coefficients, train_features, train_residuals, covariance

    mapped_rmses: list[float] = []
    mapped_cosines: list[float] = []
    oracle_rmses: list[float] = []
    oracle_cosines: list[float] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            positions = torch.arange(ids.shape[1], device=device)
            hidden = dual.base._prefix_hidden(prefix, ids, 1)
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            primary_attention = sequential._student_attention(primary, hidden, positions)
            secondary_attention = sequential._student_attention(secondary, hidden, positions)
            combined_attention = residual._combine_attention(
                primary_attention, secondary_attention, hidden
            )
            feature = primary.post_attention_norm(combined_attention).squeeze(0)
            mapped_residual = _reconstruct(feature, mean, basis, weights).unsqueeze(0)
            mapped_final = combined_attention.float() + mapped_residual
            actual_residual = (teacher_final - combined_attention).squeeze(0).float()
            oracle_residual = rank_audit.project_with_basis(
                actual_residual, mean, basis
            ).unsqueeze(0)
            oracle_final = combined_attention.float() + oracle_residual
            mapped_rmse, mapped_cosine = dual.base._metrics(
                mapped_final, teacher_final.float(), hidden.float()
            )
            oracle_rmse, oracle_cosine = dual.base._metrics(
                oracle_final, teacher_final.float(), hidden.float()
            )
            mapped_rmses.append(float(mapped_rmse))
            mapped_cosines.append(float(mapped_cosine))
            oracle_rmses.append(float(oracle_rmse))
            oracle_cosines.append(float(oracle_cosine))

    mean_mapped_rmse = sum(mapped_rmses) / len(mapped_rmses)
    mean_mapped_cosine = sum(mapped_cosines) / len(mapped_cosines)
    gate = protocol["gate"]
    oracle_passed = (
        sum(oracle_rmses) / len(oracle_rmses) <= float(gate["mean_relative_rmse_maximum"])
        and sum(oracle_cosines) / len(oracle_cosines) >= float(gate["mean_output_cosine_minimum"])
    )
    mapped_passed = (
        mean_mapped_rmse <= float(gate["mean_relative_rmse_maximum"])
        and mean_mapped_cosine >= float(gate["mean_output_cosine_minimum"])
    )
    result = {
        "format": FORMAT,
        "status": "PASS_COMBINED_ATTENTION_MLP_MAP_LOCAL" if mapped_passed else "FAIL_COMBINED_ATTENTION_MLP_MAP_LOCAL",
        "protocol_sha256": sha256_file(protocol_path),
        "training_performed": False,
        "training_authorized": False,
        "layer": 1,
        "calibration_tokens": calibration_tokens,
        "train_observations": observations,
        "rank": rank,
        "basis_energy_explained": energy,
        "effective_ridge": ridge,
        "training_coefficient_relative_rmse": training_coefficient_rmse,
        "oracle_validation": {
            "mean_relative_rmse": sum(oracle_rmses) / len(oracle_rmses),
            "mean_output_cosine": sum(oracle_cosines) / len(oracle_cosines),
            "passed": oracle_passed,
        },
        "mapped_validation": {
            "mean_relative_rmse": mean_mapped_rmse,
            "maximum_relative_rmse": max(mapped_rmses),
            "mean_output_cosine": mean_mapped_cosine,
            "minimum_output_cosine": min(mapped_cosines),
            "passed": mapped_passed,
        },
        "source_mlp_present_in_artifact": False,
        "artifact_written": False,
        "artifact_promoted": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only layer-1 combined-attention MLP-map audit only; no artifact, English quality, runtime, certificate, or superiority claim.",
    }
    _write_immutable(output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_COMBINED_ATTENTION_MLP_AUDIT_PROTOCOL_V285.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_combined_attention_mlp/audit_v286",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
