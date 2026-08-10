"""Read-only decomposition of the routed v15 layer-2 cosine failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_routed_v15_layer0_extract as layer0
from . import capability_compiler_phase3_routed_v15_progressive_extract as progressive
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-routed-v15-layer2-geometry-audit/1"


def _summary(prediction, target, source, attention):
    relative_rmse, output_cosine = dual.base._metrics(prediction, target, source)
    error = prediction.float() - target.float()
    residual_prediction = prediction.float() - attention.float()
    residual_target = target.float() - attention.float()
    target_rms = torch.sqrt(target.float().square().mean()).clamp_min(1e-8)
    residual_rms = torch.sqrt(residual_target.square().mean()).clamp_min(1e-8)
    return {
        "block_relative_rmse": float(relative_rmse),
        "output_cosine": float(output_cosine),
        "absolute_rmse": float(torch.sqrt(error.square().mean())),
        "output_relative_rmse": float(torch.sqrt(error.square().mean()) / target_rms),
        "target_output_rms": float(target_rms),
        "input_rms": float(torch.sqrt(source.float().square().mean())),
        "target_residual_rms": float(residual_rms),
        "residual_cosine": float(
            F.cosine_similarity(
                residual_prediction.reshape(-1, residual_prediction.shape[-1]),
                residual_target.reshape(-1, residual_target.shape[-1]),
                dim=-1,
            ).mean()
        ),
    }


def _mean(records, scenario, metric):
    return sum(row["scenarios"][scenario][metric] for row in records) / len(records)


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_LAYER2_GEOMETRY_AUDIT"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("layer2 geometry audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"layer2 geometry binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")
    output.mkdir(parents=True)
    extraction, extraction_sha = progressive._load_protocol(
        root, root / protocol["extraction_protocol"]
    )
    if extraction_sha != protocol["extraction_protocol_sha256"]:
        raise Phase3Error("extraction protocol identity changed")
    device = torch.device("cuda")
    set_determinism(int(extraction["training"]["seed"]))
    model, tokenizer, base, _, _ = progressive._instantiate(root, extraction, device)
    state = model.state_dict()
    checkpoint = load_file(str(root / protocol["failed_checkpoint"]["path"]), device="cpu")
    expected = {
        name for name, _ in model.named_parameters() if name.startswith("layers.2.")
    }
    if set(checkpoint) != expected:
        raise Phase3Error("failed layer2 checkpoint boundary changed")
    with torch.no_grad():
        for name, value in checkpoint.items():
            state[name].copy_(value.to(state[name].dtype))
    model.eval()
    examples = progressive.sequential.field._examples(root, base, tokenizer)
    example_by_id = {str(row["record_id"]): row for row in examples}
    cfg = base["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    cache = progressive._initial_cache(model, validation_rows, example_by_id, device)
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(
        base["source"]["parameter_count"]
    ):
        raise Phase3Error("loaded source parameter count changed")
    layer = model.layers[2]
    source_layer = teacher.model.layers[2]
    basis = layer.mlp_output_projection.weight.float()
    gram_error = float(
        torch.linalg.matrix_norm(basis.T @ basis - torch.eye(basis.shape[1], device=device))
        / basis.shape[1]
    )
    records = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            record_id = str(row["record_id"])
            hidden, route_index = cache[record_id]
            hidden = hidden.unsqueeze(0).to(device)
            positions = torch.arange(hidden.shape[1], device=device)
            teacher_attention, teacher_final = dual._teacher_components(teacher, 2, hidden)
            attention = layer0._attention(layer, hidden, positions)
            feature = layer.post_attention_norm(attention)
            exact_source_mlp = attention + source_layer.mlp(feature)
            residual = teacher_final.float() - attention.float() - layer.mlp_residual_mean.float()
            oracle_coefficients = residual @ basis
            basis_oracle = attention + layer.mlp_residual_mean + oracle_coefficients @ basis.T
            linear_coefficients = layer.linear_coefficient_projection(feature)
            linear_only = attention + layer.mlp_residual_mean + layer.mlp_output_projection(
                linear_coefficients
            )
            deployed = attention + layer._mlp_delta(attention, route_index)
            scenarios = {
                "attention_plus_exact_source_mlp": _summary(
                    exact_source_mlp, teacher_final, hidden, attention
                ),
                "rank768_basis_oracle": _summary(
                    basis_oracle, teacher_final, hidden, attention
                ),
                "linear_coefficients_only": _summary(
                    linear_only, teacher_final, hidden, attention
                ),
                "deployed_routed_map": _summary(deployed, teacher_final, hidden, attention),
            }
            records.append(
                {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "route": layer0.ROUTES[route_index],
                    "scenarios": scenarios,
                }
            )
    scenario_names = list(records[0]["scenarios"])
    aggregate = {
        scenario: {
            metric: _mean(records, scenario, metric)
            for metric in records[0]["scenarios"][scenario]
        }
        for scenario in scenario_names
    }
    gate = protocol["gate"]
    diagnosis = {
        "exact_source_mlp_passes_cosine": aggregate["attention_plus_exact_source_mlp"][
            "output_cosine"
        ] >= float(gate["mean_output_cosine_minimum"]),
        "basis_oracle_passes_cosine": aggregate["rank768_basis_oracle"]["output_cosine"]
        >= float(gate["mean_output_cosine_minimum"]),
        "linear_only_passes_cosine": aggregate["linear_coefficients_only"]["output_cosine"]
        >= float(gate["mean_output_cosine_minimum"]),
        "deployed_passes_cosine": aggregate["deployed_routed_map"]["output_cosine"]
        >= float(gate["mean_output_cosine_minimum"]),
    }
    result = {
        "format": FORMAT,
        "status": "PASS_READ_ONLY_DIAGNOSTIC_COMPLETE",
        "protocol_sha256": sha256_file(protocol_path),
        "extraction_protocol_sha256": extraction_sha,
        "calibration_tokens": calibration_tokens,
        "validation_records": len(validation_rows),
        "basis_orthonormal_gram_error": gram_error,
        "aggregate": aggregate,
        "diagnosis": diagnosis,
        "record_metrics": records,
        "training_performed": False,
        "checkpoint_written": False,
        "artifact_promoted": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only fixed-record layer2 failure decomposition only; no repair, artifact, English quality, runtime, certificate, or superiority claim.",
    }
    _write_immutable(
        output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_LAYER2_GEOMETRY_AUDIT_PROTOCOL_V311.json",
    )
    parser.add_argument(
        "--output", default="results/abi_capability_compiler_phase3_routed_v15/layer2_geometry_v312"
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
