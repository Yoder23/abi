"""Read-only perfect-label upper bound for three sparse correction routes."""

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


FORMAT = "abi-capability-compiler-phase3-three-route-label-oracle-audit/1"


def _route(capability: str, specialist_routes: tuple[str, ...]) -> str:
    return capability if capability in specialist_routes else "generic"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_THREE_ROUTE_LABEL_ORACLE_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_authorized") is not False
        or protocol.get("router_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("three-route label-oracle governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"three-route label-oracle binding changed: {name}")
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
    source_layer = teacher.model.layers[1]
    gate_up_weight = source_layer.mlp.gate_up_proj.weight.float()
    down_weight = source_layer.mlp.down_proj.weight.float()
    source_neurons = down_weight.shape[1]
    importance = torch.zeros(source_neurons, device=device)
    features_cpu: list[torch.Tensor] = []
    residuals_cpu: list[torch.Tensor] = []
    token_routes: list[str] = []
    specialists = tuple(str(value) for value in protocol["specialist_routes"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden, combined_attention, feature = nonlinear._interface(prefix, primary, secondary, ids)
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            gate, up = F.linear(feature.float(), gate_up_weight).chunk(2, dim=-1)
            activation = F.silu(gate) * up
            importance += activation.square().sum(dim=(0, 1))
            feature_cpu = feature.squeeze(0).float().cpu()
            features_cpu.append(feature_cpu)
            residuals_cpu.append((teacher_final - combined_attention).squeeze(0).float().cpu())
            token_routes.extend([_route(str(row["capability"]), specialists)] * feature_cpu.shape[0])
    importance *= down_weight.square().sum(dim=0)
    selected = torch.argsort(importance, descending=True, stable=True)[: int(protocol["selected_neurons"])]

    width = int(protocol["width"])
    rank = int(protocol["rank"])
    mean, covariance, observations = rank_audit.centered_covariance(residuals_cpu, width, device)
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
    route_names = ("generic",) + specialists
    route_weights: dict[str, torch.Tensor] = {}
    route_observations: dict[str, int] = {}
    for route_name in route_names:
        indices = torch.tensor(
            [index for index, value in enumerate(token_routes) if value == route_name],
            dtype=torch.long,
            device=device,
        )
        if indices.numel() < int(protocol["minimum_route_observations"]):
            raise Phase3Error(f"insufficient route observations: {route_name}")
        route_weights[route_name], _ = closed.solve_ridge(
            sparse_features.index_select(0, indices),
            correction_targets.index_select(0, indices),
            float(protocol["relative_ridge"]),
        )
        route_observations[route_name] = indices.numel()
    del covariance, features, coefficients, correction_targets, sparse_features

    rmses: list[float] = []
    cosines: list[float] = []
    per_route_rmses: dict[str, list[float]] = defaultdict(list)
    per_route_cosines: dict[str, list[float]] = defaultdict(list)
    record_rows: list[dict] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden, combined_attention, feature = nonlinear._interface(prefix, primary, secondary, ids)
            _, teacher_final = dual._teacher_components(teacher, 1, hidden)
            linear_coefficients = feature.float() @ linear_weights
            sparse_features = F.silu(F.linear(feature.float(), selected_gate)) * F.linear(
                feature.float(), selected_up
            )
            route_name = _route(str(row["capability"]), specialists)
            corrected_coefficients = sparse._apply_correction(
                linear_coefficients, sparse_features, route_weights[route_name]
            )
            prediction = combined_attention.float() + mean + corrected_coefficients @ basis.T
            rmse, cosine = dual.base._metrics(prediction, teacher_final.float(), hidden.float())
            rmses.append(float(rmse)); cosines.append(float(cosine))
            per_route_rmses[route_name].append(float(rmse)); per_route_cosines[route_name].append(float(cosine))
            record_rows.append({"record_id": row["record_id"], "capability": row["capability"], "oracle_route": route_name, "relative_rmse": float(rmse), "output_cosine": float(cosine)})
    mean_rmse = sum(rmses) / len(rmses)
    mean_cosine = sum(cosines) / len(cosines)
    gate = protocol["gate"]
    passed = mean_rmse <= float(gate["mean_relative_rmse_maximum"]) and mean_cosine >= float(gate["mean_output_cosine_minimum"])
    route_results = {
        name: {
            "train_observations": route_observations[name],
            "validation_records": len(per_route_cosines[name]),
            "mean_relative_rmse": sum(per_route_rmses[name]) / len(per_route_rmses[name]),
            "mean_output_cosine": sum(per_route_cosines[name]) / len(per_route_cosines[name]),
            "minimum_output_cosine": min(per_route_cosines[name]),
        }
        for name in route_names
    }
    result = {
        "format": FORMAT,
        "status": "PASS_THREE_ROUTE_LABEL_ORACLE" if passed else "FAIL_THREE_ROUTE_LABEL_ORACLE",
        "protocol_sha256": sha256_file(protocol_path),
        "training_performed": False,
        "router_trained": False,
        "artifact_written": False,
        "calibration_tokens": calibration_tokens,
        "train_observations": observations,
        "routes": route_results,
        "validation": {"mean_relative_rmse": mean_rmse, "maximum_relative_rmse": max(rmses), "mean_output_cosine": mean_cosine, "minimum_output_cosine": min(cosines), "passed": passed},
        "record_metrics": record_rows,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Perfect-label three-route layer-1 upper bound only; no inference router, artifact, English quality, runtime, certificate, or superiority claim.",
    }
    _write_immutable(output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_THREE_ROUTE_LABEL_ORACLE_PROTOCOL_V299.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_three_route/label_oracle_v300")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
