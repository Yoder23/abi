"""Read-only layer-1 audit of deterministic top-1 sparse residual extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_layer1_error_decomposition as decomposition
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-sparse-residual-layer1-audit/1"


def deterministic_kmeans(values: torch.Tensor, clusters: int, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or values.shape[0] < clusters:
        raise Phase3Error("invalid sparse clustering input")
    centroids = [values[0]]
    minimum_distance = (values - centroids[0]).square().sum(dim=1)
    for _ in range(1, clusters):
        centroids.append(values[minimum_distance.argmax()])
        candidate = (values - centroids[-1]).square().sum(dim=1)
        minimum_distance = torch.minimum(minimum_distance, candidate)
    centers = torch.stack(centroids)
    assignments = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    for _ in range(iterations):
        assignments = torch.cdist(values, centers).argmin(dim=1)
        updated = []
        for cluster in range(clusters):
            selected = values[assignments == cluster]
            if not selected.shape[0]:
                raise Phase3Error("deterministic sparse cluster became empty")
            updated.append(selected.mean(dim=0))
        centers = torch.stack(updated)
    return assignments, centers


def execute(root: Path, protocol_path: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_LAYER1_SPARSE_EXTRACTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("sparse layer-1 audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"sparse layer-1 audit binding changed: {name}")
    if not torch.cuda.is_available():
        raise Phase3Error("sparse layer-1 audit requires CUDA")
    base_protocol = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    device = torch.device("cuda")
    model, tokenizer, _, _, _ = sequential._model(root, base_protocol, device)
    state = model.state_dict()
    for layer_index in (0, 1):
        checkpoint = load_file(str(root / protocol["checkpoints"][str(layer_index)]["path"]), device="cpu")
        for name, value in checkpoint.items():
            if not name.startswith(f"layers.{layer_index}.") or name not in state:
                raise Phase3Error("sparse audit checkpoint boundary changed")
            state[name].copy_(value.to(state[name].dtype))
    model.eval(); layer_index = 1; layer = model.layers[layer_index]
    examples = sequential.field._examples(root, base_protocol, tokenizer)
    cfg = base_protocol["calibration"]
    train_rows, validation_rows, tokens = dual._calibration_examples(
        examples, seed=int(base_protocol["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        base_protocol["source"]["snapshot_path"], local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    train_deltas: list[torch.Tensor] = []; train_features: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden = dual.base._prefix_hidden(model, ids, layer_index)
            teacher_attention, teacher_final = dual._teacher_components(teacher, layer_index, hidden)
            student_attention = sequential._student_attention(layer, hidden, torch.arange(ids.shape[1], device=device))
            train_deltas.append((teacher_final - teacher_attention).squeeze(0).float().cpu())
            train_features.append(layer.post_attention_norm(student_attention).squeeze(0).float().cpu())
    global_mean, global_covariance, observations = rank_audit.centered_covariance(train_deltas, model.full_width, device)
    _, global_vectors = torch.linalg.eigh(global_covariance)
    cluster_basis = global_vectors.flip(1)[:, : int(protocol["clustering"]["projection_rank"])].contiguous()
    features = torch.cat(train_features).to(device)
    deltas = torch.cat(train_deltas).to(device)
    cluster_values = (deltas - global_mean) @ cluster_basis
    assignments, centers = deterministic_kmeans(
        cluster_values, int(protocol["experts"]), int(protocol["clustering"]["iterations"])
    )
    one_hot = torch.nn.functional.one_hot(assignments, num_classes=int(protocol["experts"])).float()
    router_weights, router_ridge = decomposition.solve_map(features, one_hot, float(protocol["relative_ridge"]))
    router_train_accuracy = float(((features @ router_weights).argmax(dim=1) == assignments).float().mean())
    expert_means = []; expert_bases = []; expert_weights = []; cluster_rows = []
    rank = int(protocol["rank_per_expert"])
    for expert in range(int(protocol["experts"])):
        selected = assignments == expert; expert_delta = deltas[selected]; expert_features = features[selected]
        mean = expert_delta.mean(dim=0)
        centered = expert_delta - mean
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.clamp_min(0).flip(0); basis = eigenvectors.flip(1)[:, :rank].contiguous()
        targets = centered @ basis
        weights, ridge = decomposition.solve_map(expert_features, targets, float(protocol["relative_ridge"]))
        prediction = expert_features @ weights
        coefficient_rmse = float(torch.sqrt((prediction - targets).square().mean() / targets.square().mean().clamp_min(1e-8)))
        expert_means.append(mean); expert_bases.append(basis); expert_weights.append(weights)
        cluster_rows.append({
            "expert": expert, "observations": int(selected.sum()),
            "energy_explained": float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12)),
            "effective_ridge": ridge, "training_coefficient_relative_rmse": coefficient_rmse,
        })
    routed_rmses = []; routed_cosines = []; oracle_rmses = []; oracle_cosines = []
    router_targets = []; router_predictions = []
    route_counts = [0 for _ in range(int(protocol["experts"]))]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden = dual.base._prefix_hidden(model, ids, layer_index)
            teacher_attention, teacher_final = dual._teacher_components(teacher, layer_index, hidden)
            student_attention = sequential._student_attention(layer, hidden, torch.arange(ids.shape[1], device=device))
            feature = layer.post_attention_norm(student_attention).float()
            teacher_delta = teacher_final.float() - teacher_attention.float()
            target_clusters = torch.cdist((teacher_delta - global_mean) @ cluster_basis, centers).argmin(dim=-1)
            routed = (feature @ router_weights).argmax(dim=-1)
            router_targets.extend(target_clusters.reshape(-1).tolist()); router_predictions.extend(routed.reshape(-1).tolist())
            candidates = []
            for expert in range(int(protocol["experts"])):
                delta = expert_means[expert] + (feature @ expert_weights[expert]) @ expert_bases[expert].T
                candidates.append(delta)
            stacked = torch.stack(candidates, dim=2)
            gather_index = routed[..., None, None].expand(*routed.shape, 1, model.full_width)
            routed_delta = stacked.gather(2, gather_index).squeeze(2)
            for expert in range(int(protocol["experts"])):
                route_counts[expert] += int((routed == expert).sum())
            per_expert_error = (stacked - teacher_delta.unsqueeze(2)).square().mean(dim=-1)
            oracle = per_expert_error.argmin(dim=2)
            oracle_index = oracle[..., None, None].expand(*oracle.shape, 1, model.full_width)
            oracle_delta = stacked.gather(2, oracle_index).squeeze(2)
            for prediction, rmses, cosines in (
                (student_attention.float() + routed_delta, routed_rmses, routed_cosines),
                (student_attention.float() + oracle_delta, oracle_rmses, oracle_cosines),
            ):
                rmse, cosine = dual.base._metrics(prediction, teacher_final.float(), hidden.float())
                rmses.append(float(rmse)); cosines.append(float(cosine))
    gate = protocol["gate"]
    def summary(rmses, cosines):
        mean_rmse = sum(rmses) / len(rmses); mean_cosine = sum(cosines) / len(cosines)
        return {
            "mean_relative_rmse": mean_rmse, "maximum_relative_rmse": max(rmses),
            "mean_output_cosine": mean_cosine, "minimum_output_cosine": min(cosines),
            "passed": mean_rmse <= gate["mean_relative_rmse_maximum"] and mean_cosine >= gate["mean_output_cosine_minimum"],
        }
    routed_summary = summary(routed_rmses, routed_cosines); oracle_summary = summary(oracle_rmses, oracle_cosines)
    validation_router_accuracy = sum(left == right for left, right in zip(router_targets, router_predictions)) / len(router_targets)
    diagnosis = "PASS_SPARSE_LOCAL_FEASIBLE" if routed_summary["passed"] else ("ROUTER_PRIMARY" if oracle_summary["passed"] else "EXPERT_MAP_PRIMARY")
    return {
        "format": FORMAT,
        "status": "PASS_SPARSE_LOCAL_FEASIBLE_NO_ARTIFACT" if routed_summary["passed"] else "FAIL_SPARSE_LOCAL_AUDIT",
        "protocol_sha256": sha256_file(protocol_path), "layer": layer_index,
        "calibration_tokens": tokens, "train_observations": observations,
        "experts": cluster_rows, "router_effective_ridge": router_ridge,
        "router_training_accuracy": router_train_accuracy,
        "router_validation_target_accuracy": validation_router_accuracy,
        "routed_validation_token_counts": route_counts,
        "routed_validation": routed_summary, "oracle_route_validation": oracle_summary,
        "diagnosis": diagnosis, "artifact_written": False, "training_performed": False,
        "teacher_present_in_artifact": False, "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only layer-1 sparse extraction audit only; no deployable artifact, autonomous English quality, measured inference, Phase 3 certificate, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_SPARSE_RESIDUAL_LAYER1_AUDIT_PROTOCOL_V263.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_sparse_residual/layer1_audit_v264.json")
    args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists(): raise Phase3Error("sparse layer-1 audit output exists")
    result = execute(root, root / args.protocol); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__": raise SystemExit(main())
