"""Read-only decomposition of the V256 layer-1 replacement error."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-layer1-error-decomposition/1"


def _passed(rmse: float, cosine: float, gate: dict[str, float]) -> bool:
    return rmse <= gate["mean_relative_rmse_maximum"] and cosine >= gate["mean_output_cosine_minimum"]


def classify_bottleneck(*, exact_residual_pass: bool, rank192_oracle_pass: bool,
                        maximum_rank_oracle_pass: bool, maximum_rank_map_pass: bool) -> str:
    if not exact_residual_pass:
        return "ATTENTION_CAPACITY_OR_OPTIMIZATION_PRIMARY"
    if not rank192_oracle_pass and maximum_rank_oracle_pass:
        return "MLP_RESIDUAL_RANK_PRIMARY"
    if maximum_rank_oracle_pass and not maximum_rank_map_pass:
        return "COEFFICIENT_MAP_PRIMARY"
    if not maximum_rank_oracle_pass:
        return "MULTIPLE_OR_UNMODELED_COMPONENTS"
    return "NO_LOCAL_CAPACITY_BLOCK_AT_MAXIMUM_AUDITED_RANK"


def execute(root: Path, protocol_path: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_LAYER1_DECOMPOSITION"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("layer-1 decomposition governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"layer-1 decomposition binding changed: {name}")
    if not torch.cuda.is_available():
        raise Phase3Error("layer-1 decomposition requires CUDA")
    base_protocol = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    device = torch.device("cuda")
    model, tokenizer, _, _, _ = sequential._model(root, base_protocol, device)
    state = model.state_dict()
    for layer_index in (0, 1):
        checkpoint = load_file(str(root / protocol["checkpoints"][str(layer_index)]["path"]), device="cpu")
        for name, value in checkpoint.items():
            if not name.startswith(f"layers.{layer_index}.") or name not in state:
                raise Phase3Error("layer checkpoint tensor boundary changed")
            state[name].copy_(value.to(state[name].dtype))
    model.eval()
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
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    layer_index = 1
    layer = model.layers[layer_index]
    train_deltas: list[torch.Tensor] = []
    train_features: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden = dual.base._prefix_hidden(model, ids, layer_index)
            teacher_attention, teacher_final = dual._teacher_components(teacher, layer_index, hidden)
            student_attention = sequential._student_attention(layer, hidden, torch.arange(ids.shape[1], device=device))
            train_deltas.append((teacher_final - teacher_attention).squeeze(0).float().cpu())
            train_features.append(layer.post_attention_norm(student_attention).squeeze(0).float().cpu())
    mean, covariance, observations = rank_audit.centered_covariance(train_deltas, model.full_width, device)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    eigenvectors = eigenvectors.flip(1).contiguous()
    ranks = [int(value) for value in protocol["ranks"]]
    maximum_rank = max(ranks)
    maximum_basis = eigenvectors[:, :maximum_rank]
    x = torch.cat(train_features).to(device)
    delta = torch.cat(train_deltas).to(device)
    maximum_targets = (delta - mean) @ maximum_basis
    gram = x.T @ x
    scale = float(torch.trace(gram) / gram.shape[0])
    ridge = float(protocol["relative_ridge"]) * scale
    factor = torch.linalg.cholesky(gram + ridge * torch.eye(gram.shape[0], device=device))
    maximum_weights = torch.cholesky_solve(x.T @ maximum_targets, factor)
    del train_features, train_deltas, delta, maximum_targets, gram, factor

    accumulators = {
        rank: {"teacher_attention_oracle": [[], []], "student_attention_oracle": [[], []], "student_attention_map": [[], []]}
        for rank in ranks
    }
    exact_residual = [[], []]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden = dual.base._prefix_hidden(model, ids, layer_index)
            teacher_attention, teacher_final = dual._teacher_components(teacher, layer_index, hidden)
            student_attention = sequential._student_attention(layer, hidden, torch.arange(ids.shape[1], device=device))
            teacher_delta = teacher_final.float() - teacher_attention.float()
            exact_prediction = student_attention.float() + teacher_delta
            rmse, cosine = dual.base._metrics(exact_prediction, teacher_final.float(), hidden.float())
            exact_residual[0].append(float(rmse)); exact_residual[1].append(float(cosine))
            student_features = layer.post_attention_norm(student_attention).float()
            for rank in ranks:
                basis = eigenvectors[:, :rank]
                oracle_delta = mean + ((teacher_delta - mean) @ basis) @ basis.T
                teacher_prediction = teacher_attention.float() + oracle_delta
                student_oracle_prediction = student_attention.float() + oracle_delta
                map_delta = mean + (student_features @ maximum_weights[:, :rank]) @ basis.T
                map_prediction = student_attention.float() + map_delta
                for name, prediction in (
                    ("teacher_attention_oracle", teacher_prediction),
                    ("student_attention_oracle", student_oracle_prediction),
                    ("student_attention_map", map_prediction),
                ):
                    rmse, cosine = dual.base._metrics(prediction, teacher_final.float(), hidden.float())
                    accumulators[rank][name][0].append(float(rmse))
                    accumulators[rank][name][1].append(float(cosine))
    gate = protocol["gate"]
    exact_mean_rmse = sum(exact_residual[0]) / len(exact_residual[0])
    exact_mean_cosine = sum(exact_residual[1]) / len(exact_residual[1])
    rank_results = []
    for rank in ranks:
        result = {"rank": rank, "training_energy_explained": float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))}
        for name, (rmses, cosines) in accumulators[rank].items():
            mean_rmse = sum(rmses) / len(rmses); mean_cosine = sum(cosines) / len(cosines)
            result[name] = {
                "mean_relative_rmse": mean_rmse, "maximum_relative_rmse": max(rmses),
                "mean_output_cosine": mean_cosine, "minimum_output_cosine": min(cosines),
                "passed": _passed(mean_rmse, mean_cosine, gate),
            }
        rank_results.append(result)
    exact_pass = _passed(exact_mean_rmse, exact_mean_cosine, gate)
    diagnosis = classify_bottleneck(
        exact_residual_pass=exact_pass,
        rank192_oracle_pass=rank_results[0]["student_attention_oracle"]["passed"],
        maximum_rank_oracle_pass=rank_results[-1]["student_attention_oracle"]["passed"],
        maximum_rank_map_pass=rank_results[-1]["student_attention_map"]["passed"],
    )
    return {
        "format": FORMAT,
        "status": "PASS_DIAGNOSTIC_COMPLETE_NO_ARTIFACT",
        "protocol_sha256": sha256_file(protocol_path),
        "layer": layer_index,
        "calibration_tokens": tokens,
        "train_observations": observations,
        "effective_ridge": ridge,
        "exact_teacher_residual_on_student_attention": {
            "mean_relative_rmse": exact_mean_rmse,
            "maximum_relative_rmse": max(exact_residual[0]),
            "mean_output_cosine": exact_mean_cosine,
            "minimum_output_cosine": min(exact_residual[1]),
            "passed": exact_pass,
        },
        "ranks": rank_results,
        "diagnosis": diagnosis,
        "teacher_present_in_artifact": False,
        "artifact_written": False,
        "training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only local error decomposition only; no deployable artifact, English quality, measured inference, Phase 3 certificate, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_LAYER1_ERROR_DECOMPOSITION_PROTOCOL_V257.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_direct_linear/layer1_error_decomposition_v258.json")
    args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists():
        raise Phase3Error("decomposition output exists")
    result = execute(root, root / args.protocol)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
