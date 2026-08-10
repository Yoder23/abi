"""Derive the single minimum layer-1 residual dimension required by the locked energy gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_existing_attention_refit as coverage
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-minimum-residual-dimension/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_MINIMUM_RESIDUAL_DIMENSION"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("rank_sweep_authorized") is not False
    ):
        raise Phase3Error("minimum residual-dimension governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"minimum residual-dimension binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("minimum residual-dimension output exists or CUDA unavailable")

    output.mkdir(parents=True)
    set_determinism(int(protocol["seed"]))
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = root / protocol["artifact"]["directory"]
    artifact_path = artifact / "model.safetensors"
    artifact_before = sha256_file(artifact_path)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str((root / protocol["layercake_host"]).resolve()))

    from layercake.routed_sparse_rank768_progressive_core_fp16 import (
        PrecisionConformantRoutedSparseRank768ProgressiveCore,
    )
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    model = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(artifact_path), device="cpu"), strict=True, assign=True)
    state = model.state_dict()
    prefix = load_file(str(root / protocol["layer0_checkpoint"]["path"]), device="cpu")
    with torch.no_grad():
        for name, value in prefix.items():
            state[name].copy_(value.to(state[name].dtype))

    layer0 = model.layers[0].float().cuda().eval()
    layer1 = model.layers[1].float().cuda().eval()
    examples = sequential.field._examples(root, base, tokenizer)
    train_rows, validation_rows = coverage.expanded_split(
        examples,
        seed=int(base["training"]["seed"]),
        maximum_tokens=int(protocol["population"]["maximum_sequence_actions"]),
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    for value in teacher.parameters():
        value.requires_grad_(False)

    terminal = int(base["source"]["terminal_token_id"])
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    route_exact = 0

    def capture(rows: list[dict], *, validation: bool) -> list[dict]:
        nonlocal route_exact, peak_rss
        values = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, row in enumerate(rows):
                host_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
                route_index = model._select_route(host_ids)
                route_exact += int(route_index == routed._route(str(row["capability"])))
                candidate = model.token_embedding(host_ids).to(device)
                positions = torch.arange(candidate.shape[1], device=device)
                candidate, _, _ = layer0.forward_with_cache(candidate, positions, route_index)
                attention = routed._attention(layer1, candidate, positions)
                source_ids = torch.tensor(
                    [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]],
                    dtype=torch.long,
                    device=device,
                )
                native = teacher.model.embed_tokens(source_ids)
                for source_index in range(2):
                    _, native = dual._teacher_components(teacher, source_index, native)
                item = {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "residual": (native - attention).squeeze(0).float().cpu(),
                }
                if validation:
                    item["attention"] = attention.squeeze(0).float().cpu()
                    item["target"] = native.squeeze(0).float().cpu()
                values.append(item)
                peak_rss = max(peak_rss, process.memory_info().rss)
                if (index + 1) % 500 == 0:
                    print(json.dumps({"capture_records": index + 1}), flush=True)
        return values

    train_cache = capture(train_rows, validation=False)
    validation_cache = capture(validation_rows, validation=True)
    del teacher
    torch.cuda.empty_cache()

    width = int(protocol["full_width"])
    mean, covariance, observations = rank_audit.centered_covariance(
        [row["residual"] for row in train_cache], width, device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    cumulative = torch.cumsum(eigenvalues, dim=0) / eigenvalues.sum().clamp_min(1e-12)
    threshold = float(protocol["residual_energy_threshold"])
    required_rank = int(
        torch.searchsorted(cumulative, torch.tensor(threshold, device=device), right=False).item()
    ) + 1
    basis = eigenvectors.flip(1)[:, :required_rank].contiguous()
    achieved_energy = float(cumulative[required_rank - 1])
    prior_rank = int(protocol["prior_rank"])
    prior_rank_energy = float(cumulative[prior_rank - 1])

    cosines = []
    rmses = []
    record_metrics = []
    with torch.inference_mode():
        for row in validation_cache:
            residual = row["residual"].to(device)
            centered = residual - mean
            predicted_residual = mean + (centered @ basis) @ basis.transpose(0, 1)
            prediction = row["attention"].to(device) + predicted_residual
            cosine, rmse = trajectory._metrics(prediction, row["target"].to(device))
            cosines.append(cosine)
            rmses.append(rmse)
            record_metrics.append(
                {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "cosine": cosine,
                    "relative_rmse": rmse,
                }
            )

    delta_rank = required_rank - prior_rank
    active_delta_per_layer = delta_rank * int(protocol["cost_model"]["active_multiply_adds_per_added_rank"])
    stored_delta_per_layer = delta_rank * int(protocol["cost_model"]["stored_parameters_per_added_rank"])
    active_baseline = int(protocol["cost_model"]["baseline_active_projection_multiply_adds_per_layer"])
    artifact_after = sha256_file(artifact_path)
    mean_cosine = sum(cosines) / len(cosines)
    mean_rmse = sum(rmses) / len(rmses)
    gates = {
        "derived_energy": achieved_energy >= threshold,
        "validation_mean_cosine": mean_cosine
        >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": mean_rmse
        <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "routes_exact": route_exact == len(train_rows) + len(validation_rows),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    result = {
        "format": FORMAT,
        "status": "PASS_MINIMUM_RESIDUAL_DIMENSION_DEFINED" if passed else "FAIL_MINIMUM_RESIDUAL_DIMENSION",
        "protocol_sha256": sha256_file(protocol_path),
        "prior_rank": prior_rank,
        "prior_rank_energy": prior_rank_energy,
        "residual_energy_threshold": threshold,
        "required_rank": required_rank,
        "required_rank_energy": achieved_energy,
        "rank_increase": delta_rank,
        "cost_model": {
            "stored_parameter_increase_per_layer": stored_delta_per_layer,
            "active_multiply_add_increase_per_token_per_layer": active_delta_per_layer,
            "baseline_active_projection_multiply_adds_per_layer": active_baseline,
            "estimated_projection_compute_increase_fraction": active_delta_per_layer / active_baseline,
        },
        "train_observations": observations,
        "validation": {
            "records": len(validation_rows),
            "mean_cosine": mean_cosine,
            "minimum_cosine": min(cosines),
            "mean_relative_rmse": mean_rmse,
            "maximum_relative_rmse": max(rmses),
            "record_metrics": record_metrics,
        },
        "route_correct": route_exact,
        "gates": gates,
        "passed": passed,
        "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": artifact_after,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "rank_sweep_performed": False,
        "training_performed": False,
        "artifact_written": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only analytically derived minimum residual-dimension threshold and direct-coefficient validation oracle only; no realizable model, physical runtime, autonomous quality, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_MINIMUM_RESIDUAL_DIMENSION_PROTOCOL_V397.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/minimum_residual_dimension_v398",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
