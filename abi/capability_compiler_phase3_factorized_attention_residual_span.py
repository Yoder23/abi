"""Read-only minimum residual span behind the fixed factorized layer-1 attention."""

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
import torch.nn.functional as F

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_existing_attention_refit as coverage
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from . import capability_compiler_phase3_within_head_vo_factorization as factorization
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-factorized-attention-residual-span/1"


def _factors(attention, heads: int, head_dimension: int, threshold: float):
    """Derive the fixed V/O factors in fp64 and return fp32 runtime tensors."""
    width = heads * head_dimension
    qkv = attention.qkv_proj.weight.detach().double()
    output = attention.o_proj.weight.detach().double()
    if tuple(qkv.shape) != (3 * width, width) or tuple(output.shape) != (width, width):
        raise Phase3Error("factorized-attention source topology changed")
    singular_values = []
    left_vectors = []
    right_vectors = []
    value_start = 2 * width
    for head in range(heads):
        value = qkv[value_start + head * head_dimension:value_start + (head + 1) * head_dimension]
        head_output = output[:, head * head_dimension:(head + 1) * head_dimension]
        q_left, r_left = torch.linalg.qr(head_output, mode="reduced")
        q_right, r_right = torch.linalg.qr(value.transpose(0, 1), mode="reduced")
        core_left, values, core_right_h = torch.linalg.svd(
            r_left @ r_right.transpose(0, 1), full_matrices=False
        )
        singular_values.append(values)
        left_vectors.append(q_left @ core_left)
        right_vectors.append(q_right @ core_right_h.transpose(0, 1))
    ranks, achieved = factorization._rank_schedule(singular_values, threshold)
    runtime = []
    for head, rank in enumerate(ranks):
        values = singular_values[head][:rank]
        root = torch.sqrt(values)
        runtime.append(
            (
                (root.unsqueeze(1) * right_vectors[head][:, :rank].transpose(0, 1)).float(),
                (left_vectors[head][:, :rank] * root.unsqueeze(0)).float(),
            )
        )
    return ranks, achieved, runtime


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_FACTORIZED_ATTENTION_RESIDUAL_SPAN"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("source_block_promotion") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("rank_sweep_authorized") is not False
    ):
        raise Phase3Error("factorized-attention residual-span governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"factorized-attention residual-span binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("factorized-attention residual-span output exists or CUDA unavailable")

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
    source_layer1 = teacher.model.layers[1]
    source_attention = source_layer1.self_attn
    heads = int(protocol["factorized_attention"]["source_heads"])
    head_dimension = int(protocol["factorized_attention"]["source_head_dimension"])
    factor_ranks, factor_energy, runtime_factors = _factors(
        source_attention,
        heads,
        head_dimension,
        float(protocol["factorized_attention"]["operator_energy_threshold"]),
    )
    if factor_ranks != [int(value) for value in protocol["factorized_attention"]["locked_rank_schedule"]]:
        raise Phase3Error("factorized-attention rank schedule changed")

    terminal = int(base["source"]["terminal_token_id"])
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    route_exact = 0

    def factorized_attention(candidate: torch.Tensor) -> torch.Tensor:
        length = candidate.shape[1]
        position_ids = torch.arange(length, device=device)[None]
        position_embeddings = teacher.model.rotary_emb(candidate, position_ids)
        mask = dual.base._causal_mask(length, device=device, dtype=candidate.dtype)
        normalized = source_layer1.input_layernorm(candidate)
        _, weights = source_attention(
            hidden_states=normalized,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        delta = torch.zeros_like(candidate, dtype=torch.float32)
        for head, (value_factor, output_factor) in enumerate(runtime_factors):
            compressed_values = F.linear(normalized.float(), value_factor)
            attended = weights[:, head].float() @ compressed_values
            delta.add_(F.linear(attended, output_factor))
        return candidate.float() + delta

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
                attention = factorized_attention(candidate)
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
                    "residual": (native.float() - attention).squeeze(0).cpu(),
                }
                if validation:
                    item["attention"] = attention.squeeze(0).cpu()
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

    cosines = []
    rmses = []
    record_metrics = []
    with torch.inference_mode():
        for row in validation_cache:
            residual = row["residual"].to(device)
            predicted_residual = rank_audit.project_with_basis(residual, mean, basis)
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

    artifact_after = sha256_file(artifact_path)
    mean_cosine = sum(cosines) / len(cosines)
    mean_rmse = sum(rmses) / len(rmses)
    gates = {
        "fixed_attention_energy": factor_energy
        >= float(protocol["factorized_attention"]["operator_energy_threshold"]),
        "derived_residual_energy": achieved_energy >= threshold,
        "strict_residual_compression": required_rank < width,
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
        "status": "PASS_FACTORIZED_ATTENTION_RESIDUAL_SPAN" if passed else "FAIL_FACTORIZED_ATTENTION_RESIDUAL_SPAN",
        "protocol_sha256": sha256_file(protocol_path),
        "factorized_attention": {
            "rank_schedule": factor_ranks,
            "total_internal_rank": sum(factor_ranks),
            "achieved_operator_energy": factor_energy,
        },
        "residual_energy_threshold": threshold,
        "required_residual_rank": required_rank,
        "required_residual_rank_energy": achieved_energy,
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
        "source_blocks_promoted": 0,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only analytically derived minimum residual span behind the fixed stable factorized layer-1 attention; direct validation coefficients only, with no installed residual mapper, model, runtime, autonomous, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_FACTORIZED_ATTENTION_RESIDUAL_SPAN_PROTOCOL_V418.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/factorized_attention_residual_span_v419",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
