"""Read-only source-weight V/O factorization oracle preserving all layer-1 Q/K heads."""

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
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-within-head-vo-factorization/1"


def _rank_schedule(singular_values: list[torch.Tensor], threshold: float) -> tuple[list[int], float]:
    ranks = [1] * len(singular_values)
    total = sum(float(values.square().sum()) for values in singular_values)
    selected = sum(float(values[0].square()) for values in singular_values)
    candidates = []
    for head, values in enumerate(singular_values):
        for component in range(1, values.numel()):
            candidates.append((float(values[component].square()), head, component))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for energy, head, component in candidates:
        if selected / max(total, 1e-300) >= threshold:
            break
        if component != ranks[head]:
            continue
        ranks[head] += 1
        selected += energy
    return ranks, selected / max(total, 1e-300)


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_WITHIN_HEAD_VO_FACTORIZATION"
        or int(protocol.get("source_heads", 0)) != 32
        or int(protocol.get("source_head_dimension", 0)) != 96
        or int(protocol.get("minimum_rank_per_head", 0)) != 1
        or protocol.get("rank_schedule_sweep_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("source_block_promotion") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("within-head V/O governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"within-head V/O binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("within-head V/O output exists or CUDA unavailable")

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
    _, validation_rows = coverage.expanded_split(
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
    heads = int(protocol["source_heads"])
    head_dimension = int(protocol["source_head_dimension"])
    full_width = heads * head_dimension
    query_width = heads * head_dimension
    key_width = int(source_attention.num_key_value_heads) * head_dimension
    qkv_weight = source_attention.qkv_proj.weight.detach().float()
    output_weight = source_attention.o_proj.weight.detach().float()
    if (
        int(source_attention.config.num_attention_heads) != heads
        or int(source_attention.num_key_value_heads) != heads
        or tuple(qkv_weight.shape) != (query_width + 2 * key_width, full_width)
        or tuple(output_weight.shape) != (full_width, full_width)
    ):
        raise Phase3Error("within-head V/O source topology changed")

    singular_values = []
    left_vectors = []
    right_vectors = []
    full_factor_error_numerator = 0.0
    full_factor_error_denominator = 0.0
    value_start = query_width + key_width
    for head in range(heads):
        value_weight = qkv_weight[
            value_start + head * head_dimension : value_start + (head + 1) * head_dimension
        ]
        head_output_weight = output_weight[
            :, head * head_dimension : (head + 1) * head_dimension
        ]
        q_left, r_left = torch.linalg.qr(head_output_weight, mode="reduced")
        q_right, r_right = torch.linalg.qr(value_weight.transpose(0, 1), mode="reduced")
        core = r_left @ r_right.transpose(0, 1)
        core_left, values, core_right_h = torch.linalg.svd(core, full_matrices=False)
        reconstructed_core = (core_left * values.unsqueeze(0)) @ core_right_h
        full_factor_error_numerator += float((reconstructed_core - core).square().sum())
        full_factor_error_denominator += float(core.square().sum())
        singular_values.append(values)
        left_vectors.append(q_left @ core_left)
        right_vectors.append(q_right @ core_right_h.transpose(0, 1))

    energy_threshold = float(protocol["operator_energy_threshold"])
    ranks, achieved_energy = _rank_schedule(singular_values, energy_threshold)
    total_rank = sum(ranks)
    full_factor_relative_error = (
        full_factor_error_numerator / max(full_factor_error_denominator, 1e-300)
    ) ** 0.5

    def exact_attention(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = candidate.shape[1]
        position_ids = torch.arange(length, device=device)[None]
        position_embeddings = teacher.model.rotary_emb(candidate, position_ids)
        mask = dual.base._causal_mask(length, device=device, dtype=candidate.dtype)
        normalized = source_layer1.input_layernorm(candidate)
        delta, weights = source_attention(
            hidden_states=normalized,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        return candidate + source_layer1.resid_attn_dropout(delta), weights, normalized

    terminal = int(base["source"]["terminal_token_id"])
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    route_exact = 0
    final_cosines = []
    final_rmses = []
    attention_cosines = []
    attention_rmses = []
    record_metrics = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, row in enumerate(validation_rows):
            host_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            route_index = model._select_route(host_ids)
            route_exact += int(route_index == routed._route(str(row["capability"])))
            candidate = model.token_embedding(host_ids).to(device)
            positions = torch.arange(candidate.shape[1], device=device)
            candidate, _, _ = layer0.forward_with_cache(candidate, positions, route_index)
            source_attention_output, attention_weights, normalized = exact_attention(candidate)
            compressed_delta = torch.zeros_like(candidate, dtype=torch.float32)
            for head, rank in enumerate(ranks):
                values = singular_values[head][:rank]
                root = torch.sqrt(values)
                value_factor = root.unsqueeze(1) * right_vectors[head][:, :rank].transpose(0, 1)
                output_factor = left_vectors[head][:, :rank] * root.unsqueeze(0)
                compressed_values = F.linear(normalized.float(), value_factor)
                attended = attention_weights[:, head].float() @ compressed_values
                compressed_delta.add_(F.linear(attended, output_factor))
            compressed_attention = candidate.float() + compressed_delta
            compressed_final = compressed_attention + source_layer1.mlp(
                source_layer1.post_attention_layernorm(compressed_attention)
            )
            source_ids = torch.tensor(
                [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]],
                dtype=torch.long,
                device=device,
            )
            native = teacher.model.embed_tokens(source_ids)
            for source_index in range(2):
                _, native = dual._teacher_components(teacher, source_index, native)
            final_cosine, final_rmse = trajectory._metrics(compressed_final.float(), native.float())
            attention_cosine, attention_rmse = trajectory._metrics(
                compressed_attention.float(), source_attention_output.float()
            )
            final_cosines.append(final_cosine)
            final_rmses.append(final_rmse)
            attention_cosines.append(attention_cosine)
            attention_rmses.append(attention_rmse)
            record_metrics.append(
                {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "final_cosine": final_cosine,
                    "final_relative_rmse": final_rmse,
                    "attention_cosine": attention_cosine,
                    "attention_relative_rmse": attention_rmse,
                }
            )
            peak_rss = max(peak_rss, process.memory_info().rss)
            if (index + 1) % 10 == 0:
                print(json.dumps({"factorized_validation_records": index + 1}), flush=True)

    mean_final_cosine = sum(final_cosines) / len(final_cosines)
    mean_final_rmse = sum(final_rmses) / len(final_rmses)
    artifact_after = sha256_file(artifact_path)
    gates = {
        "full_factorization_exact": full_factor_relative_error
        <= float(protocol["gates"]["full_factorization_relative_error_maximum"]),
        "operator_energy": achieved_energy >= energy_threshold,
        "strict_total_rank_compression": total_rank < heads * head_dimension,
        "all_head_patterns_preserved": all(rank >= 1 for rank in ranks),
        "validation_mean_cosine": mean_final_cosine
        >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": mean_final_rmse
        <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "routes_exact": route_exact == len(validation_rows),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    qk_madds = 2 * full_width * head_dimension * heads
    vo_madds = 2 * full_width * total_rank
    result = {
        "format": FORMAT,
        "status": "PASS_WITHIN_HEAD_VO_FACTORIZATION" if passed else "FAIL_WITHIN_HEAD_VO_FACTORIZATION",
        "protocol_sha256": sha256_file(protocol_path),
        "source_heads": heads,
        "source_head_dimension": head_dimension,
        "rank_schedule": ranks,
        "total_internal_rank": total_rank,
        "operator_energy_threshold": energy_threshold,
        "achieved_operator_energy": achieved_energy,
        "full_factorization_relative_error": full_factor_relative_error,
        "physical_accounting": {
            "exact_qk_projection_multiply_adds_per_token": qk_madds,
            "factored_vo_projection_multiply_adds_per_token": vo_madds,
            "total_projection_multiply_adds_per_token": qk_madds + vo_madds,
            "key_cache_width": heads * head_dimension,
            "value_cache_width": total_rank,
            "factored_vo_parameters": vo_madds,
            "copied_qk_parameters": qk_madds,
        },
        "validation": {
            "records": len(validation_rows),
            "mean_final_cosine": mean_final_cosine,
            "minimum_final_cosine": min(final_cosines),
            "mean_final_relative_rmse": mean_final_rmse,
            "maximum_final_relative_rmse": max(final_rmses),
            "mean_attention_cosine": sum(attention_cosines) / len(attention_cosines),
            "mean_attention_relative_rmse": sum(attention_rmses) / len(attention_rmses),
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
        "rank_schedule_sweep_performed": False,
        "training_performed": False,
        "artifact_written": False,
        "source_blocks_promoted": 0,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only source-weight-derived within-head V/O factorization oracle preserving exact source Q/K patterns only; no factors were installed or promoted and no physical runtime, model, Phase 3, or superiority claim is made.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_WITHIN_HEAD_VO_FACTORIZATION_PROTOCOL_V414.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/within_head_vo_v415",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
