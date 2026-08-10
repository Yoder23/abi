"""Read-only minimum exact source-head subset oracle at replacement layer 1."""

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
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-source-head-subset-oracle/1"


def _greedy_subset(gram: torch.Tensor, threshold: float) -> tuple[list[int], list[dict]]:
    heads = gram.shape[0]
    total = float(gram.sum())
    cross = gram.sum(dim=1)
    selected: list[int] = []
    curve = []
    remaining = set(range(heads))
    for step in range(heads):
        candidates = []
        for head in sorted(remaining):
            trial = selected + [head]
            indices = torch.tensor(trial, dtype=torch.long, device=gram.device)
            error = total - 2.0 * float(cross.index_select(0, indices).sum())
            error += float(gram.index_select(0, indices).index_select(1, indices).sum())
            candidates.append((max(error, 0.0), head))
        error, head = min(candidates, key=lambda item: (item[0], item[1]))
        selected.append(head)
        remaining.remove(head)
        reconstruction = 1.0 - error / max(total, 1e-300)
        curve.append(
            {
                "selected_heads": len(selected),
                "added_head": head,
                "train_reconstruction": reconstruction,
            }
        )
        if reconstruction >= threshold:
            break
    return selected, curve


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_MINIMUM_SOURCE_HEAD_SUBSET"
        or int(protocol.get("source_heads", 0)) != 32
        or int(protocol.get("head_dimension", 0)) != 96
        or protocol.get("head_count_sweep_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("source_block_promotion") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("source-head subset governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"source-head subset binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("source-head subset output exists or CUDA unavailable")

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
    heads = int(protocol["source_heads"])
    head_dimension = int(protocol["head_dimension"])
    full_width = heads * head_dimension
    if (
        int(source_attention.config.num_attention_heads) != heads
        or int(source_attention.num_key_value_heads) != heads
        or int(source_attention.head_dim) != head_dimension
        or tuple(source_attention.o_proj.weight.shape) != (full_width, full_width)
    ):
        raise Phase3Error("source-head topology changed")

    captured: dict[str, torch.Tensor] = {}

    def capture_o_input(_module, args):
        captured["o_input"] = args[0].detach()

    hook = source_attention.o_proj.register_forward_pre_hook(capture_o_input)
    output_weight = source_attention.o_proj.weight.detach().float().view(full_width, heads, head_dimension)

    def contributions(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        captured.clear()
        attention_output, _ = dual._teacher_components(teacher, 1, candidate)
        if "o_input" not in captured:
            raise Phase3Error("source attention head capture failed")
        attended = captured["o_input"].float().view(candidate.shape[0], candidate.shape[1], heads, head_dimension)
        per_head = torch.einsum("bthd,ohd->btho", attended, output_weight)
        return attention_output, per_head

    terminal = int(base["source"]["terminal_token_id"])
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    route_exact = 0
    gram = torch.zeros(heads, heads, dtype=torch.float64, device=device)
    full_reconstruction_squared_error = 0.0
    attention_delta_squared = 0.0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, row in enumerate(train_rows):
            host_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            route_index = model._select_route(host_ids)
            route_exact += int(route_index == routed._route(str(row["capability"])))
            candidate = model.token_embedding(host_ids).to(device)
            positions = torch.arange(candidate.shape[1], device=device)
            candidate, _, _ = layer0.forward_with_cache(candidate, positions, route_index)
            exact_attention, per_head = contributions(candidate)
            exact_delta = exact_attention.float() - candidate.float()
            reconstructed_delta = per_head.sum(dim=2)
            full_reconstruction_squared_error += float((reconstructed_delta - exact_delta).square().sum())
            attention_delta_squared += float(exact_delta.square().sum())
            matrix = per_head.squeeze(0).permute(0, 2, 1).reshape(-1, heads)
            gram.add_((matrix.transpose(0, 1) @ matrix).double())
            peak_rss = max(peak_rss, process.memory_info().rss)
            if (index + 1) % 500 == 0:
                print(json.dumps({"head_census_records": index + 1}), flush=True)

    threshold = float(protocol["reconstruction_threshold"])
    selected_heads, selection_curve = _greedy_subset(gram, threshold)
    selected = torch.tensor(selected_heads, dtype=torch.long, device=device)
    selected_train_reconstruction = float(selection_curve[-1]["train_reconstruction"])
    exact_reconstruction_relative_error = (
        full_reconstruction_squared_error / max(attention_delta_squared, 1e-300)
    ) ** 0.5

    final_cosines = []
    final_rmses = []
    attention_cosines = []
    attention_rmses = []
    record_metrics = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            host_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            route_index = model._select_route(host_ids)
            route_exact += int(route_index == routed._route(str(row["capability"])))
            candidate = model.token_embedding(host_ids).to(device)
            positions = torch.arange(candidate.shape[1], device=device)
            candidate, _, _ = layer0.forward_with_cache(candidate, positions, route_index)
            exact_attention, per_head = contributions(candidate)
            subset_attention = candidate.float() + per_head.index_select(2, selected).sum(dim=2)
            subset_final = subset_attention + source_layer1.mlp(
                source_layer1.post_attention_layernorm(subset_attention)
            )
            source_ids = torch.tensor(
                [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]],
                dtype=torch.long,
                device=device,
            )
            native = teacher.model.embed_tokens(source_ids)
            for source_index in range(2):
                _, native = dual._teacher_components(teacher, source_index, native)
            final_cosine, final_rmse = trajectory._metrics(subset_final.float(), native.float())
            attention_cosine, attention_rmse = trajectory._metrics(
                subset_attention.float(), exact_attention.float()
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
    hook.remove()

    mean_final_cosine = sum(final_cosines) / len(final_cosines)
    mean_final_rmse = sum(final_rmses) / len(final_rmses)
    artifact_after = sha256_file(artifact_path)
    gates = {
        "full_head_decomposition_exact": exact_reconstruction_relative_error
        <= float(protocol["gates"]["full_head_reconstruction_relative_error_maximum"]),
        "train_reconstruction": selected_train_reconstruction >= threshold,
        "strict_head_compression": len(selected_heads) < heads,
        "validation_mean_cosine": mean_final_cosine
        >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": mean_final_rmse
        <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "routes_exact": route_exact == len(train_rows) + len(validation_rows),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    selected_count = len(selected_heads)
    active_attention_madds = selected_count * 4 * full_width * head_dimension
    result = {
        "format": FORMAT,
        "status": "PASS_MINIMUM_SOURCE_HEAD_SUBSET" if passed else "FAIL_MINIMUM_SOURCE_HEAD_SUBSET",
        "protocol_sha256": sha256_file(protocol_path),
        "source_heads": heads,
        "head_dimension": head_dimension,
        "selected_heads": selected_heads,
        "selected_head_count": selected_count,
        "selection_curve": selection_curve,
        "selected_train_reconstruction": selected_train_reconstruction,
        "full_head_reconstruction_relative_error": exact_reconstruction_relative_error,
        "physical_accounting": {
            "selected_qkv_and_output_parameters": active_attention_madds,
            "active_projection_multiply_adds_per_token": active_attention_madds,
            "kv_width": selected_count * head_dimension,
            "complete_source_attention_projection_multiply_adds_per_token": heads
            * 4
            * full_width
            * head_dimension,
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
        "head_count_sweep_performed": False,
        "training_performed": False,
        "artifact_written": False,
        "source_blocks_promoted": 0,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only train-derived minimum exact source-attention-head subset oracle only; no selected weights were copied, retained, installed, or promoted and no physical runtime, model, Phase 3, or superiority claim is made.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_SOURCE_HEAD_SUBSET_PROTOCOL_V411.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/source_head_subset_v412",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
