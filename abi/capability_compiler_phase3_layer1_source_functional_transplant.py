"""Read-only layer-1 source-functional transplant diagnostic behind the passing prefix."""

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


FORMAT = "abi-capability-compiler-phase3-layer1-source-functional-transplant/1"


def _summary(records: list[dict]) -> dict:
    cosines = [row["cosine"] for row in records]
    rmses = [row["relative_rmse"] for row in records]
    return {
        "records": len(records),
        "mean_cosine": sum(cosines) / len(cosines),
        "minimum_cosine": min(cosines),
        "mean_relative_rmse": sum(rmses) / len(rmses),
        "maximum_relative_rmse": max(rmses),
        "record_metrics": records,
    }


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_LAYER1_SOURCE_FUNCTIONAL_TRANSPLANT"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("source_block_promotion") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("source-functional transplant governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"source-functional transplant binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("source-functional transplant output exists or CUDA unavailable")

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
    layer0_checkpoint = load_file(str(root / protocol["layer0_checkpoint"]["path"]), device="cpu")
    with torch.no_grad():
        for name, value in layer0_checkpoint.items():
            state[name].copy_(value.to(state[name].dtype))

    layer0 = model.layers[0].float().cuda().eval()
    compact_layer1 = model.layers[1].float().cuda().eval()
    examples = sequential.field._examples(root, base, tokenizer)
    train_rows, validation_rows = coverage.expanded_split(
        examples,
        seed=int(base["training"]["seed"]),
        maximum_tokens=int(protocol["population"]["maximum_sequence_actions"]),
    )
    validation_ids = {str(row["record_id"]) for row in validation_rows}
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

    terminal = int(base["source"]["terminal_token_id"])
    route_exact = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    arms = {
        "exact_source_block_on_replacement_prefix": {"train": [], "validation": []},
        "compact_attention_plus_exact_source_mlp": {"train": [], "validation": []},
    }
    rows = train_rows + validation_rows
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row_index, row in enumerate(rows):
            host_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            route_index = model._select_route(host_ids)
            route_exact += int(route_index == routed._route(str(row["capability"])))
            candidate = model.token_embedding(host_ids).to(device)
            positions = torch.arange(candidate.shape[1], device=device)
            candidate, _, _ = layer0.forward_with_cache(candidate, positions, route_index)

            source_ids = torch.tensor(
                [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]],
                dtype=torch.long,
                device=device,
            )
            native = teacher.model.embed_tokens(source_ids)
            for source_index in range(2):
                _, native = dual._teacher_components(teacher, source_index, native)

            _, exact_source_block = dual._teacher_components(teacher, 1, candidate)
            compact_attention = routed._attention(compact_layer1, candidate, positions)
            exact_source_mlp = compact_attention + source_layer1.mlp(
                source_layer1.post_attention_layernorm(compact_attention)
            )
            split = "validation" if str(row["record_id"]) in validation_ids else "train"
            for arm_name, prediction in (
                ("exact_source_block_on_replacement_prefix", exact_source_block),
                ("compact_attention_plus_exact_source_mlp", exact_source_mlp),
            ):
                cosine, rmse = trajectory._metrics(prediction.float(), native.float())
                arms[arm_name][split].append(
                    {
                        "record_id": row["record_id"],
                        "capability": row["capability"],
                        "cosine": cosine,
                        "relative_rmse": rmse,
                    }
                )
            peak_rss = max(peak_rss, process.memory_info().rss)
            if (row_index + 1) % 500 == 0:
                print(json.dumps({"diagnostic_records": row_index + 1}), flush=True)

    summaries = {
        arm_name: {split: _summary(metrics) for split, metrics in populations.items()}
        for arm_name, populations in arms.items()
    }
    threshold_cosine = float(protocol["gates"]["validation_mean_cosine_minimum"])
    threshold_rmse = float(protocol["gates"]["validation_mean_relative_rmse_maximum"])

    def arm_pass(name: str) -> bool:
        values = summaries[name]["validation"]
        return values["mean_cosine"] >= threshold_cosine and values["mean_relative_rmse"] <= threshold_rmse

    exact_block_pass = arm_pass("exact_source_block_on_replacement_prefix")
    compact_attention_exact_mlp_pass = arm_pass("compact_attention_plus_exact_source_mlp")
    artifact_after = sha256_file(artifact_path)
    integrity_gates = {
        "routes_exact": route_exact == len(rows),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    if not all(integrity_gates.values()):
        diagnosis = "INVALID_DIAGNOSTIC_INTEGRITY_FAILURE"
    elif not exact_block_pass:
        diagnosis = "PREFIX_PROPAGATION_BLOCKER"
    elif not compact_attention_exact_mlp_pass:
        diagnosis = "ATTENTION_COMPRESSION_BLOCKER"
    else:
        diagnosis = "RESIDUAL_FUNCTION_COMPRESSION_BLOCKER"
    completed = all(integrity_gates.values())
    result = {
        "format": FORMAT,
        "status": diagnosis,
        "protocol_sha256": sha256_file(protocol_path),
        "arms": summaries,
        "arm_gates": {
            "exact_source_block_on_replacement_prefix": exact_block_pass,
            "compact_attention_plus_exact_source_mlp": compact_attention_exact_mlp_pass,
        },
        "integrity_gates": integrity_gates,
        "diagnostic_completed": completed,
        "diagnosis": diagnosis,
        "route_correct": route_exact,
        "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": artifact_after,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_performed": False,
        "artifact_written": False,
        "source_blocks_promoted": 0,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only source-functional layer-one diagnosis only; source operations are diagnostic oracles and are not retained, promoted, packaged, or claimed as a LayerCake model.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_LAYER1_SOURCE_FUNCTIONAL_TRANSPLANT_PROTOCOL_V406.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/layer1_source_functional_v407",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
