"""Read-only rank-384 output-span oracle for exact layer-1 attention deltas."""

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


FORMAT = "abi-capability-compiler-phase3-layer1-attention-span-oracle/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_LAYER1_RANK384_ATTENTION_SPAN_ORACLE"
        or int(protocol.get("rank", 0)) != 384
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("source_block_promotion") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("layer1 attention-span governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"layer1 attention-span binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("layer1 attention-span output exists or CUDA unavailable")

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

    terminal = int(base["source"]["terminal_token_id"])
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    route_exact = 0

    def capture(rows: list[dict], *, validation: bool) -> list[dict]:
        nonlocal peak_rss, route_exact
        values = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, row in enumerate(rows):
                host_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
                route_index = model._select_route(host_ids)
                route_exact += int(route_index == routed._route(str(row["capability"])))
                candidate = model.token_embedding(host_ids).to(device)
                positions = torch.arange(candidate.shape[1], device=device)
                candidate, _, _ = layer0.forward_with_cache(candidate, positions, route_index)
                exact_attention, _ = dual._teacher_components(teacher, 1, candidate)
                item = {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "delta": (exact_attention - candidate).squeeze(0).float().cpu(),
                }
                if validation:
                    source_ids = torch.tensor(
                        [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]],
                        dtype=torch.long,
                        device=device,
                    )
                    native = teacher.model.embed_tokens(source_ids)
                    for source_index in range(2):
                        _, native = dual._teacher_components(teacher, source_index, native)
                    item["candidate"] = candidate.squeeze(0).float().cpu()
                    item["exact_attention"] = exact_attention.squeeze(0).float().cpu()
                    item["native_target"] = native.squeeze(0).float().cpu()
                values.append(item)
                peak_rss = max(peak_rss, process.memory_info().rss)
                if (index + 1) % 500 == 0:
                    print(json.dumps({"attention_span_records": index + 1}), flush=True)
        return values

    train_cache = capture(train_rows, validation=False)
    validation_cache = capture(validation_rows, validation=True)
    width = int(protocol["full_width"])
    rank = int(protocol["rank"])
    mean, covariance, observations = rank_audit.centered_covariance(
        [row["delta"] for row in train_cache], width, device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    rank_energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))

    final_cosines = []
    final_rmses = []
    attention_cosines = []
    attention_rmses = []
    record_metrics = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_cache:
            delta = row["delta"].to(device)
            reconstructed_delta = mean + ((delta - mean) @ basis) @ basis.transpose(0, 1)
            reconstructed_attention = row["candidate"].to(device) + reconstructed_delta
            prediction = reconstructed_attention + source_layer1.mlp(
                source_layer1.post_attention_layernorm(reconstructed_attention)
            )
            final_cosine, final_rmse = trajectory._metrics(
                prediction.float(), row["native_target"].to(device)
            )
            attention_cosine, attention_rmse = trajectory._metrics(
                reconstructed_attention.float(), row["exact_attention"].to(device)
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

    artifact_after = sha256_file(artifact_path)
    mean_final_cosine = sum(final_cosines) / len(final_cosines)
    mean_final_rmse = sum(final_rmses) / len(final_rmses)
    gates = {
        "rank_energy": rank_energy >= float(protocol["gates"]["rank_energy_minimum"]),
        "validation_mean_cosine": mean_final_cosine
        >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": mean_final_rmse
        <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "routes_exact": route_exact == len(train_rows) + len(validation_rows),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    result = {
        "format": FORMAT,
        "status": "PASS_RANK384_ATTENTION_OUTPUT_SPAN" if passed else "FAIL_RANK384_ATTENTION_OUTPUT_SPAN",
        "protocol_sha256": sha256_file(protocol_path),
        "rank": rank,
        "rank_energy": rank_energy,
        "train_observations": observations,
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
        "training_performed": False,
        "artifact_written": False,
        "source_blocks_promoted": 0,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only direct-coefficient rank-384 exact-attention-delta output-span oracle with exact source MLP diagnostic only; no realizable attention, source-block retention, model, runtime, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_LAYER1_ATTENTION_SPAN_ORACLE_PROTOCOL_V408.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/layer1_attention_span_v409",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
