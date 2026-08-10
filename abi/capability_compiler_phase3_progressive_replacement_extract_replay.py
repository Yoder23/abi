"""Bounded-memory exact replay of the copied-substrate GPU extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import psutil
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_progressive_replacement_extract import (
    FORMAT,
    runtime_source_rows,
    substrate_parameter_count,
)


REPAIR_FORMAT = "abi-capability-compiler-phase3-progressive-replacement-extraction-replay/1"


def _source_slice(snapshot: Path, weight_map: Mapping[str, str], key: str, start: int | None = None, end: int | None = None) -> torch.Tensor:
    relative = weight_map.get(key)
    if relative is None:
        raise Phase3Error(f"source index lacks required tensor: {key}")
    with safe_open(str(snapshot / relative), framework="pt", device="cpu") as handle:
        view = handle.get_slice(key)
        return view[:] if start is None else view[start:end]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    repair = json.loads(path.read_text(encoding="utf-8"))
    if (
        repair.get("format") != REPAIR_FORMAT
        or repair.get("status") != "PREREGISTERED_EXACT_BOUNDED_MEMORY_REPLAY"
        or repair.get("device") != "cuda"
        or repair.get("neural_training_authorized") is not False
        or repair.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("progressive extraction replay governance changed")
    for name, expected in repair["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"progressive extraction replay binding changed: {name}")
    base_path = root / repair["base_protocol"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("format") != FORMAT or base.get("status") != "PREREGISTERED_GPU_COPIED_SUBSTRATE_EXTRACTION":
        raise Phase3Error("base extraction protocol changed")
    return repair, base, sha256_file(path)


def _mapped_table(
    snapshot: Path,
    weight_map: Mapping[str, str],
    key: str,
    *,
    special_rows: list[int],
    external_actions: int,
    width: int,
    chunk_rows: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.empty((len(special_rows) + external_actions, width), dtype=torch.bfloat16)
    for index, source_row in enumerate(special_rows):
        value = _source_slice(snapshot, weight_map, key, source_row, source_row + 1).to(device)
        output[index:index + 1].copy_(value.cpu())
    cursor = len(special_rows)
    for start in range(0, external_actions, chunk_rows):
        end = min(start + chunk_rows, external_actions)
        value = _source_slice(snapshot, weight_map, key, start, end).to(device)
        output[cursor + start:cursor + end].copy_(value.cpu())
        if end == external_actions or end % (4 * chunk_rows) == 0:
            print(json.dumps({"tensor": key, "external_rows": end, "total": external_actions}), flush=True)
    return output.contiguous()


def execute(root: Path, protocol_path: Path, output_directory: Path) -> dict[str, Any]:
    repair, protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_directory.exists() or not torch.cuda.is_available():
        raise Phase3Error("progressive substrate replay output exists or CUDA unavailable")
    source = protocol["source"]
    target = protocol["target"]
    snapshot = Path(source["snapshot_path"])
    weight_map = json.loads(Path(source["index_path"]).read_text(encoding="utf-8"))["weight_map"]
    rows = runtime_source_rows(external_actions=int(target["external_actions"]), host_special_source_rows=list(target["host_special_source_rows"]))
    expected_parameters = substrate_parameter_count(runtime_vocabulary=int(target["runtime_vocabulary"]), full_width=int(target["full_width"]), layers=int(target["replacement_layers"]))
    if len(rows) != int(target["runtime_vocabulary"]) or expected_parameters != int(target["copied_substrate_parameters"]):
        raise Phase3Error("replay row or scalar accounting changed")

    device = torch.device("cuda")
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tensors: dict[str, torch.Tensor] = {}
    for source_key, target_key in (("model.embed_tokens.weight", "token_embedding.weight"), ("lm_head.weight", "lm_head.weight")):
        tensors[target_key] = _mapped_table(
            snapshot, weight_map, source_key,
            special_rows=list(target["host_special_source_rows"]),
            external_actions=int(target["external_actions"]),
            width=int(target["full_width"]),
            chunk_rows=int(repair["runtime_repair"]["chunk_rows"]),
            device=device,
        )
        peak_rss = max(peak_rss, process.memory_info().rss)
    for layer in range(int(target["replacement_layers"])):
        for source_name, target_name in (("input_layernorm", "input_norm"), ("post_attention_layernorm", "post_attention_norm")):
            key = f"model.layers.{layer}.{source_name}.weight"
            tensors[f"layers.{layer}.{target_name}.weight"] = _source_slice(snapshot, weight_map, key).to(device).cpu().contiguous()
        if (layer + 1) % 8 == 0:
            print(json.dumps({"normalization_layers": layer + 1, "total": int(target["replacement_layers"])}), flush=True)
    tensors["final_norm.weight"] = _source_slice(snapshot, weight_map, "model.norm.weight").to(device).cpu().contiguous()
    torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - started
    scalar_count = sum(value.numel() for value in tensors.values())
    if len(tensors) != 67 or scalar_count != expected_parameters or any(value.dtype != torch.bfloat16 for value in tensors.values()):
        raise Phase3Error("replay tensor count, scalar count, or dtype changed")

    output_directory.mkdir(parents=True)
    tensor_path = output_directory / "copied_substrate.safetensors"
    print(json.dumps({"serializing_scalars": scalar_count, "path": tensor_path.name}), flush=True)
    save_file(tensors, str(tensor_path), metadata={"format": "abi-progressive-replacement-copied-substrate/1", "protocol_sha256": protocol_sha, "base_protocol_sha256": sha256_file(root / repair["base_protocol"])})
    result = {
        "format": FORMAT,
        "status": "COMPLETE_UNVERIFIED_TRAINING_PROHIBITED",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha, "base_path": repair["base_protocol"], "base_sha256": sha256_file(root / repair["base_protocol"])},
        "preserved_failure": repair["preserved_failure"],
        "source": {"model": source["model"], "revision": source["revision"], "complete_blocks_retained": 0},
        "row_mapping": {"host_special_source_rows": list(target["host_special_source_rows"]), "external_source_rows": [0, int(target["external_actions"]) - 1], "runtime_vocabulary": len(rows), "unique_source_rows": len(set(rows)), "duplicate_source_rows": len(rows) - len(set(rows)), "unused_source_weight_rows_omitted": int(source["vocab_size"]) - int(target["external_actions"])},
        "accounting": {
            "source_parameters": int(source["parameter_count"]),
            "copied_substrate_parameters": scalar_count,
            "copied_lexical_parameters": 2 * int(target["runtime_vocabulary"]) * int(target["full_width"]),
            "copied_normalization_parameters": (2 * int(target["replacement_layers"]) + 1) * int(target["full_width"]),
            "future_trainable_replacement_parameters": int(target["future_trainable_replacement_parameters"]),
            "future_total_deployed_parameters": scalar_count + int(target["future_trainable_replacement_parameters"]),
            "tensor_payload_bytes": scalar_count * 2,
            "tensor_file_bytes": tensor_path.stat().st_size,
            "stored_logits": 0, "stored_activations": 0, "teacher_forward_tokens": 0, "teacher_inference_seconds": 0.0,
            "extraction_seconds": extraction_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_process_rss_bytes": max(peak_rss, process.memory_info().rss),
            "external_hardware_used": True,
            "device": torch.cuda.get_device_name(0),
            "chunk_rows": int(repair["runtime_repair"]["chunk_rows"]),
        },
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in tensors.items()},
        "tensor_sha256": sha256_file(tensor_path),
        "teacher_model_loaded": False, "teacher_present_at_inference": False, "training_performed": False,
        "phase3_certified": False, "final_test_accessed": False,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "next_gate": "Hostile exact recomputation of every copied bf16 scalar and row mapping before any replacement training.",
        "claim_boundary": "Unverified copied lexical and normalization substrate only; no replacement weights, source blocks, quality, transfer, or superiority claim."
    }
    _write_immutable(output_directory / "result.json", json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_PROGRESSIVE_REPLACEMENT_EXTRACTION_REPLAY_V221.json")
    parser.add_argument("--output-directory", default="results/abi_capability_compiler_phase3_progressive_replacement/extraction_replay_v222")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
