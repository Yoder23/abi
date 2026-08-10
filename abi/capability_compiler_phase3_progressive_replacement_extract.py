"""Extract the immutable lexical and normalization substrate for v8 replacement."""

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


FORMAT = "abi-capability-compiler-phase3-progressive-replacement-extraction/1"


def runtime_source_rows(*, external_actions: int, host_special_source_rows: list[int]) -> list[int]:
    rows = [int(value) for value in host_special_source_rows] + list(range(int(external_actions)))
    if len(rows) != external_actions + len(host_special_source_rows):
        raise Phase3Error("runtime source row accounting changed")
    return rows


def substrate_parameter_count(*, runtime_vocabulary: int, full_width: int, layers: int) -> int:
    return 2 * runtime_vocabulary * full_width + (2 * layers + 1) * full_width


def _tensor(snapshot: Path, weight_map: Mapping[str, str], key: str) -> torch.Tensor:
    relative = weight_map.get(key)
    if relative is None:
        raise Phase3Error(f"source index lacks required tensor: {key}")
    with safe_open(str(snapshot / relative), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_GPU_COPIED_SUBSTRATE_EXTRACTION"
        or protocol.get("device") != "cuda"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("progressive replacement extraction governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"progressive replacement extraction binding changed: {name}")
    return protocol, sha256_file(path)


def execute(root: Path, protocol_path: Path, output_directory: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_directory.exists() or not torch.cuda.is_available():
        raise Phase3Error("progressive substrate output exists or CUDA unavailable")
    source = protocol["source"]
    target = protocol["target"]
    snapshot = Path(source["snapshot_path"])
    weight_map = json.loads(Path(source["index_path"]).read_text(encoding="utf-8"))["weight_map"]
    rows = runtime_source_rows(
        external_actions=int(target["external_actions"]),
        host_special_source_rows=list(target["host_special_source_rows"]),
    )
    if len(rows) != int(target["runtime_vocabulary"]):
        raise Phase3Error("runtime vocabulary row mapping changed")
    expected_parameters = substrate_parameter_count(
        runtime_vocabulary=int(target["runtime_vocabulary"]),
        full_width=int(target["full_width"]),
        layers=int(target["replacement_layers"]),
    )
    if expected_parameters != int(target["copied_substrate_parameters"]):
        raise Phase3Error("copied substrate accounting changed")

    device = torch.device("cuda")
    row_index = torch.tensor(rows, dtype=torch.long, device=device)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tensors: dict[str, torch.Tensor] = {}
    for source_key, target_key in (
        ("model.embed_tokens.weight", "token_embedding.weight"),
        ("lm_head.weight", "lm_head.weight"),
    ):
        value = _tensor(snapshot, weight_map, source_key).to(device)
        tensors[target_key] = value.index_select(0, row_index).cpu().contiguous()
        del value
        peak_rss = max(peak_rss, process.memory_info().rss)
    for layer in range(int(target["replacement_layers"])):
        for source_name, target_name in (
            ("input_layernorm", "input_norm"),
            ("post_attention_layernorm", "post_attention_norm"),
        ):
            source_key = f"model.layers.{layer}.{source_name}.weight"
            tensors[f"layers.{layer}.{target_name}.weight"] = _tensor(snapshot, weight_map, source_key).contiguous()
    tensors["final_norm.weight"] = _tensor(snapshot, weight_map, "model.norm.weight").contiguous()
    torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - started
    scalar_count = sum(value.numel() for value in tensors.values())
    if scalar_count != expected_parameters or any(value.dtype != torch.bfloat16 for value in tensors.values()):
        raise Phase3Error("copied substrate scalar count or dtype changed")

    output_directory.mkdir(parents=True)
    tensor_path = output_directory / "copied_substrate.safetensors"
    save_file(tensors, str(tensor_path), metadata={"format": "abi-progressive-replacement-copied-substrate/1", "protocol_sha256": protocol_sha})
    result = {
        "format": FORMAT,
        "status": "COMPLETE_UNVERIFIED_TRAINING_PROHIBITED",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "source": {"model": source["model"], "revision": source["revision"], "complete_blocks_retained": 0},
        "row_mapping": {
            "host_special_source_rows": list(target["host_special_source_rows"]),
            "external_source_rows": [0, int(target["external_actions"]) - 1],
            "runtime_vocabulary": len(rows),
            "unique_source_rows": len(set(rows)),
            "duplicate_source_rows": len(rows) - len(set(rows)),
            "unused_source_weight_rows_omitted": int(source["vocab_size"]) - int(target["external_actions"]),
        },
        "accounting": {
            "source_parameters": int(source["parameter_count"]),
            "copied_substrate_parameters": scalar_count,
            "copied_lexical_parameters": 2 * int(target["runtime_vocabulary"]) * int(target["full_width"]),
            "copied_normalization_parameters": (2 * int(target["replacement_layers"]) + 1) * int(target["full_width"]),
            "future_trainable_replacement_parameters": int(target["future_trainable_replacement_parameters"]),
            "future_total_deployed_parameters": scalar_count + int(target["future_trainable_replacement_parameters"]),
            "tensor_payload_bytes": scalar_count * 2,
            "tensor_file_bytes": tensor_path.stat().st_size,
            "stored_logits": 0,
            "stored_activations": 0,
            "teacher_forward_tokens": 0,
            "teacher_inference_seconds": 0.0,
            "extraction_seconds": extraction_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_process_rss_bytes": max(peak_rss, process.memory_info().rss),
            "external_hardware_used": True,
            "device": torch.cuda.get_device_name(0),
        },
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in tensors.items()},
        "tensor_sha256": sha256_file(tensor_path),
        "teacher_model_loaded": False,
        "teacher_present_at_inference": False,
        "training_performed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "next_gate": "Hostile exact recomputation of every copied bf16 scalar and row mapping before any replacement training.",
        "claim_boundary": "Unverified copied lexical and normalization substrate only; no replacement weights, source blocks, quality, transfer, or superiority claim.",
    }
    _write_immutable(output_directory / "result.json", json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_PROGRESSIVE_REPLACEMENT_EXTRACTION_PROTOCOL_V219.json")
    parser.add_argument("--output-directory", default="results/abi_capability_compiler_phase3_progressive_replacement/extraction_v220")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
