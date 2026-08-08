"""Chunked GPU extraction of the preregistered projected lexical substrate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_lexical_substrate_feasibility import projection_hash


def project_rows(rows: torch.Tensor, projection: torch.Tensor, target_norm: float) -> torch.Tensor:
    projected = rows.float().matmul(projection.float())
    projected = F.normalize(projected, dim=1, eps=1e-12) * float(target_norm)
    return projected.to(torch.float16)


def _read_tensor(snapshot: Path, index: dict[str, Any], key: str) -> torch.Tensor:
    path = snapshot / index["weight_map"][key]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_EXTRACTION_ONLY" or protocol.get("neural_training_authorized") is not False or protocol.get("device") != "cuda":
        raise Phase3Error("lexical extraction governance changed")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("lexical extraction output exists or CUDA unavailable")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"lexical extraction binding changed: {relative}")
    projection_cfg = protocol["projection"]
    if projection_hash(int(projection_cfg["source_width"]), int(projection_cfg["target_width"]), int(projection_cfg["seed"])) != projection_cfg["sha256"]:
        raise Phase3Error("lexical extraction projection changed")
    generator = torch.Generator(device="cpu").manual_seed(int(projection_cfg["seed"]))
    projection = torch.randn(int(projection_cfg["source_width"]), int(projection_cfg["target_width"]), generator=generator, dtype=torch.float32)
    projection = (projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-12)).cuda()
    source = protocol["source"]
    snapshot = Path(source["snapshot_path"])
    index = json.loads(Path(source["index_path"]).read_text(encoding="utf-8"))
    rows = int(protocol["selection"]["external_actions"])
    chunk = int(protocol["chunk_rows"])
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    outputs = {}
    statistics = {}
    for output_key, source_key, norm in (
        ("input_embedding_rows_fp16", "model.embed_tokens.weight", math.sqrt(int(projection_cfg["target_width"]))),
        ("output_head_rows_fp16", "lm_head.weight", 1.0 / math.sqrt(3.0)),
    ):
        tensor = _read_tensor(snapshot, index, source_key)
        pieces = []
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            pieces.append(project_rows(tensor[start:stop].cuda(), projection, norm).cpu())
            peak_rss = max(peak_rss, process.memory_info().rss)
        result = torch.cat(pieces).contiguous()
        norms = result.float().norm(dim=1)
        outputs[output_key] = result
        statistics[output_key] = {"rows": result.shape[0], "width": result.shape[1], "dtype": str(result.dtype), "target_norm": norm, "minimum_norm": float(norms.min()), "median_norm": float(norms.median()), "maximum_norm": float(norms.max())}
        del tensor, pieces
    output.mkdir(parents=True)
    artifact = output / "projected_lexical_substrate_fp16.safetensors"
    save_file(outputs, str(artifact), metadata={"format": "abi-projected-lexical-substrate/1", "projection_sha256": projection_cfg["sha256"], "source_revision": source["revision"]})
    elapsed = time.perf_counter() - started
    metadata = {"format": "abi-capability-compiler-phase3-lexical-substrate/1", "status": "EXTRACTED_UNVERIFIED_TRAINING_PROHIBITED", "protocol_sha256": sha256_file(protocol_path), "artifact": {"path": artifact.name, "sha256": sha256_file(artifact), "bytes": artifact.stat().st_size, "tensor_payload_bytes": sum(value.numel() * value.element_size() for value in outputs.values())}, "statistics": statistics, "projection": projection_cfg, "imported_information": {"source_tensor_rows_read": rows * 2, "source_tensor_scalars_read": rows * int(projection_cfg["source_width"]) * 2, "source_parameters_copied": 0, "projected_parameters_stored": sum(value.numel() for value in outputs.values()), "projected_payload_bytes": sum(value.numel() * value.element_size() for value in outputs.values()), "source_blocks_retained": 0, "stored_logits": 0, "stored_hidden_activations": 0}, "runtime": {"device": "cuda", "gpu": torch.cuda.get_device_name(0), "machine": platform.node(), "wall_seconds": elapsed, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_process_rss_bytes": peak_rss, "chunk_rows": chunk, "teacher_inference_seconds": 0}, "teacher_model_loaded": False, "teacher_present_at_inference": False, "training_performed": False, "phase3_certified": False, "final_test_accessed": False}
    metadata["evidence_sha256"] = hashlib.sha256((json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_LEXICAL_SUBSTRATE_EXTRACTION_PROTOCOL_V82.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_lexical_substrate/extraction_v82")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = run(root, (root / args.protocol).resolve(), (root / args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
