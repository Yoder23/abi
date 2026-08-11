"""Fully CPU candidate-router repair for the matched runtime screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_cpu_runtime as base
from . import capability_compiler_phase3_cpu_runtime_v2 as qwen_repair
from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-matched-cpu-runtime-full-device-repair/1"


def _load_router_cpu(root: Path, protocol: Mapping[str, Any]):
    router_protocol = json.loads((root / protocol["router"]["protocol"]).read_text(encoding="utf-8")); _, _, tokenizer_type, _, _ = sparse._layercake_api(root, router_protocol); tokenizer = sparse._tokenizer(root, router_protocol, tokenizer_type); model = sparse._model(router_protocol, tokenizer.vocab_size); checkpoint = root / protocol["router"]["candidate_dir"] / "router.safetensors"; model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True); return model.cpu().eval(), tokenizer, router_protocol


@torch.inference_mode()
def _score_on_model_device(model: sparse.SparseRouter, tokenizer: Any, protocol: Mapping[str, Any], texts: Sequence[str]) -> torch.Tensor:
    device = next(model.parameters()).device; values = [sparse._features(tokenizer, protocol, text) for text in texts]; bpe_ids, bpe_offsets = sparse._bag([value[0] for value in values], device); character_ids, character_offsets = sparse._bag([value[1] for value in values], device); return model(bpe_ids, bpe_offsets, character_ids, character_offsets)


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    repair = protocol.get("candidate_device_repair", {})
    if protocol.get("format") != base.FORMAT or protocol.get("status") != "PREREGISTERED_MATCHED_CPU_RUNTIME_TTFT_RSS_SCREEN" or repair.get("format") != FORMAT or repair.get("router_checkpoint_device") != "cpu" or repair.get("feature_tensor_device") != "router_parameter_device" or repair.get("maximum_cuda_allocated_bytes") != 0: raise Phase3Error("full CPU device repair governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"full CPU device repair binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable fully CPU runtime output exists: {output}")
    original_loader = base._load_router; original_score = sparse._score; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); before = int(torch.cuda.memory_allocated()); base._load_router = _load_router_cpu; sparse._score = _score_on_model_device
    try: repaired = qwen_repair.run(root, protocol_path, output / "device_repaired_v512")
    finally: base._load_router = original_loader; sparse._score = original_score
    after = int(torch.cuda.memory_allocated()); peak = int(torch.cuda.max_memory_allocated()); candidate_cpu = before == 0 and after == 0 and peak == 0
    gates = dict(repaired["gates"]); gates["candidate_router_and_features_physically_cpu_only"] = candidate_cpu; passed = all(gates.values())
    result = {"format": FORMAT, "status": "PASS_FULLY_CPU_RUNTIME_PHASE3_CERTIFICATE_REVIEW_OPEN" if passed else "FAIL_FULLY_CPU_RUNTIME_GATE_CLOSED", "protocol_sha256": protocol_sha, "repaired_runtime_result_path": "device_repaired_v512/result.json", "repaired_runtime_result_sha256": sha256_file(output / "device_repaired_v512" / "result.json"), "candidate_device": {"model": "cpu", "router_checkpoint_load": "cpu", "router_feature_tensors": "router parameter device", "cuda_allocated_before_bytes": before, "cuda_allocated_after_bytes": after, "cuda_peak_allocated_bytes": peak, "fully_cpu": candidate_cpu}, "candidate": repaired["candidate"], "phase2_host_without_bridge": repaired["phase2_host_without_bridge"], "optimized_transformer": repaired["optimized_transformer"], "comparisons": repaired["comparisons"], "prompt_depth": repaired["prompt_depth"], "hardware": repaired["hardware"], "gates": gates, "passed": passed, "teacher_present_at_inference": False, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False}; result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CPU_RUNTIME_FULL_REPAIR_PROTOCOL_V511.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_cpu_runtime/runtime_v512"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
