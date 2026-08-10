"""Read-only first-microbatch timing and numeric diagnostic for V201."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from . import capability_compiler_phase3_causal_field_core as field
from . import capability_compiler_phase3_structural_core as structural
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable


def execute(root: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_READ_ONLY_DIAGNOSTIC" or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("structural runtime diagnostic governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"structural runtime diagnostic binding changed: {name}")
    base_path = root / protocol["candidate_protocol"]
    base, _ = structural.load_protocol(root, base_path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(base["training"]["cublas_workspace_config"])
    set_determinism(int(base["training"]["seed"]))
    model, tokenizer, _ = structural._model(root, base, torch.device("cuda"))
    examples = field._examples(root, base, tokenizer)
    batch = _BalancedSampler(examples, int(base["training"]["seed"])).batch(int(base["training"]["batch_size"]))
    micro = batch[: int(base["training"]["microbatch_size"])]
    packed = field._collate(micro, torch.device("cuda"))
    total_valid = sum(len(row["target_actions"]) for row in batch)
    valid = int(packed[-1].sum().item())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(packed[0])
        forward_seconds = time.perf_counter() - started
        hard, soft = field._losses(logits, *packed[1:])
        loss = (hard + float(base["training"]["probability_field_weight"]) * soft) * (valid / total_valid)
    loss_seconds = time.perf_counter() - started - forward_seconds
    backward_started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - backward_started
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = all(bool(torch.isfinite(value).all()) for value in gradients)
    maximum = max(float(value.detach().abs().max()) for value in gradients)
    result = {
        "format": "abi-capability-compiler-phase3-structural-runtime-diagnostic/1",
        "status": "PASS_DIAGNOSTIC_COMPLETE",
        "logical_batch_size": len(batch),
        "physical_microbatch_size": len(micro),
        "valid_positions_logical": total_valid,
        "valid_positions_microbatch": valid,
        "input_shape": list(packed[0].shape),
        "logits_shape": list(logits.shape),
        "hard_nll": float(hard.detach()),
        "soft_cross_entropy": float(soft.detach()),
        "weighted_loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)),
        "unscaled_gradients_finite": finite,
        "maximum_absolute_unscaled_gradient": maximum,
        "forward_seconds": forward_seconds,
        "loss_seconds": loss_seconds,
        "backward_seconds": backward_seconds,
        "projected_logical_update_seconds": 4 * (forward_seconds + loss_seconds + backward_seconds),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_updates": 0,
        "checkpoint_created": False,
        "final_test_accessed": False,
    }
    result["attribution"] = "AMP_INITIAL_SCALE_OVERFLOW_LIKELY" if finite and maximum * 65536.0 > 65504.0 else "KERNEL_RUNTIME_DOMINANT_OR_OTHER"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_STRUCTURAL_RUNTIME_DIAGNOSTIC_PROTOCOL_V202.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_structural/runtime_diagnostic_v202.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error("structural runtime diagnostic output exists")
    result = execute(root, root / args.protocol)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
