"""Time each boundary of one discarded structural optimizer update."""
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


def execute(root: Path, path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_EPHEMERAL_UPDATE_DIAGNOSTIC":
        raise Phase3Error("full-update diagnostic governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"full-update diagnostic binding changed: {name}")
    base, _ = structural.load_protocol(root, root / protocol["candidate_protocol"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_determinism(int(base["training"]["seed"]))
    model, tokenizer, _ = structural._model(root, base, torch.device("cuda"))
    examples = field._examples(root, base, tokenizer)
    batch = _BalancedSampler(examples, int(base["training"]["seed"])).batch(16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(base["training"]["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1)
    optimizer.zero_grad(set_to_none=True)
    total_valid = sum(len(row["target_actions"]) for row in batch)
    timings = []
    for offset in range(0, 16, 4):
        started = time.perf_counter()
        packed = field._collate(batch[offset:offset + 4], torch.device("cuda"))
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(packed[0])
            hard, soft = field._losses(logits, *packed[1:])
            loss = (hard + 0.5 * soft) * (int(packed[-1].sum().item()) / total_valid)
        forward_loss = time.perf_counter() - started
        backward_started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward = time.perf_counter() - backward_started
        item = {"microbatch": offset // 4, "input_shape": list(packed[0].shape), "forward_loss_seconds": forward_loss, "backward_seconds": backward, "loss": float(loss.detach())}
        timings.append(item)
        print(json.dumps(item), flush=True)
    clip_started = time.perf_counter()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.cuda.synchronize()
    clip_seconds = time.perf_counter() - clip_started
    print(json.dumps({"stage": "clip", "seconds": clip_seconds, "preclip_norm": float(norm)}), flush=True)
    step_started = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - step_started
    print(json.dumps({"stage": "optimizer", "seconds": step_seconds}), flush=True)
    return {
        "format": "abi-capability-compiler-phase3-structural-full-update-diagnostic/1",
        "status": "PASS_EPHEMERAL_UPDATE_DISCARDED",
        "microbatches": timings,
        "clip_seconds": clip_seconds,
        "preclip_gradient_norm": float(norm),
        "optimizer_step_seconds": step_seconds,
        "total_measured_seconds": sum(row["forward_loss_seconds"] + row["backward_seconds"] for row in timings) + clip_seconds + step_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "ephemeral_optimizer_updates": 1,
        "checkpoint_created": False,
        "model_discarded": True,
        "final_test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_STRUCTURAL_FULL_UPDATE_DIAGNOSTIC_PROTOCOL_V211.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_structural/full_update_diagnostic_v211.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error("full-update diagnostic output exists")
    result = execute(root, root / args.protocol)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
