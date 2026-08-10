"""Read-only fit, generalization, and structural-retention attribution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_causal_field_core as field
from . import capability_compiler_phase3_causal_field_fit as fit
from . import capability_compiler_phase3_structural_core as structural
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_READ_ONLY_ATTRIBUTION" or protocol.get("neural_training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("structural attribution governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"structural attribution binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("structural attribution output exists or CUDA unavailable")
    candidate_protocol, _ = structural.load_protocol(root, root / protocol["candidate_protocol"])
    candidate = root / protocol["candidate_dir"]
    model = structural._load_candidate(root, candidate_protocol, candidate)
    _, tokenizer_type = structural._types(root, candidate_protocol)
    tokenizer = tokenizer_type.from_document(json.loads((candidate / "tokenizer.json").read_text(encoding="utf-8")))
    acquisition = field._examples(root, candidate_protocol, tokenizer)
    development = fit._development(root, candidate_protocol, tokenizer)
    acquisition_result = fit._measure(model, acquisition, int(protocol["runtime"]["batch_size"]))
    development_result = fit._measure(model, development, int(protocol["runtime"]["batch_size"]))
    initial = load_file(str(root / protocol["initial_artifact"]), device="cpu")
    final = load_file(str(candidate / "model.safetensors"), device="cpu")
    dot = initial_norm = final_norm = delta = 0.0
    exact = total = 0
    per_tensor = {}
    for key in sorted(initial):
        left = initial[key].float().reshape(-1)
        right = final[key].float().reshape(-1)
        item_dot = float(torch.dot(left, right)); item_left = float(torch.dot(left, left)); item_right = float(torch.dot(right, right))
        item_delta = float((left - right).square().sum())
        item_exact = int(left.eq(right).sum())
        dot += item_dot; initial_norm += item_left; final_norm += item_right; delta += item_delta; exact += item_exact; total += left.numel()
        per_tensor[key] = {"cosine": item_dot / max((item_left * item_right) ** 0.5, 1e-30), "relative_l2_delta": (item_delta / max(item_left, 1e-30)) ** 0.5, "exact_entries": item_exact, "entries": left.numel()}
    gap = acquisition_result["action_accuracy"] - development_result["action_accuracy"]
    result = {
        "format": "abi-capability-compiler-phase3-structural-attribution/1",
        "status": "COMPLETE_READ_ONLY_ATTRIBUTION",
        "checkpoint_sha256": sha256_file(candidate / "model.safetensors"),
        "acquisition": acquisition_result,
        "development_teacher_forced": development_result,
        "acquisition_minus_development_action_gap": gap,
        "structural_retention": {
            "global_cosine": dot / max((initial_norm * final_norm) ** 0.5, 1e-30),
            "relative_l2_delta": (delta / max(initial_norm, 1e-30)) ** 0.5,
            "exact_entries": exact,
            "changed_entries": total - exact,
            "total_entries": total,
            "per_tensor": per_tensor,
        },
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "checkpoint_changed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
    }
    result["attribution"] = "MIXED_FIT_AND_GENERALIZATION_WITH_STRUCTURAL_INITIALIZATION_EFFECT_SMALL" if acquisition_result["action_accuracy"] < 0.99 and gap > 0.1 else "REVIEW_METRICS"
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_STRUCTURAL_ATTRIBUTION_PROTOCOL_V213.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_structural/attribution_v213.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output)
    print(json.dumps({"status": result["status"], "acquisition_action_accuracy": result["acquisition"]["action_accuracy"], "development_action_accuracy": result["development_teacher_forced"]["action_accuracy"], "global_structural_cosine": result["structural_retention"]["global_cosine"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
