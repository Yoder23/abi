"""Read-only shared-parameter gradient conflict audit for V474."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_sequence_bridge import _examples
from .capability_compiler_phase3_targeted_recovery_bridge import (
    _batch_with_prefixes,
    _load_parent,
    _set_routes,
    _weak_routes,
)
from .capability_compiler_phase3_weak_residual import (
    SharedWeakResidual,
    WEAK_CAPABILITIES,
    _attach,
)
from .capability_compiler_phase3_weak_support_audit import _load_verified_acquisition_ir
from .layercake_host import _equal_record_prompt_overlap_ce


FORMAT = "abi-capability-compiler-phase3-shared-gradient-conflict/1"
SHARED_NAMES = ("norm.weight", "norm.bias", "down.weight", "up.weight")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_GRADIENT_ATTRIBUTION"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or tuple(protocol.get("scope", {}).get("capabilities", ())) != WEAK_CAPABILITIES
    ):
        raise Phase3Error("gradient-conflict governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"gradient-conflict binding changed: {relative}")
    return protocol, sha256_file(path)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    return float(left.dot(right)) / denominator if denominator else 0.0


def _selected(examples, capability: str, count: int):
    values = sorted(
        (row for row in examples if row["capability"] == capability),
        key=lambda row: str(row["record_id"]),
    )
    if len(values) < count:
        raise Phase3Error("gradient audit sample depth changed")
    return values[:count]


def _gradient_vector(residual: SharedWeakResidual) -> torch.Tensor:
    named = dict(residual.named_parameters())
    pieces = []
    for name in SHARED_NAMES:
        gradient = named[name].grad
        if gradient is None:
            raise Phase3Error(f"shared gradient absent: {name}")
        pieces.append(gradient.detach().float().cpu().flatten())
    return torch.cat(pieces).double()


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable gradient-conflict output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("gradient-conflict CUDA unavailable")
    set_determinism(int(protocol["execution"]["seed"]))
    device = torch.device("cuda")
    model, tokenizer, _ = _load_parent(root, protocol, device)
    residual = SharedWeakResidual().to(device)
    checkpoint = root / protocol["candidate"]["checkpoint"]
    residual.load_state_dict(load_file(str(checkpoint), device="cuda"), strict=True)
    handles = _attach(model, residual)
    targeted_rows = _load_verified_acquisition_ir(root / protocol["targeted_ir"]["path"])
    anchor_rows = load_phase1_ir(root / protocol["anchor_ir"]["path"])
    target_examples = _examples(targeted_rows, tokenizer, system="A0", seed=int(protocol["execution"]["seed"]), max_tokens=int(protocol["execution"]["max_tokens"]))
    anchor_examples = _examples(anchor_rows, tokenizer, system="A0", seed=int(protocol["execution"]["seed"]) + 1, max_tokens=int(protocol["execution"]["max_tokens"]))
    count = int(protocol["execution"]["records_per_stream_per_capability"])
    batch_size = int(protocol["execution"]["batch_size"])
    vectors = {}
    losses = {}
    for capability in WEAK_CAPABILITIES:
        selected = [*_selected(target_examples, capability, count), *_selected(anchor_examples, capability, count)]
        residual.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            ids, labels, attention, prompt_lengths, routes = _batch_with_prefixes(batch, int(tokenizer.eos_token_id), device)
            _set_routes(model, _weak_routes(batch, device))
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=routes, use_cache=False)
                loss = _equal_record_prompt_overlap_ce(result["logits"], labels, ids, prompt_lengths, overlap_weight=float(protocol["execution"]["prompt_overlap_weight"]))
            (loss / (len(selected) / batch_size)).backward()
            total_loss += float(loss.detach())
            batches += 1
        vector = _gradient_vector(residual)
        vectors[capability] = vector
        losses[capability] = {
            "records": len(selected),
            "batches": batches,
            "mean_batch_loss": total_loss / batches,
            "shared_gradient_l2": float(vector.norm()),
        }
        print(json.dumps({"capability": capability, **losses[capability]}), flush=True)
    for handle in handles:
        handle.remove()
    pairwise = {}
    values = []
    for left_index, left in enumerate(WEAK_CAPABILITIES):
        for right in WEAK_CAPABILITIES[left_index + 1 :]:
            value = cosine(vectors[left], vectors[right])
            pairwise[f"{left}__{right}"] = value
            values.append(value)
    threshold = float(protocol["decision_rule"]["material_negative_cosine_maximum"])
    negative_pairs = [name for name, value in pairwise.items() if value <= threshold]
    result = {
        "format": FORMAT,
        "status": "PASS_MATERIAL_SHARED_GRADIENT_CONFLICT" if negative_pairs else "PASS_NO_MATERIAL_SHARED_GRADIENT_CONFLICT",
        "protocol_sha256": protocol_sha,
        "candidate_checkpoint_sha256": sha256_file(checkpoint),
        "measurements": losses,
        "shared_parameter_count": sum(dict(residual.named_parameters())[name].numel() for name in SHARED_NAMES),
        "pairwise_cosine": pairwise,
        "minimum_pairwise_cosine": min(values),
        "mean_pairwise_cosine": sum(values) / len(values),
        "material_negative_pairs": negative_pairs,
        "material_negative_cosine_maximum": threshold,
        "teacher_model_loaded": False,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_SHARED_GRADIENT_CONFLICT_PROTOCOL_V475.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_shared_gradient_conflict/audit_v476.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
