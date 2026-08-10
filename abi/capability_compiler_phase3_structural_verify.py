"""Hostile independent verifier for a structural causal extraction artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-structural-verifier/1"


def _source_tensor(snapshot: Path, weight_map: Mapping[str, str], key: str) -> torch.Tensor:
    relative = weight_map.get(key)
    if relative is None:
        raise Phase3Error(f"verification source tensor missing: {key}")
    with safe_open(str(snapshot / relative), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _select(scores: torch.Tensor, count: int) -> torch.Tensor:
    return torch.argsort(scores, descending=True, stable=True)[:count].sort().values


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_HOSTILE_GPU_VERIFICATION" or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("structural verifier governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"structural verifier binding changed: {name}")
    return protocol, sha256_file(path)


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = _load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("hostile verifier requires preregistered CUDA")
    extraction_protocol = json.loads((root / protocol["extraction_protocol"]).read_text(encoding="utf-8"))
    extraction_result = json.loads((root / protocol["extraction_result"]).read_text(encoding="utf-8"))
    artifact_path = root / protocol["artifact"]
    artifact = load_file(str(artifact_path), device="cpu")
    source = extraction_protocol["source"]
    target = extraction_protocol["target"]
    transforms = extraction_protocol["transforms"]
    snapshot = Path(source["snapshot_path"])
    weight_map = json.loads(Path(source["index_path"]).read_text(encoding="utf-8"))["weight_map"]
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    external = torch.arange(int(target["external_actions"]), dtype=torch.long, device=device)
    residual_score = torch.zeros(int(source["hidden_size"]), dtype=torch.float64, device=device)
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        value = _source_tensor(snapshot, weight_map, key).to(device)
        residual_score += value.index_select(0, external).double().square().sum(dim=0)
        del value
    residual = _select(residual_score, int(target["width"]))
    selection_match: dict[str, bool] = {"residual": residual.cpu().tolist() == extraction_result["selection"]["residual_ordered"]}

    width = int(source["hidden_size"])
    intermediate = int(source["intermediate_size"])
    head_dim = int(source["head_dim"])
    heads: dict[int, torch.Tensor] = {}
    neurons: dict[int, torch.Tensor] = {}
    for raw_layer in target["source_layers"]:
        layer = int(raw_layer)
        qkv = _source_tensor(snapshot, weight_map, f"model.layers.{layer}.self_attn.qkv_proj.weight").to(device)
        o = _source_tensor(snapshot, weight_map, f"model.layers.{layer}.self_attn.o_proj.weight").to(device)
        scores = []
        for head in range(int(source["num_attention_heads"])):
            total = o.index_select(0, residual)[:, head * head_dim:(head + 1) * head_dim].double().square().sum()
            for section in range(3):
                total = total + qkv[section * width + head * head_dim:section * width + (head + 1) * head_dim].index_select(1, residual).double().square().sum()
            scores.append(total)
        heads[layer] = _select(torch.stack(scores), int(target["num_attention_heads"]))
        selection_match[f"layer_{layer}_heads"] = heads[layer].cpu().tolist() == extraction_result["selection"]["attention_heads"][str(layer)]
        del qkv, o

        gate_up = _source_tensor(snapshot, weight_map, f"model.layers.{layer}.mlp.gate_up_proj.weight").to(device)
        down = _source_tensor(snapshot, weight_map, f"model.layers.{layer}.mlp.down_proj.weight").to(device)
        score = (
            gate_up[:intermediate].index_select(1, residual).double().square().sum(dim=1)
            + gate_up[intermediate:].index_select(1, residual).double().square().sum(dim=1)
            + down.index_select(0, residual).double().square().sum(dim=0)
        )
        neurons[layer] = _select(score, int(target["intermediate_size"]))
        selection_match[f"layer_{layer}_neurons"] = neurons[layer].cpu().tolist() == extraction_result["selection"]["mlp_neurons"][str(layer)]
        del gate_up, down, score

    exact_scalars = 0
    tensor_matches: dict[str, bool] = {}

    def check(name: str, value: torch.Tensor) -> None:
        nonlocal exact_scalars
        expected = value.to(torch.float16).cpu().contiguous()
        actual = artifact.get(name)
        matched = actual is not None and actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(actual, expected)
        tensor_matches[name] = bool(matched)
        if matched:
            exact_scalars += expected.numel()

    special = torch.tensor(extraction_protocol["selection"]["host_special_source_rows"], dtype=torch.long, device=device)
    rows = torch.cat((special, external))
    check("token_embedding.weight", _source_tensor(snapshot, weight_map, "model.embed_tokens.weight").to(device).index_select(0, rows).index_select(1, residual))
    check("lm_head.weight", _source_tensor(snapshot, weight_map, "lm_head.weight").to(device).index_select(0, rows).index_select(1, residual) * float(transforms["lm_head_scale"]))
    for target_layer, raw_layer in enumerate(target["source_layers"]):
        layer = int(raw_layer)
        prefix = f"layers.{target_layer}"
        head_columns = torch.tensor([column for head in heads[layer] for column in range(int(head) * head_dim, (int(head) + 1) * head_dim)], dtype=torch.long, device=device)
        qkv_rows = torch.tensor(
            [row for section in range(3) for head in heads[layer] for row in range(section * width + int(head) * head_dim, section * width + (int(head) + 1) * head_dim)],
            dtype=torch.long,
            device=device,
        )
        qkv = _source_tensor(snapshot, weight_map, f"model.layers.{layer}.self_attn.qkv_proj.weight").to(device)
        check(f"{prefix}.qkv_proj.weight", qkv.index_select(0, qkv_rows).index_select(1, residual) * float(transforms["qkv_scale"]))
        del qkv
        o = _source_tensor(snapshot, weight_map, f"model.layers.{layer}.self_attn.o_proj.weight").to(device)
        check(f"{prefix}.o_proj.weight", o.index_select(0, residual).index_select(1, head_columns) * float(transforms["o_scale"]))
        del o
        gate_up = _source_tensor(snapshot, weight_map, f"model.layers.{layer}.mlp.gate_up_proj.weight").to(device)
        gate_up_rows = torch.cat((neurons[layer], neurons[layer] + intermediate))
        check(f"{prefix}.gate_up_proj.weight", gate_up.index_select(0, gate_up_rows).index_select(1, residual) * float(transforms["gate_up_scale"]))
        del gate_up
        down = _source_tensor(snapshot, weight_map, f"model.layers.{layer}.mlp.down_proj.weight").to(device)
        check(f"{prefix}.down_proj.weight", down.index_select(0, residual).index_select(1, neurons[layer]) * float(transforms["down_scale"]))
        del down
        for source_name, target_name in (("input_layernorm", "input_norm"), ("post_attention_layernorm", "post_attention_norm")):
            norm = _source_tensor(snapshot, weight_map, f"model.layers.{layer}.{source_name}.weight").to(device)
            check(f"{prefix}.{target_name}.weight", norm.index_select(0, residual))
            del norm
    check("final_norm.weight", _source_tensor(snapshot, weight_map, "model.norm.weight").to(device).index_select(0, residual))
    expected_keys = set(tensor_matches)
    key_set_match = set(artifact) == expected_keys

    layercake_root = (root / target["layercake_repository"]).resolve()
    sys.path.insert(0, str(layercake_root))
    from layercake.structural_causal_core import StructuralCausalCore
    model = StructuralCausalCore(fixed_vocab_size=int(target["vocabulary"])).eval()
    model.load_state_dict(artifact, strict=True)
    input_ids = torch.tensor([[4, 5, 6, 1]], dtype=torch.long)
    with torch.inference_mode():
        cpu_logits = model(input_ids)
        cuda_model = StructuralCausalCore(fixed_vocab_size=int(target["vocabulary"])).cuda().eval()
        cuda_model.load_state_dict(artifact, strict=True)
        cuda_logits = cuda_model(input_ids.cuda()).cpu()
    host_checks = {
        "strict_state_dict_load": True,
        "parameter_count": model.parameter_count() == int(target["deployed_parameters"]),
        "cpu_logits_finite": bool(torch.isfinite(cpu_logits).all()),
        "cuda_logits_finite": bool(torch.isfinite(cuda_logits).all()),
        "cpu_cuda_argmax_equal": bool(torch.equal(cpu_logits.argmax(dim=-1), cuda_logits.argmax(dim=-1))),
    }
    accounting = extraction_result["accounting"]
    accounting_checks = {
        "final_parameters": accounting["final_imported_substrate_parameters"] == int(target["deployed_parameters"]) == exact_scalars,
        "payload_below_ceiling": artifact_path.stat().st_size <= int(extraction_protocol["accounting_expectation"]["payload_file_bytes_maximum"]),
        "teacher_forward_zero": accounting["teacher_forward_tokens"] == 0 and accounting["teacher_inference_seconds"] == 0,
        "logits_activations_zero": accounting["stored_logits"] == 0 and accounting["stored_activations"] == 0,
        "complete_source_blocks_zero": accounting["complete_source_blocks_retained"] == 0,
        "artifact_hash": sha256_file(artifact_path) == extraction_result["tensor_sha256"],
    }
    gates = {
        "all_selections_exact": all(selection_match.values()),
        "all_tensor_values_exact": all(tensor_matches.values()),
        "tensor_key_set_exact": key_set_match,
        "all_accounting_exact": all(accounting_checks.values()),
        "host_conformance": all(host_checks.values()),
    }
    passed = all(gates.values())
    return {
        "format": FORMAT,
        "status": "PASS_VERIFIED" if passed else "FAIL_VERIFICATION",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "selection_matches": selection_match,
        "tensor_matches": tensor_matches,
        "exact_fp16_scalars_recomputed": exact_scalars,
        "tensor_key_set_match": key_set_match,
        "accounting_checks": accounting_checks,
        "host_checks": host_checks,
        "gates": gates,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "teacher_present_at_inference": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "next_gate": "One preregistered structural-initialized acquisition candidate may be trained and autonomously screened." if passed else "Close structural extraction branch.",
        "claim_boundary": "Exact hostile extraction and host-conformance verification only; no English quality, Phase 3, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_STRUCTURAL_VERIFIER_PROTOCOL_V195.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_structural/verification_v196.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise Phase3Error("structural verification output exists")
    result = execute(root, (root / args.protocol).resolve())
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
