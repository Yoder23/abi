"""No-model feasibility gate for head-aligned structural weight extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error


def target_parameter_accounting(
    *,
    external_actions: int,
    host_special_actions: int,
    width: int,
    intermediate_size: int,
    layers: int,
) -> dict[str, int]:
    vocabulary = external_actions + host_special_actions
    lexical = 2 * vocabulary * width
    attention_per_layer = 4 * width * width
    mlp_per_layer = 3 * width * intermediate_size
    norms_per_layer = 2 * width
    body = layers * (attention_per_layer + mlp_per_layer + norms_per_layer)
    final_norm = width
    total = lexical + body + final_norm
    host_initialized = 2 * host_special_actions * width
    return {
        "vocabulary": vocabulary,
        "lexical_parameters": lexical,
        "attention_parameters_per_layer": attention_per_layer,
        "mlp_parameters_per_layer": mlp_per_layer,
        "norm_parameters_per_layer": norms_per_layer,
        "body_parameters": body,
        "final_norm_parameters": final_norm,
        "deployed_parameters": total,
        "source_derived_parameters": total - host_initialized,
        "host_initialized_special_parameters": host_initialized,
        "fp16_payload_bytes": total * 2,
    }


def _indexed_shape(snapshot: Path, weight_map: dict[str, str], key: str) -> dict[str, Any]:
    relative = weight_map.get(key)
    if relative is None:
        raise Phase3Error(f"source index lacks required tensor: {key}")
    path = snapshot / relative
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor = handle.get_slice(key)
        return {"shard": relative, "shape": list(tensor.get_shape()), "dtype": str(tensor.get_dtype())}


def required_shapes(layer: int, source: dict[str, Any]) -> dict[str, list[int]]:
    width = int(source["hidden_size"])
    intermediate = int(source["intermediate_size"])
    return {
        f"model.layers.{layer}.self_attn.qkv_proj.weight": [3 * width, width],
        f"model.layers.{layer}.self_attn.o_proj.weight": [width, width],
        f"model.layers.{layer}.mlp.gate_up_proj.weight": [2 * intermediate, width],
        f"model.layers.{layer}.mlp.down_proj.weight": [width, intermediate],
        f"model.layers.{layer}.input_layernorm.weight": [width],
        f"model.layers.{layer}.post_attention_layernorm.weight": [width],
    }


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_NO_MODEL_FEASIBILITY":
        raise Phase3Error("structural feasibility is not preregistered")
    for field in ("teacher_model_loading_authorized", "tensor_value_access_authorized", "training_authorized"):
        if protocol.get(field) is not False:
            raise Phase3Error(f"governance changed: {field}")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"structural feasibility binding changed: {name}")

    source = protocol["source"]
    target = protocol["target"]
    snapshot = Path(source["snapshot_path"])
    index = json.loads(Path(source["index_path"]).read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    observed: dict[str, Any] = {}
    expected: dict[str, list[int]] = {
        "model.embed_tokens.weight": [int(source["vocab_size"]), int(source["hidden_size"])],
        "lm_head.weight": [int(source["vocab_size"]), int(source["hidden_size"])],
        "model.norm.weight": [int(source["hidden_size"])],
    }
    for layer in target["source_layers"]:
        expected.update(required_shapes(int(layer), source))
    for key in expected:
        observed[key] = _indexed_shape(snapshot, weight_map, key)

    shapes_match = all(observed[key]["shape"] == shape for key, shape in expected.items())
    source_head_dim = int(source["hidden_size"]) // int(source["num_attention_heads"])
    geometry = {
        "source_head_dim": source_head_dim,
        "target_head_dim": int(target["width"]) // int(target["num_attention_heads"]),
        "complete_source_heads_selected": int(target["num_attention_heads"]),
        "target_width_equals_complete_heads": int(target["width"]) == int(target["num_attention_heads"]) * source_head_dim,
        "target_intermediate_within_source": int(target["intermediate_size"]) <= int(source["intermediate_size"]),
        "source_layers_strictly_increasing": list(target["source_layers"]) == sorted(set(target["source_layers"])),
        "source_layers_in_range": all(0 <= int(i) < int(source["num_hidden_layers"]) for i in target["source_layers"]),
    }
    accounting = target_parameter_accounting(
        external_actions=int(target["external_actions"]),
        host_special_actions=int(target["host_special_actions"]),
        width=int(target["width"]),
        intermediate_size=int(target["intermediate_size"]),
        layers=len(target["source_layers"]),
    )
    gates = {
        "source_tensor_shapes_match": shapes_match,
        "source_is_full_mha": int(source["num_attention_heads"]) == int(source["num_key_value_heads"]),
        "head_geometry_exact": geometry["target_width_equals_complete_heads"],
        "intermediate_selection_feasible": geometry["target_intermediate_within_source"],
        "layer_selection_valid": geometry["source_layers_strictly_increasing"] and geometry["source_layers_in_range"],
        "parameter_ceiling": accounting["deployed_parameters"] <= int(target["deployed_parameter_maximum"]),
        "payload_ceiling": accounting["fp16_payload_bytes"] <= int(target["payload_bytes_maximum"]),
        "complete_source_blocks_retained_zero": bool(target["complete_source_blocks_retained_required"] == 0),
    }
    passed = all(gates.values())
    return {
        "format": "abi-capability-compiler-phase3-structural-feasibility/1",
        "status": "PASS_FEASIBLE" if passed else "FAIL_FEASIBILITY",
        "source_model_loaded": False,
        "tensor_values_read": False,
        "training_performed": False,
        "observed_tensors": observed,
        "geometry": geometry,
        "accounting": accounting,
        "gates": gates,
        "complete_source_blocks_retained": 0,
        "teacher_present_at_inference": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "next_gate": (
            "Construct-certify the generic compact Phi-compatible LayerCake host before weight extraction."
            if passed
            else "Close the head-aligned structural extraction branch."
        ),
        "claim_boundary": "Tensor-shape, operator-geometry, and parameter feasibility only; no tensor values, quality, or transfer claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_STRUCTURAL_FEASIBILITY_PROTOCOL_V191.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_structural/feasibility_v192.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise Phase3Error("structural feasibility output exists")
    result = run(root, (root / args.protocol).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_FEASIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
