"""No-model feasibility for progressive source-block replacement.

This gate reads only the frozen source index and tensor metadata.  It does not
load the teacher, inspect tensor values, train a candidate, or touch final-test
material.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error


FORMAT = "abi-capability-compiler-phase3-progressive-replacement-feasibility/1"


def replacement_parameter_accounting(
    *,
    vocabulary: int,
    full_width: int,
    bottleneck_width: int,
    intermediate_size: int,
    layers: int,
    maximum_context: int,
) -> dict[str, int | float]:
    """Return exact deployed-size and upper-bound incremental-compute counts."""
    lexical_each = vocabulary * full_width
    copied_lexical = 2 * lexical_each
    copied_source_norms = layers * 2 * full_width
    copied_final_norm = full_width

    projection_per_layer = 2 * full_width * bottleneck_width
    attention_linear_per_layer = 4 * bottleneck_width * bottleneck_width
    mlp_linear_per_layer = 3 * bottleneck_width * intermediate_size
    latent_norm_per_layer = 2 * bottleneck_width
    trained_per_layer = (
        projection_per_layer
        + attention_linear_per_layer
        + mlp_linear_per_layer
        + latent_norm_per_layer
    )
    trained_replacement = layers * trained_per_layer
    copied_source_parameters = copied_lexical + copied_source_norms + copied_final_norm
    deployed = copied_source_parameters + trained_replacement

    # One cached autoregressive token. Embedding is a lookup, while the output
    # table is a dense projection. The attention term is the score and value
    # application at the declared maximum context.
    target_linear_macs = lexical_each + layers * (
        projection_per_layer + attention_linear_per_layer + mlp_linear_per_layer
    )
    target_attention_macs = 2 * layers * bottleneck_width * maximum_context
    target_macs = target_linear_macs + target_attention_macs

    # Phi-3 full-MHA/SwiGLU source upper bound at the same context.
    source_attention_linear_per_layer = 4 * full_width * full_width
    source_mlp_linear_per_layer = 3 * full_width * 8192
    source_linear_macs = lexical_each + layers * (
        source_attention_linear_per_layer + source_mlp_linear_per_layer
    )
    source_attention_macs = 2 * layers * full_width * maximum_context
    source_macs = source_linear_macs + source_attention_macs

    kv_cache_bytes = layers * 2 * bottleneck_width * maximum_context * 2
    return {
        "input_embedding_parameters": lexical_each,
        "output_head_parameters": lexical_each,
        "copied_source_norm_parameters": copied_source_norms,
        "copied_final_norm_parameters": copied_final_norm,
        "projection_parameters_per_layer": projection_per_layer,
        "attention_parameters_per_layer": attention_linear_per_layer,
        "mlp_parameters_per_layer": mlp_linear_per_layer,
        "latent_norm_parameters_per_layer": latent_norm_per_layer,
        "trained_parameters_per_layer": trained_per_layer,
        "trained_replacement_parameters": trained_replacement,
        "copied_source_parameters": copied_source_parameters,
        "deployed_parameters": deployed,
        "fp16_payload_bytes": deployed * 2,
        "maximum_context_kv_cache_bytes_fp16": kv_cache_bytes,
        "target_incremental_macs_at_maximum_context": target_macs,
        "source_incremental_macs_at_maximum_context": source_macs,
        "target_to_source_incremental_mac_ratio": target_macs / source_macs,
        "source_to_target_incremental_mac_ratio": source_macs / target_macs,
    }


def calibration_cache_accounting(*, tokens: int, full_width: int) -> dict[str, int]:
    one_hidden = tokens * full_width * 2
    return {
        "calibration_tokens": tokens,
        "one_fp16_hidden_field_bytes": one_hidden,
        "input_and_target_fp16_hidden_bytes": 2 * one_hidden,
    }


def _indexed_shape(snapshot: Path, weight_map: Mapping[str, str], key: str) -> dict[str, Any]:
    relative = weight_map.get(key)
    if relative is None:
        raise Phase3Error(f"source index lacks required tensor: {key}")
    with safe_open(str(snapshot / relative), framework="pt", device="cpu") as handle:
        value = handle.get_slice(key)
        return {
            "shard": relative,
            "shape": list(value.get_shape()),
            "dtype": str(value.get_dtype()),
        }


def _required_shapes(source: Mapping[str, Any]) -> dict[str, list[int]]:
    width = int(source["hidden_size"])
    intermediate = int(source["intermediate_size"])
    layers = int(source["num_hidden_layers"])
    expected: dict[str, list[int]] = {
        "model.embed_tokens.weight": [int(source["vocab_size"]), width],
        "lm_head.weight": [int(source["vocab_size"]), width],
        "model.norm.weight": [width],
    }
    for layer in range(layers):
        expected.update(
            {
                f"model.layers.{layer}.self_attn.qkv_proj.weight": [3 * width, width],
                f"model.layers.{layer}.self_attn.o_proj.weight": [width, width],
                f"model.layers.{layer}.mlp.gate_up_proj.weight": [2 * intermediate, width],
                f"model.layers.{layer}.mlp.down_proj.weight": [width, intermediate],
                f"model.layers.{layer}.input_layernorm.weight": [width],
                f"model.layers.{layer}.post_attention_layernorm.weight": [width],
            }
        )
    return expected


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_NO_MODEL_FEASIBILITY":
        raise Phase3Error("progressive-replacement feasibility is not preregistered")
    for field in (
        "teacher_model_loading_authorized",
        "tensor_value_access_authorized",
        "training_authorized",
        "final_test_access_authorized",
    ):
        if protocol.get(field) is not False:
            raise Phase3Error(f"governance changed: {field}")
    for name, expected_sha in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise Phase3Error(f"progressive-replacement binding changed: {name}")

    source = protocol["source"]
    target = protocol["target"]
    snapshot = Path(source["snapshot_path"])
    index = json.loads(Path(source["index_path"]).read_text(encoding="utf-8"))
    expected_shapes = _required_shapes(source)
    observed = {
        key: _indexed_shape(snapshot, index["weight_map"], key)
        for key in expected_shapes
    }
    shapes_match = all(observed[key]["shape"] == shape for key, shape in expected_shapes.items())

    accounting = replacement_parameter_accounting(
        vocabulary=int(source["vocab_size"]),
        full_width=int(source["hidden_size"]),
        bottleneck_width=int(target["bottleneck_width"]),
        intermediate_size=int(target["intermediate_size"]),
        layers=int(source["num_hidden_layers"]),
        maximum_context=int(target["maximum_context"]),
    )
    cache = calibration_cache_accounting(
        tokens=int(target["calibration_tokens"]),
        full_width=int(source["hidden_size"]),
    )
    gates = {
        "all_source_tensor_shapes_match": shapes_match,
        "one_replacement_per_source_block": int(target["replacement_layers"]) == int(source["num_hidden_layers"]),
        "full_residual_width_preserved": int(target["full_width"]) == int(source["hidden_size"]),
        "bottleneck_head_geometry_exact": int(target["bottleneck_width"]) % int(target["attention_heads"]) == 0,
        "bottleneck_head_dimension_even": (int(target["bottleneck_width"]) // int(target["attention_heads"])) % 2 == 0,
        "zero_complete_source_blocks_at_deployment": int(target["complete_source_blocks_retained"]) == 0,
        "payload_within_bound": int(accounting["fp16_payload_bytes"]) <= int(target["fp16_payload_bytes_maximum"]),
        "calibration_cache_within_bound": cache["input_and_target_fp16_hidden_bytes"] <= int(target["calibration_cache_bytes_maximum"]),
        "theoretical_compute_margin_at_least_four": float(accounting["source_to_target_incremental_mac_ratio"]) >= 4.0,
        "copied_parameters_explicitly_accounted": int(accounting["copied_source_parameters"]) > 0,
    }
    passed = all(gates.values())
    return {
        "format": FORMAT,
        "status": "PASS_FEASIBLE" if passed else "FAIL_FEASIBILITY",
        "source_model_loaded": False,
        "tensor_values_read": False,
        "training_performed": False,
        "final_test_accessed": False,
        "observed_tensor_count": len(observed),
        "observed_tensors": observed,
        "accounting": accounting,
        "calibration_cache": cache,
        "gates": gates,
        "complete_source_blocks_retained_at_deployment": 0,
        "teacher_present_at_inference": False,
        "phase3_certified": False,
        "next_gate": (
            "Construct-certify a generic full-residual-width bottleneck replacement host before any source tensor values are read."
            if passed
            else "Close the progressive source-block replacement branch."
        ),
        "claim_boundary": "Metadata-only architecture, payload, cache, and theoretical-compute feasibility; no tensor values, measured speed, quality, transfer, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_PROGRESSIVE_REPLACEMENT_FEASIBILITY_PROTOCOL_V215.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_progressive_replacement/feasibility_v216.json",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise Phase3Error("progressive-replacement feasibility output exists")
    result = run(root, (root / args.protocol).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_FEASIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
