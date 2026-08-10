"""No-model feasibility for source-topology-preserving dual replacement cakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-dual-path-feasibility/1"


def dual_path_accounting(*, runtime_vocabulary: int, full_width: int, bottleneck_width: int, intermediate_size: int, layers: int, maximum_context: int, source_incremental_macs: int) -> dict[str, int | float]:
    copied = 2 * runtime_vocabulary * full_width + (2 * layers + 1) * full_width
    attention_projection = 2 * full_width * bottleneck_width
    attention_operator = 4 * bottleneck_width * bottleneck_width
    attention_norm = bottleneck_width
    attention_per_layer = attention_projection + attention_operator + attention_norm
    mlp_projection = 2 * full_width * bottleneck_width
    mlp_operator = 3 * bottleneck_width * intermediate_size
    mlp_norm = bottleneck_width
    mlp_per_layer = mlp_projection + mlp_operator + mlp_norm
    trainable = layers * (attention_per_layer + mlp_per_layer)
    deployed = copied + trainable
    target_linear_macs = runtime_vocabulary * full_width + layers * (
        attention_projection + attention_operator + mlp_projection + mlp_operator
    )
    target_attention_macs = 2 * layers * bottleneck_width * maximum_context
    target_macs = target_linear_macs + target_attention_macs
    return {
        "copied_substrate_parameters": copied,
        "attention_parameters_per_layer": attention_per_layer,
        "mlp_parameters_per_layer": mlp_per_layer,
        "trainable_parameters_per_layer": attention_per_layer + mlp_per_layer,
        "trainable_replacement_parameters": trainable,
        "deployed_parameters": deployed,
        "fp16_payload_bytes": deployed * 2,
        "maximum_context_kv_cache_bytes_fp16": layers * 2 * bottleneck_width * maximum_context * 2,
        "target_incremental_macs_at_maximum_context": target_macs,
        "source_incremental_macs_at_maximum_context": source_incremental_macs,
        "source_to_target_incremental_mac_ratio": source_incremental_macs / target_macs,
    }


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_NO_MODEL_FEASIBILITY":
        raise Phase3Error("dual-path feasibility governance changed")
    for field in ("teacher_model_loading_authorized", "tensor_value_access_authorized", "training_authorized", "final_test_access_authorized"):
        if protocol.get(field) is not False: raise Phase3Error(f"governance changed: {field}")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected: raise Phase3Error(f"dual-path binding changed: {name}")
    target = protocol["target"]
    accounting = dual_path_accounting(
        runtime_vocabulary=int(target["runtime_vocabulary"]), full_width=int(target["full_width"]),
        bottleneck_width=int(target["bottleneck_width"]), intermediate_size=int(target["intermediate_size"]),
        layers=int(target["replacement_layers"]), maximum_context=int(target["maximum_context"]),
        source_incremental_macs=int(target["source_incremental_macs_at_maximum_context"]),
    )
    gates = {
        "copied_substrate_count_unchanged": accounting["copied_substrate_parameters"] == 196_899_840,
        "separate_attention_and_mlp_paths": bool(target["separate_attention_and_mlp_paths"]),
        "deployed_parameter_count": accounting["deployed_parameters"] == 291_283_968,
        "payload_within_600_mib": accounting["fp16_payload_bytes"] <= 629_145_600,
        "theoretical_compute_margin_at_least_four": accounting["source_to_target_incremental_mac_ratio"] >= 4.0,
        "zero_complete_source_blocks": int(target["complete_source_blocks_retained"]) == 0,
    }
    passed = all(gates.values())
    return {
        "format": FORMAT, "status": "PASS_FEASIBLE" if passed else "FAIL_FEASIBILITY",
        "source_model_loaded": False, "tensor_values_read": False, "training_performed": False,
        "final_test_accessed": False, "accounting": accounting, "gates": gates,
        "phase3_certified": False,
        "next_gate": "Construct-certify a generic source-aligned dual-path LayerCake host." if passed else "Close dual-path replacement.",
        "claim_boundary": "No-model parameter, payload, and theoretical-compute feasibility only; no quality, measured runtime, transfer, or superiority claim."
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_DUAL_PATH_FEASIBILITY_PROTOCOL_V231.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_dual_path/feasibility_v232.json")
    args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists(): raise Phase3Error("dual-path feasibility output exists")
    result = execute(root, root / args.protocol); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"] == "PASS_FEASIBLE" else 1


if __name__ == "__main__": raise SystemExit(main())
