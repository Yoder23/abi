"""No-model feasibility for rank-192 basis-aligned MLP residual cakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-basis-aligned-feasibility/1"


def accounting(
    *,
    copied_substrate: int,
    current_dual_trainable: int,
    full_width: int,
    rank: int,
    layers: int,
    current_target_macs: int,
    source_macs: int,
) -> dict[str, int | float]:
    basis_per_layer = full_width * rank
    mean_per_layer = full_width
    imported_basis_and_mean = layers * (basis_per_layer + mean_per_layer)
    frozen_output_projection_removed_from_trainable = layers * basis_per_layer
    trainable = current_dual_trainable - frozen_output_projection_removed_from_trainable
    deployed = copied_substrate + imported_basis_and_mean + trainable
    return {
        "copied_lexical_and_norm_parameters": copied_substrate,
        "imported_mlp_basis_parameters_per_layer": basis_per_layer,
        "imported_mlp_mean_parameters_per_layer": mean_per_layer,
        "imported_mlp_basis_and_mean_parameters": imported_basis_and_mean,
        "trainable_coefficient_and_attention_parameters": trainable,
        "deployed_parameters": deployed,
        "fp16_payload_bytes": deployed * 2,
        "active_incremental_macs_at_maximum_context": current_target_macs,
        "source_incremental_macs_at_maximum_context": source_macs,
        "source_to_target_incremental_mac_ratio": source_macs / current_target_macs,
    }


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_MODEL_FEASIBILITY"
    ):
        raise Phase3Error("basis-aligned feasibility governance changed")
    for field in (
        "teacher_model_loading_authorized",
        "tensor_value_access_authorized",
        "training_authorized",
        "final_test_access_authorized",
    ):
        if protocol.get(field) is not False:
            raise Phase3Error(f"governance changed: {field}")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"basis-aligned feasibility binding changed: {name}")
    target = protocol["target"]
    values = accounting(
        copied_substrate=int(target["copied_substrate_parameters"]),
        current_dual_trainable=int(target["current_dual_trainable_parameters"]),
        full_width=int(target["full_width"]),
        rank=int(target["mlp_output_rank"]),
        layers=int(target["replacement_layers"]),
        current_target_macs=int(target["current_target_incremental_macs_at_maximum_context"]),
        source_macs=int(target["source_incremental_macs_at_maximum_context"]),
    )
    gates = {
        "rank_unchanged_at_192": int(target["mlp_output_rank"]) == 192,
        "imported_basis_and_mean_exact": values["imported_mlp_basis_and_mean_parameters"] == 18_972_672,
        "trainable_count_reduced": values["trainable_coefficient_and_attention_parameters"] == 75_509_760,
        "deployed_count_exact": values["deployed_parameters"] == 291_382_272,
        "payload_within_600_mib": values["fp16_payload_bytes"] <= 629_145_600,
        "active_compute_not_increased": values["active_incremental_macs_at_maximum_context"] == 199_013_376,
        "theoretical_compute_margin_at_least_four": values["source_to_target_incremental_mac_ratio"] >= 4.0,
        "zero_complete_source_blocks": int(target["complete_source_blocks_retained"]) == 0,
    }
    passed = all(gates.values())
    return {
        "format": FORMAT,
        "status": "PASS_FEASIBLE" if passed else "FAIL_FEASIBILITY",
        "source_model_loaded": False,
        "tensor_values_read": False,
        "training_performed": False,
        "final_test_accessed": False,
        "accounting": values,
        "gates": gates,
        "phase3_certified": False,
        "next_gate": "Construct-certify a generic rank-192 basis-aligned LayerCake host."
        if passed
        else "Close basis-aligned replacement.",
        "claim_boundary": "No-model parameter, payload, and theoretical-compute feasibility only; no extraction, quality, measured runtime, transfer, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_BASIS_ALIGNED_FEASIBILITY_PROTOCOL_V241.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_basis_aligned/feasibility_v242.json",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error("basis-aligned feasibility output exists")
    result = execute(root, root / args.protocol)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_FEASIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
