"""No-model envelope for dual attention plus linear/nonlinear rank-768 residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-dual-attention-nonlinear-feasibility/1"


def accounting(
    *,
    full: int = 3072,
    attention_width: int = 192,
    residual_rank: int = 768,
    nonlinear_hidden: int = 384,
    layers: int = 32,
) -> dict[str, int | float]:
    copied = 196_899_840
    attention_per_layer = (
        2 * full * attention_width
        + 4 * attention_width * attention_width
        + attention_width
    )
    nonlinear_residual_per_layer = (
        2 * full * nonlinear_hidden
        + nonlinear_hidden * residual_rank
        + full * residual_rank
        + full
    )
    fixed_linear_per_layer = full * residual_rank
    deployed = copied + layers * (
        2 * attention_per_layer
        + nonlinear_residual_per_layer
        + fixed_linear_per_layer
    )
    active_macs = 307_540_992 + layers * (
        2 * full * attention_width
        + 4 * attention_width * attention_width
        + fixed_linear_per_layer
    )
    return {
        "copied_parameters": copied,
        "primary_attention_parameters": layers * attention_per_layer,
        "residual_attention_parameters": layers * attention_per_layer,
        "nonlinear_rank768_residual_parameters": layers * nonlinear_residual_per_layer,
        "fixed_linear_rank768_parameters": layers * fixed_linear_per_layer,
        "deployed_parameters": deployed,
        "fp16_payload_bytes": 2 * deployed,
        "active_incremental_macs_at_maximum_context": active_macs,
        "source_to_target_active_mac_ratio": 3_823_042_560 / active_macs,
    }


def execute(root: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_MODEL_DUAL_ATTENTION_NONLINEAR_FEASIBILITY"
        or any(
            protocol.get(name) is not False
            for name in (
                "teacher_model_loading_authorized",
                "tensor_value_access_authorized",
                "training_authorized",
                "final_test_access_authorized",
            )
        )
    ):
        raise Phase3Error("dual-attention nonlinear feasibility governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"dual-attention nonlinear feasibility binding changed: {name}")
    values = accounting(**protocol["architecture"])
    gates = {
        "payload_below_one_gib": values["fp16_payload_bytes"] < 1024**3,
        "theoretical_active_mac_margin_at_least_four": values[
            "source_to_target_active_mac_ratio"
        ]
        >= 4.0,
        "zero_source_blocks": protocol["source_blocks"] == 0,
        "rank_matches_measured_minimum": protocol["architecture"]["residual_rank"] == 768,
    }
    return {
        "format": FORMAT,
        "status": "PASS_FEASIBLE_LOCAL_FIT_MAY_BE_DESIGNED" if all(gates.values()) else "FAIL_FEASIBILITY",
        "protocol_sha256": sha256_file(protocol_path),
        "accounting": values,
        "gates": gates,
        "source_model_loaded": False,
        "tensor_values_read": False,
        "training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "No-model final-system envelope only; no host, artifact, quality, physical runtime, certificate, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_DUAL_ATTENTION_NONLINEAR_FEASIBILITY_PROTOCOL_V289.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_dual_attention_nonlinear/feasibility_v290.json",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error("output exists")
    result = execute(root, root / args.protocol)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
