"""No-model envelope for generic, abstention, and conversation correction routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-three-route-feasibility/1"


def accounting(
    *, full: int = 3072, sparse_width: int = 384, rank: int = 768, routes: int = 3, layers: int = 32
) -> dict[str, int | float]:
    base_single_route = 517_874_688
    extra_expert_outputs = layers * (routes - 1) * sparse_width * rank
    router_parameters = full * routes + routes
    deployed = base_single_route + extra_expert_outputs + router_parameters
    active_macs = 425_505_792 + full * routes
    return {
        "base_single_route_parameters": base_single_route,
        "extra_expert_output_parameters": extra_expert_outputs,
        "router_parameters": router_parameters,
        "deployed_parameters": deployed,
        "fp16_payload_bytes": 2 * deployed,
        "active_incremental_macs_at_maximum_context": active_macs,
        "source_to_target_active_mac_ratio": 3_823_042_560 / active_macs,
        "active_routes_per_request": 1,
    }


def execute(root: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_MODEL_THREE_ROUTE_FEASIBILITY"
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
        raise Phase3Error("three-route feasibility governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"three-route feasibility binding changed: {name}")
    values = accounting(**protocol["architecture"])
    gates = {
        "payload_below_one_gib": values["fp16_payload_bytes"] < 1024**3,
        "active_mac_margin_at_least_four": values["source_to_target_active_mac_ratio"] >= 4,
        "hard_top1_only": values["active_routes_per_request"] == 1,
        "zero_source_blocks": protocol["source_blocks"] == 0,
    }
    return {
        "format": FORMAT,
        "status": "PASS_FEASIBLE_LABEL_ORACLE_MAY_BE_AUDITED" if all(gates.values()) else "FAIL_FEASIBILITY",
        "protocol_sha256": sha256_file(protocol_path),
        "accounting": values,
        "gates": gates,
        "source_model_loaded": False,
        "training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "No-model three-route envelope only; no router, artifact, quality, physical runtime, certificate, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_THREE_ROUTE_FEASIBILITY_PROTOCOL_V297.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_three_route/feasibility_v298.json")
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
