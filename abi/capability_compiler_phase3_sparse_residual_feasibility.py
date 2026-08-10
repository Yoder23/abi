"""No-model accounting for a top-1 sparse residual-expert host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-sparse-residual-feasibility/1"


def accounting(*, full: int = 3072, rank: int = 192, experts: int = 4, layers: int = 32) -> dict:
    copied = 196_899_840
    attention_per_layer = 1_327_296
    router_per_layer = full * experts
    stored_expert_per_layer = full * rank + rank * full + full
    stored_sparse_mlp_per_layer = router_per_layer + experts * stored_expert_per_layer
    active_sparse_mlp_per_layer = router_per_layer + 2 * full * rank
    attention = layers * attention_per_layer
    sparse_mlp = layers * stored_sparse_mlp_per_layer
    deployed = copied + attention + sparse_mlp
    direct_linear_active_macs = 184_857_600
    direct_linear_mlp_per_layer = 2 * full * rank
    active_macs = direct_linear_active_macs + layers * (
        active_sparse_mlp_per_layer - direct_linear_mlp_per_layer
    )
    return {
        "copied_parameters": copied,
        "attention_parameters": attention,
        "stored_sparse_mlp_parameters": sparse_mlp,
        "deployed_parameters": deployed,
        "fp16_payload_bytes": 2 * deployed,
        "experts": experts,
        "rank_per_expert": rank,
        "union_rank": experts * rank,
        "active_experts_per_token": 1,
        "active_sparse_mlp_macs_per_layer": active_sparse_mlp_per_layer,
        "active_incremental_macs_at_maximum_context": active_macs,
        "active_mac_increase_over_direct_linear": active_macs / direct_linear_active_macs - 1.0,
        "source_to_target_active_mac_ratio": 3_823_042_560 / active_macs,
    }


def execute(root: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_MODEL_SPARSE_RESIDUAL_FEASIBILITY"
        or any(protocol.get(name) is not False for name in (
            "teacher_model_loading_authorized", "tensor_value_access_authorized",
            "training_authorized", "final_test_access_authorized",
        ))
    ):
        raise Phase3Error("sparse-residual feasibility governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"sparse-residual feasibility binding changed: {name}")
    values = accounting(**protocol["architecture"])
    gates = {
        "union_rank_matches_measured_minimum": values["union_rank"] >= 768,
        "top1_physical_design": values["active_experts_per_token"] == 1,
        "payload_below_800_mib": values["fp16_payload_bytes"] <= 800 * 1024 * 1024,
        "active_mac_overhead_below_two_percent": values["active_mac_increase_over_direct_linear"] <= 0.02,
        "source_active_mac_margin_at_least_four": values["source_to_target_active_mac_ratio"] >= 4.0,
        "zero_source_blocks": protocol["source_blocks"] == 0,
    }
    return {
        "format": FORMAT,
        "status": "PASS_FEASIBLE_GENERIC_HOST_MAY_BE_CONSTRUCTED" if all(gates.values()) else "FAIL_FEASIBILITY",
        "protocol_sha256": sha256_file(protocol_path),
        "accounting": values,
        "gates": gates,
        "source_model_loaded": False,
        "tensor_values_read": False,
        "training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "No-model sparse-host accounting only; no artifact, quality, physical runtime, Phase 3 certificate, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_SPARSE_RESIDUAL_FEASIBILITY_PROTOCOL_V261.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_sparse_residual/feasibility_v262.json")
    args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists():
        raise Phase3Error("sparse-residual feasibility output exists")
    result = execute(root, root / args.protocol)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
