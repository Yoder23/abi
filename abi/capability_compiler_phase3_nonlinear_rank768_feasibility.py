"""No-model accounting for a dense nonlinear rank-768 residual host."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable

FORMAT = "abi-capability-compiler-phase3-nonlinear-rank768-feasibility/1"


def accounting(*, full: int = 3072, attention_width: int = 192, residual_rank: int = 768, nonlinear_hidden: int = 384, layers: int = 32) -> dict:
    copied = 196_899_840
    attention_per_layer = 2 * full * attention_width + 4 * attention_width * attention_width + attention_width
    residual_per_layer = 2 * full * nonlinear_hidden + nonlinear_hidden * residual_rank + full * residual_rank + full
    deployed = copied + layers * (attention_per_layer + residual_per_layer)
    active_macs = 184_857_600 + layers * ((2 * full * nonlinear_hidden + nonlinear_hidden * residual_rank + full * residual_rank) - 2 * full * attention_width)
    return {
        "copied_parameters": copied, "attention_parameters": layers * attention_per_layer,
        "nonlinear_residual_parameters": layers * residual_per_layer,
        "deployed_parameters": deployed, "fp16_payload_bytes": 2 * deployed,
        "residual_rank": residual_rank, "nonlinear_hidden": nonlinear_hidden,
        "active_incremental_macs_at_maximum_context": active_macs,
        "source_to_target_active_mac_ratio": 3_823_042_560 / active_macs,
    }


def execute(root: Path, path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_NO_MODEL_NONLINEAR_RANK768_FEASIBILITY" or any(protocol.get(name) is not False for name in ("teacher_model_loading_authorized", "tensor_value_access_authorized", "training_authorized", "final_test_access_authorized")):
        raise Phase3Error("nonlinear rank-768 feasibility governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"nonlinear feasibility binding changed: {name}")
    values = accounting(**protocol["architecture"])
    gates = {
        "rank_matches_measured_minimum": values["residual_rank"] >= 768,
        "payload_below_800_mib": values["fp16_payload_bytes"] <= 800 * 1024 * 1024,
        "source_active_mac_margin_at_least_four": values["source_to_target_active_mac_ratio"] >= 4,
        "zero_source_blocks": protocol["source_blocks"] == 0,
    }
    return {"format": FORMAT, "status": "PASS_FEASIBLE_GENERIC_HOST_MAY_BE_CONSTRUCTED" if all(gates.values()) else "FAIL_FEASIBILITY", "protocol_sha256": sha256_file(path), "accounting": values, "gates": gates, "source_model_loaded": False, "tensor_values_read": False, "training_performed": False, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "No-model nonlinear host accounting only; no artifact, quality, runtime, certificate, or superiority claim."}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_NONLINEAR_RANK768_FEASIBILITY_PROTOCOL_V267.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_nonlinear_rank768/feasibility_v268.json")
    args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists(): raise Phase3Error("nonlinear feasibility output exists")
    result = execute(root, root / args.protocol); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__": raise SystemExit(main())
