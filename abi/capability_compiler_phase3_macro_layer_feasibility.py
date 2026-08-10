"""Fixed-byte accounting gate for a sixteen-stage macro-layer successor."""

from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import sys

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable

FORMAT = "abi-capability-compiler-phase3-macro-layer-feasibility/1"


def active_layer_macs(full: int, bottleneck: int, sparse: int, rank: int) -> int:
    attention = 2 * (2 * full * bottleneck + 4 * bottleneck * bottleneck)
    residual = 2 * full * sparse + full * rank + sparse * rank + full * rank
    return attention + residual


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_FIXED_BYTE_SIXTEEN_STAGE_MACRO_LAYER_FEASIBILITY" or protocol.get("training") != "PROHIBITED" or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("macro-layer feasibility governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"macro-layer binding changed: {name}")
    if output.exists():
        raise Phase3Error("macro-layer feasibility output exists")
    output.mkdir(parents=True)
    config = json.loads((root / protocol["artifact_config"]).read_text(encoding="utf-8"))["model"]
    sys.path.insert(0, str((root / protocol["layercake_host"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveCore
    current = protocol["current"]
    target = protocol["target"]
    target_parameters = RoutedSparseRank768ProgressiveCore.parameter_count_for_config(
        fixed_vocab_size=int(config["fixed_vocab_size"]), full_width=int(config["full_width"]),
        bottleneck_width=int(target["bottleneck_width"]), replacement_layers=int(target["replacement_layers"]),
        intermediate_size=int(config["intermediate_size"]), residual_rank=int(target["residual_rank"]),
        sparse_width=int(target["sparse_width"]), routes=3,
    )
    current_active = int(current["replacement_layers"]) * active_layer_macs(int(config["full_width"]), int(current["bottleneck_width"]), int(current["sparse_width"]), int(current["residual_rank"]))
    target_active = int(target["replacement_layers"]) * active_layer_macs(int(config["full_width"]), int(target["bottleneck_width"]), int(target["sparse_width"]), int(target["residual_rank"]))
    pairs = [[2 * index, 2 * index + 1] for index in range(int(target["replacement_layers"]))]
    gates = {
        "exact_sixteen_pairs": pairs == protocol["expected_source_layer_pairs"],
        "parameters_below_current": target_parameters < int(current["parameters"]),
        "fp16_bytes_below_ceiling": 2 * target_parameters <= int(protocol["maximum_model_bytes"]),
        "active_macs_not_increased": target_active <= current_active,
        "kv_width_layer_product_not_increased": int(target["replacement_layers"]) * int(target["bottleneck_width"]) <= int(current["replacement_layers"]) * int(current["bottleneck_width"]),
    }
    result = {
        "format": FORMAT, "status": "PASS_MACRO_LAYER_FEASIBILITY" if all(gates.values()) else "FAIL_MACRO_LAYER_FEASIBILITY",
        "protocol_sha256": sha256_file(protocol_path), "source_layer_pairs": pairs,
        "target_parameters": target_parameters, "target_fp16_bytes": 2 * target_parameters,
        "byte_margin": int(protocol["maximum_model_bytes"]) - 2 * target_parameters,
        "current_active_replacement_macs_per_token": current_active,
        "target_active_replacement_macs_per_token": target_active,
        "active_mac_ratio_target_over_current": target_active / current_active,
        "current_kv_width_layer_product": int(current["replacement_layers"]) * int(current["bottleneck_width"]),
        "target_kv_width_layer_product": int(target["replacement_layers"]) * int(target["bottleneck_width"]),
        "source_blocks_in_target": 0, "training_performed": False, "artifact_written": False,
        "gates": gates, "passed": all(gates.values()), "final_test_accessed": False, "phase3_certified": False,
        "claim_boundary": "Static fixed-byte macro-layer feasibility only; no fit, artifact, autonomous quality, measured runtime, Phase 3, or superiority claim."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_MACRO_LAYER_FEASIBILITY_PROTOCOL_V349.json");parser.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_macro_layer/feasibility_v350");args=parser.parse_args();root=Path.cwd().resolve();print(json.dumps(execute(root,(root/args.protocol).resolve(),(root/args.output_dir).resolve()),indent=2,sort_keys=True));return 0

if __name__ == "__main__": raise SystemExit(main())
