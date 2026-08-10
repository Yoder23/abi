"""Correct progressive-replacement feasibility to the deployable tokenizer vocabulary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error
from . import capability_compiler_phase3_progressive_replacement_feasibility as base


FORMAT = "abi-capability-compiler-phase3-progressive-replacement-feasibility-repair/1"


def corrected_accounting(base_accounting: dict[str, int | float], *, runtime_vocabulary: int, source_vocabulary: int, full_width: int) -> dict[str, int | float]:
    if runtime_vocabulary >= source_vocabulary:
        raise Phase3Error("runtime vocabulary correction must remove unused source rows")
    corrected = base.replacement_parameter_accounting(
        vocabulary=runtime_vocabulary,
        full_width=full_width,
        bottleneck_width=192,
        intermediate_size=768,
        layers=32,
        maximum_context=512,
    )
    source_macs = int(base_accounting["source_incremental_macs_at_maximum_context"])
    target_macs = int(corrected["target_incremental_macs_at_maximum_context"])
    corrected["source_incremental_macs_at_maximum_context"] = source_macs
    corrected["target_to_source_incremental_mac_ratio"] = target_macs / source_macs
    corrected["source_to_target_incremental_mac_ratio"] = source_macs / target_macs
    corrected["source_weight_vocabulary"] = source_vocabulary
    corrected["runtime_vocabulary"] = runtime_vocabulary
    corrected["omitted_unused_source_rows"] = source_vocabulary - runtime_vocabulary
    return corrected


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_NO_MODEL_ACCOUNTING_REPAIR":
        raise Phase3Error("progressive feasibility repair is not preregistered")
    for field in ("teacher_model_loading_authorized", "tensor_value_access_authorized", "training_authorized", "final_test_access_authorized"):
        if protocol.get(field) is not False:
            raise Phase3Error(f"governance changed: {field}")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"progressive feasibility repair binding changed: {name}")
    base_result = base.run(root, root / protocol["base_protocol"])
    if base_result["status"] != "PASS_FEASIBLE":
        raise Phase3Error("base metadata feasibility no longer passes")
    target = protocol["corrected_target"]
    accounting = corrected_accounting(
        base_result["accounting"],
        runtime_vocabulary=int(target["runtime_vocabulary"]),
        source_vocabulary=int(target["source_weight_vocabulary"]),
        full_width=int(target["full_width"]),
    )
    gates = {
        "base_source_metadata_feasibility": True,
        "runtime_vocabulary_equals_external_plus_specials": int(target["runtime_vocabulary"]) == int(target["external_actions"]) + int(target["host_special_actions"]),
        "unused_source_rows_omitted": int(accounting["omitted_unused_source_rows"]) == 49,
        "corrected_deployed_parameter_count": int(accounting["deployed_parameters"]) == 253_535_232,
        "corrected_payload_within_bound": int(accounting["fp16_payload_bytes"]) <= int(target["fp16_payload_bytes_maximum"]),
        "theoretical_compute_margin_at_least_four": float(accounting["source_to_target_incremental_mac_ratio"]) >= 4.0,
        "zero_complete_source_blocks_at_deployment": int(target["complete_source_blocks_retained"]) == 0,
    }
    passed = all(gates.values())
    return {
        "format": FORMAT,
        "status": "PASS_FEASIBLE_CORRECTED" if passed else "FAIL_FEASIBILITY",
        "base_result_preserved": protocol["base_result"],
        "source_model_loaded": False,
        "tensor_values_read": False,
        "training_performed": False,
        "final_test_accessed": False,
        "observed_tensor_count": base_result["observed_tensor_count"],
        "accounting": accounting,
        "calibration_cache": base_result["calibration_cache"],
        "gates": gates,
        "complete_source_blocks_retained_at_deployment": 0,
        "teacher_present_at_inference": False,
        "phase3_certified": False,
        "next_gate": "Extract only the 32,015-row mapped lexical substrate and normalization fields, then hostilely verify every copied value before training.",
        "claim_boundary": "Corrected no-model tokenizer, payload, and theoretical-compute feasibility only; no source values, quality, measured runtime, transfer, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_PROGRESSIVE_REPLACEMENT_FEASIBILITY_REPAIR_V217.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_progressive_replacement/feasibility_repair_v218.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error("progressive feasibility repair output exists")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_FEASIBLE_CORRECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
