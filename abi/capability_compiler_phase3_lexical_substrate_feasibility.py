"""No-model feasibility/accounting for projected teacher lexical substrate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error


def projection_hash(source_width: int, target_width: int, seed: int) -> str:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(source_width, target_width, generator=generator, dtype=torch.float32)
    projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return hashlib.sha256(projection.contiguous().numpy().tobytes()).hexdigest()


def projected_accounting(actions: int, source_width: int, target_width: int, deployed_parameters: int) -> dict[str, Any]:
    imported_parameters = actions * target_width * 2
    return {"source_table_scalars": actions * source_width * 2, "source_table_bytes_bf16": actions * source_width * 2 * 2, "projected_table_scalars": imported_parameters, "projected_payload_bytes_fp16": imported_parameters * 2, "final_imported_substrate_parameters": imported_parameters, "bridge_and_special_parameters": deployed_parameters - imported_parameters, "source_to_projected_payload_ratio": source_width / target_width}


def _shape(path: Path, key: str) -> tuple[list[int], str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise Phase3Error(f"source lexical tensor missing: {key}")
        tensor = handle.get_slice(key)
        return list(tensor.get_shape()), str(tensor.get_dtype())


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_NO_MODEL_FEASIBILITY" or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("tensor_extraction_authorized") is not False:
        raise Phase3Error("lexical feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"lexical feasibility binding changed: {relative}")
    index = json.loads(Path(protocol["source"]["index_path"]).read_text(encoding="utf-8"))
    snapshot = Path(protocol["source"]["snapshot_path"])
    observed = {}
    for key in protocol["source"]["tensor_keys"]:
        relative = index["weight_map"].get(key)
        if relative is None:
            raise Phase3Error(f"source index lacks {key}")
        path = (snapshot / relative).resolve()
        shape, dtype = _shape(path, key)
        observed[key] = {"file": relative, "shape": shape, "dtype": dtype, "file_bytes": path.stat().st_size}
    expected_shape = [int(protocol["source"]["vocabulary"]), int(protocol["source"]["width"])]
    shapes_match = all(value["shape"] == expected_shape for value in observed.values())
    projection = projection_hash(int(protocol["source"]["width"]), int(protocol["projection"]["target_width"]), int(protocol["projection"]["seed"]))
    accounting = projected_accounting(int(protocol["selection"]["external_actions"]), int(protocol["source"]["width"]), int(protocol["projection"]["target_width"]), int(protocol["selection"]["deployed_parameters"]))
    passes = shapes_match and projection == protocol["projection"]["sha256"] and accounting["projected_payload_bytes_fp16"] <= int(protocol["selection"]["payload_bytes_maximum"]) and accounting["bridge_and_special_parameters"] <= int(protocol["selection"]["bridge_parameters_maximum"])
    return {"format": "abi-capability-compiler-phase3-lexical-substrate-feasibility/1", "status": "PASS_FEASIBLE" if passes else "FAIL_FEASIBILITY", "source_tensors": observed, "source_model_loaded": False, "tensor_values_read": False, "projection": {**protocol["projection"], "observed_sha256": projection}, "accounting": accounting, "host_action_rows": {"external_actions": protocol["selection"]["external_actions"], "host_offset": 4, "source_rows_excluded": int(protocol["source"]["vocabulary"]) - int(protocol["selection"]["external_actions"])}, "source_blocks_retained": 0, "teacher_present_at_inference": False, "phase3_certified": False, "final_test_accessed": False, "next_gate": "Preregister one chunked GPU extraction with exact row, projection, dtype, payload, and source accounting." if passes else "Close projected lexical substrate branch."}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_LEXICAL_SUBSTRATE_FEASIBILITY_PROTOCOL_V80.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_lexical_substrate/feasibility_v80.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise Phase3Error("lexical feasibility output exists")
    result = run(root, (root / args.protocol).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
