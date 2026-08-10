"""Dtype-conformant replay of the fixed factorized layer-1 analytic realization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import capability_compiler_phase3_factorized_layer1_analytic_realization as original
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-factorized-layer1-dtype-replay/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    correction = protocol.get("dtype_correction", {})
    if (
        protocol.get("format") != original.FORMAT
        or protocol.get("status") != "PREREGISTERED_FACTORIZED_LAYER1_ANALYTIC_REALIZATION"
        or correction.get("tensor") != "sparse_features"
        or correction.get("from") != "bfloat16"
        or correction.get("to") != "float32"
        or correction.get("only_change_from_v420") != "ROUTE_CORRECTION_INPUT_DTYPE_CONFORMANCE"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("factorized layer1 dtype-replay governance changed")
    if output.exists():
        raise Phase3Error("factorized layer1 dtype-replay output exists")

    prior_cat = original.torch.cat

    def conforming_cat(values, *args, **kwargs):
        result = prior_cat(values, *args, **kwargs)
        if result.dtype == torch.bfloat16 and result.ndim >= 1 and result.shape[-1] == 384:
            return result.float()
        return result

    original.torch.cat = conforming_cat
    try:
        source_result = original.execute(root, protocol_path, output / "dtype_replay")
    finally:
        original.torch.cat = prior_cat

    source_path = output / "dtype_replay" / "result.json"
    result = dict(source_result)
    result.update(
        {
            "format": FORMAT,
            "status": "PASS_FACTORIZED_LAYER1_DTYPE_REPLAY"
            if source_result["passed"]
            else "FAIL_FACTORIZED_LAYER1_DTYPE_REPLAY",
            "dtype_correction": correction,
            "source_replay": "dtype_replay/result.json",
            "source_replay_sha256": sha256_file(source_path),
            "source_evidence_sha256": source_result["evidence_sha256"],
            "component_path": "dtype_replay/layer1_component.safetensors"
            if source_result["component_written"]
            else None,
            "phase3_certified": False,
            "claim_boundary": "Dtype-conformant replay of the single fixed layer-1 analytic realization; any passing component remains uninstalled and has no physical runtime, autonomous, complete-model, Phase 3, or superiority claim.",
        }
    )
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FACTORIZED_LAYER1_DTYPE_REPLAY_PROTOCOL_V422.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/factorized_layer1_dtype_replay_v423")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
