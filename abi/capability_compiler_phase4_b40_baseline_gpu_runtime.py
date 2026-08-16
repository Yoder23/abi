"""Measure exact-B40 LoRA checkpoints with the proven matched GPU harness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

from . import capability_compiler_phase4_b50_gpu_runtime as harness
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b40_baselines import (
    load_exact_records,
    train_exact_b40_router,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b40-baseline-gpu-runtime/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b40-baseline-gpu-runtime-result/1"
SYSTEMS = ("L0", "L1")


def _staging_path(root: Path, output: Path) -> Path:
    staging = output.parent / f"{output.name}_raw_harness"
    if root.resolve() not in staging.resolve().parents:
        raise Phase3Error("B40 runtime staging path escaped repository root")
    return staging


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    runtime = protocol.get("runtime", {})
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_SAME_CHECKPOINT_B40_LORA_CUDA_RUNTIME"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_generation_authorized") is not False
        or protocol.get("source_base_loading_for_lora_authorized") is not True
        or protocol.get("final_test_access") != "PROHIBITED"
        or int(runtime.get("distinct_prompts", 0)) != 100
        or int(runtime.get("repeated_observations", 0)) < 20
        or int(runtime.get("p95_minimum_observations", 0)) != 100
        or int(runtime.get("p99_minimum_observations", 0)) != 1000
        or set(protocol.get("systems", {})) != set(SYSTEMS)
        or protocol.get("authorized_systems") != ["L1"]
    ):
        raise Phase3Error("matched B40 LoRA CUDA runtime governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"matched B40 LoRA CUDA runtime binding changed: {relative}")
    return protocol, sha256_file(path)


def run(
    root: Path,
    protocol_path: Path,
    *,
    system: str,
    output: Path,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if system not in protocol["authorized_systems"] or output.exists():
        raise Phase3Error("invalid or existing matched B40 LoRA CUDA runtime target")
    staging = _staging_path(root, output)
    if staging.exists():
        raise Phase3Error("existing B40 runtime raw-harness staging target")
    with (
        patch.object(harness, "load_protocol", lambda _root, _path: (protocol, protocol_sha)),
        patch.object(harness, "SYSTEMS", SYSTEMS),
        patch.object(harness, "load_exact_records", load_exact_records),
        patch.object(harness, "train_exact_b50_router", train_exact_b40_router),
    ):
        measured = harness.run(
            root,
            protocol_path,
            system=system,
            output=staging,
        )
    observations = (staging / "observations.jsonl").read_bytes()
    output.mkdir(parents=True)
    observations_path = output / "observations.jsonl"
    _write_immutable(observations_path, observations)
    underlying_status = str(measured["status"])
    measured.update(
        {
            "format": RESULT_FORMAT,
            "status": "PASS_SAME_CHECKPOINT_B40_LORA_CUDA_RUNTIME"
            if underlying_status.startswith("PASS")
            else "FAIL_SAME_CHECKPOINT_B40_LORA_CUDA_RUNTIME",
            "protocol_sha256": protocol_sha,
            "underlying_harness_status": underlying_status,
            "observations_path": observations_path.relative_to(root).as_posix(),
            "observations_sha256": sha256_file(observations_path),
            "phase4_certified": False,
            "claim_boundary": "One exact-B40 same-checkpoint LoRA CUDA runtime measurement with output identity. No CPU comparison, final test, Phase 4, or ABI-superiority claim.",
        }
    )
    measured.pop("evidence_sha256", None)
    measured["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(measured)).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(measured, indent=2, sort_keys=True).encode() + b"\n",
    )
    return measured


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(
        root,
        (root / args.protocol).resolve(),
        system=args.system,
        output=(root / args.output_dir).resolve(),
    )
    print(json.dumps({
        "status": result["status"],
        "system": result["system"],
        "metrics": result["metrics"],
        "identity": result["quality_output_identities"],
    }, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
