"""GPU runtime for the quality-qualified exact-B50 conformed L1 product."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import capability_compiler_phase4_b50_gpu_runtime as base
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b50_l1_conformance import conform_output
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b50-l1-conformed-gpu-runtime/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-l1-conformed-gpu-runtime-result/1"


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_CONFORMED_L1_SAME_PRODUCT_GPU_RUNTIME"
        or protocol.get("system") != "L1"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("conformed L1 runtime governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"conformed L1 runtime binding changed: {relative}")
    verification = _json((root / str(protocol["quality_verification"])).resolve())
    if (
        verification.get("status") != "PASS_INDEPENDENT_L1_CONFORMANCE_VERIFICATION"
        or not all(bool(value) for value in verification.get("gates", {}).values())
    ):
        raise Phase3Error("conformed L1 quality verification is not passing")
    return protocol, sha256_file(path)


def _conformed_request(
    original_request: Any,
    runtime: Mapping[str, Any],
    system: str,
    probe: Mapping[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    row = original_request(runtime, system, probe)
    started = time.perf_counter()
    output, replacements = conform_output(str(row["output"]), str(row["capability"]))
    if replacements > 1:
        raise Phase3Error("conformed L1 runtime produced multiple replacements")
    authoritative_ids = [
        int(value)
        for value in runtime["tokenizer"](output, add_special_tokens=False).input_ids
    ]
    conformance_seconds = time.perf_counter() - started
    total_seconds = float(row["total_seconds"]) + conformance_seconds
    raw = output.encode("utf-8")
    row.update(
        {
            "output": output,
            "retokenized_output_token_ids": authoritative_ids,
            "output_utf8_bytes": len(raw),
            "output_characters": len(output),
            "authoritative_output_tokens": len(authoritative_ids),
            "total_seconds": total_seconds,
            "bytes_per_second": len(raw) / total_seconds,
            "characters_per_second": len(output) / total_seconds,
        }
    )
    row["execution"] = dict(row["execution"])
    row["execution"].update(
        {
            "conformance_rule_executed": True,
            "conformance_replacements": replacements,
            "conformance_seconds": conformance_seconds,
        }
    )
    audit["invocations"] += 1
    audit["replacements"] += replacements
    audit["seconds"] += conformance_seconds
    return row


def run(root: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha256 = _load_protocol(root, protocol_path)
    if output_dir.exists():
        raise Phase3Error("conformed L1 runtime output already exists")
    audit = {"invocations": 0, "replacements": 0, "seconds": 0.0}
    original_request = base._baseline_request

    def wrapped(runtime: Mapping[str, Any], system: str, probe: Mapping[str, Any]) -> dict[str, Any]:
        return _conformed_request(original_request, runtime, system, probe, audit)

    engine_dir = output_dir / "engine"
    base._baseline_request = wrapped
    try:
        engine = base.run(
            root,
            (root / str(protocol["source_runtime_protocol"])).resolve(),
            system="L1",
            output=engine_dir,
        )
    finally:
        base._baseline_request = original_request

    observations_path = root / str(engine["observations_path"])
    observations = [
        json.loads(line)
        for line in observations_path.read_bytes().splitlines()
        if line.strip()
    ]
    conformance_rows = sum(
        row.get("execution", {}).get("conformance_rule_executed") is True
        for row in observations
    )
    observation_replacements = sum(
        int(row.get("execution", {}).get("conformance_replacements", 0))
        for row in observations
    )
    gates = {
        "quality_verification_pass": True,
        "source_runtime_engine_pass": engine["status"]
        == "PASS_SAME_CHECKPOINT_B50_CUDA_RUNTIME"
        and all(bool(value) for value in engine["gates"].values()),
        "all_120_measured_rows_executed_conformance": conformance_rows == 120,
        "all_124_cold_warm_measured_requests_executed_conformance": audit["invocations"]
        == 124,
        "expected_zero_seed104729_replacements": audit["replacements"]
        == int(protocol["expected"]["total_replacements"])
        == 0
        and observation_replacements == 0,
        "conformance_time_included": audit["seconds"] > 0
        and all(
            float(row["execution"]["conformance_seconds"]) >= 0
            for row in observations
        ),
        "same_checkpoint_as_quality": sha256_file(
            root / str(protocol["checkpoint_path"])
        )
        == str(protocol["checkpoint_sha256"]),
        "depth": len(observations) == 120
        and len({str(row["probe_id"]) for row in observations}) == 100,
        "authoritative_final_output_token_accounting": all(
            int(row["authoritative_output_tokens"])
            == len(row["retokenized_output_token_ids"])
            for row in observations
        ),
        "teacher_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    result: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "status": (
            "PASS_CONFORMED_L1_SAME_PRODUCT_GPU_RUNTIME"
            if all(gates.values())
            else "FAIL_CONFORMED_L1_SAME_PRODUCT_GPU_RUNTIME"
        ),
        "protocol_sha256": protocol_sha256,
        "system": "L1",
        "seed": 104729,
        "engine_result": (engine_dir / "result.json").relative_to(root).as_posix(),
        "engine_result_sha256": sha256_file(engine_dir / "result.json"),
        "observations": observations_path.relative_to(root).as_posix(),
        "observations_sha256": sha256_file(observations_path),
        "metrics": engine["metrics"],
        "cold": engine["cold"],
        "active_tensor_bytes": engine["active_tensor_bytes"],
        "peak_process_rss_delta_bytes": engine["peak_process_rss_delta_bytes"],
        "peak_cuda_allocated_bytes": engine["peak_cuda_allocated_bytes"],
        "conformance": {
            "invocations": audit["invocations"],
            "replacements": audit["replacements"],
            "wall_seconds": audit["seconds"],
            "learned_parameters": 0,
        },
        "gates": gates,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "abi_superiority_certified": False,
        "claim_boundary": "Same-checkpoint GPU runtime for the independently quality-qualified conformed L1 product. Candidate-versus-comparator composition, final test, Phase 4, and ABI superiority remain open.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    result = run(root, args.protocol.resolve(), args.output_dir.resolve())
    print(result["status"])
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
