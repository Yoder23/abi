"""Independently compose exact-B50 v24 CPU/GPU runtime evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import capability_compiler_phase4_b50_v23_runtime_compose as v23_compose
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b50_gpu_runtime import SYSTEMS
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_b50_v24_runtime import (
    RESULT_FORMAT as V24_RUNTIME_RESULT_FORMAT,
    load_cpu_protocol,
    load_gpu_protocol,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b50-v24-runtime-compose/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-v24-runtime-compose-result/1"


def _wrapper_digest_valid(result: Mapping[str, Any]) -> bool:
    expected = str(result.get("evidence_sha256", ""))
    payload = {key: value for key, value in result.items() if key != "evidence_sha256"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == expected


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_INDEPENDENT_EXACT_B50_V24_RUNTIME_COMPOSITION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or set(protocol.get("baseline_gpu_results", {})) != set(SYSTEMS[1:])
    ):
        raise Phase3Error("exact B50 v24 runtime composition governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 v24 composition binding changed: {relative}")
    return protocol, sha256_file(path)


def _load_wrapper(
    root: Path,
    path: Path,
    *,
    mode: str,
    protocol_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wrapper = _json(path)
    engine_path = root / str(wrapper["engine_result"])
    if (
        wrapper.get("format") != V24_RUNTIME_RESULT_FORMAT
        or wrapper.get("mode") != mode
        or wrapper.get("runtime_interface") != "lc-direct-neural-core/24"
        or wrapper.get("protocol_sha256") != protocol_sha
        or not _wrapper_digest_valid(wrapper)
        or sha256_file(engine_path) != wrapper.get("engine_result_sha256")
        or sha256_file(root / str(wrapper["observations"]))
        != wrapper.get("observations_sha256")
        or wrapper.get("training_performed") is not False
        or wrapper.get("teacher_query_performed") is not False
        or wrapper.get("final_test_accessed") is not False
    ):
        raise Phase3Error(f"exact B50 v24 {mode} wrapper changed")
    engine = _json(engine_path)
    if engine != wrapper.get("engine") or not result_evidence_digest_valid(engine):
        raise Phase3Error(f"exact B50 v24 {mode} engine changed")
    return wrapper, engine


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B50 v24 composition exists: {output}")
    output.mkdir(parents=True)
    engine_path = output / "engine_result.json"
    original_loader = v23_compose.load_protocol
    original_cpu = v23_compose.load_cpu_protocol
    original_gpu = v23_compose.load_gpu_protocol
    original_wrapper = v23_compose._load_wrapper
    v23_compose.load_protocol = lambda _root, _path: (protocol, protocol_sha)
    v23_compose.load_cpu_protocol = load_cpu_protocol
    v23_compose.load_gpu_protocol = load_gpu_protocol
    v23_compose._load_wrapper = _load_wrapper
    try:
        engine = v23_compose.verify(root, protocol_path, engine_path)
    finally:
        v23_compose.load_protocol = original_loader
        v23_compose.load_cpu_protocol = original_cpu
        v23_compose.load_gpu_protocol = original_gpu
        v23_compose._load_wrapper = original_wrapper
    if not result_evidence_digest_valid(engine):
        raise Phase3Error("exact B50 v24 composition engine digest changed")
    passed = engine["status"].startswith("PASS") and all(engine["gates"].values())
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_EXACT_B50_V24_CPU_GPU_RUNTIME_COMPOSED"
        if passed
        else "FAIL_EXACT_B50_V24_RUNTIME_GATE_CLOSED",
        "protocol_sha256": protocol_sha,
        "runtime_interface": "lc-direct-neural-core/24",
        "candidate_package": dict(protocol["candidate_package"]),
        "engine_status": engine["status"],
        "engine_result": engine_path.relative_to(root).as_posix(),
        "engine_result_sha256": sha256_file(engine_path),
        "quality_qualified_gpu_baselines": engine[
            "quality_qualified_gpu_baselines"
        ],
        "gpu_comparisons": engine["gpu_comparisons"],
        "cpu": engine["cpu"],
        "cpu_gpu_candidate_output_identities": engine[
            "cpu_gpu_candidate_output_identities"
        ],
        "gates": engine["gates"],
        "training_performed": False,
        "model_inference_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "abi_superiority_certified": False,
        "claim_boundary": "Independent exact-V24 same-artifact CPU/GPU runtime composition on development prompts. A quality-qualified matched comparator, final test, Phase 4, and unconditional ABI superiority remain open unless explicitly passed here.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
