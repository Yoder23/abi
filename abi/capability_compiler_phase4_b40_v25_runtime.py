"""Measure the exact signed B40 five-route LayerCake v25 artifact on CPU and CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from . import capability_compiler_phase4_b50_cpu_runtime as cpu_engine
from . import capability_compiler_phase4_b50_gpu_runtime as gpu_engine
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b20_v25_physical_screen import _api
from .capability_compiler_phase4_b40_v25_product_conformance import _package
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_v19_frontier_rescreen import _json


PROTOCOL_FORMAT = "abi-capability-compiler-phase4-b40-v25-runtime/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b40-v25-runtime-result/1"


def _load_overlay(
    root: Path, path: Path, *, mode: str
) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status")
        != f"PREREGISTERED_EXACT_B40_V25_{mode.upper()}_RUNTIME"
        or protocol.get("runtime_interface") != "lc-direct-neural-core/25"
        or protocol.get("v25_runtime_mode") != mode
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_generation_authorized") is not False
        or protocol.get("candidate_construction_authorized") is not False
        or protocol.get("deterministic_package_rebuild_authorized") is not True
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("exact B40 v25 runtime governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B40 v25 runtime binding changed: {relative}")
    product = _json(root / str(protocol["v25_product_result"]))
    if (
        product.get("status") != "PASS_STABLE_B40_SIGNED_V25_PRODUCT_CONFORMANCE"
        or not result_evidence_digest_valid(product)
    ):
        raise Phase3Error("exact B40 v25 product prerequisite changed")
    expected = protocol["systems"]["ABI"]
    system = next(
        row for row in product["systems"]
        if int(row["seed"]) == int(expected["seed"])
    )
    if (
        system["package"]["archive_sha256"] != expected["archive_sha256"]
        or system["package"]["tensor_payload_hash"]
        != expected["tensor_payload_hash"]
        or system["outputs"]["sha256"] != expected["quality_reference_sha256"]
    ):
        raise Phase3Error("exact B40 v25 runtime package or output identity changed")
    return protocol, sha256_file(path)


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged["systems"] = dict(base["systems"])
    merged["systems"]["ABI"] = dict(overlay["systems"]["ABI"])
    merged["runtime_interface"] = "lc-direct-neural-core/25"
    merged["v25_runtime_mode"] = overlay["v25_runtime_mode"]
    merged["v25_product_result"] = overlay["v25_product_result"]
    merged["candidate_screen_protocol"] = overlay["candidate_screen_protocol"]
    if "locked_phase2_runtime" in overlay:
        merged["locked_phase2_runtime"] = dict(overlay["locked_phase2_runtime"])
    return merged


def load_cpu_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    overlay, protocol_sha = _load_overlay(root, path, mode="cpu")
    base, _ = cpu_engine.load_protocol(
        root, root / str(overlay["base_runtime_protocol"])
    )
    return _merge(base, overlay), protocol_sha


def load_gpu_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    overlay, protocol_sha = _load_overlay(root, path, mode="gpu")
    base, _ = gpu_engine.load_protocol(
        root, root / str(overlay["base_runtime_protocol"])
    )
    return _merge(base, overlay), protocol_sha


def _build_candidate_v25(
    root: Path,
    protocol: Mapping[str, Any],
    temporary: Path,
):
    source = _json(root / str(protocol["candidate_screen_protocol"]))
    for relative, expected_hash in source["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected_hash:
            raise Phase3Error(f"exact B40 v25 source binding changed: {relative}")
    expected = protocol["systems"]["ABI"]
    spec = next(
        row for row in source["systems"]
        if int(row["seed"]) == int(expected["seed"])
    )
    layercake_root = (root / str(source["layercake_root"])).resolve()
    api = _api(layercake_root)
    api["Host"] = api["ClarificationRouteAllocationBoundedCoreHost"]
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(source["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    package_path = temporary / "candidate.cake"
    built = _package(
        root,
        source,
        spec,
        package_path,
        api,
        private,
        public_pem,
    )
    if (
        built["archive_sha256"] != expected["archive_sha256"]
        or built["tensor_payload_hash"] != expected["tensor_payload_hash"]
        or built["archive_bytes"] != int(expected["archive_bytes"])
    ):
        raise Phase3Error("exact B40 v25 runtime rebuild changed")
    signer = api["key_id"](public_pem)
    return package_path, built, public_pem, signer, api, source


def _wrapper_result(
    root: Path,
    protocol_sha: str,
    output: Path,
    engine: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    engine_path = output / "engine" / "result.json"
    observations = output / "engine" / "observations.jsonl"
    passed = str(engine["status"]).startswith("PASS")
    result = {
        "format": RESULT_FORMAT,
        "status": f"PASS_SAME_ARTIFACT_B40_V25_{mode.upper()}_RUNTIME"
        if passed
        else f"FAIL_SAME_ARTIFACT_B40_V25_{mode.upper()}_RUNTIME",
        "protocol_sha256": protocol_sha,
        "runtime_interface": "lc-direct-neural-core/25",
        "mode": mode,
        "engine_status": engine["status"],
        "engine_result": engine_path.relative_to(root).as_posix(),
        "engine_result_sha256": sha256_file(engine_path),
        "observations": observations.relative_to(root).as_posix(),
        "observations_sha256": sha256_file(observations),
        "engine": dict(engine),
        "training_performed": False,
        "teacher_query_performed": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Exact same-artifact B40 v25 development runtime. Matched B40 baselines, external human review, final test, Phase 4, and ABI superiority remain open.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def preflight_cpu(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_cpu_protocol(root, protocol_path)
    original = cpu_engine.load_protocol
    cpu_engine.load_protocol = lambda _root, _path: (protocol, protocol_sha)
    try:
        result = cpu_engine.preflight(root, protocol_path)
    finally:
        cpu_engine.load_protocol = original
    engine_status = str(result["status"])
    result["engine_status"] = engine_status
    result["status"] = (
        "PASS_B40_V25_CPU_RUNTIME_PREFLIGHT"
        if engine_status.startswith("PASS")
        else "FAIL_B40_V25_CPU_RUNTIME_PREFLIGHT"
    )
    result["v25_protocol_sha256"] = protocol_sha
    result["runtime_interface"] = "lc-direct-neural-core/25"
    return result


def preflight_gpu(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_gpu_protocol(root, protocol_path)
    distinct, scheduled = gpu_engine.runtime_schedule(root, protocol)
    spec = protocol["systems"]["ABI"]
    reference = gpu_engine._reference(
        root / str(spec["quality_reference_outputs"]),
        str(spec["quality_reference_sha256"]),
    )
    gates = {
        "cuda_available": gpu_engine.torch.cuda.is_available(),
        "depth": len(distinct) == 100 and len(scheduled) == 120,
        "quality_reference_complete": len(reference) == 1400,
        "training_prohibited": True,
        "teacher_query_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "format": "abi-capability-compiler-phase4-b40-v25-gpu-runtime-preflight/1",
        "status": "PASS_B40_V25_GPU_RUNTIME_PREFLIGHT"
        if all(gates.values())
        else "FAIL_B40_V25_GPU_RUNTIME_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "runtime_interface": "lc-direct-neural-core/25",
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "final_test_accessed": False,
    }


def run_cpu(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_cpu_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B40 v25 CPU output exists: {output}")
    original_builder = cpu_engine._build_candidate
    original_loader = cpu_engine.load_protocol
    cpu_engine._build_candidate = _build_candidate_v25
    cpu_engine.load_protocol = lambda _root, _path: (protocol, protocol_sha)
    try:
        engine = cpu_engine.run(root, protocol_path, output / "engine")
    finally:
        cpu_engine._build_candidate = original_builder
        cpu_engine.load_protocol = original_loader
    return _wrapper_result(root, protocol_sha, output, engine, mode="cpu")


def run_gpu(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_gpu_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B40 v25 GPU output exists: {output}")
    original_builder = gpu_engine._build_candidate
    original_loader = gpu_engine.load_protocol
    gpu_engine._build_candidate = _build_candidate_v25
    gpu_engine.load_protocol = lambda _root, _path: (protocol, protocol_sha)
    try:
        engine = gpu_engine.run(
            root, protocol_path, system="ABI", output=output / "engine"
        )
    finally:
        gpu_engine._build_candidate = original_builder
        gpu_engine.load_protocol = original_loader
    return _wrapper_result(root, protocol_sha, output, engine, mode="gpu")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "command", choices=("preflight-cpu", "preflight-gpu", "run-cpu", "run-gpu")
    )
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = (root / args.protocol).resolve()
    if args.command == "preflight-cpu":
        result = preflight_cpu(root, protocol_path)
    elif args.command == "preflight-gpu":
        result = preflight_gpu(root, protocol_path)
    else:
        if not args.output_dir:
            raise Phase3Error("B40 v25 runtime output directory is required")
        output = (root / args.output_dir).resolve()
        result = (
            run_cpu(root, protocol_path, output)
            if args.command == "run-cpu"
            else run_gpu(root, protocol_path, output)
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
