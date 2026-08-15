"""Run same-artifact CPU/GPU runtime for the exact B50 v24 package."""

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
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_b50_v24_conformance import (
    _api_v24,
    _repackage_v24,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase4_v22_b50_rescreen import (
    _api as _api_v22,
    load_protocol as _load_v22_protocol,
)


PROTOCOL_FORMAT = "abi-capability-compiler-phase4-b50-v24-runtime/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-v24-runtime-result/1"


def _load_overlay(
    root: Path, path: Path, *, mode: str
) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status")
        != f"PREREGISTERED_EXACT_B50_V24_{mode.upper()}_RUNTIME"
        or protocol.get("runtime_interface") != "lc-direct-neural-core/24"
        or protocol.get("v24_runtime_mode") != mode
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_generation_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("exact B50 v24 runtime governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 v24 runtime binding changed: {relative}")
    conformance = _json(root / str(protocol["v24_conformance_result"]))
    if (
        conformance.get("status")
        != "PASS_EXACT_B50_V24_THREE_SEED_OUTPUT_CONFORMANCE"
        or not result_evidence_digest_valid(conformance)
    ):
        raise Phase3Error("exact B50 v24 conformance prerequisite changed")
    candidate = protocol["systems"]["ABI"]
    system = next(
        row for row in conformance["systems"]
        if int(row["seed"]) == int(candidate["seed"])
    )
    if (
        system["package"]["archive_sha256"] != candidate["archive_sha256"]
        or system["package"]["tensor_payload_hash"]
        != candidate["tensor_payload_hash"]
    ):
        raise Phase3Error("exact B50 v24 runtime package identity changed")
    return protocol, sha256_file(path)


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged["systems"] = dict(base["systems"])
    merged["systems"]["ABI"] = dict(overlay["systems"]["ABI"])
    merged["runtime_interface"] = "lc-direct-neural-core/24"
    merged["v24_runtime_mode"] = overlay["v24_runtime_mode"]
    merged["v24_conformance_result"] = overlay["v24_conformance_result"]
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


def _build_candidate_v24(
    root: Path,
    protocol: Mapping[str, Any],
    temporary: Path,
):
    source, _ = _load_v22_protocol(
        root, root / str(protocol["candidate_screen_protocol"])
    )
    layercake_root = (root / str(source["layercake_root"])).resolve()
    api_v22 = _api_v22(layercake_root)
    api_v24 = _api_v24(layercake_root)
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(source["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api_v24["key_id"](public_pem)
    expected = protocol["systems"]["ABI"]
    spec = next(
        row for row in source["systems"]
        if int(row["seed"]) == int(expected["seed"])
    )
    package_path, package = _repackage_v24(
        root,
        source,
        spec,
        temporary,
        api_v22,
        api_v24,
        private,
        public_pem,
    )
    if (
        package["archive_sha256"] != expected["archive_sha256"]
        or package["tensor_payload_hash"] != expected["tensor_payload_hash"]
        or int(package["archive_bytes"]) != int(expected["archive_bytes"])
    ):
        raise Phase3Error("exact B50 v24 runtime rebuild changed")
    return package_path, package, public_pem, signer, api_v24, source


def _wrapper_result(
    root: Path,
    protocol_sha: str,
    output: Path,
    engine: Mapping[str, Any],
    *,
    mode: str,
    retention_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    engine_path = output / "engine" / "result.json"
    observations = output / "engine" / "observations.jsonl"
    passed = str(engine["status"]).startswith("PASS")
    result = {
        "format": RESULT_FORMAT,
        "status": f"PASS_SAME_ARTIFACT_B50_V24_{mode.upper()}_RUNTIME"
        if passed
        else f"FAIL_SAME_ARTIFACT_B50_V24_{mode.upper()}_RUNTIME",
        "protocol_sha256": protocol_sha,
        "runtime_interface": "lc-direct-neural-core/24",
        "mode": mode,
        "engine_status": engine["status"],
        "engine_result": engine_path.relative_to(root).as_posix(),
        "engine_result_sha256": sha256_file(engine_path),
        "observations": observations.relative_to(root).as_posix(),
        "observations_sha256": sha256_file(observations),
        "engine": dict(engine),
        "retention_reference": dict(retention_reference or {}),
        "training_performed": False,
        "teacher_query_performed": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Exact same-artifact v24 development runtime rescreen. No final test, matched-quality baseline endpoint, Phase 4, or unconditional ABI-superiority claim.",
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
    result["v24_protocol_sha256"] = protocol_sha
    result["runtime_interface"] = "lc-direct-neural-core/24"
    return result


def run_cpu(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_cpu_protocol(root, protocol_path)
    overlay = _json(protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B50 v24 CPU output exists: {output}")
    original_builder = cpu_engine._build_candidate
    original_loader = cpu_engine.load_protocol
    cpu_engine._build_candidate = _build_candidate_v24
    cpu_engine.load_protocol = lambda _root, _path: (protocol, protocol_sha)
    try:
        engine = cpu_engine.run(root, protocol_path, output / "engine")
    finally:
        cpu_engine._build_candidate = original_builder
        cpu_engine.load_protocol = original_loader
    return _wrapper_result(
        root,
        protocol_sha,
        output,
        engine,
        mode="cpu",
        retention_reference=overlay.get("retention_reference"),
    )


def run_gpu(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_gpu_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B50 v24 GPU output exists: {output}")
    original_builder = gpu_engine._build_candidate
    original_loader = gpu_engine.load_protocol
    gpu_engine._build_candidate = _build_candidate_v24
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
    parser.add_argument("command", choices=("preflight-cpu", "run-cpu", "run-gpu"))
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = (root / args.protocol).resolve()
    if args.command == "preflight-cpu":
        result = preflight_cpu(root, protocol_path)
    else:
        if not args.output_dir:
            raise Phase3Error("v24 runtime output directory is required")
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
