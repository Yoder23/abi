"""Materialize the exact signed Phase 7 core before a serving process starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase7_integrated_runtime import _build_core_archive


FORMAT = "abi-capability-compiler-phase7-product-materialization/1"
RESULT_FORMAT = "abi-capability-compiler-phase7-product-materialization-result/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_EXACT_CORE_OUT_OF_PROCESS_MATERIALIZATION"
        or protocol.get("source_model_loading_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_content_change_authorized") is not False
    ):
        raise Phase3Error("Phase 7 materialization governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 7 materialization binding changed: {relative}")
    return protocol, sha256_file(path)


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    target = (root / protocol["output_archive"]).resolve()
    result = (root / protocol["output_result"]).resolve()
    gates = {
        "outputs_absent": not target.exists() and not result.exists(),
        "exact_archive_hash_registered": len(protocol["expected_archive_sha256"]) == 64,
        "exact_payload_hash_registered": len(protocol["expected_payload_sha256"]) == 64,
        "training_absent": True,
        "teacher_absent": True,
    }
    return {
        "format": "abi-capability-compiler-phase7-product-materialization-preflight/1",
        "status": "PASS_PHASE7_PRODUCT_MATERIALIZATION_PREFLIGHT"
        if all(gates.values())
        else "FAIL_PHASE7_PRODUCT_MATERIALIZATION_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "gates": gates,
    }


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    target = (root / protocol["output_archive"]).resolve()
    result_path = (root / protocol["output_result"]).resolve()
    if target.exists() or result_path.exists():
        raise Phase3Error("immutable Phase 7 materialization output exists")
    runtime_protocol = _json(root / protocol["integrated_runtime_protocol"])
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="abi-phase7-materialize-") as raw:
        _, _, built, _, archive = _build_core_archive(
            root, runtime_protocol, Path(raw)
        )
        if (
            built["archive_sha256"] != protocol["expected_archive_sha256"]
            or built["tensor_payload_hash"] != protocol["expected_payload_sha256"]
        ):
            raise Phase3Error("materialized Phase 7 product identity changed")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            shutil.copyfile(archive, temporary)
            if sha256_file(temporary) != protocol["expected_archive_sha256"]:
                raise Phase3Error("copied Phase 7 product identity changed")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    elapsed = time.perf_counter() - started
    gates = {
        "archive_identity_exact": sha256_file(target)
        == protocol["expected_archive_sha256"],
        "payload_identity_exact": built["tensor_payload_hash"]
        == protocol["expected_payload_sha256"],
        "core_component_inventory_exact": built["component_parameters"]
        == {"model": 61655050, "router": 1058040, "residual": 124416},
        "training_absent": True,
        "teacher_absent": True,
        "content_change_absent": True,
    }
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_PHASE7_PRODUCT_MATERIALIZATION"
        if all(gates.values())
        else "FAIL_PHASE7_PRODUCT_MATERIALIZATION",
        "protocol_sha256": protocol_sha,
        "archive": target.relative_to(root).as_posix(),
        "archive_sha256": sha256_file(target),
        "archive_bytes": target.stat().st_size,
        "tensor_payload_sha256": built["tensor_payload_hash"],
        "package_hash": built["package_hash"],
        "component_parameters": built["component_parameters"],
        "total_parameters": built["total_parameters"],
        "materialization_seconds": elapsed,
        "gates": gates,
        "teacher_model_loaded": False,
        "training_performed": False,
        "model_artifact_mutated": False,
        "claim_boundary": "One-time deterministic materialization of the already-certified exact core. Serving screens must run in new processes and may only consume this hash-bound archive.",
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    _write_immutable(
        result_path,
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = (root / args.protocol).resolve()
    result = preflight(root, protocol_path) if args.preflight else run(root, protocol_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
