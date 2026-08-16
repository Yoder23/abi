"""Prepare and collect a clean-export Phase 8 rehearsal on development hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase7_direct_artifact_runtime import (
    preflight as runtime_preflight,
)
from .capability_compiler_phase8_release_readiness import (
    hardware_document,
    verify_manifest,
)


FORMAT = "abi-capability-compiler-phase8-local-clean-rehearsal/1"


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_entry(
    abi_root: Path, layercake_root: Path, relative: str
) -> tuple[str, Path]:
    posix = PurePosixPath(relative)
    if posix.is_absolute():
        raise Phase3Error(f"absolute release path rejected: {relative}")
    parts = posix.parts
    if parts[:2] == ("..", "layercake_release"):
        remainder = parts[2:]
        if not remainder or ".." in remainder:
            raise Phase3Error(f"invalid LayerCake release path: {relative}")
        target = (layercake_root / Path(*remainder)).resolve()
        try:
            target.relative_to(layercake_root.resolve())
        except ValueError as exc:
            raise Phase3Error(f"LayerCake release path escaped root: {relative}") from exc
        return "layercake", target
    if ".." in parts or not parts:
        raise Phase3Error(f"invalid ABI release path: {relative}")
    target = (abi_root / Path(*parts)).resolve()
    try:
        target.relative_to(abi_root.resolve())
    except ValueError as exc:
        raise Phase3Error(f"ABI release path escaped root: {relative}") from exc
    return "abi", target


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    original = protocol.get("status") == "PREREGISTERED_PHASE8_LOCAL_CLEAN_REHEARSAL"
    polarity_repair = (
        protocol.get("status")
        == "PREREGISTERED_PHASE8_LOCAL_CLEAN_REHEARSAL_POLARITY_REPAIR"
        and protocol.get("repair_of")
        == "ABI_CAPABILITY_COMPILER_PHASE8_LOCAL_CLEAN_REHEARSAL_PROTOCOL_V1076.json"
        and protocol.get("repair_scope") == "FALSE_POLARITY_GATE_LABEL_ONLY"
    )
    collection_path_repair = (
        protocol.get("status")
        == "PREREGISTERED_PHASE8_LOCAL_CLEAN_REHEARSAL_COLLECTION_PATH_REPAIR"
        and protocol.get("repair_of")
        == "ABI_CAPABILITY_COMPILER_PHASE8_LOCAL_CLEAN_REHEARSAL_REPAIR_PROTOCOL_V1079.json"
        and protocol.get("repair_scope")
        == "WINDOWS_MAX_PATH_COLLECTION_ROOT_SHORTENING_ONLY"
        and protocol.get("inference_reexecution_authorized") is False
    )
    if (
        protocol.get("format") != FORMAT
        or not (original or polarity_repair or collection_path_repair)
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_authorized") is not False
        or protocol.get("phase8_certification_authorized") is not False
        or (
            protocol.get("model_inference_authorized")
            != "ONE_EXACT_CPU_AND_ONE_EXACT_CUDA_SAME_MACHINE_REHEARSAL_ONLY"
            if not collection_path_repair
            else protocol.get("model_inference_authorized") is not False
        )
    ):
        raise Phase3Error("Phase 8 local-rehearsal governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 8 local-rehearsal binding changed: {relative}")
    if polarity_repair:
        failure = _json(root / protocol["preserved_failure"])
        failure_gates = dict(failure.get("gates", {}))
        claimed = failure_gates.pop("independent_hardware_claimed", None)
        if (
            failure.get("status") != protocol["preserved_failure_status"]
            or claimed is not False
            or not failure_gates
            or not all(value is True for value in failure_gates.values())
        ):
            raise Phase3Error("Phase 8 polarity repair exceeded its registered scope")
    return protocol, sha256_file(path)


def _target_roots(protocol: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    parent = Path(str(protocol["clean_parent"])).resolve()
    return parent, parent / "abi_release", parent / "layercake_release"


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    parent, abi_clean, layercake_clean = _target_roots(protocol)
    hardware = hardware_document()
    repair = protocol.get("repair_scope") in {
        "FALSE_POLARITY_GATE_LABEL_ONLY",
        "WINDOWS_MAX_PATH_COLLECTION_ROOT_SHORTENING_ONLY",
    }
    gates = {
        "development_hardware_exact": hardware["fingerprint_sha256"]
        == protocol["development_hardware_fingerprint_sha256"],
        "cuda_available": torch.cuda.is_available(),
        "abi_packet_source_is_ancestor": subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.as_posix()}",
                "merge-base",
                "--is-ancestor",
                protocol["abi_packet_source_commit"],
                "HEAD",
            ],
            cwd=root,
        ).returncode
        == 0,
        "layercake_source_head_exact": _git(
            (root / "../layercake_release").resolve(), "rev-parse", "HEAD"
        )
        == protocol["layercake_commit"],
        "clean_parent_state_expected": parent.exists() is repair,
        "abi_clean_state_expected": abi_clean.exists() is repair,
        "layercake_clean_state_expected": layercake_clean.exists() is repair,
        "preserved_failure_state_expected": (
            (root / protocol["preserved_failure"]).is_file()
            if repair
            else True
        ),
        "collection_output_absent": not (root / protocol["collection_output"]).exists(),
        "verification_output_absent": not (
            root / protocol["verification_output"]
        ).exists(),
        "teacher_absent": True,
        "training_absent": True,
        "external_independence_not_claimed": True,
    }
    if protocol.get("repair_scope") == "WINDOWS_MAX_PATH_COLLECTION_ROOT_SHORTENING_ONLY":
        gates.update(
            clean_device_results_present=all(
                (abi_clean / protocol["clean_device_outputs"][device] / "result.json").is_file()
                and (
                    abi_clean
                    / protocol["clean_device_outputs"][device]
                    / "observations.jsonl"
                ).is_file()
                for device in ("cpu", "cuda")
            ),
            collection_targets_below_260_characters=max(
                len(
                    str(
                        (
                            root
                            / protocol["collection_root"]
                            / relative
                        ).resolve()
                    )
                )
                for relative in protocol["collect_files"]
            )
            < 260,
        )
    return {
        "format": "abi-capability-compiler-phase8-local-clean-rehearsal-preflight/1",
        "status": "PASS_PHASE8_LOCAL_CLEAN_REHEARSAL_PREFLIGHT"
        if all(gates.values())
        else "FAIL_PHASE8_LOCAL_CLEAN_REHEARSAL_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "hardware": hardware,
        "clean_parent": str(parent),
        "gates": gates,
        "phase8_certified": False,
    }


def prepare(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    _, abi_clean, layercake_clean = _target_roots(protocol)
    if output.exists():
        raise Phase3Error("immutable local-rehearsal preparation output exists")
    if not abi_clean.is_dir() or not layercake_clean.is_dir():
        raise Phase3Error("clean worktrees do not exist")
    if _git(abi_clean, "rev-parse", "HEAD") != protocol["abi_packet_source_commit"]:
        raise Phase3Error("clean ABI worktree commit changed")
    if _git(layercake_clean, "rev-parse", "HEAD") != protocol["layercake_commit"]:
        raise Phase3Error("clean LayerCake worktree commit changed")
    if _git(abi_clean, "diff", "--name-only") or _git(
        layercake_clean, "diff", "--name-only"
    ):
        raise Phase3Error("clean tracked worktree bytes changed before preparation")

    manifest_source = (root / protocol["release_manifest"]).resolve()
    manifest = _json(manifest_source)
    copied: list[dict[str, Any]] = []
    verified_tracked = 0
    for relative, specification in manifest["files"].items():
        repository, target = _resolve_entry(abi_clean, layercake_clean, relative)
        expected = str(specification["sha256"])
        if specification["tracked"]:
            if not target.is_file() or sha256_file(target) != expected:
                raise Phase3Error(f"clean tracked inventory mismatch: {relative}")
            verified_tracked += 1
            continue
        if repository != "abi":
            raise Phase3Error("untracked LayerCake release payload is prohibited")
        source = (root / relative).resolve()
        if not source.is_file() or sha256_file(source) != expected:
            raise Phase3Error(f"source payload mismatch: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if sha256_file(target) != expected:
            raise Phase3Error(f"copied payload mismatch: {relative}")
        copied.append(
            {"path": relative, "bytes": target.stat().st_size, "sha256": expected}
        )

    clean_manifest = (abi_clean / protocol["release_manifest"]).resolve()
    clean_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not clean_manifest.exists():
        shutil.copy2(manifest_source, clean_manifest)
    if sha256_file(clean_manifest) != protocol["release_manifest_sha256"]:
        raise Phase3Error("copied release manifest changed")
    manifest_verification = verify_manifest(
        abi_clean,
        abi_clean / protocol["release_readiness_protocol"],
        clean_manifest,
    )
    clean_runtime_preflight_path = (
        abi_clean / protocol["clean_runtime_preflight"]
    ).resolve()
    if clean_runtime_preflight_path.exists():
        clean_runtime_preflight = _json(clean_runtime_preflight_path)
    else:
        clean_runtime_preflight = runtime_preflight(
            abi_clean, abi_clean / protocol["product_protocol"]
        )
        clean_runtime_preflight_path.parent.mkdir(parents=True, exist_ok=True)
        _write_immutable(
            clean_runtime_preflight_path,
            json.dumps(clean_runtime_preflight, indent=2, sort_keys=True).encode()
            + b"\n",
        )
    expected_outputs = {
        device: abi_clean / protocol["clean_device_outputs"][device]
        for device in ("cpu", "cuda")
    }
    gates = {
        "abi_commit_exact": _git(abi_clean, "rev-parse", "HEAD")
        == protocol["abi_packet_source_commit"],
        "layercake_commit_exact": _git(layercake_clean, "rev-parse", "HEAD")
        == protocol["layercake_commit"],
        "tracked_inventory_exact": verified_tracked
        == int(manifest["file_count"]) - len(copied),
        "untracked_inventory_exact": len(copied) == 3,
        "manifest_verification_pass": manifest_verification["status"]
        == "PASS_PHASE8_RELEASE_MANIFEST_VERIFICATION",
        "clean_runtime_preflight_pass": clean_runtime_preflight["status"]
        == "PASS_PHASE7_DIRECT_ARTIFACT_PREFLIGHT",
        "registered_outputs_absent": all(
            not path.exists() for path in expected_outputs.values()
        ),
        "development_hardware_exact": hardware_document()["fingerprint_sha256"]
        == protocol["development_hardware_fingerprint_sha256"],
        "teacher_absent": True,
        "training_absent": True,
        "independent_hardware_not_claimed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase8-local-clean-rehearsal-preparation/1",
        "status": "PASS_PHASE8_LOCAL_CLEAN_REHEARSAL_PREPARATION"
        if all(gates.values())
        else "FAIL_PHASE8_LOCAL_CLEAN_REHEARSAL_PREPARATION",
        "protocol_sha256": protocol_sha,
        "clean_abi_root": str(abi_clean),
        "clean_layercake_root": str(layercake_clean),
        "abi_head": _git(abi_clean, "rev-parse", "HEAD"),
        "layercake_head": _git(layercake_clean, "rev-parse", "HEAD"),
        "hardware": hardware_document(),
        "manifest_verification": manifest_verification,
        "clean_runtime_preflight": clean_runtime_preflight,
        "tracked_files_verified": verified_tracked,
        "untracked_files_copied": copied,
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "phase8_certified": False,
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def collect(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    _, abi_clean, layercake_clean = _target_roots(protocol)
    if output.exists():
        raise Phase3Error("immutable local-rehearsal collection output exists")
    collected: dict[str, dict[str, Any]] = {}
    for relative in protocol["collect_files"]:
        source = (abi_clean / relative).resolve()
        if not source.is_file():
            raise Phase3Error(f"clean rehearsal evidence missing: {relative}")
        target = (root / protocol["collection_root"] / relative).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_immutable(target, source.read_bytes())
        collected[relative] = {
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    cpu = _json(
        root
        / protocol["collection_root"]
        / protocol["clean_device_outputs"]["cpu"]
        / "result.json"
    )
    cuda = _json(
        root
        / protocol["collection_root"]
        / protocol["clean_device_outputs"]["cuda"]
        / "result.json"
    )
    gates = {
        "abi_commit_exact": _git(abi_clean, "rev-parse", "HEAD")
        == protocol["abi_packet_source_commit"],
        "layercake_commit_exact": _git(layercake_clean, "rev-parse", "HEAD")
        == protocol["layercake_commit"],
        "tracked_abi_diff_absent": not _git(abi_clean, "diff", "--name-only"),
        "tracked_layercake_diff_absent": not _git(
            layercake_clean, "diff", "--name-only"
        ),
        "cpu_result_pass": cpu.get("status") == "PASS_PHASE7_INTEGRATED_RUNTIME",
        "cuda_result_pass": cuda.get("status") == "PASS_PHASE7_INTEGRATED_RUNTIME",
        "all_registered_files_collected": len(collected)
        == len(protocol["collect_files"]),
        "development_hardware_exact": hardware_document()["fingerprint_sha256"]
        == protocol["development_hardware_fingerprint_sha256"],
        "teacher_absent": cpu.get("teacher_model_loaded") is False
        and cuda.get("teacher_model_loaded") is False,
        "training_absent": cpu.get("training_performed") is False
        and cuda.get("training_performed") is False,
        "independent_hardware_not_claimed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase8-local-clean-rehearsal-collection/1",
        "status": "PASS_PHASE8_LOCAL_CLEAN_REHEARSAL_COLLECTION"
        if all(gates.values())
        else "FAIL_PHASE8_LOCAL_CLEAN_REHEARSAL_COLLECTION",
        "protocol_sha256": protocol_sha,
        "hardware": hardware_document(),
        "collected": collected,
        "gates": gates,
        "phase8_certified": False,
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--prepare")
    parser.add_argument("--collect")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = (root / args.protocol).resolve()
    if args.preflight:
        result = preflight(root, protocol_path)
    elif args.prepare:
        result = prepare(root, protocol_path, (root / args.prepare).resolve())
    elif args.collect:
        result = collect(root, protocol_path, (root / args.collect).resolve())
    else:
        raise Phase3Error("select preflight, prepare, or collect")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
