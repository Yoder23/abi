"""Evaluate preregistered B80 cross-seed parent/bridge compatibility pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
from typing import Any, Iterable, Mapping

import torch

from . import capability_compiler_phase3_final_controls as final_controls
from . import capability_compiler_phase3_route_isolated as isolated
from . import capability_compiler_phase3_sparse_router as sparse
from . import capability_compiler_phase4_abi_lineage as lineage
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-b80-compatibility/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_B80_PARENT_BRIDGE_MATRIX"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_construction_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B80 compatibility governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B80 compatibility binding changed: {relative}")
    base_path = root / protocol["base_lineage_protocol"]
    base, _ = lineage.load_protocol(root, base_path)
    if sha256_file(base_path) != protocol["base_lineage_protocol_sha256"]:
        raise Phase3Error("B80 compatibility base lineage changed")
    return protocol, sha256_file(path), base


def _source(protocol: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    found = [row for row in protocol["sources"] if int(row["seed"]) == seed]
    if len(found) != 1:
        raise Phase3Error("unregistered compatibility seed")
    return found[0]


def _validate_source(root: Path, source: Mapping[str, Any]) -> None:
    for component in ("parent", "router", "bridge", "parent_outputs", "guard_artifact"):
        item = source[component]
        if sha256_file(root / item["path"]) != item["sha256"]:
            raise Phase3Error(f"compatibility {component} source changed")


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, _ = load_protocol(root, protocol_path)
    for source in protocol["sources"]:
        _validate_source(root, source)
    seeds = sorted(int(row["seed"]) for row in protocol["sources"])
    pairs = [(parent, bridge) for parent in seeds for bridge in seeds]
    off_diagonal = [(parent, bridge) for parent, bridge in pairs if parent != bridge]
    if len(pairs) != 9 or len(off_diagonal) != 6:
        raise Phase3Error("compatibility matrix depth changed")
    return {"status": "PASS_B80_COMPATIBILITY_PREFLIGHT", "protocol_sha256": protocol_sha, "seeds": seeds, "matrix_cells": 9, "existing_diagonal_cells": 3, "authorized_off_diagonal_cells": 6, "training_performed": False, "model_construction_performed": False, "final_test_accessed": False}


def evaluate_pair(root: Path, protocol_path: Path, parent_seed: int, bridge_seed: int, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if parent_seed == bridge_seed or output.exists() or not torch.cuda.is_available():
        raise Phase3Error("only immutable off-diagonal CUDA evaluations are authorized")
    parent = _source(protocol, parent_seed)
    bridge = _source(protocol, bridge_seed)
    _validate_source(root, parent)
    _validate_source(root, bridge)
    output.mkdir(parents=True)
    candidate = output / "candidate"
    candidate.mkdir()
    checkpoint = candidate / "control_bridge.safetensors"
    shutil.copyfile(root / bridge["bridge"]["path"], checkpoint)

    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    v440["training"]["seed"] = parent_seed
    parent_dir = (root / parent["parent"]["path"]).parent
    router_dir = (root / parent["router"]["path"]).parent
    router_protocol = _json(root / lineage_protocol["base_protocols"]["router"])
    router_protocol["training"]["seed"] = parent_seed
    base = _json(root / lineage_protocol["base_protocols"]["v526_base"])
    base["training"]["seed"] = parent_seed
    base["parent"]["checkpoint_sha256"] = parent["parent"]["sha256"]
    base["parent"]["development_outputs"] = parent["parent_outputs"]["path"]
    base["supervision"]["artifact"] = parent["guard_artifact"]["path"]
    control = _json(root / lineage_protocol["base_protocols"]["v526_control"])
    control["guard"]["artifact"] = parent["guard_artifact"]["path"]
    control["systems"] = [isolated.SYSTEM]
    stage_sha = hashlib.sha256(canonical_json_bytes({"protocol_sha256": protocol_sha, "parent_seed": parent_seed, "bridge_seed": bridge_seed, "parent_sha256": parent["parent"]["sha256"], "bridge_sha256": bridge["bridge"]["sha256"]})).hexdigest()
    metadata = {"format": FORMAT, "status": "EXISTING_BRIDGE_COPIED_FOR_READ_ONLY_COMPATIBILITY_EVALUATION", "system": isolated.SYSTEM, "protocol_sha256": stage_sha, "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size}, "training_performed": False, "source_bridge_seed": bridge_seed, "final_test_accessed": False}
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(candidate / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")

    def load_parent(device: torch.device):
        return lineage._load_candidate(root, v440, parent_dir, device)

    def load_router(*_: Any):
        return (*sparse._load(root, router_protocol, router_dir), router_protocol)

    with lineage._patch(final_controls, SharedWeakResidual=isolated.RouteIsolatedResidual, EXPECTED_PARAMETERS=isolated.PARAMETERS, SYSTEMS=(isolated.SYSTEM,), load_protocol=lambda *_: (control, stage_sha, base), _load_parent=lambda _root, _protocol, device: load_parent(device), _load_router=load_router):
        evaluation = final_controls.evaluate(root, protocol_path, isolated.SYSTEM, candidate, output / "evaluation")
    gates, relative = lineage._evaluate_gates(root, base, evaluation, output / "evaluation" / "development_outputs.jsonl", int(protocol["bootstrap_seed_base"]) + parent_seed + bridge_seed)
    result = {
        "format": "abi-capability-compiler-phase4-b80-compatibility-result/1",
        "status": "PASS_B80_CROSS_SEED_MACHINE_GATES" if all(gates.values()) else "FAIL_B80_CROSS_SEED_MACHINE_GATES",
        "protocol_sha256": protocol_sha, "parent_seed": parent_seed, "bridge_seed": bridge_seed,
        "parent_checkpoint_sha256": parent["parent"]["sha256"], "router_checkpoint_sha256": parent["router"]["sha256"], "bridge_checkpoint_sha256": bridge["bridge"]["sha256"],
        "functional_passes_v1": evaluation["functional_passes_v1"], "repetition_collapses_v2": evaluation["repetition_collapses_v2"],
        "router_correct": evaluation["router_correct"], "strong_routes_exact": evaluation["strong_routes_exact"], "teacher_comparison_v1": relative, "gates": gates,
        "training_performed": False, "model_construction_performed": False, "new_teacher_information": 0, "teacher_present_at_inference": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "final_test_accessed": False, "phase4_certified": False,
        "claim_boundary": "One read-only B80 parent/bridge compatibility cell; no candidate promotion, minimum, runtime, matched-baseline, final-test, Phase 4 certificate, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("evaluate")
    run.add_argument("--parent-seed", type=int, required=True)
    run.add_argument("--bridge-seed", type=int, required=True)
    run.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = preflight(root, root / args.protocol) if args.command == "preflight" else evaluate_pair(root, root / args.protocol, args.parent_seed, args.bridge_seed, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
