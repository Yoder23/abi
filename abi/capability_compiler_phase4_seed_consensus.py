"""Build and evaluate one equal-weight consensus of aligned Phase 4 seed states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch

from . import capability_compiler_phase3_copy_balanced_transition as balanced
from . import capability_compiler_phase3_final_controls as final_controls
from . import capability_compiler_phase3_route_isolated as isolated
from . import capability_compiler_phase3_sparse_router as sparse
from . import capability_compiler_phase4_abi_lineage as lineage
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-seed-consensus/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EQUAL_WEIGHT_ALIGNED_SEED_CONSENSUS"
        or protocol.get("training_authorized") is not False
        or protocol.get("coefficient_search_authorized") is not False
        or protocol.get("new_teacher_information_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("seed-consensus governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"seed-consensus binding changed: {relative}")
    base_path = root / protocol["base_lineage_protocol"]
    base, _ = lineage.load_protocol(root, base_path)
    if sha256_file(base_path) != protocol["base_lineage_protocol_sha256"]:
        raise Phase3Error("seed-consensus base lineage changed")
    return protocol, sha256_file(path), base


def mean_states(paths: Sequence[Path]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if len(paths) != 3:
        raise Phase3Error("consensus requires exactly three states")
    states = [load_file(str(path), device="cpu") for path in paths]
    keys = tuple(sorted(states[0]))
    if any(tuple(sorted(state)) != keys for state in states[1:]):
        raise Phase3Error("consensus state keys are not aligned")
    result: dict[str, torch.Tensor] = {}
    maximum_source_delta = 0.0
    for key in keys:
        values = [state[key] for state in states]
        if any(value.shape != values[0].shape or value.dtype != values[0].dtype for value in values[1:]):
            raise Phase3Error("consensus tensor schema is not aligned")
        if not values[0].is_floating_point():
            if any(not torch.equal(value, values[0]) for value in values[1:]):
                raise Phase3Error("nonfloating consensus tensor differs")
            result[key] = values[0].contiguous()
            continue
        accumulator = torch.zeros_like(values[0], dtype=torch.float64)
        for value in values:
            accumulator.add_(value.to(torch.float64))
        averaged = (accumulator / 3.0).to(values[0].dtype).contiguous()
        result[key] = averaged
        maximum_source_delta = max(maximum_source_delta, max(float((value - averaged).abs().max()) for value in values))
    return result, {"tensors": len(keys), "parameters": sum(value.numel() for value in result.values()), "accumulation_dtype": "float64", "output_dtype": str(next(iter(result.values())).dtype), "maximum_absolute_source_to_consensus_delta": maximum_source_delta}


def _budget(protocol: Mapping[str, Any], budget: str) -> Mapping[str, Any]:
    found = [row for row in protocol["budgets"] if row["id"] == budget]
    if len(found) != 1:
        raise Phase3Error("unregistered consensus budget")
    return found[0]


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, _ = load_protocol(root, protocol_path)
    checks = []
    for budget in protocol["budgets"]:
        for component in ("parent", "router", "bridge"):
            paths = [root / row[component]["path"] for row in budget["sources"]]
            for path, row in zip(paths, budget["sources"]):
                if sha256_file(path) != row[component]["sha256"]:
                    raise Phase3Error("consensus source checkpoint changed")
            states = [load_file(str(path), device="cpu") for path in paths]
            keys = tuple(sorted(states[0]))
            if any(tuple(sorted(state)) != keys for state in states[1:]):
                raise Phase3Error("consensus source keys differ")
            checks.append({"budget": budget["id"], "component": component, "tensors": len(keys), "parameters": sum(value.numel() for value in states[0].values()), "schema_aligned": True})
    return {"status": "PASS_SEED_CONSENSUS_PREFLIGHT", "protocol_sha256": protocol_sha, "checks": checks, "weights": [1 / 3, 1 / 3, 1 / 3], "training_performed": False, "new_teacher_information": 0, "final_test_accessed": False}


@torch.inference_mode()
def _parent_outputs(root: Path, base: Mapping[str, Any], parent_dir: Path, output: Path) -> dict[str, Any]:
    device = torch.device("cuda")
    model, tokenizer, _ = lineage._load_candidate(root, base, parent_dir, device)
    model.eval()
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(development_probes(root / base["development"]["catalog_path"])):
        value, tokens, route = balanced._generate(model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), device)
        rows.append({"probe_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]), "output": value, "output_token_ids": tokens, "automatic_route": route})
        if (index + 1) % 100 == 0:
            print(json.dumps({"consensus_parent_evaluated": index + 1}), flush=True)
    output.mkdir(parents=True)
    raw = output / "development_outputs.jsonl"
    _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in rows))
    result = {"format": "abi-capability-compiler-phase4-consensus-parent-evaluation/1", "status": "COMPLETE_DEVELOPMENT_ONLY", "observations": len(rows), "raw_outputs_sha256": sha256_file(raw), "wall_seconds": time.perf_counter() - started, "final_test_accessed": False}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def build_and_evaluate(root: Path, protocol_path: Path, budget_id: str, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable consensus output exists or CUDA unavailable")
    budget = _budget(protocol, budget_id)
    output.mkdir(parents=True)
    receipts = {}
    for component, filename in (("parent", "model.safetensors"), ("router", "router.safetensors"), ("bridge", "control_bridge.safetensors")):
        paths = [root / row[component]["path"] for row in budget["sources"]]
        states, arithmetic = mean_states(paths)
        directory = output / component
        directory.mkdir()
        checkpoint = directory / filename
        save_file(states, str(checkpoint), metadata={"format": FORMAT, "component": component, "weights": "1/3,1/3,1/3"})
        receipts[component] = {"checkpoint": {"path": str(checkpoint.relative_to(output)).replace("\\", "/"), "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size}, "arithmetic": arithmetic, "sources": [{"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(path)} for path in paths]}

    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    v440["training"]["seed"] = int(protocol["construction_seed_label"])
    parent_sha = receipts["parent"]["checkpoint"]["sha256"]
    _write_immutable(output / "parent" / "metadata.json", json.dumps({"format": FORMAT, "protocol_sha256": protocol_sha, "checkpoint": receipts["parent"]["checkpoint"], "training_performed": False}, indent=2, sort_keys=True).encode() + b"\n")
    parent_evaluation = _parent_outputs(root, v440, output / "parent", output / "parent_evaluation")

    router_protocol = _json(root / lineage_protocol["base_protocols"]["router"])
    router_protocol["training"]["seed"] = int(protocol["construction_seed_label"])
    _write_immutable(output / "router" / "config.json", json.dumps({"vocabulary": router_protocol["representation"].get("bpe_embedding_buckets"), **router_protocol["representation"]}, indent=2, sort_keys=True).encode() + b"\n")

    base = _json(root / lineage_protocol["base_protocols"]["v526_base"])
    base["training"]["seed"] = int(protocol["evaluation_bootstrap_seed"])
    base["parent"]["checkpoint_sha256"] = parent_sha
    base["parent"]["development_outputs"] = str((output / "parent_evaluation" / "development_outputs.jsonl").relative_to(root)).replace("\\", "/")
    base["supervision"]["artifact"] = budget["guard_artifact"]
    control = _json(root / lineage_protocol["base_protocols"]["v526_control"])
    control["guard"]["artifact"] = budget["guard_artifact"]
    control["systems"] = [isolated.SYSTEM]
    stage_sha = hashlib.sha256(canonical_json_bytes({"protocol_sha256": protocol_sha, "budget": budget_id, "receipts": receipts})).hexdigest()
    bridge_checkpoint = output / "bridge" / "control_bridge.safetensors"
    bridge_metadata = {"format": FORMAT, "status": "BUILT_EQUAL_WEIGHT_CONSENSUS_NO_TRAINING", "system": isolated.SYSTEM, "protocol_sha256": stage_sha, "checkpoint": {"path": bridge_checkpoint.name, "sha256": sha256_file(bridge_checkpoint), "bytes": bridge_checkpoint.stat().st_size}, "teacher_present": False, "training_performed": False, "final_test_accessed": False}
    bridge_metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(bridge_metadata)).hexdigest()
    _write_immutable(output / "bridge" / "metadata.json", json.dumps(bridge_metadata, indent=2, sort_keys=True).encode() + b"\n")

    def load_parent(device: torch.device):
        return lineage._load_candidate(root, v440, output / "parent", device)

    def load_router(*_: Any):
        return (*sparse._load(root, router_protocol, output / "router"), router_protocol)

    started = time.perf_counter()
    with lineage._patch(final_controls, SharedWeakResidual=isolated.RouteIsolatedResidual, EXPECTED_PARAMETERS=isolated.PARAMETERS, SYSTEMS=(isolated.SYSTEM,), load_protocol=lambda *_: (control, stage_sha, base), _load_parent=lambda _root, _protocol, device: load_parent(device), _load_router=load_router):
        evaluation = final_controls.evaluate(root, protocol_path, isolated.SYSTEM, output / "bridge", output / "evaluation")
    gates, relative = lineage._evaluate_gates(root, base, evaluation, output / "evaluation" / "development_outputs.jsonl", int(protocol["evaluation_bootstrap_seed"]))
    result = {
        "format": "abi-capability-compiler-phase4-seed-consensus-result/1",
        "status": "PASS_PHASE4_SEED_CONSENSUS_MACHINE_GATES" if all(gates.values()) else "FAIL_PHASE4_SEED_CONSENSUS_MACHINE_GATES",
        "protocol_sha256": protocol_sha, "budget": budget_id, "weights": [1 / 3, 1 / 3, 1 / 3], "components": receipts,
        "parent_evaluation": parent_evaluation, "functional_passes_v1": evaluation["functional_passes_v1"], "repetition_collapses_v2": evaluation["repetition_collapses_v2"],
        "router_correct": evaluation["router_correct"], "strong_routes_exact": evaluation["strong_routes_exact"], "teacher_comparison_v1": relative, "gates": gates,
        "training_performed": False, "new_teacher_information": 0, "coefficient_search_performed": False, "architecture_changed": False,
        "single_deployed_checkpoint": True, "source_parameters_copied": 0, "teacher_present_at_inference": False,
        "wall_seconds_excluding_construction_and_parent_evaluation": time.perf_counter() - started,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "final_test_accessed": False, "phase4_certified": False,
        "claim_boundary": "One preregistered no-training seed-consensus development result; no stable frontier, minimum, runtime, matched-baseline, final-test, Phase 4 certificate, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("build-evaluate")
    run.add_argument("--budget", required=True)
    run.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = preflight(root, root / args.protocol) if args.command == "preflight" else build_and_evaluate(root, root / args.protocol, args.budget, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
