"""Run the single preregistered exposure-balanced Phase 4 bridge stabilization."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from . import capability_compiler_phase3_final_controls as final_controls
from . import capability_compiler_phase3_route_isolated as isolated
from . import capability_compiler_phase3_sparse_router as sparse
from . import capability_compiler_phase4_abi_lineage as lineage
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_weak_residual import WEAK_CAPABILITIES


FORMAT = "abi-capability-compiler-phase4-uniform-final-bridge/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SINGLE_EXPOSURE_BALANCED_STABILIZATION"
        or protocol.get("training_device") != "cuda"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("new_teacher_information_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
    ):
        raise Phase3Error("uniform final-bridge governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"uniform final-bridge binding changed: {relative}")
    lineage_path = root / protocol["base_lineage_protocol"]
    base_lineage, _ = lineage.load_protocol(root, lineage_path)
    if sha256_file(lineage_path) != protocol["base_lineage_protocol_sha256"]:
        raise Phase3Error("base lineage protocol changed")
    registered = {(str(row["budget"]), int(row["seed"])) for row in protocol["runs"]}
    if registered != {(budget, seed) for budget in ("B40", "B80") for seed in base_lineage["seeds"]}:
        raise Phase3Error("stabilization run matrix changed")
    return protocol, sha256_file(path), base_lineage


class UniformDualViewSampler:
    """Cycle through each view stratum; recovery prefixes stay on that step's row."""

    last_instance: "UniformDualViewSampler | None" = None

    def __init__(self, rows: Sequence[Mapping[str, Any]], seed: int):
        del seed
        self.groups = {
            (capability, builder, view): sorted(
                [row for row in rows if row["capability"] == capability and int(row["builder"]) == builder and row["view"] == view],
                key=lambda row: str(row["record_id"]),
            )
            for capability in WEAK_CAPABILITIES
            for builder in range(4)
            for view in ("host_projected", "source_wrapped")
        }
        if any(not values for values in self.groups.values()):
            raise Phase3Error("uniform sampler lost a dual-view stratum")
        self.recovery_strata = tuple(self.groups)
        self.cursors = {key: 0 for key in self.recovery_strata}
        self.recovery_index = 0
        self.last_teacher: dict[tuple[str, int, str], Mapping[str, Any]] = {}
        self.exposures: Counter[str] = Counter()
        UniformDualViewSampler.last_instance = self

    def teacher_forced_batch(self) -> list[Mapping[str, Any]]:
        result = []
        for key in self.recovery_strata:
            values = self.groups[key]
            row = values[self.cursors[key] % len(values)]
            self.cursors[key] += 1
            self.last_teacher[key] = row
            self.exposures[str(row["record_id"])] += 1
            result.append(row)
        return result

    def recovery_batch(self, size: int) -> list[Mapping[str, Any]]:
        result = []
        for _ in range(size):
            key = self.recovery_strata[self.recovery_index % len(self.recovery_strata)]
            self.recovery_index += 1
            if key not in self.last_teacher:
                raise Phase3Error("recovery requested before teacher batch")
            result.append(self.last_teacher[key])
        return result

    def profile(self) -> dict[str, Any]:
        strata = {}
        for key, values in self.groups.items():
            counts = [self.exposures[str(row["record_id"])] for row in values]
            strata[":".join(map(str, key))] = {
                "records": len(values),
                "minimum_exposures": min(counts),
                "maximum_exposures": max(counts),
                "range": max(counts) - min(counts),
            }
        return {
            "ordering": "lexicographic_record_id",
            "seed_dependent_sampling": False,
            "recovery_uses_same_step_record": True,
            "strata": dict(sorted(strata.items())),
            "maximum_within_stratum_exposure_range": max(row["range"] for row in strata.values()),
        }


def _source(protocol: Mapping[str, Any], budget: str, seed: int) -> Mapping[str, Any]:
    found = [row for row in protocol["runs"] if row["budget"] == budget and int(row["seed"]) == seed]
    if len(found) != 1:
        raise Phase3Error("unregistered stabilization source")
    return found[0]


def _validate_source(root: Path, source: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    directory = root / source["source_dir"]
    result_path = directory / "result.json"
    if sha256_file(result_path) != source["source_result_sha256"]:
        raise Phase3Error("source lineage result changed")
    result = _json(result_path)
    if result.get("budget", {}).get("id") != source["budget"] or int(result.get("seed", -1)) != int(source["seed"]):
        raise Phase3Error("source lineage identity changed")
    files = {
        "v463": directory / "v463" / "model.safetensors",
        "router": directory / "router" / "router.safetensors",
        "v484": directory / "v484" / "host_recovery_bridge.safetensors",
    }
    for stage, target in files.items():
        if sha256_file(target) != result["stage_checkpoints"][stage]:
            raise Phase3Error(f"source {stage} checkpoint changed")
        if sha256_file(target.parent / "metadata.json") != result["stage_metadata_sha256"][stage]:
            raise Phase3Error(f"source {stage} metadata changed")
    if sha256_file(directory / "v463_evaluation" / "development_outputs.jsonl") != source["parent_development_outputs_sha256"]:
        raise Phase3Error("source parent development outputs changed")
    if sha256_file(directory / "budget_host_supervision.abicir") != source["host_artifact_sha256"]:
        raise Phase3Error("source host artifact changed")
    return directory, result


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, base_lineage = load_protocol(root, protocol_path)
    manifest = _json(root / base_lineage["budget_manifest"])
    runs = []
    for source in protocol["runs"]:
        directory, result = _validate_source(root, source)
        selected, budget = lineage._selected_rows(root, base_lineage, manifest, source["budget"])
        if budget["selection_sha256"] != result["selection_sha256"]:
            raise Phase3Error("source selection changed")
        per_stratum = len(selected["v480_host_supervision"]) // 16
        runs.append({
            "budget": source["budget"], "seed": source["seed"], "source_dir": str(directory.relative_to(root)).replace("\\", "/"),
            "host_records": len(selected["v480_host_supervision"]), "records_per_view_stratum": per_stratum,
            "teacher_exposure_floor": 2000 // per_stratum, "teacher_exposure_ceiling": (2000 + per_stratum - 1) // per_stratum,
        })
    return {
        "status": "PASS_UNIFORM_FINAL_BRIDGE_PREFLIGHT", "protocol_sha256": protocol_sha, "runs": runs,
        "same_information_budget": True, "same_architecture": True, "same_steps": 2000,
        "maximum_expected_within_stratum_exposure_range": 1, "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, budget: str, seed: int, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, base_lineage = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable stabilization output exists or CUDA unavailable")
    source = _source(protocol, budget, seed)
    source_dir, source_result = _validate_source(root, source)
    manifest = _json(root / base_lineage["budget_manifest"])
    selected, budget_spec = lineage._selected_rows(root, base_lineage, manifest, budget)
    if budget_spec["selection_sha256"] != source_result["selection_sha256"]:
        raise Phase3Error("source and budget selection differ")

    v440 = _json(root / base_lineage["base_protocols"]["v443"])
    v440["training"]["seed"] = seed
    v463 = source_dir / "v463"
    parent_evaluation = source_dir / "v463_evaluation" / "development_outputs.jsonl"
    router_dir = source_dir / "router"
    v484 = source_dir / "v484"
    host_artifact = source_dir / "budget_host_supervision.abicir"

    def load_v463(device: torch.device):
        return lineage._load_candidate(root, v440, v463, device)

    router_protocol = _json(root / base_lineage["base_protocols"]["router"])
    router_protocol["training"]["seed"] = seed

    base = _json(root / base_lineage["base_protocols"]["v526_base"])
    base["training"]["seed"] = seed
    base["parent"]["checkpoint_sha256"] = source_result["stage_checkpoints"]["v463"]
    base["parent"]["development_outputs"] = str(parent_evaluation.relative_to(root)).replace("\\", "/")
    base["initialization"]["checkpoint"] = str((v484 / "host_recovery_bridge.safetensors").relative_to(root)).replace("\\", "/")
    base["initialization"]["checkpoint_sha256"] = source_result["stage_checkpoints"]["v484"]
    base["supervision"]["artifact"] = str(host_artifact.relative_to(root)).replace("\\", "/")
    control = _json(root / base_lineage["base_protocols"]["v526_control"])
    control["guard"]["artifact"] = base["supervision"]["artifact"]
    control["systems"] = [isolated.SYSTEM]
    stage_sha = hashlib.sha256(canonical_json_bytes({
        "stage": "uniform_final_bridge", "protocol_sha256": protocol_sha, "budget": budget, "seed": seed,
        "training": base["training"], "source_checkpoints": {key: source_result["stage_checkpoints"][key] for key in ("v463", "router", "v484")},
    })).hexdigest()

    def load_router(*_: Any):
        return (*sparse._load(root, router_protocol, router_dir), router_protocol)

    started = time.perf_counter()
    UniformDualViewSampler.last_instance = None
    with lineage._patch(
        final_controls,
        SharedWeakResidual=isolated.RouteIsolatedResidual,
        EXPECTED_PARAMETERS=isolated.PARAMETERS,
        SYSTEMS=(isolated.SYSTEM,),
        load_protocol=lambda *_: (control, stage_sha, base),
        _load_parent=lambda _root, _protocol, device: load_v463(device),
        _load_router=load_router,
        _artifact_rows=lambda *_: selected["v480_host_supervision"],
        DualViewSampler=UniformDualViewSampler,
    ):
        training = final_controls.train(root, protocol_path, isolated.SYSTEM, output / "v526")
        evaluation = final_controls.evaluate(root, protocol_path, isolated.SYSTEM, output / "v526", output / "evaluation")
    sampler = UniformDualViewSampler.last_instance
    if sampler is None:
        raise Phase3Error("uniform sampler was not instantiated")
    exposure = sampler.profile()
    if exposure["maximum_within_stratum_exposure_range"] > 1:
        raise Phase3Error("uniform exposure invariant failed")
    gates, relative = lineage._evaluate_gates(root, base, evaluation, output / "evaluation" / "development_outputs.jsonl", seed + 4_100_000)
    result = {
        "format": "abi-capability-compiler-phase4-uniform-final-bridge-result/1",
        "status": "PASS_PHASE4_STABILIZED_ABI_BUDGET_MACHINE_GATES" if all(gates.values()) else "FAIL_PHASE4_STABILIZED_ABI_BUDGET_MACHINE_GATES",
        "protocol_sha256": protocol_sha, "budget": budget_spec, "seed": seed,
        "source_lineage": {"path": source["source_dir"], "result_sha256": source["source_result_sha256"], "reused_stages": ["v463", "router", "v484", "budget_host_supervision"]},
        "changed_component": "final_v526_sampler_only", "new_teacher_information": 0,
        "architecture_changed": False, "steps_changed": False, "deployed_parameter_count_changed": False,
        "checkpoint": training["checkpoint"], "training_metadata_sha256": sha256_file(output / "v526" / "metadata.json"),
        "functional_passes_v1": evaluation["functional_passes_v1"], "repetition_collapses_v2": evaluation["repetition_collapses_v2"],
        "router_correct": evaluation["router_correct"], "strong_routes_exact": evaluation["strong_routes_exact"],
        "teacher_comparison_v1": relative, "gates": gates, "exposure_balance": exposure,
        "wall_seconds": time.perf_counter() - started,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "teacher_present_at_inference": False, "final_test_accessed": False, "phase4_certified": False,
        "claim_boundary": "Single preregistered final-bridge stabilization run; no stable frontier, minimum, runtime, matched-baseline, final-test, certificate, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("train")
    run.add_argument("--budget", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = preflight(root, root / args.protocol) if args.command == "preflight" else train(root, root / args.protocol, args.budget, args.seed, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
