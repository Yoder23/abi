"""Single physically route-isolated rank-16 Phase 3 successor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from torch import nn

from . import capability_compiler_phase3_final_controls as controls
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap


FORMAT = "abi-capability-compiler-phase3-route-isolated/1"
RANK = 16
ROUTES = 4
WIDTH = 768
PARAMETERS = 99_840
SYSTEM = "A0_route_isolated"
CONTROL_SYSTEMS = ("A1_label_free", "A2_shuffled", "A3_bridge_only", "A4_monolithic")


class RouteIsolatedResidual(nn.Module):
    """Four disjoint experts; a token executes exactly one rank-16 expert."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.down = nn.Parameter(torch.empty(ROUTES, RANK, WIDTH))
        self.up = nn.Parameter(torch.empty(ROUTES, WIDTH, RANK))
        nn.init.normal_(self.down, mean=0.0, std=0.02)
        nn.init.zeros_(self.up)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):
        if "down.weight" in state_dict:
            mapped = {
                "norm.weight": state_dict["norm.weight"],
                "norm.bias": state_dict["norm.bias"],
                "down": torch.stack([state_dict["down.weight"][route * RANK:(route + 1) * RANK] for route in range(ROUTES)]),
                "up": torch.stack([state_dict["up.weight"][:, route * RANK:(route + 1) * RANK] for route in range(ROUTES)]),
            }
            return super().load_state_dict(mapped, strict=True, assign=assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def delta(self, hidden: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden)
        down = self.down.index_select(0, routes)
        up = self.up.index_select(0, routes)
        low = torch.einsum("bsw,brw->bsr", normalized, down)
        return torch.einsum("bsr,bwr->bsw", F.silu(low), up)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, tuple[dict[str, Any], dict[str, Any]]]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") not in {"PREREGISTERED_SINGLE_ROUTE_ISOLATED_SUCCESSOR", "PREREGISTERED_ROUTE_ISOLATED_MATCHED_CONTROLS", "PREREGISTERED_ROUTE_ISOLATED_PAIRED_SEED_MATRIX"} or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("nearby_sweeps_authorized") is not False:
        raise Phase3Error("route-isolated governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"route-isolated binding changed: {relative}")
    control_protocol, _, base = controls.load_protocol(root, root / protocol["base_control_protocol"])
    control_protocol = copy.deepcopy(control_protocol)
    if protocol["status"] in {"PREREGISTERED_ROUTE_ISOLATED_MATCHED_CONTROLS", "PREREGISTERED_ROUTE_ISOLATED_PAIRED_SEED_MATRIX"}:
        expected = CONTROL_SYSTEMS if protocol["status"] == "PREREGISTERED_ROUTE_ISOLATED_MATCHED_CONTROLS" else (SYSTEM, *CONTROL_SYSTEMS)
        if tuple(protocol.get("systems", ())) != expected:
            raise Phase3Error("route-isolated control systems changed")
        control_protocol["control_outputs"] = copy.deepcopy(protocol["control_outputs"])
        control_protocol["A0_outputs"] = protocol["A0_outputs"]
        control_protocol["A0_checkpoint_sha256"] = protocol["A0_checkpoint_sha256"]
        if protocol["status"] == "PREREGISTERED_ROUTE_ISOLATED_PAIRED_SEED_MATRIX":
            base = copy.deepcopy(base)
            base["training"]["seed"] = int(protocol["training_seed"])
    else:
        control_protocol["control_outputs"] = {SYSTEM: protocol["evaluation_output"]}
    return protocol, sha256_file(path), (control_protocol, base)


def _patch(protocol_sha: str, bundle: tuple[dict[str, Any], dict[str, Any]], systems: tuple[str, ...]):
    control_protocol, base = bundle
    old = (controls.SharedWeakResidual, controls.EXPECTED_PARAMETERS, controls.SYSTEMS, controls.load_protocol)
    controls.SharedWeakResidual = RouteIsolatedResidual
    controls.EXPECTED_PARAMETERS = PARAMETERS
    controls.SYSTEMS = systems
    controls.load_protocol = lambda root, path: (control_protocol, protocol_sha, base)
    return old


def _restore(old) -> None:
    controls.SharedWeakResidual, controls.EXPECTED_PARAMETERS, controls.SYSTEMS, controls.load_protocol = old


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, bundle = load_protocol(root, protocol_path)
    _, base = bundle
    residual = RouteIsolatedResidual()
    initialization = root / base["initialization"]["checkpoint"]
    residual.load_state_dict(load_file(str(initialization), device="cpu"), strict=True)
    count = sum(value.numel() for value in residual.parameters())
    if count != PARAMETERS:
        raise Phase3Error("route-isolated parameter count changed")
    return {"status": "PASS_PREFLIGHT", "protocol_sha256": protocol_sha, "parameters": count, "total_parameter_ratio_to_V488": count / 100352, "active_rank": RANK, "current_active_rank": 64, "active_rank_ratio": RANK / 64, "physical_experts": ROUTES, "shared_trainable_parameters": int(residual.norm.weight.numel() + residual.norm.bias.numel()), "final_test_accessed": False}


def train(root: Path, protocol_path: Path, system: str, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, bundle = load_protocol(root, protocol_path)
    allowed = (SYSTEM,) if protocol["status"] == "PREREGISTERED_SINGLE_ROUTE_ISOLATED_SUCCESSOR" else CONTROL_SYSTEMS if protocol["status"] == "PREREGISTERED_ROUTE_ISOLATED_MATCHED_CONTROLS" else (SYSTEM, *CONTROL_SYSTEMS)
    if system not in allowed:
        raise Phase3Error("route-isolated system is not authorized")
    old = _patch(protocol_sha, bundle, allowed)
    try:
        return controls.train(root, protocol_path, system, output)
    finally:
        _restore(old)


def evaluate(root: Path, protocol_path: Path, system: str, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, bundle = load_protocol(root, protocol_path)
    allowed = (SYSTEM,) if protocol["status"] == "PREREGISTERED_SINGLE_ROUTE_ISOLATED_SUCCESSOR" else CONTROL_SYSTEMS if protocol["status"] == "PREREGISTERED_ROUTE_ISOLATED_MATCHED_CONTROLS" else (SYSTEM, *CONTROL_SYSTEMS)
    if system not in allowed:
        raise Phase3Error("route-isolated system is not authorized")
    old = _patch(protocol_sha, bundle, allowed)
    try:
        return controls.evaluate(root, protocol_path, system, candidate, output)
    finally:
        _restore(old)


def decide(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, bundle = load_protocol(root, protocol_path)
    _, base = bundle
    if output.exists():
        raise Phase3Error("immutable route-isolated decision exists")
    evaluation = _json(root / protocol["evaluation_output"] / "result.json")
    rows = [json.loads(line) for line in (root / protocol["evaluation_output"] / "development_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    probes = {str(row["probe_id"]): row for row in development_probes(root / base["development"]["catalog_path"])}
    teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / base["development"]["teacher_reference"]).read_text(encoding="utf-8").splitlines())}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass_v1"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probes[row["probe_id"]]["evaluator"])} for row in rows]
    relative = paired_stratified_bootstrap(paired, replicates=10_000, seed=5251729)
    thresholds = base["absolute_screen"]
    gates = {
        "per_capability": all(value["wilson_v1"]["point"] >= thresholds["per_capability_functional_point_estimate_minimum"] and value["wilson_v1"]["lower_95"] >= thresholds["per_capability_functional_wilson_lower_minimum"] for value in evaluation["per_capability"].values()),
        "critical": all(evaluation["per_capability"][name]["wilson_v1"]["point"] >= thresholds["critical_point_minimum"] and evaluation["per_capability"][name]["wilson_v1"]["lower_95"] >= thresholds["critical_wilson_lower_minimum"] for name in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_collapse": evaluation["repetition_collapses_v2"] == 0,
        "router_exact": evaluation["router_correct"] == 1400,
        "strong_exact": evaluation["strong_routes_exact"] == 1000,
        "teacher_noninferior": relative["lower_95"] >= base["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"],
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {"format": "abi-capability-compiler-phase3-route-isolated-decision/1", "status": "PASS_ROUTE_ISOLATED_A0_ABSOLUTE_SCREEN_CONTROLS_OPEN" if passed else "FAIL_ROUTE_ISOLATED_A0_ABSOLUTE_SCREEN_CLOSED", "protocol_sha256": protocol_sha, "checkpoint_sha256": evaluation["checkpoint_sha256"], "parameters": PARAMETERS, "active_rank": RANK, "functional_passes_v1": evaluation["functional_passes_v1"], "teacher_comparison": relative, "gates": gates, "controls_authorized": passed, "final_test_accessed": False}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def decide_controls(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, bundle = load_protocol(root, protocol_path)
    if protocol["status"] not in {"PREREGISTERED_ROUTE_ISOLATED_MATCHED_CONTROLS", "PREREGISTERED_ROUTE_ISOLATED_PAIRED_SEED_MATRIX"}:
        raise Phase3Error("matched-control decision is not authorized")
    systems = CONTROL_SYSTEMS if protocol["status"] == "PREREGISTERED_ROUTE_ISOLATED_MATCHED_CONTROLS" else (SYSTEM, *CONTROL_SYSTEMS)
    if protocol["status"] == "PREREGISTERED_ROUTE_ISOLATED_PAIRED_SEED_MATRIX":
        control_protocol, base = bundle
        control_protocol = copy.deepcopy(control_protocol)
        metadata = _json(root / protocol["A0_metadata"])
        control_protocol["A0_checkpoint_sha256"] = metadata["checkpoint"]["sha256"]
        bundle = (control_protocol, base)
    old = _patch(protocol_sha, bundle, systems)
    try:
        return controls.decide(root, protocol_path, output)
    finally:
        _restore(old)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); p=sub.add_parser("train"); p.add_argument("--system", default=SYSTEM, choices=(SYSTEM, *CONTROL_SYSTEMS)); p.add_argument("--output-dir", required=True); p=sub.add_parser("evaluate"); p.add_argument("--system", default=SYSTEM, choices=(SYSTEM, *CONTROL_SYSTEMS)); p.add_argument("--candidate-dir", required=True); p.add_argument("--output-dir", required=True); p=sub.add_parser("decide"); p.add_argument("--output", required=True); p=sub.add_parser("decide-controls"); p.add_argument("--output", required=True); args=parser.parse_args(argv); root=Path.cwd().resolve(); protocol=root/args.protocol
    result = preflight(root, protocol) if args.command=="preflight" else train(root, protocol, args.system, root/args.output_dir) if args.command=="train" else evaluate(root, protocol, args.system, root/args.candidate_dir, root/args.output_dir) if args.command=="evaluate" else decide(root, protocol, root/args.output) if args.command=="decide" else decide_controls(root, protocol, root/args.output)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
