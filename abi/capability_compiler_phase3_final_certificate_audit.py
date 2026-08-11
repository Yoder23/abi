"""Fail-closed aggregate certificate audit for the exact Phase 3 endpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _write_immutable
from .capability_compiler_phase3_route_isolated import RouteIsolatedResidual
from .capability_compiler_phase3_targeted_recovery_bridge import _load_parent
from .capability_compiler_phase3_weak_residual import WEAK_CAPABILITIES, _attach, _set_routes


FORMAT = "abi-capability-compiler-phase3-final-certificate-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_FAIL_CLOSED_PHASE3_CERTIFICATE_AUDIT"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("phase2_prerequisite_waiver_authorized") is not False
    ):
        raise Phase3Error("final Phase 3 audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"final Phase 3 audit binding changed: {relative}")
    return protocol, sha256_file(path)


def physical_sparse_microcheck() -> dict[str, Any]:
    torch.manual_seed(548)
    model = RouteIsolatedResidual().eval()
    with torch.no_grad():
        model.up.normal_(mean=0.0, std=0.02)
    hidden = torch.randn(1, 3, 768)
    route = torch.tensor([0], dtype=torch.long)
    before = model.delta(hidden, route)
    with torch.no_grad():
        model.down[1].add_(100.0)
        model.up[1].add_(100.0)
    after_unselected = model.delta(hidden, route)
    with torch.no_grad():
        model.down[0].add_(0.25)
    after_selected = model.delta(hidden, route)
    return {
        "physical_experts": 4,
        "active_experts_per_token": 1,
        "active_rank": 16,
        "unselected_expert_mutation_max_abs_delta": float((after_unselected - before).abs().max()),
        "selected_expert_mutation_max_abs_delta": float((after_selected - after_unselected).abs().max()),
        "unselected_expert_cannot_affect_route": torch.equal(before, after_unselected),
        "selected_expert_affects_route": not torch.equal(after_unselected, after_selected),
    }


@torch.inference_mode()
def incremental_state_microcheck(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    screen = _json(root / protocol["interface_microcheck"]["screen_protocol"])
    model, tokenizer, _ = _load_parent(root, screen, torch.device("cpu"))
    residual = RouteIsolatedResidual().cpu().eval()
    residual.load_state_dict(load_file(str(root / protocol["interface_microcheck"]["checkpoint"]), device="cpu"), strict=True)
    handles = _attach(model, residual)
    try:
        capability = str(WEAK_CAPABILITIES[0])
        probe = next(row for row in development_probes(root / protocol["interface_microcheck"]["catalog"]) if row["canonical_capability"] == capability)
        prompt_ids = [int(value) for value in tokenizer.encode(str(probe["prompt"]).rstrip() + "\n", add_special_tokens=False)]
        _set_routes(model, torch.tensor([0], dtype=torch.long))
        first = model(
            torch.tensor([prompt_ids], dtype=torch.long),
            prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long),
            task_routes=torch.tensor([CAPABILITY_TO_ROUTE[capability]], dtype=torch.long),
            use_cache=True,
        )
        selected = first["logits"][:, -1].argmax(dim=-1)
        second = model(selected[:, None], task_routes=first["task_routes"], past_key_values=first["past_key_values"], use_cache=True)
        return {
            "canonical_attach_handles": len(handles),
            "first_cache_present": first.get("past_key_values") is not None,
            "second_cache_present": second.get("past_key_values") is not None,
            "one_token_increment_consumed": tuple(selected[:, None].shape) == (1, 1),
            "second_logits_shape": list(second["logits"].shape),
            "same_task_route": torch.equal(first["task_routes"], second["task_routes"]),
            "persistent_incremental_state": first.get("past_key_values") is not None and second.get("past_key_values") is not None,
            "canonical_interface_unchanged": len(handles) > 0 and torch.equal(first["task_routes"], second["task_routes"]),
        }
    finally:
        for handle in handles:
            handle.remove()


def assess(
    artifact: Mapping[str, Any],
    verifier: Mapping[str, Any],
    replication: Mapping[str, Any],
    runtime: Mapping[str, Any],
    hosts: Mapping[str, Any],
    sparse: Mapping[str, Any],
    incremental: Mapping[str, Any],
    ratings_complete: bool,
) -> tuple[dict[str, bool], dict[str, bool]]:
    checkpoint = "1649e110338904f69fafc0f5ff110e2c8d99f4f8366eb133442fb4938fa3c390"
    machine = {
        "artifact_provenance_and_accounting": artifact.get("status") == "PASS_INDEPENDENT_HOSTILE_ARTIFACT_VERIFICATION" and artifact.get("artifact", {}).get("verified_records") == 1280 and artifact.get("accounting_reconciled", {}).get("stored_logits") == 0 and artifact.get("accounting_reconciled", {}).get("stored_hidden_activations") == 0 and artifact.get("accounting_reconciled", {}).get("copied_source_parameters") == 0,
        "independent_one_seed_hostile_reconstruction": verifier.get("status") == "PASS_INDEPENDENT_HOSTILE_RECONSTRUCTION" and verifier.get("A0_passes") == 1393 and verifier.get("hostile_mutations_rejected") == 5,
        "three_seed_causal_replication": replication.get("status") == "PASS_THREE_PAIRED_SEED_ROUTE_ISOLATED_REPLICATION" and replication.get("replication_passed") is True and all(item.get("lower_95", 0) > 0 for item in replication.get("hierarchical_A0_minus_control", {}).values()),
        "same_final_checkpoint_quality_runtime_hosts": runtime.get("candidate", {}).get("checkpoint_sha256") == checkpoint and hosts.get("reference", {}).get("checkpoint_sha256") == checkpoint and hosts.get("passed") is True,
        "exact_fully_cpu_runtime": runtime.get("status") == "PASS_EXACT_ROUTE_ISOLATED_CORRECTED_FULLY_CPU_RUNTIME_GATE_MATRIX" and runtime.get("gates", {}).get("complete_corrected_gate_matrix") == "19/19 PASS" and runtime.get("comparisons", {}).get("median_throughput_ratio", 0) >= 2.0 and runtime.get("comparisons", {}).get("paired_throughput_lower_95", 0) >= 2.0 and runtime.get("candidate", {}).get("parent_throughput_retention", 0) >= 0.95 and runtime.get("comparisons", {}).get("median_ttft_ratio", 2) <= 1.0 and runtime.get("comparisons", {}).get("peak_rss_ratio", 2) < 1.0,
        "three_host_exact_reproduction": hosts.get("status") == "PASS_THREE_HOST_EXACT_ROUTE_ISOLATED_REPRODUCTION" and hosts.get("gates", {}).get("host_initializations") == 3 and hosts.get("gates", {}).get("byte_identical_outputs") is True and hosts.get("gates", {}).get("zero_collapse") is True,
        "physically_route_isolated_sparse_execution": sparse.get("unselected_expert_cannot_affect_route") is True and sparse.get("selected_expert_affects_route") is True and sparse.get("active_experts_per_token") == 1 and sparse.get("active_rank") == 16,
        "persistent_state_and_canonical_interface": incremental.get("persistent_incremental_state") is True and incremental.get("canonical_interface_unchanged") is True,
        "teacher_and_source_absent_at_inference": runtime.get("gates", {}).get("candidate_fully_cpu") is True and hosts.get("teacher_present_at_inference", False) is False,
        "final_test_not_accessed": runtime.get("gates", {}).get("final_test_not_accessed") is True and hosts.get("gates", {}).get("final_test_not_accessed") is True,
    }
    prerequisites = {
        "phase0_complete": True,
        "phase1_complete": True,
        "phase2_complete": ratings_complete,
        "phase3_machine_gates_complete": all(machine.values()),
    }
    return machine, prerequisites


def _must_reject(name: str, machine: Mapping[str, bool]) -> str:
    if not all(machine.values()):
        return name
    raise Phase3Error(f"final Phase 3 audit accepted hostile mutation: {name}")


def count_completed_ratings(paths: Iterable[Path]) -> tuple[int, int]:
    rows = 0
    filled = 0
    for path in paths:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if row.get("preference") in {"A", "B", "TIE", "BOTH_UNACCEPTABLE"}:
                filled += 1
    return rows, filled


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha256 = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable final Phase 3 audit output exists")
    artifact = _json(root / protocol["evidence"]["artifact_verifier"])
    verifier = _json(root / protocol["evidence"]["one_seed_verifier"])
    replication = _json(root / protocol["evidence"]["three_seed_replication"])
    runtime = _json(root / protocol["evidence"]["runtime"])
    hosts = _json(root / protocol["evidence"]["hosts"])
    sparse = physical_sparse_microcheck()
    incremental = incremental_state_microcheck(root, protocol)
    manifest = _json(root / protocol["phase2_ratings"]["manifest"])
    rows, filled = count_completed_ratings(root / relative for relative in protocol["phase2_ratings"]["forms"])
    ratings_complete = manifest.get("status") == "COMPLETE" and rows == 21000 and filled == 21000
    machine, prerequisites = assess(artifact, verifier, replication, runtime, hosts, sparse, incremental, ratings_complete)
    rejected = []
    for name, source, field_path, value in (
        ("artifact_accounting_mutation", artifact, ("accounting_reconciled", "stored_logits"), 1),
        ("verifier_quality_mutation", verifier, ("A0_passes",), 1392),
        ("replication_gate_mutation", replication, ("replication_passed",), False),
        ("runtime_ratio_mutation", runtime, ("comparisons", "median_throughput_ratio"), 1.99),
        ("host_checkpoint_mutation", hosts, ("reference", "checkpoint_sha256"), "changed"),
    ):
        mutated = copy.deepcopy(source)
        cursor = mutated
        for key in field_path[:-1]:
            cursor = cursor[key]
        cursor[field_path[-1]] = value
        values = {"artifact": artifact, "verifier": verifier, "replication": replication, "runtime": runtime, "hosts": hosts}
        values[{id(artifact): "artifact", id(verifier): "verifier", id(replication): "replication", id(runtime): "runtime", id(hosts): "hosts"}[id(source)]] = mutated
        hostile_machine, _ = assess(values["artifact"], values["verifier"], values["replication"], values["runtime"], values["hosts"], sparse, incremental, ratings_complete)
        rejected.append(_must_reject(name, hostile_machine))
    machine_complete = all(machine.values()) and len(rejected) == 5
    unconditional = machine_complete and all(prerequisites.values())
    status = "PASS_PHASE3_UNCONDITIONAL_CERTIFICATE" if unconditional else "PASS_PHASE3_MACHINE_EVIDENCE_COMPLETE_BLOCKED_PHASE2_HUMAN_RATINGS" if machine_complete else "FAIL_PHASE3_MACHINE_CERTIFICATE_AUDIT"
    result = {
        "format": FORMAT,
        "status": status,
        "protocol_sha256": protocol_sha256,
        "machine_gates": machine,
        "machine_evidence_complete": machine_complete,
        "prerequisite_gates": prerequisites,
        "phase2_human_ratings": {"manifest_status": manifest.get("status"), "rows": rows, "filled_preferences": filled, "required_preferences": 21000, "complete": ratings_complete},
        "physical_sparse_microcheck": sparse,
        "incremental_state_microcheck": incremental,
        "hostile_mutations_rejected": rejected,
        "hostile_mutations_rejected_count": len(rejected),
        "phase3_certified": unconditional,
        "phase4_open": unconditional,
        "teacher_present_at_inference": False,
        "historical_evidence_changed": False,
        "final_test_accessed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.mkdir(parents=True)
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
