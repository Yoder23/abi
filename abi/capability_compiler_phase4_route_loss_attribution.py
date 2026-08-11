"""Read-only acquisition-loss attribution for inherited versus adapted routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_route_isolated as isolated
from . import capability_compiler_phase3_weak_residual as weak
from . import capability_compiler_phase4_abi_lineage as lineage
from . import capability_compiler_phase4_capability_isolated_adaptation as adapted
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_sequence_bridge import _batch


FORMAT = "abi-capability-compiler-phase4-route-loss-attribution/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_ACQUISITION_LOSS_ATTRIBUTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("promotion_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("route-loss attribution governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"route-loss attribution binding changed: {relative}")
    lineage_protocol, _ = lineage.load_protocol(root, root / protocol["lineage_protocol"])
    return protocol, sha256_file(path), lineage_protocol


def _partition(record_id: str, modulus: int) -> int:
    return int.from_bytes(hashlib.sha256(record_id.encode()).digest()[:8], "big") % modulus


def _per_record_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_labels = labels[:, 1:]
    losses = F.cross_entropy(logits[:, :-1].flatten(0, 1), shifted_labels.flatten(), ignore_index=-100, reduction="none").reshape_as(shifted_labels)
    active = shifted_labels.ge(0)
    return (losses * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)


@torch.inference_mode()
def _losses(model: Any, tokenizer: Any, residual: Any, examples: Sequence[Mapping[str, Any]], device: torch.device, *, system: str, batch_size: int) -> dict[str, float]:
    handles = weak._attach(model, residual) if system == "inherited" else adapted._attach(model, residual)
    result: dict[str, float] = {}
    try:
        for start in range(0, len(examples), batch_size):
            rows = list(examples[start:start + batch_size])
            ids, labels, attention, prompt_lengths, task_routes = _batch(rows, int(tokenizer.eos_token_id), device)
            if system == "inherited":
                mapping = {name: index for index, name in enumerate(weak.WEAK_CAPABILITIES)}
                routes = torch.tensor([mapping.get(str(row["capability"]), -1) for row in rows], dtype=torch.long, device=device)
                weak._set_routes(model, routes)
            else:
                adapted._set_routes(model, adapted._route_tensor(rows, device))
            output = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=task_routes, use_cache=False)
            values = _per_record_loss(output["logits"].float(), labels)
            result.update({str(row["record_id"]): float(value) for row, value in zip(rows, values)})
    finally:
        for handle in handles:
            handle.remove()
    return result


def audit(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable output exists or CUDA unavailable")
    run_dir = root / protocol["lineage_dir"]
    manifest = _json(root / lineage_protocol["budget_manifest"])
    selected, _ = lineage._selected_rows(root, lineage_protocol, manifest, protocol["budget"])
    device = torch.device("cuda")
    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    model, tokenizer, _ = lineage._load_candidate(root, v440, run_dir / "v463", device)
    examples = lineage._examples_subset(selected["phase1_ir"], tokenizer, system="A0", seed=int(protocol["seed"]), max_tokens=int(protocol["max_tokens"]))
    audit_examples = sorted((row for row in examples if _partition(str(row["record_id"]), int(protocol["partition_modulus"])) == int(protocol["audit_partition"])), key=lambda row: str(row["record_id"]))
    if any(sum(row["capability"] == capability for row in audit_examples) < int(protocol["minimum_records_per_capability"]) for capability in CAPABILITIES):
        raise Phase3Error("acquisition audit partition lacks capability depth")
    inherited = isolated.RouteIsolatedResidual().to(device)
    inherited.load_state_dict(load_file(str(run_dir / "v526" / "control_bridge.safetensors"), device="cuda"), strict=True); inherited.eval()
    trained = adapted.CapabilityIsolatedResidual().to(device)
    trained.load_state_dict(load_file(str(root / protocol["adapted_checkpoint"]), device="cuda"), strict=True); trained.eval()
    inherited_losses = _losses(model, tokenizer, inherited, audit_examples, device, system="inherited", batch_size=int(protocol["batch_size"]))
    adapted_losses = _losses(model, tokenizer, trained, audit_examples, device, system="adapted", batch_size=int(protocol["batch_size"]))
    selection = {}
    for capability in CAPABILITIES:
        ids = [str(row["record_id"]) for row in audit_examples if row["capability"] == capability]
        old = sum(inherited_losses[key] for key in ids) / len(ids); new = sum(adapted_losses[key] for key in ids) / len(ids)
        selection[capability] = {"records": len(ids), "inherited_mean_response_ce": old, "adapted_mean_response_ce": new, "selected": "adapted" if new < old else "inherited"}
    outputs = {"inherited": {row["probe_id"]: row for row in _rows(root / protocol["development_outputs"]["inherited"])}, "adapted": {row["probe_id"]: row for row in _rows(root / protocol["development_outputs"]["adapted"])}}
    probes = {row["probe_id"]: row for row in development_probes(root / protocol["catalog"])}
    teacher = {row["probe_id"]: row for row in _rows(root / protocol["teacher_reference"])}
    composed = []
    for probe_id in sorted(probes):
        capability = str(probes[probe_id]["canonical_capability"]); system = selection[capability]["selected"]; source = outputs[system][probe_id]
        value = str(source["output"])
        composed.append({"probe_id": probe_id, "capability": capability, "selected_system": system, "output": value, "functional_pass_v1": evaluate_functional(value, probes[probe_id]["evaluator"]), "repetition_collapse_v2": bool(source["repetition_collapse_v2"])})
    per = {}
    for capability in CAPABILITIES:
        rows = [row for row in composed if row["capability"] == capability]; passed = sum(row["functional_pass_v1"] for row in rows)
        per[capability] = {"passes": passed, "observations": len(rows), "wilson": wilson(passed, len(rows)), "collapses": sum(row["repetition_collapse_v2"] for row in rows)}
    paired = [{"capability": row["capability"], "candidate_pass": row["functional_pass_v1"], "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probes[row["probe_id"]]["evaluator"])} for row in composed]
    relative = paired_stratified_bootstrap(paired, replicates=10_000, seed=int(protocol["bootstrap_seed"]))
    threshold = protocol["thresholds"]
    gates = {"per_capability": all(value["wilson"]["point"] >= threshold["per_capability_point"] and value["wilson"]["lower_95"] >= threshold["per_capability_lower"] for value in per.values()), "critical": all(per[name]["wilson"]["point"] >= threshold["critical_point"] and per[name]["wilson"]["lower_95"] >= threshold["critical_lower"] for name in protocol["critical_capabilities"]), "zero_collapse": sum(row["repetition_collapse_v2"] for row in composed) == 0, "teacher_noninferior": relative["lower_95"] >= threshold["teacher_relative_lower"], "selection_uses_no_development_labels": True, "no_training": True, "no_promotion": True, "final_test_not_accessed": True}
    raw = output.parent / "selected_outputs.jsonl"; output.parent.mkdir(parents=True); _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in composed))
    result = {"format": "abi-capability-compiler-phase4-route-loss-attribution-result/1", "status": "PASS_PROSPECTIVE_ROUTE_LOCAL_VALIDATION_DESIGN_SUPPORTED" if all(gates.values()) else "FAIL_ROUTE_LOCAL_LOSS_SELECTION_NOT_SUPPORTED", "protocol_sha256": protocol_sha, "audit_partition_records": len(audit_examples), "selection": selection, "functional_passes": sum(row["functional_pass_v1"] for row in composed), "observations": len(composed), "per_capability": per, "repetition_collapses_v2": sum(row["repetition_collapse_v2"] for row in composed), "teacher_comparison": relative, "gates": gates, "selected_outputs_sha256": sha256_file(raw), "training_performed": False, "promotion_authorized": False, "final_test_accessed": False, "interpretation": "The acquisition partition was consumed by the existing candidate and is not a true validation holdout. A pass supports one future prospectively split train/validation architecture only; it does not promote this composition.", "claim_boundary": "Read-only in-sample acquisition-loss attribution; no candidate, stable frontier, matched baseline, final test, Phase 4 certificate, or superiority claim."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = audit(root, root / args.protocol, root / args.output); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__": raise SystemExit(main())
