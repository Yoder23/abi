"""Prospective route-local autonomous functional validation for Phase 4."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file, save_file
import torch

from . import capability_compiler_phase3_route_isolated as isolated
from . import capability_compiler_phase3_weak_residual as weak
from . import capability_compiler_phase4_abi_lineage as lineage
from . import capability_compiler_phase4_capability_isolated_adaptation as adapted
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_contract_guard_v2_audit import truncate_at_first_v2_collapse
from .capability_compiler_phase3_guarded_screen import artifact_markers
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-functional-validation/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _partition(record_id: str, modulus: int) -> int:
    return int.from_bytes(hashlib.sha256(record_id.encode()).digest()[:8], "big") % modulus


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_ROUTE_LOCAL_AUTONOMOUS_FUNCTIONAL_VALIDATION"
        or protocol.get("training_device") != "cuda"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
    ):
        raise Phase3Error("functional-validation governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"functional-validation binding changed: {relative}")
    lineage_protocol, _ = lineage.load_protocol(root, root / protocol["lineage_protocol"])
    return protocol, sha256_file(path), lineage_protocol


def _selected_rows(root: Path, protocol: Mapping[str, Any], lineage_protocol: Mapping[str, Any]):
    manifest = _json(root / lineage_protocol["budget_manifest"])
    selected, accounting = lineage._selected_rows(root, lineage_protocol, manifest, str(protocol["budget"]))
    modulus = int(protocol["split"]["modulus"]); validation = int(protocol["split"]["validation_partition"])
    train = [row for row in selected["phase1_ir"] if _partition(str(row["ir_record_id"]), modulus) != validation]
    validate = [row for row in selected["phase1_ir"] if _partition(str(row["ir_record_id"]), modulus) == validation]
    return selected, train, validate, accounting


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    selected, train, validate, accounting = _selected_rows(root, protocol, lineage_protocol)
    train_ids = {row["ir_record_id"] for row in train}; validation_ids = {row["ir_record_id"] for row in validate}
    if train_ids & validation_ids or train_ids | validation_ids != {row["ir_record_id"] for row in selected["phase1_ir"]}:
        raise Phase3Error("functional-validation split is not disjoint and exhaustive")
    counts = {capability: {"train": sum(row["capability"] == capability for row in train), "validation": sum(row["capability"] == capability for row in validate)} for capability in CAPABILITIES}
    if any(value["validation"] < int(protocol["split"]["minimum_validation_per_capability"]) for value in counts.values()):
        raise Phase3Error("functional-validation split lacks capability depth")
    if any(not isinstance(row.get("functional_evaluator"), dict) for row in validate):
        raise Phase3Error("validation evaluator missing")
    return {"status": "PASS_FUNCTIONAL_VALIDATION_PREFLIGHT", "protocol_sha256": protocol_sha, "budget": protocol["budget"], "seed": protocol["seed"], "train_records": len(train), "validation_records": len(validate), "counts": counts, "selection_sha256": accounting["selection_sha256"], "disjoint": True, "exhaustive": True, "functional_evaluators_bound": True, "training_performed": False, "final_test_accessed": False}


def _patch_train(protocol: dict[str, Any], protocol_sha: str, lineage_protocol: dict[str, Any]):
    original_load = adapted.load_protocol; original_examples = lineage._examples_subset
    modulus = int(protocol["split"]["modulus"]); validation = int(protocol["split"]["validation_partition"])
    def examples(rows, tokenizer, **kwargs):
        filtered = [row for row in rows if _partition(str(row["ir_record_id"]), modulus) != validation]
        return original_examples(filtered, tokenizer, **kwargs)
    adapted.load_protocol = lambda *_: (protocol, protocol_sha, lineage_protocol)
    lineage._examples_subset = examples
    return original_load, original_examples


def _restore_train(original) -> None:
    adapted.load_protocol, lineage._examples_subset = original


def _guarded_generate(model: Any, tokenizer: Any, residual: Any, rows: list[dict[str, Any]], run_dir: Path, protocol: Mapping[str, Any], device: torch.device, *, system: str) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    handles = weak._attach(model, residual) if system == "inherited" else adapted._attach(model, residual)
    markers = artifact_markers(run_dir / "budget_host_supervision.abicir"); clause = str(protocol["guard"]["canonical_abstention_clause"])
    passes: dict[str, bool] = {}; evidence = []
    try:
        for row in sorted(rows, key=lambda item: str(item["ir_record_id"])):
            capability = str(row["capability"]); prompt = str(row["normalized_generation_prompt"]); maximum = int(row["generation_max_new_tokens"])
            if system == "inherited":
                mapping = {name: index for index, name in enumerate(weak.WEAK_CAPABILITIES)}; route = mapping.get(capability, -1); weak._set_routes(model, torch.tensor([route], dtype=torch.long, device=device))
                task = torch.tensor([lineage.CAPABILITY_TO_ROUTE[capability]], dtype=torch.long, device=device)
                prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]; ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                output = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], device=device), task_routes=task, use_cache=True); cache, logits, tokens = output["past_key_values"], output["logits"][:, -1], []
                for _ in range(maximum):
                    selected = logits.argmax(dim=-1); token = int(selected.item())
                    if token == int(tokenizer.eos_token_id): break
                    tokens.append(token); output = model(selected[:, None], task_routes=task, past_key_values=cache, use_cache=True); cache, logits = output["past_key_values"], output["logits"][:, -1]
                original = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                original, _, _ = adapted._generate(model, tokenizer, prompt, maximum, capability, device)
            value, terminated = truncate_at_first_v2_collapse(original)
            if capability == "abstention" and not any(marker.casefold() in value.casefold() for marker in markers): value = clause + (" " + value if value else "")
            passed = evaluate_functional(value, row["functional_evaluator"]); record_id = str(row["ir_record_id"]); passes[record_id] = passed
            evidence.append({"record_id": record_id, "capability": capability, "system": system, "functional_pass_v1": passed, "repetition_collapse_v2": repetition_collapse_v2(value), "guard_terminated": terminated, "output": value})
    finally:
        for handle in handles: handle.remove()
    return passes, evidence


def train_select(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available(): raise Phase3Error("immutable output exists or CUDA unavailable")
    output.mkdir(parents=True); trained_dir = output / "trained"
    original = _patch_train(protocol, protocol_sha, lineage_protocol)
    try: training = adapted.train(root, protocol_path, str(protocol["budget"]), int(protocol["seed"]), trained_dir)
    finally: _restore_train(original)
    _, train_rows, validation, accounting = _selected_rows(root, protocol, lineage_protocol)
    device = torch.device("cuda"); run_dir = root / protocol["runs"][0]["lineage_dir"]
    model, tokenizer, _, _, _ = adapted._load_components(root, protocol, lineage_protocol, protocol["runs"][0], device)
    inherited = isolated.RouteIsolatedResidual().to(device); inherited.load_state_dict(load_file(str(run_dir / "v526" / "control_bridge.safetensors"), device="cuda"), strict=True); inherited.eval()
    trained = adapted.CapabilityIsolatedResidual().to(device); trained.load_state_dict(load_file(str(trained_dir / "capability_isolated_residual.safetensors"), device="cuda"), strict=True); trained.eval()
    inherited_pass, inherited_rows = _guarded_generate(model, tokenizer, inherited, validation, run_dir, protocol, device, system="inherited")
    adapted_pass, adapted_rows = _guarded_generate(model, tokenizer, trained, validation, run_dir, protocol, device, system="adapted")
    selections = {}; critical = set(protocol["critical_capabilities"])
    for capability in CAPABILITIES:
        ids = [str(row["ir_record_id"]) for row in validation if row["capability"] == capability]; old = sum(inherited_pass[key] for key in ids); new = sum(adapted_pass[key] for key in ids); interval = wilson(new, len(ids))
        point_min = float(protocol["validation_thresholds"]["critical_point"] if capability in critical else protocol["validation_thresholds"]["per_capability_point"]); lower_min = float(protocol["validation_thresholds"]["critical_lower"] if capability in critical else protocol["validation_thresholds"]["per_capability_lower"])
        accepted = new > old and interval["point"] >= point_min and interval["lower_95"] >= lower_min
        selections[capability] = {"observations": len(ids), "inherited_passes": old, "adapted_passes": new, "adapted_wilson": interval, "accepted": accepted, "selected": "adapted" if accepted else "inherited"}
    initial = adapted.CapabilityIsolatedResidual(); adapted._initialize(initial, run_dir / "v526" / "control_bridge.safetensors"); trained_cpu = adapted.CapabilityIsolatedResidual(); trained_cpu.load_state_dict(load_file(str(trained_dir / "capability_isolated_residual.safetensors"), device="cpu"), strict=True)
    with torch.no_grad():
        for index, capability in enumerate(CAPABILITIES):
            if selections[capability]["accepted"]:
                initial.norm_weight[index].copy_(trained_cpu.norm_weight[index]); initial.norm_bias[index].copy_(trained_cpu.norm_bias[index]); initial.down[index].copy_(trained_cpu.down[index]); initial.up[index].copy_(trained_cpu.up[index])
    checkpoint = output / "selected_capability_residual.safetensors"; save_file({name: value.detach().contiguous() for name, value in initial.state_dict().items()}, str(checkpoint), metadata={"format": FORMAT})
    validation_path = output / "validation_outputs.jsonl"; _write_immutable(validation_path, b"".join(canonical_json_bytes(row) for row in inherited_rows + adapted_rows))
    metadata = {"format": FORMAT, "status": "TRAINED_AND_SELECTED_DEVELOPMENT_UNSEEN", "protocol_sha256": protocol_sha, "budget": protocol["budget"], "seed": protocol["seed"], "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size}, "trained_checkpoint_sha256": training["checkpoint"]["sha256"], "selections": selections, "validation_outputs_sha256": sha256_file(validation_path), "train_records": len(train_rows), "validation_records": len(validation), "imported_information": {"unique_source_attempts": accounting["unique_source_attempts"], "teacher_output_tokens": accounting["authoritative_teacher_output_tokens"], "stored_logits": 0, "stored_hidden_activations": 0, "source_parameters_copied": 0}, "installed_bridge_parameters": adapted.PARAMETERS, "active_bridge_parameters_per_token": adapted.PARAMETERS_PER_ROUTE, "active_routes_per_token": 1, "teacher_present": False, "final_test_accessed": False}
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(); _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n"); return metadata


def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path); metadata = _json(candidate / "metadata.json")
    if metadata["protocol_sha256"] != protocol_sha: raise Phase3Error("functional-validation candidate protocol changed")
    adapted_metadata = {"format": adapted.FORMAT, "status": "TRAINED_DEVELOPMENT_ONLY", "protocol_sha256": protocol_sha, "budget": protocol["budget"], "seed": protocol["seed"], "checkpoint": metadata["checkpoint"]}
    original_load = adapted.load_protocol; adapted.load_protocol = lambda *_: (protocol, protocol_sha, lineage_protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="phase4-functional-validation-", dir=output.parent) as temporary:
            shim = Path(temporary)
            shutil.copy2(candidate / metadata["checkpoint"]["path"], shim / metadata["checkpoint"]["path"])
            (shim / "metadata.json").write_text(json.dumps(adapted_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = adapted.evaluate(root, protocol_path, str(protocol["budget"]), int(protocol["seed"]), shim, output)
    finally: adapted.load_protocol = original_load
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", required=True); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); p=sub.add_parser("train-select"); p.add_argument("--output-dir", required=True); p=sub.add_parser("evaluate"); p.add_argument("--candidate-dir", required=True); p.add_argument("--output-dir", required=True); args=parser.parse_args(argv); root=Path.cwd().resolve(); protocol=root/args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else train_select(root, protocol, root/args.output_dir) if args.command == "train-select" else evaluate(root, protocol, root/args.candidate_dir, root/args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__": raise SystemExit(main())
