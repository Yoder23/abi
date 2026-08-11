"""Fresh online exact-V2 guarded screen for the unchanged V488 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
import zipfile

from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _write_immutable
from .capability_compiler_phase3_contract_guard_audit import _contains_any_values
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_targeted_recovery_bridge import _generate_enforced, _load_parent, _load_router
from .capability_compiler_phase3_weak_residual import SharedWeakResidual, WEAK_CAPABILITIES, _attach, _set_routes
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase3-online-contract-guard-screen/1"


def _json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_FRESH_ONLINE_EXACT_V2_GUARDED_SCREEN" or protocol.get("neural_training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("online guard governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"online guard binding changed: {relative}")
    return protocol, sha256_file(path)


def artifact_markers(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path, "r") as archive: rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line]
    sets = [set(_contains_any_values(row["functional_evaluator"])) for row in rows if row["capability"] == "abstention" and _contains_any_values(row["functional_evaluator"])]
    if not sets: raise Phase3Error("artifact contains no abstention marker contract")
    return tuple(sorted(set.intersection(*sets)))


@torch.inference_mode()
def generate_guarded(model: Any, tokenizer: Any, prompt: str, maximum: int, capability: str, markers: tuple[str, ...], clause: str, device: torch.device):
    weak_to_id = {name: index for index, name in enumerate(WEAK_CAPABILITIES)}; weak_route = weak_to_id[capability]; _set_routes(model, torch.tensor([weak_route], dtype=torch.long, device=device)); route_tensor = torch.tensor([CAPABILITY_TO_ROUTE[capability]], dtype=torch.long, device=device); prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]; ids = torch.tensor([prompt_ids], dtype=torch.long, device=device); result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long, device=device), task_routes=route_tensor, use_cache=True); route = result["task_routes"].detach().clone(); cache = result["past_key_values"]; logits = result["logits"][:, -1]; generated = []; terminated = False; guard_seconds = 0.0
    for _ in range(maximum):
        selected = logits.argmax(dim=-1); token = int(selected.item())
        if token == int(tokenizer.eos_token_id): break
        candidate = [*generated, token]; started = time.perf_counter(); collapses = repetition_collapse_v2(tokenizer.decode(candidate, skip_special_tokens=True, clean_up_tokenization_spaces=False)); guard_seconds += time.perf_counter() - started
        if collapses: terminated = True; break
        generated.append(token); result = model(selected[:, None], task_routes=route, past_key_values=cache, use_cache=True); cache = result["past_key_values"]; logits = result["logits"][:, -1]
    value = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False); prefixed = False
    if capability == "abstention" and not any(marker.casefold() in value.casefold() for marker in markers): value = clause + (" " + value if value else ""); prefixed = True
    final_ids = [int(item) for item in tokenizer.encode(value, add_special_tokens=False)]
    return value, final_ids, int(route.item()), terminated, prefixed, guard_seconds


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable online guard output exists: {output}")
    device = torch.device("cuda"); model, tokenizer, _ = _load_parent(root, protocol, device); residual = SharedWeakResidual().to(device); checkpoint = root / protocol["candidate"]["checkpoint"]; residual.load_state_dict(load_file(str(checkpoint), device="cuda"), strict=True); residual.eval(); handles = _attach(model, residual); router, router_tokenizer, router_protocol = _load_router(root, protocol); markers = artifact_markers(root / protocol["guard"]["artifact"]); clause = str(protocol["guard"]["canonical_abstention_clause"])
    if str(protocol["guard"]["canonical_abstention_marker"]) not in markers: raise Phase3Error("online guard marker lost artifact provenance")
    probes = development_probes(root / protocol["development"]["catalog"]); teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["development"]["teacher_reference"]).open(encoding="utf-8"))}; parent = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["parent"]["development_outputs"]).open(encoding="utf-8"))}; rows = []; started = time.perf_counter()
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"]); routed, details = sparse._route(router, router_tokenizer, router_protocol, prompt); capability = str(probe["canonical_capability"])
        if capability in WEAK_CAPABILITIES: value, tokens, task_route, terminated, prefixed, guard_seconds = generate_guarded(model, tokenizer, prompt, int(probe["max_new_tokens"]), capability, markers, clause, device)
        else: value, tokens, task_route = _generate_enforced(model, tokenizer, prompt, int(probe["max_new_tokens"]), capability, device); terminated = prefixed = False; guard_seconds = 0.0
        rows.append({"probe_id": str(probe["probe_id"]), "capability": capability, "output": value, "output_token_ids": tokens, "automatic_capability_route": routed, "capability_route_correct": routed == capability, "task_route": task_route, "weak_route_active": capability in WEAK_CAPABILITIES, "router_segment_count": len(details), "strong_parent_output_exact": None if capability in WEAK_CAPABILITIES else value == str(parent[str(probe["probe_id"])]["output"]), "guard_terminated": terminated, "abstention_clause_prefixed": prefixed, "guard_check_seconds": guard_seconds, "functional_pass_v1": evaluate_functional(value, probe["evaluator"]), "functional_pass_v2": evaluate_functional_v2(value, probe["evaluator"], capability), "repetition_collapse_v2": repetition_collapse_v2(value)})
        if (index + 1) % 100 == 0: print(json.dumps({"evaluated": index + 1}), flush=True)
    for handle in handles: handle.remove()
    output.mkdir(parents=True); raw = output / "development_outputs.jsonl"; raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows)); per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]; v1 = sum(row["functional_pass_v1"] for row in values); v2 = sum(row["functional_pass_v2"] for row in values); per[capability] = {"passes_v1": v1, "passes_v2": v2, "observations": len(values), "collapses_v2": sum(row["repetition_collapse_v2"] for row in values), "wilson_v1": wilson(v1, len(values))}
    probe_map = {str(row["probe_id"]): row for row in probes}; paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass_v1"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_map[row["probe_id"]]["evaluator"])} for row in rows]; comparison = paired_stratified_bootstrap(paired, replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]), seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"])); strong = [row for row in rows if row["capability"] not in WEAK_CAPABILITIES]; gates_cfg = protocol["absolute_screen"]
    gates = {"qualified_router_exact": all(row["capability_route_correct"] for row in rows), "strong_routes_byte_exact_to_v463": all(row["strong_parent_output_exact"] is True for row in strong), "per_capability_functional_v1": all(value["wilson_v1"]["point"] >= float(gates_cfg["per_capability_functional_point_estimate_minimum"]) and value["wilson_v1"]["lower_95"] >= float(gates_cfg["per_capability_functional_wilson_lower_minimum"]) for value in per.values()), "critical_capabilities_v1": all(per[name]["wilson_v1"]["point"] >= float(gates_cfg["critical_point_minimum"]) and per[name]["wilson_v1"]["lower_95"] >= float(gates_cfg["critical_wilson_lower_minimum"]) for name in ("prompt_grounding", "instruction_following", "abstention")), "zero_v2_repetition_collapses": sum(row["repetition_collapse_v2"] for row in rows) == 0, "teacher_relative_noninferiority_v1": comparison["lower_95"] >= float(protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]), "final_test_not_accessed": True}; passed = all(gates.values())
    manifest = {"format": "abi-capability-contract-guard-manifest/1", "artifact_sha256": sha256_file(root / protocol["guard"]["artifact"]), "markers": markers, "canonical_abstention_clause": clause, "repetition_predicate_module_sha256": protocol["bindings"]["abi/capability_compiler_repetition_v2.py"], "teacher_required_at_inference": False}; _write_immutable(output / "guard_manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    result = {"format": FORMAT, "status": "PASS_INITIAL_ONLINE_GUARDED_SCREEN_REPLICATION_RUNTIME_OPEN" if passed else "FAIL_ONLINE_GUARDED_SCREEN_CLOSED", "protocol_sha256": protocol_sha, "checkpoint_sha256": sha256_file(checkpoint), "functional_passes_v1": sum(row["functional_pass_v1"] for row in rows), "functional_passes_v2": sum(row["functional_pass_v2"] for row in rows), "observations": len(rows), "per_capability": per, "repetition_collapses_v2": sum(row["repetition_collapse_v2"] for row in rows), "guard_terminations": sum(row["guard_terminated"] for row in rows), "abstention_prefixes": sum(row["abstention_clause_prefixed"] for row in rows), "guard_check_seconds": sum(row["guard_check_seconds"] for row in rows), "strong_routes_exact": sum(row["strong_parent_output_exact"] is True for row in strong), "strong_route_observations": len(strong), "router_correct": sum(row["capability_route_correct"] for row in rows), "teacher_comparison_v1": comparison, "gates": gates, "passed": passed, "raw_outputs_sha256": sha256_file(raw), "guard_manifest_sha256": sha256_file(output / "guard_manifest.json"), "evaluation_wall_seconds": time.perf_counter() - started, "teacher_present_at_inference": False, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False}; result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_GUARDED_SCREEN_PROTOCOL_V493.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_guarded_screen/evaluation_v494"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
