"""Prospective 14-route sparse adaptation for the conditional Phase 4 frontier."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any, Iterable, Mapping, Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn

from . import capability_compiler_phase3_sparse_router as sparse
from . import capability_compiler_phase3_weak_residual as weak
from . import capability_compiler_phase4_abi_lineage as lineage
from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _write_immutable
from .capability_compiler_phase3_contract_guard_v2_audit import truncate_at_first_v2_collapse
from .capability_compiler_phase3_guarded_screen import artifact_markers
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_sequence_bridge import _batch
from .capability_compiler_phase3_targeted_recovery_bridge import _batch_with_prefixes
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .layercake_host import _equal_record_prompt_overlap_ce


FORMAT = "abi-capability-compiler-phase4-capability-isolated-adaptation/1"
WIDTH = 768
RANK = 16
ROUTES = len(CAPABILITIES)
PARAMETERS_PER_ROUTE = 2 * WIDTH + 2 * WIDTH * RANK
PARAMETERS = ROUTES * PARAMETERS_PER_ROUTE


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_ONE_CAPABILITY_ISOLATED_SPARSE_DESIGN"
        or protocol.get("training_device") != "cuda"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
        or int(protocol["architecture"]["routes"]) != ROUTES
        or int(protocol["architecture"]["rank"]) != RANK
        or int(protocol["architecture"]["trainable_parameters"]) != PARAMETERS
    ):
        raise Phase3Error("capability-isolated adaptation governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"capability-isolated binding changed: {relative}")
    lineage_path = root / protocol["lineage_protocol"]
    lineage_protocol, _ = lineage.load_protocol(root, lineage_path)
    return protocol, sha256_file(path), lineage_protocol


class CapabilityIsolatedResidual(nn.Module):
    """Fourteen disjoint experts; selection occurs before tensor execution."""

    def __init__(self) -> None:
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(ROUTES, WIDTH))
        self.norm_bias = nn.Parameter(torch.zeros(ROUTES, WIDTH))
        self.down = nn.Parameter(torch.empty(ROUTES, RANK, WIDTH))
        self.up = nn.Parameter(torch.zeros(ROUTES, WIDTH, RANK))
        nn.init.normal_(self.down, mean=0.0, std=0.02)

    def delta(self, hidden: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        weight = self.norm_weight.index_select(0, routes)[:, None, :]
        bias = self.norm_bias.index_select(0, routes)[:, None, :]
        mean = hidden.mean(dim=-1, keepdim=True)
        variance = (hidden - mean).square().mean(dim=-1, keepdim=True)
        normalized = (hidden - mean) * torch.rsqrt(variance + 1e-5)
        normalized = normalized * weight + bias
        down = self.down.index_select(0, routes)
        up = self.up.index_select(0, routes)
        low = torch.einsum("bsw,brw->bsr", normalized, down)
        return torch.einsum("bsr,bwr->bsw", F.silu(low), up)


def _initialize(residual: CapabilityIsolatedResidual, checkpoint: Path) -> None:
    inherited = load_file(str(checkpoint), device="cpu")
    required = {"norm.weight", "norm.bias", "down", "up"}
    if set(inherited) != required or tuple(inherited["down"].shape) != (4, RANK, WIDTH):
        raise Phase3Error("inherited four-route residual schema changed")
    mapping = {name: index for index, name in enumerate(weak.WEAK_CAPABILITIES)}
    with torch.no_grad():
        for capability, source in mapping.items():
            target = CAPABILITIES.index(capability)
            residual.norm_weight[target].copy_(inherited["norm.weight"])
            residual.norm_bias[target].copy_(inherited["norm.bias"])
            residual.down[target].copy_(inherited["down"][source])
            residual.up[target].copy_(inherited["up"][source])


def _set_routes(model: nn.Module, routes: torch.Tensor) -> None:
    if routes.ndim != 1:
        raise Phase3Error("capability routes must be one-dimensional")
    for block in model.transformer.h:
        block._abi_capability_isolated_routes = routes.long()


def _hook(residual: CapabilityIsolatedResidual):
    def apply(module, args, kwargs):
        hidden = args[0]
        routes = getattr(module, "_abi_capability_isolated_routes", None)
        if routes is None or routes.shape[0] != hidden.shape[0] or bool((routes < 0).any()) or bool((routes >= ROUTES).any()):
            raise Phase3Error("capability route is absent or malformed")
        return (hidden + residual.delta(hidden, routes), *args[1:]), kwargs
    return apply


def _attach(model: nn.Module, residual: CapabilityIsolatedResidual):
    return [block.register_forward_pre_hook(_hook(residual), with_kwargs=True) for block in model.transformer.h]


def _route_tensor(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([CAPABILITIES.index(str(row["capability"])) for row in rows], dtype=torch.long, device=device)


class CapabilitySampler:
    def __init__(self, rows: Sequence[Mapping[str, Any]], seed: int) -> None:
        self.groups = {name: [row for row in rows if row["capability"] == name] for name in CAPABILITIES}
        if any(not values for values in self.groups.values()):
            raise Phase3Error("capability-isolated training lost a capability")
        self.rng = random.Random(seed)
        self.recovery_index = 0

    def teacher_forced_batch(self) -> list[Mapping[str, Any]]:
        return [self.rng.choice(self.groups[name]) for name in CAPABILITIES]

    def recovery_batch(self, size: int) -> list[Mapping[str, Any]]:
        result = []
        for _ in range(size):
            name = CAPABILITIES[self.recovery_index % ROUTES]
            self.recovery_index += 1
            result.append(self.rng.choice(self.groups[name]))
        return result


@torch.inference_mode()
def _prefixes(model: Any, rows: Sequence[Mapping[str, Any]], horizon: int, device: torch.device, eos: int) -> list[list[int]]:
    result_rows: list[list[int]] = []
    model.eval()
    for row in rows:
        prompt_count = int(row["prompt_tokens"])
        prompt = list(row["input_ids"][:prompt_count])
        _set_routes(model, _route_tensor([row], device))
        task = torch.tensor([int(row["route"])], dtype=torch.long, device=device)
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        result = model(ids, prompt_lengths=torch.tensor([prompt_count], device=device), task_routes=task, use_cache=True)
        cache, logits, generated = result["past_key_values"], result["logits"][:, -1], []
        for _ in range(horizon):
            selected = logits.argmax(dim=-1)
            token = int(selected.item())
            if token == eos:
                break
            generated.append(token)
            result = model(selected[:, None], task_routes=task, past_key_values=cache, use_cache=True)
            cache, logits = result["past_key_values"], result["logits"][:, -1]
        result_rows.append(generated)
    model.train()
    return result_rows


def _run(protocol: Mapping[str, Any], budget: str, seed: int) -> Mapping[str, Any]:
    match = next((row for row in protocol["runs"] if row["budget"] == budget and int(row["seed"]) == seed), None)
    if match is None:
        raise Phase3Error("unregistered capability-isolated run")
    return match


def _load_components(root: Path, protocol: Mapping[str, Any], lineage_protocol: Mapping[str, Any], run: Mapping[str, Any], device: torch.device):
    run_dir = root / str(run["lineage_dir"])
    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    model, tokenizer, _ = lineage._load_candidate(root, v440, run_dir / "v463", device)
    router_protocol = _json(root / lineage_protocol["base_protocols"]["router"])
    router = sparse._load(root, router_protocol, run_dir / "router")
    return model, tokenizer, router, router_protocol, run_dir


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    residual = CapabilityIsolatedResidual()
    if sum(value.numel() for value in residual.parameters()) != PARAMETERS:
        raise Phase3Error("capability-isolated parameter count changed")
    checks = []
    manifest = _json(root / lineage_protocol["budget_manifest"])
    for run in protocol["runs"]:
        selected, accounting = lineage._selected_rows(root, lineage_protocol, manifest, str(run["budget"]))
        checkpoint = root / str(run["lineage_dir"]) / "v526" / "control_bridge.safetensors"
        probe = CapabilityIsolatedResidual(); _initialize(probe, checkpoint)
        checks.append({"budget": run["budget"], "seed": run["seed"], "phase1_records": len(selected["phase1_ir"]), "unique_source_attempts": accounting["unique_source_attempts"], "initialization_sha256": sha256_file(checkpoint)})
    hidden = torch.randn(2, 3, WIDTH)
    with torch.no_grad():
        residual.up.zero_(); residual.up[0].fill_(0.01)
    delta = residual.delta(hidden, torch.tensor([0, 1]))
    if not bool(delta[0].abs().sum() > 0) or not torch.equal(delta[1], torch.zeros_like(delta[1])):
        raise Phase3Error("physical route isolation preflight failed")
    return {
        "status": "PASS_CAPABILITY_ISOLATED_ADAPTATION_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "routes": ROUTES,
        "rank": RANK,
        "installed_bridge_parameters": PARAMETERS,
        "active_bridge_parameters_per_token": PARAMETERS_PER_ROUTE,
        "active_bridge_parameter_fraction": PARAMETERS_PER_ROUTE / PARAMETERS,
        "one_active_path_verified": True,
        "runs": checks,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, budget: str, seed: int, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    run = _run(protocol, budget, seed)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable output exists or CUDA unavailable")
    cfg = protocol["training"]
    set_determinism(seed)
    device = torch.device("cuda")
    model, tokenizer, _, _, run_dir = _load_components(root, protocol, lineage_protocol, run, device)
    residual = CapabilityIsolatedResidual(); _initialize(residual, run_dir / "v526" / "control_bridge.safetensors"); residual.to(device)
    handles = _attach(model, residual)
    manifest = _json(root / lineage_protocol["budget_manifest"])
    selected, accounting = lineage._selected_rows(root, lineage_protocol, manifest, budget)
    examples = lineage._examples_subset(selected["phase1_ir"], tokenizer, system="A0", seed=seed, max_tokens=int(cfg["max_tokens"]))
    sampler = CapabilitySampler(examples, seed)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(cfg["weight_decay"]))
    parent_before = weak._state_hash(model.state_dict())
    process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats()
    observations, recovery_counts, sequence = Counter(), Counter(), hashlib.sha256()
    teacher_tokens = prefix_tokens = recovery_batches = 0; curves = []; started = time.perf_counter()
    residual.train()
    for step in range(1, int(cfg["steps"]) + 1):
        batch = sampler.teacher_forced_batch(); prefixes = [[] for _ in batch]; horizon = None
        if step >= int(cfg["recovery_start_step"]) and (step - int(cfg["recovery_start_step"])) % int(cfg["recovery_interval"]) == 0:
            recovery = sampler.recovery_batch(int(cfg["recovery_batch_size"]))
            choices = [int(value) for value in cfg["recovery_horizons"]]; horizon = choices[recovery_batches % len(choices)]
            generated = _prefixes(model, recovery, horizon, device, int(tokenizer.eos_token_id))
            slots = {str(row["capability"]): index for index, row in enumerate(batch)}
            for replacement, prefix in zip(recovery, generated):
                index = slots[str(replacement["capability"])]; batch[index] = replacement; prefixes[index] = prefix
                recovery_counts[str(replacement["capability"])] += 1
            recovery_batches += 1; prefix_tokens += sum(len(value) for value in generated)
        ids, labels, attention, prompt_lengths, task_routes = _batch_with_prefixes(batch, int(tokenizer.eos_token_id), device, prefixes)
        _set_routes(model, _route_tensor(batch, device)); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=task_routes, use_cache=False)
            loss = _equal_record_prompt_overlap_ce(result["logits"], labels, ids, prompt_lengths, overlap_weight=float(cfg["prompt_overlap_weight"]))
        loss.backward(); torch.nn.utils.clip_grad_norm_(residual.parameters(), float(cfg["gradient_clip_norm"])); optimizer.step()
        for row in batch:
            observations[str(row["capability"])] += 1; teacher_tokens += int(row["response_tokens"]); sequence.update((str(row["record_id"]) + "\n").encode())
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {"step": step, "loss": float(loss.detach()), "recovery_horizon": horizon, "wall_seconds": time.perf_counter() - started}; curves.append(curve); print(json.dumps(curve), flush=True)
    parent_after = weak._state_hash(model.state_dict())
    for handle in handles: handle.remove()
    if parent_before != parent_after:
        raise Phase3Error("immutable parent changed")
    output.mkdir(parents=True); checkpoint = output / "capability_isolated_residual.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in residual.state_dict().items()}, str(checkpoint), metadata={"format": FORMAT})
    metadata = {
        "format": FORMAT, "status": "TRAINED_DEVELOPMENT_ONLY", "protocol_sha256": protocol_sha, "budget": budget, "seed": seed,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parent": {"checkpoint_sha256": sha256_file(run_dir / "v463" / "model.safetensors"), "mutated": False},
        "router": {"checkpoint_sha256": sha256_file(run_dir / "router" / "router.safetensors"), "mutated": False},
        "initialization": {"checkpoint_sha256": sha256_file(run_dir / "v526" / "control_bridge.safetensors")},
        "architecture": {"installed_bridge_parameters": PARAMETERS, "active_bridge_parameters_per_token": PARAMETERS_PER_ROUTE, "routes": ROUTES, "active_routes_per_token": 1, "rank": RANK},
        "training": {"steps": int(cfg["steps"]), "observations": sum(observations.values()), "observations_by_capability": dict(observations), "teacher_response_tokens_in_loss": teacher_tokens, "recovery_batches": recovery_batches, "recovery_by_capability": dict(recovery_counts), "autonomous_prefix_tokens": prefix_tokens, "record_sequence_sha256": sequence.hexdigest(), "wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "curves": curves},
        "imported_information": {"unique_source_attempts": accounting["unique_source_attempts"], "teacher_output_tokens": accounting["authoritative_teacher_output_tokens"], "stored_logits": 0, "stored_hidden_activations": 0, "source_parameters_copied": 0},
        "teacher_present": False, "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(); _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n"); return metadata


@torch.inference_mode()
def _generate(model: Any, tokenizer: Any, prompt: str, maximum: int, capability: str, device: torch.device) -> tuple[str, list[int], int]:
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
    _set_routes(model, torch.tensor([CAPABILITIES.index(capability)], dtype=torch.long, device=device))
    task = torch.tensor([CAPABILITY_TO_ROUTE[capability]], dtype=torch.long, device=device)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], device=device), task_routes=task, use_cache=True)
    cache, logits, generated = result["past_key_values"], result["logits"][:, -1], []
    for _ in range(maximum):
        selected = logits.argmax(dim=-1); token = int(selected.item())
        if token == int(tokenizer.eos_token_id): break
        generated.append(token); result = model(selected[:, None], task_routes=task, past_key_values=cache, use_cache=True); cache, logits = result["past_key_values"], result["logits"][:, -1]
    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False), generated, int(task.item())


def evaluate(root: Path, protocol_path: Path, budget: str, seed: int, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path); run = _run(protocol, budget, seed)
    if output.exists() or not torch.cuda.is_available(): raise Phase3Error("immutable evaluation exists or CUDA unavailable")
    metadata = _json(candidate / "metadata.json"); checkpoint = candidate / str(metadata["checkpoint"]["path"])
    if metadata["protocol_sha256"] != protocol_sha or metadata["budget"] != budget or int(metadata["seed"]) != seed or sha256_file(checkpoint) != metadata["checkpoint"]["sha256"]: raise Phase3Error("candidate lineage changed")
    device = torch.device("cuda"); model, tokenizer, router_bundle, router_protocol, run_dir = _load_components(root, protocol, lineage_protocol, run, device)
    residual = CapabilityIsolatedResidual().to(device); residual.load_state_dict(load_file(str(checkpoint), device="cuda"), strict=True); residual.eval(); handles = _attach(model, residual)
    router, router_tokenizer, _ = router_bundle; markers = artifact_markers(run_dir / "budget_host_supervision.abicir"); clause = str(protocol["guard"]["canonical_abstention_clause"])
    probes = development_probes(root / protocol["development"]["catalog"]); teacher = {row["probe_id"]: row for row in map(json.loads, (root / protocol["development"]["teacher_reference"]).open(encoding="utf-8"))}
    rows = []; started = time.perf_counter()
    for probe in probes:
        capability = str(probe["canonical_capability"]); prompt = str(probe["prompt"]); routed, _ = sparse._route(router, router_tokenizer, router_protocol, prompt)
        original, _, task_route = _generate(model, tokenizer, prompt, int(probe["max_new_tokens"]), capability, device); value, terminated = truncate_at_first_v2_collapse(original); prefixed = False
        if capability == "abstention" and not any(marker.casefold() in value.casefold() for marker in markers): value = clause + (" " + value if value else ""); prefixed = True
        tokens = [int(item) for item in tokenizer.encode(value, add_special_tokens=False)]
        rows.append({"probe_id": str(probe["probe_id"]), "capability": capability, "output": value, "original_output": original, "output_token_ids": tokens, "automatic_capability_route": routed, "capability_route_correct": routed == capability, "physical_residual_route": CAPABILITIES.index(capability), "active_residual_routes": 1, "task_route": task_route, "guard_terminated": terminated, "abstention_prefixed": prefixed, "functional_pass_v1": evaluate_functional(value, probe["evaluator"]), "functional_pass_v2": evaluate_functional_v2(value, probe["evaluator"]), "repetition_collapse_v2": repetition_collapse_v2(value)})
    for handle in handles: handle.remove()
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]; passed = sum(row["functional_pass_v1"] for row in values)
        per[capability] = {"passes": passed, "observations": len(values), "wilson_v1": wilson(passed, len(values)), "collapses_v2": sum(row["repetition_collapse_v2"] for row in values)}
    paired = [{"capability": row["capability"], "candidate_pass": row["functional_pass_v1"], "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), next(probe["evaluator"] for probe in probes if probe["probe_id"] == row["probe_id"]))} for row in rows]
    relative = paired_stratified_bootstrap(paired, replicates=10_000, seed=int(protocol["development"]["bootstrap_seed"]) + seed)
    thresholds = protocol["thresholds"]
    gates = {"per_capability": all(value["wilson_v1"]["point"] >= thresholds["per_capability_point"] and value["wilson_v1"]["lower_95"] >= thresholds["per_capability_lower"] for value in per.values()), "critical": all(per[name]["wilson_v1"]["point"] >= thresholds["critical_point"] and per[name]["wilson_v1"]["lower_95"] >= thresholds["critical_lower"] for name in protocol["critical_capabilities"]), "zero_collapse": sum(row["repetition_collapse_v2"] for row in rows) == 0, "teacher_noninferior": relative["lower_95"] >= thresholds["teacher_relative_lower"], "router_exact": sum(row["capability_route_correct"] for row in rows) == len(rows), "one_active_path": all(row["active_residual_routes"] == 1 for row in rows), "teacher_absent": True, "final_test_not_accessed": True}
    output.mkdir(parents=True); raw = output / "development_outputs.jsonl"; _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in rows))
    result = {"format": "abi-capability-compiler-phase4-capability-isolated-adaptation-result/1", "status": "PASS_CAPABILITY_ISOLATED_MACHINE_GATES" if all(gates.values()) else "FAIL_CAPABILITY_ISOLATED_MACHINE_GATES", "protocol_sha256": protocol_sha, "budget": budget, "seed": seed, "checkpoint_sha256": sha256_file(checkpoint), "functional_passes_v1": sum(row["functional_pass_v1"] for row in rows), "observations": len(rows), "per_capability": per, "repetition_collapses_v2": sum(row["repetition_collapse_v2"] for row in rows), "guard_terminations": sum(row["guard_terminated"] for row in rows), "router_correct": sum(row["capability_route_correct"] for row in rows), "teacher_comparison_v1": relative, "gates": gates, "installed_bridge_parameters": PARAMETERS, "active_bridge_parameters_per_token": PARAMETERS_PER_ROUTE, "active_routes_per_token": 1, "evaluation_wall_seconds": time.perf_counter() - started, "raw_outputs_sha256": sha256_file(raw), "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda}, "teacher_present_at_inference": False, "final_test_accessed": False, "phase4_certified": False, "claim_boundary": "One prospective capability-isolated development screen; no stable frontier, adjacent lower failure, matched baseline, runtime certification, final test, Phase 4 certificate, or superiority claim."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", required=True); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); p = sub.add_parser("train"); p.add_argument("--budget", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--output-dir", required=True); p = sub.add_parser("evaluate"); p.add_argument("--budget", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--candidate-dir", required=True); p.add_argument("--output-dir", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = root / args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else train(root, protocol, args.budget, args.seed, root / args.output_dir) if args.command == "train" else evaluate(root, protocol, args.budget, args.seed, root / args.candidate_dir, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__": raise SystemExit(main())
