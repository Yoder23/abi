"""Host-conformant all-strata autonomous-recovery successor to V474."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence
import zipfile

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch

from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_targeted_recovery_bridge import _autonomous_prefixes, _batch_with_prefixes, _generate_enforced, _load_parent, _load_router, _weak_routes
from .capability_compiler_phase3_weak_residual import EXPECTED_PARAMETERS, SharedWeakResidual, WEAK_CAPABILITIES, _attach, _parameter_count, _set_routes, _state_hash
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .layercake_host import _equal_record_prompt_overlap_ce


FORMAT = "abi-capability-compiler-phase3-host-recovery-bridge/1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SINGLE_HOST_CONFORMANT_ALL_STRATA_RECOVERY_SUCCESSOR"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
        or tuple(protocol.get("architecture", {}).get("weak_capabilities", ())) != WEAK_CAPABILITIES
        or int(protocol.get("architecture", {}).get("trainable_parameters", -1)) != EXPECTED_PARAMETERS
    ):
        raise Phase3Error("host recovery governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"host recovery binding changed: {relative}")
    return protocol, sha256_file(path)


def _artifact_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as archive:
        if tuple(sorted(archive.namelist())) != ("accounting.json", "manifest.json", "records.jsonl"):
            raise Phase3Error("host supervision archive entries changed")
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line]
    counts = Counter((str(row["capability"]), int(row["builder"])) for row in rows)
    if len(rows) != 1280 or counts != Counter({(capability, builder): 80 for capability in WEAK_CAPABILITIES for builder in range(4)}):
        raise Phase3Error("host supervision balance changed")
    return rows


def _examples(rows: Sequence[Mapping[str, Any]], tokenizer: Any, max_tokens: int) -> list[dict[str, Any]]:
    eos = int(tokenizer.eos_token_id)
    result = []
    for row in rows:
        prompt_ids = [int(value) for value in tokenizer.encode(str(row["host_prompt"]).rstrip() + "\n", add_special_tokens=False)]
        response_ids = [int(value) for value in tokenizer.encode(str(row["output"]), add_special_tokens=False)] + [eos]
        available = max_tokens - len(prompt_ids)
        if available < 2:
            raise Phase3Error("verified host supervision exceeds model context")
        response_ids = response_ids[:available]
        if response_ids[-1] != eos:
            response_ids[-1] = eos
        result.append({
            "record_id": str(row["record_id"]),
            "capability": str(row["capability"]),
            "builder": int(row["builder"]),
            "route": CAPABILITY_TO_ROUTE[str(row["capability"])],
            "input_ids": prompt_ids + response_ids,
            "labels": [-100] * len(prompt_ids) + response_ids,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
        })
    return result


class AllStrataSampler:
    def __init__(self, rows: Sequence[Mapping[str, Any]], seed: int):
        self.groups = {(capability, builder): [row for row in rows if row["capability"] == capability and int(row["builder"]) == builder] for capability in WEAK_CAPABILITIES for builder in range(4)}
        if set(map(len, self.groups.values())) != {80}:
            raise Phase3Error("all-strata sampler depth changed")
        self.strata = tuple(self.groups)
        self.rng = random.Random(seed)
        self.recovery_index = 0

    def teacher_forced_batch(self) -> list[Mapping[str, Any]]:
        return [self.rng.choice(self.groups[key]) for key in self.strata]

    def recovery_batch(self, size: int) -> list[Mapping[str, Any]]:
        result = []
        for _ in range(size):
            key = self.strata[self.recovery_index % len(self.strata)]
            self.recovery_index += 1
            result.append(self.rng.choice(self.groups[key]))
        return result


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer, _ = _load_parent(root, protocol, torch.device("cpu"))
    rows = _artifact_rows(root / protocol["supervision"]["artifact"])
    examples = _examples(rows, tokenizer, int(protocol["training"]["max_tokens"]))
    residual = SharedWeakResidual()
    initialization = root / protocol["initialization"]["checkpoint"]
    residual.load_state_dict(load_file(str(initialization), device="cpu"), strict=True)
    return {"status": "PASS_PREFLIGHT", "protocol_sha256": protocol_sha, "frozen_parent_parameters": sum(value.numel() for value in model.parameters()), "bridge_parameters": _parameter_count(residual), "supervision_records": len(examples), "strata": 16, "tokenizer_vocabulary": len(tokenizer), "final_test_accessed": False}


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable host recovery candidate exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("host recovery CUDA unavailable")
    cfg = protocol["training"]; seed = int(cfg["seed"]); set_determinism(seed); device = torch.device("cuda")
    model, tokenizer, _ = _load_parent(root, protocol, device)
    residual = SharedWeakResidual().to(device)
    initialization = root / protocol["initialization"]["checkpoint"]
    residual.load_state_dict(load_file(str(initialization), device="cuda"), strict=True)
    initial_residual_hash = _state_hash(residual.state_dict())
    handles = _attach(model, residual)
    rows = _artifact_rows(root / protocol["supervision"]["artifact"])
    examples = _examples(rows, tokenizer, int(cfg["max_tokens"]))
    sampler = AllStrataSampler(examples, seed)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(cfg["weight_decay"]))
    parent_before = _state_hash(model.state_dict()); process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats()
    counts = Counter(); recovery_counts = Counter(); recovery_horizon_counts = Counter(); record_sequence = hashlib.sha256(); teacher_tokens = recovery_prefix_tokens = recovery_batches = 0; curves = []; started = time.perf_counter()
    for step in range(1, int(cfg["steps"]) + 1):
        selected = sampler.teacher_forced_batch(); prefixes = [[] for _ in selected]; horizon = None
        if step >= int(cfg["recovery_start_step"]) and (step - int(cfg["recovery_start_step"])) % int(cfg["recovery_interval"]) == 0:
            recovery = sampler.recovery_batch(int(cfg["recovery_batch_size"])); horizons = tuple(int(value) for value in cfg["recovery_horizons"]); horizon = horizons[recovery_batches % len(horizons)]
            generated = _autonomous_prefixes(model, tokenizer, recovery, horizon, device); location = {(str(row["record_id"]), str(row["capability"]), int(row["builder"])): prefix for row, prefix in zip(recovery, generated)}
            prefixes = [location.get((str(row["record_id"]), str(row["capability"]), int(row["builder"])), []) for row in selected]
            # Recovery records replace the corresponding stratum member so every generated prefix is trained.
            by_stratum = {(str(row["capability"]), int(row["builder"])): (row, prefix) for row, prefix in zip(recovery, generated)}
            for index, row in enumerate(selected):
                replacement = by_stratum.get((str(row["capability"]), int(row["builder"])))
                if replacement is not None:
                    selected[index], prefixes[index] = replacement
                    recovery_counts[f"{row['capability']}:{row['builder']}"] += 1
            recovery_batches += 1; recovery_horizon_counts[str(horizon)] += 1; recovery_prefix_tokens += sum(len(value) for value in generated)
        ids, labels, attention, prompt_lengths, routes = _batch_with_prefixes(selected, int(tokenizer.eos_token_id), device, prefixes)
        _set_routes(model, _weak_routes(selected, device)); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=routes, use_cache=False)
            loss = _equal_record_prompt_overlap_ce(result["logits"], labels, ids, prompt_lengths, overlap_weight=float(cfg["prompt_overlap_weight"]))
        loss.backward(); torch.nn.utils.clip_grad_norm_(residual.parameters(), float(cfg["gradient_clip_norm"])); optimizer.step()
        for row in selected:
            key = f"{row['capability']}:{row['builder']}"; counts[key] += 1; teacher_tokens += int(row["response_tokens"]); record_sequence.update((str(row["record_id"]) + "\n").encode())
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {"step": step, "language_loss": float(loss.detach()), "recovery_horizon": horizon, "wall_seconds": time.perf_counter() - started}; curves.append(curve); print(json.dumps(curve), flush=True)
    parent_after = _state_hash(model.state_dict())
    for handle in handles: handle.remove()
    if parent_before != parent_after:
        raise Phase3Error("frozen V463 parent changed")
    output.mkdir(parents=True); checkpoint = output / "host_recovery_bridge.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in residual.state_dict().items()}, str(checkpoint), metadata={"format": FORMAT})
    wall = time.perf_counter() - started
    metadata = {
        "format": FORMAT, "status": "TRAINED_DEVELOPMENT_ONLY_NONPROMOTIONAL", "protocol_sha256": protocol_sha,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parent": {"checkpoint_sha256": protocol["parent"]["checkpoint_sha256"], "state_sha256_before": parent_before, "state_sha256_after": parent_after, "mutated": False},
        "initialization": {"checkpoint_sha256": sha256_file(initialization), "state_sha256": initial_residual_hash},
        "supervision": {"artifact_sha256": sha256_file(root / protocol["supervision"]["artifact"]), "records": len(rows), "observations_by_stratum": dict(sorted(counts.items())), "teacher_tokens_seen": teacher_tokens},
        "bridge": {"parameters": EXPECTED_PARAMETERS, "weak_capabilities": WEAK_CAPABILITIES, "source_parameters_copied": 0},
        "training": {"device": "cuda", "seed": seed, "steps": int(cfg["steps"]), "observations": sum(counts.values()), "autonomous_prefix_tokens_seen": recovery_prefix_tokens, "recovery_batches": recovery_batches, "recovery_by_stratum": dict(sorted(recovery_counts.items())), "recovery_horizon_batches": dict(sorted(recovery_horizon_counts.items())), "record_sequence_sha256": record_sequence.hexdigest(), "wall_seconds": wall, "active_parameter_seconds": EXPECTED_PARAMETERS * wall, "peak_process_rss_bytes": int(peak_rss), "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "curves": curves},
        "teacher_present_during_training": False, "teacher_present_at_inference": False, "source_blocks_retained": 0, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(); _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n"); return metadata


@torch.inference_mode()
def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable host recovery evaluation exists: {output}")
    metadata = _json(candidate / "metadata.json"); checkpoint = candidate / metadata["checkpoint"]["path"]
    if metadata["protocol_sha256"] != protocol_sha or sha256_file(checkpoint) != metadata["checkpoint"]["sha256"]: raise Phase3Error("host recovery lineage changed")
    device = torch.device("cuda"); model, tokenizer, _ = _load_parent(root, protocol, device); residual = SharedWeakResidual().to(device); residual.load_state_dict(load_file(str(checkpoint), device="cuda"), strict=True); residual.eval(); handles = _attach(model, residual)
    router, router_tokenizer, router_protocol = _load_router(root, protocol); probes = development_probes(root / protocol["development"]["catalog_path"])
    teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["development"]["teacher_reference"]).open(encoding="utf-8"))}; parent_rows = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["parent"]["development_outputs"]).open(encoding="utf-8"))}
    rows = []; started = time.perf_counter()
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"]); routed, details = sparse._route(router, router_tokenizer, router_protocol, prompt); value, tokens, task_route = _generate_enforced(model, tokenizer, prompt, int(probe["max_new_tokens"]), routed, device); capability = str(probe["canonical_capability"])
        rows.append({"probe_id": str(probe["probe_id"]), "capability": capability, "output": value, "output_token_ids": tokens, "automatic_capability_route": routed, "capability_route_correct": routed == capability, "task_route": task_route, "weak_route_active": routed in WEAK_CAPABILITIES, "router_segment_count": len(details), "strong_parent_output_exact": None if capability in WEAK_CAPABILITIES else value == str(parent_rows[str(probe["probe_id"])]["output"]), "functional_pass_v1": evaluate_functional(value, probe["evaluator"]), "functional_pass_v2": evaluate_functional_v2(value, probe["evaluator"], capability), "repetition_collapse_v2": repetition_collapse_v2(value)})
        if (index + 1) % 100 == 0: print(json.dumps({"evaluated": index + 1}), flush=True)
    for handle in handles: handle.remove()
    output.mkdir(parents=True); raw = output / "development_outputs.jsonl"; raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]; v1 = sum(row["functional_pass_v1"] for row in values); v2 = sum(row["functional_pass_v2"] for row in values); per[capability] = {"passes_v1": v1, "passes_v2": v2, "observations": len(values), "v2_collapses": sum(row["repetition_collapse_v2"] for row in values), "wilson_v1": wilson(v1, len(values))}
    probe_by_id = {str(row["probe_id"]): row for row in probes}; paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass_v1"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_by_id[row["probe_id"]]["evaluator"])} for row in rows]; relative = paired_stratified_bootstrap(paired, replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]), seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    a = protocol["absolute_screen"]; collapses = sum(row["repetition_collapse_v2"] for row in rows); strong = [row for row in rows if row["capability"] not in WEAK_CAPABILITIES]
    gates = {"qualified_router_exact": all(row["capability_route_correct"] for row in rows), "strong_routes_byte_exact_to_v463": all(row["strong_parent_output_exact"] is True for row in strong), "per_capability_functional_v1": all(value["wilson_v1"]["point"] >= float(a["per_capability_functional_point_estimate_minimum"]) and value["wilson_v1"]["lower_95"] >= float(a["per_capability_functional_wilson_lower_minimum"]) for value in per.values()), "critical_capabilities_v1": all(per[name]["wilson_v1"]["point"] >= float(a["critical_point_minimum"]) and per[name]["wilson_v1"]["lower_95"] >= float(a["critical_wilson_lower_minimum"]) for name in ("prompt_grounding", "instruction_following", "abstention")), "zero_v2_repetition_collapses": collapses <= int(a["repetition_collapse_v2_count_maximum"]), "teacher_relative_noninferiority_v1": relative["lower_95"] >= float(protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]), "frozen_parent": metadata["parent"]["mutated"] is False, "final_test_not_accessed": True}
    passed = all(gates.values()); result = {"format": "abi-capability-compiler-phase3-host-recovery-bridge-result/1", "status": "PASS_INITIAL_HOST_RECOVERY_SCREEN_REPLICATION_RUNTIME_OPEN" if passed else "FAIL_HOST_RECOVERY_SUCCESSOR_CLOSED", "protocol_sha256": protocol_sha, "checkpoint_sha256": metadata["checkpoint"]["sha256"], "functional_passes_v1": sum(row["functional_pass_v1"] for row in rows), "functional_passes_v2": sum(row["functional_pass_v2"] for row in rows), "observations": len(rows), "per_capability": per, "repetition_collapses_v2": collapses, "strong_routes_exact": sum(row["strong_parent_output_exact"] is True for row in strong), "strong_route_observations": len(strong), "router_correct": sum(row["capability_route_correct"] for row in rows), "teacher_comparison_v1": relative, "gates": gates, "passed": passed, "raw_outputs_sha256": sha256_file(raw), "evaluation_wall_seconds": time.perf_counter() - started, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False, "claim_boundary": "Single development-only host-conformant all-strata recovery successor; replication, CPU runtime, TTFT, RSS, final quality, minimum information, and Phase 3 remain unproven."}; result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_HOST_RECOVERY_BRIDGE_PROTOCOL_V483.json"); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); train_parser = sub.add_parser("train"); train_parser.add_argument("--output-dir", required=True); eval_parser = sub.add_parser("evaluate"); eval_parser.add_argument("--candidate-dir", required=True); eval_parser.add_argument("--output-dir", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = root / args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else train(root, protocol, root / args.output_dir) if args.command == "train" else evaluate(root, protocol, root / args.candidate_dir, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
