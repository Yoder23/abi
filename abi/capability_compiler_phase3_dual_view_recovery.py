"""Dual-view prompt-invariant recovery successor warm-started from V484."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _write_immutable
from .capability_compiler_phase3_host_recovery_bridge import _artifact_rows, evaluate as base_evaluate, load_protocol as base_load_protocol
from .capability_compiler_phase3_targeted_recovery_bridge import _autonomous_prefixes, _batch_with_prefixes, _load_parent, _weak_routes
from .capability_compiler_phase3_weak_residual import EXPECTED_PARAMETERS, SharedWeakResidual, WEAK_CAPABILITIES, _attach, _parameter_count, _set_routes, _state_hash
from .layercake_host import _equal_record_prompt_overlap_ce


META_MARKERS = ("<prior_answer>", "<machine_requirements>", "Repair one answer")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol, protocol_sha = base_load_protocol(root, path)
    if (
        protocol.get("protocol_id") != "abi-capability-compiler-phase3-dual-view-recovery-v487"
        or tuple(protocol.get("training", {}).get("prompt_views", ())) != ("host_projected", "source_wrapped")
        or int(protocol.get("training", {}).get("recovery_view_strata", -1)) != 32
    ):
        raise Phase3Error("dual-view recovery governance changed")
    return protocol, protocol_sha


def dual_examples(rows: Sequence[Mapping[str, Any]], catalog: Sequence[Mapping[str, Any]], tokenizer: Any, max_tokens: int) -> list[dict[str, Any]]:
    eos = int(tokenizer.eos_token_id)
    probes = {str(row["probe_id"]): row for row in catalog}
    result = []
    for row in rows:
        probe = probes[str(row["probe_id"])]
        prompts = {"host_projected": str(row["host_prompt"]), "source_wrapped": str(probe["prompt"])}
        if any(marker in prompts["source_wrapped"] for marker in META_MARKERS):
            raise Phase3Error("teacher repair meta-prompt crossed dual-view firewall")
        for view, prompt in prompts.items():
            prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
            response_ids = [int(value) for value in tokenizer.encode(str(row["output"]), add_special_tokens=False)] + [eos]
            available = max_tokens - len(prompt_ids)
            if available < 2:
                raise Phase3Error("dual-view record exceeds model context")
            response_ids = response_ids[:available]
            if response_ids[-1] != eos:
                response_ids[-1] = eos
            result.append({"record_id": f"{row['record_id']}:{view}", "source_record_id": str(row["record_id"]), "capability": str(row["capability"]), "builder": int(row["builder"]), "view": view, "route": CAPABILITY_TO_ROUTE[str(row["capability"])], "input_ids": prompt_ids + response_ids, "labels": [-100] * len(prompt_ids) + response_ids, "prompt_tokens": len(prompt_ids), "response_tokens": len(response_ids)})
    return result


class DualViewSampler:
    def __init__(self, rows: Sequence[Mapping[str, Any]], seed: int):
        self.groups = {(capability, builder, view): [row for row in rows if row["capability"] == capability and int(row["builder"]) == builder and row["view"] == view] for capability in WEAK_CAPABILITIES for builder in range(4) for view in ("host_projected", "source_wrapped")}
        if set(map(len, self.groups.values())) != {80}:
            raise Phase3Error("dual-view sampler depth changed")
        self.recovery_strata = tuple(self.groups)
        self.rng = random.Random(seed); self.recovery_index = 0

    def teacher_forced_batch(self) -> list[Mapping[str, Any]]:
        return [self.rng.choice(self.groups[key]) for key in self.recovery_strata]

    def recovery_batch(self, size: int) -> list[Mapping[str, Any]]:
        result = []
        for _ in range(size):
            key = self.recovery_strata[self.recovery_index % len(self.recovery_strata)]; self.recovery_index += 1; result.append(self.rng.choice(self.groups[key]))
        return result


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path); model, tokenizer, _ = _load_parent(root, protocol, torch.device("cpu")); rows = _artifact_rows(root / protocol["supervision"]["artifact"]); catalog = _json(root / protocol["supervision"]["source_catalog"])["probes"]; examples = dual_examples(rows, catalog, tokenizer, int(protocol["training"]["max_tokens"])); residual = SharedWeakResidual(); residual.load_state_dict(load_file(str(root / protocol["initialization"]["checkpoint"]), device="cpu"), strict=True)
    return {"status": "PASS_PREFLIGHT", "protocol_sha256": protocol_sha, "frozen_parent_parameters": sum(value.numel() for value in model.parameters()), "bridge_parameters": _parameter_count(residual), "source_records": len(rows), "dual_view_examples": len(examples), "view_strata": 32, "final_test_accessed": False}


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable dual-view candidate exists: {output}")
    if not torch.cuda.is_available(): raise Phase3Error("dual-view CUDA unavailable")
    cfg = protocol["training"]; seed = int(cfg["seed"]); set_determinism(seed); device = torch.device("cuda"); model, tokenizer, _ = _load_parent(root, protocol, device); residual = SharedWeakResidual().to(device); initialization = root / protocol["initialization"]["checkpoint"]; residual.load_state_dict(load_file(str(initialization), device="cuda"), strict=True); initial_hash = _state_hash(residual.state_dict()); handles = _attach(model, residual)
    rows = _artifact_rows(root / protocol["supervision"]["artifact"]); catalog = _json(root / protocol["supervision"]["source_catalog"])["probes"]; examples = dual_examples(rows, catalog, tokenizer, int(cfg["max_tokens"])); sampler = DualViewSampler(examples, seed); optimizer = torch.optim.AdamW(residual.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(cfg["weight_decay"])); parent_before = _state_hash(model.state_dict()); process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats(); counts = Counter(); recovery_counts = Counter(); horizon_counts = Counter(); sequence = hashlib.sha256(); teacher_tokens = prefix_tokens = recovery_batches = 0; curves = []; started = time.perf_counter()
    for step in range(1, int(cfg["steps"]) + 1):
        selected = sampler.teacher_forced_batch(); prefixes = [[] for _ in selected]; horizon = None
        if step >= int(cfg["recovery_start_step"]) and (step - int(cfg["recovery_start_step"])) % int(cfg["recovery_interval"]) == 0:
            recovery = sampler.recovery_batch(int(cfg["recovery_batch_size"])); horizons = tuple(int(value) for value in cfg["recovery_horizons"]); horizon = horizons[recovery_batches % len(horizons)]; generated = _autonomous_prefixes(model, tokenizer, recovery, horizon, device); by_stratum = {(str(row["capability"]), int(row["builder"]), str(row["view"])): (row, prefix) for row, prefix in zip(recovery, generated)}
            selected_keys = {(str(row["capability"]), int(row["builder"]), str(row["view"])): index for index, row in enumerate(selected)}
            for key, replacement in by_stratum.items():
                if key not in selected_keys:
                    raise Phase3Error("paired teacher batch lost a recovery view-stratum")
                index = selected_keys[key]; selected[index], prefixes[index] = replacement
                recovery_counts[f"{key[0]}:{key[1]}:{key[2]}"] += 1
            recovery_batches += 1; horizon_counts[str(horizon)] += 1; prefix_tokens += sum(len(value) for value in generated)
        ids, labels, attention, prompt_lengths, routes = _batch_with_prefixes(selected, int(tokenizer.eos_token_id), device, prefixes); _set_routes(model, _weak_routes(selected, device)); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=routes, use_cache=False); loss = _equal_record_prompt_overlap_ce(result["logits"], labels, ids, prompt_lengths, overlap_weight=float(cfg["prompt_overlap_weight"]))
        loss.backward(); torch.nn.utils.clip_grad_norm_(residual.parameters(), float(cfg["gradient_clip_norm"])); optimizer.step()
        for row in selected:
            key = f"{row['capability']}:{row['builder']}:{row['view']}"; counts[key] += 1; teacher_tokens += int(row["response_tokens"]); sequence.update((str(row["record_id"]) + "\n").encode())
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {"step": step, "teacher_forced_view": "both_paired", "language_loss": float(loss.detach()), "recovery_horizon": horizon, "wall_seconds": time.perf_counter() - started}; curves.append(curve); print(json.dumps(curve), flush=True)
    parent_after = _state_hash(model.state_dict())
    for handle in handles: handle.remove()
    if parent_before != parent_after: raise Phase3Error("frozen V463 parent changed")
    output.mkdir(parents=True); checkpoint = output / "dual_view_recovery_bridge.safetensors"; save_file({name: value.detach().cpu().contiguous() for name, value in residual.state_dict().items()}, str(checkpoint), metadata={"format": protocol["format"]}); wall = time.perf_counter() - started
    metadata = {"format": protocol["format"], "status": "TRAINED_DEVELOPMENT_ONLY_NONPROMOTIONAL", "protocol_sha256": protocol_sha, "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size}, "parent": {"checkpoint_sha256": protocol["parent"]["checkpoint_sha256"], "state_sha256_before": parent_before, "state_sha256_after": parent_after, "mutated": False}, "initialization": {"checkpoint_sha256": sha256_file(initialization), "state_sha256": initial_hash}, "supervision": {"artifact_sha256": sha256_file(root / protocol["supervision"]["artifact"]), "source_catalog_sha256": sha256_file(root / protocol["supervision"]["source_catalog"]), "source_records": len(rows), "dual_view_examples": len(examples), "observations_by_view_stratum": dict(sorted(counts.items())), "teacher_tokens_seen": teacher_tokens}, "bridge": {"parameters": EXPECTED_PARAMETERS, "weak_capabilities": WEAK_CAPABILITIES, "source_parameters_copied": 0}, "training": {"device": "cuda", "seed": seed, "steps": int(cfg["steps"]), "observations": sum(counts.values()), "autonomous_prefix_tokens_seen": prefix_tokens, "recovery_batches": recovery_batches, "recovery_by_view_stratum": dict(sorted(recovery_counts.items())), "recovery_horizon_batches": dict(sorted(horizon_counts.items())), "record_sequence_sha256": sequence.hexdigest(), "wall_seconds": wall, "active_parameter_seconds": EXPECTED_PARAMETERS * wall, "peak_process_rss_bytes": int(peak_rss), "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "curves": curves}, "teacher_present_during_training": False, "teacher_present_at_inference": False, "source_blocks_retained": 0, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False}; metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(); _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n"); return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_DUAL_VIEW_RECOVERY_PROTOCOL_V487.json"); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); train_parser = sub.add_parser("train"); train_parser.add_argument("--output-dir", required=True); eval_parser = sub.add_parser("evaluate"); eval_parser.add_argument("--candidate-dir", required=True); eval_parser.add_argument("--output-dir", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = root / args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else train(root, protocol, root / args.output_dir) if args.command == "train" else base_evaluate(root, protocol, root / args.candidate_dir, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
