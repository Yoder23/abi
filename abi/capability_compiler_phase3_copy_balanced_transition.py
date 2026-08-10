"""Single copy-balanced conformance repair of the existing V443 transition."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_qualified_transition_control import FROZEN_PREFIXES, _configure_trainable, _state_hash
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_sequence_bridge import _BalancedSampler, _batch, _examples, _generate
from .capability_compiler_phase3_v443_prompt_pointer import _load_v443
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .layercake_host import _equal_record_prompt_overlap_ce


FORMAT = "abi-capability-compiler-phase3-copy-balanced-transition/1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_SINGLE_BOUNDED_REPAIR" or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("nearby_sweeps_authorized") is not False:
        raise Phase3Error("copy-balanced governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"copy-balanced binding changed: {name}")
    return protocol, sha256_file(path)


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, _, _ = _load_v443(root, protocol, torch.device("cpu"))
    trainable = _configure_trainable(model)
    return {"status": "PASS_PREFLIGHT", "protocol_sha256": protocol_sha, "total_parameters": sum(p.numel() for p in model.parameters()), "trainable_existing_parameters": sum(p.numel() for p in trainable), "new_parameters": 0, "final_test_accessed": False}


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable copy-balanced output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("copy-balanced CUDA unavailable")
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model, tokenizer, _ = _load_v443(root, protocol, device)
    trainable = _configure_trainable(model)
    rows = load_phase1_ir(root / protocol["phase1_ir"]["path"])
    examples = _examples(rows, tokenizer, system="A0", seed=seed, max_tokens=int(cfg["max_tokens"]))
    sampler = _BalancedSampler(examples, seed)
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(cfg["weight_decay"]))
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    frozen_before = {name: value for name, value in before.items() if name.startswith(FROZEN_PREFIXES)}
    process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats()
    sampled = Counter(); sequence_sha = hashlib.sha256(); language_tokens = 0; curves = []; started = time.perf_counter()
    model.train()
    for step in range(1, int(cfg["steps"]) + 1):
        selected = sampler.batch(int(cfg["batch_size"]))
        ids, labels, attention, prompt_lengths, routes = _batch(selected, int(tokenizer.eos_token_id), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=routes, use_cache=False)
            language_loss = _equal_record_prompt_overlap_ce(result["logits"], labels, ids, prompt_lengths, overlap_weight=float(cfg["prompt_overlap_weight"]))
            classifier_loss = F.cross_entropy(result["task_logits"].float(), routes)
            loss = language_loss + float(cfg["classifier_loss_weight"]) * classifier_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, float(cfg["gradient_clip_norm"])); optimizer.step()
        for row in selected:
            sampled[str(row["capability"])] += 1; sequence_sha.update(str(row["record_id"]).encode("ascii") + b"\n"); language_tokens += int(row["response_tokens"])
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {"step": step, "language_loss": float(language_loss.detach()), "classifier_loss": float(classifier_loss.detach()), "wall_seconds": time.perf_counter() - started}; curves.append(curve); print(json.dumps(curve), flush=True)
    model.eval()
    after = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    changed = sorted(name for name in before if not torch.equal(before[name], after[name]))
    if not changed or any(name.startswith(FROZEN_PREFIXES) for name in changed):
        raise Phase3Error("copy-balanced repair changed frozen identity substrate")
    frozen_after = {name: value for name, value in after.items() if name.startswith(FROZEN_PREFIXES)}
    if _state_hash(frozen_before) != _state_hash(frozen_after):
        raise Phase3Error("copy-balanced embedding state changed")
    output.mkdir(parents=True); checkpoint = output / "model.safetensors"; save_file(after, str(checkpoint), metadata={"format": FORMAT})
    wall = time.perf_counter() - started
    metadata = {"format": FORMAT, "status": "TRAINED_DEVELOPMENT_ONLY_NONPROMOTIONAL", "protocol_sha256": protocol_sha, "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size}, "parent_checkpoint_sha256": protocol["parent"]["checkpoint_sha256"], "training": {"seed": seed, "steps": int(cfg["steps"]), "batch_size": int(cfg["batch_size"]), "prompt_overlap_weight": float(cfg["prompt_overlap_weight"]), "teacher_response_tokens_seen": language_tokens, "sampled_records_by_capability": dict(sorted(sampled.items())), "successful_record_sequence_sha256": sequence_sha.hexdigest(), "wall_seconds": wall, "peak_process_rss_bytes": int(peak_rss), "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "curves": curves}, "isolation": {"changed_tensor_count": len(changed), "new_parameters": 0, "embedding_state_sha256_before": _state_hash(frozen_before), "embedding_state_sha256_after": _state_hash(frozen_after)}, "teacher_present_at_training_or_inference": False, "source_blocks_retained": 0, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False}
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(); _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n"); return metadata


def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable copy-balanced evaluation exists: {output}")
    metadata = _json(candidate / "metadata.json"); checkpoint = candidate / "model.safetensors"
    if metadata["protocol_sha256"] != protocol_sha or sha256_file(checkpoint) != metadata["checkpoint"]["sha256"]: raise Phase3Error("copy-balanced lineage changed")
    device = torch.device("cuda"); model, tokenizer, _ = _load_v443(root, protocol, device); model.load_state_dict(load_file(str(checkpoint), device="cuda"), strict=True); model.eval()
    probes = development_probes(root / protocol["development"]["catalog_path"]); teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["development"]["teacher_reference"]).open(encoding="utf-8"))}
    rows = []; started = time.perf_counter()
    for index, probe in enumerate(probes):
        value, tokens, route = _generate(model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), device)
        rows.append({"probe_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]), "output": value, "output_token_ids": tokens, "automatic_route": route, "functional_pass": evaluate_functional(value, probe["evaluator"]), "repetition_collapse_v2": repetition_collapse_v2(value)})
        if (index + 1) % 100 == 0: print(json.dumps({"evaluated": index + 1}), flush=True)
    output.mkdir(parents=True); raw = output / "development_outputs.jsonl"; raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]; passes = sum(row["functional_pass"] for row in values); per[capability] = {"passes": passes, "observations": len(values), "v2_collapses": sum(row["repetition_collapse_v2"] for row in values), "wilson": wilson(passes, len(values))}
    probe_by_id = {str(row["probe_id"]): row for row in probes}; paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_by_id[row["probe_id"]]["evaluator"])} for row in rows]; relative = paired_stratified_bootstrap(paired, replicates=10000, seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    a = protocol["absolute_screen"]; collapses = sum(row["repetition_collapse_v2"] for row in rows)
    gates = {"per_capability_functional": all(v["wilson"]["point"] >= a["per_capability_functional_point_estimate_minimum"] and v["wilson"]["lower_95"] >= a["per_capability_functional_wilson_lower_minimum"] for v in per.values()), "critical_capabilities": all(per[n]["wilson"]["point"] >= a["critical_point_minimum"] and per[n]["wilson"]["lower_95"] >= a["critical_wilson_lower_minimum"] for n in ("prompt_grounding", "instruction_following", "abstention")), "zero_v2_repetition_collapses": collapses == 0, "teacher_relative_noninferiority": relative["lower_95"] >= protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"], "zero_new_inference_parameters": metadata["isolation"]["new_parameters"] == 0, "final_test_not_accessed": True}
    passed = all(gates.values()); result = {"format": "abi-capability-compiler-phase3-copy-balanced-transition-result/1", "status": "PASS_INITIAL_COPY_BALANCED_SCREEN" if passed else "FAIL_COPY_BALANCED_REPAIR_CLOSED", "protocol_sha256": protocol_sha, "checkpoint_sha256": metadata["checkpoint"]["sha256"], "functional_passes": sum(row["functional_pass"] for row in rows), "observations": len(rows), "per_capability": per, "repetition_collapses_v2": collapses, "teacher_comparison": relative, "gates": gates, "passed": passed, "raw_outputs_sha256": sha256_file(raw), "evaluation_wall_seconds": time.perf_counter() - started, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False, "claim_boundary": "Single development-only copy-balanced transition repair; replication, runtime, final quality, minimum information, and Phase 3 remain unproven."}; result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_COPY_BALANCED_TRANSITION_PROTOCOL_V458.json"); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); tp = sub.add_parser("train"); tp.add_argument("--output-dir", required=True); ep = sub.add_parser("evaluate"); ep.add_argument("--candidate-dir", required=True); ep.add_argument("--output-dir", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = root / args.protocol; result = preflight(root, protocol) if args.command == "preflight" else train(root, protocol, root / args.output_dir) if args.command == "train" else evaluate(root, protocol, root / args.candidate_dir, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
