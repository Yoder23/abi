"""Train and screen one matched-capacity selective-boundary BPE core."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, repetition_collapse, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_bpe_core import _json
from .capability_compiler_phase3_bpe_core_analysis import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_bpe_pointer_resilience import _pointer_targets
from .capability_compiler_phase3_route_bridge import _collate, _select_controls, BOS_ID, PAD_ID
from .capability_compiler_phase3_segment_router import _semantic_segments
from .capability_compiler_phase3_unicode_span_copy_feasibility import _targeted_rows


FORMAT = "abi-capability-compiler-phase3-selective-bpe-core/1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_INITIAL_GPU_SCREEN" or protocol.get("training", {}).get("device") != "cuda" or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("promotion_eligible") is not False:
        raise Phase3Error("selective BPE core governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"selective BPE core binding changed: {relative}")
    return protocol, sha256_file(path)


def _types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.portable_token_plan import PortableTokenPlan
    from layercake_extensions.selective_boundary_bpe_direct_neural_core import SelectiveBoundaryBpeTokenizer
    return PortableTokenPlan, SelectiveBoundaryBpeTokenizer


def _tokenizer(root: Path, protocol: Mapping[str, Any], tokenizer_type: Any):
    raw = json.loads((root / protocol["tokenizer"]["path"]).read_text(encoding="utf-8"))
    tokenizer = tokenizer_type(raw)
    if tokenizer.hash() != protocol["tokenizer"]["canonical_sha256"] or tokenizer.vocab_size != int(protocol["tokenizer"]["fixed_actions"]):
        raise Phase3Error("selective BPE tokenizer identity changed")
    return tokenizer


def _examples(root: Path, protocol: Mapping[str, Any], tokenizer: Any):
    original = load_phase1_ir(root / protocol["phase1_ir"])
    targeted = _targeted_rows(root / protocol["targeted_ir"])
    controls = _select_controls(original, tokenizer)
    by_capability = {capability: controls[index] for index, capability in enumerate(CAPABILITIES)}
    examples = []
    rows = []
    for row in original:
        body = "\n".join(str(row["normalized_acquisition_prompt"]).splitlines()[1:]).strip()
        rows.append((str(row["ir_record_id"]), str(row["capability"]), body, str(row["normalized_output"]), int(row["authoritative_teacher_tokens"])))
    for row in targeted:
        rows.append((str(row["ir_record_id"]), str(row["capability"]), str(row["host_conformant_acquisition_prompt"]), str(row["normalized_output"]), int(row["authoritative_teacher_tokens"])))
    for record_id, capability, body, output, teacher_tokens in rows:
        control_id, control_piece = by_capability[capability]
        body_pieces = tokenizer.split("\n" + body)
        source_lexemes = [control_piece] + body_pieces
        source_ids = [control_id] + [tokenizer.lexeme_to_id[piece] for piece in body_pieces]
        target = _pointer_targets(source_lexemes, tokenizer.split(output), tokenizer.vocab_size, tokenizer)
        if len(source_ids) > int(protocol["architecture"]["maximum_source_lexemes"]) or len(target) > int(protocol["architecture"]["maximum_target_actions"]):
            raise Phase3Error(f"selective BPE example exceeds host bound: {record_id}")
        if tokenizer.decode_actions(target, source_lexemes) != output.encode("utf-8"):
            raise Phase3Error(f"selective BPE target is not exact: {record_id}")
        examples.append({"record_id": record_id, "capability": capability, "source_ids": source_ids, "source_lexemes": source_lexemes, "target_actions": target, "pointer_actions": sum(action >= tokenizer.vocab_size for action in target), "teacher_tokens": teacher_tokens, "prompt_bytes": len(body.encode("utf-8")), "output_bytes": len(output.encode("utf-8"))})
    if len(examples) != 14000 or len({row["record_id"] for row in examples}) != 14000:
        raise Phase3Error("selective BPE combined inventory changed")
    return examples, controls


def _model(protocol: Mapping[str, Any], tokenizer: Any, model_type: Any, device: torch.device):
    model = model_type(fixed_vocab_size=tokenizer.vocab_size, **protocol["architecture"]).bind_tokenizer(tokenizer).to(device)
    if model.parameter_count() != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error("selective BPE parameter count changed")
    return model


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path); model_type, tokenizer_type = _types(root, protocol); tokenizer = _tokenizer(root, protocol, tokenizer_type); examples, controls = _examples(root, protocol, tokenizer); model = _model(protocol, tokenizer, model_type, torch.device("cpu"))
    control_doc = [{"capability": capability, "token_id": controls[index][0], "piece_hex": controls[index][1].hex()} for index, capability in enumerate(CAPABILITIES)]
    return {"status": "PASS", "protocol_sha256": protocol_sha, "records": len(examples), "parameters": model.parameter_count(), "vocabulary": tokenizer.vocab_size, "maximum_source_actions": max(len(row["source_ids"]) for row in examples), "maximum_target_actions": max(len(row["target_actions"]) for row in examples), "pointer_actions": sum(row["pointer_actions"] for row in examples), "records_with_pointers": sum(row["pointer_actions"] > 0 for row in examples), "route_control_selection_sha256": hashlib.sha256(canonical_json_bytes(control_doc)).hexdigest(), "all_targets_exact": True, "final_test_accessed": False}


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available(): raise Phase3Error("selective BPE output exists or CUDA unavailable")
    model_type, tokenizer_type = _types(root, protocol); tokenizer = _tokenizer(root, protocol, tokenizer_type); examples, controls = _examples(root, protocol, tokenizer); device = torch.device("cuda"); seed = int(protocol["training"]["seed"]); set_determinism(seed); model = _model(protocol, tokenizer, model_type, device); model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol["training"]["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1); scaler = torch.amp.GradScaler("cuda", enabled=True); sampler = _BalancedSampler(examples, seed); process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats(); successful = skipped = 0; sequence = hashlib.sha256(); sampled = Counter(); curves = []; started = time.perf_counter()
    while successful < int(protocol["training"]["steps"]):
        batch = sampler.batch(int(protocol["training"]["batch_size"])); source, targets = _collate(batch, device); previous = torch.full_like(targets, PAD_ID); previous[:, 0] = BOS_ID
        if targets.shape[1] > 1: previous[:, 1:] = torch.where(targets[:, :-1].ge(0), targets[:, :-1], torch.full_like(targets[:, :-1], PAD_ID))
        while True:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16): log_probs = model.action_log_probs(source, previous)["log_probs"]; loss = F.nll_loss(log_probs.float().reshape(-1, log_probs.shape[-1]), targets.reshape(-1), ignore_index=-100)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < before: skipped += 1; continue
            break
        successful += 1
        for row in batch: sampled[row["capability"]] += 1; sequence.update(row["record_id"].encode() + b"\n")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(protocol["training"]["curve_interval"]) == 0:
            curve = {"step": successful, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - started}; curves.append(curve); print(json.dumps(curve), flush=True)
    output.mkdir(parents=True); checkpoint = output / "model.safetensors"; save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(checkpoint)); router_source = root / protocol["router"]["checkpoint_path"]; (output / "router.safetensors").write_bytes(router_source.read_bytes()); _write_json(output / "tokenizer.json", tokenizer.canonical_dict()); _write_json(output / "model_config.json", {**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size}); control_doc = [{"capability": capability, "token_id": controls[index][0], "piece_hex": controls[index][1].hex()} for index, capability in enumerate(CAPABILITIES)]; _write_json(output / "route_controls.json", {"controls": control_doc})
    metadata = {"format": "abi-capability-compiler-phase3-selective-bpe-candidate/1", "status": "TRAINED_INITIAL_DEVELOPMENT_SCREEN", "protocol_sha256": protocol_sha, "seed": seed, "checkpoint": {"path": "model.safetensors", "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size}, "tokenizer": {"sha256": sha256_file(output / "tokenizer.json"), "canonical_sha256": tokenizer.hash(), "vocabulary": tokenizer.vocab_size}, "parameters": {"generator": model.parameter_count(), "router": int(protocol["router"]["parameters"])}, "pointer_supervision": {"actions": sum(row["pointer_actions"] for row in examples), "records": sum(row["pointer_actions"] > 0 for row in examples)}, "training": {"steps": successful, "batch_size": int(protocol["training"]["batch_size"]), "wall_seconds": time.perf_counter() - started, "skipped_amp_steps": skipped, "record_sequence_sha256": sequence.hexdigest(), "sampled_by_capability": dict(sorted(sampled.items())), "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "curves": curves}, "imported_information": {"records": len(examples), "raw_prompt_bytes": sum(row["prompt_bytes"] for row in examples), "teacher_output_bytes": sum(row["output_bytes"] for row in examples), "authoritative_teacher_tokens": sum(row["teacher_tokens"] for row in examples), "stored_logits": 0, "stored_activations": 0, "source_parameters_copied": 0}, "teacher_present_at_inference": False, "source_blocks_retained": 0, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False, "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)}}; metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(); _write_json(output / "metadata.json", metadata); return metadata


def _load_candidate(root: Path, protocol: Mapping[str, Any], candidate: Path):
    model_type, tokenizer_type = _types(root, protocol); tokenizer = tokenizer_type.from_document(_json(candidate / "tokenizer.json")); model = _model(protocol, tokenizer, model_type, torch.device("cuda")); model.load_state_dict(load_file(str(candidate / "model.safetensors"), device="cuda"), strict=True); model.eval(); return model, tokenizer


@torch.inference_mode()
def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error("selective BPE evaluation exists")
    metadata = _json(candidate / "metadata.json")
    if metadata.get("protocol_sha256") != protocol_sha or sha256_file(candidate / "model.safetensors") != metadata["checkpoint"]["sha256"]: raise Phase3Error("selective BPE candidate identity changed")
    model, tokenizer = _load_candidate(root, protocol, candidate); router_protocol = _json(root / protocol["router"]["protocol_path"]); router, router_tokenizer = sparse._load(root, router_protocol, root / protocol["router"]["candidate_dir"]); controls = {row["capability"]: bytes.fromhex(row["piece_hex"]).decode("utf-8") for row in _json(candidate / "route_controls.json")["controls"]}; probes = development_probes(root / protocol["development_catalog"]); rows = []; started = time.perf_counter()
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"]); route, _ = sparse._route(router, router_tokenizer, router_protocol, prompt); controlled = controls[route] + "\n" + _semantic_segments(prompt)[-1]; error = None
        try: value = model.generate_bytes(controlled, maximum_actions=min(int(probe["max_new_tokens"]), int(protocol["architecture"]["maximum_target_actions"]))).decode("utf-8", errors="strict")
        except Exception as exc: value = ""; error = f"{type(exc).__name__}: {exc}"
        rows.append({"probe_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]), "predicted_route": route, "route_correct": route == str(probe["canonical_capability"]), "output": value, "generation_error": error, "functional_pass": evaluate_functional(value, probe["evaluator"]), "repetition_collapse": repetition_collapse(value)})
        if (index + 1) % 100 == 0: print(json.dumps({"evaluated": index + 1}), flush=True)
    output.mkdir(parents=True); raw = output / "development_outputs.jsonl"; raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows)); per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]; passes = sum(row["functional_pass"] for row in values); per[capability] = {"passes": passes, "observations": len(values), "collapses": sum(row["repetition_collapse"] for row in values), "wilson": wilson(passes, len(values))}
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))}; probe_map = {str(row["probe_id"]): row for row in probes}; paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_map[row["probe_id"]]["evaluator"])} for row in rows]; comparison = paired_stratified_bootstrap(paired, replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]), seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"])); gate = protocol["absolute_screen"]; collapses = sum(row["repetition_collapse"] for row in rows); errors = sum(row["generation_error"] is not None for row in rows); gates = {"per_capability_functional": all(value["wilson"]["point"] >= gate["per_capability_functional_point_estimate_minimum"] and value["wilson"]["lower_95"] >= gate["per_capability_functional_wilson_lower_minimum"] for value in per.values()), "critical_capabilities": all(per[value]["wilson"]["point"] >= gate["critical_point_minimum"] and per[value]["wilson"]["lower_95"] >= gate["critical_wilson_lower_minimum"] for value in ("prompt_grounding", "instruction_following", "abstention")), "zero_repetition_collapses": collapses == 0, "zero_generation_errors": errors == 0, "router_accuracy": sum(row["route_correct"] for row in rows) == len(rows), "teacher_relative_noninferiority": comparison["lower_95"] >= protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]}; passed = all(gates.values()); decision = {"format": "abi-capability-compiler-phase3-selective-bpe-decision/1", "status": "PASS_INITIAL_SCREEN_REPLICATION_AND_HOST_CERTIFICATION_REQUIRED" if passed else "FAIL_INITIAL_SCREEN_SELECTIVE_BPE_CLOSED", "protocol_sha256": protocol_sha, "checkpoint_sha256": metadata["checkpoint"]["sha256"], "functional_passes": sum(row["functional_pass"] for row in rows), "observations": len(rows), "per_capability": per, "repetition_collapses": collapses, "generation_errors": errors, "route_correct": sum(row["route_correct"] for row in rows), "teacher_comparison": comparison, "gates": gates, "initial_screen_pass": passed, "promotion_eligible": False, "outputs_sha256": sha256_file(raw), "evaluation_wall_seconds": time.perf_counter() - started, "teacher_present_at_inference": False, "phase3_certified": False, "phase4_open": False, "final_test_accessed": False}; decision["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(decision)).hexdigest(); _write_json(output / "decision.json", decision); return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("inventory", "train", "evaluate")); parser.add_argument("--protocol", required=True); parser.add_argument("--candidate-dir", required=True); parser.add_argument("--output-dir", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = root / args.protocol
    result = inventory(root, protocol) if args.command == "inventory" else train(root, protocol, root / args.candidate_dir) if args.command == "train" else evaluate(root, protocol, root / args.candidate_dir, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
