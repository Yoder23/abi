"""Bridge-only LayerCake v4 candidate over the verified lexical substrate."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, repetition_collapse, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable
from .capability_compiler_phase3_bpe_core_analysis import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_segment_router import _semantic_segments
from .capability_compiler_phase3_teacher_native_core import BOS_ID, PAD_ID, _collate, _examples, _json, _layercake_api, _model, _tokenizer, controlled_prompt


FORMAT = "abi-capability-compiler-phase3-lexical-substrate-core/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_INITIAL_BRIDGE_ONLY_SCREEN" or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("training", {}).get("device") != "cuda":
        raise Phase3Error("lexical substrate core governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"lexical substrate core binding changed: {relative}")
    return protocol, sha256_file(path)


def initialize_substrate(model, substrate: Mapping[str, torch.Tensor]) -> None:
    if substrate["input_embedding_rows_fp16"].shape != (32011, 192) or substrate["output_head_rows_fp16"].shape != (32011, 192):
        raise Phase3Error("lexical substrate shape changed")
    with torch.no_grad():
        model.lexeme_embedding.weight[4:].copy_(substrate["input_embedding_rows_fp16"].to(model.lexeme_embedding.weight))
        model.fixed_output.weight[4:].copy_(substrate["output_head_rows_fp16"].to(model.fixed_output.weight))
    model.lexeme_embedding.weight.requires_grad_(False)
    model.fixed_output.weight.requires_grad_(False)


def _build(root: Path, protocol: Mapping[str, Any], device: torch.device):
    model_type, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    model = _model(protocol, tokenizer, model_type).to(device)
    substrate = load_file(str((root / protocol["substrate"]["path"]).resolve()), device="cpu")
    initialize_substrate(model, substrate)
    return model, tokenizer, substrate


def inventory(root: Path, path: Path) -> Mapping[str, Any]:
    protocol, protocol_hash = load_protocol(root, path)
    set_determinism(int(protocol["training"]["seed"]))
    model, tokenizer, substrate = _build(root, protocol, torch.device("cpu"))
    examples = _examples(root, protocol, tokenizer)
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    imported = sum(value.numel() for value in substrate.values())
    if trainable != int(protocol["training"]["trainable_parameters"]) or imported != int(protocol["substrate"]["imported_parameters"]):
        raise Phase3Error("lexical substrate parameter accounting changed")
    return {"status": "PASS", "protocol_sha256": protocol_hash, "records": len(examples), "deployed_parameters": model.parameter_count(), "trainable_parameters": trainable, "imported_parameters": imported, "frozen_parameter_values": model.parameter_count() - trainable, "maximum_source_actions": max(len(row["source_ids"]) for row in examples), "maximum_target_actions": max(len(row["target_actions"]) for row in examples), "teacher_model_loaded": False, "final_test_accessed": False}


def train(root: Path, path: Path, output: Path) -> Mapping[str, Any]:
    protocol, protocol_hash = load_protocol(root, path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("lexical substrate core output exists or CUDA unavailable")
    cfg = protocol["training"]
    set_determinism(int(cfg["seed"]))
    model, tokenizer, substrate = _build(root, protocol, torch.device("cuda"))
    examples = _examples(root, protocol, tokenizer)
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = _BalancedSampler(examples, int(cfg["seed"]))
    curves = []
    sampled = Counter()
    sequence = hashlib.sha256()
    process = psutil.Process()
    peak = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    successful = skipped = 0
    while successful < int(cfg["steps"]):
        batch = sampler.batch(int(cfg["batch_size"]))
        source, targets = _collate(batch, torch.device("cuda"))
        previous = torch.full_like(targets, PAD_ID)
        previous[:, 0] = BOS_ID
        if targets.shape[1] > 1:
            previous[:, 1:] = torch.where(targets[:, :-1].ge(0), targets[:, :-1], torch.full_like(targets[:, :-1], PAD_ID))
        while True:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                log_probs = model.action_log_probs(source, previous)["log_probs"]
                loss = F.nll_loss(log_probs.float().reshape(-1, log_probs.shape[-1]), targets.reshape(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < before:
                skipped += 1
                continue
            break
        successful += 1
        for row in batch:
            sampled[row["capability"]] += 1
            sequence.update(row["record_id"].encode() + b"\n")
        peak = max(peak, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            value = {"step": successful, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - started}
            curves.append(value)
            print(json.dumps(value), flush=True)
    input_unchanged = torch.equal(model.lexeme_embedding.weight[4:].detach().cpu(), substrate["input_embedding_rows_fp16"].float())
    output_unchanged = torch.equal(model.fixed_output.weight[4:].detach().cpu(), substrate["output_head_rows_fp16"].float())
    if not input_unchanged or not output_unchanged:
        raise Phase3Error("frozen imported substrate changed during training")
    output.mkdir(parents=True)
    checkpoint = output / "model.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(checkpoint))
    router_source = (root / protocol["router"]["checkpoint_path"]).resolve()
    router_path = output / "router.safetensors"
    router_path.write_bytes(router_source.read_bytes())
    tokenizer_path = output / "tokenizer.json"
    _write_immutable(tokenizer_path, json.dumps(tokenizer.canonical_dict(), indent=2, sort_keys=True).encode() + b"\n")
    config_path = output / "model_config.json"
    _write_immutable(config_path, json.dumps({**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size}, indent=2, sort_keys=True).encode() + b"\n")
    metadata = {"format": FORMAT, "status": "TRAINED_INITIAL_BRIDGE_ONLY_SCREEN", "protocol_sha256": protocol_hash, "seed": int(cfg["seed"]), "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size}, "router": {"path": router_path.name, "sha256": sha256_file(router_path), "bytes": router_path.stat().st_size, "parameters": 1058040}, "tokenizer": {"path": tokenizer_path.name, "sha256": sha256_file(tokenizer_path), "canonical_sha256": tokenizer.hash(), "vocabulary": tokenizer.vocab_size}, "model_config": {"path": config_path.name, "sha256": sha256_file(config_path), "deployed_parameters": model.parameter_count(), "trainable_parameters": sum(value.numel() for value in trainable)}, "substrate": {"artifact_sha256": protocol["substrate"]["sha256"], "imported_parameters": sum(value.numel() for value in substrate.values()), "input_table_unchanged": input_unchanged, "output_table_unchanged": output_unchanged, "deployed": True}, "training": {"steps": successful, "batch_size": int(cfg["batch_size"]), "wall_seconds": time.perf_counter() - started, "skipped_amp_steps": skipped, "record_sequence_sha256": sequence.hexdigest(), "sampled_by_capability": dict(sorted(sampled.items())), "peak_process_rss_bytes": peak, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "curves": curves}, "imported_information": {"records": 7000, "teacher_input_tokens": 576925, "teacher_output_tokens": 215647, "stored_logits": 0, "stored_hidden_activations": 0, "source_parameters_copied": 0, "final_imported_substrate_parameters": sum(value.numel() for value in substrate.values()), "source_blocks_retained": 0}, "teacher_present_at_inference": False, "receiver_training_steps": 0, "layercake_host_interface": protocol["layercake_host"]["interface"], "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False, "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)}}
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


def _load_candidate(root: Path, protocol: Mapping[str, Any], candidate: Path):
    model_type, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = tokenizer_type.from_document(_json(candidate / "tokenizer.json"))
    model = model_type(**_json(candidate / "model_config.json")).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(candidate / "model.safetensors"), device="cuda"), strict=True)
    return model.cuda().eval(), tokenizer


def evaluate(root: Path, path: Path, candidate: Path, output: Path) -> Mapping[str, Any]:
    protocol, protocol_hash = load_protocol(root, path)
    metadata = _json(candidate / "metadata.json")
    if output.exists() or metadata.get("protocol_sha256") != protocol_hash or sha256_file(candidate / "model.safetensors") != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("lexical substrate evaluation identity failed")
    model, _ = _load_candidate(root, protocol, candidate)
    substrate = load_file(str((root / protocol["substrate"]["path"]).resolve()), device="cpu")
    if not torch.equal(model.lexeme_embedding.weight[4:].cpu(), substrate["input_embedding_rows_fp16"].float()) or not torch.equal(model.fixed_output.weight[4:].cpu(), substrate["output_head_rows_fp16"].float()):
        raise Phase3Error("deployed substrate identity changed")
    router_protocol = _json(root / protocol["router"]["protocol_path"])
    router, router_tokenizer = sparse._load(root, router_protocol, (root / protocol["router"]["candidate_dir"]).resolve())
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"])
        route, _ = sparse._route(router, router_tokenizer, router_protocol, prompt)
        error = None
        try:
            value = model.generate_bytes(controlled_prompt(route, _semantic_segments(prompt)[-1]), maximum_actions=min(int(probe["max_new_tokens"]), int(protocol["architecture"]["maximum_target_actions"]))).decode("utf-8", errors="strict")
        except Exception as exc:
            value = ""
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"probe_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]), "predicted_route": route, "route_correct": route == str(probe["canonical_capability"]), "output": value, "generation_error": error, "functional_pass": evaluate_functional(value, probe["evaluator"]), "repetition_collapse": repetition_collapse(value)})
        if (index + 1) % 100 == 0:
            print(json.dumps({"evaluated": index + 1}), flush=True)
    output.mkdir(parents=True)
    raw = output / "development_outputs.jsonl"
    raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passes = sum(row["functional_pass"] for row in values)
        per[capability] = {"passes": passes, "observations": len(values), "collapses": sum(row["repetition_collapse"] for row in values), "wilson": wilson(passes, len(values))}
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))}
    probe_map = {str(probe["probe_id"]): probe for probe in probes}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_map[row["probe_id"]]["evaluator"])} for row in rows]
    comparison = paired_stratified_bootstrap(paired, replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]), seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    gate = protocol["absolute_screen"]
    ordinary = all(value["wilson"]["point"] >= gate["per_capability_functional_point_estimate_minimum"] and value["wilson"]["lower_95"] >= gate["per_capability_functional_wilson_lower_minimum"] for value in per.values())
    critical = all(per[name]["wilson"]["point"] >= gate["critical_point_minimum"] and per[name]["wilson"]["lower_95"] >= gate["critical_wilson_lower_minimum"] for name in ("prompt_grounding", "instruction_following", "abstention"))
    collapses = sum(row["repetition_collapse"] for row in rows)
    errors = sum(row["generation_error"] is not None for row in rows)
    gates = {"per_capability_functional": ordinary, "critical_capabilities": critical, "zero_repetition_collapses": collapses == 0, "zero_generation_errors": errors == 0, "router_accuracy": sum(row["route_correct"] for row in rows) == len(rows), "teacher_relative_noninferiority": comparison["lower_95"] >= protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]}
    passed = all(gates.values())
    decision = {"format": "abi-capability-compiler-phase3-lexical-substrate-core-decision/1", "status": "PASS_INITIAL_SCREEN_REPLICATIONS_AND_HOST_CERTIFICATION_REQUIRED" if passed else "FAIL_INITIAL_SCREEN_BRANCH_CLOSED", "protocol": {"path": path.name, "sha256": protocol_hash}, "checkpoint_sha256": metadata["checkpoint"]["sha256"], "functional_passes": sum(row["functional_pass"] for row in rows), "observations": len(rows), "per_capability": per, "repetition_collapses": collapses, "generation_errors": errors, "route_correct": sum(row["route_correct"] for row in rows), "teacher_comparison": comparison, "gates": gates, "initial_screen_pass": passed, "promotion_eligible": passed, "outputs_sha256": sha256_file(raw), "evaluation_wall_seconds": time.perf_counter() - started, "teacher_present_at_inference": False, "imported_substrate_unchanged": True, "phase3_certified": False, "final_test_accessed": False, "next_step": "Preregister two paired seeds and same-artifact CPU host certification." if passed else "Preserve failure; no nearby lexical projection variant."}
    decision["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(decision)).hexdigest()
    _write_immutable(output / "decision.json", json.dumps(decision, indent=2, sort_keys=True).encode() + b"\n")
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train", "evaluate"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_LEXICAL_SUBSTRATE_CORE_PROTOCOL_V86.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_lexical_substrate_core/development_v86/B0-seed240075")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_lexical_substrate_core/evaluation_v86/B0-seed240075")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    path = (root / args.protocol).resolve()
    result = inventory(root, path) if args.command == "inventory" else train(root, path, (root / args.candidate_dir).resolve()) if args.command == "train" else evaluate(root, path, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
