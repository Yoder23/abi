"""Unicode-safe BPE pointer-supervision resilience screen."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import time
from typing import Any, Iterable, Mapping, Sequence

import psutil
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, repetition_collapse, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_bpe_core_analysis import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_route_bridge import _base, _collate, _load_candidate, _select_controls, BOS_ID, PAD_ID
from .capability_compiler_phase3_segment_router import _semantic_segments
from .capability_compiler_phase3_bpe_core import _json


FORMAT = "abi-capability-compiler-phase3-bpe-pointer-resilience/1"
IDENTITY_PIECE = re.compile(rb"^[A-Za-z0-9_]+$")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_UTF8_BPE_POINTER_SCREEN"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("training", {}).get("device") != "cuda"
        or protocol.get("promotion_eligible") is not False
        or protocol.get("phase4_status") != "LOCKED"
    ):
        raise Phase3Error("BPE pointer resilience governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"BPE pointer resilience binding changed: {relative}")
    return protocol, sha256_file(path)


def _pointer_targets(source_lexemes: Sequence[bytes], output_lexemes: Sequence[bytes], vocabulary: int, tokenizer: Any) -> list[int]:
    counts = Counter(source_lexemes)
    positions = {piece: index for index, piece in enumerate(source_lexemes) if counts[piece] == 1 and IDENTITY_PIECE.fullmatch(piece) is not None}
    return [vocabulary + positions[piece] if piece in positions else tokenizer.lexeme_to_id[piece] for piece in output_lexemes] + [2]


def _examples(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], list[tuple[int, bytes]]]:
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    controls = _select_controls(rows, tokenizer)
    control_by_capability = {capability: controls[index] for index, capability in enumerate(CAPABILITIES)}
    examples = []
    for row in rows:
        capability = str(row["capability"])
        lines = str(row["normalized_acquisition_prompt"]).splitlines()
        body = "\n".join(lines[1:]).strip()
        control_id, control_piece = control_by_capability[capability]
        body_pieces = tokenizer.split("\n" + body)
        source_lexemes = [control_piece] + body_pieces
        source_ids = [control_id] + [tokenizer.lexeme_to_id[piece] for piece in body_pieces]
        output = str(row["normalized_output"])
        output_lexemes = tokenizer.split(output)
        target = _pointer_targets(source_lexemes, output_lexemes, tokenizer.vocab_size, tokenizer)
        if len(source_ids) > int(protocol["architecture"]["maximum_source_lexemes"]) or len(target) > int(protocol["architecture"]["maximum_target_actions"]):
            raise Phase3Error("BPE pointer example exceeds host bound")
        if tokenizer.decode_actions(target, source_lexemes) != output.encode("utf-8"):
            raise Phase3Error("BPE pointer target does not reconstruct losslessly")
        examples.append({"record_id": str(row["ir_record_id"]), "capability": capability, "source_ids": source_ids, "source_lexemes": source_lexemes, "target_actions": target, "pointer_actions": sum(action >= tokenizer.vocab_size for action in target)})
    if len(examples) != 7000:
        raise Phase3Error("BPE pointer inventory changed")
    return examples, controls


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer = _base(root, protocol, torch.device("cpu"))
    examples, controls = _examples(root, protocol, tokenizer)
    selection = hashlib.sha256(canonical_json_bytes([{"capability": capability, "token_id": controls[index][0], "piece_hex": controls[index][1].hex()} for index, capability in enumerate(CAPABILITIES)])).hexdigest()
    if selection != protocol["route_controls"]["selection_sha256"]:
        raise Phase3Error("route-control selection changed")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "records": len(examples),
        "trainable_parameters": model.parameter_count(),
        "pointer_actions": sum(row["pointer_actions"] for row in examples),
        "records_with_pointer_actions": sum(row["pointer_actions"] > 0 for row in examples),
        "pointer_actions_by_capability": {capability: sum(row["pointer_actions"] for row in examples if row["capability"] == capability) for capability in CAPABILITIES},
        "maximum_source_actions": max(len(row["source_ids"]) for row in examples),
        "maximum_target_actions": max(len(row["target_actions"]) for row in examples),
        "all_targets_losslessly_reconstructed": True,
        "teacher_outputs_added": 0,
        "source_parameters_copied": 0,
        "layercake_host_changed": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("BPE pointer output exists or CUDA unavailable")
    device = torch.device("cuda")
    model, tokenizer = _base(root, protocol, device)
    examples, controls = _examples(root, protocol, tokenizer)
    config = protocol["training"]
    seed = int(config["seed"])
    set_determinism(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = _BalancedSampler(examples, seed)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    successful = skipped = 0
    sequence = hashlib.sha256()
    sampled = Counter()
    curves = []
    started = time.perf_counter()
    while successful < int(config["steps"]):
        batch = sampler.batch(int(config["batch_size"]))
        source, targets = _collate(batch, device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            sequence.update(row["record_id"].encode("utf-8") + b"\n")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(config["curve_interval"]) == 0:
            value = {"step": successful, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - started}
            curves.append(value)
            print(json.dumps(value), flush=True)
    output.mkdir(parents=True)
    checkpoint = output / "model.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(checkpoint))
    router_source = (root / protocol["router"]["checkpoint_path"]).resolve()
    router_target = output / "router.safetensors"
    router_target.write_bytes(router_source.read_bytes())
    tokenizer_path = output / "tokenizer.json"
    _write_json(tokenizer_path, tokenizer.canonical_dict())
    config_path = output / "model_config.json"
    _write_json(config_path, {**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size})
    controls_path = output / "route_controls.json"
    _write_json(controls_path, {"controls": [{"capability": capability, "token_id": controls[index][0], "piece_hex": controls[index][1].hex()} for index, capability in enumerate(CAPABILITIES)]})
    metadata = {
        "format": "abi-capability-compiler-phase3-bpe-pointer-candidate/1",
        "status": "TRAINED_INITIAL_DEVELOPMENT_SCREEN",
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "checkpoint": {"path": "model.safetensors", "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "router": {"sha256": sha256_file(router_target), "parameters": int(protocol["router"]["parameters"])},
        "tokenizer": {"sha256": sha256_file(tokenizer_path), "canonical_sha256": tokenizer.hash(), "vocabulary": tokenizer.vocab_size},
        "model_config_sha256": sha256_file(config_path),
        "route_controls_sha256": sha256_file(controls_path),
        "parameters": {"generator": model.parameter_count(), "router": int(protocol["router"]["parameters"])},
        "pointer_supervision": {"actions": sum(row["pointer_actions"] for row in examples), "records": sum(row["pointer_actions"] > 0 for row in examples)},
        "training": {"steps": successful, "batch_size": int(config["batch_size"]), "wall_seconds": time.perf_counter() - started, "skipped_amp_steps": skipped, "record_sequence_sha256": sequence.hexdigest(), "sampled_by_capability": dict(sorted(sampled.items())), "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "curves": curves},
        "imported_information": {"records": 7000, "teacher_outputs_added": 0, "stored_logits": 0, "stored_activations": 0, "source_parameters_copied": 0},
        "teacher_present_at_inference": False,
        "source_blocks_retained": 0,
        "promotion_eligible": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_json(output / "metadata.json", metadata)
    return metadata


@torch.inference_mode()
def _training_fit(root: Path, protocol: Mapping[str, Any], model: torch.nn.Module, tokenizer: Any) -> dict[str, Any]:
    examples, _ = _examples(root, protocol, tokenizer)
    actions = correct = exact = 0
    for start in range(0, len(examples), int(protocol["fit_attribution"]["batch_size"])):
        batch = examples[start:start + int(protocol["fit_attribution"]["batch_size"])]
        source, targets = _collate(batch, torch.device("cuda"))
        predicted = model(source, targets)["log_probs"].argmax(-1)
        mask = targets.ge(0)
        matches = predicted.eq(targets) & mask
        actions += int(mask.sum())
        correct += int(matches.sum())
        exact += sum(int(matches[index].sum()) == int(mask[index].sum()) for index in range(len(batch)))
    return {"records": len(examples), "actions": actions, "correct_actions": correct, "action_accuracy": correct / actions, "exact_sequences": exact, "exact_sequence_rate": exact / len(examples)}


@torch.inference_mode()
def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("BPE pointer evaluation output exists")
    metadata = _json(candidate / "metadata.json")
    if metadata.get("protocol_sha256") != protocol_sha or sha256_file(candidate / "model.safetensors") != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("BPE pointer candidate identity changed")
    model, tokenizer = _load_candidate(root, protocol, candidate)
    router_protocol = _json(root / protocol["router"]["protocol_path"])
    router, router_tokenizer = sparse._load(root, router_protocol, (root / protocol["router"]["candidate_dir"]).resolve())
    controls_doc = _json(candidate / "route_controls.json")["controls"]
    controls = {row["capability"]: bytes.fromhex(row["piece_hex"]).decode("utf-8") for row in controls_doc}
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"])
        route, _ = sparse._route(router, router_tokenizer, router_protocol, prompt)
        body = _semantic_segments(prompt)[-1]
        controlled = controls[route] + "\n" + body
        error = None
        try:
            value = model.generate_bytes(controlled, maximum_actions=min(int(probe["max_new_tokens"]), int(protocol["architecture"]["maximum_target_actions"]))).decode("utf-8", errors="strict")
        except Exception as exc:
            value = ""
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"probe_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]), "predicted_route": route, "route_correct": route == str(probe["canonical_capability"]), "output": value, "generation_error": error, "functional_pass": evaluate_functional(value, probe["evaluator"]), "repetition_collapse": repetition_collapse(value)})
        if (index + 1) % 100 == 0:
            print(json.dumps({"evaluated": index + 1}), flush=True)
    fit = _training_fit(root, protocol, model, tokenizer)
    output.mkdir(parents=True)
    raw = output / "development_outputs.jsonl"
    raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passes = sum(row["functional_pass"] for row in values)
        per[capability] = {"passes": passes, "observations": len(values), "collapses": sum(row["repetition_collapse"] for row in values), "wilson": wilson(passes, len(values))}
    teacher = {str(row["probe_id"]): row for row in map(json.loads, open(root / protocol["teacher_reference"], encoding="utf-8"))}
    probe_map = {str(row["probe_id"]): row for row in probes}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_map[row["probe_id"]]["evaluator"])} for row in rows]
    comparison = paired_stratified_bootstrap(paired, replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]), seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    gate = protocol["absolute_screen"]
    collapses = sum(row["repetition_collapse"] for row in rows)
    errors = sum(row["generation_error"] is not None for row in rows)
    gates = {
        "per_capability_functional": all(value["wilson"]["point"] >= gate["per_capability_functional_point_estimate_minimum"] and value["wilson"]["lower_95"] >= gate["per_capability_functional_wilson_lower_minimum"] for value in per.values()),
        "critical_capabilities": all(per[value]["wilson"]["point"] >= gate["critical_point_minimum"] and per[value]["wilson"]["lower_95"] >= gate["critical_wilson_lower_minimum"] for value in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_repetition_collapses": collapses == 0,
        "zero_generation_errors": errors == 0,
        "router_accuracy": sum(row["route_correct"] for row in rows) == len(rows),
        "teacher_relative_noninferiority": comparison["lower_95"] >= protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"],
    }
    passed = all(gates.values())
    decision = {"format": "abi-capability-compiler-phase3-bpe-pointer-decision/1", "status": "PASS_INITIAL_SCREEN_REPLICATION_AND_HOST_CERTIFICATION_REQUIRED" if passed else "FAIL_INITIAL_SCREEN_BPE_POINTER_CLOSED", "protocol_sha256": protocol_sha, "checkpoint_sha256": metadata["checkpoint"]["sha256"], "functional_passes": sum(row["functional_pass"] for row in rows), "observations": len(rows), "per_capability": per, "repetition_collapses": collapses, "generation_errors": errors, "route_correct": sum(row["route_correct"] for row in rows), "teacher_comparison": comparison, "training_fit_attribution": fit, "gates": gates, "initial_screen_pass": passed, "promotion_eligible": False, "outputs_sha256": sha256_file(raw), "evaluation_wall_seconds": time.perf_counter() - started, "teacher_present_at_inference": False, "layercake_host_changed": False, "phase3_certified": False, "phase4_open": False, "final_test_accessed": False, "next_step": "Preregister paired seeds and same-candidate LayerCake host/runtime certification." if passed else "Preserve failure; do not run remaining seeds or nearby pointer variants."}
    decision["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(decision)).hexdigest()
    _write_json(output / "decision.json", decision)
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train", "evaluate"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_BPE_POINTER_PROTOCOL_V54.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_bpe_pointer/development_v54/P0-seed240050")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_bpe_pointer/evaluation_v54/P0-seed240050")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    result = inventory(root, protocol) if args.command == "inventory" else train(root, protocol, (root / args.candidate_dir).resolve()) if args.command == "train" else evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
