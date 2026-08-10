"""Train and autonomously screen one structurally initialized causal core."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping

import psutil
from safetensors.torch import load_file, save_file
import torch

from . import capability_compiler_phase3_causal_field_core as field
from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, repetition_collapse, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable
from .capability_compiler_phase3_bpe_core_analysis import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_segment_router import _semantic_segments


FORMAT = "abi-capability-compiler-phase3-structural-core/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    raw_protocol = _json(path)
    if raw_protocol.get("format") == "abi-capability-compiler-phase3-structural-core-runtime-repair/1":
        if raw_protocol.get("status") != "PREREGISTERED_EXACT_RUNTIME_REPLAY":
            raise Phase3Error("structural runtime repair governance changed")
        protocol = _json(root / raw_protocol["base_protocol"])
        protocol["training"] = {**protocol["training"], **raw_protocol["training_override"]}
        protocol["bindings"] = raw_protocol["bindings"]
    else:
        protocol = raw_protocol
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INITIAL_GPU_SCREEN"
        or protocol.get("training", {}).get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("promotion_eligible") is not False
    ):
        raise Phase3Error("structural core governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"structural core binding changed: {name}")
    return protocol, sha256_file(path)


def _types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.structural_causal_core import StructuralCausalCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    return StructuralCausalCore, DecoderAwareExternalTokenizer


def _model(root: Path, protocol: Mapping[str, Any], device: torch.device):
    model_type, tokenizer_type = _types(root, protocol)
    tokenizer = field._tokenizer(protocol, tokenizer_type)
    model = model_type(fixed_vocab_size=tokenizer.vocab_size, **protocol["architecture"]).bind_tokenizer(tokenizer).to(device)
    initial = load_file(str(root / protocol["structural_artifact"]["path"]), device=str(device))
    model.load_state_dict(initial, strict=True)
    if model.parameter_count() != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error("structural model parameter count changed")
    return model, tokenizer, initial


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer, initial = _model(root, protocol, torch.device("cpu"))
    examples = field._examples(root, protocol, tokenizer)
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "records": len(examples),
        "parameters": model.parameter_count(),
        "initial_tensor_keys": len(initial),
        "vocabulary": tokenizer.vocab_size,
        "maximum_source_actions": max(len(row["source_ids"]) for row in examples),
        "maximum_target_actions": max(len(row["target_actions"]) for row in examples),
        "maximum_sequence_actions": max(len(row["source_ids"]) + len(row["target_actions"]) for row in examples),
        "probability_positions": sum(len(row["target_actions"]) for row in examples),
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if protocol["training"].get("cublas_workspace_config"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(protocol["training"]["cublas_workspace_config"])
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("structural candidate exists or CUDA unavailable")
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model, tokenizer, initial = _model(root, protocol, device)
    examples = field._examples(root, protocol, tokenizer)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("amp_grad_scaling", True)), init_scale=float(cfg.get("amp_initial_scale", 65536.0)))
    sampler = _BalancedSampler(examples, seed)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    successful = skipped = 0
    sequence = hashlib.sha256()
    sampled = Counter()
    curves = []
    started = time.perf_counter()
    while successful < int(cfg["steps"]):
        batch = sampler.batch(int(cfg["batch_size"]))
        microbatch_size = int(cfg.get("microbatch_size", cfg["batch_size"]))
        if int(cfg["batch_size"]) % microbatch_size:
            raise Phase3Error("logical batch must divide structural microbatch")
        total_valid = sum(len(row["target_actions"]) for row in batch)
        while True:
            optimizer.zero_grad(set_to_none=True)
            loss_value = hard_value = soft_value = 0.0
            for offset in range(0, len(batch), microbatch_size):
                packed = field._collate(batch[offset:offset + microbatch_size], device)
                valid_count = int(packed[-1].sum().item())
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(packed[0])
                    hard, soft = field._losses(logits, *packed[1:])
                    loss = hard + float(cfg["probability_field_weight"]) * soft
                    weighted = loss * (valid_count / total_valid)
                scaler.scale(weighted).backward()
                loss_value += float(loss.detach()) * valid_count / total_valid
                hard_value += float(hard.detach()) * valid_count / total_valid
                soft_value += float(soft.detach()) * valid_count / total_valid
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
            sequence.update(row["record_id"].encode() + b"\n")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curve = {"step": successful, "loss": loss_value, "hard_nll": hard_value, "soft_cross_entropy": soft_value, "wall_seconds": time.perf_counter() - started}
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    torch.cuda.synchronize()
    final_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    initial_cpu = {key: value.detach().cpu().to(final_state[key].dtype) for key, value in initial.items()}
    changed = sum(int(final_state[key].ne(initial_cpu[key]).sum()) for key in final_state)
    squared_delta = sum(float((final_state[key].float() - initial_cpu[key].float()).square().sum()) for key in final_state)
    output.mkdir(parents=True)
    checkpoint = output / "model.safetensors"
    save_file(final_state, str(checkpoint))
    router_source = root / protocol["router"]["checkpoint_path"]
    (output / "router.safetensors").write_bytes(router_source.read_bytes())
    _write_json(output / "tokenizer.json", tokenizer.canonical_dict())
    _write_json(output / "model_config.json", {**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size})
    extraction = _json(root / protocol["structural_artifact"]["extraction_result"])
    metadata = {
        "format": "abi-capability-compiler-phase3-structural-candidate/1",
        "status": "TRAINED_INITIAL_DEVELOPMENT_SCREEN",
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "initial_structural_artifact": {"path": protocol["structural_artifact"]["path"], "sha256": sha256_file(root / protocol["structural_artifact"]["path"]), "unchanged": True},
        "parameters": {"generator": model.parameter_count(), "router": int(protocol["router"]["parameters"]), "trainable": sum(value.numel() for value in model.parameters() if value.requires_grad)},
        "conformance_changes": {"changed_scalar_entries": changed, "squared_l2_delta": squared_delta, "original_artifact_mutated": False},
        "training": {"steps": successful, "logical_batch_size": int(cfg["batch_size"]), "physical_microbatch_size": int(cfg.get("microbatch_size", cfg["batch_size"])), "gradient_accumulation_steps": int(cfg["batch_size"]) // int(cfg.get("microbatch_size", cfg["batch_size"])), "wall_seconds": time.perf_counter() - started, "skipped_amp_steps": skipped, "record_sequence_sha256": sequence.hexdigest(), "sampled_by_capability": dict(sorted(sampled.items())), "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "curves": curves},
        "imported_information": {
            "records": len(examples),
            "raw_prompt_bytes": sum(row["prompt_bytes"] for row in examples),
            "teacher_output_bytes": sum(row["output_bytes"] for row in examples),
            "authoritative_teacher_tokens": sum(row["teacher_tokens"] for row in examples),
            "stored_logits": int(protocol["probability_field"]["stored_logits"]),
            "stored_probability_scalars": int(protocol["probability_field"]["stored_probability_scalars"]),
            "probability_field_bytes": int(protocol["probability_field"]["tensor_payload_bytes"]),
            "stored_activations": 0,
            "source_parameters": extraction["accounting"]["source_parameters"],
            "final_imported_substrate_parameters": extraction["accounting"]["final_imported_substrate_parameters"],
            "exact_source_scalar_entries": extraction["accounting"]["exact_source_scalar_entries_in_deployed_artifact"],
            "transformed_source_derived_scalar_entries": extraction["accounting"]["transformed_source_derived_scalar_entries"],
            "source_extraction_seconds": extraction["accounting"]["extraction_seconds"],
            "teacher_forward_tokens_for_probability_field": int(protocol["probability_field"]["teacher_forward_tokens"]),
            "teacher_forward_seconds_for_probability_field": float(protocol["probability_field"]["teacher_forward_seconds"]),
        },
        "teacher_present_at_inference": False,
        "complete_source_blocks_retained": 0,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_json(output / "metadata.json", metadata)
    return metadata


def _load_candidate(root: Path, protocol: Mapping[str, Any], candidate: Path):
    model_type, tokenizer_type = _types(root, protocol)
    tokenizer = tokenizer_type.from_document(_json(candidate / "tokenizer.json"))
    model = model_type(**_json(candidate / "model_config.json")).bind_tokenizer(tokenizer).cuda()
    model.load_state_dict(load_file(str(candidate / "model.safetensors"), device="cuda"), strict=True)
    return model.eval()


@torch.inference_mode()
def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    metadata = _json(candidate / "metadata.json")
    if output.exists() or metadata.get("protocol_sha256") != protocol_sha or sha256_file(candidate / "model.safetensors") != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("structural candidate identity changed")
    model = _load_candidate(root, protocol, candidate)
    router_protocol = _json(root / protocol["router"]["protocol_path"])
    router, router_tokenizer = sparse._load(root, router_protocol, root / protocol["router"]["candidate_dir"])
    probes = development_probes(root / protocol["development_catalog"])
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        prompt = str(probe["prompt"])
        route, _ = sparse._route(router, router_tokenizer, router_protocol, prompt)
        controlled = f"Capability route: {route}\n{_semantic_segments(prompt)[-1]}"
        error = None
        try:
            value = model.generate_bytes(controlled, maximum_actions=min(int(probe["max_new_tokens"]), int(protocol["architecture"]["maximum_target_actions"]))).decode("utf-8", errors="strict")
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
    decision = {
        "format": "abi-capability-compiler-phase3-structural-core-decision/1",
        "status": "PASS_INITIAL_SCREEN_REPLICATION_AND_HOST_CERTIFICATION_REQUIRED" if passed else "FAIL_INITIAL_SCREEN_STRUCTURAL_BRANCH_REQUIRES_ATTRIBUTION",
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "functional_passes": sum(row["functional_pass"] for row in rows),
        "observations": len(rows),
        "per_capability": per,
        "repetition_collapses": collapses,
        "generation_errors": errors,
        "route_correct": sum(row["route_correct"] for row in rows),
        "teacher_comparison": comparison,
        "gates": gates,
        "initial_screen_pass": passed,
        "promotion_eligible": False,
        "outputs_sha256": sha256_file(raw),
        "evaluation_wall_seconds": time.perf_counter() - started,
        "teacher_present_at_inference": False,
        "complete_source_blocks_retained": 0,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
    }
    decision["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(decision)).hexdigest()
    _write_json(output / "decision.json", decision)
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train", "evaluate"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_STRUCTURAL_CORE_PROTOCOL_V197.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_structural/core_v197/C0-seed240184")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_structural/core_v197/evaluation_C0-seed240184")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    result = inventory(root, protocol) if args.command == "inventory" else train(root, protocol, (root / args.candidate_dir).resolve()) if args.command == "train" else evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
