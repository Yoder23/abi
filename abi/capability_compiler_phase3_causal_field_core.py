"""Train and screen one decoder-only causal probability-field candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, repetition_collapse, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable
from .capability_compiler_phase3_bpe_core_analysis import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_causal_field_feasibility import _rows
from .capability_compiler_phase3_segment_router import _semantic_segments


FORMAT = "abi-capability-compiler-phase3-causal-field-core/1"
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
EXTERNAL_OFFSET = 4


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INITIAL_GPU_SCREEN"
        or protocol.get("training", {}).get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("promotion_eligible") is not False
    ):
        raise Phase3Error("causal-field core governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"causal-field core binding changed: {relative}")
    return protocol, sha256_file(path)


def _layercake_types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.native_causal_core import TiedNativeCausalCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    return TiedNativeCausalCore, DecoderAwareExternalTokenizer


def _tokenizer(protocol: Mapping[str, Any], tokenizer_type: Any):
    document = _json(Path(protocol["source"]["tokenizer_json"]))
    tokenizer = tokenizer_type(document)
    if tokenizer.hash() != protocol["source"]["layercake_tokenizer_sha256"] or tokenizer.vocab_size != int(protocol["source"]["host_fixed_actions"]):
        raise Phase3Error("causal-field tokenizer identity changed")
    return tokenizer


def _map_external(value: int, terminal: int) -> int:
    return EOS_ID if int(value) == terminal else EXTERNAL_OFFSET + int(value)


def _examples(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    rows = _rows(root, protocol)
    field_dir = (root / protocol["probability_field"]["directory"]).resolve()
    records = [json.loads(line) for line in (field_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    tensors = load_file(str(field_dir / "top32_probability_field.safetensors"), device="cpu")
    terminal = int(protocol["source"]["terminal_token_id"])
    offsets = tensors["offsets"]
    examples = []
    for index, (row, field) in enumerate(zip(rows, records)):
        if row["record_id"] != field["record_id"] or row["capability"] != field["capability"]:
            raise Phase3Error("causal-field record order changed")
        source_ids, _ = tokenizer.encode_source(row["host_prompt"])
        targets = [_map_external(value, terminal) for value in row["generated_ids"]]
        lo = int(offsets[index]); hi = int(offsets[index + 1])
        if hi - lo != len(targets):
            raise Phase3Error("causal-field target offset changed")
        field_ids = tensors["token_ids"][lo:hi].to(torch.int64)
        mapped_ids = torch.where(field_ids.eq(terminal), torch.full_like(field_ids, EOS_ID), field_ids + EXTERNAL_OFFSET)
        if len(source_ids) > int(protocol["architecture"]["maximum_source_actions"]) or len(targets) > int(protocol["architecture"]["maximum_target_actions"]) or len(source_ids) + len(targets) > int(protocol["architecture"]["maximum_sequence_actions"]):
            raise Phase3Error("causal-field example exceeds host bound")
        examples.append(
            {
                "record_id": row["record_id"],
                "capability": row["capability"],
                "source_ids": source_ids,
                "target_actions": targets,
                "field_ids": mapped_ids,
                "field_probabilities": tensors["probabilities"][lo:hi],
                "field_residual": tensors["residual_mass"][lo:hi],
                "teacher_tokens": int(row["teacher_output_tokens"]),
                "prompt_bytes": len(row["rendered_prompt"].encode("utf-8")),
                "output_bytes": len(row["output"].encode("utf-8")),
            }
        )
    if len(examples) != 14000 or len(records) != 14000:
        raise Phase3Error("causal-field example inventory changed")
    return examples


def _model(protocol: Mapping[str, Any], tokenizer: Any, model_type: Any, device: torch.device):
    model = model_type(fixed_vocab_size=tokenizer.vocab_size, **protocol["architecture"]).bind_tokenizer(tokenizer).to(device)
    if model.parameter_count() != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error("causal-field model parameter count changed")
    return model


def _collate(rows: Sequence[Mapping[str, Any]], device: torch.device):
    input_width = max(len(row["source_ids"]) + len(row["target_actions"]) for row in rows)
    target_width = max(len(row["target_actions"]) for row in rows)
    top_k = int(rows[0]["field_ids"].shape[1])
    inputs = torch.full((len(rows), input_width), PAD_ID, dtype=torch.long, device=device)
    target_positions = torch.zeros((len(rows), target_width), dtype=torch.long, device=device)
    targets = torch.full((len(rows), target_width), -100, dtype=torch.long, device=device)
    field_ids = torch.zeros((len(rows), target_width, top_k), dtype=torch.long, device=device)
    field_probabilities = torch.zeros((len(rows), target_width, top_k), dtype=torch.float16, device=device)
    field_residual = torch.zeros((len(rows), target_width), dtype=torch.float16, device=device)
    valid = torch.zeros((len(rows), target_width), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        source = list(row["source_ids"])
        target = list(row["target_actions"])
        packed = source + [BOS_ID] + target[:-1]
        inputs[index, : len(packed)] = torch.tensor(packed, dtype=torch.long, device=device)
        count = len(target)
        target_positions[index, :count] = torch.arange(len(source), len(source) + count, device=device)
        targets[index, :count] = torch.tensor(target, dtype=torch.long, device=device)
        field_ids[index, :count] = row["field_ids"].to(device=device, dtype=torch.long)
        field_probabilities[index, :count] = row["field_probabilities"].to(device)
        field_residual[index, :count] = row["field_residual"].to(device)
        valid[index, :count] = True
    return inputs, target_positions, targets, field_ids, field_probabilities, field_residual, valid


def _losses(logits: torch.Tensor, positions: torch.Tensor, targets: torch.Tensor, field_ids: torch.Tensor, field_probabilities: torch.Tensor, field_residual: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    selected_logits = torch.gather(logits, 1, positions[:, :, None].expand(-1, -1, logits.shape[-1]))
    log_probs = F.log_softmax(selected_logits.float(), dim=-1)
    hard = F.nll_loss(log_probs.reshape(-1, log_probs.shape[-1]), targets.reshape(-1), ignore_index=-100)
    selected_log_probs = torch.gather(log_probs, -1, field_ids)
    selected_student_mass = torch.exp(selected_log_probs).sum(dim=-1).clamp(max=1.0 - 1e-7)
    other_log_prob = torch.log1p(-selected_student_mass)
    soft_rows = -(field_probabilities.float() * selected_log_probs).sum(dim=-1) - field_residual.float() * other_log_prob
    soft = soft_rows[valid].mean()
    return hard, soft


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model_type, tokenizer_type = _layercake_types(root, protocol)
    tokenizer = _tokenizer(protocol, tokenizer_type)
    examples = _examples(root, protocol, tokenizer)
    model = _model(protocol, tokenizer, model_type, torch.device("cpu"))
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "records": len(examples),
        "parameters": model.parameter_count(),
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
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("causal-field candidate exists or CUDA unavailable")
    model_type, tokenizer_type = _layercake_types(root, protocol)
    tokenizer = _tokenizer(protocol, tokenizer_type)
    examples = _examples(root, protocol, tokenizer)
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model = _model(protocol, tokenizer, model_type, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1)
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
    while successful < int(cfg["steps"]):
        batch = sampler.batch(int(cfg["batch_size"]))
        inputs, positions, targets, field_ids, field_probabilities, field_residual, valid = _collate(batch, device)
        while True:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(inputs)
                hard, soft = _losses(logits, positions, targets, field_ids, field_probabilities, field_residual, valid)
                loss = hard + float(cfg["probability_field_weight"]) * soft
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
            sequence.update(row["record_id"].encode() + b"\n")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curve = {"step": successful, "loss": float(loss.detach()), "hard_nll": float(hard.detach()), "soft_cross_entropy": float(soft.detach()), "wall_seconds": time.perf_counter() - started}
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    output.mkdir(parents=True)
    checkpoint = output / "model.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(checkpoint))
    router_source = root / protocol["router"]["checkpoint_path"]
    (output / "router.safetensors").write_bytes(router_source.read_bytes())
    _write_json(output / "tokenizer.json", tokenizer.canonical_dict())
    _write_json(output / "model_config.json", {**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size})
    metadata = {
        "format": "abi-capability-compiler-phase3-causal-field-candidate/1",
        "status": "TRAINED_INITIAL_DEVELOPMENT_SCREEN",
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parameters": {"generator": model.parameter_count(), "router": int(protocol["router"]["parameters"])},
        "training": {"steps": successful, "batch_size": int(cfg["batch_size"]), "wall_seconds": time.perf_counter() - started, "skipped_amp_steps": skipped, "record_sequence_sha256": sequence.hexdigest(), "sampled_by_capability": dict(sorted(sampled.items())), "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "curves": curves},
        "imported_information": {"records": len(examples), "raw_prompt_bytes": sum(row["prompt_bytes"] for row in examples), "teacher_output_bytes": sum(row["output_bytes"] for row in examples), "authoritative_teacher_tokens": sum(row["teacher_tokens"] for row in examples), "stored_logits": int(protocol["probability_field"]["stored_logits"]), "stored_probability_scalars": int(protocol["probability_field"]["stored_probability_scalars"]), "probability_field_bytes": int(protocol["probability_field"]["tensor_payload_bytes"]), "stored_activations": 0, "source_parameters_copied": 0},
        "teacher_present_at_inference": False,
        "source_blocks_retained": 0,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_json(output / "metadata.json", metadata)
    return metadata


def _load_candidate(root: Path, protocol: Mapping[str, Any], candidate: Path):
    model_type, tokenizer_type = _layercake_types(root, protocol)
    tokenizer = tokenizer_type.from_document(_json(candidate / "tokenizer.json"))
    model = model_type(**_json(candidate / "model_config.json")).bind_tokenizer(tokenizer).cuda()
    model.load_state_dict(load_file(str(candidate / "model.safetensors"), device="cuda"), strict=True)
    return model.eval(), tokenizer


@torch.inference_mode()
def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    metadata = _json(candidate / "metadata.json")
    if output.exists() or metadata.get("protocol_sha256") != protocol_sha or sha256_file(candidate / "model.safetensors") != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("causal-field candidate identity changed")
    model, _ = _load_candidate(root, protocol, candidate)
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
        "format": "abi-capability-compiler-phase3-causal-field-decision/1",
        "status": "PASS_INITIAL_SCREEN_REPLICATION_AND_HOST_CERTIFICATION_REQUIRED" if passed else "FAIL_INITIAL_SCREEN_CAUSAL_FIELD_BRANCH_CLOSED",
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
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CAUSAL_FIELD_CORE_PROTOCOL_V184.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3/causal_field_core_v184/C0-seed240184")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3/causal_field_core_v184/evaluation_C0-seed240184")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    result = inventory(root, protocol) if args.command == "inventory" else train(root, protocol, (root / args.candidate_dir).resolve()) if args.command == "train" else evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
