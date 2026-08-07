"""V62: one LayerCake candidate trained with the verified pooled teacher substrate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_route_bridge import (
    BOS_ID,
    PAD_ID,
    _base,
    _collate,
    _examples,
    evaluate as evaluate_bridge,
    load_protocol,
)


def _projection(source_width: int, target_width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randn((source_width, target_width), generator=generator, dtype=torch.float32)
    return F.normalize(value, dim=0)


def _projection_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or valid.shape != value.shape[:2] or not bool(valid.any(dim=1).all()):
        raise Phase3Error("representation alignment mask is invalid")
    weights = valid.to(value.dtype).unsqueeze(-1)
    return (value * weights).sum(1) / weights.sum(1)


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer = _base(root, protocol, torch.device("cpu"))
    examples, controls = _examples(root, protocol, tokenizer)
    substrate = load_file(str((root / protocol["teacher_substrate"]["tensor_path"]).resolve()), device="cpu")
    if tuple(substrate["prompt_pooled"].shape) != (7000, 3072) or tuple(substrate["response_pooled"].shape) != (7000, 3072):
        raise Phase3Error("V62 teacher substrate shape changed")
    projection = _projection(3072, model.model_width, int(protocol["alignment"]["projection_seed"]))
    if _projection_hash(projection) != protocol["alignment"]["projection_sha256"]:
        raise Phase3Error("V62 fixed projection changed")
    if model.parameter_count() != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error("V62 LayerCake parameter count changed")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "records": len(examples),
        "route_controls": len(controls),
        "trainable_parameters": model.parameter_count(),
        "deployed_parameters": model.parameter_count(),
        "training_only_projection_scalars": projection.numel(),
        "training_only_projection_sha256": _projection_hash(projection),
        "teacher_substrate_scalars": sum(value.numel() for value in substrate.values()),
        "maximum_source_actions": max(len(row["source_ids"]) for row in examples),
        "maximum_target_actions": max(len(row["target_actions"]) for row in examples),
        "teacher_present_at_inference": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("V62 output exists or CUDA unavailable")
    device = torch.device("cuda")
    model, tokenizer = _base(root, protocol, device)
    examples, controls = _examples(root, protocol, tokenizer)
    phase1_rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    row_index = {str(row["ir_record_id"]): index for index, row in enumerate(phase1_rows)}
    if len(row_index) != len(examples) or any(row["record_id"] not in row_index for row in examples):
        raise Phase3Error("V62 teacher-substrate join changed")
    substrate = load_file(str((root / protocol["teacher_substrate"]["tensor_path"]).resolve()), device="cpu")
    teacher_prompt = substrate["prompt_pooled"]
    teacher_response = substrate["response_pooled"]
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    projection = _projection(teacher_prompt.shape[1], model.model_width, int(protocol["alignment"]["projection_seed"])).to(device)
    if _projection_hash(projection) != protocol["alignment"]["projection_sha256"]:
        raise Phase3Error("V62 fixed projection changed after device transfer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = _BalancedSampler(examples, seed)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    successful = skipped = 0
    sampled = Counter()
    sequence = hashlib.sha256()
    curves = []
    captured: dict[str, torch.Tensor] = {}

    def capture_decoder(_module, _inputs, value):
        captured["decoded"] = value

    hook = model.decoder.register_forward_hook(capture_decoder)
    started = time.perf_counter()
    model.train()
    try:
        while successful < int(cfg["steps"]):
            batch = sampler.batch(int(cfg["batch_size"]))
            source, targets = _collate(batch, device)
            previous = torch.full_like(targets, PAD_ID)
            previous[:, 0] = BOS_ID
            if targets.shape[1] > 1:
                previous[:, 1:] = torch.where(targets[:, :-1].ge(0), targets[:, :-1], torch.full_like(targets[:, :-1], PAD_ID))
            indices = [row_index[row["record_id"]] for row in batch]
            target_prompt = teacher_prompt[indices].float().to(device) @ projection
            target_response = teacher_response[indices].float().to(device) @ projection
            while True:
                captured.clear()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    result = model.action_log_probs(source, previous)
                    log_probs = result["log_probs"]
                    nll = F.nll_loss(log_probs.float().reshape(-1, log_probs.shape[-1]), targets.reshape(-1), ignore_index=-100)
                decoded = captured.get("decoded")
                if decoded is None:
                    raise Phase3Error("V62 decoder hook did not capture hidden states")
                source_mean = _masked_mean(result["encoded"].float(), ~result["source_padding"])
                response_mean = _masked_mean(decoded.float(), targets.ne(-100))
                prompt_loss = (1.0 - F.cosine_similarity(source_mean, target_prompt.float(), dim=-1)).mean()
                response_loss = (1.0 - F.cosine_similarity(response_mean, target_response.float(), dim=-1)).mean()
                loss = nll + float(protocol["alignment"]["prompt_weight"]) * prompt_loss + float(protocol["alignment"]["response_weight"]) * response_loss
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
                sequence.update(row["record_id"].encode("ascii") + b"\n")
            peak_rss = max(peak_rss, process.memory_info().rss)
            if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
                curve = {"step": successful, "loss": float(loss.detach()), "nll": float(nll.detach()), "prompt_alignment_loss": float(prompt_loss.detach()), "response_alignment_loss": float(response_loss.detach()), "wall_seconds": time.perf_counter() - started}
                curves.append(curve)
                print(json.dumps(curve), flush=True)
    finally:
        hook.remove()
    output.mkdir(parents=True)
    checkpoint = output / "model.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(checkpoint))
    router_source = (root / protocol["router"]["checkpoint_path"]).resolve()
    router_path = output / "router.safetensors"
    router_path.write_bytes(router_source.read_bytes())
    tokenizer_path = output / "tokenizer.json"
    _write_immutable(tokenizer_path, json.dumps(tokenizer.canonical_dict(), sort_keys=True, indent=2).encode("utf-8") + b"\n")
    config_path = output / "model_config.json"
    _write_immutable(config_path, json.dumps({**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size}, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    control_doc = [{"capability": capability, "token_id": controls[index][0], "piece_hex": controls[index][1].hex()} for index, capability in enumerate(protocol["capabilities"])]
    controls_path = output / "route_controls.json"
    _write_immutable(controls_path, json.dumps(control_doc, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    metadata = {
        "format": "abi-capability-compiler-phase3-representation-aligned-core/1",
        "status": "TRAINED_INITIAL_REPRESENTATION_ALIGNED_SCREEN",
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "router": {"path": router_path.name, "sha256": sha256_file(router_path), "bytes": router_path.stat().st_size, "parameters": 1058040},
        "tokenizer": {"path": tokenizer_path.name, "sha256": sha256_file(tokenizer_path), "canonical_sha256": tokenizer.hash(), "vocabulary": tokenizer.vocab_size},
        "model_config": {"path": config_path.name, "sha256": sha256_file(config_path), "trainable_parameters": model.parameter_count()},
        "route_controls": {"path": controls_path.name, "sha256": sha256_file(controls_path), "selection_sha256": protocol["route_controls"]["selection_sha256"]},
        "alignment": {"method": "fixed Gaussian column-normalized projection plus prompt and response cosine losses", "projection_seed": int(protocol["alignment"]["projection_seed"]), "projection_sha256": protocol["alignment"]["projection_sha256"], "projection_parameters_deployed": 0, "prompt_weight": float(protocol["alignment"]["prompt_weight"]), "response_weight": float(protocol["alignment"]["response_weight"])},
        "training": {"steps": successful, "batch_size": int(cfg["batch_size"]), "wall_seconds": time.perf_counter() - started, "skipped_amp_steps": skipped, "record_sequence_sha256": sequence.hexdigest(), "sampled_by_capability": dict(sorted(sampled.items())), "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "curves": curves},
        "imported_information": {"records": 7000, "teacher_input_tokens": 576925, "teacher_output_tokens": 215647, "stored_logits": 0, "stored_activations": 14000, "stored_activation_scalars": 43008000, "stored_activation_bytes": 86016000, "source_parameters_copied": 0},
        "teacher_present_at_inference": False,
        "training_only_projection_discarded": True,
        "source_blocks_retained": 0,
        "deployed_parameters": model.parameter_count(),
        "layercake_host_changed": False,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train", "evaluate"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_REPRESENTATION_ALIGNED_CORE_PROTOCOL_V62.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_representation_aligned_core/development_v62/A0-seed240050")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_representation_aligned_core/evaluation_v62/A0-seed240050")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    candidate = (root / args.candidate_dir).resolve()
    output = (root / args.output_dir).resolve()
    if args.command == "inventory":
        result = inventory(root, protocol)
    elif args.command == "train":
        result = train(root, protocol, candidate)
    else:
        result = evaluate_bridge(root, protocol, candidate, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
