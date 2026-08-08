"""V70 matched LayerCake screen using verified action-aligned causal states."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable
from .capability_compiler_phase3_route_bridge import BOS_ID, PAD_ID, _base, _collate, _examples, evaluate as evaluate_bridge, load_protocol


def _alignment_batch(
    batch: Sequence[Mapping[str, Any]], index_by_id: Mapping[str, int], tensors: Mapping[str, torch.Tensor], source_width: int, target_width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_teacher = torch.zeros((len(batch), source_width, 192), dtype=torch.float32, device=device)
    target_teacher = torch.zeros((len(batch), target_width, 192), dtype=torch.float32, device=device)
    source_mask = torch.zeros((len(batch), source_width), dtype=torch.bool, device=device)
    target_mask = torch.zeros((len(batch), target_width), dtype=torch.bool, device=device)
    for offset, row in enumerate(batch):
        index = index_by_id[str(row["record_id"])]
        s0, s1 = int(tensors["source_offsets"][index]), int(tensors["source_offsets"][index + 1])
        t0, t1 = int(tensors["target_offsets"][index]), int(tensors["target_offsets"][index + 1])
        source_count = s1 - s0
        target_count = t1 - t0
        if source_count != len(row["source_ids"]) or target_count != len(row["target_actions"]) - 1:
            raise Phase3Error("V70 action-aligned join length changed")
        source_teacher[offset, :source_count] = tensors["source_values"][s0:s1].float().to(device)
        target_teacher[offset, :target_count] = tensors["target_values"][t0:t1].float().to(device)
        source_mask[offset, :source_count] = True
        target_mask[offset, :target_count] = True
    return source_teacher, target_teacher, source_mask, target_mask


def _cosine_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if student.shape != teacher.shape or mask.shape != student.shape[:2] or not bool(mask.any()):
        raise Phase3Error("V70 alignment tensors changed")
    return (1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=-1))[mask].mean()


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer = _base(root, protocol, torch.device("cpu"))
    examples, controls = _examples(root, protocol, tokenizer)
    tensors = load_file(str((root / protocol["teacher_substrate"]["tensor_path"]).resolve()), device="cpu")
    if tuple(tensors["source_values"].shape) != (50403, 192) or tuple(tensors["target_values"].shape) != (49029, 192):
        raise Phase3Error("V70 substrate shape changed")
    if model.parameter_count() != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error("V70 model parameter count changed")
    records = [json.loads(line) for line in (root / protocol["teacher_substrate"]["records_path"]).read_text(encoding="utf-8").splitlines()]
    if [row["record_id"] for row in records] != [row["record_id"] for row in examples]:
        raise Phase3Error("V70 substrate record order changed")
    return {"status": "PASS", "protocol_sha256": protocol_sha, "records": len(examples), "route_controls": len(controls), "trainable_parameters": model.parameter_count(), "deployed_parameters": model.parameter_count(), "source_teacher_vectors": tensors["source_values"].shape[0], "target_teacher_vectors": tensors["target_values"].shape[0], "teacher_present_at_inference": False, "final_test_accessed": False}


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("V70 output exists or CUDA unavailable")
    device = torch.device("cuda")
    model, tokenizer = _base(root, protocol, device)
    examples, controls = _examples(root, protocol, tokenizer)
    tensors = load_file(str((root / protocol["teacher_substrate"]["tensor_path"]).resolve()), device="cpu")
    records = [json.loads(line) for line in (root / protocol["teacher_substrate"]["records_path"]).read_text(encoding="utf-8").splitlines()]
    index_by_id = {str(row["record_id"]): index for index, row in enumerate(records)}
    if len(index_by_id) != 7000 or any(row["record_id"] not in index_by_id for row in examples):
        raise Phase3Error("V70 substrate join changed")
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
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
            teacher_source, teacher_target, source_mask, target_mask = _alignment_batch(batch, index_by_id, tensors, source.shape[1], targets.shape[1], device)
            while True:
                captured.clear()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    result = model.action_log_probs(source, previous)
                    nll = F.nll_loss(result["log_probs"].float().reshape(-1, result["log_probs"].shape[-1]), targets.reshape(-1), ignore_index=-100)
                decoded = captured.get("decoded")
                if decoded is None:
                    raise Phase3Error("V70 decoder hook failed")
                source_loss = _cosine_loss(result["encoded"], teacher_source, source_mask)
                target_loss = _cosine_loss(decoded, teacher_target, target_mask)
                loss = nll + float(protocol["alignment"]["source_weight"]) * source_loss + float(protocol["alignment"]["target_weight"]) * target_loss
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
                curve = {"step": successful, "loss": float(loss.detach()), "nll": float(nll.detach()), "source_alignment_loss": float(source_loss.detach()), "target_alignment_loss": float(target_loss.detach()), "wall_seconds": time.perf_counter() - started}
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
    controls_path = output / "route_controls.json"
    control_doc = [{"capability": capability, "token_id": controls[index][0], "piece_hex": controls[index][1].hex()} for index, capability in enumerate(CAPABILITIES)]
    _write_immutable(controls_path, json.dumps(control_doc, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    metadata = {"format": "abi-capability-compiler-phase3-action-aligned-core/1", "status": "TRAINED_INITIAL_ACTION_ALIGNED_SCREEN", "protocol_sha256": protocol_sha, "seed": seed, "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size}, "router": {"path": router_path.name, "sha256": sha256_file(router_path), "bytes": router_path.stat().st_size, "parameters": 1058040}, "tokenizer": {"path": tokenizer_path.name, "sha256": sha256_file(tokenizer_path), "canonical_sha256": tokenizer.hash(), "vocabulary": tokenizer.vocab_size}, "model_config": {"path": config_path.name, "sha256": sha256_file(config_path), "trainable_parameters": model.parameter_count()}, "route_controls": {"path": controls_path.name, "sha256": sha256_file(controls_path), "selection_sha256": protocol["route_controls"]["selection_sha256"]}, "alignment": {"method": "per-action causal representation distillation", "source_weight": float(protocol["alignment"]["source_weight"]), "target_weight": float(protocol["alignment"]["target_weight"]), "deployed_alignment_parameters": 0}, "training": {"steps": successful, "batch_size": int(cfg["batch_size"]), "wall_seconds": time.perf_counter() - started, "skipped_amp_steps": skipped, "record_sequence_sha256": sequence.hexdigest(), "sampled_by_capability": dict(sorted(sampled.items())), "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "curves": curves}, "imported_information": {"records": 7000, "teacher_input_tokens": 576925, "teacher_output_tokens": 215647, "stored_logits": 0, "stored_activations": 99432, "stored_activation_scalars": 19090944, "stored_activation_bytes": 38181888, "source_parameters_copied": 0}, "teacher_present_at_inference": False, "source_blocks_retained": 0, "deployed_parameters": model.parameter_count(), "layercake_host_changed": False, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False, "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)}}
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train", "evaluate"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_CORE_PROTOCOL_V70.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_action_aligned_core/development_v70/A0-seed240050")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_action_aligned_core/evaluation_v70/A0-seed240050")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    candidate = (root / args.candidate_dir).resolve()
    output = (root / args.output_dir).resolve()
    result = inventory(root, protocol) if args.command == "inventory" else train(root, protocol, candidate) if args.command == "train" else evaluate_bridge(root, protocol, candidate, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
