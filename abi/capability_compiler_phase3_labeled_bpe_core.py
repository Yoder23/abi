"""Train one label-aware, header-robust, causal-recovery BPE candidate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping

import psutil
from safetensors.torch import save_file
import torch
from torch import nn
import torch.nn.functional as F

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_bpe_core import (
    FORMAT,
    _collate,
    _json,
    _layercake_api,
    _model,
    _tokenizer,
    load_protocol,
)


BOS_ID = 1
PAD_ID = 0
UNK_ID = 3


def use_body_view(record_id: str, step: int, probability: float) -> bool:
    digest = hashlib.sha256(f"{record_id}\0{step}\0header-view".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < probability


def _examples(
    rows: list[Mapping[str, Any]],
    tokenizer: Any,
    eos_id: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["normalized_acquisition_prompt"])
        lines = prompt.splitlines()
        if len(lines) < 2:
            raise Phase3Error(f"labeled BPE prompt lacks synthetic header: {row['ir_record_id']}")
        body = "\n".join(lines[1:])
        output = str(row["normalized_output"])
        source_lexemes = tokenizer.split(prompt)
        body_lexemes = tokenizer.split(body)
        output_lexemes = tokenizer.split(output)
        source_ids = [tokenizer.lexeme_to_id[value] for value in source_lexemes]
        body_ids = [tokenizer.lexeme_to_id[value] for value in body_lexemes]
        target_actions = [tokenizer.lexeme_to_id[value] for value in output_lexemes] + [eos_id]
        if (
            max(len(source_ids), len(body_ids)) > int(config["maximum_source_lexemes"])
            or len(target_actions) > int(config["maximum_target_actions"])
        ):
            raise Phase3Error(f"labeled BPE example exceeds bound: {row['ir_record_id']}")
        examples.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "source_ids": source_ids,
                "body_source_ids": body_ids,
                "target_actions": target_actions,
                "teacher_tokens": int(row["authoritative_teacher_tokens"]),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "output_bytes_count": len(output.encode("utf-8")),
            }
        )
    if len(examples) != 7000 or len({row["record_id"] for row in examples}) != 7000:
        raise Phase3Error("labeled BPE inventory changed")
    return examples


def _previous_actions(targets: torch.Tensor, corruption_probability: float) -> tuple[torch.Tensor, int, int]:
    previous = torch.full_like(targets, PAD_ID)
    previous[:, 0] = BOS_ID
    if targets.shape[1] > 1:
        shifted = targets[:, :-1]
        previous[:, 1:] = torch.where(shifted.ge(0), shifted, torch.full_like(shifted, PAD_ID))
    eligible = previous.ne(PAD_ID)
    eligible[:, 0] = False
    corrupt = eligible & torch.rand(previous.shape, device=previous.device).lt(corruption_probability)
    previous = torch.where(corrupt, torch.full_like(previous, UNK_ID), previous)
    return previous, int(corrupt.sum().item()), int(eligible.sum().item())


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    eos_id, model_type, tokenizer_type, abi_version, abi_sha = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    examples = _examples(load_phase1_ir((root / protocol["phase1_ir"]).resolve()), tokenizer, eos_id, protocol["architecture"])
    model = _model(protocol, tokenizer, model_type)
    auxiliary = nn.Linear(int(protocol["architecture"]["model_width"]), len(CAPABILITIES))
    deployed = sum(value.numel() for value in model.parameters())
    auxiliary_parameters = sum(value.numel() for value in auxiliary.parameters())
    if (
        deployed != int(protocol["training"]["deployed_trainable_parameters"])
        or auxiliary_parameters != int(protocol["training"]["training_only_auxiliary_parameters"])
    ):
        raise Phase3Error("labeled BPE parameter inventory changed")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "layercake_interface": abi_version,
        "layercake_interface_sha256": abi_sha,
        "records": len(examples),
        "capabilities": len(CAPABILITIES),
        "deployed_trainable_parameters": deployed,
        "training_only_auxiliary_parameters": auxiliary_parameters,
        "maximum_full_source_actions": max(len(row["source_ids"]) for row in examples),
        "maximum_body_source_actions": max(len(row["body_source_ids"]) for row in examples),
        "maximum_target_actions": max(len(row["target_actions"]) for row in examples),
        "teacher_present_at_inference": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_dir.exists() or not torch.cuda.is_available():
        raise Phase3Error("labeled BPE output exists or CUDA unavailable")
    eos_id, model_type, tokenizer_type, abi_version, abi_sha = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    examples = _examples(load_phase1_ir((root / protocol["phase1_ir"]).resolve()), tokenizer, eos_id, protocol["architecture"])
    cfg = protocol["training"]
    strategy = protocol["labeled_acquisition"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model = _model(protocol, tokenizer, model_type).to(device)
    auxiliary = nn.Linear(int(protocol["architecture"]["model_width"]), len(CAPABILITIES)).to(device)
    deployed_parameters = sum(value.numel() for value in model.parameters())
    auxiliary_parameters = sum(value.numel() for value in auxiliary.parameters())
    if deployed_parameters != int(cfg["deployed_trainable_parameters"]) or auxiliary_parameters != int(cfg["training_only_auxiliary_parameters"]):
        raise Phase3Error("labeled BPE parameter count changed")
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(auxiliary.parameters()),
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = _BalancedSampler(examples, seed)
    capability_to_id = {name: index for index, name in enumerate(CAPABILITIES)}
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    successful = skipped = corrupted_actions = eligible_history_actions = 0
    header_views: Counter[str] = Counter()
    sampled: Counter[str] = Counter()
    sequence_hash = hashlib.sha256()
    curves: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    auxiliary.train()
    while successful < int(cfg["steps"]):
        selected = sampler.batch(int(cfg["batch_size"]))
        selected_views: list[dict[str, Any]] = []
        step = successful + 1
        for row in selected:
            body = use_body_view(str(row["record_id"]), step, float(strategy["header_dropout_probability"]))
            selected_views.append({**row, "source_ids": row["body_source_ids"] if body else row["source_ids"]})
            header_views["body" if body else "full"] += 1
        while True:
            source, targets = _collate(selected_views, device)
            labels = torch.tensor([capability_to_id[str(row["capability"])] for row in selected], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                previous, corrupt_count, eligible_count = _previous_actions(targets, float(strategy["causal_history_corruption_probability"]))
                result = model.action_log_probs(source, previous)
                log_probs = result["log_probs"]
                token_loss = F.nll_loss(log_probs.float().reshape(-1, log_probs.shape[-1]), targets.reshape(-1), ignore_index=-100)
                source_mask = source.ne(PAD_ID)
                pooled = (result["encoded"] * source_mask[:, :, None]).sum(dim=1) / source_mask.sum(dim=1, keepdim=True).clamp_min(1)
                capability_logits = auxiliary(pooled)
                capability_loss = F.cross_entropy(capability_logits.float(), labels)
                loss = token_loss + float(strategy["capability_auxiliary_loss_weight"]) * capability_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(auxiliary.parameters()), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1
                continue
            break
        successful += 1
        corrupted_actions += corrupt_count
        eligible_history_actions += eligible_count
        for row in selected:
            sampled[row["capability"]] += 1
            sequence_hash.update(row["record_id"].encode("ascii") + b"\n")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curve = {
                "step": successful,
                "loss": float(loss.detach()),
                "token_loss": float(token_loss.detach()),
                "capability_loss": float(capability_loss.detach()),
                "capability_accuracy": float(capability_logits.argmax(dim=-1).eq(labels).float().mean().detach()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "model.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(checkpoint))
    auxiliary_path = output_dir / "training_auxiliary.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in auxiliary.state_dict().items()}, str(auxiliary_path))
    tokenizer_path = output_dir / "tokenizer.json"
    _write_immutable(tokenizer_path, json.dumps(tokenizer.canonical_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n")
    config_path = output_dir / "model_config.json"
    model_config = {**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size}
    _write_immutable(config_path, json.dumps(model_config, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    metadata: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-bpe-core-candidate/1",
        "status": "TRAINED_CONDITIONAL_DEVELOPMENT_SCREEN",
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "checkpoint": {"path": "model.safetensors", "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "tokenizer": {"path": "tokenizer.json", "sha256": sha256_file(tokenizer_path), "canonical_sha256": tokenizer.hash(), "vocabulary": tokenizer.vocab_size},
        "model_config": {"path": "model_config.json", "sha256": sha256_file(config_path), "trainable_parameters": deployed_parameters},
        "training_auxiliary": {"path": "training_auxiliary.safetensors", "sha256": sha256_file(auxiliary_path), "bytes": auxiliary_path.stat().st_size, "parameters": auxiliary_parameters, "present_at_inference": False},
        "layercake_interface": {"version": abi_version, "sha256": abi_sha},
        "imported_information": {
            "records": len(examples),
            "raw_prompt_bytes": sum(row["prompt_bytes"] for row in examples),
            "teacher_output_bytes": sum(row["output_bytes_count"] for row in examples),
            "authoritative_teacher_tokens": sum(row["teacher_tokens"] for row in examples),
            "stored_logits": 0,
            "stored_activations": 0,
            "source_parameters_copied": 0,
            "capability_labels": len(examples),
        },
        "representation": {"pointer_supervision": False, "label_aware_training": True, "header_dropout": True, "causal_history_corruption": True},
        "training": {
            "steps": successful,
            "batch_size": int(cfg["batch_size"]),
            "wall_seconds": time.perf_counter() - started,
            "skipped_amp_steps": skipped,
            "record_sequence_sha256": sequence_hash.hexdigest(),
            "sampled_by_capability": dict(sorted(sampled.items())),
            "header_views": dict(sorted(header_views.items())),
            "corrupted_history_actions": corrupted_actions,
            "eligible_history_actions": eligible_history_actions,
            "realized_history_corruption_rate": corrupted_actions / eligible_history_actions,
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "curves": curves,
        },
        "teacher_present_at_inference": False,
        "source_blocks_retained": 0,
        "development_only": True,
        "promotion_eligible": False,
        "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    metadata["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output_dir / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "train"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_LABELED_BPE_CORE_PROTOCOL_V41.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_labeled_bpe_core/development_v41/L0-seed240017")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = inventory(root, (root / args.protocol).resolve()) if args.command == "inventory" else train(root, (root / args.protocol).resolve(), (root / args.candidate_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
