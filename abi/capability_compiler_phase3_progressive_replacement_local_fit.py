"""Layer-local progressive replacement fit against frozen source blocks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_causal_field_core as field
from .capability_compiler_phase2_common import CAPABILITIES, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-progressive-replacement-local-fit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_BOUNDED_LAYER_LOCAL_GPU_FIT"
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("progressive local-fit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"progressive local-fit binding changed: {name}")
    return protocol, sha256_file(path)


def _types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.source_aligned_progressive_replacement_core import SourceAlignedProgressiveReplacementCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    return SourceAlignedProgressiveReplacementCore, DecoderAwareExternalTokenizer


def replacement_trainable_keys(model: torch.nn.Module) -> set[str]:
    frozen_suffixes = ("input_norm.weight", "post_attention_norm.weight")
    return {
        name for name, _ in model.named_parameters()
        if name.startswith("layers.") and not name.endswith(frozen_suffixes)
    }


def _calibration_examples(examples: list[dict[str, Any]], *, seed: int, train_per_capability: int, validation_per_capability: int, maximum_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        grouped[str(row["capability"])].append(row)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    token_count = 0
    for capability in CAPABILITIES:
        ranked = sorted(
            grouped[capability],
            key=lambda row: hashlib.sha256(f"{seed}:{row['record_id']}".encode()).digest(),
        )
        selected = ranked[: train_per_capability + validation_per_capability]
        if len(selected) != train_per_capability + validation_per_capability:
            raise Phase3Error("insufficient capability calibration records")
        for index, row in enumerate(selected):
            packed = list(row["source_ids"]) + list(row["target_actions"])[:-1]
            packed = packed[:maximum_tokens]
            if not packed:
                raise Phase3Error("empty local-fit sequence")
            value = {"record_id": row["record_id"], "capability": capability, "input_ids": packed}
            token_count += len(packed)
            (train if index < train_per_capability else validation).append(value)
    train.sort(key=lambda row: hashlib.sha256(f"train:{seed}:{row['record_id']}".encode()).digest())
    validation.sort(key=lambda row: hashlib.sha256(f"validation:{seed}:{row['record_id']}".encode()).digest())
    return train, validation, token_count


def _model(root: Path, protocol: Mapping[str, Any], device: torch.device):
    model_type, tokenizer_type = _types(root, protocol)
    tokenizer = field._tokenizer(protocol, tokenizer_type)
    set_determinism(int(protocol["training"]["seed"]))
    model = model_type(fixed_vocab_size=tokenizer.vocab_size, **protocol["architecture"]).bind_tokenizer(tokenizer)
    substrate = load_file(str(root / protocol["substrate"]["path"]), device="cpu")
    missing, unexpected = model.load_state_dict(substrate, strict=False, assign=True)
    expected_trainable = replacement_trainable_keys(model)
    if set(missing) != expected_trainable or unexpected:
        raise Phase3Error("copied substrate and replacement state boundary changed")
    with torch.no_grad():
        for layer in model.layers:
            layer.output_projection.weight.zero_()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in expected_trainable)
    return model.to(device), tokenizer, substrate, expected_trainable


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer, substrate, trainable = _model(root, protocol, torch.device("cpu"))
    examples = field._examples(root, protocol, tokenizer)
    cfg = protocol["calibration"]
    train, validation, tokens = _calibration_examples(
        examples, seed=int(protocol["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "records": len(examples),
        "calibration_train_records": len(train),
        "calibration_validation_records": len(validation),
        "calibration_tokens": tokens,
        "runtime_vocabulary": tokenizer.vocab_size,
        "deployed_parameters": model.parameter_count(),
        "copied_substrate_parameters": sum(value.numel() for value in substrate.values()),
        "trainable_replacement_parameters": sum(parameter.numel() for name, parameter in model.named_parameters() if name in trainable),
        "trainable_tensor_keys": len(trainable),
        "source_model_loaded": False,
        "tensor_values_read": True,
        "training_performed": False,
        "final_test_accessed": False,
    }


def _causal_mask(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = torch.full((length, length), torch.finfo(dtype).min, device=device, dtype=dtype)
    return torch.triu(value, diagonal=1)[None, None]


def _prefix_hidden(model: Any, input_ids: torch.Tensor, layer_index: int) -> torch.Tensor:
    hidden = model.token_embedding(input_ids)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    for layer in model.layers[:layer_index]:
        hidden, _, _ = layer.forward_with_cache(hidden, positions)
    return hidden


def _teacher_target(teacher: Any, layer_index: int, hidden: torch.Tensor) -> torch.Tensor:
    length = hidden.shape[1]
    position_ids = torch.arange(length, device=hidden.device)[None]
    position_embeddings = teacher.model.rotary_emb(hidden, position_ids)
    mask = _causal_mask(length, device=hidden.device, dtype=hidden.dtype)
    return teacher.model.layers[layer_index](
        hidden, attention_mask=mask, position_ids=position_ids,
        use_cache=False, position_embeddings=position_embeddings,
    )


def _metrics(prediction: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = (prediction.float() - target.float()).square().mean()
    denominator = (target.float() - source.float()).square().mean().clamp_min(1e-8)
    relative_rmse = torch.sqrt(numerator / denominator)
    cosine = F.cosine_similarity(prediction.float().reshape(-1, prediction.shape[-1]), target.float().reshape(-1, target.shape[-1]), dim=-1).mean()
    return relative_rmse, cosine


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("local-fit output exists or CUDA unavailable")
    device = torch.device("cuda")
    model, tokenizer, substrate, trainable_keys = _model(root, protocol, device)
    examples = field._examples(root, protocol, tokenizer)
    cfg = protocol["calibration"]
    train_rows, validation_rows, calibration_tokens = _calibration_examples(
        examples, seed=int(protocol["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    process = psutil.Process(); peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats(); load_started = time.perf_counter()
    teacher = AutoModelForCausalLM.from_pretrained(
        protocol["source"]["snapshot_path"], local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(protocol["source"]["parameter_count"]):
        raise Phase3Error("loaded source parameter count changed")
    teacher_load_seconds = time.perf_counter() - load_started

    steps_per_layer = int(protocol["training"]["steps_per_layer"])
    positions_by_layer: list[dict[str, Any]] = []
    started = time.perf_counter()
    all_pass = True
    for layer_index, layer in enumerate(model.layers):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(f"layers.{layer_index}.") and name in trainable_keys)
        parameters = [parameter for parameter in layer.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=float(protocol["training"]["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(protocol["training"]["weight_decay"]))
        layer.train(); curves = []
        for step in range(steps_per_layer):
            row = train_rows[(step + layer_index * steps_per_layer) % len(train_rows)]
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = _prefix_hidden(model, ids, layer_index)
                target = _teacher_target(teacher, layer_index, hidden)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                positions = torch.arange(ids.shape[1], device=device)
                prediction, _, _ = layer.forward_with_cache(hidden, positions)
                relative_rmse, cosine = _metrics(prediction, target, hidden)
                loss = relative_rmse.square() + float(protocol["training"]["cosine_weight"]) * (1.0 - cosine)
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, float(protocol["training"]["gradient_clip_norm"])); optimizer.step()
            if step == 0 or (step + 1) % int(protocol["training"]["curve_interval"]) == 0:
                curve = {"step": step + 1, "relative_rmse": float(relative_rmse.detach()), "cosine": float(cosine.detach()), "loss": float(loss.detach())}
                curves.append(curve)
        layer.eval()
        validation_rmse = []; validation_cosine = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in validation_rows:
                ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
                hidden = _prefix_hidden(model, ids, layer_index)
                target = _teacher_target(teacher, layer_index, hidden)
                prediction, _, _ = layer.forward_with_cache(hidden, torch.arange(ids.shape[1], device=device))
                relative_rmse, cosine = _metrics(prediction, target, hidden)
                validation_rmse.append(float(relative_rmse)); validation_cosine.append(float(cosine))
        mean_rmse = sum(validation_rmse) / len(validation_rmse); mean_cosine = sum(validation_cosine) / len(validation_cosine)
        passed = mean_rmse <= float(protocol["local_gate"]["mean_relative_rmse_maximum"]) and mean_cosine >= float(protocol["local_gate"]["mean_output_cosine_minimum"])
        layer_result = {"layer": layer_index, "steps": steps_per_layer, "mean_validation_relative_rmse": mean_rmse, "maximum_validation_relative_rmse": max(validation_rmse), "mean_validation_output_cosine": mean_cosine, "minimum_validation_output_cosine": min(validation_cosine), "passed": passed, "curves": curves}
        positions_by_layer.append(layer_result); all_pass = all_pass and passed
        print(json.dumps(layer_result), flush=True)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if not passed and protocol["local_gate"]["fail_fast"]:
            break

    output.mkdir(parents=True)
    replacement = {
        name: parameter.detach().to(torch.float16).cpu().contiguous()
        for name, parameter in model.named_parameters() if name in trainable_keys
    }
    replacement_path = output / "replacement_weights.safetensors"
    save_file(replacement, str(replacement_path), metadata={"format": "abi-progressive-replacement-local-fit/1", "protocol_sha256": protocol_sha})
    unchanged = all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in substrate.items())
    metadata = {
        "format": FORMAT,
        "status": "PASS_LOCAL_FIT_END_TO_END_PROTOCOL_MAY_BE_DESIGNED" if all_pass and len(positions_by_layer) == int(protocol["architecture"]["replacement_layers"]) else "FAIL_LOCAL_FIT_BRANCH_CLOSED",
        "protocol_sha256": protocol_sha,
        "source": {"model": protocol["source"]["model"], "revision": protocol["source"]["revision"], "teacher_load_seconds": teacher_load_seconds, "teacher_present_in_artifact": False},
        "calibration": {"train_records": len(train_rows), "validation_records": len(validation_rows), "tokens": calibration_tokens, "maximum_sequence_tokens": int(cfg["maximum_sequence_tokens"])},
        "layers": positions_by_layer,
        "replacement": {"path": replacement_path.name, "sha256": sha256_file(replacement_path), "file_bytes": replacement_path.stat().st_size, "parameters": sum(value.numel() for value in replacement.values()), "tensor_keys": len(replacement)},
        "copied_substrate": {"path": protocol["substrate"]["path"], "sha256": sha256_file(root / protocol["substrate"]["path"]), "unchanged_after_training": unchanged, "parameters": sum(value.numel() for value in substrate.values())},
        "accounting": {"training_steps_completed": sum(row["steps"] for row in positions_by_layer), "training_wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "source_model_inference_hours": (time.perf_counter() - started) / 3600.0, "stored_logits": 0, "stored_activations": 0, "complete_source_blocks_in_artifact": 0},
        "teacher_required_at_inference": False, "training_performed": True, "phase3_certified": False, "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
        "next_gate": "Preregister bounded end-to-end cached-field conformance only if every local layer passed." if all_pass else "Close this exact progressive replacement architecture and preserve the local failure.",
        "claim_boundary": "Layer-local source-block approximation only; no autonomous English quality, speed, transfer, Phase 3 certificate, or superiority claim."
    }
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("inventory", "train")); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_PROGRESSIVE_REPLACEMENT_LOCAL_FIT_PROTOCOL_V225.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_progressive_replacement/local_fit_v226")
    args = parser.parse_args(); root = Path.cwd().resolve(); result = inventory(root, root / args.protocol) if args.command == "inventory" else train(root, root / args.protocol, root / args.output); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if not str(result["status"]).startswith("FAIL") else 1


if __name__ == "__main__": raise SystemExit(main())
