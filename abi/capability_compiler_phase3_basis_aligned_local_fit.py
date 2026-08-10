"""Per-layer basis extraction and direct coefficient fit for the v11 host."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_causal_field_core as field
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-basis-aligned-local-fit/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_BOUNDED_BASIS_ALIGNED_GPU_FIT"
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("basis-aligned local-fit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"basis-aligned local-fit binding changed: {name}")
    return protocol, sha256_file(path)


def _types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.basis_aligned_progressive_core import BasisAlignedProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    return BasisAlignedProgressiveCore, DecoderAwareExternalTokenizer


def key_classes(model: torch.nn.Module) -> tuple[set[str], set[str]]:
    copied = {
        f"layers.{index}.{name}.weight"
        for index in range(len(model.layers))
        for name in ("input_norm", "post_attention_norm")
    }
    imported = {
        f"layers.{index}.{name}"
        for index in range(len(model.layers))
        for name in ("mlp_output_projection.weight", "mlp_residual_mean")
    }
    trainable = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("layers.") and name not in copied and name not in imported
    }
    return trainable, imported


def _model(root: Path, protocol: Mapping[str, Any], device: torch.device):
    model_type, tokenizer_type = _types(root, protocol)
    tokenizer = field._tokenizer(protocol, tokenizer_type)
    set_determinism(int(protocol["training"]["seed"]))
    model = model_type(fixed_vocab_size=tokenizer.vocab_size, **protocol["architecture"]).bind_tokenizer(tokenizer)
    substrate = load_file(str(root / protocol["substrate"]["path"]), device="cpu")
    missing, unexpected = model.load_state_dict(substrate, strict=False, assign=True)
    trainable, imported = key_classes(model)
    if set(missing) != trainable | imported or unexpected:
        raise Phase3Error("basis-aligned copied/imported/trainable boundary changed")
    with torch.no_grad():
        for layer in model.layers:
            layer.attention_output_projection.weight.zero_()
            layer.mlp_output_projection.weight.zero_()
            layer.mlp_residual_mean.zero_()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)
    return model.to(device), tokenizer, substrate, trainable, imported


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer, substrate, trainable, imported = _model(root, protocol, torch.device("cpu"))
    examples = field._examples(root, protocol, tokenizer)
    cfg = protocol["calibration"]
    train, validation, tokens = dual._calibration_examples(
        examples,
        seed=int(protocol["training"]["seed"]),
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
        "imported_basis_and_mean_parameters": sum(model.get_parameter(name).numel() for name in imported),
        "trainable_parameters": sum(model.get_parameter(name).numel() for name in trainable),
        "imported_tensor_keys": len(imported),
        "trainable_tensor_keys": len(trainable),
        "source_model_loaded": False,
        "training_performed": False,
        "final_test_accessed": False,
    }


def _extract_basis(
    teacher: Any,
    model: Any,
    rows: list[dict[str, Any]],
    layer_index: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    deltas: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden = dual.base._prefix_hidden(model, ids, layer_index)
            attention, final = dual._teacher_components(teacher, layer_index, hidden)
            deltas.append((final - attention).squeeze(0).float().cpu())
    mean, covariance, observations = rank_audit.centered_covariance(deltas, model.full_width, device)
    del deltas
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    explained = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    return mean, basis, observations, explained


def _student(layer: Any, hidden: torch.Tensor, positions: torch.Tensor):
    attention, _ = dual._student_components(layer, hidden, positions)
    latent = layer.mlp_input_projection(layer.post_attention_norm(attention))
    gate, up = layer.gate_up_proj(layer.mlp_norm(latent)).chunk(2, dim=-1)
    coefficients = layer.down_proj(F.silu(gate) * up)
    final = attention + layer.mlp_residual_mean + layer.mlp_output_projection(coefficients)
    return attention, coefficients, final


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("basis-aligned local-fit output exists or CUDA unavailable")
    device = torch.device("cuda")
    model, tokenizer, substrate, trainable_keys, imported_keys = _model(root, protocol, device)
    examples = field._examples(root, protocol, tokenizer)
    cfg = protocol["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(protocol["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    teacher = AutoModelForCausalLM.from_pretrained(
        protocol["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(protocol["source"]["parameter_count"]):
        raise Phase3Error("loaded source parameter count changed")
    teacher_load_seconds = time.perf_counter() - load_started

    steps_per_layer = int(protocol["training"]["steps_per_layer"])
    results = []
    started = time.perf_counter()
    complete = True
    for layer_index, layer in enumerate(model.layers):
        mean, basis, basis_observations, basis_energy = _extract_basis(
            teacher, model, train_rows, layer_index, int(protocol["training"]["basis_rank"]), device
        )
        with torch.no_grad():
            layer.mlp_residual_mean.copy_(mean.to(layer.mlp_residual_mean.dtype))
            layer.mlp_output_projection.weight.copy_(basis.to(layer.mlp_output_projection.weight.dtype))
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(f"layers.{layer_index}.") and name in trainable_keys)
        parameters = [parameter for parameter in layer.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=float(protocol["training"]["learning_rate"]), betas=(0.9, 0.95),
            weight_decay=float(protocol["training"]["weight_decay"]),
        )
        curves = []
        layer.train()
        for step in range(steps_per_layer):
            row = train_rows[(step + layer_index * steps_per_layer) % len(train_rows)]
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = dual.base._prefix_hidden(model, ids, layer_index)
                attention_target, final_target = dual._teacher_components(teacher, layer_index, hidden)
                coefficient_target = (final_target.float() - attention_target.float() - mean) @ basis
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                attention, coefficients, final = _student(layer, hidden, torch.arange(ids.shape[1], device=device))
                attention_rmse, attention_cosine = dual.base._metrics(attention, attention_target, hidden)
                final_rmse, final_cosine = dual.base._metrics(final, final_target, hidden)
                coefficient_rmse = torch.sqrt(
                    (coefficients.float() - coefficient_target).square().mean()
                    / coefficient_target.square().mean().clamp_min(1e-8)
                )
                loss = attention_rmse.square() + final_rmse.square() + float(protocol["training"]["coefficient_weight"]) * coefficient_rmse.square() + float(protocol["training"]["cosine_weight"]) * (2.0 - attention_cosine - final_cosine)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(protocol["training"]["gradient_clip_norm"]))
            optimizer.step()
            if step == 0 or (step + 1) % int(protocol["training"]["curve_interval"]) == 0:
                curves.append({"step": step + 1, "attention_relative_rmse": float(attention_rmse.detach()), "final_relative_rmse": float(final_rmse.detach()), "coefficient_relative_rmse": float(coefficient_rmse.detach()), "final_cosine": float(final_cosine.detach()), "loss": float(loss.detach())})
        layer.eval()
        rmses, cosines, coefficient_rmses = [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in validation_rows:
                ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
                hidden = dual.base._prefix_hidden(model, ids, layer_index)
                attention_target, final_target = dual._teacher_components(teacher, layer_index, hidden)
                coefficient_target = (final_target.float() - attention_target.float() - mean) @ basis
                _, coefficients, final = _student(layer, hidden, torch.arange(ids.shape[1], device=device))
                rmse, cosine = dual.base._metrics(final, final_target, hidden)
                coefficient_rmse = torch.sqrt((coefficients.float() - coefficient_target).square().mean() / coefficient_target.square().mean().clamp_min(1e-8))
                rmses.append(float(rmse)); cosines.append(float(cosine)); coefficient_rmses.append(float(coefficient_rmse))
        mean_rmse = sum(rmses) / len(rmses); mean_cosine = sum(cosines) / len(cosines)
        passed = mean_rmse <= float(protocol["local_gate"]["mean_relative_rmse_maximum"]) and mean_cosine >= float(protocol["local_gate"]["mean_output_cosine_minimum"])
        row_result = {"layer": layer_index, "steps": steps_per_layer, "basis_observations": basis_observations, "training_basis_energy_explained": basis_energy, "mean_validation_coefficient_relative_rmse": sum(coefficient_rmses) / len(coefficient_rmses), "mean_validation_relative_rmse": mean_rmse, "maximum_validation_relative_rmse": max(rmses), "mean_validation_output_cosine": mean_cosine, "minimum_validation_output_cosine": min(cosines), "passed": passed, "curves": curves}
        results.append(row_result); print(json.dumps(row_result), flush=True)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if not passed:
            complete = False
            if protocol["local_gate"]["fail_fast"]:
                break

    output.mkdir(parents=True)
    replacement_keys = trainable_keys | imported_keys
    replacement = {name: parameter.detach().to(torch.float16).cpu().contiguous() for name, parameter in model.named_parameters() if name in replacement_keys}
    path = output / "replacement_weights.safetensors"
    save_file(replacement, str(path), metadata={"format": "abi-basis-aligned-local-fit/1", "protocol_sha256": protocol_sha})
    unchanged = all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in substrate.items())
    all_pass = complete and len(results) == int(protocol["architecture"]["replacement_layers"])
    metadata = {
        "format": FORMAT,
        "status": "PASS_LOCAL_FIT_END_TO_END_PROTOCOL_MAY_BE_DESIGNED" if all_pass else "FAIL_LOCAL_FIT_BRANCH_CLOSED",
        "protocol_sha256": protocol_sha,
        "source": {"model": protocol["source"]["model"], "revision": protocol["source"]["revision"], "teacher_load_seconds": teacher_load_seconds, "teacher_present_in_artifact": False},
        "calibration": {"train_records": len(train_rows), "validation_records": len(validation_rows), "tokens": calibration_tokens},
        "layers": results,
        "replacement": {"path": path.name, "sha256": sha256_file(path), "file_bytes": path.stat().st_size, "parameters": sum(value.numel() for value in replacement.values()), "tensor_keys": len(replacement), "imported_basis_and_mean_parameters": sum(model.get_parameter(name).numel() for name in imported_keys), "trainable_parameters": sum(model.get_parameter(name).numel() for name in trainable_keys)},
        "copied_substrate": {"sha256": sha256_file(root / protocol["substrate"]["path"]), "unchanged_after_training": unchanged, "parameters": sum(value.numel() for value in substrate.values())},
        "accounting": {"training_steps_completed": sum(row["steps"] for row in results), "training_wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "stored_logits": 0, "stored_raw_activations": 0, "complete_source_blocks_in_artifact": 0},
        "teacher_required_at_inference": False, "training_performed": True, "phase3_certified": False, "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
        "next_gate": "Preregister bounded end-to-end conformance only if every local layer passed." if all_pass else "Close this exact basis-aligned architecture and preserve the local failure.",
        "claim_boundary": "Basis-aligned layer-local approximation only; no autonomous English quality, speed, transfer, Phase 3 certificate, or superiority claim.",
    }
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_BASIS_ALIGNED_LOCAL_FIT_PROTOCOL_V243.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_basis_aligned/local_fit_v244")
    args = parser.parse_args(); root = Path.cwd().resolve()
    result = inventory(root, root / args.protocol) if args.command == "inventory" else train(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if not str(result["status"]).startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
