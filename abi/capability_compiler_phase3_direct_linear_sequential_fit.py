"""Sequential compact-attention fit and replacement-conditioned analytic MLP extraction.

This is deliberately a single preregistered path: attention is optimized first, the
linear MLP map is then solved on the exact student-attention interface, and the
integrated replacement is validated before the next source layer is exposed.
"""

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

from . import capability_compiler_phase3_causal_field_core as field
from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-direct-linear-sequential-fit/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_FAIL_FAST_32_LAYER_GPU_FIT"
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("direct-linear sequential-fit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"direct-linear sequential-fit binding changed: {name}")
    return protocol, sha256_file(path)


def _types(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.direct_linear_progressive_core import DirectLinearProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    return DirectLinearProgressiveCore, DecoderAwareExternalTokenizer


def key_classes(model: torch.nn.Module) -> tuple[set[str], set[str]]:
    """Return the fitted-attention and analytically imported tensor boundaries."""
    attention_modules = {
        "attention_input_projection", "attention_norm", "qkv_proj", "o_proj",
        "attention_output_projection",
    }
    imported_modules = {
        "mlp_coefficient_projection", "mlp_output_projection", "mlp_residual_mean",
    }
    def module_name(name: str) -> str:
        pieces = name.split(".")
        return pieces[-1] if pieces[-1] == "mlp_residual_mean" else pieces[-2]
    attention = {
        name for name, _ in model.named_parameters()
        if name.startswith("layers.") and module_name(name) in attention_modules
    }
    imported = {
        name for name, _ in model.named_parameters()
        if name.startswith("layers.") and module_name(name) in imported_modules
    }
    return attention, imported


def _model(root: Path, protocol: Mapping[str, Any], device: torch.device):
    model_type, tokenizer_type = _types(root, protocol)
    tokenizer = field._tokenizer(protocol, tokenizer_type)
    set_determinism(int(protocol["training"]["seed"]))
    model = model_type(fixed_vocab_size=tokenizer.vocab_size, **protocol["architecture"]).bind_tokenizer(tokenizer)
    substrate = load_file(str(root / protocol["substrate"]["path"]), device="cpu")
    missing, unexpected = model.load_state_dict(substrate, strict=False, assign=True)
    attention, imported = key_classes(model)
    if set(missing) != attention | imported or unexpected:
        raise Phase3Error("direct-linear copied/fitted/imported tensor boundary changed")
    with torch.no_grad():
        for layer in model.layers:
            layer.attention_output_projection.weight.zero_()
            layer.mlp_coefficient_projection.weight.zero_()
            layer.mlp_output_projection.weight.zero_()
            layer.mlp_residual_mean.zero_()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in attention)
    return model.to(device), tokenizer, substrate, attention, imported


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, tokenizer, substrate, attention, imported = _model(root, protocol, torch.device("cpu"))
    examples = field._examples(root, protocol, tokenizer)
    cfg = protocol["calibration"]
    train, validation, tokens = dual._calibration_examples(
        examples,
        seed=int(protocol["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    attention_parameters = sum(model.get_parameter(name).numel() for name in attention)
    imported_parameters = sum(model.get_parameter(name).numel() for name in imported)
    copied_parameters = sum(value.numel() for value in substrate.values())
    return {
        "format": FORMAT,
        "status": "PASS_INVENTORY_NO_SOURCE_MODEL_LOADED",
        "protocol_sha256": protocol_sha,
        "records": len(examples),
        "calibration_train_records": len(train),
        "calibration_validation_records": len(validation),
        "calibration_tokens": tokens,
        "runtime_vocabulary": tokenizer.vocab_size,
        "deployed_parameters": model.parameter_count(),
        "copied_substrate_parameters": copied_parameters,
        "fitted_attention_parameters": attention_parameters,
        "analytically_imported_parameters": imported_parameters,
        "replacement_parameters": attention_parameters + imported_parameters,
        "fitted_attention_tensor_keys": len(attention),
        "analytically_imported_tensor_keys": len(imported),
        "source_model_loaded": False,
        "training_performed": False,
        "final_test_accessed": False,
    }


def _student_attention(layer: Any, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    latent = layer.attention_input_projection(layer.input_norm(hidden))
    query, key, value = layer._qkv(latent, positions)
    attended = layer._attention(query, key, value, causal=True).transpose(1, 2).contiguous().view_as(latent)
    return hidden + layer.attention_output_projection(layer.o_proj(attended))


def _student_final(layer: Any, attention: torch.Tensor) -> torch.Tensor:
    return attention + layer.mlp_residual_mean + layer.mlp_output_projection(
        layer.mlp_coefficient_projection(layer.post_attention_norm(attention))
    )


def _fit_replacement_conditioned_map(
    teacher: Any,
    model: Any,
    rows: list[dict[str, Any]],
    layer_index: int,
    rank: int,
    relative_ridge: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float, float]:
    layer = model.layers[layer_index]
    deltas: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            hidden = dual.base._prefix_hidden(model, ids, layer_index)
            teacher_attention, teacher_final = dual._teacher_components(teacher, layer_index, hidden)
            student_attention = _student_attention(layer, hidden, torch.arange(ids.shape[1], device=device))
            deltas.append((teacher_final - teacher_attention).squeeze(0).float().cpu())
            features.append(layer.post_attention_norm(student_attention).squeeze(0).float().cpu())
    mean, covariance, observations = rank_audit.centered_covariance(deltas, model.full_width, device)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    x = torch.cat(features).to(device)
    y = (torch.cat(deltas).to(device) - mean) @ basis
    weights, ridge = closed.solve_ridge(x, y, relative_ridge)
    prediction = x @ weights
    coefficient_rmse = float(torch.sqrt((prediction - y).square().mean() / y.square().mean().clamp_min(1e-8)))
    return mean, basis, weights, observations, energy, ridge, coefficient_rmse


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("direct-linear sequential-fit output exists or CUDA unavailable")
    output.mkdir(parents=True)
    device = torch.device("cuda")
    model, tokenizer, substrate, attention_keys, imported_keys = _model(root, protocol, device)
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
        protocol["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(protocol["source"]["parameter_count"]):
        raise Phase3Error("loaded source parameter count changed")
    teacher_load_seconds = time.perf_counter() - load_started

    steps_per_layer = int(protocol["training"]["attention_steps_per_layer"])
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    complete = True
    for layer_index, layer in enumerate(model.layers):
        current_prefix = f"layers.{layer_index}."
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(current_prefix) and name in attention_keys)
        parameters = [parameter for parameter in layer.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(protocol["training"]["learning_rate"]),
            betas=(0.9, 0.95),
            weight_decay=float(protocol["training"]["weight_decay"]),
        )
        curves: list[dict[str, float | int]] = []
        layer.train()
        for step in range(steps_per_layer):
            row = train_rows[(step + layer_index * steps_per_layer) % len(train_rows)]
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = dual.base._prefix_hidden(model, ids, layer_index)
                attention_target, _ = dual._teacher_components(teacher, layer_index, hidden)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = _student_attention(layer, hidden, torch.arange(ids.shape[1], device=device))
                relative_rmse, cosine = dual.base._metrics(prediction, attention_target, hidden)
                loss = relative_rmse.square() + float(protocol["training"]["cosine_weight"]) * (1.0 - cosine)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(protocol["training"]["gradient_clip_norm"]))
            optimizer.step()
            if step == 0 or (step + 1) % int(protocol["training"]["curve_interval"]) == 0:
                curves.append({
                    "step": step + 1,
                    "attention_relative_rmse": float(relative_rmse.detach()),
                    "attention_cosine": float(cosine.detach()),
                    "loss": float(loss.detach()),
                })
        layer.eval()
        mean, basis, weights, observations, energy, ridge, training_coefficient_rmse = _fit_replacement_conditioned_map(
            teacher, model, train_rows, layer_index,
            int(protocol["training"]["basis_rank"]),
            float(protocol["training"]["relative_ridge"]), device,
        )
        with torch.no_grad():
            layer.mlp_residual_mean.copy_(mean.to(layer.mlp_residual_mean.dtype))
            layer.mlp_output_projection.weight.copy_(basis.to(layer.mlp_output_projection.weight.dtype))
            layer.mlp_coefficient_projection.weight.copy_(weights.T.to(layer.mlp_coefficient_projection.weight.dtype))

        attention_rmses: list[float] = []
        attention_cosines: list[float] = []
        final_rmses: list[float] = []
        final_cosines: list[float] = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in validation_rows:
                ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
                hidden = dual.base._prefix_hidden(model, ids, layer_index)
                attention_target, final_target = dual._teacher_components(teacher, layer_index, hidden)
                attention = _student_attention(layer, hidden, torch.arange(ids.shape[1], device=device))
                final = _student_final(layer, attention)
                attention_rmse, attention_cosine = dual.base._metrics(attention, attention_target, hidden)
                final_rmse, final_cosine = dual.base._metrics(final, final_target, hidden)
                attention_rmses.append(float(attention_rmse)); attention_cosines.append(float(attention_cosine))
                final_rmses.append(float(final_rmse)); final_cosines.append(float(final_cosine))
        mean_rmse = sum(final_rmses) / len(final_rmses)
        mean_cosine = sum(final_cosines) / len(final_cosines)
        passed = (
            mean_rmse <= float(protocol["local_gate"]["mean_relative_rmse_maximum"])
            and mean_cosine >= float(protocol["local_gate"]["mean_output_cosine_minimum"])
        )
        result = {
            "layer": layer_index,
            "attention_steps": steps_per_layer,
            "map_train_observations": observations,
            "basis_energy_explained": energy,
            "effective_ridge": ridge,
            "training_coefficient_relative_rmse": training_coefficient_rmse,
            "mean_validation_attention_relative_rmse": sum(attention_rmses) / len(attention_rmses),
            "maximum_validation_attention_relative_rmse": max(attention_rmses),
            "mean_validation_attention_cosine": sum(attention_cosines) / len(attention_cosines),
            "minimum_validation_attention_cosine": min(attention_cosines),
            "mean_validation_relative_rmse": mean_rmse,
            "maximum_validation_relative_rmse": max(final_rmses),
            "mean_validation_output_cosine": mean_cosine,
            "minimum_validation_output_cosine": min(final_cosines),
            "passed": passed,
            "curves": curves,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        layer_state = {
            name: parameter.detach().to(torch.float16).cpu().contiguous()
            for name, parameter in model.named_parameters()
            if name.startswith(current_prefix) and name in attention_keys | imported_keys
        }
        save_file(layer_state, str(output / f"layer_{layer_index:02d}.safetensors"), metadata={
            "format": "abi-direct-linear-sequential-layer/1",
            "protocol_sha256": protocol_sha,
            "passed": str(passed).lower(),
        })
        peak_rss = max(peak_rss, process.memory_info().rss)
        if not passed:
            complete = False
            if protocol["local_gate"]["fail_fast"]:
                break

    replacement = {
        name: parameter.detach().to(torch.float16).cpu().contiguous()
        for name, parameter in model.named_parameters()
        if name in attention_keys | imported_keys
    }
    replacement_path = output / "replacement_weights.safetensors"
    save_file(replacement, str(replacement_path), metadata={
        "format": "abi-direct-linear-sequential-fit/1", "protocol_sha256": protocol_sha,
    })
    unchanged = all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in substrate.items())
    all_pass = complete and len(results) == int(protocol["architecture"]["replacement_layers"])
    metadata = {
        "format": FORMAT,
        "status": "PASS_LOCAL_FIT_END_TO_END_PROTOCOL_MAY_BE_DESIGNED" if all_pass else "FAIL_LOCAL_FIT_BRANCH_CLOSED",
        "protocol_sha256": protocol_sha,
        "source": {
            "model": protocol["source"]["model"], "revision": protocol["source"]["revision"],
            "teacher_load_seconds": teacher_load_seconds, "teacher_present_in_artifact": False,
        },
        "calibration": {
            "train_records": len(train_rows), "validation_records": len(validation_rows),
            "tokens": calibration_tokens, "maximum_sequence_tokens": int(cfg["maximum_sequence_tokens"]),
        },
        "layers": results,
        "replacement": {
            "path": replacement_path.name, "sha256": sha256_file(replacement_path),
            "file_bytes": replacement_path.stat().st_size,
            "parameters": sum(value.numel() for value in replacement.values()),
            "tensor_keys": len(replacement),
            "fitted_attention_parameters": sum(model.get_parameter(name).numel() for name in attention_keys),
            "analytically_imported_parameters": sum(model.get_parameter(name).numel() for name in imported_keys),
        },
        "copied_substrate": {
            "path": protocol["substrate"]["path"], "sha256": sha256_file(root / protocol["substrate"]["path"]),
            "unchanged_after_fit": unchanged, "parameters": sum(value.numel() for value in substrate.values()),
        },
        "accounting": {
            "attention_steps_completed": sum(row["attention_steps"] for row in results),
            "fit_wall_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "stored_logits": 0, "stored_raw_activations": 0, "complete_source_blocks_in_artifact": 0,
        },
        "teacher_required_at_inference": False,
        "training_performed": True,
        "phase3_certified": False,
        "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
        "next_gate": "Preregister end-to-end cached-field conformance." if all_pass else "Close this exact branch and preserve its failure.",
        "claim_boundary": "Sequential layer-local replacement evidence only; no autonomous English quality, measured inference, Phase 3 certificate, or superiority claim.",
    }
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_LINEAR_SEQUENTIAL_FIT_PROTOCOL_V255.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_direct_linear/sequential_fit_v256")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = inventory(root, root / args.protocol) if args.command == "inventory" else train(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not str(result["status"]).startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
