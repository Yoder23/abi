"""Bounded residual-attention correction of the frozen qualified layer-1 path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from safetensors.torch import load_file, save_file
import torch

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-residual-attention-fit/1"


def _combine_attention(
    primary_attention: torch.Tensor,
    secondary_attention: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Add only the secondary layer's residual delta to the frozen primary path."""
    if primary_attention.shape != secondary_attention.shape or hidden.shape != primary_attention.shape:
        raise Phase3Error("residual-attention tensor shape changed")
    return primary_attention + (secondary_attention - hidden)


def _secondary_layer(device: torch.device, primary: Any, protocol: dict[str, Any]) -> Any:
    from layercake.dual_path_progressive_core import DualPathProgressiveLayer

    architecture = protocol["secondary_architecture"]
    layer = DualPathProgressiveLayer(
        int(architecture["full_width"]),
        int(architecture["bottleneck_width"]),
        int(architecture["attention_heads"]),
        int(architecture["intermediate_size"]),
        rms_epsilon=float(architecture["rms_epsilon"]),
        rope_theta=float(architecture["rope_theta"]),
    )
    with torch.no_grad():
        layer.input_norm.weight.copy_(primary.input_norm.weight.detach().cpu())
        layer.post_attention_norm.weight.copy_(primary.post_attention_norm.weight.detach().cpu())
        layer.attention_output_projection.weight.zero_()
    del (
        layer.mlp_input_projection,
        layer.mlp_norm,
        layer.gate_up_proj,
        layer.down_proj,
        layer.mlp_output_projection,
    )
    layer.input_norm.weight.requires_grad_(False)
    layer.post_attention_norm.weight.requires_grad_(False)
    return layer.to(device)


def execute(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_BOUNDED_RESIDUAL_ATTENTION_FIT"
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("residual-attention governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"residual-attention binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")

    output.mkdir(parents=True)
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    device = torch.device("cuda")
    prefix, tokenizer, _, attention_keys, _ = sequential._model(root, base, device)
    state = prefix.state_dict()
    for layer_index in (0, 1):
        checkpoint = load_file(
            str(root / protocol["base_checkpoints"][str(layer_index)]["path"]),
            device="cpu",
        )
        for name, value in checkpoint.items():
            if name in state:
                state[name].copy_(value.to(state[name].dtype))

    qualified = load_file(str(root / protocol["qualified_primary_checkpoint"]["path"]), device="cpu")
    expected_primary_names = {
        name for name in attention_keys if name.startswith("layers.1.")
    }
    if set(qualified) != expected_primary_names:
        raise Phase3Error("qualified primary attention tensor boundary changed")
    with torch.no_grad():
        for name, value in qualified.items():
            state[name].copy_(value.to(state[name].dtype))
    for parameter in prefix.parameters():
        parameter.requires_grad_(False)
    prefix.eval()
    primary = prefix.layers[1]

    set_determinism(int(protocol["training"]["seed"]))
    secondary = _secondary_layer(device, primary, protocol)
    trainable = [parameter for parameter in secondary.parameters() if parameter.requires_grad]
    zero_output_at_start = bool(torch.count_nonzero(secondary.attention_output_projection.weight).item() == 0)
    if not zero_output_at_start:
        raise Phase3Error("secondary output projection is not exactly zero initialized")

    examples = sequential.field._examples(root, base, tokenizer)
    calibration = base["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(calibration["train_records_per_capability"]),
        validation_per_capability=int(calibration["validation_records_per_capability"]),
        maximum_tokens=int(calibration["maximum_sequence_tokens"]),
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    source_layer = teacher.model.layers[1]
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["steps"])
    offset = int(training["record_offset"])
    curves: list[dict[str, float | int]] = []
    secondary.train()
    for step in range(steps):
        row = train_rows[(step + offset) % len(train_rows)]
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        positions = torch.arange(ids.shape[1], device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = dual.base._prefix_hidden(prefix, ids, 1)
            attention_target, final_target = dual._teacher_components(teacher, 1, hidden)
            feature_target = source_layer.post_attention_layernorm(attention_target)
            primary_attention = sequential._student_attention(primary, hidden, positions)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            secondary_attention = sequential._student_attention(secondary, hidden, positions)
            attention = _combine_attention(primary_attention, secondary_attention, hidden)
            feature = primary.post_attention_norm(attention)
            final = attention + source_layer.mlp(feature)
            attention_rmse, attention_cosine = dual.base._metrics(attention, attention_target, hidden)
            final_rmse, final_cosine = dual.base._metrics(final, final_target, hidden)
            feature_rmse = torch.sqrt(
                (feature.float() - feature_target.float()).square().mean()
                / feature_target.float().square().mean().clamp_min(1e-8)
            )
            loss = (
                attention_rmse.square()
                + final_rmse.square()
                + feature_rmse.square()
                + float(training["cosine_weight"])
                * (2.0 - attention_cosine - final_cosine)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
        optimizer.step()
        if step == 0 or (step + 1) % int(training["curve_interval"]) == 0:
            curves.append(
                {
                    "step": step + 1,
                    "attention_relative_rmse": float(attention_rmse.detach()),
                    "feature_relative_rmse": float(feature_rmse.detach()),
                    "final_relative_rmse": float(final_rmse.detach()),
                    "final_cosine": float(final_cosine.detach()),
                    "loss": float(loss.detach()),
                }
            )

    secondary.eval()
    attention_rmses: list[float] = []
    feature_rmses: list[float] = []
    final_rmses: list[float] = []
    final_cosines: list[float] = []
    primary_rmses: list[float] = []
    primary_cosines: list[float] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            positions = torch.arange(ids.shape[1], device=device)
            hidden = dual.base._prefix_hidden(prefix, ids, 1)
            attention_target, final_target = dual._teacher_components(teacher, 1, hidden)
            feature_target = source_layer.post_attention_layernorm(attention_target)
            primary_attention = sequential._student_attention(primary, hidden, positions)
            primary_feature = primary.post_attention_norm(primary_attention)
            primary_final = primary_attention + source_layer.mlp(primary_feature)
            primary_rmse, primary_cosine = dual.base._metrics(primary_final, final_target, hidden)
            secondary_attention = sequential._student_attention(secondary, hidden, positions)
            attention = _combine_attention(primary_attention, secondary_attention, hidden)
            feature = primary.post_attention_norm(attention)
            final = attention + source_layer.mlp(feature)
            attention_rmse, _ = dual.base._metrics(attention, attention_target, hidden)
            final_rmse, final_cosine = dual.base._metrics(final, final_target, hidden)
            feature_rmse = torch.sqrt(
                (feature.float() - feature_target.float()).square().mean()
                / feature_target.float().square().mean().clamp_min(1e-8)
            )
            primary_rmses.append(float(primary_rmse))
            primary_cosines.append(float(primary_cosine))
            attention_rmses.append(float(attention_rmse))
            feature_rmses.append(float(feature_rmse))
            final_rmses.append(float(final_rmse))
            final_cosines.append(float(final_cosine))

    mean_rmse = sum(final_rmses) / len(final_rmses)
    mean_cosine = sum(final_cosines) / len(final_cosines)
    gate = protocol["gate"]
    passed = (
        mean_rmse <= float(gate["mean_relative_rmse_maximum"])
        and mean_cosine >= float(gate["mean_output_cosine_minimum"])
    )
    weights = {
        name: value.detach().to(torch.float16).cpu().contiguous()
        for name, value in secondary.named_parameters()
    }
    checkpoint_path = output / "layer1_residual_attention.safetensors"
    save_file(
        weights,
        str(checkpoint_path),
        metadata={"format": FORMAT, "protocol_sha256": sha256_file(protocol_path)},
    )
    result = {
        "format": FORMAT,
        "status": "PASS_RESIDUAL_ATTENTION_INTERFACE" if passed else "FAIL_RESIDUAL_ATTENTION_INTERFACE",
        "protocol_sha256": sha256_file(protocol_path),
        "steps": steps,
        "record_offset": offset,
        "calibration_tokens": calibration_tokens,
        "zero_output_at_start": zero_output_at_start,
        "primary_checkpoint_unchanged": sha256_file(root / protocol["qualified_primary_checkpoint"]["path"])
        == protocol["qualified_primary_checkpoint"]["sha256"],
        "curves": curves,
        "primary_validation": {
            "mean_relative_rmse": sum(primary_rmses) / len(primary_rmses),
            "mean_output_cosine": sum(primary_cosines) / len(primary_cosines),
        },
        "combined_validation": {
            "mean_attention_relative_rmse": sum(attention_rmses) / len(attention_rmses),
            "mean_feature_relative_rmse": sum(feature_rmses) / len(feature_rmses),
            "mean_relative_rmse": mean_rmse,
            "maximum_relative_rmse": max(final_rmses),
            "mean_output_cosine": mean_cosine,
            "minimum_output_cosine": min(final_cosines),
            "passed": passed,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "parameters": sum(value.numel() for value in weights.values()),
        },
        "complete_source_mlp_promoted": False,
        "artifact_promoted": False,
        "training_performed": True,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Bounded layer-1 residual-attention interface fit only; no host, artifact, English quality, runtime, certificate, or superiority claim.",
    }
    _write_immutable(
        output / "metadata.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_RESIDUAL_ATTENTION_FIT_PROTOCOL_V283.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_residual_attention/layer1_fit_v284",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
