"""Prepare a GPU-aligned same-tokenizer instruction source for ABI acquisition.

The source is a temporary acquisition aid.  A frozen open-weight GPT-2 source
receives low-rank updates on its attention and MLP matrices from segregated
English-form response text.  Those updates are merged into a standalone source
checkpoint for online logit distillation; no source block is deployable in the
resulting LayerCake host.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import psutil
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer

from .layercake_full_core_acquisition import _DeterministicRowSampler
from .layercake_host import _batch, _canonical_json_bytes, _sha256_file
from .layercake_host_v3 import load_english_training_rows


FORMAT = "abi-same-tokenizer-instruction-source-alignment/1"
TARGET_SUFFIXES = (
    ".attn.c_attn",
    ".attn.c_proj",
    ".mlp.c_fc",
    ".mlp.c_proj",
)


class SourceAlignmentError(RuntimeError):
    """Raised when a source-alignment invariant is not met."""


class _LowRankWeightDelta(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, *, rank: int, alpha: float):
        super().__init__()
        if weight.ndim != 2 or min(weight.shape) < rank:
            raise SourceAlignmentError("source matrix is not rank-compatible")
        self.scale = float(alpha) / float(rank)
        self.a = torch.nn.Parameter(
            torch.empty(
                int(weight.shape[0]),
                rank,
                device=weight.device,
                dtype=torch.float32,
            )
        )
        self.b = torch.nn.Parameter(
            torch.zeros(
                rank,
                int(weight.shape[1]),
                device=weight.device,
                dtype=torch.float32,
            )
        )
        torch.nn.init.normal_(self.a, mean=0.0, std=0.02)

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        delta = (self.scale * (self.a @ self.b)).to(base_weight.dtype)
        return base_weight + delta


def _target_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if any(name.endswith(suffix) for suffix in TARGET_SUFFIXES)
        and hasattr(module, "weight")
    ]
    expected = int(model.config.n_layer) * len(TARGET_SUFFIXES)
    if len(targets) != expected:
        raise SourceAlignmentError(
            f"expected {expected} source matrices, found {len(targets)}"
        )
    return targets


def install_source_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: float,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    """Freeze the source and install temporary low-rank matrix deltas."""

    model.requires_grad_(False)
    parameters: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, module in _target_modules(model):
        delta = _LowRankWeightDelta(module.weight, rank=rank, alpha=alpha)
        parametrize.register_parametrization(module, "weight", delta)
        parameters.extend(delta.parameters())
        names.append(name)
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise SourceAlignmentError("source low-rank parameters are incomplete")
    return parameters, names


def merge_source_lora(model: torch.nn.Module) -> None:
    """Merge every temporary parametrization into its original matrix shape."""

    for _, module in _target_modules(model):
        if not parametrize.is_parametrized(module, "weight"):
            raise SourceAlignmentError("source low-rank parametrization is absent")
        parametrize.remove_parametrizations(
            module, "weight", leave_parametrized=True
        )
    if any(parametrize.is_parametrized(module, "weight") for _, module in _target_modules(model)):
        raise SourceAlignmentError("source low-rank parametrization survived merge")


def _response_loss(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def align_same_tokenizer_source(
    *,
    source_path: str | Path,
    main_bundle_path: str | Path,
    anchor_bundle_path: str | Path,
    output_path: str | Path,
    main_budget_index: int,
    anchor_budget_index: int,
    seed: int,
    steps: int,
    batch_size: int,
    anchor_batch_size: int,
    gradient_accumulation_steps: int,
    max_tokens: int,
    learning_rate: float,
    rank: int,
    alpha: float,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if device_name != "cuda" or not torch.cuda.is_available():
        raise SourceAlignmentError("same-tokenizer source alignment requires CUDA")
    if min(
        steps,
        batch_size,
        anchor_batch_size,
        gradient_accumulation_steps,
        max_tokens,
        rank,
    ) <= 0:
        raise SourceAlignmentError("alignment sizes must be positive")
    if learning_rate <= 0 or alpha <= 0:
        raise SourceAlignmentError("alignment optimization values must be positive")

    source_path = Path(source_path).resolve()
    main_bundle_path = Path(main_bundle_path).resolve()
    anchor_bundle_path = Path(anchor_bundle_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise SourceAlignmentError("alignment output already exists")
    source_checkpoint = source_path / "model.safetensors"
    source_config = source_path / "config.json"
    source_tokenizer_file = source_path / "tokenizer.json"
    for required in (source_checkpoint, source_config, source_tokenizer_file):
        if not required.is_file():
            raise SourceAlignmentError(f"source file is absent: {required.name}")

    source_checkpoint_sha_before = _sha256_file(source_checkpoint)
    main_sha_before = _sha256_file(main_bundle_path)
    anchor_sha_before = _sha256_file(anchor_bundle_path)
    main_rows, main_budget, main_bundle = load_english_training_rows(
        main_bundle_path, budget_index=main_budget_index
    )
    anchor_rows, anchor_budget, anchor_bundle = load_english_training_rows(
        anchor_bundle_path, budget_index=anchor_budget_index
    )
    main_sampler = _DeterministicRowSampler(
        main_rows, seed=seed, strategy="balanced_capabilities"
    )
    anchor_sampler = _DeterministicRowSampler(
        anchor_rows, seed=seed + 1, strategy="balanced_capabilities"
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(source_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source_path,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.config.use_cache = False
    # Reentrant checkpointing drops this graph because token IDs do not require
    # gradients and every base weight is frozen.  Non-reentrant checkpointing
    # correctly tracks the trainable parametrizations inside each block.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora_parameters, target_names = install_source_lora(
        model, rank=rank, alpha=alpha
    )
    optimizer = torch.optim.AdamW(
        lora_parameters, lr=learning_rate, weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    autocast = lambda: torch.autocast(
        device_type="cuda", dtype=torch.float16
    )

    successful_steps = 0
    attempted_steps = 0
    skipped_steps = 0
    supervised_tokens = 0
    anchor_supervised_tokens = 0
    main_utf8_bytes = 0
    anchor_utf8_bytes = 0
    sampled_main: Counter[str] = Counter()
    sampled_anchor: Counter[str] = Counter()
    unique_main: set[str] = set()
    unique_anchor: set[str] = set()
    curves: list[dict[str, Any]] = []
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    cpu_before = process.cpu_times()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()

    while successful_steps < steps:
        attempted_steps += 1
        if attempted_steps > steps + 1000:
            raise SourceAlignmentError("too many skipped alignment steps")
        main_snapshot = main_sampler.snapshot()
        anchor_snapshot = anchor_sampler.snapshot()
        optimizer.zero_grad(set_to_none=True)
        scale_before = scaler.get_scale()
        selected_main: list[Mapping[str, Any]] = []
        selected_anchor: list[Mapping[str, Any]] = []
        token_count = 0
        anchor_token_count = 0
        loss_sum = 0.0
        main_loss_sum = 0.0
        anchor_loss_sum = 0.0
        for _ in range(gradient_accumulation_steps):
            main_batch_rows = main_sampler.batch(batch_size)
            anchor_batch_rows = anchor_sampler.batch(anchor_batch_size)
            main_batch = _batch(
                tokenizer,
                main_batch_rows,
                device=device,
                max_tokens=max_tokens,
            )
            anchor_batch = _batch(
                tokenizer,
                anchor_batch_rows,
                device=device,
                max_tokens=max_tokens,
            )
            main_ids, main_labels, main_attention, _, _, main_tokens = main_batch
            anchor_ids, anchor_labels, anchor_attention, _, _, anchor_tokens = anchor_batch
            with autocast():
                main_loss = _response_loss(
                    model,
                    input_ids=main_ids,
                    labels=main_labels,
                    attention_mask=main_attention,
                )
                anchor_loss = _response_loss(
                    model,
                    input_ids=anchor_ids,
                    labels=anchor_labels,
                    attention_mask=anchor_attention,
                )
                loss = main_loss + anchor_loss
            scaler.scale(loss / gradient_accumulation_steps).backward()
            selected_main.extend(main_batch_rows)
            selected_anchor.extend(anchor_batch_rows)
            token_count += int(main_tokens)
            anchor_token_count += int(anchor_tokens)
            loss_sum += float(loss.detach())
            main_loss_sum += float(main_loss.detach())
            anchor_loss_sum += float(anchor_loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            main_sampler.restore(main_snapshot)
            anchor_sampler.restore(anchor_snapshot)
            skipped_steps += 1
            continue

        successful_steps += 1
        supervised_tokens += token_count
        anchor_supervised_tokens += anchor_token_count
        for row in selected_main:
            sampled_main[str(row["capability"])] += 1
            unique_main.add(str(row["record_id"]))
            main_utf8_bytes += len(
                (str(row["prompt"]) + str(row["response"])).encode("utf-8")
            )
        for row in selected_anchor:
            sampled_anchor[str(row["capability"])] += 1
            unique_anchor.add(str(row["record_id"]))
            anchor_utf8_bytes += len(
                (str(row["prompt"]) + str(row["response"])).encode("utf-8")
            )
        if successful_steps == 1 or successful_steps % 100 == 0:
            point = {
                "step": successful_steps,
                "total_loss": loss_sum / gradient_accumulation_steps,
                "main_loss": main_loss_sum / gradient_accumulation_steps,
                "anchor_loss": anchor_loss_sum / gradient_accumulation_steps,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(point)
            print(json.dumps(point), flush=True)

    model.eval()
    merge_source_lora(model)
    if any("parametrizations" in name for name, _ in model.named_parameters()):
        raise SourceAlignmentError("temporary source adapter survived merge")
    output_path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        output_path, safe_serialization=True, max_shard_size="4GB"
    )
    tokenizer.save_pretrained(output_path)
    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    checkpoint_out = output_path / "model.safetensors"
    if not checkpoint_out.is_file():
        raise SourceAlignmentError("aligned source checkpoint is absent")
    if _sha256_file(source_checkpoint) != source_checkpoint_sha_before:
        raise SourceAlignmentError("open-weight source changed during alignment")
    if _sha256_file(main_bundle_path) != main_sha_before:
        raise SourceAlignmentError("main alignment bundle changed")
    if _sha256_file(anchor_bundle_path) != anchor_sha_before:
        raise SourceAlignmentError("anchor alignment bundle changed")

    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "ALIGNED_SOURCE_NOT_DEPLOYABLE_NOT_YET_LAYERCAKE_CERTIFIED",
        "source": {
            "path_at_alignment": str(source_path),
            "checkpoint_sha256": source_checkpoint_sha_before,
            "checkpoint_bytes": source_checkpoint.stat().st_size,
            "config_sha256": _sha256_file(source_config),
            "tokenizer_sha256": _sha256_file(source_tokenizer_file),
            "unchanged_on_disk": True,
        },
        "aligned_source": {
            "checkpoint": "model.safetensors",
            "checkpoint_sha256": _sha256_file(checkpoint_out),
            "checkpoint_bytes": checkpoint_out.stat().st_size,
            "config_sha256": _sha256_file(output_path / "config.json"),
            "tokenizer_sha256": _sha256_file(output_path / "tokenizer.json"),
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "precision": "float16",
            "deployable_in_layercake": False,
            "required_at_layercake_inference": False,
        },
        "temporary_adapter": {
            "method": "merged_low_rank_matrix_alignment",
            "rank": rank,
            "alpha": alpha,
            "target_matrix_names": target_names,
            "target_matrix_count": len(target_names),
            "trainable_parameter_count": int(
                sum(parameter.numel() for parameter in lora_parameters)
            ),
            "adapter_parameters_retained_separately": 0,
            "merged_into_source_matrix_shapes": True,
        },
        "main_artifact": {
            "path_at_alignment": str(main_bundle_path),
            "archive_sha256": main_sha_before,
            "manifest_sha256": main_bundle["verification"]["manifest_sha256"],
            "budget_id": main_budget["budget_id"],
            "budget_index": main_budget_index,
            "selected_records": len(main_rows),
            "unique_records_seen": len(unique_main),
            "sampled_records_by_capability": dict(sorted(sampled_main.items())),
        },
        "anchor_artifact": {
            "path_at_alignment": str(anchor_bundle_path),
            "archive_sha256": anchor_sha_before,
            "manifest_sha256": anchor_bundle["verification"]["manifest_sha256"],
            "budget_id": anchor_budget["budget_id"],
            "budget_index": anchor_budget_index,
            "selected_records": len(anchor_rows),
            "unique_records_seen": len(unique_anchor),
            "sampled_records_by_capability": dict(sorted(sampled_anchor.items())),
        },
        "training": {
            "device": "cuda",
            "seed": seed,
            "successful_optimizer_steps": successful_steps,
            "attempted_optimizer_steps": attempted_steps,
            "skipped_amp_optimizer_steps": skipped_steps,
            "microbatch_size": batch_size,
            "anchor_microbatch_size": anchor_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_tokens": max_tokens,
            "learning_rate": learning_rate,
            "supervised_response_tokens_seen": supervised_tokens,
            "anchor_supervised_response_tokens_seen": anchor_supervised_tokens,
            "main_utf8_bytes_seen": main_utf8_bytes,
            "anchor_utf8_bytes_seen": anchor_utf8_bytes,
            "wall_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "cpu_seconds": (
                cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
            ),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": int(process.memory_info().rss),
            "curves": curves,
        },
        "imported_information_accounting": {
            "teacher_generated_output_bytes_stored_in_bundles": sum(
                len(str(row["response"]).encode("utf-8"))
                for row in main_rows + anchor_rows
            ),
            "teacher_tokens_stored_in_bundles": sum(
                int(row["teacher_tokens"]) for row in main_rows + anchor_rows
            ),
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "foreign_source_parameters_copied_into_layercake": 0,
            "foreign_source_blocks_retained_in_layercake": 0,
        },
        "claim_boundary": (
            "This is a GPU-prepared, same-tokenizer acquisition source derived from "
            "segregated English-form text. It is not a deployable LayerCake, a domain "
            "package, foreign-teacher losslessness, or proof of final English quality."
        ),
    }
    payload = dict(manifest)
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    (output_path / "alignment_metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    del model
    torch.cuda.empty_cache()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--main-bundle", required=True)
    parser.add_argument("--anchor-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--main-budget-index", type=int, default=-1)
    parser.add_argument("--anchor-budget-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--anchor-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    result = align_same_tokenizer_source(
        source_path=args.source,
        main_bundle_path=args.main_bundle,
        anchor_bundle_path=args.anchor_bundle,
        output_path=args.output,
        main_budget_index=args.main_budget_index,
        anchor_budget_index=args.anchor_budget_index,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        anchor_batch_size=args.anchor_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_tokens=args.max_tokens,
        learning_rate=args.learning_rate,
        rank=args.rank,
        alpha=args.alpha,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "checkpoint_sha256": result["aligned_source"]["checkpoint_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
