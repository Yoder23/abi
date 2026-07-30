"""Acquire English into the existing LayerCake core without changing its graph.

This is a bounded capacity diagnostic after the small-delta v47 lineage failed
generalization.  The complete LayerCake core is trainable, but its architecture,
tokenizer, sparse task-cake topology, canonical ABI, and source artifact remain
fixed.  No foreign-teacher parameter is copied or retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .artifacts import module_state_sha256
from .layercake_host import (
    _autonomous_prefixes,
    _batch,
    _canonical_json_bytes,
    _equal_record_prompt_overlap_ce,
    _import_layercake_runtime,
    _is_within,
    _sha256_file,
)
from .layercake_host_v3 import load_english_training_rows
from .layercake_core_loader import load_layercake_core


ARTIFACT_FORMAT = "abi-layercake-full-english-core-acquisition/1"


class FullCoreAcquisitionError(RuntimeError):
    """Raised when the bounded full-core acquisition contract is violated."""


class _DeterministicRowSampler:
    """Draw records uniformly or give every declared capability equal weight."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        strategy: str,
    ) -> None:
        if strategy not in {"uniform_records", "balanced_capabilities"}:
            raise FullCoreAcquisitionError("unknown sampling strategy")
        if not rows:
            raise FullCoreAcquisitionError("cannot sample an empty corpus")
        self.rows = rows
        self.rng = random.Random(seed)
        self.strategy = strategy
        self.order = list(range(len(rows)))
        self.cursor = len(self.order)
        self.by_capability: dict[str, list[int]] = {}
        self.capability_cursors: dict[str, int] = {}
        for index, row in enumerate(rows):
            self.by_capability.setdefault(
                str(row["capability"]), []
            ).append(index)
        self.capability_order = sorted(self.by_capability)
        self.capability_cursor = len(self.capability_order)
        for capability, indices in self.by_capability.items():
            self.capability_cursors[capability] = len(indices)

    def _next_uniform(self) -> int:
        if self.cursor >= len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
        index = self.order[self.cursor]
        self.cursor += 1
        return index

    def _next_capability(self) -> str:
        if self.capability_cursor >= len(self.capability_order):
            self.rng.shuffle(self.capability_order)
            self.capability_cursor = 0
        capability = self.capability_order[self.capability_cursor]
        self.capability_cursor += 1
        return capability

    def _next_balanced(self) -> int:
        capability = self._next_capability()
        indices = self.by_capability[capability]
        cursor = self.capability_cursors[capability]
        if cursor >= len(indices):
            self.rng.shuffle(indices)
            cursor = 0
        index = indices[cursor]
        self.capability_cursors[capability] = cursor + 1
        return index

    def batch(self, size: int) -> list[Mapping[str, Any]]:
        if size <= 0:
            raise FullCoreAcquisitionError("sample size must be positive")
        selector = (
            self._next_balanced
            if self.strategy == "balanced_capabilities"
            else self._next_uniform
        )
        return [self.rows[selector()] for _ in range(size)]


def _manifest_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def train_full_core(
    *,
    bundle_path: str | Path,
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    output_path: str | Path,
    budget_index: int,
    seed: int,
    steps: int,
    batch_size: int = 8,
    shared_learning_rate: float = 2.0e-5,
    cake_learning_rate: float = 1.0e-4,
    classifier_loss_weight: float = 0.25,
    prompt_overlap_loss_weight: float = 1.0,
    max_tokens: int = 256,
    recovery_start_step: int = 400,
    recovery_interval: int = 8,
    recovery_horizons: Sequence[int] = (8, 16, 32),
    sampling_strategy: str = "uniform_records",
    device_name: str = "cuda",
) -> dict[str, Any]:
    if min(steps, batch_size, max_tokens) <= 0:
        raise FullCoreAcquisitionError("steps, batch size, and token limit must be positive")
    if min(shared_learning_rate, cake_learning_rate) <= 0:
        raise FullCoreAcquisitionError("learning rates must be positive")
    if classifier_loss_weight < 0 or prompt_overlap_loss_weight < 0:
        raise FullCoreAcquisitionError("loss weights must be non-negative")
    if recovery_start_step < 0 or recovery_interval < 0:
        raise FullCoreAcquisitionError("recovery schedule is invalid")
    if recovery_interval and (
        not recovery_horizons
        or any(int(horizon) <= 0 for horizon in recovery_horizons)
    ):
        raise FullCoreAcquisitionError("recovery horizons must be positive")
    if sampling_strategy not in {
        "uniform_records",
        "balanced_capabilities",
    }:
        raise FullCoreAcquisitionError("unknown sampling strategy")

    bundle_path = Path(bundle_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    parent_path = Path(parent_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    output_path = Path(output_path).resolve()
    parent_in_sealed_tree = _is_within(parent_path, layercake_root)
    abi_root = Path(__file__).resolve().parents[1]
    if not parent_in_sealed_tree and not _is_within(parent_path, abi_root):
        raise FullCoreAcquisitionError(
            "parent must belong to LayerCake or the ABI evidence tree"
        )
    if _is_within(output_path, layercake_root):
        raise FullCoreAcquisitionError("ABI acquisition may not modify the sealed LayerCake tree")
    if output_path.exists():
        raise FullCoreAcquisitionError(f"core artifact is immutable: {output_path}")
    if not canonical_abi_path.is_file():
        raise FullCoreAcquisitionError("canonical semantic ABI is missing")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise FullCoreAcquisitionError("CUDA was requested but is unavailable")

    archive_sha_before = _sha256_file(bundle_path)
    rows, budget, bundle = load_english_training_rows(
        bundle_path, budget_index=budget_index
    )
    verification = bundle["verification"]
    if (
        verification["domain_segregation_verified"] is not True
        or verification["training_eligible"] is not True
    ):
        raise FullCoreAcquisitionError("training bundle did not pass segregation")

    parent_metadata_path = parent_path / "metadata.json"
    parent_checkpoint_path = parent_path / "model.safetensors"
    parent_metadata = json.loads(
        parent_metadata_path.read_text(encoding="utf-8")
    )
    parent_checkpoint_sha = _sha256_file(parent_checkpoint_path)
    if parent_metadata["checkpoint"]["sha256"] != parent_checkpoint_sha:
        raise FullCoreAcquisitionError("sealed parent checkpoint hash changed")
    if not parent_in_sealed_tree and (
        parent_metadata.get("format")
        != "abi-layercake-six-block-capacity-base/1"
        or parent_metadata.get("canonical_semantic_abi", {}).get("sha256")
        != _sha256_file(canonical_abi_path)
    ):
        raise FullCoreAcquisitionError(
            "ABI-owned parent is not the hash-bound six-block base"
        )

    device = torch.device(device_name)
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    _import_layercake_runtime(layercake_root)
    model, tokenizer, _ = load_layercake_core(
        parent_path,
        layercake_root=layercake_root,
        device=device,
    )
    model.train()
    initial_state_sha = module_state_sha256(model)

    cake_parameters = list(model.task_classifier.parameters())
    for cake in model.task_cakes:
        cake_parameters.extend(cake.parameters())
    cake_ids = {id(parameter) for parameter in cake_parameters}
    shared_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in cake_ids
    ]
    if not shared_parameters or not cake_parameters:
        raise FullCoreAcquisitionError("LayerCake parameter groups are incomplete")
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": shared_parameters,
                "lr": shared_learning_rate,
            },
            {
                "params": cake_parameters,
                "lr": cake_learning_rate,
            },
        ],
        weight_decay=0.01,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.float16))
        if use_amp
        else (lambda: nullcontext())
    )

    sampler = _DeterministicRowSampler(
        rows,
        seed=seed,
        strategy=sampling_strategy,
    )
    unique_seen: set[str] = set()
    sampled_records_by_capability: Counter[str] = Counter()
    supervised_tokens_seen = 0
    raw_utf8_bytes_seen = 0
    autonomous_prefix_tokens_seen = 0
    recovery_batches = 0
    horizon_counts = {
        str(int(horizon)): 0 for horizon in recovery_horizons
    }
    successful_steps = 0
    skipped_amp_steps = 0
    attempted_batches = 0
    curves: list[dict[str, Any]] = []
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    cpu_before = process.cpu_times()
    started = time.perf_counter()

    while successful_steps < steps:
        attempted_batches += 1
        if attempted_batches > steps + 1000:
            raise FullCoreAcquisitionError("too many non-finite optimizer attempts")
        selected = sampler.batch(batch_size)
        sampled_records_by_capability.update(
            str(row["capability"]) for row in selected
        )
        generated_prefixes = None
        if (
            recovery_interval > 0
            and successful_steps >= recovery_start_step
            and (successful_steps - recovery_start_step) % recovery_interval == 0
        ):
            horizon = int(
                recovery_horizons[
                    recovery_batches % len(recovery_horizons)
                ]
            )
            generated_prefixes = _autonomous_prefixes(
                model,
                tokenizer,
                selected,
                horizon=horizon,
                device=device,
            )
            recovery_batches += 1
            horizon_counts[str(horizon)] += 1
            autonomous_prefix_tokens_seen += sum(
                len(prefix) for prefix in generated_prefixes
            )
        (
            ids,
            labels,
            attention,
            prompt_lengths,
            routes,
            observed,
        ) = _batch(
            tokenizer,
            selected,
            device=device,
            max_tokens=max_tokens,
            generated_prefixes=generated_prefixes,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            result = model(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                task_routes=routes,
                use_cache=False,
            )
            language_loss = _equal_record_prompt_overlap_ce(
                result["logits"],
                labels,
                ids,
                prompt_lengths,
                overlap_weight=prompt_overlap_loss_weight,
            )
            classifier_loss = F.cross_entropy(
                result["task_logits"], routes
            )
            loss = language_loss + classifier_loss_weight * classifier_loss
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped_amp_steps += 1
            continue
        successful_steps += 1
        unique_seen.update(str(row["record_id"]) for row in selected)
        supervised_tokens_seen += observed
        raw_utf8_bytes_seen += sum(
            len((str(row["prompt"]) + str(row["response"])).encode("utf-8"))
            for row in selected
        )
        if (
            successful_steps == 1
            or successful_steps % 100 == 0
            or successful_steps == steps
        ):
            curve = {
                "step": successful_steps,
                "total_loss": float(loss.detach()),
                "language_loss": float(language_loss.detach()),
                "classifier_loss": float(classifier_loss.detach()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)

    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    model.eval()
    final_state_sha = module_state_sha256(model)
    if final_state_sha == initial_state_sha:
        raise FullCoreAcquisitionError("full-core acquisition changed no parameters")
    if _sha256_file(bundle_path) != archive_sha_before:
        raise FullCoreAcquisitionError("imported ABI artifact changed during training")
    if _sha256_file(parent_checkpoint_path) != parent_checkpoint_sha:
        raise FullCoreAcquisitionError("sealed parent changed during training")

    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_path / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(checkpoint_path),
    )
    tokenizer.save_pretrained(output_path)
    checkpoint_sha = _sha256_file(checkpoint_path)
    tokenizer_path = output_path / "tokenizer.json"
    cpu_seconds = (
        cpu_after.user
        + cpu_after.system
        - cpu_before.user
        - cpu_before.system
    )
    manifest: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "status": "TRAINED_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED",
        "architecture": copy.deepcopy(parent_metadata["architecture"]),
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha,
            "bytes": checkpoint_path.stat().st_size,
        },
        "tokenizer": {
            "path": tokenizer_path.name,
            "sha256": _sha256_file(tokenizer_path),
        },
        "parent_layercake": {
            "path_at_training": str(parent_path),
            "checkpoint_sha256": parent_checkpoint_sha,
            "metadata_sha256": _sha256_file(parent_metadata_path),
            "logical_state_sha256_before": initial_state_sha,
            "unchanged_on_disk": True,
        },
        "acquired_core": {
            "logical_state_sha256_after": final_state_sha,
            "total_parameter_count": trainable_parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "active_parameter_count": int(
                parent_metadata["parameters"]["active"]
            ),
            "graph_topology_changed": False,
            "task_cake_count": int(
                parent_metadata["architecture"]["task_cakes"]
            ),
            "maximum_active_task_cakes_per_sequence": 1,
            "physical_sparse_topology_preserved": True,
        },
        "canonical_semantic_abi": {
            "path_at_training": str(canonical_abi_path),
            "sha256": _sha256_file(canonical_abi_path),
            "changed": False,
        },
        "imported_artifact": {
            "path_at_training": str(bundle_path),
            "archive_sha256_before": archive_sha_before,
            "archive_sha256_after": _sha256_file(bundle_path),
            "manifest_sha256": verification["manifest_sha256"],
            "artifact_role": verification["artifact_role"],
            "domain_segregation_verified": True,
            "budget_id": budget["budget_id"],
            "budget_index": budget_index,
            "selected_english_records": len(rows),
            "unique_selected_records_seen": len(unique_seen),
            "all_selected_records_seen": len(unique_seen) == len(rows),
            "selected_teacher_tokens": sum(
                int(row["teacher_tokens"]) for row in rows
            ),
            "selected_teacher_output_bytes": sum(
                len(str(row["response"]).encode("utf-8")) for row in rows
            ),
            "teacher_logits_stored": 0,
            "teacher_hidden_activation_bytes_stored": 0,
        },
        "foreign_source_boundary": {
            "teacher_present_at_inference": False,
            "source_transformer_blocks_retained": 0,
            "source_parameters_copied": 0,
            "source_generated_text_retained_in_deployment": False,
            "teacher_tokenizer_required_at_inference": False,
        },
        "training": {
            "seed": seed,
            "device": str(device),
            "steps": steps,
            "successful_optimizer_steps": successful_steps,
            "skipped_amp_optimizer_steps": skipped_amp_steps,
            "attempted_batches": attempted_batches,
            "batch_size": batch_size,
            "sampling_strategy": sampling_strategy,
            "sampled_records_by_capability": dict(
                sorted(sampled_records_by_capability.items())
            ),
            "shared_learning_rate": shared_learning_rate,
            "cake_learning_rate": cake_learning_rate,
            "classifier_loss_weight": classifier_loss_weight,
            "prompt_overlap_loss_weight": prompt_overlap_loss_weight,
            "weight_decay": 0.01,
            "max_tokens": max_tokens,
            "supervised_layercake_tokens_seen": supervised_tokens_seen,
            "raw_utf8_bytes_seen": raw_utf8_bytes_seen,
            "self_generated_prefix_recovery": {
                "start_step": recovery_start_step,
                "interval": recovery_interval,
                "horizons": [int(value) for value in recovery_horizons],
                "batches": recovery_batches,
                "horizon_batches": horizon_counts,
                "autonomous_prefix_tokens_seen": autonomous_prefix_tokens_seen,
            },
            "wall_seconds": elapsed,
            "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0,
            "cpu_seconds": cpu_seconds,
            "cpu_core_hours": cpu_seconds / 3600,
            "active_parameter_seconds": trainable_parameter_count * elapsed,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": int(process.memory_info().rss),
            "peak_device_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "curves": curves,
        },
        "claim_boundary": (
            "This capacity diagnostic preserves the existing LayerCake graph "
            "but trains the complete core because the sealed parent and small-"
            "delta lineage both failed autonomous English. It is not a small-"
            "bridge Phase-3 pass, a fluent-core claim, or runtime certification."
        ),
    }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    _write_json(output_path / "metadata.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shared-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--cake-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--classifier-loss-weight", type=float, default=0.25)
    parser.add_argument("--prompt-overlap-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--recovery-start-step", type=int, default=400)
    parser.add_argument("--recovery-interval", type=int, default=8)
    parser.add_argument("--recovery-horizons", default="8,16,32")
    parser.add_argument(
        "--sampling-strategy",
        choices=("uniform_records", "balanced_capabilities"),
        default="uniform_records",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    manifest = train_full_core(
        bundle_path=args.bundle,
        layercake_root=args.layercake_root,
        parent_path=args.parent,
        canonical_abi_path=args.canonical_abi,
        output_path=args.output,
        budget_index=args.budget_index,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        shared_learning_rate=args.shared_learning_rate,
        cake_learning_rate=args.cake_learning_rate,
        classifier_loss_weight=args.classifier_loss_weight,
        prompt_overlap_loss_weight=args.prompt_overlap_loss_weight,
        max_tokens=args.max_tokens,
        recovery_start_step=args.recovery_start_step,
        recovery_interval=args.recovery_interval,
        recovery_horizons=tuple(
            int(value)
            for value in args.recovery_horizons.split(",")
            if value.strip()
        ),
        sampling_strategy=args.sampling_strategy,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
