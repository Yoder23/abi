"""Train the preregistered R58 controller without updating the R55 host."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import psutil
import torch
from safetensors.torch import save_file

from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core
from abi.layercake_full_core_acquisition import _DeterministicRowSampler
from abi.layercake_host import _batch, _sha256_file
from abi.layercake_host_v3 import load_english_training_rows
from experiments.foreign_capability_r14.core import write_json_once
from experiments.sparse_termination_controller_r58.controller import (
    RANK,
    ROUTES,
    WIDTH,
    SparseTerminationController,
    balanced_terminal_bce,
    complete_terminal_rows,
)


HOST_SHA256 = "4f302650a86964042996db1990e753388c84af7326253088f2a4bbeb698f1be6"
HOST_METADATA_SHA256 = "262fa8994dad29b8ded7039c5ce52a6e32af5ec91e2581b1bce220f06682b4ae"
BROAD_SHA256 = "82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150"
ANCHOR_SHA256 = "f6a27cae529d990ba67f1a26bb9cd79f97ba5ca0965c9debd0fc98dab8dba820"
SEED = 58_001
STEPS = 1_200
MAIN_BATCH = 16
ANCHOR_BATCH = 4
MAX_TOKENS = 256


class R58Error(RuntimeError):
    pass


class EpochSampler:
    def __init__(self, rows: list[dict], *, seed: int) -> None:
        self.rows = rows
        self.rng = random.Random(seed)
        self.order = list(range(len(rows)))
        self.cursor = len(self.order)

    def batch(self, size: int) -> list[dict]:
        selected = []
        for _ in range(size):
            if self.cursor >= len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            selected.append(self.rows[self.order[self.cursor]])
            self.cursor += 1
        return selected


def _prepare(tokenizer, path: Path) -> tuple[list[dict], dict, dict]:
    rows, budget, bundle = load_english_training_rows(path, budget_index=-1)
    route = {capability: index for index, capability in enumerate(CAPABILITY_CAKE_ORDER)}
    for row in rows:
        row["route"] = route[str(row["capability"])]
    rows = complete_terminal_rows(tokenizer, rows, max_tokens=MAX_TOKENS)
    return rows, budget, bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--broad-bundle", type=Path, required=True)
    parser.add_argument("--anchor-bundle", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    host = args.host.resolve()
    broad = args.broad_bundle.resolve()
    anchor = args.anchor_bundle.resolve()
    output = args.output.resolve()
    if output.exists():
        raise R58Error(f"immutable R58 output exists: {output}")
    frozen = (
        (host / "model.safetensors", HOST_SHA256),
        (host / "metadata.json", HOST_METADATA_SHA256),
        (broad, BROAD_SHA256),
        (anchor, ANCHOR_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or _sha256_file(path) != digest:
            raise R58Error(f"R58 frozen input changed: {path}")
    if not torch.cuda.is_available():
        raise R58Error("R58 training requires CUDA")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(
        host,
        layercake_root=args.layercake_root.resolve(),
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    main_rows, broad_budget, broad_artifact = _prepare(tokenizer, broad)
    anchor_rows, anchor_budget, anchor_artifact = _prepare(tokenizer, anchor)
    if len(main_rows) != 16_134 or len(anchor_rows) != 1_438:
        raise R58Error("R58 terminal-observable row inventory changed")

    controller = SparseTerminationController(seed=SEED).to(device)
    optimizer = torch.optim.AdamW(
        controller.parameters(), lr=1.0e-3, weight_decay=0.01
    )
    main_sampler = EpochSampler(main_rows, seed=SEED)
    anchor_sampler = _DeterministicRowSampler(
        anchor_rows,
        seed=SEED + 1,
        strategy="balanced_capabilities",
    )
    seen_main: set[str] = set()
    seen_anchor: set[str] = set()
    sampled_main: Counter[str] = Counter()
    sampled_anchor: Counter[str] = Counter()
    curves = []
    process = psutil.Process()
    rss_before = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cpu_started = time.process_time()
    for step in range(1, STEPS + 1):
        selected_main = main_sampler.batch(MAIN_BATCH)
        selected_anchor = anchor_sampler.batch(ANCHOR_BATCH)
        selected = selected_main + selected_anchor
        for row in selected_main:
            seen_main.add(str(row["record_id"]))
            sampled_main[str(row["capability"])] += 1
        for row in selected_anchor:
            seen_anchor.add(str(row["record_id"]))
            sampled_anchor[str(row["capability"])] += 1
        ids, labels, attention, prompt_lengths, routes, _ = _batch(
            tokenizer,
            selected,
            device=device,
            max_tokens=MAX_TOKENS,
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            result = model(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                task_routes=routes,
                use_cache=False,
            )
            hidden = result["hidden"][:, :-1].float().detach()
        logits = controller(hidden, routes)
        loss = balanced_terminal_bce(logits, labels, tokenizer.eos_token_id)
        if not torch.isfinite(loss):
            raise R58Error(f"non-finite controller loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0).item()
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == STEPS:
            with torch.no_grad():
                shifted = labels[:, 1:]
                active = shifted >= 0
                terminal = active & shifted.eq(tokenizer.eos_token_id)
                content = active & ~terminal
                terminal_accuracy = float((logits[terminal] >= 0).float().mean())
                content_accuracy = float((logits[content] < 0).float().mean())
            curve = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "terminal_accuracy": terminal_accuracy,
                "content_accuracy": content_accuracy,
                "gradient_norm": gradient_norm,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve, sort_keys=True), flush=True)

    if len(seen_main) != len(main_rows) or len(seen_anchor) != len(anchor_rows):
        raise R58Error("R58 did not observe every registered training row")
    if any(_sha256_file(path) != digest for path, digest in frozen):
        raise R58Error("R58 changed a frozen input")
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in controller.state_dict().items()
    }
    output.mkdir(parents=True)
    controller_path = output / "termination_controller.safetensors"
    save_file(state, str(controller_path))
    controller_sha = _sha256_file(controller_path)
    elapsed = time.perf_counter() - started
    metadata = {
        "format": "abi-r58-sparse-termination-controller/1",
        "status": "TRAINED_NOT_YET_CERTIFIED",
        "controller": {
            "path": controller_path.name,
            "sha256": controller_sha,
            "routes": ROUTES,
            "width": WIDTH,
            "rank": RANK,
            "installed_parameters": sum(value.numel() for value in state.values()),
            "active_parameters_per_sequence": RANK * WIDTH + RANK + RANK + 1,
            "maximum_active_routes_per_decision": 1,
            "threshold_logit": 0.0,
            "can_modify_token_logits": False,
            "can_select_or_rewrite_tokens": False,
        },
        "frozen_host": {
            "checkpoint_sha256": HOST_SHA256,
            "metadata_sha256": HOST_METADATA_SHA256,
            "unchanged_after_training": True,
            "teacher_present_at_inference": False,
        },
        "training": {
            "seed": SEED,
            "device": "cuda",
            "steps": STEPS,
            "successful_optimizer_steps": STEPS,
            "main_batch_size": MAIN_BATCH,
            "anchor_batch_size": ANCHOR_BATCH,
            "max_tokens": MAX_TOKENS,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.01,
            "gradient_norm_ceiling": 1.0,
            "loss": "equal_record_half_terminal_half_content_bce",
            "main_sampling": "deterministic_shuffled_epoch",
            "anchor_sampling": "balanced_capabilities",
            "main_rows": len(main_rows),
            "anchor_rows": len(anchor_rows),
            "unique_main_rows_seen": len(seen_main),
            "unique_anchor_rows_seen": len(seen_anchor),
            "sampled_main_by_capability": dict(sorted(sampled_main.items())),
            "sampled_anchor_by_capability": dict(sorted(sampled_anchor.items())),
            "wall_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "cpu_seconds": time.process_time() - cpu_started,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": process.memory_info().rss,
            "peak_device_memory_bytes": torch.cuda.max_memory_allocated(device),
            "curves": curves,
        },
        "inputs": {
            "broad_archive_sha256": BROAD_SHA256,
            "broad_budget_id": broad_budget["budget_id"],
            "broad_manifest_sha256": broad_artifact["manifest_sha256"],
            "anchor_archive_sha256": ANCHOR_SHA256,
            "anchor_budget_id": anchor_budget["budget_id"],
            "anchor_manifest_sha256": anchor_artifact["manifest_sha256"],
            "validation_or_final_outputs_used": 0,
            "teacher_calls": 0,
            "teacher_logits_stored": 0,
            "teacher_activations_stored": 0,
            "source_parameters_copied": 0,
        },
        "claim_boundary": (
            "A frozen-host termination diagnostic only; not yet an English, "
            "prospective, minimality, or LoRA/distillation claim."
        ),
    }
    unsigned = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    metadata["manifest_sha256"] = hashlib.sha256(unsigned).hexdigest()
    write_json_once(output / "metadata.json", metadata)
    print(json.dumps({"controller_sha256": controller_sha, "status": metadata["status"]}, indent=2))


if __name__ == "__main__":
    main()
