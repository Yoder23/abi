"""Train public R8 neural transition source states from examples only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .capability_generator import (
    OpaqueCapability,
    canonical_json_bytes,
    generate_rows,
    public_capabilities,
)
from .native_host import SPECS, FrozenNeuralHost, NativeHostError, sha256_file
from .source_transition import (
    NeuralTransitionSource,
    SourceTransitionError,
    controller_state,
    ensure_source_base_frozen,
    pack_controller_states,
)


class TransitionTrainingError(RuntimeError):
    """Raised when source transition acquisition violates its registration."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TransitionTrainingError(f"expected JSON object: {path}")
    return value


@torch.inference_mode()
def _base_cache(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    digit_chunks = []
    other_chunks = []
    target_ids = torch.tensor(host.target_token_ids, dtype=torch.long, device=host.device)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        digits = logits.index_select(-1, target_ids)
        masked = logits.clone()
        masked.index_fill_(-1, target_ids, float("-inf"))
        digit_chunks.append(digits)
        other_chunks.append(torch.logsumexp(masked, dim=-1))
    return torch.cat(digit_chunks), torch.cat(other_chunks)


def _structured(rows: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> tuple[list[int], list[list[int]]]:
    return (
        [int(rows[index]["start"]) for index in indices],
        [[int(value) for value in rows[index]["program"]] for index in indices],
    )


def _loss_from_cache(
    controller: NeuralTransitionSource,
    digit_logits: torch.Tensor,
    other_logsumexp: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> torch.Tensor:
    starts, programs = _structured(rows, indices)
    index_tensor = torch.tensor(indices, dtype=torch.long, device=digit_logits.device)
    selected_digits = digit_logits.index_select(0, index_tensor)
    selected_other = other_logsumexp.index_select(0, index_tensor)
    scores = selected_digits + controller.canonical_addition(starts, programs)
    answers = torch.tensor(
        [int(rows[index]["answer"]) for index in indices],
        dtype=torch.long,
        device=digit_logits.device,
    )
    correct = scores.gather(1, answers.unsqueeze(1)).squeeze(1)
    denominator = torch.logaddexp(selected_other, torch.logsumexp(scores, dim=-1))
    return (denominator - correct).mean()


def fit_one(
    controller: NeuralTransitionSource,
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    controller.reset()
    controller.to(host.device)
    cache_started = time.perf_counter()
    digit_logits, other_logsumexp = _base_cache(host, rows, batch_size=batch_size)
    cache_seconds = time.perf_counter() - cache_started
    optimizer = torch.optim.AdamW(controller.parameters(), lr=learning_rate, weight_decay=0.0)
    generator = random.Random(seed)
    first_loss = None
    final_loss = None
    started = time.perf_counter()
    for _ in range(steps):
        indices = [generator.randrange(len(rows)) for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_from_cache(
            controller, digit_logits, other_logsumexp, rows, indices
        )
        if not torch.isfinite(loss):
            raise TransitionTrainingError("source transition loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 10.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        final_loss = value
    ensure_source_base_frozen(host, host.model_state_sha256)
    return {
        "optimizer_steps": steps,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "base_logit_cache_seconds": cache_seconds,
        "optimization_wall_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def _accuracy(
    controller: NeuralTransitionSource,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    correct = 0
    nll = 0.0
    predictions = []
    host = controller.host
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        logits = controller.logits(
            [str(row["prompt"]) for row in batch],
            [int(row["start"]) for row in batch],
            [[int(value) for value in row["program"]] for row in batch],
        )
        targets = host.target_ids([int(row["answer"]) for row in batch])
        predicted = logits.argmax(dim=-1)
        correct += int((predicted == targets).sum().item())
        nll += float(F.cross_entropy(logits, targets, reduction="sum").item())
        predictions.extend(int(value) for value in predicted.cpu().tolist())
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_nll": nll / len(rows),
        "prediction_ids_sha256": hashlib.sha256(
            canonical_json_bytes(predictions)
        ).hexdigest(),
    }


def train_public(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise TransitionTrainingError(f"immutable transition output exists: {output}")
    config = _json(config_path)
    if config["training"].get("source_capability_method") != "neural_transition_adapter":
        raise TransitionTrainingError("active config does not register transition source")
    split = config["splits"]
    meta = public_capabilities(
        int(split["meta_seed"]), split="meta_train", count=int(split["meta_train_capabilities"])
    )
    development = public_capabilities(
        int(split["development_seed"]),
        split="development",
        count=int(split["development_capabilities"]),
    )
    capabilities: list[OpaqueCapability] = [*meta, *development]
    training_rows = [
        generate_rows(
            capability,
            split="source_train",
            rows=int(split["source_train_rows_per_capability"]),
            depths=config["capability_family"]["source_train_depths"],
            seed=int(config["training"]["seed"]) + 1009 * index,
        )
        for index, capability in enumerate(capabilities)
    ]
    evaluation_rows = [
        generate_rows(
            capability,
            split="public_source_evaluation",
            rows=256,
            depths=config["capability_family"]["evaluation_depths"],
            seed=int(config["training"]["seed"]) + 9001 * index,
        )
        for index, capability in enumerate(capabilities)
    ]
    host = FrozenNeuralHost(SPECS["source"], device=config["training"]["device"])
    controller = NeuralTransitionSource(host, seed=int(config["training"]["seed"])).to(
        host.device
    )
    before = [
        _accuracy(controller, rows, batch_size=int(config["training"]["batch_size"]))
        for rows in evaluation_rows
    ]
    states = []
    assessments = []
    started = time.perf_counter()
    for index, capability in enumerate(capabilities):
        training = fit_one(
            controller,
            host,
            training_rows[index],
            steps=int(config["training"]["source_steps_per_capability"]),
            learning_rate=float(config["training"]["source_transition_learning_rate"]),
            batch_size=int(config["training"]["batch_size"]),
            seed=int(config["training"]["seed"]) + 12007 * index,
        )
        after = _accuracy(
            controller,
            evaluation_rows[index],
            batch_size=int(config["training"]["batch_size"]),
        )
        states.append(controller_state(controller))
        row = {
            "capability_index": index,
            "capability_id": capability.capability_id,
            "before": before[index],
            "after": after,
            "training": training,
            "controller_state_sha256": controller.state_sha256(),
        }
        assessments.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    ensure_source_base_frozen(host, host.model_state_sha256)
    output.mkdir(parents=True)
    tensor_path = output / "meta_source_transition_states.safetensors"
    save_file(pack_controller_states(states), str(tensor_path))
    receipt = {
        "format": "abi-native-transfer-r8-source-meta-transition-training/1",
        "config_sha256": sha256_file(config_path),
        "split": "public_pre_reveal",
        "capabilities": {
            "meta_train": len(meta),
            "development": len(development),
            "total": len(capabilities),
        },
        "source": {
            "model_id": host.spec.model_id,
            "revision": host.spec.revision,
            "architecture_family": host.spec.architecture_family,
            "snapshot_inventory": host.inventory,
            "base_model_state_sha256_before": host.model_state_sha256,
            "base_model_state_sha256_after": host.model_state_sha256,
            "base_parameters_trainable": 0,
        },
        "controller": controller.schema(),
        "states": {
            "path": tensor_path.name,
            "sha256": sha256_file(tensor_path),
            "capability_states": len(states),
        },
        "assessments": assessments,
        "wall_seconds": time.perf_counter() - started,
        "heldout_accessed": False,
        "heldout_reveal_present": False,
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    (output / "receipt.json").write_bytes(
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = train_public(Path(args.config).resolve(), Path(args.output).resolve())
    except (TransitionTrainingError, SourceTransitionError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
