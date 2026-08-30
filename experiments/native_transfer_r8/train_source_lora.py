"""Train public R8 source capabilities as isolated LoRA adapter states."""

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
from .source_adapter import (
    SourceAdapterError,
    SourceLoRASet,
    ensure_only_adapters_trainable,
    pack_capability_states,
)


class SourceLoRATrainingError(RuntimeError):
    """Raised when the source adapter acquisition control is invalid."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceLoRATrainingError(f"expected JSON object: {path}")
    return value


@torch.inference_mode()
def _accuracy(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    correct = 0
    nll = 0.0
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
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


def fit_one(
    host: FrozenNeuralHost,
    adapters: SourceLoRASet,
    rows: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    adapters.reset(seed)
    ensure_only_adapters_trainable(host.model, adapters)
    parameters = adapters.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = random.Random(seed)
    started = time.perf_counter()
    first_loss = None
    final_loss = None
    for _ in range(steps):
        batch = [rows[generator.randrange(len(rows))] for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        targets = host.target_ids([int(row["answer"]) for row in batch])
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise SourceLoRATrainingError("source LoRA loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        final_loss = value
    adapters.verify_base_frozen()
    return {
        "optimizer_steps": steps,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "wall_seconds": time.perf_counter() - started,
    }


def train_public(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise SourceLoRATrainingError(f"immutable source LoRA output exists: {output}")
    config = _json(config_path)
    if config["training"].get("source_capability_method") != "lora_all_gpt2_conv1d":
        raise SourceLoRATrainingError("active config does not register source LoRA")
    split = config["splits"]
    meta = public_capabilities(
        int(split["meta_seed"]),
        split="meta_train",
        count=int(split["meta_train_capabilities"]),
    )
    development = public_capabilities(
        int(split["development_seed"]),
        split="development",
        count=int(split["development_capabilities"]),
    )
    capabilities: list[OpaqueCapability] = [*meta, *development]
    train_rows = [
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
    before = [
        _accuracy(
            host,
            rows,
            batch_size=int(config["training"]["batch_size"]),
        )
        for rows in evaluation_rows
    ]
    adapters = SourceLoRASet(
        host.model,
        rank=int(config["training"]["source_lora_rank"]),
        expected_base_sha256=host.model_state_sha256,
    )
    states = []
    assessments = []
    started = time.perf_counter()
    for index, capability in enumerate(capabilities):
        training = fit_one(
            host,
            adapters,
            train_rows[index],
            steps=int(config["training"]["source_steps_per_capability"]),
            learning_rate=float(config["training"]["source_lora_learning_rate"]),
            batch_size=int(config["training"]["batch_size"]),
            seed=int(config["training"]["seed"]) + 12007 * index,
        )
        state = adapters.state()
        states.append(state)
        after = _accuracy(
            host,
            evaluation_rows[index],
            batch_size=int(config["training"]["batch_size"]),
        )
        row = {
            "capability_index": index,
            "capability_id": capability.capability_id,
            "before": before[index],
            "after": after,
            "training": training,
            "adapter_state_sha256": adapters.state_sha256(),
        }
        assessments.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    adapters.verify_base_frozen()
    output.mkdir(parents=True)
    tensor_path = output / "meta_source_lora_adapters.safetensors"
    save_file(pack_capability_states(states), str(tensor_path))
    receipt = {
        "format": "abi-native-transfer-r8-source-meta-lora-training/1",
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
            "base_model_state_sha256_after": adapters.base_state_sha256(),
            "base_parameters_trainable": 0,
        },
        "adapter": adapters.inventory(),
        "adapters": {
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
    except (SourceLoRATrainingError, SourceAdapterError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
