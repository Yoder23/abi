"""Train only source-side capability prefixes for R8."""

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


class SourceTrainingError(RuntimeError):
    """Raised when source training violates an R8 freeze or evidence rule."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceTrainingError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise SourceTrainingError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _capability_rows(
    capabilities: Sequence[OpaqueCapability], config: Mapping[str, Any]
) -> list[list[dict[str, Any]]]:
    count = int(config["splits"]["source_train_rows_per_capability"])
    base_seed = int(config["training"]["seed"])
    return [
        generate_rows(
            capability,
            split="source_train",
            rows=count,
            depths=config["capability_family"]["source_train_depths"],
            seed=base_seed + 1009 * index,
        )
        for index, capability in enumerate(capabilities)
    ]


def _batch(
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    generator: random.Random,
    batch_size: int,
) -> tuple[list[int], list[Mapping[str, Any]]]:
    indices = [generator.randrange(len(rows)) for _ in range(batch_size)]
    selected = [rows[index][generator.randrange(len(rows[index]))] for index in indices]
    return indices, selected


def _accuracy(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    prefix: torch.Tensor | None,
    *,
    batch_size: int,
) -> dict[str, Any]:
    correct = 0
    nll = 0.0
    predictions: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=prefix)
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
        "prediction_ids_sha256": hashlib.sha256(canonical_json_bytes(predictions)).hexdigest(),
    }


def fit_prefixes(
    host: FrozenNeuralHost,
    capabilities: Sequence[OpaqueCapability],
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    prefix_length: int,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = random.Random(seed)
    prefixes = torch.nn.Parameter(
        torch.zeros(
            len(capabilities),
            prefix_length,
            host.hidden_width,
            dtype=torch.float32,
            device=host.device,
        )
    )
    torch.nn.init.normal_(prefixes, mean=0.0, std=0.02)
    optimizer = torch.optim.AdamW([prefixes], lr=learning_rate, weight_decay=0.0)
    curves = []
    started = time.perf_counter()
    peak_cuda = 0
    for step in range(1, steps + 1):
        capability_indices, selected = _batch(rows, generator=generator, batch_size=batch_size)
        index_tensor = torch.tensor(capability_indices, dtype=torch.long, device=host.device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = host.logits(
            [str(row["prompt"]) for row in selected],
            prefix=prefixes.index_select(0, index_tensor),
        )
        targets = host.target_ids([int(row["answer"]) for row in selected])
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise SourceTrainingError("source prefix loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([prefixes], 1.0)
        optimizer.step()
        if host.device.type == "cuda":
            peak_cuda = max(peak_cuda, int(torch.cuda.max_memory_allocated()))
        if step == 1 or step % max(1, steps // 20) == 0 or step == steps:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(record)
            print(json.dumps(record), flush=True)
    host.verify_frozen()
    return prefixes.detach().cpu(), {
        "optimizer_steps": steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": peak_cuda,
        "curves": curves,
    }


def train_public(config_path: Path, output: Path) -> dict[str, Any]:
    config = _json(config_path)
    if output.exists():
        raise SourceTrainingError(f"immutable public source output exists: {output}")
    split = config["splits"]
    meta_capabilities = public_capabilities(
        int(split["meta_seed"]),
        split="meta_train",
        count=int(split["meta_train_capabilities"]),
    )
    development_capabilities = public_capabilities(
        int(split["development_seed"]),
        split="development",
        count=int(split["development_capabilities"]),
    )
    capabilities = [*meta_capabilities, *development_capabilities]
    rows = _capability_rows(capabilities, config)
    host = FrozenNeuralHost(SPECS["source"], device=config["training"]["device"])
    meta_steps = int(config["training"].get("source_meta_steps", 3000))
    prefixes, training = fit_prefixes(
        host,
        capabilities,
        rows,
        prefix_length=int(config["training"]["source_prefix_length"]),
        steps=meta_steps,
        learning_rate=float(config["training"]["source_learning_rate"]),
        batch_size=int(config["training"]["batch_size"]),
        seed=int(config["training"]["seed"]),
    )
    output.mkdir(parents=True)
    tensor_path = output / "meta_source_prefixes.safetensors"
    save_file({"prefixes": prefixes.contiguous()}, str(tensor_path))
    assessments = []
    zero = torch.zeros_like(prefixes[0]).to(host.device)
    for index, capability in enumerate(capabilities):
        sampled = rows[index][: min(256, len(rows[index]))]
        assessments.append(
            {
                "capability_id": capability.capability_id,
                "before": _accuracy(
                    host,
                    sampled,
                    None,
                    batch_size=int(config["training"]["batch_size"]),
                ),
                "zero_prefix": _accuracy(
                    host,
                    sampled,
                    zero,
                    batch_size=int(config["training"]["batch_size"]),
                ),
                "after": _accuracy(
                    host,
                    sampled,
                    prefixes[index].to(host.device),
                    batch_size=int(config["training"]["batch_size"]),
                ),
            }
        )
    receipt = {
        "format": "abi-native-transfer-r8-source-meta-training/1",
        "config_sha256": sha256_file(config_path),
        "split": "public_pre_reveal",
        "heldout_reveal_present": False,
        "capabilities": {
            "meta_train": len(meta_capabilities),
            "development": len(development_capabilities),
            "total": len(capabilities),
        },
        "source": {
            "model_id": host.spec.model_id,
            "revision": host.spec.revision,
            "architecture_family": host.spec.architecture_family,
            "snapshot_inventory": host.inventory,
            "model_state_sha256": host.model_state_sha256,
            "parameters_frozen": True,
        },
        "prefixes": {
            "path": tensor_path.name,
            "sha256": sha256_file(tensor_path),
            "shape": list(prefixes.shape),
        },
        "training": training,
        "assessments": assessments,
        "teacher_present_at_recipient_inference": False,
        "heldout_accessed": False,
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    _write_once(
        output / "receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    del host
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = train_public(Path(args.config).resolve(), Path(args.output).resolve())
    except (SourceTrainingError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
