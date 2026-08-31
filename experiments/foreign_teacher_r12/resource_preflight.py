"""One-step CUDA coexistence diagnostic; this is not an R12 scientific gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from experiments.native_transfer_r8.capability_generator import (
    generate_rows,
    public_capabilities,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
)

from .public_preflight import _json


def _memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "process_allocated_bytes": int(torch.cuda.memory_allocated()),
        "process_reserved_bytes": int(torch.cuda.memory_reserved()),
        "process_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _json(config_path)
    torch.cuda.reset_peak_memory_stats()
    before_load = _memory()
    capability = public_capabilities(
        int(config["data"]["capability_seed"]), split="development", count=1
    )[0]
    rows = generate_rows(
        capability,
        split="source_train",
        rows=int(config["data"]["training_rows"]),
        depths=config["data"]["training_depths"],
        seed=int(config["data"]["training_seed"]),
    )
    host = FrozenNeuralHost(SPECS["qwen2"], device="cuda")
    adapters = GenericRecipientAdapterSet(
        host, rank=int(config["training"]["lora_rank"])
    )
    parameters = adapters.parameters()
    permitted = {id(parameter) for parameter in parameters}
    for parameter in host.model.parameters():
        parameter.requires_grad_(id(parameter) in permitted)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=0.0,
    )
    batch = rows[: int(config["training"]["batch_size"])]
    logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
    targets = host.target_ids([int(row["answer"]) for row in batch])
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    adapters.verify_base_frozen()
    return {
        "format": "abi-r12a-cuda-coexistence-diagnostic/1",
        "scientific_gate": False,
        "optimizer_steps": 1,
        "batch_size": len(batch),
        "loss": float(loss.detach()),
        "before_load": before_load,
        "after_step": _memory(),
        "device": torch.cuda.get_device_name(0),
        "base_state_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        result = run(Path(args.config).resolve())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    "format": "abi-r12a-cuda-coexistence-diagnostic/1",
                    "scientific_gate": False,
                    "status": "RESOURCE_BLOCKED",
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
