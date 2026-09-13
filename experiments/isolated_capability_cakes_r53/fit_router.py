"""Fit and seal R53's response-blind 14-way capability-label router."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from abi.hf_extraction import load_probe_catalog
from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER
from abi.layercake_host import _sha256_file
from experiments.external_router_cake_r49.run_v1 import FEATURES, _feature
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once


CATALOG_SHA256 = "8992c7de94d3733d66f2083f96ec8d3cba31be3943afe1849f42220f33ff8d08"
SEED = 53_002
STEPS = 300
LEARNING_RATE = 0.05
WEIGHT_DECAY = 0.001
CAPABILITY_TO_INDEX = {
    capability: index for index, capability in enumerate(CAPABILITY_CAKE_ORDER)
}


class RouterError(RuntimeError):
    """Raised when the frozen R53 routing contract is not reproducible."""


def matrix(rows: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack([_feature(str(row["prompt"])) for row in rows])
    labels = torch.tensor(
        [CAPABILITY_TO_INDEX[str(row["capability"])] for row in rows],
        dtype=torch.long,
    )
    return inputs, labels


def score(
    router: torch.nn.Linear,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    with torch.inference_mode():
        predictions = router(inputs).argmax(dim=-1)
    correct = int((predictions == labels).sum())
    rotated = (labels + 1) % len(CAPABILITY_CAKE_ORDER)
    return {
        "rows": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels),
        "rotated_correct": int((predictions == rotated).sum()),
        "rotated_accuracy": float((predictions == rotated).float().mean()),
        "predictions_sha256": hashlib.sha256(
            bytes(int(value) for value in predictions.cpu())
        ).hexdigest(),
    }


def run(catalog_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RouterError(f"immutable R53 router output exists: {output}")
    if not catalog_path.is_file() or _sha256_file(catalog_path) != CATALOG_SHA256:
        raise RouterError("R53 catalog changed")
    catalog = load_probe_catalog(catalog_path)
    by_split = {
        split: [dict(row) for row in catalog["probes"] if row["split"] == split]
        for split in ("search", "validation", "final_test")
    }
    expected_capabilities = set(CAPABILITY_CAKE_ORDER)
    for split, rows in by_split.items():
        counts = {
            capability: sum(row["capability"] == capability for row in rows)
            for capability in CAPABILITY_CAKE_ORDER
        }
        if (
            len(rows) != 1_400
            or set(row["capability"] for row in rows) != expected_capabilities
            or any(value != 100 for value in counts.values())
        ):
            raise RouterError(f"R53 {split} routing matrix changed")

    matrices = {split: matrix(rows) for split, rows in by_split.items()}
    torch.manual_seed(SEED)
    router = torch.nn.Linear(FEATURES, len(CAPABILITY_CAKE_ORDER))
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_x, train_y = matrices["search"]
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = router(train_x)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "accuracy": float((logits.argmax(dim=-1) == train_y).float().mean()),
            }
            trace.append(row)
            print(json.dumps(row), flush=True)
    router.eval()
    scores = {
        split: score(router, inputs, labels)
        for split, (inputs, labels) in matrices.items()
    }
    passed = all(
        value["accuracy"] == 1.0 and value["rotated_correct"] == 0
        for value in scores.values()
    )

    output.mkdir(parents=True)
    router_path = output / "router.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in router.state_dict().items()
        },
        str(router_path),
        metadata={"format": "abi-r53-capability-label-router/1"},
    )
    result = {
        "format": "abi-r53-capability-label-router/1",
        "verdict": "PASS_R53_ROUTER_PREREQUISITE" if passed else "FAIL_R53_ROUTER_PREREQUISITE",
        "catalog_sha256": CATALOG_SHA256,
        "router_sha256": _sha256_file(router_path),
        "router_bytes": router_path.stat().st_size,
        "parameters": sum(value.numel() for value in router.parameters()),
        "capability_order": list(CAPABILITY_CAKE_ORDER),
        "features": FEATURES,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "teacher_calls": 0,
        "response_targets": 0,
        "scores": scores,
        "trace": trace,
        "wall_seconds": time.perf_counter() - started,
        "passed": passed,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RouterError("R53 exact response-blind routing prerequisite failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.catalog.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
