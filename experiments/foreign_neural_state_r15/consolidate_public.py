"""Consolidate an immutable R15 public event journal for release verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file


def consolidate(receipt_path: Path, journal: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise R14Error(f"immutable R15 public dataset exists: {output}")
    receipt = json_object(receipt_path)
    events = receipt.get("source", {}).get("events", [])
    if len(events) != 320:
        raise R14Error("R15 public source event count changed")
    features = []
    for index, event in enumerate(events):
        tensor_path = journal / f"event-{index:04d}.safetensors"
        event_path = journal / f"event-{index:04d}.json"
        if (
            not tensor_path.is_file()
            or not event_path.is_file()
            or json_object(event_path) != event
            or sha256_file(tensor_path) != event.get("feature_sha256")
        ):
            raise R14Error(f"R15 public journal event changed: {index}")
        state = load_file(str(tensor_path), device="cpu")
        if set(state) != {"delta"} or state["delta"].numel() != 7168:
            raise R14Error(f"R15 public journal tensor changed: {index}")
        features.append(state["delta"].float().contiguous())
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"features": torch.stack(features)},
        str(output),
        metadata={
            "format": "abi-r15a-public-weight-delta-dataset/1",
            "receipt_sha256": sha256_file(receipt_path),
            "receipt_evidence_sha256": str(receipt["evidence_sha256"]),
        },
    )
    return {
        "format": "abi-r15a-public-weight-delta-dataset/1",
        "events": len(features),
        "feature_elements": int(features[0].numel()),
        "receipt_sha256": sha256_file(receipt_path),
        "dataset_sha256": sha256_file(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = consolidate(
            Path(args.receipt).resolve(),
            Path(args.journal).resolve(),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
