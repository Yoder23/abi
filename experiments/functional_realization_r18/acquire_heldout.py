"""Acquire and seal the frozen source outputs for R18 held-out replication."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from experiments.factual_semantic_r16.public_qualification import _load_source
from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.linguistic_realization_r17.public_qualification import (
    _source_observations,
)

from .heldout_protocol import load_bound_inputs
from .hidden_frames import heldout_rows
from .verify import _parse_template


def _parseable_by_signature(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row["split"] != "extraction":
            continue
        signature = str(row["signature"])
        counts.setdefault(signature, 0)
        counts[signature] += int(_parse_template(str(row["output"]), row["slots"]) is not None)
    return counts


def run(config_path: Path, reveal_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R18 source output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config, reveal = load_bound_inputs(root, config_path, reveal_path)
    source = config["source"]
    commitment = str(config["heldout_seed_commitment"])

    def builder(split: str) -> list[dict[str, Any]]:
        return heldout_rows(str(reveal["secret_hex"]), commitment, split)

    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(source["model_id"], source["revision"])
    source_parameters = sum(parameter.numel() for parameter in model.parameters())
    observations, compiler_records, counters = _source_observations(tokenizer, model, builder)
    peak_gpu = int(torch.cuda.max_memory_allocated())
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    rows_path = output / "source_observations.jsonl"
    write_jsonl_once(rows_path, observations)
    parseable = _parseable_by_signature(observations)
    minimum = min(parseable.values()) if parseable else 0
    authorized = (
        len(observations) == 120
        and len(compiler_records) == 72
        and len(parseable) == 24
        and minimum >= int(config["gates"]["minimum_parseable_per_signature"])
    )
    receipt = {
        "format": "abi-r18-heldout-source-acquisition/1",
        "verdict": "PASS_SOURCE_COMPILABILITY" if authorized else "FAIL_SOURCE_COMPILABILITY",
        "compiler_authorized": authorized,
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "protocol_sha256": sha256_file(Path(__file__).with_name("HELDOUT_PROTOCOL.md")),
        "source": {
            "model_id": source["model_id"],
            "revision": source["revision"],
            "snapshot_path_name": snapshot.name,
            "parameters": source_parameters,
            "training_steps": 0,
            "device": "cuda",
            "present_at_compilation": False,
            "present_at_package_execution": False,
        },
        "metrics": {
            "source_rows": len(observations),
            "extraction_rows": len(compiler_records),
            "evaluation_rows": len(observations) - len(compiler_records),
            "source_functional_exact": sum(row["functional_exact"] for row in observations),
            "extraction_functional_exact": sum(
                row["functional_exact"] for row in observations if row["split"] == "extraction"
            ),
            "evaluation_functional_exact": sum(
                row["functional_exact"] for row in observations if row["split"] == "evaluation"
            ),
            "parseable_by_signature": parseable,
            "minimum_parseable_per_signature": minimum,
        },
        "artifacts": {"source_rows": {"path": rows_path.name, "sha256": sha256_file(rows_path)}},
        "information_accounting": {
            **counters,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_gpu,
            "source_parameters": source_parameters,
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "frozen_source_parameters_copied": 0,
        },
        "full_abi_moonshot": "OPEN",
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.reveal, args.output), indent=2))


if __name__ == "__main__":
    main()
