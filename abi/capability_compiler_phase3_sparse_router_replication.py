"""Locked V46 replications for the passing V45 sparse router architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_bpe_core import _json
from .capability_compiler_phase3_segment_router import METADATA
from . import capability_compiler_phase3_sparse_router as base


def _replication_decision(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    metadata: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    raw_hash: str,
) -> Mapping[str, Any]:
    result = dict(base._decision(protocol, protocol_hash, metadata, rows, raw_hash))
    passed = bool(result["gates"]["router_gate_pass"])
    result["format"] = "abi-capability-compiler-phase3-sparse-router-replication-decision/1"
    result["status"] = (
        "PASS_SPARSE_ROUTER_REPLICATION"
        if passed
        else "FAIL_SPARSE_ROUTER_REPLICATION_ARCHITECTURE_CLOSED"
    )
    result["protocol"] = {"path": protocol_path.name, "sha256": protocol_hash}
    result["replication_seed"] = int(protocol["training"]["seed"])
    result["next_step"] = (
        "Combine with the other locked seeds; only a three-seed pass may open a host construct."
        if passed
        else "Close the architecture; a routed host is prohibited."
    )
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


@torch.inference_mode()
def evaluate(
    root: Path, protocol_path: Path, candidate: Path, output: Path
) -> Mapping[str, Any]:
    protocol, protocol_hash = base.load_protocol(root, protocol_path)
    metadata = _json(candidate / "metadata.json")
    if (
        output.exists()
        or metadata.get("protocol_sha256") != protocol_hash
        or sha256_file(candidate / "router.safetensors") != metadata["checkpoint"]["sha256"]
    ):
        raise Phase3Error("sparse-router replication identity failed")
    model, tokenizer = base._load(root, protocol, candidate)
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    rows = []
    for probe in probes:
        capability = str(probe["canonical_capability"])
        prompt = str(probe["prompt"])
        lines = prompt.splitlines()
        body = "\n".join(lines[1:]).strip()
        metadata_segment = lines[0].strip()
        for variant, text in (("original", prompt), ("body", body)):
            prediction, details = base._route(model, tokenizer, protocol, text)
            rows.append(
                {
                    "probe_id": str(probe["probe_id"]),
                    "capability": capability,
                    "variant": variant,
                    "predicted": prediction,
                    "correct": prediction == capability,
                    "segments": details,
                }
            )
        metadata_prediction = int(
            base._score(model, tokenizer, protocol, [metadata_segment]).argmax(dim=-1)[0]
        )
        rows.append(
            {
                "probe_id": str(probe["probe_id"]),
                "capability": capability,
                "variant": "metadata",
                "predicted": (*CAPABILITIES, METADATA)[metadata_prediction],
                "correct": metadata_prediction == len(CAPABILITIES),
            }
        )
    output.mkdir(parents=True)
    raw_path = output / "rows.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    decision = _replication_decision(
        protocol_path,
        protocol,
        protocol_hash,
        metadata,
        rows,
        sha256_file(raw_path),
    )
    _write_immutable(
        output / "decision.json",
        json.dumps(decision, sort_keys=True, indent=2).encode() + b"\n",
    )
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train", "evaluate"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = (root / arguments.protocol).resolve()
    candidate = (root / arguments.candidate_dir).resolve()
    output = (root / arguments.output_dir).resolve()
    if arguments.command == "inventory":
        result = base.inventory(root, protocol_path)
    elif arguments.command == "train":
        result = base.train(root, protocol_path, candidate)
    else:
        result = evaluate(root, protocol_path, candidate, output)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
