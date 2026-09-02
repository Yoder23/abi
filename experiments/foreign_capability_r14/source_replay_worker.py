"""Replay one frozen R14 source adapter in a fresh process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
)

from .capability import heldout_capabilities
from .core import (
    R14Error,
    capability_rows,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from .run import _adapter_state_sha256, _tag_observations
from .source import observe_queries


def run(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    capability_index: int,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable replay output exists: {output}")
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    receipt = json_object(run_dir / "receipt.json")
    capabilities = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    if not 0 <= capability_index < len(capabilities):
        raise R14Error("replay capability index changed")
    rows = capability_rows(config, capabilities)[capability_index]
    capability = capabilities[capability_index]
    source = receipt["source_acquisitions"][capability_index]
    if source["capability_id"] != capability.capability_id:
        raise R14Error("replay source order changed")
    host = FrozenNeuralHost(SPECS["qwen2"], device=str(config["source_acquisition"]["device"]))
    if host.model_state_sha256 != source["base_state_sha256_before"]:
        raise R14Error("replay base source changed")
    before, _ = observe_queries(
        host,
        rows["evaluation"][: int(config["data"]["before_rows_per_capability"])],
        batch_size=int(config["source_acquisition"]["evaluation_batch_size"]),
    )
    adapters = GenericRecipientAdapterSet(host, rank=int(config["source_acquisition"]["lora_rank"]))
    adapter_path = run_dir / source["adapter_artifact"]["path"]
    state = load_file(str(adapter_path), device="cpu")
    if (
        sha256_file(adapter_path) != source["adapter_artifact"]["sha256"]
        or _adapter_state_sha256(state) != source["adapter_state_sha256"]
    ):
        raise R14Error("replay source adapter changed")
    adapters.load_state(state)
    adapters.freeze()
    queries, _ = observe_queries(
        host,
        rows["queries"],
        batch_size=int(config["source_acquisition"]["evaluation_batch_size"]),
    )
    evaluation, _ = observe_queries(
        host,
        rows["evaluation"],
        batch_size=int(config["source_acquisition"]["evaluation_batch_size"]),
    )
    counterfactual, _ = observe_queries(
        host,
        rows["counterfactual"],
        batch_size=int(config["source_acquisition"]["evaluation_batch_size"]),
    )
    observations = [
        *_tag_observations(before, capability_id=capability.capability_id, split="BEFORE"),
        *_tag_observations(queries, capability_id=capability.capability_id, split="QUERY"),
        *_tag_observations(evaluation, capability_id=capability.capability_id, split="EVALUATION"),
        *_tag_observations(
            counterfactual,
            capability_id=capability.capability_id,
            split="COUNTERFACTUAL",
        ),
    ]
    output.mkdir(parents=True)
    observations_path = output / "observations.jsonl"
    write_jsonl_once(observations_path, observations)
    replay = {
        "format": "abi-r14-source-live-replay-worker/1",
        "capability_id": capability.capability_id,
        "pid": os.getpid(),
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "adapter_sha256": sha256_file(adapter_path),
        "base_state_sha256": host.model_state_sha256,
        "optimizer_steps": 0,
        "observations": {
            "path": observations_path.name,
            "rows": len(observations),
            "sha256": sha256_file(observations_path),
        },
    }
    replay["evidence_sha256"] = evidence_hash(replay)
    write_json_once(output / "receipt.json", replay)
    return replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--capability-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.run_dir).resolve(),
            int(args.capability_index),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
