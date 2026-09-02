"""Replay one frozen R13 source adapter without training."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from experiments.native_transfer_r8.capability_generator import committed_heldout_capabilities
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
    sha256_file,
)

from .core import (
    R13Error,
    capability_rows,
    evidence_hash,
    json_object,
    observe_source,
    write_json_once,
    write_jsonl_once,
)


def run(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    capability_index: int,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R13Error(f"immutable source replay exists: {output}")
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    receipt = json_object(run_dir / "receipt.json")
    capabilities = committed_heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    if capability_index < 0 or capability_index >= len(capabilities):
        raise R13Error("invalid source replay capability index")
    _, evaluation, atomic = capability_rows(config, capabilities)
    capability = capabilities[capability_index]
    source = receipt["source_acquisitions"][capability_index]
    if source["capability_id"] != capability.capability_id:
        raise R13Error("source replay capability identity changed")
    artifact = run_dir / str(source["adapter_artifact"]["path"])
    if (
        not artifact.is_file()
        or artifact.stat().st_size != source["adapter_artifact"]["bytes"]
        or sha256_file(artifact) != source["adapter_artifact"]["sha256"]
    ):
        raise R13Error("source replay adapter changed")
    source_config = config["source_acquisition"]
    host = FrozenNeuralHost(SPECS["qwen2"], device=str(source_config["device"]))
    observations = []
    metrics = {}
    for condition, rows in (
        ("BEFORE_EVALUATION", evaluation[capability_index]),
        ("BEFORE_ATOMIC", atomic[capability_index]),
    ):
        measurement, values = observe_source(
            host,
            rows,
            capability_id=capability.capability_id,
            condition=condition,
            batch_size=int(source_config["evaluation_batch_size"]),
        )
        metrics[condition] = measurement
        observations.extend(values)
    adapters = GenericRecipientAdapterSet(host, rank=int(source_config["lora_rank"]))
    adapters.load_state(load_file(str(artifact), device="cpu"))
    adapters.freeze()
    adapters.verify_base_frozen()
    for condition, rows in (
        ("AFTER_EVALUATION", evaluation[capability_index]),
        ("AFTER_ATOMIC", atomic[capability_index]),
    ):
        measurement, values = observe_source(
            host,
            rows,
            capability_id=capability.capability_id,
            condition=condition,
            batch_size=int(source_config["evaluation_batch_size"]),
        )
        metrics[condition] = measurement
        observations.extend(values)
    output.mkdir(parents=True)
    rows_path = output / "observations.jsonl"
    write_jsonl_once(rows_path, observations)
    result = {
        "format": "abi-r13-source-live-replay/1",
        "capability_id": capability.capability_id,
        "pid": os.getpid(),
        "source_optimizer_steps": 0,
        "adapter_sha256": sha256_file(artifact),
        "base_state_sha256": adapters.base_state_sha256(),
        "metrics": metrics,
        "observations": {
            "path": rows_path.name,
            "rows": len(observations),
            "sha256": sha256_file(rows_path),
        },
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "receipt.json", result)
    del adapters, host
    gc.collect()
    torch.cuda.empty_cache()
    return result


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
            args.capability_index,
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
