"""Independent live replay of R15A source, extractor, and recipient execution."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.foreign_capability_r14.source import observe_queries
from experiments.native_transfer_r8.native_host import SPECS, FrozenNeuralHost

from .isolation import run_wsl_isolated_extraction
from .protocol import capability_rows, heldout_capabilities
from .run import _random_state, _shuffled_state
from .source import QwenLearningEvent
from .verify import SOURCE_FIELDS, verify


def _source_groups(run_dir: Path, receipt: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    reference = receipt["source_observations"]
    path = run_dir / reference["path"]
    if not path.is_file() or sha256_file(path) != reference["sha256"]:
        raise R14Error("R15A live replay source rows unavailable")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != reference["rows"]:
        raise R14Error("R15A live replay source row count changed")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if set(row) != SOURCE_FIELDS:
            raise R14Error("R15A live replay source schema changed")
        grouped[(str(row["capability_id"]), str(row["condition"]))].append(
            {key: row[key] for key in SOURCE_FIELDS - {"capability_id", "condition"}}
        )
    return grouped


def _compare_observations(
    observed: list[dict[str, Any]], expected: list[dict[str, Any]], label: str
) -> None:
    if observed != expected:
        raise R14Error(f"R15A live source replay changed: {label}")


def verify_live(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R15A live verification exists: {output}")
    static = verify(config_path, reveal_path, run_dir)
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    receipt = json_object(run_dir / "receipt.json")
    capabilities = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    rows_by_capability = capability_rows(config, capabilities)
    grouped = _source_groups(run_dir, receipt)
    host = FrozenNeuralHost(SPECS["qwen2"], device=str(config["runtime"]["device"]))
    event = QwenLearningEvent(
        host,
        rank=int(config["source_acquisition"]["lora_rank"]),
        initialization_seed=int(config["source_acquisition"]["initialization_seed"]),
    )
    source_replays = []
    states = [
        load_file(str(run_dir / item["adapter_artifact"]["path"]), device="cpu")
        for item in receipt["source_acquisitions"]
    ]
    for index, (capability, rows, state) in enumerate(
        zip(capabilities, rows_by_capability, states)
    ):
        conditions = {
            "BEFORE": event.before_state,
            "AFTER": state,
            "REMOVED": event.before_state,
            "RESTORED": state,
            "WRONG": states[(index + 1) % len(states)],
            "RANDOM": _random_state(state, capability.capability_id),
            "SHUFFLED": _shuffled_state(state, capability.capability_id),
        }
        for condition, condition_state in conditions.items():
            event.load_state(condition_state)
            observations, _ = observe_queries(
                host, rows["atomic"], batch_size=int(config["runtime"]["batch_size"])
            )
            _compare_observations(
                observations,
                grouped[(capability.capability_id, condition)],
                f"{capability.capability_id}/{condition}",
            )
        event.load_state(state)
        for condition, split in (
            ("QUERY", "queries"),
            ("SOURCE_EVALUATION", "source_evaluation"),
            ("SOURCE_COUNTERFACTUAL", "counterfactual"),
        ):
            observations, _ = observe_queries(
                host, rows[split], batch_size=int(config["runtime"]["batch_size"])
            )
            _compare_observations(
                observations,
                grouped[(capability.capability_id, condition)],
                f"{capability.capability_id}/{condition}",
            )
        source_replays.append(
            {
                "capability_id": capability.capability_id,
                "conditions_replayed": list(conditions) + [
                    "QUERY",
                    "SOURCE_EVALUATION",
                    "SOURCE_COUNTERFACTUAL",
                ],
                "rows_replayed": sum(len(rows[key]) for key in ("queries", "source_evaluation", "counterfactual"))
                + 7 * len(rows["atomic"]),
            }
        )
    event.verify_base_frozen()
    del event, host
    gc.collect()
    torch.cuda.empty_cache()

    output.mkdir(parents=True)
    public_receipt = json_object(config_path.parents[3] / config["public_frontend"]["receipt"])
    extraction_replays = []
    for index, item in enumerate(receipt["source_acquisitions"]):
        destination = output / "extractions" / f"event-{index:02d}"
        replay = run_wsl_isolated_extraction(
            config_path.parents[3],
            frontend=config_path.parents[3] / config["public_frontend"]["tensors"],
            frontend_spec=public_receipt["frontend"],
            delta=run_dir / item["delta_artifact"]["path"],
            destination=destination,
        )
        original = json_object(
            run_dir / receipt["neural_state_extractions"][index]["isolated_result"]["path"]
        )
        if (
            replay["result"]["labels"] != original["labels"]
            or replay["result"]["frontend_spec_sha256"]
            != original["frontend_spec_sha256"]
            or replay["result"]["weight_delta_sha256"]
            != original["weight_delta_sha256"]
        ):
            raise R14Error("R15A isolated extraction live replay changed")
        extraction_replays.append(
            {
                "capability_id": capabilities[index].capability_id,
                "result_sha256": sha256_file(destination / "result.json"),
                "launcher_sha256": sha256_file(destination / "launcher.json"),
                "labels": replay["result"]["labels"],
            }
        )

    recipient_replays = []
    for host_key in config["recipient_hosts"]:
        destination = output / "recipients" / str(host_key)
        command = [
            sys.executable,
            "-m",
            "experiments.foreign_neural_state_r15.recipient_worker",
            "--config",
            str(config_path),
            "--reveal",
            str(reveal_path),
            "--manifest",
            str(run_dir / "package_manifest.json"),
            "--host",
            str(host_key),
            "--output",
            str(destination),
        ]
        environment = dict(os.environ)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        completed = subprocess.run(
            command,
            cwd=config_path.parents[3],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise R14Error(f"R15A recipient live replay failed: {host_key}")
        replay = json_object(destination / "receipt.json")
        original = next(
            item for item in receipt["recipient_workers"] if item["host"] == host_key
        )
        if replay["summary"] != original["summary"]:
            raise R14Error(f"R15A recipient live replay changed: {host_key}")
        recipient_replays.append(
            {
                "host": host_key,
                "receipt_sha256": sha256_file(destination / "receipt.json"),
                "observations_sha256": sha256_file(destination / "observations.jsonl"),
                "summary": replay["summary"],
            }
        )
    result = {
        "format": "abi-r15a-live-verification/1",
        "verdict": "PASS",
        "static_verification_evidence_sha256": static["evidence_sha256"],
        "source_replays": source_replays,
        "isolated_extraction_replays": extraction_replays,
        "recipient_replays": recipient_replays,
        "source_base_model_sha256": receipt["source_acquisitions"][0][
            "base_model_sha256_before"
        ],
        "run_evidence_sha256": receipt["evidence_sha256"],
        "stored_scientific_status_booleans_consumed": 0,
        "claim_ceiling": "NOT_TEACHER_BEHAVIOR_CLONING_NOT_PREEXISTING_KNOWLEDGE",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = verify_live(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.run_dir).resolve(),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
