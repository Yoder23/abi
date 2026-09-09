"""Execute the preregistered R15A foreign-weight-delta campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    source_metrics,
    write_json_once,
    write_jsonl_once,
)
from experiments.foreign_capability_r14.extractor import extract_transition
from experiments.foreign_capability_r14.recipient_worker import summarize
from experiments.foreign_capability_r14.source import observe_queries
from experiments.foreign_teacher_r12.custody import verify_r11_freeze
from experiments.native_isa_r11.core import (
    sha256_bytes,
    transition_accuracy,
    transition_bytes,
    write_package_once,
)
from experiments.native_transfer_r8.native_host import SPECS, FrozenNeuralHost

from .frontend import (
    load_frozen_frontend,
    predict_labels,
    transition_from_labels,
)
from .isolation import run_wsl_isolated_extraction
from .protocol import (
    capability_rows,
    heldout_capabilities,
    operations_commitment,
)
from .source import (
    QwenLearningEvent,
    cache_output_training_inputs,
    train_cached_output_learning_event,
)


def _tag(
    rows: list[dict[str, Any]], *, capability_id: str, condition: str
) -> list[dict[str, Any]]:
    return [
        {"capability_id": capability_id, "condition": condition, **row}
        for row in rows
    ]


def _observation_identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": row["row_id"],
            "prompt_sha256": row["prompt_sha256"],
            "start": row["start"],
            "program": row["program"],
            "canonical_probabilities": row["canonical_probabilities"],
        }
        for row in rows
    ]


def _shuffled_state(
    state: dict[str, torch.Tensor], capability_id: str
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int.from_bytes(hashlib.sha256((capability_id + "/shuffle").encode()).digest()[:8], "big")
    )
    permutation = torch.randperm(state["b"].shape[1], generator=generator)
    return {"a": state["a"].clone(), "b": state["b"].index_select(1, permutation)}


def _random_state(
    state: dict[str, torch.Tensor], capability_id: str
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int.from_bytes(hashlib.sha256((capability_id + "/random").encode()).digest()[:8], "big")
    )
    return {
        name: torch.randn(value.shape, generator=generator, dtype=value.dtype)
        * value.float().std().clamp_min(1e-6)
        for name, value in state.items()
    }


def _delta_controls(
    frontend: dict[str, Any],
    deltas: list[torch.Tensor],
    capabilities: list[Any],
    rows_by_capability: list[dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    controls = []
    for index, (delta, capability, rows) in enumerate(
        zip(deltas, capabilities, rows_by_capability)
    ):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int.from_bytes(
                hashlib.sha256((capability.capability_id + "/delta-random").encode()).digest()[:8],
                "big",
            )
        )
        random_delta = torch.randn(delta.shape, generator=generator) * delta.std().clamp_min(1e-8)
        row_generator = torch.Generator(device="cpu")
        row_generator.manual_seed(
            int.from_bytes(
                hashlib.sha256((capability.capability_id + "/delta-shuffle").encode()).digest()[:8],
                "big",
            )
        )
        shuffled_delta = delta.reshape(8, -1).index_select(
            0, torch.randperm(8, generator=row_generator)
        ).flatten()
        candidates = {
            "WRONG": deltas[(index + 1) % len(deltas)],
            "RANDOM": random_delta,
            "SHUFFLED": shuffled_delta,
        }
        values = {}
        for condition, candidate in candidates.items():
            labels = predict_labels(frontend, candidate).tolist()
            transition = transition_from_labels(labels)
            values[condition] = {
                "labels": labels,
                "unseen_accuracy": transition_accuracy(transition, rows["evaluation"]),
                "counterfactual_accuracy": transition_accuracy(
                    transition, rows["counterfactual"]
                ),
            }
        zero_rejected = float(torch.zeros_like(delta).norm()) <= 1e-12
        controls.append(
            {
                "capability_id": capability.capability_id,
                "zero_delta_rejected": zero_rejected,
                **values,
            }
        )
    return controls


def run(config_path: Path, reveal_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R15A output exists: {output}")
    config = json_object(config_path)
    if config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL":
        raise R14Error("R15A preregistration status changed")
    reveal = json_object(reveal_path)
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R14Error("R15A held-out reveal is malformed") from exc
    if (
        len(secret) != 32
        or hashlib.sha256(secret).hexdigest() != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        raise R14Error("R15A held-out reveal does not match preregistration")
    root = config_path.parents[3]
    r11_freeze = verify_r11_freeze(root, config)
    public = config["public_frontend"]
    public_receipt_path = root / str(public["receipt"])
    public_frontend_path = root / str(public["tensors"])
    if (
        sha256_file(public_receipt_path) != public["receipt_sha256"]
        or sha256_file(public_frontend_path) != public["tensors_sha256"]
    ):
        raise R14Error("R15A public frontend binding changed")
    public_receipt = json_object(public_receipt_path)
    if (
        public_receipt.get("evidence_sha256") != public["evidence_sha256"]
        or public_receipt.get("gates", {}).get("development_exact_observed") != 64
        or public_receipt.get("gates", {}).get("development_exact_required") != 64
    ):
        raise R14Error("R15A public qualification changed")
    frontend = load_frozen_frontend(public_frontend_path, public_receipt["frontend"])
    capabilities = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    rows_by_capability = capability_rows(config, capabilities)
    output.mkdir(parents=True)
    preserved_reveal = output / "heldout_reveal.json"
    shutil.copyfile(reveal_path, preserved_reveal)
    source_dir = output / "source"
    source_dir.mkdir()
    host = FrozenNeuralHost(SPECS["qwen2"], device=str(config["runtime"]["device"]))
    event = QwenLearningEvent(
        host,
        rank=int(config["source_acquisition"]["lora_rank"]),
        initialization_seed=int(config["source_acquisition"]["initialization_seed"]),
    )
    source_config = config["source_acquisition"]
    cache = cache_output_training_inputs(host, event, rows_by_capability[0]["atomic"])
    source_receipts = []
    source_observations: list[dict[str, Any]] = []
    deltas = []
    states = []
    for index, (capability, rows) in enumerate(zip(capabilities, rows_by_capability)):
        event.reset()
        before_atomic, _ = observe_queries(
            host, rows["atomic"], batch_size=int(config["runtime"]["batch_size"])
        )
        training = train_cached_output_learning_event(
            event,
            rows["atomic"],
            cache,
            steps=int(source_config["steps"]),
            learning_rate=float(source_config["learning_rate"]),
        )
        state = event.state()
        delta = event.delta().flatten()
        state_path = source_dir / f"event-{index:02d}-adapter.safetensors"
        delta_path = source_dir / f"event-{index:02d}-delta.safetensors"
        save_file(state, str(state_path))
        save_file({"delta": delta}, str(delta_path))
        after_atomic, _ = observe_queries(
            host, rows["atomic"], batch_size=int(config["runtime"]["batch_size"])
        )
        query_observations, _ = observe_queries(
            host, rows["queries"], batch_size=int(config["runtime"]["batch_size"])
        )
        source_evaluation, _ = observe_queries(
            host,
            rows["source_evaluation"],
            batch_size=int(config["runtime"]["batch_size"]),
        )
        source_counterfactual, _ = observe_queries(
            host, rows["counterfactual"], batch_size=int(config["runtime"]["batch_size"])
        )
        event.reset()
        removed_atomic, _ = observe_queries(
            host, rows["atomic"], batch_size=int(config["runtime"]["batch_size"])
        )
        event.load_state(state)
        restored_atomic, _ = observe_queries(
            host, rows["atomic"], batch_size=int(config["runtime"]["batch_size"])
        )
        if _observation_identity(before_atomic) != _observation_identity(removed_atomic):
            raise R14Error("R15A source adapter removal did not restore base behavior")
        if _observation_identity(after_atomic) != _observation_identity(restored_atomic):
            raise R14Error("R15A source adapter restoration changed behavior")
        source_observations.extend(_tag(before_atomic, capability_id=capability.capability_id, condition="BEFORE"))
        source_observations.extend(_tag(after_atomic, capability_id=capability.capability_id, condition="AFTER"))
        source_observations.extend(_tag(removed_atomic, capability_id=capability.capability_id, condition="REMOVED"))
        source_observations.extend(_tag(restored_atomic, capability_id=capability.capability_id, condition="RESTORED"))
        source_observations.extend(_tag(query_observations, capability_id=capability.capability_id, condition="QUERY"))
        source_observations.extend(_tag(source_evaluation, capability_id=capability.capability_id, condition="SOURCE_EVALUATION"))
        source_observations.extend(_tag(source_counterfactual, capability_id=capability.capability_id, condition="SOURCE_COUNTERFACTUAL"))
        source_receipts.append(
            {
                "capability_id": capability.capability_id,
                "operations_commitment": operations_commitment(capability),
                "model_id": host.spec.model_id,
                "revision": host.spec.revision,
                "base_model_sha256_before": event.base_model_sha256,
                "base_model_sha256_after": host.model_state_sha256,
                "adapter_state_sha256": event.state_sha256(),
                "adapter_artifact": {
                    "path": state_path.relative_to(output).as_posix(),
                    "bytes": state_path.stat().st_size,
                    "sha256": sha256_file(state_path),
                },
                "delta_artifact": {
                    "path": delta_path.relative_to(output).as_posix(),
                    "bytes": delta_path.stat().st_size,
                    "sha256": sha256_file(delta_path),
                    "effective_delta_sha256": event.delta_sha256(),
                    "elements": int(delta.numel()),
                },
                "training": training,
                "before_atomic": source_metrics(before_atomic, rows["atomic"]),
                "after_atomic": source_metrics(after_atomic, rows["atomic"]),
                "source_evaluation": source_metrics(source_evaluation, rows["source_evaluation"]),
                "source_counterfactual": source_metrics(source_counterfactual, rows["counterfactual"]),
                "removal_exact_to_before": True,
                "restoration_exact_to_after": True,
            }
        )
        deltas.append(delta)
        states.append(state)
        print(json.dumps({"source_event": index + 1, "events": len(capabilities)}), flush=True)

    for index, (capability, rows, state) in enumerate(
        zip(capabilities, rows_by_capability, states)
    ):
        condition_states = {
            "WRONG": states[(index + 1) % len(states)],
            "RANDOM": _random_state(state, capability.capability_id),
            "SHUFFLED": _shuffled_state(state, capability.capability_id),
        }
        for condition, condition_state in condition_states.items():
            event.load_state(condition_state)
            observations, _ = observe_queries(
                host, rows["atomic"], batch_size=int(config["runtime"]["batch_size"])
            )
            source_observations.extend(
                _tag(observations, capability_id=capability.capability_id, condition=condition)
            )
            source_receipts[index].setdefault("state_controls", {})[condition] = source_metrics(
                observations, rows["atomic"]
            )
    event.verify_base_frozen()

    package_items = []
    extraction_receipts = []
    black_box_receipts = []
    for index, (capability, rows, source) in enumerate(
        zip(capabilities, rows_by_capability, source_receipts)
    ):
        isolation_dir = output / "extractions" / f"event-{index:02d}"
        isolated = run_wsl_isolated_extraction(
            root,
            frontend=public_frontend_path,
            frontend_spec=public_receipt["frontend"],
            delta=output / source["delta_artifact"]["path"],
            destination=isolation_dir,
        )
        labels = [int(value) for value in isolated["result"]["labels"]]
        transition = transition_from_labels(labels)
        package = write_package_once(
            output / "packages",
            transition,
            {
                "teacher_before_sha256": source["base_model_sha256_before"],
                "teacher_after_sha256": source["adapter_state_sha256"],
            },
        )
        package["capability_id"] = capability.capability_id
        package_items.append(package)
        extraction_receipts.append(
            {
                "capability_id": capability.capability_id,
                "labels": labels,
                "labels_commitment": hashlib.sha256(
                    json.dumps(labels, separators=(",", ":")).encode()
                ).hexdigest(),
                "isolated_result": {
                    "path": (isolation_dir / "result.json").relative_to(output).as_posix(),
                    "sha256": sha256_file(isolation_dir / "result.json"),
                },
                "isolated_launcher": {
                    "path": (isolation_dir / "launcher.json").relative_to(output).as_posix(),
                    "sha256": sha256_file(isolation_dir / "launcher.json"),
                },
                "package_unseen_accuracy": transition_accuracy(transition, rows["evaluation"]),
                "package_counterfactual_accuracy": transition_accuracy(
                    transition, rows["counterfactual"]
                ),
            }
        )
        query_rows = [
            row
            for row in source_observations
            if row["capability_id"] == capability.capability_id and row["condition"] == "QUERY"
        ]
        query_rows = [
            {key: value for key, value in row.items() if key not in {"capability_id", "condition"}}
            for row in query_rows
        ]
        baseline_transition, baseline = extract_transition(query_rows)
        black_box_receipts.append(
            {
                "capability_id": capability.capability_id,
                "extraction": baseline,
                "operations_exact": baseline["selected_operations_commitment"]
                == operations_commitment(capability),
                "package_unseen_accuracy": transition_accuracy(
                    baseline_transition, rows["evaluation"]
                ),
                "package_counterfactual_accuracy": transition_accuracy(
                    baseline_transition, rows["counterfactual"]
                ),
            }
        )
        print(json.dumps({"extraction": index + 1, "events": len(capabilities)}), flush=True)

    controls = _delta_controls(frontend, deltas, capabilities, rows_by_capability)
    uniform = torch.full((3, 8, 8), 1.0 / 8.0)
    before_package = write_package_once(
        output / "packages",
        uniform,
        {
            "teacher_before_sha256": sha256_bytes(transition_bytes(uniform)),
            "teacher_after_sha256": sha256_bytes(transition_bytes(uniform)),
        },
    )
    manifest = {"before": before_package, "after": package_items}
    manifest_path = output / "package_manifest.json"
    write_json_once(manifest_path, manifest)
    observations_path = output / "source_observations.jsonl"
    write_jsonl_once(observations_path, source_observations)
    recipient_receipts = []
    for host_key in config["recipient_hosts"]:
        worker_output = output / "recipients" / str(host_key)
        command = [
            sys.executable,
            "-m",
            "experiments.foreign_neural_state_r15.recipient_worker",
            "--config",
            str(config_path),
            "--reveal",
            str(preserved_reveal),
            "--manifest",
            str(manifest_path),
            "--host",
            str(host_key),
            "--output",
            str(worker_output),
        ]
        environment = dict(os.environ)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        logs = output / "worker_logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{host_key}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (logs / f"{host_key}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise R14Error(f"R15A recipient worker failed: {host_key}")
        receipt = json_object(worker_output / "receipt.json")
        receipt["recomputed_summary"] = summarize(
            [
                json.loads(line)
                for line in (worker_output / "observations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ],
            [item["recipient"] for item in rows_by_capability],
        )
        recipient_receipts.append(receipt)
    receipt = {
        "format": "abi-r15a-foreign-weight-delta-extraction/1",
        "verification_status": "UNVERIFIED_RUN_OUTPUT",
        "claim_target": "BOUNDED_FOREIGN_NEURAL_STATE_CAPABILITY_RECOVERY",
        "claim_ceiling": "NOT_TEACHER_BEHAVIOR_CLONING_NOT_PREEXISTING_KNOWLEDGE",
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(preserved_reveal),
        "manifest_sha256": sha256_file(manifest_path),
        "r11_freeze": r11_freeze,
        "public_frontend": public,
        "data": {
            "capabilities": [item.capability_id for item in capabilities],
            "atomic_rows_per_capability": 24,
            "black_box_queries_per_capability": len(rows_by_capability[0]["queries"]),
            "evaluation_rows_per_capability": len(rows_by_capability[0]["evaluation"]),
            "counterfactual_rows_per_capability": len(rows_by_capability[0]["counterfactual"]),
            "recipient_rows_per_capability": len(rows_by_capability[0]["recipient"]),
        },
        "source_acquisitions": source_receipts,
        "source_observations": {
            "path": observations_path.name,
            "rows": len(source_observations),
            "sha256": sha256_file(observations_path),
        },
        "neural_state_extractions": extraction_receipts,
        "black_box_baseline": black_box_receipts,
        "delta_controls": controls,
        "packages": manifest,
        "recipient_workers": recipient_receipts,
        "heldout": {
            "commitment": config["heldout_seed_commitment"],
            "reveal_path": preserved_reveal.name,
        },
        "hardware": {
            "source_device": str(config["runtime"]["device"]),
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    del event, host
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
