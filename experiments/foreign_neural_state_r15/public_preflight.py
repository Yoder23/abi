"""Run a bounded public R15A feasibility study before preregistration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from experiments.foreign_capability_r14.capability import (
    AffineCapability,
    capability_from_seed,
    generate_rows,
)
from experiments.native_isa_r11.core import transition_accuracy
from experiments.native_transfer_r8.capability_generator import canonical_json_bytes
from experiments.native_transfer_r8.native_host import SPECS, FrozenNeuralHost

from .frontend import (
    frontend_spec,
    labels_for_capability,
    predict_labels,
    train_affine_pair_frontend,
    train_affine_table_frontend,
    train_bilinear_frontend,
    train_frontend,
    transition_from_labels,
)
from .source import (
    FullQwenLoRALearningEvent,
    QwenLearningEvent,
    cache_output_training_inputs,
    canonical_accuracy,
    train_cached_output_learning_event,
    train_full_lora_learning_event,
    train_learning_event,
)


def _public_capabilities(count: int, *, seed: int, split: str) -> list[AffineCapability]:
    result = []
    seen = set()
    index = 0
    while len(result) < count:
        capability = capability_from_seed(seed + 104729 * index, split=split, index=index)
        index += 1
        if capability.operations in seen:
            continue
        seen.add(capability.operations)
        result.append(capability)
    return result


def _rows(
    capability: AffineCapability,
    *,
    index: int,
    curriculum: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    atomic = generate_rows(
        capability,
        split="r15_public_atomic",
        rows=24,
        depths=[1],
        seed=15015001 + 1009 * index,
    )
    if curriculum == "mixed":
        training = generate_rows(
            capability,
            split="r15_public_source_train",
            rows=4096,
            depths=range(1, 10),
            seed=15015011 + 1009 * index,
        )
    elif curriculum == "atomic_balanced":
        training = atomic
    else:
        raise RuntimeError(f"unknown source curriculum: {curriculum}")
    evaluation = generate_rows(
        capability,
        split="r15_public_unseen",
        rows=1024,
        depths=range(10, 15),
        seed=15015002 + 1009 * index,
        excluded_keys={str(row["program_key"]) for row in training + atomic},
    )
    return training, atomic, evaluation


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"immutable public preflight exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    output: Path,
    frontend_output: Path,
    *,
    train_events: int,
    development_events: int,
    steps: int,
    channel: str,
    curriculum: str,
    frontend_kind: str,
    long_source_eval_events: int,
    public_seed: int,
) -> dict[str, Any]:
    if output.exists() or frontend_output.exists():
        raise RuntimeError("immutable public R15A output already exists")
    journal = output.with_suffix(output.suffix + ".journal")
    journal.mkdir(parents=True, exist_ok=True)
    journal_meta = {
        "format": "abi-r15a-public-event-journal/1",
        "train_events": train_events,
        "development_events": development_events,
        "steps": steps,
        "channel": channel,
        "curriculum": curriculum,
        "frontend_kind": frontend_kind,
        "long_source_eval_events": long_source_eval_events,
        "public_seed": public_seed,
    }
    meta_path = journal / "meta.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != journal_meta:
            raise RuntimeError("public event journal contract changed")
    else:
        _write_json_once(meta_path, journal_meta)
    capabilities = _public_capabilities(
        train_events + development_events,
        seed=public_seed,
        split="r15_public",
    )
    host = FrozenNeuralHost(SPECS["qwen2"], device="cuda")
    if channel == "targeted_output":
        event: QwenLearningEvent | FullQwenLoRALearningEvent = QwenLearningEvent(
            host, rank=16, initialization_seed=15015003
        )
        adapter_description = "rank16_targeted_output_lora"
    elif channel == "cached_output_delta":
        event = QwenLearningEvent(host, rank=16, initialization_seed=15015003)
        adapter_description = "rank16_targeted_output_lora_cached_frozen_residual_training"
    elif channel == "full_lora_bilinear":
        event = FullQwenLoRALearningEvent(
            host,
            rank=8,
            initialization_seed=15015003,
            projections_per_module=8,
        )
        adapter_description = "rank8_all_linear_lora_effective_delta_bilinear_sketch"
    else:
        raise RuntimeError(f"unknown R15A neural-state channel: {channel}")
    feature_rows = []
    label_rows = []
    source_receipts = []
    output_cache = None
    for index, capability in enumerate(capabilities):
        feature_path = journal / f"event-{index:04d}.safetensors"
        event_receipt_path = journal / f"event-{index:04d}.json"
        if feature_path.exists() != event_receipt_path.exists():
            raise RuntimeError(f"incomplete journal event: {index}")
        if feature_path.exists():
            saved_receipt = json.loads(event_receipt_path.read_text(encoding="utf-8"))
            saved_feature = load_file(str(feature_path))["delta"].float().cpu()
            if (
                saved_receipt.get("capability_id") != capability.capability_id
                or saved_receipt.get("feature_sha256") != _sha256_file(feature_path)
            ):
                raise RuntimeError(f"journal event binding changed: {index}")
            feature_rows.append(saved_feature)
            label_rows.append(labels_for_capability(capability))
            source_receipts.append(saved_receipt)
            print(
                json.dumps({"event": index + 1, "events": len(capabilities), "resumed": True}),
                flush=True,
            )
            continue
        training, atomic, evaluation = _rows(
            capability,
            index=index,
            curriculum=curriculum,
        )
        if channel == "cached_output_delta":
            if curriculum != "atomic_balanced":
                raise RuntimeError("cached output channel requires the atomic curriculum")
            if output_cache is None:
                output_cache = cache_output_training_inputs(host, event, atomic)
            acquisition = train_cached_output_learning_event(
                event,
                training,
                output_cache,
                steps=steps,
                learning_rate=0.01,
            )
        elif isinstance(event, QwenLearningEvent):
            acquisition = train_learning_event(
                host,
                event,
                training,
                steps=steps,
                learning_rate=0.002,
                batch_size=32,
                seed=15015004 + 7919 * index,
            )
        else:
            acquisition = train_full_lora_learning_event(
                event,
                training,
                steps=steps,
                learning_rate=0.0002,
                batch_size=27,
                seed=15015004 + 7919 * index,
            )
        source_atomic = canonical_accuracy(host, atomic, batch_size=24)
        development_index = index - train_events
        if channel == "cached_output_delta" and (
            index < train_events or development_index >= long_source_eval_events
        ):
            source = {"rows": 0, "correct": 0, "accuracy": None, "not_run": True}
        else:
            source = canonical_accuracy(host, evaluation, batch_size=64)
        feature = event.delta().flatten()
        feature_rows.append(feature)
        label_rows.append(labels_for_capability(capability))
        saved_receipt = {
            "capability_id": capability.capability_id,
            "split": "frontend_train" if index < train_events else "development",
            "operations_commitment": hashlib.sha256(
                canonical_json_bytes([list(value) for value in capability.operations])
            ).hexdigest(),
            "source_atomic": source_atomic,
            "source_unseen": source,
            "acquisition": acquisition,
        }
        save_file({"delta": feature}, str(feature_path))
        saved_receipt["feature_sha256"] = _sha256_file(feature_path)
        _write_json_once(event_receipt_path, saved_receipt)
        source_receipts.append(saved_receipt)
        print(
            json.dumps(
                {
                    "event": index + 1,
                    "events": len(capabilities),
                    "source_accuracy": source["accuracy"],
                    "source_atomic_accuracy": source_atomic["accuracy"],
                    "seconds": acquisition["wall_seconds"],
                }
            ),
            flush=True,
        )
    features = torch.stack(feature_rows)
    labels = torch.stack(label_rows)
    train_x = features[:train_events]
    train_y = labels[:train_events]
    development_x = features[train_events:]
    development_y = labels[train_events:]
    trials = []
    best = None
    if frontend_kind == "kernel":
        states = [
            train_frontend(train_x, train_y, gamma=gamma, ridge=ridge)
            for gamma in (1.0, 4.0, 16.0, 64.0)
            for ridge in (0.0001, 0.01, 1.0)
        ]
    elif frontend_kind in {"bilinear", "bilinear_affine"}:
        states = [
            train_bilinear_frontend(
                train_x,
                train_y,
                affine_codebook=frontend_kind == "bilinear_affine",
            )
        ]
    elif frontend_kind == "affine_pair":
        states = [train_affine_pair_frontend(train_x, train_y)]
    elif frontend_kind == "affine_table":
        states = [train_affine_table_frontend(train_x, train_y)]
    else:
        raise RuntimeError(f"unknown frontend kind: {frontend_kind}")
    for state in states:
        predictions = torch.stack([predict_labels(state, row) for row in development_x])
        heads_correct = int(predictions.eq(development_y).sum())
        exact = int(predictions.eq(development_y).all(dim=1).sum())
        trial = {
            "frontend_format": state["format"],
            "head_accuracy": heads_correct / development_y.numel(),
            "exact_capabilities": exact,
            "capabilities": development_events,
        }
        if "gamma" in state:
            trial.update({"gamma": state["gamma"], "ridge": state["ridge"]})
        trials.append(trial)
        if best is None or (trial["exact_capabilities"], trial["head_accuracy"]) > (
            best[0]["exact_capabilities"],
            best[0]["head_accuracy"],
        ):
            best = (trial, state, predictions)
    assert best is not None
    selected, state, predictions = best
    development = []
    for offset, capability in enumerate(capabilities[train_events:]):
        labels_predicted = predictions[offset].tolist()
        try:
            transition = transition_from_labels(labels_predicted)
            _, _, evaluation = _rows(
                capability,
                index=train_events + offset,
                curriculum=curriculum,
            )
            package_accuracy = transition_accuracy(transition, evaluation)
            valid = True
        except RuntimeError:
            package_accuracy = 0.0
            valid = False
        development.append(
            {
                "capability_id": capability.capability_id,
                "labels_exact": labels_predicted == development_y[offset].tolist(),
                "valid_affine_program": valid,
                "package_unseen_accuracy": package_accuracy,
            }
        )
    tensor_path = frontend_output
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    if state["format"] == "abi-r15a-frozen-weight-delta-kernel-frontend/1":
        tensor_names = ("train_features", "coefficients")
    elif state["format"] == "abi-r15a-frozen-weight-delta-affine-pair-frontend/3":
        tensor_names = ("left", "right", "cross", "bias")
    else:
        tensor_names = ("readout", "bias")
    save_file(
        {name: state[name] for name in tensor_names},
        str(tensor_path),
        metadata={
            "format": state["format"],
            "frontend_spec_sha256": frontend_spec(state)["evidence_sha256"],
        },
    )
    result = {
        "format": "abi-r15a-public-weight-delta-preflight/1",
        "scope": "PUBLIC_DEVELOPMENT_ONLY_NOT_CERTIFICATION",
        "source": {
            "model_id": host.spec.model_id,
            "revision": host.spec.revision,
            "base_model_sha256": event.base_model_sha256,
            "adapter": adapter_description,
            "curriculum": curriculum,
            "allowed_delta_shape": list(event.delta().shape),
            "events": source_receipts,
        },
        "frontend": frontend_spec(state),
        "frontend_trials": trials,
        "selected_trial": selected,
        "development": development,
        "gates": {
            "development_exact_required": development_events,
            "development_exact_observed": sum(item["labels_exact"] for item in development),
            "source_atomic_exact_events": sum(
                item["source_atomic"]["accuracy"] == 1.0 for item in source_receipts
            ),
            "source_atomic_events": len(source_receipts),
            "public_preflight_pass": all(item["labels_exact"] for item in development),
        },
        "information_boundary": {
            "extractor_behavioral_queries": 0,
            "extractor_answers": 0,
            "extractor_oracle_calls": 0,
            "candidate_program_search": False,
            "input": "effective_qwen_output_weight_delta_only",
        },
        "hardware": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "claim_ceiling": "PUBLIC_FEASIBILITY_NOT_R15A_CERTIFICATION",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_json_once(output, result)
    del event, host
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frontend-output", required=True)
    parser.add_argument("--train-events", type=int, default=24)
    parser.add_argument("--development-events", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--channel",
        choices=("targeted_output", "cached_output_delta", "full_lora_bilinear"),
        default="targeted_output",
    )
    parser.add_argument(
        "--curriculum",
        choices=("mixed", "atomic_balanced"),
        default="mixed",
    )
    parser.add_argument(
        "--frontend-kind",
        choices=(
            "kernel",
            "bilinear",
            "bilinear_affine",
            "affine_pair",
            "affine_table",
        ),
        default="kernel",
    )
    parser.add_argument("--long-source-eval-events", type=int, default=32)
    parser.add_argument("--public-seed", type=int, default=15015000)
    args = parser.parse_args()
    result = run(
        Path(args.output).resolve(),
        Path(args.frontend_output).resolve(),
        train_events=args.train_events,
        development_events=args.development_events,
        steps=args.steps,
        channel=args.channel,
        curriculum=args.curriculum,
        frontend_kind=args.frontend_kind,
        long_source_eval_events=args.long_source_eval_events,
        public_seed=args.public_seed,
    )
    print(json.dumps(result["gates"], indent=2, sort_keys=True))
    return 0 if result["gates"]["public_preflight_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
