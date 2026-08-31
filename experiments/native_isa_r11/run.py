"""Run R11 teacher learning, extraction, and native neural recipient transfer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from experiments.copy_paste_r10.runtime import canonical_prediction
from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    committed_heldout_capabilities,
    generate_rows,
)
from experiments.native_transfer_r8.recipient_worker import _disable_network

from .core import (
    NativeResidualISAHost,
    R11Error,
    RecurrentTransitionNeuralISA,
    sha256_bytes,
    sha256_file,
    train_teacher_transition,
    transition_accuracy,
    transition_bytes,
    write_package_once,
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R11Error(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise R11Error(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise R11Error(f"registered path escapes repository: {relative}") from exc
    return path


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise R11Error(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise R11Error(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _bind(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    bindings = config.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise R11Error("R11 bindings missing")
    actual = {}
    for relative, expected in bindings.items():
        path = _resolve(root, str(relative))
        if not path.is_file() or sha256_file(path) != expected:
            raise R11Error(f"registered R11 binding changed: {relative}")
        actual[str(relative)] = str(expected)
    return actual


def _rows(
    config: Mapping[str, Any], capabilities: Sequence[Any]
) -> tuple[list[list[Any]], list[list[Any]]]:
    data = config["data"]
    training = [
        generate_rows(
            capability,
            split="r11_teacher_training",
            rows=int(data["training_rows_per_capability"]),
            depths=data["training_depths"],
            seed=int(data["training_seed"]) + 1009 * index,
        )
        for index, capability in enumerate(capabilities)
    ]
    evaluation = [
        generate_rows(
            capability,
            split="r11_heldout_evaluation",
            rows=int(data["evaluation_rows_per_capability"]),
            depths=data["evaluation_depths"],
            seed=int(data["evaluation_seed"]) + 9001 * index,
        )
        for index, capability in enumerate(capabilities)
    ]
    return training, evaluation


def _random_transition(reference: torch.Tensor, label: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big"))
    return torch.softmax(torch.randn(reference.shape, generator=generator), dim=-1)


def _shuffled_transition(reference: torch.Tensor, label: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(hashlib.sha256(label.encode()).digest()[8:16], "big"))
    result = reference.clone()
    for operation in range(3):
        result[operation] = result[operation].index_select(
            0, torch.randperm(8, generator=generator)
        )
    return result


def _conditions(
    before: torch.Tensor,
    after: torch.Tensor,
    wrong: torch.Tensor,
    capability_id: str,
) -> dict[str, torch.Tensor | None]:
    return {
        "BASE": None,
        "AFTER": after,
        "BEFORE": before,
        "WRONG": wrong,
        "ZERO": torch.zeros_like(after),
        "RANDOM": _random_transition(after, capability_id),
        "SHUFFLED": _shuffled_transition(after, capability_id),
        "REMOVED": None,
        "BACKEND_REMOVED": None,
        "CODEC_REMOVED": None,
        "RESTORED": after,
    }


@torch.inference_mode()
def _host_matrix(
    host_key: str,
    config: Mapping[str, Any],
    codec_tensors: Mapping[str, torch.Tensor],
    codec_receipt: Mapping[str, Any],
    capabilities: Sequence[Any],
    evaluation_rows: Sequence[Sequence[Mapping[str, Any]]],
    before: torch.Tensor,
    after: Sequence[torch.Tensor],
    package_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    host_records = {item["host"]: item for item in codec_receipt["hosts"]}
    frozen = host_records[host_key]
    host = NativeResidualISAHost(
        host_key,
        codec=codec_tensors[host_key],
        codec_sha256=frozen["codec_sha256"],
        scale=float(frozen["residual_scale"]),
        device="cuda",
    )
    executor = RecurrentTransitionNeuralISA().to(host.device).eval()
    observations = []
    started = time.perf_counter()
    for capability_index, (capability, rows) in enumerate(zip(capabilities, evaluation_rows)):
        wrong_index = (capability_index + 1) % len(after)
        conditions = _conditions(
            before, after[capability_index], after[wrong_index], capability.capability_id
        )
        if list(conditions) != list(config["conditions"]):
            raise R11Error("registered condition order changed")
        hashes = {
            "BASE": None,
            "AFTER": package_manifest["after"][capability_index]["sha256"],
            "BEFORE": package_manifest["before"]["sha256"],
            "WRONG": package_manifest["after"][wrong_index]["sha256"],
            "ZERO": "CONTROL_ZERO",
            "RANDOM": "CONTROL_RANDOM",
            "SHUFFLED": "CONTROL_SHUFFLED",
            "REMOVED": None,
            "BACKEND_REMOVED": package_manifest["after"][capability_index]["sha256"],
            "CODEC_REMOVED": package_manifest["after"][capability_index]["sha256"],
            "RESTORED": package_manifest["after"][capability_index]["sha256"],
        }
        batch_size = int(config["runtime"]["batch_size"])
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            prompts = [str(row["prompt"]) for row in batch]
            base_logits, hidden = host.base_state(prompts)
            for condition, transition in conditions.items():
                active = condition not in {
                    "BASE",
                    "REMOVED",
                    "BACKEND_REMOVED",
                    "CODEC_REMOVED",
                }
                if active:
                    distribution = executor(transition.to(host.device), prompts)
                    logits = host.realize(hidden, distribution)
                else:
                    logits = base_logits
                    distribution = host.canonical_probabilities(base_logits)
                predictions = logits.argmax(dim=-1)
                canonical_probabilities = host.canonical_probabilities(logits)
                for row, prediction, probability, neural_state in zip(
                    batch, predictions, canonical_probabilities, distribution
                ):
                    token_id = int(prediction.cpu())
                    canonical = canonical_prediction(token_id, host.target_token_ids)
                    if canonical is None:
                        output_bytes = host.host.tokenizer.decode([token_id]).encode(
                            "utf-8", errors="replace"
                        )
                    else:
                        output_bytes = str(canonical).encode("utf-8")
                    observations.append(
                        {
                            "host": host_key,
                            "capability_id": capability.capability_id,
                            "condition": condition,
                            "row_id": row["row_id"],
                            "prompt_sha256": row["prompt_sha256"],
                            "package_sha256": hashes[condition],
                            "backend_active": active,
                            "codec_active": active,
                            "prediction_token_id": token_id,
                            "canonical_prediction": canonical,
                            "canonical_output_utf8_hex": output_bytes.hex(),
                            "canonical_probabilities": [
                                float(value) for value in probability.cpu()
                            ],
                            "neural_state": [float(value) for value in neural_state.cpu()],
                        }
                    )
    host.verify_frozen()
    receipt = {
        "host": host_key,
        "model_id": host.host.spec.model_id,
        "revision": host.host.spec.revision,
        "architecture_family": host.host.spec.architecture_family,
        "model_state_sha256_before": host.model_state_sha256,
        "model_state_sha256_after": host.model_state_sha256,
        "codec_sha256_before": host.codec_sha256,
        "codec_sha256_after": host.codec_sha256,
        "target_token_ids": list(host.target_token_ids),
        "recipient_optimizer_steps": 0,
        "backend_learned_parameters": executor.learned_parameters,
        "source_model_loaded": host_key == "source",
        "rows": len(observations),
        "wall_seconds": time.perf_counter() - started,
    }
    del host, executor
    gc.collect()
    torch.cuda.empty_cache()
    return observations, receipt


def run(config_path: Path, reveal_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R11Error(f"immutable R11 output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = _json(config_path)
    if config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL":
        raise R11Error("R11 final preregistration is not frozen")
    bindings = _bind(root, config)
    reveal = _json(reveal_path)
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R11Error("held-out reveal is not hexadecimal") from exc
    if (
        len(secret) != 32
        or sha256_bytes(secret) != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        raise R11Error("held-out reveal does not match preregistration")
    capabilities = committed_heldout_capabilities(
        reveal["secret_hex"],
        expected_commitment=config["heldout_seed_commitment"],
        count=int(config["data"]["heldout_capabilities"]),
    )
    training_rows, evaluation_rows = _rows(config, capabilities)
    codec_receipt = _json(_resolve(root, config["codec_freeze"]["receipt"]))
    codec_tensors = load_file(str(_resolve(root, config["codec_freeze"]["tensors"])), device="cpu")
    before = torch.full((3, 8, 8), 1.0 / 8.0)
    after = []
    training_receipts = []
    for index, (capability, rows) in enumerate(zip(capabilities, training_rows)):
        transition, training = train_teacher_transition(
            rows,
            steps=int(config["teacher_training"]["steps"]),
            learning_rate=float(config["teacher_training"]["learning_rate"]),
            seed=int(config["teacher_training"]["seed"]) + 4001 * index,
            device="cuda",
        )
        before_accuracy = transition_accuracy(before, evaluation_rows[index])
        after_accuracy = transition_accuracy(transition, evaluation_rows[index])
        after.append(transition)
        training_receipts.append(
            {
                "capability_id": capability.capability_id,
                "teacher_before_sha256": sha256_bytes(transition_bytes(before)),
                "teacher_after_sha256": sha256_bytes(transition_bytes(transition)),
                "canonical_before_accuracy": before_accuracy,
                "canonical_after_accuracy": after_accuracy,
                "training": training,
            }
        )
    output.mkdir(parents=True)
    package_dir = output / "packages"
    before_package = write_package_once(
        package_dir,
        before,
        {
            "teacher_before_sha256": sha256_bytes(transition_bytes(before)),
            "teacher_after_sha256": sha256_bytes(transition_bytes(before)),
        },
    )
    after_packages = []
    for capability, transition in zip(capabilities, after):
        item = write_package_once(
            package_dir,
            transition,
            {
                "teacher_before_sha256": sha256_bytes(transition_bytes(before)),
                "teacher_after_sha256": sha256_bytes(transition_bytes(transition)),
            },
        )
        item["capability_id"] = capability.capability_id
        after_packages.append(item)
    manifest = {"before": before_package, "after": after_packages}
    _disable_network()
    teacher_started = time.perf_counter()
    source_rows, source_receipt = _host_matrix(
        "source",
        config,
        codec_tensors,
        codec_receipt,
        capabilities,
        evaluation_rows,
        before,
        after,
        manifest,
    )
    teacher_finished = time.perf_counter()
    source_path = output / "teacher_observations.jsonl"
    _write_jsonl_once(source_path, source_rows)
    recipient_rows = []
    recipient_receipts = []
    recipient_started = time.perf_counter()
    for host_key in config["recipient_hosts"]:
        rows, receipt = _host_matrix(
            str(host_key),
            config,
            codec_tensors,
            codec_receipt,
            capabilities,
            evaluation_rows,
            before,
            after,
            manifest,
        )
        recipient_rows.extend(rows)
        recipient_receipts.append(receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    recipient_finished = time.perf_counter()
    recipient_path = output / "recipient_observations.jsonl"
    _write_jsonl_once(recipient_path, recipient_rows)
    receipt = {
        "format": "abi-native-neural-isa-r11-run/1",
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "bindings": bindings,
        "claim_target": "LOSSLESS_OUTPUT_EQUIVALENCE_SYNTHETIC_ABI_NATIVE_TEACHER",
        "claim_ceiling": "LEVEL_1_OUTPUT_EQUIVALENCE_ONLY",
        "packages": manifest,
        "teacher_training": training_receipts,
        "teacher_execution": {
            "started": teacher_started,
            "finished": teacher_finished,
            "host": source_receipt,
            "observations": {
                "path": source_path.name,
                "sha256": sha256_file(source_path),
                "rows": len(source_rows),
            },
        },
        "recipient_execution": {
            "started": recipient_started,
            "finished": recipient_finished,
            "teacher_finished_before_recipient_started": teacher_finished <= recipient_started,
            "teacher_loaded_in_recipient_process": False,
            "hosts": recipient_receipts,
            "observations": {
                "path": recipient_path.name,
                "sha256": sha256_file(recipient_path),
                "rows": len(recipient_rows),
            },
        },
        "heldout": {
            "commitment": config["heldout_seed_commitment"],
            "capabilities": [item.capability_id for item in capabilities],
            "revealed_after_codec_freeze": True,
        },
        "hardware": {
            "device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    _write_json_once(output / "receipt.json", receipt)
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
