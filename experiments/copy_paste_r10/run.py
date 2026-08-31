"""Run the R10 source-to-package-to-frozen-host copy/paste matrix."""

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

from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    generate_rows,
    public_capabilities,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    module_sha256,
)
from experiments.native_transfer_r8.recipient_worker import (
    _disable_network,
    _random_latent,
    _shuffled_latent,
)
from experiments.native_transfer_r8.source_transition import (
    NeuralTransitionSource,
    load_controller_state,
    unpack_controller_state,
)

from .runtime import (
    CanonicalTransitionVM,
    CopyPasteRuntimeError,
    apply_host_codec,
    canonical_prediction,
    discover_canonical_token_map,
    load_package,
    sha256_file,
    write_package_once,
)


class R10RunError(RuntimeError):
    """Raised when the registered R10 execution boundary changes."""


class R10FrozenNeuralHost(FrozenNeuralHost):
    """R8 frozen host with an exact, tokenizer-generic canonical output map."""

    def _target_tokens(self) -> tuple[list[int], list[str]]:
        return discover_canonical_token_map(
            self.tokenizer, encoder_decoder=self.spec.encoder_decoder
        )


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R10RunError(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise R10RunError(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise R10RunError(f"registered path escapes repository: {relative}") from exc
    return path


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise R10RunError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise R10RunError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _bind_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    reference = config["r8_reference"]
    pairs = (
        ("config", "config_sha256"),
        ("source_receipt", "source_receipt_sha256"),
        ("source_states", "source_states_sha256"),
        ("extraction_receipt", "extraction_receipt_sha256"),
        ("canonical_latents", "canonical_latents_sha256"),
    )
    result = {}
    for path_key, hash_key in pairs:
        path = _resolve(root, str(reference[path_key]))
        actual = sha256_file(path)
        if not path.is_file() or actual != reference[hash_key]:
            raise R10RunError(f"registered R8 input changed: {path_key}")
        result[str(path.relative_to(root)).replace("\\", "/")] = actual
    implementation = config.get("implementation")
    if not isinstance(implementation, dict) or not implementation:
        raise R10RunError("implementation hashes are not frozen")
    for relative, expected in implementation.items():
        path = _resolve(root, str(relative))
        actual = sha256_file(path)
        if actual != expected:
            raise R10RunError(f"registered implementation changed: {relative}")
        result[str(relative)] = actual
    return result


def _evaluation_rows(
    config: Mapping[str, Any], r8: Mapping[str, Any]
) -> tuple[list[Any], list[list[dict[str, Any]]]]:
    matrix = config["public_matrix"]
    split = r8["splits"]
    capabilities = public_capabilities(
        int(split["development_seed"]),
        split="development",
        count=int(matrix["development_capabilities"]),
    )
    rows = [
        generate_rows(
            capability,
            split="r10_copy_paste_evaluation",
            rows=int(matrix["evaluation_rows_per_capability"]),
            depths=matrix["evaluation_depths"],
            seed=int(matrix["evaluation_seed"]) + 4001 * index,
        )
        for index, capability in enumerate(capabilities)
    ]
    return capabilities, rows


@torch.inference_mode()
def _source_observations(
    config: Mapping[str, Any],
    r8: Mapping[str, Any],
    source_states_path: Path,
    capabilities: Sequence[Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = config["runtime"]
    host = R10FrozenNeuralHost(SPECS["source"], device="cuda")
    controller = NeuralTransitionSource(host, seed=int(r8["training"]["seed"])).to(host.device)
    packed = load_file(str(source_states_path), device="cpu")
    meta_count = int(r8["splits"]["meta_train_capabilities"])
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()
    for capability_index, (capability, capability_rows) in enumerate(zip(capabilities, rows)):
        for condition in ("BEFORE", "AFTER"):
            if condition == "BEFORE":
                controller.reset()
            else:
                load_controller_state(
                    controller,
                    unpack_controller_state(packed, meta_count + capability_index),
                )
            for start in range(0, len(capability_rows), int(settings["batch_size"])):
                batch = capability_rows[start : start + int(settings["batch_size"])]
                logits = controller.logits(
                    [str(row["prompt"]) for row in batch],
                    [int(row["start"]) for row in batch],
                    [[int(value) for value in row["program"]] for row in batch],
                )
                probabilities = host.canonical_probabilities(logits)
                predictions = logits.argmax(dim=-1)
                for row, probability, prediction in zip(batch, probabilities, predictions):
                    observations.append(
                        {
                            "capability_id": capability.capability_id,
                            "condition": condition,
                            "row_id": row["row_id"],
                            "prompt_sha256": row["prompt_sha256"],
                            "prediction_token_id": int(prediction.cpu()),
                            "canonical_prediction": canonical_prediction(
                                int(prediction.cpu()), host.target_token_ids
                            ),
                            "canonical_output_probabilities": [
                                float(value) for value in probability.cpu()
                            ],
                        }
                    )
    host.verify_frozen()
    receipt = {
        "model_id": host.spec.model_id,
        "revision": host.spec.revision,
        "target_token_ids": list(host.target_token_ids),
        "model_state_sha256_before": host.model_state_sha256,
        "model_state_sha256_after": module_sha256(host.model),
        "controller_learned_parameters": sum(value.numel() for value in controller.parameters()),
        "recipient_phase_optimizer_steps": 0,
        "rows": len(observations),
        "wall_seconds": time.perf_counter() - started,
    }
    del controller, host, packed
    gc.collect()
    torch.cuda.empty_cache()
    return observations, receipt


def _packages(
    output: Path,
    config: Mapping[str, Any],
    latents: Mapping[str, torch.Tensor],
    capabilities: Sequence[Any],
) -> dict[str, Any]:
    reference = config["r8_reference"]
    provenance = {
        "source_receipt_sha256": reference["source_receipt_sha256"],
        "extraction_receipt_sha256": reference["extraction_receipt_sha256"],
    }
    directory = output / "packages"
    before = write_package_once(directory, latents["before"], provenance)
    after = []
    for index, capability in enumerate(capabilities):
        item = write_package_once(directory, latents["development_after"][index], provenance)
        item["capability_id"] = capability.capability_id
        after.append(item)
    return {"before": before, "after": after}


def _load_manifest_packages(
    output: Path, manifest: Mapping[str, Any]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    _, before = load_package(output / "packages" / str(manifest["before"]["path"]))
    after = [load_package(output / "packages" / str(item["path"]))[1] for item in manifest["after"]]
    return before, after


def _condition_latents(
    after: torch.Tensor,
    before: torch.Tensor,
    wrong: torch.Tensor,
    capability_id: str,
) -> dict[str, torch.Tensor | None]:
    return {
        "BASE": None,
        "AFTER": after,
        "BEFORE": before,
        "WRONG": wrong,
        "ZERO": torch.zeros_like(after),
        "RANDOM": _random_latent(after, capability_id + "-r10"),
        "SHUFFLED": _shuffled_latent(after, capability_id + "-r10"),
        "REMOVED": None,
        "INTERPRETER_REMOVED": None,
        "RESTORED": after,
    }


@torch.inference_mode()
def _host_observations(
    host_key: str,
    config: Mapping[str, Any],
    output: Path,
    manifest: Mapping[str, Any],
    capabilities: Sequence[Any],
    rows: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = config["runtime"]
    host = R10FrozenNeuralHost(SPECS[host_key], device="cuda")
    state_before = host.model_state_sha256
    vm = CanonicalTransitionVM()
    before, after_packages = _load_manifest_packages(output, manifest)
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()
    for capability_index, (capability, capability_rows) in enumerate(zip(capabilities, rows)):
        after = after_packages[capability_index]
        wrong_index = (capability_index + 1) % len(after_packages)
        conditions = _condition_latents(
            after, before, after_packages[wrong_index], capability.capability_id
        )
        if tuple(conditions) != tuple(config["public_matrix"]["conditions"]):
            raise R10RunError("condition order changed")
        after_sha = manifest["after"][capability_index]["sha256"]
        wrong_sha = manifest["after"][wrong_index]["sha256"]
        package_hashes = {
            "BASE": None,
            "AFTER": after_sha,
            "BEFORE": manifest["before"]["sha256"],
            "WRONG": wrong_sha,
            "ZERO": "CONTROL_ZERO",
            "RANDOM": "CONTROL_RANDOM",
            "SHUFFLED": "CONTROL_SHUFFLED",
            "REMOVED": None,
            "INTERPRETER_REMOVED": after_sha,
            "RESTORED": after_sha,
        }
        for start in range(0, len(capability_rows), int(settings["batch_size"])):
            batch = capability_rows[start : start + int(settings["batch_size"])]
            prompts = [str(row["prompt"]) for row in batch]
            base_logits, _ = host.logits(prompts, prefix=None)
            for condition, latent in conditions.items():
                interpreter_active = condition not in {
                    "BASE",
                    "REMOVED",
                    "INTERPRETER_REMOVED",
                }
                if interpreter_active:
                    distribution = vm.execute(latent, prompts)
                    logits = apply_host_codec(
                        base_logits,
                        distribution,
                        host.target_token_ids,
                        margin=float(settings["host_logit_margin"]),
                    )
                else:
                    logits = base_logits
                    distribution = host.canonical_probabilities(logits).cpu()
                canonical_probabilities = host.canonical_probabilities(logits)
                predictions = logits.argmax(dim=-1)
                for row, prediction, probability, vm_probability in zip(
                    batch, predictions, canonical_probabilities, distribution
                ):
                    token_id = int(prediction.cpu())
                    canonical = canonical_prediction(token_id, host.target_token_ids)
                    if canonical is None:
                        output_bytes = host.tokenizer.decode([token_id]).encode("utf-8")
                    else:
                        output_bytes = str(canonical).encode("utf-8")
                    observations.append(
                        {
                            "host": host_key,
                            "capability_id": capability.capability_id,
                            "condition": condition,
                            "row_id": row["row_id"],
                            "prompt_sha256": row["prompt_sha256"],
                            "package_sha256": package_hashes[condition],
                            "interpreter_active": interpreter_active,
                            "prediction_token_id": token_id,
                            "canonical_prediction": canonical,
                            "canonical_output_utf8_hex": output_bytes.hex(),
                            "canonical_output_probabilities": [
                                float(value) for value in probability.cpu()
                            ],
                            "vm_output_probabilities": [
                                float(value) for value in vm_probability.cpu()
                            ],
                        }
                    )
    host.verify_frozen()
    state_after = module_sha256(host.model)
    if state_before != state_after:
        raise R10RunError(f"recipient model changed: {host_key}")
    receipt = {
        "host": host_key,
        "model_id": host.spec.model_id,
        "revision": host.spec.revision,
        "architecture_family": host.spec.architecture_family,
        "target_token_ids": list(host.target_token_ids),
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "recipient_optimizer_steps": 0,
        "interpreter_learned_parameters": vm.learned_parameters,
        "source_model_loaded": False,
        "rows": len(observations),
        "wall_seconds": time.perf_counter() - started,
    }
    del host, vm, before, after_packages
    gc.collect()
    torch.cuda.empty_cache()
    return observations, receipt


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R10RunError(f"immutable output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = _json(config_path)
    if config.get("status") != "PREREGISTERED_BEFORE_EXECUTION":
        raise R10RunError("R10 preregistration is not frozen")
    bindings = _bind_inputs(root, config)
    r8 = _json(_resolve(root, str(config["r8_reference"]["config"])))
    capabilities, rows = _evaluation_rows(config, r8)
    latents = load_file(
        str(_resolve(root, str(config["r8_reference"]["canonical_latents"]))),
        device="cpu",
    )
    output.mkdir(parents=True)
    manifest = _packages(output, config, latents, capabilities)
    _disable_network()
    source_started = time.perf_counter()
    source_rows, source_receipt = _source_observations(
        config,
        r8,
        _resolve(root, str(config["r8_reference"]["source_states"])),
        capabilities,
        rows,
    )
    source_finished = time.perf_counter()
    source_path = output / "source_observations.jsonl"
    _write_jsonl_once(source_path, source_rows)
    recipient_rows: list[dict[str, Any]] = []
    host_receipts = []
    recipient_started = time.perf_counter()
    for host_key in config["public_matrix"]["hosts"]:
        host_rows, host_receipt = _host_observations(
            str(host_key), config, output, manifest, capabilities, rows
        )
        recipient_rows.extend(host_rows)
        host_receipts.append(host_receipt)
        print(json.dumps(host_receipt, sort_keys=True), flush=True)
    recipient_finished = time.perf_counter()
    recipient_path = output / "recipient_observations.jsonl"
    _write_jsonl_once(recipient_path, recipient_rows)
    receipt = {
        "format": "abi-copy-paste-r10-run/1",
        "config_sha256": sha256_file(config_path),
        "bindings": bindings,
        "claim_target": "ABI-C4",
        "claim_ceiling": "RUNTIME_OWNED_COPY_PASTE_EXECUTION_ONLY",
        "packages": manifest,
        "source": source_receipt,
        "source_execution": {
            "started": source_started,
            "finished": source_finished,
            "observations": {
                "path": source_path.name,
                "sha256": sha256_file(source_path),
                "rows": len(source_rows),
            },
        },
        "recipient_execution": {
            "started": recipient_started,
            "finished": recipient_finished,
            "source_finished_before_recipient_started": source_finished <= recipient_started,
            "physical_source_file_absence_claimed": False,
            "hosts": host_receipts,
            "observations": {
                "path": recipient_path.name,
                "sha256": sha256_file(recipient_path),
                "rows": len(recipient_rows),
            },
        },
        "hardware": {
            "device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    _write_once(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = run(Path(args.config).resolve(), Path(args.output).resolve())
    except (OSError, ValueError, CopyPasteRuntimeError, R10RunError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
