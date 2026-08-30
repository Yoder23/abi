"""Label-free native recipient worker for R8.

This module deliberately does not import the capability generator or scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from .native_host import (
    SPECS,
    CanonicalLatentBridge,
    FrozenNeuralHost,
    NativeHostError,
    canonical_json_bytes,
    module_sha256,
    sha256_file,
)

PACKAGE_FORMAT = "abi-native-neural-capability-package/1"
PACKAGE_SHAPE = (3, 8, 8)
FORBIDDEN_PACKAGE_TERMS = (
    "qwen",
    "pythia",
    "gpt",
    "t5",
    "llama",
    "tokenizer",
    "hidden_width",
    "layer_count",
    "recipient",
    "model_id",
)
CONDITIONS = (
    "BASE",
    "BEFORE",
    "AFTER",
    "PERMUTED_TEACHER_DELTA",
    "ZERO",
    "RANDOM",
    "SHUFFLED",
    "WRONG",
    "REMOVED",
    "BRIDGE_REMOVED",
    "MODEL_REMOVED",
    "RUNTIME_ONLY",
)


class RecipientWorkerError(RuntimeError):
    """Raised whenever label-free recipient execution cannot be trusted."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecipientWorkerError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_bytes().splitlines()
    if not lines:
        raise RecipientWorkerError(f"empty worker input: {path}")
    rows = [json.loads(line) for line in lines]
    forbidden = {"answer", "offsets", "seed", "program", "start", "label"}
    if any(not isinstance(row, dict) or forbidden.intersection(row) for row in rows):
        raise RecipientWorkerError(f"private evaluator field entered worker input: {path}")
    return rows


def _evidence_valid(value: Mapping[str, Any]) -> bool:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    return stored == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_package(path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("format") != PACKAGE_FORMAT:
        raise RecipientWorkerError(f"package format changed: {path}")
    if not _evidence_valid(value):
        raise RecipientWorkerError(f"package evidence hash changed: {path}")
    lowered = raw.decode("utf-8").casefold()
    matches = [term for term in FORBIDDEN_PACKAGE_TERMS if term in lowered]
    if matches or value.get("host_specific_payloads") != 0 or value.get("executable_payloads") != 0:
        raise RecipientWorkerError(f"host-private or executable package payload: {matches}")
    payload = bytes.fromhex(str(value["latent_hex"]))
    if hashlib.sha256(payload).hexdigest() != value.get("latent_sha256"):
        raise RecipientWorkerError(f"package latent changed: {path}")
    latent = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
    if latent.numel() != 3 * 8 * 8:
        raise RecipientWorkerError("package latent shape changed")
    return value, latent.reshape(PACKAGE_SHAPE)


def _disable_network() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)

    class DeniedSocket(socket.socket):
        def connect(self, *_args: Any, **_kwargs: Any) -> None:
            raise RecipientWorkerError("network is disabled during R8 evaluation")

        def connect_ex(self, *_args: Any, **_kwargs: Any) -> int:
            raise RecipientWorkerError("network is disabled during R8 evaluation")

    socket.socket = DeniedSocket


def _random_latent(reference: torch.Tensor, capability_id: str) -> torch.Tensor:
    generator = random.Random(
        hashlib.sha256(f"r8-random:{capability_id}".encode("utf-8")).hexdigest()
    )
    values = torch.tensor(
        [generator.random() for _ in range(reference.numel())], dtype=torch.float32
    ).reshape(reference.shape)
    return values / values.sum(dim=-1, keepdim=True)


def _shuffled_latent(reference: torch.Tensor, capability_id: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    seed = int.from_bytes(
        hashlib.sha256(f"r8-shuffle:{capability_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    generator.manual_seed(seed)
    flat = reference.reshape(-1, 8)
    order = torch.randperm(flat.shape[0], generator=generator)
    columns = torch.randperm(8, generator=generator)
    return flat.index_select(0, order).index_select(1, columns).reshape(reference.shape)


def _condition_latent(
    condition: str,
    *,
    before: torch.Tensor,
    after: torch.Tensor,
    permuted: torch.Tensor,
    wrong: torch.Tensor,
    capability_id: str,
) -> torch.Tensor | None:
    if condition in {"BASE", "REMOVED", "BRIDGE_REMOVED", "MODEL_REMOVED", "RUNTIME_ONLY"}:
        return None
    if condition == "BEFORE":
        return before
    if condition == "AFTER":
        return after
    if condition == "PERMUTED_TEACHER_DELTA":
        return permuted
    if condition == "ZERO":
        return torch.zeros_like(after)
    if condition == "RANDOM":
        return _random_latent(after, capability_id)
    if condition == "SHUFFLED":
        return _shuffled_latent(after, capability_id)
    if condition == "WRONG":
        return wrong
    raise RecipientWorkerError(f"unknown condition: {condition}")


def run(
    config_path: Path,
    campaign_root: Path,
    source_dir: Path,
    output: Path,
    *,
    host_key: str,
) -> dict[str, Any]:
    if output.exists():
        raise RecipientWorkerError(f"immutable recipient output exists: {output}")
    if host_key not in {"qwen2", "pythia", "t5"}:
        raise RecipientWorkerError(f"unregistered host: {host_key}")
    config = _json(config_path)
    freeze_path = campaign_root / "freeze_receipt.json"
    freeze = _json(freeze_path)
    bridge_dir = campaign_root / "pre_reveal/bridges" / host_key
    bridge_receipt = _json(bridge_dir / "receipt.json")
    bridge_path = bridge_dir / bridge_receipt["bridge"]["path"]
    if (
        not _evidence_valid(freeze)
        or not _evidence_valid(bridge_receipt)
        or sha256_file(bridge_path) != bridge_receipt["bridge"]["sha256"]
        or freeze["bridges"][host_key]["bridge_sha256"] != sha256_file(bridge_path)
    ):
        raise RecipientWorkerError("frozen bridge or receipt identity changed")
    package_dirs = sorted((source_dir / "packages").iterdir(), key=lambda path: path.name)
    if len(package_dirs) != int(config["splits"]["heldout_capabilities"]):
        raise RecipientWorkerError("held-out package count changed")
    packages = {}
    package_hashes = {}
    for directory in package_dirs:
        before_doc, before = _load_package(directory / "before.abipkg")
        after_doc, after = _load_package(directory / "after.abipkg")
        permuted_doc, permuted = _load_package(directory / "permuted_teacher_delta.abipkg")
        capability_id = directory.name
        if {
            str(before_doc["capability_id"]),
            str(after_doc["capability_id"]),
            str(permuted_doc["capability_id"]),
        } != {capability_id}:
            raise RecipientWorkerError("package capability identity changed")
        packages[capability_id] = {
            "before": before,
            "after": after,
            "permuted": permuted,
        }
        package_hashes[capability_id] = {
            name: sha256_file(directory / f"{name}.abipkg") for name in ("before", "after")
        }
        package_hashes[capability_id]["permuted_teacher_delta"] = sha256_file(
            directory / "permuted_teacher_delta.abipkg"
        )
    host = FrozenNeuralHost(SPECS[host_key], device=config["training"]["device"])
    bridge = CanonicalLatentBridge(host).to(host.device)
    bridge.load_state_dict(load_file(str(bridge_path), device=str(host.device)), strict=True)
    bridge.freeze()
    if module_sha256(bridge) != bridge_receipt["bridge"]["state_sha256"]:
        raise RecipientWorkerError("loaded bridge state differs from frozen receipt")
    _disable_network()
    rows_out = []
    batch_size = int(config["training"]["batch_size"])
    capability_ids = sorted(packages)
    started = time.perf_counter()
    for capability_index, capability_id in enumerate(capability_ids):
        rows = _jsonl(source_dir / "worker_inputs" / f"{capability_id}.jsonl")
        current = packages[capability_id]
        wrong = packages[capability_ids[(capability_index + 1) % len(capability_ids)]]["after"]
        for condition in CONDITIONS:
            latent = _condition_latent(
                condition,
                before=current["before"],
                after=current["after"],
                permuted=current["permuted"],
                wrong=wrong,
                capability_id=capability_id,
            )
            if condition == "BASE":
                condition_package_sha256 = None
            elif condition == "BEFORE":
                condition_package_sha256 = package_hashes[capability_id]["before"]
            elif condition == "PERMUTED_TEACHER_DELTA":
                condition_package_sha256 = package_hashes[capability_id][
                    "permuted_teacher_delta"
                ]
            elif condition == "WRONG":
                wrong_id = capability_ids[(capability_index + 1) % len(capability_ids)]
                condition_package_sha256 = package_hashes[wrong_id]["after"]
            else:
                condition_package_sha256 = package_hashes[capability_id]["after"]
            if condition in {"BRIDGE_REMOVED", "MODEL_REMOVED", "RUNTIME_ONLY"}:
                for row in rows:
                    rows_out.append(
                        {
                            "host": host_key,
                            "architecture_family": host.spec.architecture_family,
                            "capability_id": capability_id,
                            "row_id": row["row_id"],
                            "prompt_sha256": row["prompt_sha256"],
                            "condition": condition,
                            "prediction_token_id": None,
                            "prediction_text": None,
                            "canonical_output_probabilities": None,
                            "exception_type": f"{condition}_INTENTIONALLY_ABSENT",
                            "package_sha256": condition_package_sha256,
                        }
                    )
                continue
            prefix = None if latent is None else bridge(latent.to(host.device)).squeeze(0)
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                with torch.inference_mode():
                    logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=prefix)
                    predicted = logits.argmax(dim=-1)
                    probabilities = host.canonical_probabilities(logits)
                for row, token_id, probability in zip(batch, predicted, probabilities):
                    token = int(token_id.item())
                    rows_out.append(
                        {
                            "host": host_key,
                            "architecture_family": host.spec.architecture_family,
                            "capability_id": capability_id,
                            "row_id": row["row_id"],
                            "prompt_sha256": row["prompt_sha256"],
                            "condition": condition,
                            "prediction_token_id": token,
                            "prediction_text": host.tokenizer.decode([token]),
                            "canonical_output_probabilities": [
                                float(value) for value in probability.cpu().tolist()
                            ],
                            "exception_type": None,
                            "package_sha256": condition_package_sha256,
                        }
                    )
    bridge.verify_frozen()
    host.verify_frozen()
    output.mkdir(parents=True)
    raw_path = output / "observations.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows_out))
    manifest = {
        "format": "abi-native-transfer-r8-recipient-worker/1",
        "status": "RAW_LABEL_FREE_EXECUTION_COMPLETE",
        "host": host_key,
        "architecture_family": host.spec.architecture_family,
        "revision": host.spec.revision,
        "config_sha256": sha256_file(config_path),
        "freeze_receipt_sha256": sha256_file(freeze_path),
        "bridge_receipt_sha256": sha256_file(bridge_dir / "receipt.json"),
        "bridge_sha256_before": sha256_file(bridge_path),
        "bridge_sha256_after": sha256_file(bridge_path),
        "host_snapshot_inventory": host.inventory,
        "host_model_state_sha256_before": host.model_state_sha256,
        "host_model_state_sha256_after": host.model_state_sha256,
        "target_token_ids": host.target_token_ids,
        "packages": package_hashes,
        "conditions": list(CONDITIONS),
        "rows": len(rows_out),
        "observations_sha256": sha256_file(raw_path),
        "recipient_parameters_trainable": 0,
        "recipient_optimizer_steps": 0,
        "bridge_optimizer_steps_after_reveal": 0,
        "teacher_loaded": False,
        "generator_imported": False,
        "test_labels_available": False,
        "network_disabled_in_process": True,
        "physical_filesystem_isolation": False,
        "wall_seconds": time.perf_counter() - started,
    }
    manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    (output / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    del host, bridge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", required=True, choices=("qwen2", "pythia", "t5"))
    args = parser.parse_args()
    try:
        value = run(
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
            Path(args.source_dir).resolve(),
            Path(args.output).resolve(),
            host_key=args.host,
        )
    except (RecipientWorkerError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
