"""Frozen atomic extraction from public neural transition source states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .capability_generator import MODULUS, OPERATORS, canonical_json_bytes, render_prompt
from .extract_capability import extractor_spec
from .native_host import SPECS, FrozenNeuralHost, NativeHostError, sha256_file
from .source_transition import (
    NeuralTransitionSource,
    SourceTransitionError,
    ensure_source_base_frozen,
    load_controller_state,
    unpack_controller_state,
)


class TransitionExtractionError(RuntimeError):
    """Raised when transition source extraction is not immutable."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TransitionExtractionError(f"expected JSON object: {path}")
    return value


@torch.inference_mode()
def extract_atomic_latent(controller: NeuralTransitionSource) -> torch.Tensor:
    starts = [start for _operator in range(len(OPERATORS)) for start in range(MODULUS)]
    programs = [[operator] for operator in range(len(OPERATORS)) for _ in range(MODULUS)]
    prompts = [
        render_prompt(start, program)
        for start, program in zip(starts, programs)
    ]
    logits = controller.logits(prompts, starts, programs)
    latent = controller.host.canonical_probabilities(logits).reshape(3, 8, 8)
    if not torch.isfinite(latent).all():
        raise TransitionExtractionError("transition canonical latent is non-finite")
    return latent.detach().cpu().float().contiguous()


def extract_meta(config_path: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise TransitionExtractionError(f"immutable transition extraction exists: {output}")
    config = _json(config_path)
    source_receipt_path = source_dir / "receipt.json"
    source_receipt = _json(source_receipt_path)
    state_path = source_dir / source_receipt["states"]["path"]
    if (
        source_receipt.get("config_sha256") != sha256_file(config_path)
        or source_receipt["states"]["sha256"] != sha256_file(state_path)
    ):
        raise TransitionExtractionError("source transition receipt binding changed")
    packed = load_file(str(state_path), device="cpu")
    count = int(config["splits"]["meta_train_capabilities"]) + int(
        config["splits"]["development_capabilities"]
    )
    host = FrozenNeuralHost(SPECS["source"], device=config["training"]["device"])
    controller = NeuralTransitionSource(host, seed=int(config["training"]["seed"])).to(
        host.device
    )
    before = extract_atomic_latent(controller)
    latents = []
    state_hashes = []
    for index in range(count):
        load_controller_state(controller, unpack_controller_state(packed, index))
        latents.append(extract_atomic_latent(controller))
        state_hashes.append(controller.state_sha256())
    ensure_source_base_frozen(host, host.model_state_sha256)
    meta_count = int(config["splits"]["meta_train_capabilities"])
    output.mkdir(parents=True)
    tensor_path = output / "meta_canonical_latents.safetensors"
    save_file(
        {
            "before": before,
            "meta_after": torch.stack(latents[:meta_count]).contiguous(),
            "development_after": torch.stack(latents[meta_count:]).contiguous(),
        },
        str(tensor_path),
    )
    receipt = {
        "format": "abi-native-transfer-r8-meta-transition-extraction/1",
        "config_sha256": sha256_file(config_path),
        "source_receipt_sha256": sha256_file(source_receipt_path),
        "source_states_sha256": sha256_file(state_path),
        "source_state_sha256s": state_hashes,
        "source_base_model_state_sha256_before": host.model_state_sha256,
        "source_base_model_state_sha256_after": host.model_state_sha256,
        "extractor": extractor_spec(),
        "latents": {
            "path": tensor_path.name,
            "sha256": sha256_file(tensor_path),
            "before_shape": list(before.shape),
            "meta_after_shape": [meta_count, 3, 8, 8],
            "development_after_shape": [count - meta_count, 3, 8, 8],
        },
        "heldout_accessed": False,
        "learned_extractor_parameters": 0,
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    (output / "receipt.json").write_bytes(
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = extract_meta(
            Path(args.config).resolve(),
            Path(args.source_dir).resolve(),
            Path(args.output).resolve(),
        )
    except (TransitionExtractionError, SourceTransitionError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
