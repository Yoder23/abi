"""Apply the frozen R8 atomic extractor to source LoRA capability states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .capability_generator import canonical_json_bytes
from .extract_capability import extract_atomic_latent, extractor_spec
from .native_host import SPECS, FrozenNeuralHost, NativeHostError, sha256_file
from .source_adapter import (
    SourceAdapterError,
    SourceLoRASet,
    unpack_capability_state,
)


class LoRAExtractionError(RuntimeError):
    """Raised when source LoRA extraction custody fails."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LoRAExtractionError(f"expected JSON object: {path}")
    return value


def extract_meta(config_path: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise LoRAExtractionError(f"immutable LoRA meta extraction exists: {output}")
    config = _json(config_path)
    receipt_path = source_dir / "receipt.json"
    source_receipt = _json(receipt_path)
    adapter_path = source_dir / source_receipt["adapters"]["path"]
    if (
        source_receipt.get("config_sha256") != sha256_file(config_path)
        or source_receipt["adapters"]["sha256"] != sha256_file(adapter_path)
    ):
        raise LoRAExtractionError("source LoRA receipt binding changed")
    packed = load_file(str(adapter_path), device="cpu")
    count = int(config["splits"]["meta_train_capabilities"]) + int(
        config["splits"]["development_capabilities"]
    )
    host = FrozenNeuralHost(SPECS["source"], device=config["training"]["device"])
    adapters = SourceLoRASet(
        host.model,
        rank=int(config["training"]["source_lora_rank"]),
        expected_base_sha256=host.model_state_sha256,
    )
    before = extract_atomic_latent(host, None)
    latents = []
    state_hashes = []
    for index in range(count):
        adapters.load_state(unpack_capability_state(packed, index))
        adapters.verify_base_frozen()
        latents.append(extract_atomic_latent(host, None))
        state_hashes.append(adapters.state_sha256())
    meta_count = int(config["splits"]["meta_train_capabilities"])
    output.mkdir(parents=True)
    tensor_path = output / "meta_canonical_latents.safetensors"
    save_file(
        {
            "before": before.contiguous(),
            "meta_after": torch.stack(latents[:meta_count]).contiguous(),
            "development_after": torch.stack(latents[meta_count:]).contiguous(),
        },
        str(tensor_path),
    )
    receipt = {
        "format": "abi-native-transfer-r8-meta-lora-extraction/1",
        "config_sha256": sha256_file(config_path),
        "source_receipt_sha256": sha256_file(receipt_path),
        "source_adapter_sha256": sha256_file(adapter_path),
        "source_adapter_state_sha256s": state_hashes,
        "source_base_model_state_sha256_before": host.model_state_sha256,
        "source_base_model_state_sha256_after": adapters.base_state_sha256(),
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
    except (LoRAExtractionError, SourceAdapterError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
