"""Freeze capability-blind native residual codecs before R11 held-out reveal."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from experiments.copy_paste_r10.run import R10FrozenNeuralHost
from experiments.native_transfer_r8.capability_generator import canonical_json_bytes
from experiments.native_transfer_r8.native_host import SPECS, module_sha256

from .core import R11Error, sha256_bytes, sha256_file


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R11Error(f"expected JSON object: {path}")
    return value


def _bind(root: Path, config: dict[str, Any]) -> dict[str, str]:
    bindings = config.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise R11Error("R11 codec-freeze bindings missing")
    actual = {}
    for relative, expected in bindings.items():
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise R11Error(f"registered path escapes repository: {relative}") from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise R11Error(f"registered codec-freeze binding changed: {relative}")
        actual[str(relative)] = str(expected)
    return actual


def freeze(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R11Error(f"immutable codec output exists: {output}")
    config = _json(config_path)
    if config.get("status") != "PREREGISTERED_BEFORE_CODEC_FREEZE":
        raise R11Error("R11 codec preregistration is not frozen")
    root = Path(__file__).resolve().parents[2]
    bindings = _bind(root, config)
    tensors = {}
    hosts = []
    for host_key in config["hosts"]:
        host = R10FrozenNeuralHost(SPECS[str(host_key)], device="cuda")
        state_before = host.model_state_sha256
        weight = host.model.get_output_embeddings().weight.detach().float()
        indices = torch.tensor(host.target_token_ids, device=weight.device)
        codec = weight.index_select(0, indices)
        codec = codec / codec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        codec = codec.cpu().float().contiguous()
        digest = sha256_bytes(codec.numpy().tobytes())
        # Generic pre-capability proof: each isolated codec row must already be
        # classified as its corresponding native output token by the frozen head.
        logits = torch.nn.functional.linear(
            codec.to(host.device, dtype=host.model.get_output_embeddings().weight.dtype),
            host.model.get_output_embeddings().weight,
        ).float()
        predictions = logits.argmax(dim=-1).cpu().tolist()
        if predictions != host.target_token_ids:
            raise R11Error(f"pre-capability native codec is not exact: {host_key}")
        host.verify_frozen()
        state_after = module_sha256(host.model)
        if state_before != state_after:
            raise R11Error(f"host changed during codec freeze: {host_key}")
        tensors[str(host_key)] = codec
        hosts.append(
            {
                "host": host_key,
                "model_id": host.spec.model_id,
                "revision": host.spec.revision,
                "architecture_family": host.spec.architecture_family,
                "model_state_sha256_before": state_before,
                "model_state_sha256_after": state_after,
                "target_token_ids": list(host.target_token_ids),
                "codec_shape": list(codec.shape),
                "codec_sha256": digest,
                "isolated_native_head_accuracy": 1.0,
                "capability_examples_seen": 0,
                "capability_ids_seen": 0,
                "learned_parameters": 0,
                "optimizer_steps": 0,
                "residual_scale": float(config["codec_scales"][host_key]),
            }
        )
        del host, weight, codec, logits
        gc.collect()
        torch.cuda.empty_cache()
    output.mkdir(parents=True)
    tensor_path = output / "host_codecs.safetensors"
    save_file(tensors, str(tensor_path))
    receipt = {
        "format": "abi-native-neural-isa-r11-codec-freeze/1",
        "config_sha256": sha256_file(config_path),
        "bindings": bindings,
        "heldout_seed_commitment": config["heldout_seed_commitment"],
        "heldout_seed_revealed": False,
        "host_codecs": {
            "path": tensor_path.name,
            "sha256": sha256_file(tensor_path),
            "bytes": tensor_path.stat().st_size,
        },
        "hosts": hosts,
        "network_used": False,
        "hardware": {
            "device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    (output / "receipt.json").write_bytes(
        json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = freeze(Path(args.config).resolve(), Path(args.output).resolve())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
