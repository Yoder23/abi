"""Frozen atomic-logit extractor and immutable R8 package format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .capability_generator import MODULUS, OPERATORS, canonical_json_bytes, render_prompt
from .native_host import SPECS, FrozenNeuralHost, NativeHostError, sha256_file

PACKAGE_FORMAT = "abi-native-neural-capability-package/1"
EXTRACTOR_FORMAT = "abi-native-transfer-r8-atomic-logit-extractor/1"
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


class ExtractionError(RuntimeError):
    """Raised when an extractor or package invariant changes."""


def extractor_spec() -> dict[str, Any]:
    value = {
        "format": EXTRACTOR_FORMAT,
        "family": "opaque_modular_micro_language/1",
        "probe_order": [
            {"operation_slot": operator, "input_slot": start}
            for operator in range(len(OPERATORS))
            for start in range(MODULUS)
        ],
        "output_basis": list(range(MODULUS)),
        "normalization": "softmax_over_canonical_output_token_logits",
        "learned_parameters": 0,
    }
    value["extractor_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def extractor_sha256() -> str:
    return str(extractor_spec()["extractor_sha256"])


@torch.inference_mode()
def extract_atomic_latent(
    host: FrozenNeuralHost,
    prefix: torch.Tensor | None,
    *,
    batch_size: int = 24,
) -> torch.Tensor:
    prompts = [
        render_prompt(start, (operator,))
        for operator in range(len(OPERATORS))
        for start in range(MODULUS)
    ]
    chunks = []
    for offset in range(0, len(prompts), batch_size):
        batch = prompts[offset : offset + batch_size]
        selected_prefix = prefix
        if prefix is not None and prefix.ndim == 3:
            selected_prefix = prefix[offset : offset + len(batch)]
        logits, _ = host.logits(batch, prefix=selected_prefix)
        chunks.append(host.canonical_probabilities(logits).cpu())
    latent = torch.cat(chunks, dim=0).reshape(len(OPERATORS), MODULUS, MODULUS)
    if not torch.isfinite(latent).all() or tuple(latent.shape) != (3, 8, 8):
        raise ExtractionError("canonical latent is invalid")
    return latent.float().contiguous()


def _package_core(
    latent: torch.Tensor,
    *,
    capability_id: str,
    reveal_commitment_sha256: str,
    source_before_sha256: str,
    source_after_sha256: str,
) -> dict[str, Any]:
    if tuple(latent.shape) != (len(OPERATORS), MODULUS, MODULUS):
        raise ExtractionError("package latent shape changed")
    payload = latent.detach().cpu().float().contiguous().numpy().tobytes()
    return {
        "format": PACKAGE_FORMAT,
        "capability_id": capability_id,
        "family": "opaque_modular_micro_language/1",
        "latent_schema": {
            "shape": [len(OPERATORS), MODULUS, MODULUS],
            "dtype": "float32-little-endian",
            "row_normalization": "probability_simplex",
        },
        "latent_hex": payload.hex(),
        "latent_sha256": hashlib.sha256(payload).hexdigest(),
        "extractor_sha256": extractor_sha256(),
        "reveal_commitment_sha256": reveal_commitment_sha256,
        "source_before_state_sha256": source_before_sha256,
        "source_after_state_sha256": source_after_sha256,
        "host_specific_payloads": 0,
        "executable_payloads": 0,
        "test_rows": 0,
        "answers": 0,
    }


def write_package_once(
    path: Path,
    latent: torch.Tensor,
    *,
    capability_id: str,
    reveal_commitment_sha256: str,
    source_before_sha256: str,
    source_after_sha256: str,
) -> dict[str, Any]:
    if path.exists():
        raise ExtractionError(f"immutable package exists: {path}")
    value = _package_core(
        latent,
        capability_id=capability_id,
        reveal_commitment_sha256=reveal_commitment_sha256,
        source_before_sha256=source_before_sha256,
        source_after_sha256=source_after_sha256,
    )
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    lowered = payload.decode("utf-8").casefold()
    matches = [term for term in FORBIDDEN_PACKAGE_TERMS if term in lowered]
    if matches:
        raise ExtractionError(f"host-private term entered package: {matches}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "latent_sha256": value["latent_sha256"],
        "forbidden_term_matches": [],
    }


def load_package(path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("format") != PACKAGE_FORMAT:
        raise ExtractionError("capability package format changed")
    evidence = dict(value)
    stored = evidence.pop("evidence_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(evidence)).hexdigest():
        raise ExtractionError("capability package evidence hash changed")
    lowered = raw.decode("utf-8").casefold()
    matches = [term for term in FORBIDDEN_PACKAGE_TERMS if term in lowered]
    if matches or value.get("host_specific_payloads") != 0 or value.get("executable_payloads") != 0:
        raise ExtractionError(f"capability package is not representation-neutral: {matches}")
    payload = bytes.fromhex(str(value["latent_hex"]))
    if hashlib.sha256(payload).hexdigest() != value.get("latent_sha256"):
        raise ExtractionError("capability latent bytes changed")
    latent = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
    if latent.numel() != len(OPERATORS) * MODULUS * MODULUS:
        raise ExtractionError("capability latent size changed")
    latent = latent.reshape(len(OPERATORS), MODULUS, MODULUS)
    if not torch.isfinite(latent).all():
        raise ExtractionError("capability latent is non-finite")
    return value, latent


def extract_meta(config_path: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ExtractionError(f"immutable meta extraction exists: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_receipt = json.loads((source_dir / "receipt.json").read_text(encoding="utf-8"))
    prefix_path = source_dir / source_receipt["prefixes"]["path"]
    if sha256_file(prefix_path) != source_receipt["prefixes"]["sha256"]:
        raise ExtractionError("meta source prefixes changed")
    prefixes = load_file(str(prefix_path), device="cpu")["prefixes"]
    meta_count = int(config["splits"]["meta_train_capabilities"])
    development_count = int(config["splits"]["development_capabilities"])
    count = meta_count + development_count
    if prefixes.shape[0] != count:
        raise ExtractionError("public source prefix count changed")
    host = FrozenNeuralHost(SPECS["source"], device=config["training"]["device"])
    before = extract_atomic_latent(host, None)
    latents = []
    for index in range(count):
        latents.append(extract_atomic_latent(host, prefixes[index].to(host.device)))
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
        "format": "abi-native-transfer-r8-meta-extraction/1",
        "config_sha256": sha256_file(config_path),
        "source_receipt_sha256": sha256_file(source_dir / "receipt.json"),
        "extractor": extractor_spec(),
        "latents": {
            "path": tensor_path.name,
            "sha256": sha256_file(tensor_path),
            "before_shape": list(before.shape),
            "meta_after_shape": [meta_count, len(OPERATORS), MODULUS, MODULUS],
            "development_after_shape": [development_count, len(OPERATORS), MODULUS, MODULUS],
        },
        "heldout_accessed": False,
        "learned_extractor_parameters": 0,
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    (output / "receipt.json").write_bytes(
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    del host
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    except (ExtractionError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
