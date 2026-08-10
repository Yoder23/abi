"""Assemble the immutable teacher-free routed v15 English-core candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from . import capability_compiler_phase3_routed_v15_progressive_extract as progressive
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-routed-v15-artifact-assembly/1"
ARTIFACT_FORMAT = "layercake-routed-sparse-rank768-english-core/1"


def _self_hash(value: dict) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_IMMUTABLE_ROUTED_V15_ARTIFACT_ASSEMBLY"
        or protocol.get("device") != "cpu"
        or protocol.get("source_model_access") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("routed v15 artifact assembly governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"routed v15 artifact binding changed: {name}")
    if output.exists():
        raise Phase3Error("artifact output already exists")
    output.mkdir(parents=True)
    extraction, extraction_sha = progressive._load_protocol(
        root, root / protocol["extraction_protocol"]
    )
    if extraction_sha != protocol["extraction_protocol_sha256"]:
        raise Phase3Error("source-aligned extraction protocol identity changed")
    model, tokenizer, base, _, _ = progressive._instantiate(
        root, extraction, torch.device("cpu")
    )
    progressive_metadata_path = root / protocol["progressive_metadata"]["path"]
    progressive_metadata = json.loads(progressive_metadata_path.read_text(encoding="utf-8"))
    if (
        progressive_metadata.get("status") != "PASS_ROUTED_V15_LAYERS_02_31"
        or progressive_metadata.get("layers_completed") != list(range(3, 32))
    ):
        raise Phase3Error("progressive local-pass evidence changed")
    state = model.state_dict()
    progressive_root = progressive_metadata_path.parent.resolve()
    loaded_layers = []
    checkpoint_bindings = []
    with torch.no_grad():
        for row in progressive_metadata["layer_results"]:
            layer_index = int(row["layer"])
            relative = Path(row["checkpoint"]["path"])
            checkpoint_path = (progressive_root / relative).resolve()
            if not checkpoint_path.is_relative_to(progressive_root):
                raise Phase3Error("progressive checkpoint escapes evidence directory")
            expected_hash = str(row["checkpoint"]["sha256"])
            if sha256_file(checkpoint_path) != expected_hash:
                raise Phase3Error(f"progressive layer{layer_index} checkpoint changed")
            checkpoint = load_file(str(checkpoint_path), device="cpu")
            expected_keys = {
                name for name, _ in model.named_parameters()
                if name.startswith(f"layers.{layer_index}.")
            }
            if set(checkpoint) != expected_keys:
                raise Phase3Error(f"progressive layer{layer_index} tensor boundary changed")
            for name, value in checkpoint.items():
                if state[name].shape != value.shape:
                    raise Phase3Error(f"progressive layer{layer_index} tensor shape changed")
                state[name].copy_(value.to(state[name].dtype))
            loaded_layers.append(layer_index)
            checkpoint_bindings.append(
                {
                    "layer": layer_index,
                    "path": str(checkpoint_path.relative_to(root)).replace("\\", "/"),
                    "sha256": expected_hash,
                }
            )
    if loaded_layers != list(range(3, 32)):
        raise Phase3Error("assembled progressive layer order changed")
    tensors = {
        name: parameter.detach().to(torch.float16).cpu().contiguous()
        for name, parameter in model.named_parameters()
    }
    expected_keys = {name for name, _ in model.named_parameters()}
    if set(tensors) != expected_keys:
        raise Phase3Error("assembled tensor boundary is incomplete")
    parameter_count = sum(value.numel() for value in tensors.values())
    if parameter_count != int(protocol["artifact"]["expected_parameters"]):
        raise Phase3Error("assembled parameter count changed")
    if len(tensors) != int(protocol["artifact"]["expected_tensor_keys"]):
        raise Phase3Error("assembled tensor-key count changed")
    tensor_path = output / "model.safetensors"
    save_file(
        tensors,
        str(tensor_path),
        metadata={
            "format": ARTIFACT_FORMAT,
            "abi_version": protocol["artifact"]["abi_version"],
            "abi_sha256": protocol["artifact"]["abi_sha256"],
            "protocol_sha256": sha256_file(protocol_path),
        },
    )
    if tensor_path.stat().st_size > int(protocol["artifact"]["maximum_model_bytes"]):
        raise Phase3Error("assembled model exceeds the locked artifact byte ceiling")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        saved_keys = set(handle.keys())
        saved_metadata = handle.metadata()
        saved_specs = {
            name: {
                "shape": list(handle.get_slice(name).get_shape()),
                "dtype": str(tensors[name].dtype).removeprefix("torch."),
            }
            for name in sorted(saved_keys)
        }
    if saved_keys != expected_keys or saved_metadata.get("format") != ARTIFACT_FORMAT:
        raise Phase3Error("saved artifact header failed replay")
    abi_path = (root / protocol["artifact"]["abi_contract"]).resolve()
    abi_contract = json.loads(abi_path.read_text(encoding="utf-8"))
    config = {
        "format": ARTIFACT_FORMAT,
        "artifact_role": "english-core",
        "abi_version": protocol["artifact"]["abi_version"],
        "abi_sha256": protocol["artifact"]["abi_sha256"],
        "model": model.canonical_config(),
        "tokenizer": tokenizer.canonical_dict(),
        "tensor_dtype": "float16",
        "strict_state_dict": True,
        "source_transformer_blocks": 0,
        "teacher_required_at_inference": False,
    }
    config_path = output / "config.json"
    _write_immutable(config_path, canonical_json_bytes(config))
    _write_immutable(output / "abi_contract.json", abi_path.read_bytes())
    copied_parameters = sum(value.numel() for value in load_file(
        str(root / base["substrate"]["path"]), device="cpu"
    ).values())
    manifest = {
        "format": "abi-capability-compiler-routed-v15-artifact-manifest/1",
        "artifact_format": ARTIFACT_FORMAT,
        "status": "ASSEMBLED_UNVERIFIED_NOT_PROMOTED",
        "protocol_sha256": sha256_file(protocol_path),
        "files": {
            "model.safetensors": {"sha256": sha256_file(tensor_path), "bytes": tensor_path.stat().st_size},
            "config.json": {"sha256": sha256_file(config_path), "bytes": config_path.stat().st_size},
            "abi_contract.json": {
                "sha256": sha256_file(output / "abi_contract.json"),
                "bytes": (output / "abi_contract.json").stat().st_size,
            },
        },
        "tensor_specs": saved_specs,
        "tensor_keys": len(saved_specs),
        "parameters": parameter_count,
        "raw_fp16_parameter_bytes": parameter_count * 2,
        "source": {
            "model": base["source"]["model"],
            "revision": base["source"]["revision"],
            "source_parameter_count": base["source"]["parameter_count"],
            "source_blocks_in_artifact": 0,
            "teacher_present_in_artifact": False,
            "teacher_required_at_inference": False,
        },
        "imported_information": {
            "raw_source_prompts": 14000,
            "teacher_input_tokens": 1211686,
            "teacher_output_tokens": 432371,
            "stored_probability_scalars_used_as_corpus_evidence": 14268243,
            "stored_logits_in_final_artifact": 0,
            "stored_hidden_activations_in_final_artifact": 0,
            "frozen_source_parameters_copied": copied_parameters,
            "final_artifact_parameters": parameter_count,
            "source_transformer_blocks_copied": 0,
        },
        "checkpoint_lineage": {
            "prefix": extraction["prefix_checkpoints"],
            "progressive": checkpoint_bindings,
        },
        "host": {
            "repository": extraction["layercake_host"]["repository"],
            "commit": extraction["layercake_host"]["commit"],
            "abi_version": extraction["layercake_host"]["interface"],
            "abi_contract_sha256": sha256_file(output / "abi_contract.json"),
        },
        "artifact_promoted": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = _self_hash(manifest)
    manifest_path = output / "manifest.json"
    _write_immutable(manifest_path, canonical_json_bytes(manifest))
    result = {
        "format": FORMAT,
        "status": "PASS_ASSEMBLY_AWAITING_HOSTILE_VERIFICATION",
        "protocol_sha256": sha256_file(protocol_path),
        "artifact": {
            "directory": str(output.relative_to(root)).replace("\\", "/"),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_self_hash": manifest["manifest_sha256"],
            "model_sha256": sha256_file(tensor_path),
            "model_bytes": tensor_path.stat().st_size,
            "parameters": parameter_count,
            "tensor_keys": len(saved_specs),
        },
        "source_model_loaded": False,
        "teacher_present_in_artifact": False,
        "source_blocks_in_artifact": 0,
        "strict_tensor_header_replay": True,
        "artifact_promoted": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Immutable routed v15 candidate assembly only; hostile integrity, execution, English quality, runtime, certificate, and superiority remain unproven.",
    }
    _write_immutable(
        output / "assembly_metadata.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_ARTIFACT_ASSEMBLY_PROTOCOL_V321.json",
    )
    parser.add_argument(
        "--output", default="results/abi_capability_compiler_phase3_routed_v15/artifact_v322"
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
