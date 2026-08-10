"""Independent hostile verifier for the assembled routed v15 core candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil
from safetensors import safe_open
from safetensors.torch import load_file
import torch

FORMAT = "abi-capability-compiler-phase3-routed-v15-artifact-verifier/2"
ARTIFACT_FORMAT = "layercake-routed-sparse-rank768-english-core/1"


class VerifierError(RuntimeError):
    pass


def canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        raise VerifierError(f"verifier evidence is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _self_hash(value: dict) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_documents(manifest: dict, config: dict, specs: dict) -> None:
    required_manifest = {
        "format", "artifact_format", "status", "protocol_sha256", "files",
        "tensor_specs", "tensor_keys", "parameters", "raw_fp16_parameter_bytes",
        "source", "imported_information", "checkpoint_lineage", "host",
        "artifact_promoted", "final_test_accessed", "phase3_certified", "manifest_sha256",
    }
    if set(manifest) != required_manifest or manifest.get("manifest_sha256") != _self_hash(manifest):
        raise VerifierError("artifact manifest schema or self hash changed")
    if (
        manifest.get("format") != "abi-capability-compiler-routed-v15-artifact-manifest/1"
        or manifest.get("artifact_format") != ARTIFACT_FORMAT
        or manifest.get("status") != "ASSEMBLED_UNVERIFIED_NOT_PROMOTED"
        or manifest.get("artifact_promoted") is not False
        or manifest.get("final_test_accessed") is not False
        or manifest.get("phase3_certified") is not False
        or manifest.get("tensor_specs") != specs
        or manifest.get("tensor_keys") != 613
        or manifest.get("parameters") != 536758275
        or manifest.get("raw_fp16_parameter_bytes") != 1073516550
    ):
        raise VerifierError("artifact manifest invariant changed")
    source = manifest.get("source", {})
    if (
        source.get("source_blocks_in_artifact") != 0
        or source.get("teacher_present_in_artifact") is not False
        or source.get("teacher_required_at_inference") is not False
    ):
        raise VerifierError("artifact source/teacher boundary changed")
    required_config = {
        "format", "artifact_role", "abi_version", "abi_sha256", "model", "tokenizer",
        "tensor_dtype", "strict_state_dict", "source_transformer_blocks",
        "teacher_required_at_inference",
    }
    if set(config) != required_config or (
        config.get("format") != ARTIFACT_FORMAT
        or config.get("artifact_role") != "english-core"
        or config.get("abi_version") != "lc-direct-neural-core/15"
        or config.get("tensor_dtype") != "float16"
        or config.get("strict_state_dict") is not True
        or config.get("source_transformer_blocks") != 0
        or config.get("teacher_required_at_inference") is not False
    ):
        raise VerifierError("artifact config invariant changed")


def _expect_rejection(manifest: dict, config: dict, specs: dict, mutation: str) -> bool:
    bad_manifest = copy.deepcopy(manifest)
    bad_config = copy.deepcopy(config)
    if mutation == "manifest_self_hash":
        bad_manifest["manifest_sha256"] = "0" * 64
    elif mutation == "tensor_spec_deletion":
        bad_manifest["tensor_specs"].pop(next(iter(bad_manifest["tensor_specs"])))
        bad_manifest["manifest_sha256"] = _self_hash(bad_manifest)
    elif mutation == "false_promotion":
        bad_manifest["artifact_promoted"] = True
        bad_manifest["manifest_sha256"] = _self_hash(bad_manifest)
    elif mutation == "source_block_injection":
        bad_manifest["source"]["source_blocks_in_artifact"] = 1
        bad_manifest["manifest_sha256"] = _self_hash(bad_manifest)
    elif mutation == "teacher_dependency":
        bad_manifest["source"]["teacher_required_at_inference"] = True
        bad_manifest["manifest_sha256"] = _self_hash(bad_manifest)
    elif mutation == "abi_substitution":
        bad_config["abi_sha256"] = "0" * 64
    else:
        raise AssertionError(mutation)
    try:
        _validate_documents(bad_manifest, bad_config, specs)
        if bad_config.get("abi_sha256") != manifest["host"]["abi_contract_sha256"]:
            raise VerifierError("artifact ABI substitution")
    except VerifierError:
        return True
    return False


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_IMPORT_ISOLATED_HOSTILE_ROUTED_V15_ARTIFACT_VERIFIER"
        or protocol.get("device") != "cpu"
        or protocol.get("source_model_access") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise VerifierError("routed v15 artifact verifier governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise VerifierError(f"routed v15 verifier binding changed: {name}")
    if output.exists() or "transformers" in sys.modules:
        raise VerifierError("verifier output exists or source-model runtime was imported")
    output.mkdir(parents=True)
    artifact = (root / protocol["artifact"]["directory"]).resolve()
    expected_members = {
        "abi_contract.json", "assembly_metadata.json", "config.json", "manifest.json",
        "model.safetensors",
    }
    if {path.name for path in artifact.iterdir()} != expected_members:
        raise VerifierError("artifact directory member set changed")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    abi = json.loads((artifact / "abi_contract.json").read_text(encoding="utf-8"))
    for name, binding in manifest["files"].items():
        path = artifact / name
        if set(binding) != {"sha256", "bytes"} or (
            sha256_file(path) != binding["sha256"] or path.stat().st_size != binding["bytes"]
        ):
            raise VerifierError(f"artifact member identity changed: {name}")
    tensor_path = artifact / "model.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        specs = {
            name: {
                "shape": list(handle.get_slice(name).get_shape()),
                "dtype": "float16",
            }
            for name in sorted(handle.keys())
        }
    _validate_documents(manifest, config, specs)
    if (
        config["abi_sha256"] != sha256_file(artifact / "abi_contract.json")
        or config["abi_sha256"] != manifest["host"]["abi_contract_sha256"]
        or abi.get("abi_version") != "lc-direct-neural-core/15"
        or metadata.get("format") != ARTIFACT_FORMAT
        or metadata.get("abi_sha256") != config["abi_sha256"]
    ):
        raise VerifierError("artifact ABI or safetensors metadata changed")
    mutations = [
        "manifest_self_hash", "tensor_spec_deletion", "false_promotion",
        "source_block_injection", "teacher_dependency", "abi_substitution",
    ]
    mutation_results = {name: _expect_rejection(manifest, config, specs, name) for name in mutations}
    if not all(mutation_results.values()):
        raise VerifierError("hostile metadata mutation was accepted")
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    sys.path.insert(0, str(layercake_root))
    from layercake.routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    model = RoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    process = psutil.Process()
    load_started = time.perf_counter()
    tensors = load_file(str(tensor_path), device="cpu")
    incompatible = model.load_state_dict(tensors, strict=True)
    load_seconds = time.perf_counter() - load_started
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise VerifierError("strict host load returned incompatible keys")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in model.parameters()) != 536758275:
        raise VerifierError("strict-loaded parameter count changed")
    source_ids, source_lexemes = tokenizer.encode_source(protocol["execution_probe"])
    values = torch.tensor([source_ids], dtype=torch.long)
    route_index = model._select_route(values)
    with torch.inference_mode():
        batch_logits = model.forward_routed(values, route_index)[:, -1]
        state = model.prefill_ids(source_ids, source_lexemes)
        prefill_difference = float((batch_logits.float() - state.next_logits.float()).abs().max())
        appended_action = int(protocol["incremental_probe_action"])
        token = torch.tensor([[appended_action]], dtype=torch.long)
        position = torch.tensor([state.sequence_length], dtype=torch.long)
        hidden = model.token_embedding(token)
        keys = []
        cached_values = []
        for layer, previous_key, previous_value in zip(
            model.layers, state.layer_keys, state.layer_values
        ):
            hidden, key, value = layer.incremental(
                hidden, position, previous_key, previous_value, route_index
            )
            keys.append(key); cached_values.append(value)
        incremental_logits = model.lm_head(model.final_norm(hidden[:, -1]))
        full_values = torch.tensor([source_ids + [appended_action]], dtype=torch.long)
        full_logits = model.forward_routed(full_values, route_index)[:, -1]
        incremental_difference = float(
            (incremental_logits.float() - full_logits.float()).abs().max()
        )
        layer = model.layers[0]
        for index, projection in enumerate(layer.route_coefficient_projections):
            if index != route_index:
                projection.weight.fill_(float("nan"))
        sparse_selected_finite = bool(torch.isfinite(model.forward_routed(values, route_index)).all())
        poisoned_route = (route_index + 1) % len(model.route_names)
        sparse_poison_observed = bool(
            not torch.isfinite(layer._mlp_delta(model.token_embedding(values), poisoned_route)).all()
        )
    tolerance = float(protocol["execution_tolerance_maximum_absolute_error"])
    execution_passed = (
        prefill_difference <= tolerance
        and incremental_difference <= tolerance
        and int(state.route_index) == route_index
        and len(state.layer_keys) == 32
        and all(key.shape[1] == 4 for key in state.layer_keys)
        and sparse_selected_finite
        and sparse_poison_observed
    )
    if not execution_passed or "transformers" in sys.modules:
        raise VerifierError("teacher-free execution or sparse isolation failed")
    result = {
        "format": FORMAT,
        "status": "PASS_HOSTILE_ARTIFACT_VERIFICATION",
        "protocol_sha256": sha256_file(protocol_path),
        "artifact": {
            "model_sha256": sha256_file(tensor_path),
            "parameters": 536758275,
            "tensor_keys": 613,
            "strict_state_dict": True,
            "manifest_self_hash_valid": True,
            "abi_hash_valid": True,
        },
        "hostile_mutations": mutation_results,
        "execution": {
            "probe": protocol["execution_probe"],
            "route": model.route_names[route_index],
            "prefill_batch_maximum_absolute_error": prefill_difference,
            "incremental_batch_maximum_absolute_error": incremental_difference,
            "tolerance": tolerance,
            "persistent_route": True,
            "persistent_dual_kv_layers": len(state.layer_keys),
            "physical_sparse_selected_route_finite_with_unselected_nan": sparse_selected_finite,
            "physical_sparse_poison_observed_when_selected": sparse_poison_observed,
        },
        "runtime": {
            "strict_load_seconds": load_seconds,
            "process_rss_bytes_after_execution": process.memory_info().rss,
            "device": "cpu",
        },
        "transformers_imported": False,
        "source_model_loaded": False,
        "teacher_present_in_artifact": False,
        "source_blocks_in_artifact": 0,
        "artifact_promoted": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Hostile artifact integrity and teacher-free execution pass only; autonomous English quality, benchmarked runtime, Phase 3 certificate, and superiority remain unproven.",
    }
    _write_immutable(
        output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_ARTIFACT_VERIFIER_PROTOCOL_V325.json",
    )
    parser.add_argument(
        "--output", default="results/abi_capability_compiler_phase3_routed_v15/artifact_verify_v326"
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
