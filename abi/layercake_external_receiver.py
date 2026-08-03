"""Causal receiver controls for ABI-to-LayerCake English transfer.

The receiving model class is imported from the separately sealed LayerCake
checkout.  This module owns only immutable control packages and evidence.  It
does not modify or reimplement the LayerCake model, and a passing control does
not qualify an ABI-derived English candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch

from .english_generalization_evaluation import _collapse_metrics
from .failure_attribution import AttributionError, verify_contract
from .hf_extraction import evaluate_output, load_probe_catalog


RECEIVER_FORMAT = "abi-layercake-capability-naive-receiver/1"
PAYLOAD_FORMAT = "abi-layercake-state-payload/1"
CONTROL_FORMAT = "abi-layercake-receiver-controls/1"
NATIVE_PAYLOAD_ROLE = "known_good_layercake_native_payload"
DECODING = {
    "algorithm": "greedy",
    "no_repeat_ngram_size": 4,
    "allow_prompt_ngrams": False,
    "lexical_repetition_truncation_threshold": 0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _manifest_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return _canonical_sha(payload)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttributionError(f"invalid receiver JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AttributionError(f"receiver JSON must be an object: {path}")
    return value


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    """Normalize optional manifest sections without accepting non-mappings."""

    return dict(value) if isinstance(value, Mapping) else {}


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise AttributionError(f"receiver evidence is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AttributionError(f"receiver path escapes its root: {relative}") from exc
    return path


def _exact_external_modules(layercake_root: Path):
    existing = sys.modules.get("layercake")
    if existing is not None:
        origin = Path(str(getattr(existing, "__file__", ""))).resolve()
        try:
            origin.relative_to(layercake_root)
        except ValueError as exc:
            raise AttributionError(
                f"a non-control LayerCake package is already imported: {origin}"
            ) from exc
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    model_module = importlib.import_module(
        "layercake.models.shallow_sparse_english"
    )
    training_module = importlib.import_module(
        "layercake.training.phase2_shallow_sparse"
    )
    for module in (model_module, training_module):
        try:
            Path(module.__file__).resolve().relative_to(layercake_root)
        except ValueError as exc:
            raise AttributionError(
                "receiver imported LayerCake code outside the sealed checkout"
            ) from exc
    return model_module, training_module


def _control_context(contract_path: Path, layercake_root: Path) -> dict[str, Any]:
    verification = verify_contract(contract_path, layercake_root=layercake_root)
    contract = _read(contract_path)
    control = contract["sealed_layercake_control"]
    commit = _git(layercake_root, "rev-parse", "HEAD").strip()
    porcelain = _git(layercake_root, "status", "--porcelain")
    if commit != control["repository_commit"] or porcelain:
        raise AttributionError("LayerCake receiver control is not the clean sealed checkout")
    metadata_path = _within(
        layercake_root,
        control["native_checkpoint_metadata"]["path"],
    )
    metadata = _read(metadata_path)
    checkpoint = _within(layercake_root, metadata["checkpoint"]["path"])
    tokenizer = _within(
        layercake_root,
        control["native_runtime_artifact"]["tokenizer"]["path"],
    )
    checkpoint_directory = checkpoint.parent
    tokenizer_asset_names = (
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    tokenizer_assets = [checkpoint_directory / name for name in tokenizer_asset_names]
    if any(not path.is_file() for path in tokenizer_assets):
        raise AttributionError("sealed PyTorch tokenizer assets are incomplete")
    if (
        _sha256_file(checkpoint) != control["primary_checkpoint_sha256"]
        or _sha256_file(tokenizer)
        != control["native_runtime_artifact"]["tokenizer"]["sha256"]
    ):
        raise AttributionError("sealed receiver checkpoint or tokenizer changed")
    return {
        "verification": verification,
        "contract": contract,
        "control": control,
        "commit": commit,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "tokenizer_assets": tokenizer_assets,
    }


def _state_contract(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    tensors = []
    parameters = 0
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        count = int(value.numel())
        parameters += count
        tensors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "parameters": count,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return {
        "tensor_count": len(tensors),
        "parameter_count": parameters,
        "state_sha256": _canonical_sha(tensors),
        "tensors": tensors,
    }


def _architecture(control: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    architecture = dict(metadata["architecture"])
    if architecture.get("architecture_version") != control["architecture_id"]:
        raise AttributionError("receiver architecture is not the sealed production host")
    return architecture


def create_capability_naive_receiver(
    *,
    contract_path: str | Path,
    layercake_root: str | Path,
    output_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Create one exact-architecture receiver containing no learned source state."""

    contract_path = Path(contract_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise AttributionError(f"receiver package already exists: {output_path}")
    if seed < 0:
        raise AttributionError("receiver seed must be non-negative")
    context = _control_context(contract_path, layercake_root)
    control = context["control"]
    architecture = _architecture(control, context["metadata"])
    model_module, _ = _exact_external_modules(layercake_root)
    torch.manual_seed(seed)
    model = model_module.ShallowSparseEnglishCore(
        model_module.ShallowSparseEnglishConfig(**architecture)
    ).cpu().eval()
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    state_contract = _state_contract(state)

    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint = output_path / "model.safetensors"
    save_file(state, str(checkpoint))
    for source in context["tokenizer_assets"]:
        shutil.copyfile(source, output_path / source.name)
    tokenizer_assets = [
        {
            "path": source.name,
            "bytes": (output_path / source.name).stat().st_size,
            "sha256": _sha256_file(output_path / source.name),
        }
        for source in context["tokenizer_assets"]
    ]
    manifest: dict[str, Any] = {
        "format": RECEIVER_FORMAT,
        "status": "SEALED_CAUSAL_NEGATIVE_CONTROL",
        "role": "capability_naive_receiver",
        "seed": seed,
        "layercake_host": {
            "repository_commit": context["commit"],
            "architecture_id": control["architecture_id"],
            "architecture_hash": control["architecture_hash"],
            "architecture": architecture,
            "canonical_semantic_abi_sha256": control[
                "canonical_semantic_abi_file"
            ]["sha256"],
            "model_class_source": str(
                Path(model_module.__file__).resolve().relative_to(layercake_root)
            ).replace("\\", "/"),
        },
        "checkpoint": {
            "path": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256_file(checkpoint),
            **state_contract,
        },
        "tokenizer": {
            "assets": tokenizer_assets,
            "tokenizer_json_sha256": next(
                item["sha256"]
                for item in tokenizer_assets
                if item["path"] == "tokenizer.json"
            ),
            "provenance": "copied_bit_exact_from_sealed_layercake_host",
        },
        "imported_information": {
            "foreign_teacher_prompts": 0,
            "foreign_teacher_output_bytes": 0,
            "foreign_teacher_tokens": 0,
            "foreign_teacher_logits": 0,
            "foreign_teacher_activations": 0,
            "foreign_teacher_parameters_copied": 0,
            "layercake_learned_parameters_copied": 0,
            "bridge_parameters": 0,
            "training_steps": 0,
            "training_tokens": 0,
            "training_hardware": None,
        },
        "claim_boundary": (
            "This is the exact production LayerCake architecture and tokenizer "
            "with deterministic random neural state. It is not a trained model, "
            "an ABI artifact, or evidence of English transfer."
        ),
    }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    _write_immutable(output_path / "manifest.json", manifest)
    return manifest


def create_native_state_payload(
    *,
    contract_path: str | Path,
    layercake_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Package the sealed native checkpoint as a same-path positive control."""

    contract_path = Path(contract_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise AttributionError(f"native payload package already exists: {output_path}")
    context = _control_context(contract_path, layercake_root)
    control = context["control"]
    architecture = _architecture(control, context["metadata"])
    state = load_file(str(context["checkpoint"]), device="cpu")
    state_contract = _state_contract(state)

    output_path.mkdir(parents=True, exist_ok=False)
    payload = output_path / "payload.safetensors"
    shutil.copyfile(context["checkpoint"], payload)
    manifest: dict[str, Any] = {
        "format": PAYLOAD_FORMAT,
        "status": "SEALED_NATIVE_POSITIVE_CONTROL",
        "role": "known_good_layercake_native_payload",
        "source": {
            "kind": "sealed_layercake_checkpoint",
            "repository_commit": context["commit"],
            "checkpoint_sha256": control["primary_checkpoint_sha256"],
        },
        "target": {
            "architecture_id": control["architecture_id"],
            "architecture_hash": control["architecture_hash"],
            "architecture": architecture,
            "canonical_semantic_abi_sha256": control[
                "canonical_semantic_abi_file"
            ]["sha256"],
        },
        "payload": {
            "path": payload.name,
            "bytes": payload.stat().st_size,
            "sha256": _sha256_file(payload),
            **state_contract,
        },
        "bridge": {
            "kind": "identity_state_installation",
            "parameters": 0,
            "training_steps": 0,
            "training_tokens": 0,
        },
        "imported_information": {
            "foreign_teacher_prompts": 0,
            "foreign_teacher_output_bytes": 0,
            "foreign_teacher_tokens": 0,
            "foreign_teacher_logits": 0,
            "foreign_teacher_activations": 0,
            "foreign_teacher_parameters_copied": 0,
            "layercake_native_parameters_copied": state_contract[
                "parameter_count"
            ],
        },
        "claim_boundary": (
            "This payload proves only that the receiving path carries one "
            "known-good native LayerCake state exactly. It is not ABI extraction."
        ),
    }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    _write_immutable(output_path / "manifest.json", manifest)
    return manifest


def create_abi_state_payload(
    *,
    contract_path: str | Path,
    layercake_root: str | Path,
    source_artifact_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Package one preserved ABI-owned exact-architecture state for the receiver."""

    contract_path = Path(contract_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    source_artifact_path = Path(source_artifact_path).resolve()
    output_path = Path(output_path).resolve()
    abi_root = Path(__file__).resolve().parents[1]
    try:
        source_artifact_path.relative_to(abi_root)
    except ValueError as exc:
        raise AttributionError("ABI candidate must belong to the ABI evidence tree") from exc
    if output_path.exists():
        raise AttributionError(f"ABI payload package already exists: {output_path}")
    source_metadata_path = source_artifact_path / "metadata.json"
    source_checkpoint = source_artifact_path / "model.safetensors"
    source_metadata = _read(source_metadata_path)
    allowed_formats = {
        "abi-layercake-direct-source-initialization/1",
        "abi-layercake-full-english-core-acquisition/1",
        "abi-layercake-component-graft/1",
    }
    if source_metadata.get("format") not in allowed_formats:
        raise AttributionError("ABI candidate format is not an exact-core artifact")
    declared_checkpoint = source_metadata.get("checkpoint", {})
    if (
        not source_checkpoint.is_file()
        or _sha256_file(source_checkpoint) != declared_checkpoint.get("sha256")
        or source_checkpoint.stat().st_size != int(declared_checkpoint.get("bytes", -1))
    ):
        raise AttributionError("ABI candidate checkpoint bytes changed")
    context = _control_context(contract_path, layercake_root)
    control = context["control"]
    production_architecture = _architecture(control, context["metadata"])
    candidate_architecture = dict(source_metadata.get("architecture", {}))
    for name, expected in production_architecture.items():
        if candidate_architecture.get(name) != expected:
            raise AttributionError(f"ABI candidate changed production architecture: {name}")
    allowed_zero_extensions = {
        "capability_adapter_rank": 0,
        "capability_adapter_shared_across_layers": False,
        "capability_cake_canonical_routes": [],
        "capability_cake_order": [],
        "capability_control_width": 0,
        "capability_prefix_length": 0,
        "capability_router_buckets": 0,
        "capability_router_width": 0,
        "deep_cake_gate_layers": 0,
        "deep_reused_capability_cakes": False,
        "prompt_identity_rank": 0,
        "prompt_identity_selective": False,
        "task_route_layerwise_control": False,
    }
    unexpected = {
        name: value
        for name, value in candidate_architecture.items()
        if name not in production_architecture
        and (name not in allowed_zero_extensions or value != allowed_zero_extensions[name])
    }
    if unexpected:
        raise AttributionError(
            f"ABI candidate contains non-production extensions: {sorted(unexpected)}"
        )
    canonical = source_metadata.get("canonical_semantic_abi", {})
    if canonical.get("sha256") != control["canonical_semantic_abi_file"]["sha256"]:
        raise AttributionError("ABI candidate changed the canonical semantic ABI")

    state = load_file(str(source_checkpoint), device="cpu")
    state_contract = _state_contract(state)
    model_module, _ = _exact_external_modules(layercake_root)
    model = model_module.ShallowSparseEnglishCore(
        model_module.ShallowSparseEnglishConfig(**production_architecture)
    ).cpu().eval()
    if set(state) != set(model.state_dict()):
        raise AttributionError("ABI candidate state does not match the exact host")
    model.load_state_dict(state, strict=True)
    del model

    output_path.mkdir(parents=True, exist_ok=False)
    payload = output_path / "payload.safetensors"
    shutil.copyfile(source_checkpoint, payload)
    direct = _mapping_or_empty(
        source_metadata.get("direct_source_initialization")
    )
    foreign = _mapping_or_empty(
        source_metadata.get("foreign_source_boundary")
    )
    imported_artifact = source_metadata.get("imported_artifact")
    broad_anchor = source_metadata.get("broad_behavior_anchor")
    online_source_sections = {
        name: value
        for name, value in source_metadata.items()
        if isinstance(value, dict)
        and name
        not in {
            "direct_source_initialization",
            "foreign_source_boundary",
            "training",
        }
        and (
            "source_distillation" in name
            or "source_teacher_forward_tokens" in value
            or "source_model_inference_seconds" in value
        )
    }
    if source_metadata["format"] == "abi-layercake-full-english-core-acquisition/1":
        if not isinstance(imported_artifact, dict):
            raise AttributionError(
                "full-core ABI candidate omitted imported-artifact accounting"
            )
        if imported_artifact.get("archive_sha256_before") != imported_artifact.get(
            "archive_sha256_after"
        ):
            raise AttributionError("imported ABI training artifact changed")
    imported_artifact = imported_artifact if isinstance(imported_artifact, dict) else {}
    broad_anchor = broad_anchor if isinstance(broad_anchor, dict) else {}
    online_forward_tokens = sum(
        int(section.get("source_teacher_forward_tokens", 0))
        for section in online_source_sections.values()
    )
    online_inference_seconds = sum(
        float(section.get("source_model_inference_seconds", 0.0))
        for section in online_source_sections.values()
    )
    online_source_parameters = max(
        (
            int(section.get("source_parameter_count", 0))
            for section in online_source_sections.values()
        ),
        default=0,
    )
    cached_teacher_output_bytes = int(
        imported_artifact.get("selected_teacher_output_bytes", 0)
    ) + int(broad_anchor.get("selected_teacher_output_bytes", 0))
    cached_teacher_tokens = int(
        imported_artifact.get("selected_teacher_tokens", 0)
    ) + int(broad_anchor.get("selected_teacher_tokens", 0))
    logits_stored = int(imported_artifact.get("teacher_logits_stored", 0)) + int(
        broad_anchor.get("teacher_logits_stored", 0)
    ) + sum(
        int(section.get("logits_stored", 0))
        for section in online_source_sections.values()
    )
    hidden_activation_bytes = int(
        imported_artifact.get("teacher_hidden_activation_bytes_stored", 0)
    ) + int(broad_anchor.get("teacher_hidden_activation_bytes_stored", 0)) + sum(
        int(section.get("hidden_activations_stored", 0))
        for section in online_source_sections.values()
    )
    manifest: dict[str, Any] = {
        "format": PAYLOAD_FORMAT,
        "status": "PACKAGED_HISTORICAL_ABI_CANDIDATE_NOT_QUALIFIED",
        "role": "abi_english_substrate_candidate",
        "source": {
            "kind": "abi_owned_exact_host_checkpoint",
            "artifact_path": str(source_artifact_path),
            "artifact_format": source_metadata["format"],
            "metadata_sha256": _sha256_file(source_metadata_path),
            "checkpoint_sha256": declared_checkpoint["sha256"],
            "source_checkpoint_sha256": direct.get("source_checkpoint_sha256"),
            "teacher_present_at_inference": foreign.get(
                "teacher_present_at_inference"
            ),
            "source_transformer_blocks_retained": foreign.get(
                "source_transformer_blocks_retained"
            ),
            "source_promotion_eligible": source_metadata.get("promotion_eligible"),
        },
        "target": {
            "architecture_id": control["architecture_id"],
            "architecture_hash": control["architecture_hash"],
            "architecture": production_architecture,
            "canonical_semantic_abi_sha256": control[
                "canonical_semantic_abi_file"
            ]["sha256"],
        },
        "payload": {
            "path": payload.name,
            "bytes": payload.stat().st_size,
            "sha256": _sha256_file(payload),
            **state_contract,
        },
        "bridge": {
            "kind": "identity_state_installation",
            "parameters": 0,
            "training_steps": 0,
            "training_tokens": 0,
        },
        "imported_information": {
            "cached_teacher_record_count": int(
                imported_artifact.get("selected_english_records", 0)
            ),
            "unique_cached_teacher_records_seen": int(
                imported_artifact.get("unique_selected_records_seen", 0)
            ),
            "cached_teacher_output_bytes": cached_teacher_output_bytes,
            "cached_teacher_tokens": cached_teacher_tokens,
            "online_source_teacher_forward_tokens": online_forward_tokens,
            "online_source_model_inference_seconds": online_inference_seconds,
            "online_source_parameter_count": online_source_parameters,
            "teacher_logits_stored": logits_stored,
            "teacher_hidden_activation_bytes_stored": hidden_activation_bytes,
            "foreign_teacher_parameters_copied": foreign.get(
                "source_parameters_copied", 0
            ),
            "source_transformer_blocks_retained": foreign.get(
                "source_transformer_blocks_retained", 0
            ),
            "bridge_parameters": 0,
            "training": {
                key: source_metadata.get("training", {}).get(key)
                for key in (
                    "successful_optimizer_steps",
                    "raw_utf8_bytes_seen",
                    "supervised_layercake_tokens_seen",
                    "anchor_raw_utf8_bytes_seen",
                    "anchor_supervised_layercake_tokens_seen",
                    "general_preservation_utf8_bytes_seen",
                    "general_preservation_tokens_seen",
                    "wall_seconds",
                    "gpu_hours",
                    "peak_device_memory_bytes",
                    "cpu_seconds",
                    "cpu_core_hours",
                    "active_parameter_seconds",
                )
                if key in source_metadata.get("training", {})
            },
            "raw_accounting_sections": {
                "direct_source_initialization": direct or None,
                "imported_artifact": imported_artifact or None,
                "broad_behavior_anchor": broad_anchor or None,
                "online_sources": online_source_sections,
            },
        },
        "claim_boundary": (
            "This package preserves and rescreens an ABI-owned candidate through "
            "the exact external LayerCake receiver. Packaging is not qualification; "
            "all original negative evidence and promotion restrictions remain."
        ),
    }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    _write_immutable(output_path / "manifest.json", manifest)
    if _sha256_file(source_checkpoint) != declared_checkpoint["sha256"]:
        raise AttributionError("ABI source candidate changed while packaging")
    return manifest


def _verified_package(
    path: Path,
    *,
    expected_format: str,
    control: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    manifest_path = path / "manifest.json"
    manifest = _read(manifest_path)
    if (
        manifest.get("format") != expected_format
        or manifest.get("manifest_sha256") != _manifest_sha(manifest)
    ):
        raise AttributionError(f"receiver package manifest is invalid: {path}")
    section_name = "checkpoint" if expected_format == RECEIVER_FORMAT else "payload"
    section = manifest.get(section_name)
    if not isinstance(section, dict):
        raise AttributionError("receiver package payload declaration is missing")
    payload = _within(path, str(section.get("path", "")))
    if (
        not payload.is_file()
        or payload.stat().st_size != int(section.get("bytes", -1))
        or _sha256_file(payload) != section.get("sha256")
    ):
        raise AttributionError("receiver package payload bytes changed")
    target = manifest.get("layercake_host", manifest.get("target", {}))
    if (
        target.get("architecture_id") != control["architecture_id"]
        or target.get("architecture_hash") != control["architecture_hash"]
        or target.get("canonical_semantic_abi_sha256")
        != control["canonical_semantic_abi_file"]["sha256"]
    ):
        raise AttributionError("receiver package targets a different LayerCake host")
    if expected_format == RECEIVER_FORMAT:
        tokenizer = manifest.get("tokenizer")
        assets = tokenizer.get("assets") if isinstance(tokenizer, dict) else None
        if not isinstance(assets, list) or not assets:
            raise AttributionError("receiver tokenizer assets are missing")
        for item in assets:
            if not isinstance(item, dict):
                raise AttributionError("receiver tokenizer asset declaration is invalid")
            asset = _within(path, str(item.get("path", "")))
            if (
                not asset.is_file()
                or asset.stat().st_size != int(item.get("bytes", -1))
                or _sha256_file(asset) != item.get("sha256")
            ):
                raise AttributionError("receiver tokenizer asset changed")
        if tokenizer.get("tokenizer_json_sha256") != control[
            "native_runtime_artifact"
        ]["tokenizer"]["sha256"]:
            raise AttributionError("receiver tokenizer is not the sealed tokenizer")
    return manifest, payload


def _assert_exact_state_contract(
    state: Mapping[str, torch.Tensor],
    declared: Mapping[str, Any],
) -> dict[str, Any]:
    actual = _state_contract(state)
    for key in ("tensor_count", "parameter_count", "state_sha256"):
        if actual[key] != declared.get(key):
            raise AttributionError(f"payload state contract changed: {key}")
    return actual


def _load_receiver_and_install(
    *,
    receiver_path: Path,
    payload_path: Path | None,
    control: Mapping[str, Any],
    layercake_root: Path,
):
    receiver_manifest, receiver_checkpoint = _verified_package(
        receiver_path,
        expected_format=RECEIVER_FORMAT,
        control=control,
    )
    model_module, training_module = _exact_external_modules(layercake_root)
    architecture = receiver_manifest["layercake_host"]["architecture"]
    model = model_module.ShallowSparseEnglishCore(
        model_module.ShallowSparseEnglishConfig(**architecture)
    ).cpu().eval()
    receiver_state = load_file(str(receiver_checkpoint), device="cpu")
    _assert_exact_state_contract(receiver_state, receiver_manifest["checkpoint"])
    model.load_state_dict(receiver_state, strict=True)
    installed_manifest = None
    installed_checkpoint = receiver_checkpoint
    if payload_path is not None:
        installed_manifest, installed_checkpoint = _verified_package(
            payload_path,
            expected_format=PAYLOAD_FORMAT,
            control=control,
        )
        payload_state = load_file(str(installed_checkpoint), device="cpu")
        _assert_exact_state_contract(payload_state, installed_manifest["payload"])
        if set(payload_state) != set(model.state_dict()):
            raise AttributionError("payload tensor keys do not match the receiver")
        for name, value in model.state_dict().items():
            incoming = payload_state[name]
            if incoming.shape != value.shape or incoming.dtype != value.dtype:
                raise AttributionError(f"payload tensor contract changed: {name}")
        model.load_state_dict(payload_state, strict=True)
    tokenizer = training_module._tokenizer(receiver_path)
    return model, tokenizer, receiver_manifest, installed_manifest, installed_checkpoint


def _select_token(
    logits: torch.Tensor,
    generated: list[int],
    *,
    no_repeat_ngram_size: int = 4,
) -> int:
    """Match the disclosed LayerCake natural-screen greedy decoder."""

    values = logits[0].detach().clone()
    if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size - 1:
        prefix = tuple(generated[-(no_repeat_ngram_size - 1) :])
        blocked = set()
        for index in range(len(generated) - no_repeat_ngram_size + 1):
            if tuple(generated[index : index + no_repeat_ngram_size - 1]) == prefix:
                blocked.add(generated[index + no_repeat_ngram_size - 1])
        if blocked:
            values[list(blocked)] = -torch.inf
    return int(values.argmax().item())


@torch.inference_mode()
def _generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str, list[int], int, list[int]]:
    prompt_ids = tokenizer.encode(prompt + "\n")
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    state = model.prefill(ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token_id = _select_token(
            state["next_logits"],
            generated,
            no_repeat_ngram_size=DECODING["no_repeat_ngram_size"],
        )
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        selected = torch.tensor([token_id], dtype=torch.long)
        _, state = model.decode_step(state, next_token=selected)
    output = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    cache_lengths = [int(layer[0].shape[2]) for layer in state["past_key_values"]]
    return output, generated, int(state["task_routes"].item()), cache_lengths


def evaluate_capability_naive_receiver(
    *,
    contract_path: str | Path,
    layercake_root: str | Path,
    receiver_path: str | Path,
    catalog_path: str | Path,
    output_path: str | Path,
    control_name: str = "capability_naive_receiver",
) -> dict[str, Any]:
    """Evaluate the no-capability receiver on a previously disclosed audit."""

    contract_path = Path(contract_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    receiver_path = Path(receiver_path).resolve()
    catalog_path = Path(catalog_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise AttributionError(f"receiver evaluation is immutable: {output_path}")
    context = _control_context(contract_path, layercake_root)
    control = context["control"]
    model, tokenizer, receiver, _, checkpoint = _load_receiver_and_install(
        receiver_path=receiver_path,
        payload_path=None,
        control=control,
        layercake_root=layercake_root,
    )
    checkpoint_before = _sha256_file(checkpoint)
    catalog = load_probe_catalog(catalog_path)
    observations = []
    started = time.perf_counter()
    for probe in catalog["probes"]:
        generated_started = time.perf_counter()
        output, token_ids, route, cache_lengths = _generate(
            model,
            tokenizer,
            str(probe["prompt"]),
            max_new_tokens=int(probe["max_new_tokens"]),
        )
        passed, score = evaluate_output(output, probe["evaluator"])
        observations.append(
            {
                "probe_id": str(probe["probe_id"]),
                "capability": str(probe["capability"]),
                "prompt_sha256": hashlib.sha256(
                    str(probe["prompt"]).encode("utf-8")
                ).hexdigest(),
                "evaluator": probe["evaluator"],
                "output": output,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "authoritative_generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "automatic_route": route,
                "cache_lengths": cache_lengths,
                "passed": passed,
                "score": score,
                "collapse": _collapse_metrics(
                    token_ids,
                    output,
                    prompt_token_ids=tokenizer.encode(str(probe["prompt"]) + "\n"),
                    prompt=str(probe["prompt"]),
                ),
                "seconds": time.perf_counter() - generated_started,
            }
        )
    passes = sum(bool(row["passed"]) for row in observations)
    quality_result = "PASS" if passes == len(observations) else "FAIL"
    if control_name not in {"capability_naive_receiver", "bridge_only"}:
        raise AttributionError("unsupported no-payload receiver control")
    expected = context["contract"]["required_control_matrix"][control_name][
        "expected_english_quality_result"
    ]
    checks = {
        "exact_external_host_verified": context["verification"]["status"] == "PASS",
        "receiver_contains_no_teacher_or_layercake_learned_parameters": all(
            receiver["imported_information"][key] == 0
            for key in (
                "foreign_teacher_parameters_copied",
                "layercake_learned_parameters_copied",
                "bridge_parameters",
                "training_tokens",
            )
        ),
        "quality_result_matches_preregistered_negative_expectation": quality_result
        == expected,
        "persistent_three_layer_incremental_state": all(
            len(row["cache_lengths"]) == 3
            and len(set(row["cache_lengths"])) == 1
            for row in observations
        ),
        "receiver_checkpoint_immutable": checkpoint_before == _sha256_file(checkpoint),
        "sealed_layercake_checkout_still_clean": not bool(
            _git(layercake_root, "status", "--porcelain")
        ),
    }
    evidence: dict[str, Any] = {
        "format": CONTROL_FORMAT,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "control": control_name,
        "english_quality_result": quality_result,
        "expected_english_quality_result": expected,
        "claim_scope": "CAUSAL_NEGATIVE_CONTROL_ONLY_NOT_ABI_TRANSFER",
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "observations": len(observations),
        },
        "receiver": {
            "path": str(receiver_path),
            "manifest_sha256": receiver["manifest_sha256"],
            "checkpoint_sha256": checkpoint_before,
            "seed": receiver["seed"],
        },
        "bridge": {
            "kind": "identity_state_installation",
            "parameters": 0,
            "training_steps": 0,
            "training_tokens": 0,
            "artifact_payload_present": False,
        }
        if control_name == "bridge_only"
        else None,
        "metrics": {
            "passes": passes,
            "failures": len(observations) - passes,
            "pass_rate": passes / max(1, len(observations)),
            "collapse_count": sum(
                bool(row["collapse"]["collapse_detected"])
                for row in observations
            ),
            "wall_seconds": time.perf_counter() - started,
        },
        "decoding": DECODING,
        "checks": checks,
        "observations": observations,
        "promotion_eligible": False,
        "abi_transfer_proven": False,
        "moonshot_complete": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    _write_immutable(output_path, evidence)
    return evidence


def _verified_control_evidence(
    path: Path,
    *,
    control_name: str,
) -> tuple[dict[str, Any], str]:
    evidence = _read(path)
    claimed = evidence.pop("evidence_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha(evidence):
        raise AttributionError(f"control evidence hash mismatch: {path}")
    if (
        evidence.get("format") != CONTROL_FORMAT
        or evidence.get("status") != "PASS"
        or evidence.get("control") != control_name
    ):
        raise AttributionError(f"required receiver control did not pass: {control_name}")
    return evidence, claimed


def evaluate_abi_candidate_baseline(
    *,
    contract_path: str | Path,
    layercake_root: str | Path,
    receiver_path: str | Path,
    payload_path: str | Path,
    catalog_path: str | Path,
    native_same_path_evidence_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Attribute one preserved ABI candidate's quality on the exact host."""

    contract_path = Path(contract_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    receiver_path = Path(receiver_path).resolve()
    payload_path = Path(payload_path).resolve()
    catalog_path = Path(catalog_path).resolve()
    native_same_path_evidence_path = Path(native_same_path_evidence_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise AttributionError(f"ABI baseline evidence is immutable: {output_path}")
    context = _control_context(contract_path, layercake_root)
    control = context["control"]
    same_path, same_path_sha = _verified_control_evidence(
        native_same_path_evidence_path,
        control_name="native_payload_same_path",
    )
    model, tokenizer, receiver, payload, checkpoint = _load_receiver_and_install(
        receiver_path=receiver_path,
        payload_path=payload_path,
        control=control,
        layercake_root=layercake_root,
    )
    if payload is None or payload.get("role") != "abi_english_substrate_candidate":
        raise AttributionError("baseline payload is not an ABI English candidate")
    checkpoint_before = _sha256_file(checkpoint)
    source_artifact = Path(payload["source"]["artifact_path"]).resolve()
    source_checkpoint = source_artifact / "model.safetensors"
    source_before = _sha256_file(source_checkpoint)
    if source_before != payload["source"]["checkpoint_sha256"]:
        raise AttributionError("ABI source artifact changed after packaging")
    catalog = load_probe_catalog(catalog_path)
    observations = []
    started = time.perf_counter()
    for probe in catalog["probes"]:
        generated_started = time.perf_counter()
        output, token_ids, route, cache_lengths = _generate(
            model,
            tokenizer,
            str(probe["prompt"]),
            max_new_tokens=int(probe["max_new_tokens"]),
        )
        passed, score = evaluate_output(output, probe["evaluator"])
        observations.append(
            {
                "probe_id": str(probe["probe_id"]),
                "capability": str(probe["capability"]),
                "prompt_sha256": hashlib.sha256(
                    str(probe["prompt"]).encode("utf-8")
                ).hexdigest(),
                "evaluator": probe["evaluator"],
                "output": output,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "authoritative_generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "automatic_route": route,
                "cache_lengths": cache_lengths,
                "passed": passed,
                "score": score,
                "collapse": _collapse_metrics(
                    token_ids,
                    output,
                    prompt_token_ids=tokenizer.encode(str(probe["prompt"]) + "\n"),
                    prompt=str(probe["prompt"]),
                ),
                "seconds": time.perf_counter() - generated_started,
            }
        )
    passes = sum(bool(row["passed"]) for row in observations)
    quality_result = "PASS" if passes == len(observations) else "FAIL"
    retained_blocks = int(
        payload["imported_information"].get("source_transformer_blocks_retained", 0)
    )
    teacher_absent = payload["source"].get("teacher_present_at_inference") is False
    source_after = _sha256_file(source_checkpoint)
    checks = {
        "exact_external_host_verified": context["verification"]["status"] == "PASS",
        "known_good_native_payload_passed_same_path": same_path.get("result") == "PASS",
        "candidate_uses_exact_production_architecture": payload["target"][
            "architecture_id"
        ]
        == control["architecture_id"],
        "canonical_abi_identity_bound": payload["target"][
            "canonical_semantic_abi_sha256"
        ]
        == control["canonical_semantic_abi_file"]["sha256"],
        "teacher_absent_at_inference": teacher_absent,
        "candidate_payload_immutable": checkpoint_before == _sha256_file(checkpoint),
        "source_artifact_immutable": source_before == source_after,
        "persistent_three_layer_incremental_state": all(
            len(row["cache_lengths"]) == 3
            and len(set(row["cache_lengths"])) == 1
            for row in observations
        ),
        "sealed_layercake_checkout_still_clean": not bool(
            _git(layercake_root, "status", "--porcelain")
        ),
    }
    rejection_reasons = []
    if quality_result != "PASS":
        rejection_reasons.append("locked_28_prompt_english_quality_failed")
    if retained_blocks:
        rejection_reasons.append("foreign_source_transformer_blocks_retained_exact")
    if not teacher_absent:
        rejection_reasons.append("teacher_present_at_inference")
    evidence: dict[str, Any] = {
        "format": CONTROL_FORMAT,
        "status": "PASS_ATTRIBUTION" if all(checks.values()) else "FAIL_ATTRIBUTION",
        "control": "historical_abi_candidate_exact_host_baseline",
        "english_quality_result": quality_result,
        "preliminary_attribution": (
            "ABI_EXTRACTION_FAILURE" if rejection_reasons else "OPEN_NOT_PROMOTED"
        ),
        "claim_scope": "ABI_CANDIDATE_BASELINE_ONLY_NOT_PROMOTION",
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "observations": len(observations),
        },
        "receiver": {
            "path": str(receiver_path),
            "manifest_sha256": receiver["manifest_sha256"],
        },
        "payload": {
            "path": str(payload_path),
            "manifest_sha256": payload["manifest_sha256"],
            "checkpoint_sha256": checkpoint_before,
            "source": payload["source"],
            "imported_information": payload["imported_information"],
        },
        "native_same_path_control": {
            "path": str(native_same_path_evidence_path),
            "evidence_sha256": same_path_sha,
        },
        "metrics": {
            "passes": passes,
            "failures": len(observations) - passes,
            "pass_rate": passes / max(1, len(observations)),
            "collapse_count": sum(
                bool(row["collapse"]["collapse_detected"])
                for row in observations
            ),
            "wall_seconds": time.perf_counter() - started,
        },
        "decoding": DECODING,
        "checks": checks,
        "rejection_reasons": rejection_reasons,
        "observations": observations,
        "promotion_eligible": False,
        "abi_transfer_proven": False,
        "moonshot_complete": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    _write_immutable(output_path, evidence)
    return evidence


@torch.inference_mode()
def evaluate_native_payload_quality_scope_control(
    *,
    contract_path: str | Path,
    layercake_root: str | Path,
    receiver_path: str | Path,
    payload_path: str | Path,
    catalog_path: str | Path,
    native_same_path_evidence_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Measure the sealed native payload on the ABI candidate quality scope.

    This is an observational attribution control, not a new LayerCake
    certification.  Execution and identity checks determine whether the
    control ran validly; quality is reported separately so a new ABI target
    cannot silently inherit the narrower native Phase 2 certificate.
    """

    contract_path = Path(contract_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    receiver_path = Path(receiver_path).resolve()
    payload_path = Path(payload_path).resolve()
    catalog_path = Path(catalog_path).resolve()
    native_same_path_evidence_path = Path(native_same_path_evidence_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise AttributionError(f"native quality control is immutable: {output_path}")
    context = _control_context(contract_path, layercake_root)
    control = context["control"]
    _, same_path_sha = _verified_control_evidence(
        native_same_path_evidence_path,
        control_name="native_payload_same_path",
    )
    model, tokenizer, receiver, payload, checkpoint = _load_receiver_and_install(
        receiver_path=receiver_path,
        payload_path=payload_path,
        control=control,
        layercake_root=layercake_root,
    )
    if payload is None or payload.get("role") != NATIVE_PAYLOAD_ROLE:
        raise AttributionError("quality-scope control payload is not sealed native state")
    checkpoint_before = _sha256_file(checkpoint)
    if checkpoint_before != control["primary_checkpoint_sha256"]:
        raise AttributionError("quality-scope control is not the sealed native checkpoint")

    catalog = load_probe_catalog(catalog_path)
    observations = []
    started = time.perf_counter()
    for probe in catalog["probes"]:
        generated_started = time.perf_counter()
        prompt = str(probe["prompt"])
        output, token_ids, route, cache_lengths = _generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=int(probe["max_new_tokens"]),
        )
        passed, score = evaluate_output(output, probe["evaluator"])
        observations.append(
            {
                "probe_id": str(probe["probe_id"]),
                "capability": str(probe["capability"]),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "evaluator": probe["evaluator"],
                "output": output,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "authoritative_generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "automatic_route": route,
                "cache_lengths": cache_lengths,
                "passed": passed,
                "score": score,
                "collapse": _collapse_metrics(
                    token_ids,
                    output,
                    prompt_token_ids=tokenizer.encode(prompt + "\n"),
                    prompt=prompt,
                ),
                "seconds": time.perf_counter() - generated_started,
            }
        )
    passes = sum(bool(row["passed"]) for row in observations)
    collapse_count = sum(
        bool(row["collapse"]["collapse_detected"]) for row in observations
    )
    checks = {
        "exact_external_host_verified": context["verification"]["status"] == "PASS",
        "payload_is_sealed_native_checkpoint": checkpoint_before
        == control["primary_checkpoint_sha256"],
        "native_same_path_control_bound": isinstance(same_path_sha, str),
        "canonical_abi_identity_bound": payload["target"][
            "canonical_semantic_abi_sha256"
        ]
        == control["canonical_semantic_abi_file"]["sha256"],
        "persistent_three_layer_incremental_state": all(
            len(row["cache_lengths"]) == 3
            and len(set(row["cache_lengths"])) == 1
            for row in observations
        ),
        "payload_immutable": checkpoint_before == _sha256_file(checkpoint),
        "sealed_layercake_checkout_still_clean": not bool(
            _git(layercake_root, "status", "--porcelain")
        ),
    }
    evidence: dict[str, Any] = {
        "format": CONTROL_FORMAT,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "control": "sealed_native_payload_matched_quality_scope",
        "claim_scope": (
            "OBSERVATIONAL_MATCHED_QUALITY_SCOPE_CONTROL_ONLY; this does not "
            "change or broaden the sealed LayerCake Phase 2 certificate"
        ),
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "observations": len(observations),
        },
        "layercake": {
            "repository_commit": context["commit"],
            "checkpoint_sha256": checkpoint_before,
            "architecture_id": control["architecture_id"],
        },
        "receiver": {
            "path": str(receiver_path),
            "manifest_sha256": receiver["manifest_sha256"],
        },
        "payload": {
            "path": str(payload_path),
            "manifest_sha256": payload["manifest_sha256"],
            "checkpoint_sha256": checkpoint_before,
        },
        "native_same_path_control": {
            "path": str(native_same_path_evidence_path),
            "evidence_sha256": same_path_sha,
        },
        "quality_scope": {
            "passes": passes,
            "failures": len(observations) - passes,
            "pass_rate": passes / max(1, len(observations)),
            "collapse_count": collapse_count,
            "all_prompts_pass": passes == len(observations),
            "minimum_10_of_28_screen_pass": passes >= 10 and collapse_count == 0,
            "wall_seconds": time.perf_counter() - started,
        },
        "decoding": DECODING,
        "checks": checks,
        "observations": observations,
        "promotion_eligible": False,
        "abi_transfer_proven": False,
        "moonshot_complete": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    _write_immutable(output_path, evidence)
    return evidence


@torch.inference_mode()
def run_native_payload_same_path_control(
    *,
    contract_path: str | Path,
    layercake_root: str | Path,
    receiver_path: str | Path,
    payload_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Prove exact state and behavior through the receiver installation path."""

    contract_path = Path(contract_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    receiver_path = Path(receiver_path).resolve()
    payload_path = Path(payload_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise AttributionError(f"same-path evidence is immutable: {output_path}")
    context = _control_context(contract_path, layercake_root)
    control = context["control"]
    installed, tokenizer, receiver, payload, installed_checkpoint = (
        _load_receiver_and_install(
            receiver_path=receiver_path,
            payload_path=payload_path,
            control=control,
            layercake_root=layercake_root,
        )
    )
    _, training_module = _exact_external_modules(layercake_root)
    direct, direct_tokenizer, direct_metadata = training_module.load_student(
        context["checkpoint"].parent, device="cpu"
    )
    installed_state = installed.state_dict()
    direct_state = direct.state_dict()
    state_equal = set(installed_state) == set(direct_state) and all(
        torch.equal(installed_state[name], direct_state[name])
        for name in direct_state
    )
    prompts = (
        "Rewrite this politely: send the revised note by noon.",
        "Use only this memo: Project Kestrel moved to 14:30. State the project and time.",
        "Continue this supplied sequence coherently: first gather notes, then",
    )
    comparisons = []
    for prompt in prompts:
        installed_output, installed_ids, installed_route, installed_cache = _generate(
            installed, tokenizer, prompt, max_new_tokens=32
        )
        direct_output, direct_ids, direct_route, direct_cache = _generate(
            direct, direct_tokenizer, prompt, max_new_tokens=32
        )
        comparisons.append(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "installed_output_sha256": hashlib.sha256(
                    installed_output.encode("utf-8")
                ).hexdigest(),
                "direct_output_sha256": hashlib.sha256(
                    direct_output.encode("utf-8")
                ).hexdigest(),
                "token_ids_exact": installed_ids == direct_ids,
                "output_exact": installed_output == direct_output,
                "route_exact": installed_route == direct_route,
                "cache_lengths_exact": installed_cache == direct_cache,
            }
        )
    sparse = installed.physical_sparse_contract()
    payload_checkpoint_before = _sha256_file(installed_checkpoint)
    checks = {
        "exact_external_host_verified": context["verification"]["status"] == "PASS",
        "payload_is_sealed_native_checkpoint": payload["source"][
            "checkpoint_sha256"
        ]
        == control["primary_checkpoint_sha256"],
        "payload_file_is_bit_exact_native_checkpoint": payload_checkpoint_before
        == control["primary_checkpoint_sha256"],
        "installed_parameter_state_exact": state_equal,
        "three_prompt_token_output_route_and_state_exact": all(
            all(
                row[key]
                for key in (
                    "token_ids_exact",
                    "output_exact",
                    "route_exact",
                    "cache_lengths_exact",
                )
            )
            for row in comparisons
        ),
        "canonical_abi_identity_bound": payload["target"][
            "canonical_semantic_abi_sha256"
        ]
        == control["canonical_semantic_abi_file"]["sha256"],
        "physical_sparse_contract_retained": sparse[
            "maximum_active_task_cakes_per_sequence"
        ]
        == 1
        and sparse["inactive_cakes_called"] == 0,
        "identity_bridge_has_zero_parameters": payload["bridge"]["parameters"] == 0,
        "payload_immutable": payload_checkpoint_before
        == _sha256_file(installed_checkpoint),
        "sealed_layercake_checkout_still_clean": not bool(
            _git(layercake_root, "status", "--porcelain")
        ),
    }
    evidence: dict[str, Any] = {
        "format": CONTROL_FORMAT,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "control": "native_payload_same_path",
        "result": "PASS" if all(checks.values()) else "FAIL",
        "claim_scope": "NATIVE_PAYLOAD_RECEIVER_PATH_CONTROL_ONLY_NOT_ABI_TRANSFER",
        "layercake": {
            "repository_commit": context["commit"],
            "checkpoint_sha256": control["primary_checkpoint_sha256"],
            "architecture_id": control["architecture_id"],
            "canonical_semantic_abi_sha256": control[
                "canonical_semantic_abi_file"
            ]["sha256"],
            "direct_metadata_checkpoint_sha256": direct_metadata["checkpoint"][
                "sha256"
            ],
        },
        "receiver": {
            "path": str(receiver_path),
            "manifest_sha256": receiver["manifest_sha256"],
        },
        "payload": {
            "path": str(payload_path),
            "manifest_sha256": payload["manifest_sha256"],
            "checkpoint_sha256": payload_checkpoint_before,
            "parameter_count": payload["payload"]["parameter_count"],
        },
        "physical_sparse_contract": sparse,
        "comparisons": comparisons,
        "checks": checks,
        "promotion_eligible": False,
        "abi_transfer_proven": False,
        "moonshot_complete": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    _write_immutable(output_path, evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--contract", required=True)
    common.add_argument("--layercake-root", required=True)
    create_receiver = subparsers.add_parser("create-receiver", parents=[common])
    create_receiver.add_argument("--output", required=True)
    create_receiver.add_argument("--seed", type=int, required=True)
    create_payload = subparsers.add_parser("create-native-payload", parents=[common])
    create_payload.add_argument("--output", required=True)
    create_abi_payload = subparsers.add_parser(
        "create-abi-payload", parents=[common]
    )
    create_abi_payload.add_argument("--source-artifact", required=True)
    create_abi_payload.add_argument("--output", required=True)
    evaluate_receiver = subparsers.add_parser("evaluate-naive", parents=[common])
    evaluate_receiver.add_argument("--receiver", required=True)
    evaluate_receiver.add_argument("--catalog", required=True)
    evaluate_receiver.add_argument("--output", required=True)
    evaluate_bridge = subparsers.add_parser(
        "evaluate-bridge-only", parents=[common]
    )
    evaluate_bridge.add_argument("--receiver", required=True)
    evaluate_bridge.add_argument("--catalog", required=True)
    evaluate_bridge.add_argument("--output", required=True)
    same_path = subparsers.add_parser("same-path", parents=[common])
    same_path.add_argument("--receiver", required=True)
    same_path.add_argument("--payload", required=True)
    same_path.add_argument("--output", required=True)
    baseline = subparsers.add_parser("evaluate-abi-baseline", parents=[common])
    baseline.add_argument("--receiver", required=True)
    baseline.add_argument("--payload", required=True)
    baseline.add_argument("--catalog", required=True)
    baseline.add_argument("--native-same-path-evidence", required=True)
    baseline.add_argument("--output", required=True)
    native_quality = subparsers.add_parser(
        "evaluate-native-quality-scope", parents=[common]
    )
    native_quality.add_argument("--receiver", required=True)
    native_quality.add_argument("--payload", required=True)
    native_quality.add_argument("--catalog", required=True)
    native_quality.add_argument("--native-same-path-evidence", required=True)
    native_quality.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "contract_path": args.contract,
        "layercake_root": args.layercake_root,
    }
    if args.command == "create-receiver":
        result = create_capability_naive_receiver(
            **common, output_path=args.output, seed=args.seed
        )
    elif args.command == "create-native-payload":
        result = create_native_state_payload(**common, output_path=args.output)
    elif args.command == "create-abi-payload":
        result = create_abi_state_payload(
            **common,
            source_artifact_path=args.source_artifact,
            output_path=args.output,
        )
    elif args.command in {"evaluate-naive", "evaluate-bridge-only"}:
        result = evaluate_capability_naive_receiver(
            **common,
            receiver_path=args.receiver,
            catalog_path=args.catalog,
            output_path=args.output,
            control_name=(
                "bridge_only"
                if args.command == "evaluate-bridge-only"
                else "capability_naive_receiver"
            ),
        )
    elif args.command == "same-path":
        result = run_native_payload_same_path_control(
            **common,
            receiver_path=args.receiver,
            payload_path=args.payload,
            output_path=args.output,
        )
    elif args.command == "evaluate-native-quality-scope":
        result = evaluate_native_payload_quality_scope_control(
            **common,
            receiver_path=args.receiver,
            payload_path=args.payload,
            catalog_path=args.catalog,
            native_same_path_evidence_path=args.native_same_path_evidence,
            output_path=args.output,
        )
    else:
        result = evaluate_abi_candidate_baseline(
            **common,
            receiver_path=args.receiver,
            payload_path=args.payload,
            catalog_path=args.catalog,
            native_same_path_evidence_path=args.native_same_path_evidence,
            output_path=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {
        "PASS",
        "PASS_ATTRIBUTION",
        "SEALED_CAUSAL_NEGATIVE_CONTROL",
        "SEALED_NATIVE_POSITIVE_CONTROL",
        "PACKAGED_HISTORICAL_ABI_CANDIDATE_NOT_QUALIFIED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
