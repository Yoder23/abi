"""Compress one teacher-free six-block LayerCake core to Phase-2 depth.

The operation copies only LayerCake parameters. It imports no foreign teacher
weights, logits, activations, tokenizer service, or response cache. The
resulting three-block checkpoint is a training initialization, never a quality
or runtime claim by itself.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from .artifacts import module_state_sha256
from .layercake_core_loader import ABIEnglishCoreConfig, load_layercake_core
from .layercake_host import (
    _canonical_json_bytes,
    _import_layercake_runtime,
    _is_within,
    _sha256_file,
)


DEPTH_COMPRESSION_BASE_FORMAT = (
    "abi-layercake-three-block-depth-compression-base/1"
)


def _manifest_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def parse_selected_layers(
    value: str | Sequence[int], *, source_layers: int = 6, target_layers: int = 3
) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            selected = tuple(
                int(item.strip()) for item in value.split(",") if item.strip()
            )
        except ValueError as exc:
            raise ValueError("selected layers must be comma-separated integers") from exc
    else:
        selected = tuple(int(item) for item in value)
    if (
        len(selected) != target_layers
        or tuple(sorted(set(selected))) != selected
        or selected[0] < 0
        or selected[-1] >= source_layers
    ):
        raise ValueError(
            "selected layers must be exactly three unique ascending indices "
            "within the six-block source"
        )
    return selected


def build_depth_compression_base(
    *,
    source_path: str | Path,
    layercake_root: str | Path,
    canonical_abi_path: str | Path,
    output_path: str | Path,
    selected_layers: Sequence[int],
) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    output_path = Path(output_path).resolve()
    selected = parse_selected_layers(selected_layers)
    abi_root = Path(__file__).resolve().parents[1]
    if not _is_within(source_path, abi_root):
        raise RuntimeError("depth-compression source must belong to ABI evidence")
    if _is_within(output_path, layercake_root):
        raise RuntimeError("ABI may not modify the sealed LayerCake tree")
    if output_path.exists():
        raise RuntimeError(f"compression base is immutable: {output_path}")
    if not canonical_abi_path.is_file():
        raise RuntimeError("canonical semantic ABI is missing")

    source_metadata_path = source_path / "metadata.json"
    source_checkpoint_path = source_path / "model.safetensors"
    source_metadata = json.loads(
        source_metadata_path.read_text(encoding="utf-8")
    )
    source_checkpoint_sha = _sha256_file(source_checkpoint_path)
    foreign_boundary = source_metadata.get("foreign_source_boundary", {})
    if (
        source_metadata.get("format")
        != "abi-layercake-full-english-core-acquisition/1"
        or source_metadata.get("architecture", {}).get("layers") != 6
        or source_metadata.get("checkpoint", {}).get("sha256")
        != source_checkpoint_sha
        or source_metadata.get("canonical_semantic_abi", {}).get("sha256")
        != _sha256_file(canonical_abi_path)
        or foreign_boundary.get("teacher_present_at_inference") is not False
        or int(foreign_boundary.get("source_parameters_copied", -1)) != 0
        or int(foreign_boundary.get("source_transformer_blocks_retained", -1))
        != 0
    ):
        raise RuntimeError("six-block LayerCake source identity or boundary changed")

    _import_layercake_runtime(layercake_root)
    from layercake.models.shallow_sparse_english import ShallowSparseEnglishCore

    source, tokenizer, _ = load_layercake_core(
        source_path,
        layercake_root=layercake_root,
        device="cpu",
    )
    architecture = copy.deepcopy(source_metadata["architecture"])
    architecture["layers"] = 3
    architecture["architecture_version"] = (
        "layercake-shallow-sparse-english/1-three-block-task-cakes"
    )
    target = ShallowSparseEnglishCore(ABIEnglishCoreConfig(**architecture))
    with torch.no_grad():
        target.transformer.wte.load_state_dict(
            source.transformer.wte.state_dict()
        )
        target.transformer.wpe.load_state_dict(
            source.transformer.wpe.state_dict()
        )
        target.transformer.ln_f.load_state_dict(
            source.transformer.ln_f.state_dict()
        )
        for target_block, source_index in zip(
            target.transformer.h, selected, strict=True
        ):
            target_block.load_state_dict(
                source.transformer.h[source_index].state_dict()
            )
        target.task_classifier.load_state_dict(
            source.task_classifier.state_dict()
        )
        target.task_cakes.load_state_dict(source.task_cakes.state_dict())
    target.eval()

    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_path / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in target.state_dict().items()
        },
        str(checkpoint_path),
    )
    tokenizer.save_pretrained(output_path)
    tokenizer_path = output_path / "tokenizer.json"
    total_parameters = sum(
        parameter.numel() for parameter in target.parameters()
    )
    active_parameters = int(target.active_parameter_count())
    metadata: dict[str, Any] = {
        "format": DEPTH_COMPRESSION_BASE_FORMAT,
        "status": "INITIALIZED_THREE_BLOCK_COMPRESSION_NOT_TRAINED_OR_CERTIFIED",
        "architecture": target.config.canonical_dict(),
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "tokenizer": {
            "path": tokenizer_path.name,
            "sha256": _sha256_file(tokenizer_path),
        },
        "parameters": {
            "total": total_parameters,
            "active": active_parameters,
            "active_fraction": active_parameters / total_parameters,
        },
        "physical_sparsity": target.physical_sparse_contract(),
        "incremental_state": {
            "implemented": True,
            "mechanism": "three-block GPT-2-compatible KV cache plus cached task route",
        },
        "compression": {
            "method": "ordered_layer_selection_then_bounded_sequence_finetuning",
            "selected_source_layer_indices": list(selected),
            "source_layer_count": 6,
            "target_layer_count": 3,
            "source_path": str(source_path),
            "source_checkpoint_sha256": source_checkpoint_sha,
            "source_metadata_file_sha256": _sha256_file(
                source_metadata_path
            ),
            "source_manifest_sha256": source_metadata["manifest_sha256"],
            "source_logical_state_sha256": source_metadata["acquired_core"][
                "logical_state_sha256_after"
            ],
            "target_logical_state_sha256": module_state_sha256(target),
            "additional_teacher_tokens": 0,
            "teacher_logits_or_activations_imported": 0,
            "foreign_source_parameters_copied": 0,
            "layercake_blocks_copied": 3,
        },
        "canonical_semantic_abi": {
            "path_at_creation": str(canonical_abi_path),
            "sha256": _sha256_file(canonical_abi_path),
            "changed": False,
        },
        "foreign_source_boundary": {
            "teacher_present_at_inference": False,
            "source_transformer_blocks_retained": 0,
            "source_parameters_copied": 0,
            "source_generated_text_retained_in_deployment": False,
            "teacher_tokenizer_required_at_inference": False,
        },
        "final_test_accessed": False,
        "claim_boundary": (
            "This is a teacher-free LayerCake-to-LayerCake depth-compression "
            "initialization. It preserves Phase-2 depth but makes no fluency, "
            "teacher-equivalence, speed, or promotion claim."
        ),
    }
    metadata["manifest_sha256"] = _manifest_sha(metadata)
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    if _sha256_file(source_checkpoint_path) != source_checkpoint_sha:
        raise RuntimeError("source checkpoint changed during compression")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-layers", default="0,2,5")
    args = parser.parse_args(argv)
    metadata = build_depth_compression_base(
        source_path=args.source,
        layercake_root=args.layercake_root,
        canonical_abi_path=args.canonical_abi,
        output_path=args.output,
        selected_layers=parse_selected_layers(args.selected_layers),
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "manifest_sha256": metadata["manifest_sha256"],
                "selected_layers": metadata["compression"][
                    "selected_source_layer_indices"
                ],
                "parameters": metadata["parameters"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
