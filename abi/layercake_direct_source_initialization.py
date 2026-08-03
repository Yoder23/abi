"""Initialize a LayerCake English substrate from compatible source weights.

This is an acquisition-only operation.  The result deliberately remains
non-promotable until GPU conformance has changed every imported source block
and the exact deployed descendant has passed the locked quality/runtime gates.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from .artifacts import _tensor_bytes, module_state_sha256
from .layercake_core_loader import load_layercake_core
from .layercake_host import (
    _canonical_json_bytes,
    _is_within,
    _sha256_file,
)


DIRECT_SOURCE_BASE_FORMAT = "abi-layercake-direct-source-initialization/1"


class DirectSourceInitializationError(RuntimeError):
    """Raised when the exact import boundary or compatible topology changes."""


def _manifest_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def parse_selected_layers(
    value: str | Sequence[int], *, source_layers: int, target_layers: int
) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            selected = tuple(
                int(item.strip()) for item in value.split(",") if item.strip()
            )
        except ValueError as exc:
            raise DirectSourceInitializationError(
                "selected layers must be comma-separated integers"
            ) from exc
    else:
        selected = tuple(int(item) for item in value)
    if (
        len(selected) != target_layers
        or tuple(sorted(set(selected))) != selected
        or not selected
        or selected[0] < 0
        or selected[-1] >= source_layers
    ):
        raise DirectSourceInitializationError(
            "selected layers must be unique ascending source indices with "
            "one entry per target block"
        )
    return selected


def _copy_module(
    *,
    target: torch.nn.Module,
    source: torch.nn.Module,
    target_prefix: str,
    source_prefix: str,
) -> tuple[list[dict[str, Any]], int]:
    target_state = target.state_dict()
    source_state = source.state_dict()
    if target_state.keys() != source_state.keys():
        raise DirectSourceInitializationError(
            f"incompatible module state: {target_prefix} <- {source_prefix}"
        )
    parameter_names = set(dict(target.named_parameters()).keys())
    ledger: list[dict[str, Any]] = []
    parameter_count = 0
    with torch.no_grad():
        for name in target_state:
            target_value = target_state[name]
            source_value = source_state[name]
            if (
                target_value.shape != source_value.shape
                or target_value.dtype != source_value.dtype
            ):
                raise DirectSourceInitializationError(
                    f"incompatible source tensor: {source_prefix}.{name}"
                )
            target_value.copy_(source_value)
            if not torch.equal(target_value, source_value):
                raise DirectSourceInitializationError(
                    f"source tensor copy was not exact: {source_prefix}.{name}"
                )
            is_parameter = name in parameter_names
            if is_parameter:
                parameter_count += target_value.numel()
            ledger.append(
                {
                    "target_tensor": f"{target_prefix}.{name}",
                    "source_tensor": f"{source_prefix}.{name}",
                    "kind": "parameter" if is_parameter else "buffer",
                    "numel": target_value.numel(),
                    "sha256_after_copy": tensor_sha256(target_value),
                }
            )
    return ledger, parameter_count


def copy_source_substrate(
    *,
    target: torch.nn.Module,
    source: torch.nn.Module,
    selected_layers: Sequence[int],
) -> dict[str, Any]:
    """Copy embeddings, selected blocks, and final norm with exact proof."""

    target_transformer = target.transformer
    source_transformer = source.transformer
    selected = parse_selected_layers(
        selected_layers,
        source_layers=len(source_transformer.h),
        target_layers=len(target_transformer.h),
    )
    modules: list[tuple[torch.nn.Module, torch.nn.Module, str, str]] = [
        (
            target_transformer.wte,
            source_transformer.wte,
            "transformer.wte",
            "transformer.wte",
        ),
        (
            target_transformer.wpe,
            source_transformer.wpe,
            "transformer.wpe",
            "transformer.wpe",
        ),
        (
            target_transformer.ln_f,
            source_transformer.ln_f,
            "transformer.ln_f",
            "transformer.ln_f",
        ),
    ]
    for target_index, source_index in enumerate(selected):
        modules.append(
            (
                target_transformer.h[target_index],
                source_transformer.h[source_index],
                f"transformer.h.{target_index}",
                f"transformer.h.{source_index}",
            )
        )

    copied_tensors: list[dict[str, Any]] = []
    copied_parameters = 0
    for target_module, source_module, target_prefix, source_prefix in modules:
        ledger, count = _copy_module(
            target=target_module,
            source=source_module,
            target_prefix=target_prefix,
            source_prefix=source_prefix,
        )
        copied_tensors.extend(ledger)
        copied_parameters += count
    return {
        "method": "exact_shape_compatible_weight_copy_then_gpu_conformance",
        "selected_source_layer_indices": list(selected),
        "source_layer_count": len(source_transformer.h),
        "target_layer_count": len(target_transformer.h),
        "source_parameters_copied_at_initialization": copied_parameters,
        "source_transformer_blocks_retained_exact_at_initialization": len(
            selected
        ),
        "copied_target_tensors": copied_tensors,
        "target_block_state_sha256_after_copy": [
            module_state_sha256(block) for block in target_transformer.h
        ],
    }


def build_direct_source_base(
    *,
    source_path: str | Path,
    parent_path: str | Path,
    layercake_root: str | Path,
    canonical_abi_path: str | Path,
    output_path: str | Path,
    selected_layers: Sequence[int],
    device_name: str = "cuda",
) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    parent_path = Path(parent_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    output_path = Path(output_path).resolve()
    abi_root = Path(__file__).resolve().parents[1]
    if device_name != "cuda" or not torch.cuda.is_available():
        raise DirectSourceInitializationError(
            "ABI source-weight acquisition requires an available CUDA device"
        )
    if not (
        _is_within(source_path, layercake_root)
        or _is_within(source_path, abi_root)
    ):
        raise DirectSourceInitializationError(
            "source must belong to the sealed LayerCake or ABI evidence tree"
        )
    if not (
        _is_within(parent_path, layercake_root)
        or _is_within(parent_path, abi_root)
    ):
        raise DirectSourceInitializationError(
            "parent must belong to the sealed LayerCake or ABI evidence tree"
        )
    if _is_within(output_path, layercake_root):
        raise DirectSourceInitializationError(
            "ABI acquisition may not modify the sealed LayerCake tree"
        )
    if output_path.exists():
        raise DirectSourceInitializationError(
            f"direct-source base is immutable: {output_path}"
        )
    if not canonical_abi_path.is_file():
        raise DirectSourceInitializationError("canonical semantic ABI is missing")

    source_checkpoint = source_path / "model.safetensors"
    source_tokenizer_file = source_path / "tokenizer.json"
    source_config = source_path / "config.json"
    parent_checkpoint = parent_path / "model.safetensors"
    parent_metadata_path = parent_path / "metadata.json"
    for required in (
        source_checkpoint,
        source_tokenizer_file,
        source_config,
        parent_checkpoint,
        parent_metadata_path,
    ):
        if not required.is_file():
            raise DirectSourceInitializationError(
                f"required source or parent file is absent: {required}"
            )
    source_checkpoint_sha = _sha256_file(source_checkpoint)
    parent_checkpoint_sha = _sha256_file(parent_checkpoint)
    parent_metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    if parent_metadata.get("checkpoint", {}).get("sha256") != parent_checkpoint_sha:
        raise DirectSourceInitializationError("parent checkpoint hash changed")

    device = torch.device("cuda")
    parent, parent_tokenizer, _ = load_layercake_core(
        parent_path, layercake_root=layercake_root, device=device
    )
    source_tokenizer = AutoTokenizer.from_pretrained(
        source_path, local_files_only=True
    )
    if (
        source_tokenizer.get_vocab() != parent_tokenizer.get_vocab()
        or source_tokenizer.eos_token_id != parent_tokenizer.eos_token_id
    ):
        raise DirectSourceInitializationError(
            "source and LayerCake token-id vocabularies are not exact"
        )
    source = AutoModelForCausalLM.from_pretrained(
        source_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    source.eval()
    source.requires_grad_(False)
    if (
        int(source.config.vocab_size) != int(parent.config.vocab_size)
        or int(source.config.n_embd) != int(parent.config.width)
        or int(source.config.n_head) != int(parent.config.heads)
        or int(source.config.n_positions) < int(parent.config.max_tokens)
    ):
        raise DirectSourceInitializationError(
            "source vocabulary, width, heads, or context is incompatible"
        )

    sparse_before = module_state_sha256(parent.task_cakes)
    classifier_before = module_state_sha256(parent.task_classifier)
    import_ledger = copy_source_substrate(
        target=parent,
        source=source,
        selected_layers=selected_layers,
    )
    if (
        module_state_sha256(parent.task_cakes) != sparse_before
        or module_state_sha256(parent.task_classifier) != classifier_before
    ):
        raise DirectSourceInitializationError(
            "direct source import changed LayerCake sparse components"
        )
    source_parameter_count = sum(p.numel() for p in source.parameters())
    del source
    torch.cuda.empty_cache()
    parent.eval()

    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_path / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in parent.state_dict().items()
        },
        str(checkpoint_path),
    )
    parent_tokenizer.save_pretrained(output_path)
    tokenizer_path = output_path / "tokenizer.json"
    manifest: dict[str, Any] = {
        "format": DIRECT_SOURCE_BASE_FORMAT,
        "status": "INITIALIZED_FROM_SOURCE_NOT_CONFORMED_OR_CERTIFIED",
        "architecture": parent.config.canonical_dict(),
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "tokenizer": {
            "path": tokenizer_path.name,
            "sha256": _sha256_file(tokenizer_path),
            "token_id_vocabulary_exact_to_source": True,
        },
        "parameters": {
            "total": sum(p.numel() for p in parent.parameters()),
            "active": int(parent.active_parameter_count()),
        },
        "parent_layercake": {
            "path_at_initialization": str(parent_path),
            "checkpoint_sha256": parent_checkpoint_sha,
            "metadata_sha256": _sha256_file(parent_metadata_path),
            "sparse_components_preserved_exact": True,
            "unchanged_on_disk": True,
        },
        "direct_source_initialization": {
            **import_ledger,
            "device": "cuda",
            "source_path_at_initialization": str(source_path),
            "source_checkpoint_sha256": source_checkpoint_sha,
            "source_checkpoint_bytes": source_checkpoint.stat().st_size,
            "source_config_sha256": _sha256_file(source_config),
            "source_tokenizer_sha256": _sha256_file(source_tokenizer_file),
            "source_parameter_count": source_parameter_count,
            "source_logits_stored": 0,
            "source_hidden_activations_stored": 0,
            "source_generated_output_bytes": 0,
            "source_teacher_tokens": 0,
        },
        "canonical_semantic_abi": {
            "path_at_initialization": str(canonical_abi_path),
            "sha256": _sha256_file(canonical_abi_path),
            "changed": False,
        },
        "physical_sparsity": parent.physical_sparse_contract(),
        "foreign_source_boundary": {
            "teacher_present_at_inference": False,
            "source_parameters_copied": import_ledger[
                "source_parameters_copied_at_initialization"
            ],
            "source_parameters_retained_exact": import_ledger[
                "source_parameters_copied_at_initialization"
            ],
            "source_transformer_blocks_retained": import_ledger[
                "source_transformer_blocks_retained_exact_at_initialization"
            ],
            "source_generated_text_retained_in_deployment": False,
            "teacher_tokenizer_required_at_inference": False,
        },
        "final_test_accessed": False,
        "promotion_eligible": False,
        "claim_boundary": (
            "This is a GPU-created, teacher-free LayerCake initialization "
            "containing exactly copied source weights. It is not deployable or "
            "a quality claim. Promotion requires GPU conformance to eliminate "
            "bit-exact source-block retention and all locked LayerCake gates."
        ),
    }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    (output_path / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if (
        _sha256_file(source_checkpoint) != source_checkpoint_sha
        or _sha256_file(parent_checkpoint) != parent_checkpoint_sha
    ):
        raise DirectSourceInitializationError(
            "source or parent checkpoint changed during initialization"
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-layers", default="1,3,5")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    manifest = build_direct_source_base(
        source_path=args.source,
        parent_path=args.parent,
        layercake_root=args.layercake_root,
        canonical_abi_path=args.canonical_abi,
        output_path=args.output,
        selected_layers=tuple(
            int(value) for value in args.selected_layers.split(",") if value.strip()
        ),
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "foreign_source_boundary": manifest["foreign_source_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
