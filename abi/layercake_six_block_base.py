"""Build the preregistered six-block LayerCake capacity diagnostic base."""

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

from .artifacts import module_state_sha256
from .layercake_host import (
    _canonical_json_bytes,
    _import_layercake_runtime,
    _is_within,
    _sha256_file,
)
from .layercake_core_loader import (
    ABIEnglishCoreConfig,
    load_layercake_core,
)


BASE_FORMAT = "abi-layercake-six-block-capacity-base/1"


def _manifest_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def build_six_block_base(
    *,
    layercake_root: str | Path,
    three_block_parent_path: str | Path,
    six_block_control_path: str | Path,
    canonical_abi_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    layercake_root = Path(layercake_root).resolve()
    three_block_parent_path = Path(three_block_parent_path).resolve()
    six_block_control_path = Path(six_block_control_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    output_path = Path(output_path).resolve()
    if not _is_within(three_block_parent_path, layercake_root):
        raise RuntimeError("three-block parent must belong to LayerCake")
    if not _is_within(six_block_control_path, layercake_root):
        raise RuntimeError("six-block control must belong to LayerCake")
    if _is_within(output_path, layercake_root):
        raise RuntimeError("ABI may not modify the sealed LayerCake tree")
    if output_path.exists():
        raise RuntimeError(f"base artifact is immutable: {output_path}")
    if not canonical_abi_path.is_file():
        raise RuntimeError("canonical semantic ABI is missing")

    _import_layercake_runtime(layercake_root)
    from layercake.models.shallow_sparse_english import ShallowSparseEnglishCore

    parent, _, parent_metadata = load_layercake_core(
        three_block_parent_path,
        layercake_root=layercake_root,
        device="cpu",
    )
    control_metadata_path = six_block_control_path / "metadata.json"
    control_metadata = json.loads(
        control_metadata_path.read_text(encoding="utf-8")
    )
    control_checkpoint_path = six_block_control_path / "model.safetensors"
    control_checkpoint_sha = _sha256_file(control_checkpoint_path)
    if (
        control_metadata.get("format")
        != "layercake-phase2-nonpromotable-capacity-control/1"
        or control_metadata["checkpoint"]["sha256"] != control_checkpoint_sha
        or int(control_metadata["parameters"]) != 81_912_576
    ):
        raise RuntimeError("six-block capacity control identity changed")
    control = AutoModelForCausalLM.from_pretrained(
        six_block_control_path, local_files_only=True
    ).eval()
    if int(control.config.n_layer) != 6 or int(control.config.n_embd) != 768:
        raise RuntimeError("six-block control architecture changed")

    architecture = copy.deepcopy(parent_metadata["architecture"])
    architecture["layers"] = 6
    architecture["architecture_version"] = (
        "layercake-shallow-sparse-english/2-six-block-task-cakes"
    )
    model = ShallowSparseEnglishCore(
        ABIEnglishCoreConfig(**architecture)
    )
    with torch.no_grad():
        model.transformer.wte.weight.copy_(
            control.transformer.wte.weight
        )
        model.transformer.wpe.weight.copy_(
            control.transformer.wpe.weight
        )
        model.transformer.ln_f.load_state_dict(
            control.transformer.ln_f.state_dict()
        )
        for target, source in zip(
            model.transformer.h, control.transformer.h, strict=True
        ):
            target.load_state_dict(source.state_dict())
        model.task_classifier.load_state_dict(
            parent.task_classifier.state_dict()
        )
        model.task_cakes.load_state_dict(parent.task_cakes.state_dict())
    model.eval()

    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_path / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(checkpoint_path),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        six_block_control_path, local_files_only=True
    )
    tokenizer.save_pretrained(output_path)
    tokenizer_path = output_path / "tokenizer.json"
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    active_parameters = int(model.active_parameter_count())
    manifest: dict[str, Any] = {
        "format": BASE_FORMAT,
        "status": "INITIALIZED_CAPACITY_DIAGNOSTIC_NOT_TRAINED_OR_CERTIFIED",
        "architecture": model.config.canonical_dict(),
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
        "physical_sparsity": model.physical_sparse_contract(),
        "incremental_state": {
            "implemented": True,
            "mechanism": "six-block GPT-2-compatible KV cache plus cached task route",
        },
        "initialization": {
            "kind": "six_block_capacity_control_plus_existing_layercake_task_cakes",
            "logical_state_sha256": module_state_sha256(model),
            "three_block_parent": {
                "path_at_creation": str(three_block_parent_path),
                "checkpoint_sha256": _sha256_file(
                    three_block_parent_path / "model.safetensors"
                ),
                "task_classifier_and_cakes_copied": True,
                "transformer_blocks_copied": 0,
            },
            "six_block_control": {
                "path_at_creation": str(six_block_control_path),
                "checkpoint_sha256": control_checkpoint_sha,
                "metadata_sha256": _sha256_file(control_metadata_path),
                "transformer_blocks_copied": 6,
                "source_kind": "preexisting_layercake_capacity_control_initialized_from_distilgpt2",
            },
            "current_phi3_teacher_loaded": False,
            "current_phi3_teacher_parameters_copied": 0,
        },
        "canonical_semantic_abi": {
            "path_at_creation": str(canonical_abi_path),
            "sha256": _sha256_file(canonical_abi_path),
            "changed": False,
        },
        "quality": None,
        "training": None,
        "test_accessed": False,
        "claim_boundary": (
            "This is a materially larger sequential-capacity diagnostic. It "
            "retains six blocks from the already disclosed LayerCake capacity "
            "control and is neither a small-bridge acquisition result nor a "
            "promoted speed/quality candidate."
        ),
    }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    (output_path / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--three-block-parent", required=True)
    parser.add_argument("--six-block-control", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest = build_six_block_base(
        layercake_root=args.layercake_root,
        three_block_parent_path=args.three_block_parent,
        six_block_control_path=args.six_block_control,
        canonical_abi_path=args.canonical_abi,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "parameters": manifest["parameters"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
