"""Build an immutable LayerCake parent by restoring isolated task-cake routes.

This operation performs no training. It copies only explicitly selected
``task_cakes.<route>.`` tensors from a hash-verified donor into a hash-verified
target and proves that every unselected tensor remains byte-identical.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from .layercake_full_core_acquisition import (
    ARTIFACT_FORMAT,
    FullCoreAcquisitionError,
    _manifest_sha,
)
from .layercake_host import _canonical_json_bytes, _sha256_file


TOKENIZER_FILES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.dtype is torch.bfloat16:
        return value.view(torch.int16).numpy().tobytes()
    return value.numpy().tobytes()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _load_bound_artifact(
    path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], str, str]:
    metadata_path = path / "metadata.json"
    checkpoint_path = path / "model.safetensors"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise FullCoreAcquisitionError(
            f"LayerCake artifact is incomplete: {path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint_sha = _sha256_file(checkpoint_path)
    if metadata.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise FullCoreAcquisitionError(
            f"LayerCake checkpoint identity changed: {path}"
        )
    state = load_file(str(checkpoint_path), device="cpu")
    return metadata, state, checkpoint_sha, _sha256_file(metadata_path)


def build_component_rollback_parent(
    *,
    target_path: str | Path,
    donor_path: str | Path,
    output_path: str | Path,
    selected_task_cake_routes: Sequence[int],
) -> dict[str, Any]:
    """Restore selected route cakes without changing any other tensor."""

    target_path = Path(target_path).resolve()
    donor_path = Path(donor_path).resolve()
    output_path = Path(output_path).resolve()
    routes = tuple(int(route) for route in selected_task_cake_routes)
    if not routes or len(set(routes)) != len(routes) or any(
        route < 0 for route in routes
    ):
        raise FullCoreAcquisitionError(
            "rollback routes must be unique non-negative integers"
        )
    if output_path.exists():
        raise FullCoreAcquisitionError(
            f"component rollback artifact is immutable: {output_path}"
        )
    if output_path == target_path or output_path == donor_path:
        raise FullCoreAcquisitionError(
            "component rollback output must differ from both inputs"
        )

    (
        target_metadata,
        target_state,
        target_checkpoint_sha,
        target_metadata_sha,
    ) = _load_bound_artifact(target_path)
    (
        donor_metadata,
        donor_state,
        donor_checkpoint_sha,
        donor_metadata_sha,
    ) = _load_bound_artifact(donor_path)
    if target_metadata.get("architecture") != donor_metadata.get(
        "architecture"
    ):
        raise FullCoreAcquisitionError(
            "component rollback architectures differ"
        )
    if target_metadata.get("canonical_semantic_abi", {}).get(
        "sha256"
    ) != donor_metadata.get("canonical_semantic_abi", {}).get("sha256"):
        raise FullCoreAcquisitionError(
            "component rollback canonical ABI identities differ"
        )
    if set(target_state) != set(donor_state):
        raise FullCoreAcquisitionError(
            "component rollback checkpoint tensor keys differ"
        )
    for name in target_state:
        if (
            target_state[name].shape != donor_state[name].shape
            or target_state[name].dtype != donor_state[name].dtype
        ):
            raise FullCoreAcquisitionError(
                f"component rollback tensor schema differs: {name}"
            )

    tokenizer_hashes: dict[str, str] = {}
    for filename in TOKENIZER_FILES:
        target_file = target_path / filename
        donor_file = donor_path / filename
        if not target_file.is_file() or not donor_file.is_file():
            raise FullCoreAcquisitionError(
                f"component rollback tokenizer file is missing: {filename}"
            )
        target_hash = _sha256_file(target_file)
        if target_hash != _sha256_file(donor_file):
            raise FullCoreAcquisitionError(
                f"component rollback tokenizer differs: {filename}"
            )
        tokenizer_hashes[filename] = target_hash

    selected_prefixes = tuple(
        f"task_cakes.{route}." for route in routes
    )
    selected_names = sorted(
        name
        for name in target_state
        if name.startswith(selected_prefixes)
    )
    if not selected_names:
        raise FullCoreAcquisitionError(
            "rollback routes selected no checkpoint tensors"
        )
    output_state = {
        name: tensor.detach().clone()
        for name, tensor in target_state.items()
    }
    for name in selected_names:
        output_state[name] = donor_state[name].detach().clone()
    changed_names = [
        name
        for name in selected_names
        if not torch.equal(target_state[name], output_state[name])
    ]
    if not changed_names:
        raise FullCoreAcquisitionError(
            "selected donor components already equal the target"
        )

    unselected_names = sorted(set(target_state) - set(selected_names))
    if any(
        not torch.equal(target_state[name], output_state[name])
        for name in unselected_names
    ):
        raise FullCoreAcquisitionError(
            "unselected target tensor changed before persistence"
        )
    if any(
        not torch.equal(donor_state[name], output_state[name])
        for name in selected_names
    ):
        raise FullCoreAcquisitionError(
            "selected output tensor does not equal donor"
        )

    target_state_sha = _state_sha256(target_state)
    donor_state_sha = _state_sha256(donor_state)
    output_state_sha = _state_sha256(output_state)
    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_path / "model.safetensors"
    save_file(
        {
            name: tensor.contiguous()
            for name, tensor in sorted(output_state.items())
        },
        str(checkpoint_path),
    )
    persisted_state = load_file(str(checkpoint_path), device="cpu")
    if _state_sha256(persisted_state) != output_state_sha:
        raise FullCoreAcquisitionError(
            "persisted component rollback state is not identical"
        )
    for filename in TOKENIZER_FILES:
        shutil.copy2(target_path / filename, output_path / filename)
        if _sha256_file(output_path / filename) != tokenizer_hashes[filename]:
            raise FullCoreAcquisitionError(
                f"persisted tokenizer identity changed: {filename}"
            )

    total_parameter_count = sum(
        int(tensor.numel()) for tensor in output_state.values()
    )
    manifest = copy.deepcopy(target_metadata)
    manifest.pop("manifest_sha256", None)
    manifest["format"] = ARTIFACT_FORMAT
    manifest["status"] = (
        "COMPONENT_ROLLBACK_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED"
    )
    manifest["checkpoint"] = {
        "path": checkpoint_path.name,
        "sha256": _sha256_file(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
    }
    manifest["tokenizer"] = {
        "path": "tokenizer.json",
        "sha256": tokenizer_hashes["tokenizer.json"],
    }
    manifest["parent_layercake"] = {
        "path_at_rollback": str(target_path),
        "checkpoint_sha256": target_checkpoint_sha,
        "metadata_sha256": target_metadata_sha,
        "logical_state_sha256_before": target_state_sha,
        "unchanged_on_disk": True,
    }
    acquired_core = dict(manifest.get("acquired_core", {}))
    acquired_core.update(
        {
            "logical_state_sha256_after": output_state_sha,
            "total_parameter_count": total_parameter_count,
            "trainable_parameter_count": 0,
            "frozen_parameter_count": total_parameter_count,
            "trainable_scope": "component_rollback_no_training",
            "trainable_task_cake_routes": [],
            "graph_topology_changed": False,
            "physical_sparse_topology_preserved": True,
        }
    )
    manifest["acquired_core"] = acquired_core
    manifest["component_rollback"] = {
        "operation": "exact_task_cake_route_tensor_replacement",
        "training_performed": False,
        "selected_task_cake_routes": list(routes),
        "selected_tensor_prefixes": list(selected_prefixes),
        "selected_tensor_count": len(selected_names),
        "changed_tensor_count": len(changed_names),
        "changed_tensor_names": changed_names,
        "all_unselected_tensors_byte_identical_to_target": True,
        "all_selected_tensors_byte_identical_to_donor": True,
        "target": {
            "path_at_rollback": str(target_path),
            "checkpoint_sha256": target_checkpoint_sha,
            "metadata_sha256": target_metadata_sha,
            "logical_state_sha256": target_state_sha,
            "unchanged_on_disk": True,
        },
        "donor": {
            "path_at_rollback": str(donor_path),
            "checkpoint_sha256": donor_checkpoint_sha,
            "metadata_sha256": donor_metadata_sha,
            "logical_state_sha256": donor_state_sha,
            "unchanged_on_disk": True,
        },
        "selected_tensor_sha256": {
            name: _tensor_sha256(output_state[name])
            for name in selected_names
        },
        "output_logical_state_sha256": output_state_sha,
        "tokenizer_files_sha256": tokenizer_hashes,
    }
    manifest["claim_boundary"] = (
        "This immutable parent restores only explicitly named LayerCake "
        "task-cake routes from a compatible LayerCake donor. It performs no "
        "teacher extraction or training and earns no semantic, runtime, or "
        "moonshot promotion credit without downstream certification."
    )
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if _sha256_file(target_path / "model.safetensors") != target_checkpoint_sha:
        raise FullCoreAcquisitionError(
            "target checkpoint changed during component rollback"
        )
    if _sha256_file(donor_path / "model.safetensors") != donor_checkpoint_sha:
        raise FullCoreAcquisitionError(
            "donor checkpoint changed during component rollback"
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--routes", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_component_rollback_parent(
            target_path=args.target,
            donor_path=args.donor,
            output_path=args.output,
            selected_task_cake_routes=tuple(
                int(value)
                for value in args.routes.split(",")
                if value.strip()
            ),
        )
    except (FullCoreAcquisitionError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "component_rollback": manifest["component_rollback"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
