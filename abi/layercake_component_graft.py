"""Compose compatible teacher-free LayerCake checkpoints by exact component graft.

This is not foreign-teacher extraction.  It copies named, hash-bound LayerCake
components between checkpoints with identical architecture, tokenizer, and ABI.
The default graft scope is the physically sparse task cakes and their router.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch


FORMAT = "abi-layercake-component-graft/1"
GRAFT_PREFIXES = ("task_cakes.", "task_classifier.")
TOKENIZER_FILES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class ComponentGraftError(RuntimeError):
    """Raised when an exact compatible LayerCake graft cannot be proven."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        if value.dtype is torch.bfloat16:
            value = value.view(torch.int16)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def graft_state_dict(
    base: Mapping[str, torch.Tensor],
    donor: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Return base state with only task cakes and classifier copied from donor."""

    if set(base) != set(donor):
        raise ComponentGraftError("checkpoint tensor names differ")
    selected = sorted(
        name for name in base if name.startswith(GRAFT_PREFIXES)
    )
    if not selected or not any(
        name.startswith("task_classifier.") for name in selected
    ):
        raise ComponentGraftError("task-cake/router graft scope is incomplete")
    output: dict[str, torch.Tensor] = {}
    changed: list[str] = []
    for name in sorted(base):
        base_tensor = base[name]
        donor_tensor = donor[name]
        if base_tensor.shape != donor_tensor.shape or base_tensor.dtype != donor_tensor.dtype:
            raise ComponentGraftError(f"incompatible tensor: {name}")
        source = donor_tensor if name in selected else base_tensor
        output[name] = source.detach().clone()
        if not torch.equal(base_tensor, source):
            changed.append(name)
    if set(changed) - set(selected):
        raise ComponentGraftError("a non-selected tensor changed")
    return output, selected, changed


def compose_layercake_components(
    *,
    base_path: str | Path,
    donor_path: str | Path,
    canonical_abi_path: str | Path,
    output_path: str | Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Create one provenance-bound component-graft checkpoint."""

    if device_name != "cuda" or not torch.cuda.is_available():
        raise ComponentGraftError("ABI component composition requires CUDA")
    base_path = Path(base_path).resolve()
    donor_path = Path(donor_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ComponentGraftError("output already exists")
    if base_path == donor_path:
        raise ComponentGraftError("base and donor must be distinct")

    base_metadata_path = base_path / "metadata.json"
    donor_metadata_path = donor_path / "metadata.json"
    base_checkpoint_path = base_path / "model.safetensors"
    donor_checkpoint_path = donor_path / "model.safetensors"
    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    donor_metadata = json.loads(donor_metadata_path.read_text(encoding="utf-8"))
    for label, metadata, checkpoint in (
        ("base", base_metadata, base_checkpoint_path),
        ("donor", donor_metadata, donor_checkpoint_path),
    ):
        if metadata.get("checkpoint", {}).get("sha256") != _sha256_file(checkpoint):
            raise ComponentGraftError(f"{label} checkpoint identity changed")
        boundary = metadata.get("foreign_source_boundary", {})
        if boundary.get("teacher_present_at_inference") is not False:
            raise ComponentGraftError(f"{label} is not teacher-free")
        if boundary.get("source_transformer_blocks_retained") != 0:
            raise ComponentGraftError(f"{label} retains source blocks")

    if base_metadata.get("architecture") != donor_metadata.get("architecture"):
        raise ComponentGraftError("LayerCake architectures differ")
    if base_metadata.get("tokenizer", {}).get("sha256") != donor_metadata.get(
        "tokenizer", {}
    ).get("sha256"):
        raise ComponentGraftError("LayerCake tokenizers differ")
    canonical_sha = _sha256_file(canonical_abi_path)
    if any(
        metadata.get("canonical_semantic_abi", {}).get("sha256") != canonical_sha
        for metadata in (base_metadata, donor_metadata)
    ):
        raise ComponentGraftError("canonical ABI identity differs")

    base_checkpoint_sha_before = _sha256_file(base_checkpoint_path)
    donor_checkpoint_sha_before = _sha256_file(donor_checkpoint_path)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    base_state = load_file(str(base_checkpoint_path), device=device_name)
    donor_state = load_file(str(donor_checkpoint_path), device=device_name)
    output_state, selected, changed = graft_state_dict(base_state, donor_state)
    selected_parameter_count = sum(output_state[name].numel() for name in selected)
    changed_parameter_count = sum(output_state[name].numel() for name in changed)
    total_parameter_count = sum(value.numel() for value in output_state.values())
    logical_state_sha = _state_sha256(output_state)

    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_path / "model.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in output_state.items()},
        str(checkpoint_path),
    )
    for name in TOKENIZER_FILES:
        source = base_path / name
        if source.is_file():
            shutil.copy2(source, output_path / name)
    elapsed = time.perf_counter() - start
    peak_gpu = int(torch.cuda.max_memory_allocated())
    del base_state, donor_state, output_state
    torch.cuda.empty_cache()

    if _sha256_file(base_checkpoint_path) != base_checkpoint_sha_before:
        raise ComponentGraftError("base changed during composition")
    if _sha256_file(donor_checkpoint_path) != donor_checkpoint_sha_before:
        raise ComponentGraftError("donor changed during composition")

    checkpoint_sha = _sha256_file(checkpoint_path)
    tokenizer_path = output_path / "tokenizer.json"
    manifest = copy.deepcopy(base_metadata)
    manifest.update(
        {
            "format": FORMAT,
            "status": "COMPOSED_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED",
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": checkpoint_sha,
                "bytes": checkpoint_path.stat().st_size,
            },
            "tokenizer": {
                "path": tokenizer_path.name,
                "sha256": _sha256_file(tokenizer_path),
            },
            "canonical_semantic_abi": {
                "path_at_composition": str(canonical_abi_path),
                "sha256": canonical_sha,
                "changed": False,
            },
            "component_graft": {
                "method": "exact_layercake_task_cakes_and_router_graft",
                "device": device_name,
                "base_path": str(base_path),
                "base_checkpoint_sha256": base_checkpoint_sha_before,
                "base_manifest_sha256": base_metadata.get("manifest_sha256"),
                "donor_path": str(donor_path),
                "donor_checkpoint_sha256": donor_checkpoint_sha_before,
                "donor_manifest_sha256": donor_metadata.get("manifest_sha256"),
                "selected_tensor_names": selected,
                "selected_tensor_count": len(selected),
                "selected_parameter_count": int(selected_parameter_count),
                "changed_tensor_names": changed,
                "changed_tensor_count": len(changed),
                "changed_parameter_count": int(changed_parameter_count),
                "all_nonselected_tensors_base_exact": True,
                "all_selected_tensors_donor_exact": True,
                "layercake_parameters_copied": int(selected_parameter_count),
                "foreign_source_parameters_copied": 0,
                "foreign_source_blocks_retained": 0,
                "teacher_present_at_inference": False,
                "wall_seconds": elapsed,
                "gpu_hours": elapsed / 3600.0,
                "peak_gpu_memory_bytes": peak_gpu,
            },
            "claim_boundary": (
                "This artifact exactly grafts compatible teacher-free LayerCake task "
                "cakes and their router onto a hash-bound LayerCake shared core. It is "
                "not foreign-teacher losslessness, semantic certification, or promotion."
            ),
        }
    )
    acquired = manifest.setdefault("acquired_core", {})
    acquired["logical_state_sha256_after"] = logical_state_sha
    acquired["total_parameter_count"] = int(total_parameter_count)
    acquired["graph_topology_changed"] = False
    acquired["parameter_shapes_changed"] = False
    acquired["physical_sparse_topology_preserved"] = True
    foreign = manifest.setdefault("foreign_source_boundary", {})
    foreign.update(
        {
            "teacher_present_at_inference": False,
            "source_transformer_blocks_retained": 0,
            "source_parameters_copied": 0,
            "teacher_tokenizer_required_at_inference": False,
        }
    )
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    (output_path / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    result = compose_layercake_components(
        base_path=args.base,
        donor_path=args.donor,
        canonical_abi_path=args.canonical_abi,
        output_path=args.output,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "checkpoint_sha256": result["checkpoint"]["sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
