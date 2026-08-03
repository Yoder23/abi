"""Graft one immutable English-form substrate onto an acquired LayerCake core."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from safetensors.torch import load_file

from .layercake_full_core_acquisition import (
    ARTIFACT_FORMAT,
    _manifest_sha,
)
from .layercake_host import (
    SYMBOLIC_SURFACE_STATE_KEY,
    _canonical_json_bytes,
    _decode_symbolic_surface,
    _sha256_file,
    _validate_deployment_manifest,
)


GRAFT_FORMAT = "abi-layercake-symbolic-substrate-graft/1"


class SymbolicSubstrateGraftError(RuntimeError):
    """Raised when a symbolic substrate cannot be grafted exactly."""


def graft_symbolic_substrate(
    *,
    parent_path: str | Path,
    source_host_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    parent_path = Path(parent_path).resolve()
    source_host_path = Path(source_host_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise SymbolicSubstrateGraftError(
            f"graft artifact is immutable: {output_path}"
        )

    parent_metadata_path = parent_path / "metadata.json"
    parent_checkpoint_path = parent_path / "model.safetensors"
    parent_metadata = json.loads(
        parent_metadata_path.read_text(encoding="utf-8")
    )
    if (
        parent_metadata.get("format") != ARTIFACT_FORMAT
        or parent_metadata.get("checkpoint", {}).get("sha256")
        != _sha256_file(parent_checkpoint_path)
        or parent_metadata.get("foreign_source_boundary", {}).get(
            "teacher_present_at_inference"
        )
        is not False
    ):
        raise SymbolicSubstrateGraftError(
            "parent is not an exact teacher-free acquired LayerCake core"
        )

    source_manifest_path = source_host_path / "deployment_manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    _validate_deployment_manifest(source_manifest)
    source_delta_path = (
        source_host_path / source_manifest["host_delta"]["path"]
    )
    source_delta_sha = _sha256_file(source_delta_path)
    symbolic_manifest = source_manifest["host_delta"].get(
        "symbolic_surface", {}
    )
    if (
        source_delta_sha != source_manifest["host_delta"]["sha256"]
        or symbolic_manifest.get("mode")
        != "learned_rules_and_schema_realizers"
        or symbolic_manifest.get("source_teacher_text_retained") is not False
        or source_manifest.get("teacher_present_at_inference") is not False
    ):
        raise SymbolicSubstrateGraftError(
            "source symbolic substrate identity or teacher boundary changed"
        )
    source_state = load_file(str(source_delta_path), device="cpu")
    payload_tensor = source_state.get(SYMBOLIC_SURFACE_STATE_KEY)
    if payload_tensor is None:
        raise SymbolicSubstrateGraftError(
            "source host omitted its declared symbolic payload"
        )
    contract = _decode_symbolic_surface(payload_tensor)
    payload_bytes = _canonical_json_bytes(contract)
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    if (
        payload_sha != symbolic_manifest.get("payload_sha256")
        or len(payload_bytes) != int(symbolic_manifest.get("payload_bytes", -1))
        or list(contract.get("handlers", ()))
        != list(symbolic_manifest.get("handlers", ()))
    ):
        raise SymbolicSubstrateGraftError(
            "decoded symbolic payload does not match its source manifest"
        )

    output_path.mkdir(parents=True, exist_ok=False)
    copied_files: dict[str, str] = {}
    for source in sorted(parent_path.iterdir()):
        if not source.is_file() or source.name == "metadata.json":
            continue
        destination = output_path / source.name
        shutil.copy2(source, destination)
        source_sha = _sha256_file(source)
        if _sha256_file(destination) != source_sha:
            raise SymbolicSubstrateGraftError(
                f"parent file changed during graft: {source.name}"
            )
        copied_files[source.name] = source_sha
    payload_path = output_path / "symbolic_surface.json"
    payload_path.write_bytes(payload_bytes)
    if _sha256_file(payload_path) != payload_sha:
        raise SymbolicSubstrateGraftError(
            "symbolic payload changed during graft"
        )

    metadata = copy.deepcopy(parent_metadata)
    metadata["status"] = "DERIVED_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED"
    metadata["checkpoint"]["sha256"] = _sha256_file(
        output_path / metadata["checkpoint"]["path"]
    )
    metadata["symbolic_surface_substrate"] = {
        "format": GRAFT_FORMAT,
        "path": payload_path.name,
        "payload_bytes": len(payload_bytes),
        "payload_sha256": payload_sha,
        "handlers": list(contract["handlers"]),
        "maximum_active_handlers_per_sequence": 1,
        "neural_fallback_checkpoint_sha256": (
            parent_metadata["checkpoint"]["sha256"]
        ),
        "parent_files_copied_byte_exact": copied_files,
        "source_host_path_at_graft": str(source_host_path),
        "source_host_manifest_file_sha256": _sha256_file(
            source_manifest_path
        ),
        "source_host_manifest_sha256": source_manifest["manifest_sha256"],
        "source_host_delta_sha256": source_delta_sha,
        "source_neural_parameters_copied": 0,
        "source_task_cakes_copied": 0,
        "source_classifier_parameters_copied": 0,
        "trained_parameters": 0,
        "source_teacher_text_retained": False,
        "teacher_present_at_inference": False,
        "canonical_abi_changed": False,
    }
    metadata["acquired_core"]["graph_topology_changed"] = True
    metadata["acquired_core"]["parameter_shapes_changed"] = False
    metadata["claim_boundary"] = (
        "This exact graft keeps every v51 neural and tokenizer file byte-identical "
        "and adds only one hash-bound, teacher-free English-form capability "
        "substrate. It is not semantic or runtime certification."
    )
    metadata.pop("manifest_sha256", None)
    metadata["manifest_sha256"] = _manifest_sha(metadata)
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if _sha256_file(parent_checkpoint_path) != metadata["checkpoint"]["sha256"]:
        raise SymbolicSubstrateGraftError(
            "grafted checkpoint is not byte-identical to its parent"
        )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    metadata = graft_symbolic_substrate(
        parent_path=args.parent,
        source_host_path=args.source_host,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "manifest_sha256": metadata["manifest_sha256"],
                "payload_sha256": metadata["symbolic_surface_substrate"][
                    "payload_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
