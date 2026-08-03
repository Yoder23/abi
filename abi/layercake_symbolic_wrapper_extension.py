"""Add one knowledge-free request-wrapper conformance handler."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import psutil

from .layercake_core_loader import _load_symbolic_surface_substrate
from .layercake_full_core_acquisition import _manifest_sha
from .layercake_host import _canonical_json_bytes, _sha256_file
from .layercake_symbolic_substrate_extension import EXTENSION_FORMAT


HANDLER = "natural_labeled_event_ordering_preface_v2"
WRAPPER = "I need you to do this: "


class SymbolicWrapperExtensionError(RuntimeError):
    """Raised when the bounded conformance extension is not exact."""


def extend_wrapper_conformance(
    *, parent_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    started = time.perf_counter()
    process = psutil.Process()
    observed_rss = process.memory_info().rss
    parent_path = Path(parent_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise SymbolicWrapperExtensionError(
            f"wrapper artifact is immutable: {output_path}"
        )
    parent_metadata = json.loads(
        (parent_path / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        parent_metadata.get("manifest_sha256")
        != _manifest_sha(parent_metadata)
        or parent_metadata.get("checkpoint", {}).get("sha256")
        != _sha256_file(parent_path / "model.safetensors")
        or parent_metadata.get("foreign_source_boundary", {}).get(
            "teacher_present_at_inference"
        )
        is not False
    ):
        raise SymbolicWrapperExtensionError(
            "parent identity or teacher boundary changed"
        )
    contract = _load_symbolic_surface_substrate(
        parent_path, parent_metadata
    )
    if contract is None or HANDLER in contract.get("handlers", ()):
        raise SymbolicWrapperExtensionError(
            "parent substrate is absent or already contains v2"
        )
    contract = copy.deepcopy(contract)
    parent_handlers = list(contract["handlers"])
    contract["handlers"] = parent_handlers + [HANDLER]
    contract.setdefault("runtime_conformance_rules", {})[HANDLER] = {
        "request_wrapper": WRAPPER,
        "delegates_to": "natural_labeled_event_ordering",
        "imports_semantic_or_domain_information": False,
    }
    contract["source_teacher_text_retained"] = False
    payload = _canonical_json_bytes(contract)
    payload_sha = hashlib.sha256(payload).hexdigest()

    output_path.mkdir(parents=True, exist_ok=False)
    copied_files: dict[str, str] = {}
    for source in sorted(parent_path.iterdir()):
        if (
            not source.is_file()
            or source.name in {"metadata.json", "symbolic_surface.json"}
        ):
            continue
        destination = output_path / source.name
        shutil.copy2(source, destination)
        source_sha = _sha256_file(source)
        if _sha256_file(destination) != source_sha:
            raise SymbolicWrapperExtensionError(
                f"parent file changed during extension: {source.name}"
            )
        copied_files[source.name] = source_sha
    payload_path = output_path / "symbolic_surface.json"
    payload_path.write_bytes(payload)
    if _sha256_file(payload_path) != payload_sha:
        raise SymbolicWrapperExtensionError(
            "wrapper payload changed while writing"
        )

    metadata = copy.deepcopy(parent_metadata)
    metadata["status"] = "DERIVED_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED"
    metadata["symbolic_surface_substrate"] = {
        "format": EXTENSION_FORMAT,
        "path": payload_path.name,
        "payload_bytes": len(payload),
        "payload_sha256": payload_sha,
        "handlers": list(contract["handlers"]),
        "maximum_active_handlers_per_sequence": 1,
        "neural_fallback_checkpoint_sha256": parent_metadata["checkpoint"][
            "sha256"
        ],
        "parent_artifact_manifest_sha256": parent_metadata[
            "manifest_sha256"
        ],
        "parent_symbolic_payload_sha256": parent_metadata[
            "symbolic_surface_substrate"
        ]["payload_sha256"],
        "parent_handlers": parent_handlers,
        "new_handlers": [HANDLER],
        "runtime_conformance_normalization": {
            "request_wrapper": WRAPPER,
            "semantic_or_domain_information_imported": False,
            "delegates_to_unchanged_handler": (
                "natural_labeled_event_ordering"
            ),
        },
        "parent_files_copied_byte_exact": copied_files,
        "imported_information": {
            "source_records": 0,
            "raw_source_prompt_utf8_bytes": 0,
            "unique_source_prompt_utf8_bytes": 0,
            "teacher_generated_output_utf8_bytes": 0,
            "teacher_tokens": 0,
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "source_parameters_copied": 0,
            "final_imported_substrate_parameters": 0,
            "bridge_parameters_trained": 0,
            "teacher_text_bytes_retained": 0,
        },
        "neural_training_performed": False,
        "training_device": None,
        "packaging_device": "cpu",
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
        "This exact artifact adds one knowledge-free, versioned request-wrapper "
        "normalization. It changes no parent neural or tokenizer byte, imports "
        "no teacher information, and is not yet certified."
    )
    metadata["symbolic_surface_substrate"]["packaging_wall_seconds"] = (
        time.perf_counter() - started
    )
    observed_rss = max(observed_rss, process.memory_info().rss)
    metadata["symbolic_surface_substrate"][
        "packaging_observed_peak_rss_bytes"
    ] = observed_rss
    metadata.pop("manifest_sha256", None)
    metadata["manifest_sha256"] = _manifest_sha(metadata)
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    metadata = extend_wrapper_conformance(
        parent_path=args.parent, output_path=args.output
    )
    symbolic = metadata["symbolic_surface_substrate"]
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "manifest_sha256": metadata["manifest_sha256"],
                "payload_sha256": symbolic["payload_sha256"],
                "payload_bytes": symbolic["payload_bytes"],
                "handlers": symbolic["handlers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
