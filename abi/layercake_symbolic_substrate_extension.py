"""Extend an immutable LayerCake English-form substrate from training data."""

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
from .layercake_host import (
    _build_symbolic_surface,
    _canonical_json_bytes,
    _natural_concise_statement_combination,
    _natural_ordered_event_labels,
    _sha256_file,
)
from .layercake_host_v3 import load_english_training_rows


EXTENSION_FORMAT = "abi-layercake-symbolic-substrate-extension/1"
NEW_HANDLERS = (
    "natural_labeled_event_ordering",
    "natural_concise_statement_combination",
)


class SymbolicSubstrateExtensionError(RuntimeError):
    """Raised when a symbolic substrate extension is not exact and bounded."""


def _utf8_size(values: Sequence[str], *, unique: bool) -> int:
    selected = set(values) if unique else values
    return sum(len(value.encode("utf-8")) for value in selected)


def extend_symbolic_substrate(
    *,
    parent_path: str | Path,
    training_bundle_path: str | Path,
    output_path: str | Path,
    budget_index: int = -1,
) -> dict[str, Any]:
    started = time.perf_counter()
    process = psutil.Process()
    observed_rss = process.memory_info().rss
    parent_path = Path(parent_path).resolve()
    training_bundle_path = Path(training_bundle_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise SymbolicSubstrateExtensionError(
            f"extension artifact is immutable: {output_path}"
        )

    parent_metadata_path = parent_path / "metadata.json"
    parent_metadata = json.loads(
        parent_metadata_path.read_text(encoding="utf-8")
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
        raise SymbolicSubstrateExtensionError(
            "parent core identity or teacher boundary changed"
        )
    parent_contract = _load_symbolic_surface_substrate(
        parent_path, parent_metadata
    )
    if parent_contract is None:
        raise SymbolicSubstrateExtensionError(
            "parent omitted its declared symbolic substrate"
        )

    rows, budget, bundle = load_english_training_rows(
        training_bundle_path, budget_index=budget_index
    )
    support_rows = [
        row
        for row in rows
        if (
            row["capability"] == "coherence"
            and _natural_ordered_event_labels(str(row["prompt"])) is not None
        )
        or (
            row["capability"] == "rewriting"
            and _natural_concise_statement_combination(str(row["prompt"]))
            is not None
        )
    ]
    extracted = _build_symbolic_surface(support_rows)
    if tuple(extracted["handlers"]) != NEW_HANDLERS:
        raise SymbolicSubstrateExtensionError(
            "search-only extraction did not produce the preregistered handlers"
        )
    observed_rss = max(observed_rss, process.memory_info().rss)

    contract = copy.deepcopy(parent_contract)
    parent_handlers = list(contract.get("handlers", ()))
    if set(parent_handlers).intersection(NEW_HANDLERS):
        raise SymbolicSubstrateExtensionError(
            "parent already contains the proposed extension"
        )
    contract["handlers"] = parent_handlers + list(NEW_HANDLERS)
    support = contract.setdefault(
        "schema_supporting_search_records", {}
    )
    extracted_support = extracted["schema_supporting_search_records"]
    for handler in NEW_HANDLERS:
        count = int(extracted_support.get(handler, 0))
        if count <= 0:
            raise SymbolicSubstrateExtensionError(
                f"handler lacks passing search support: {handler}"
            )
        support[handler] = count
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
            raise SymbolicSubstrateExtensionError(
                f"parent file changed during extension: {source.name}"
            )
        copied_files[source.name] = source_sha
    payload_path = output_path / "symbolic_surface.json"
    payload_path.write_bytes(payload)
    if _sha256_file(payload_path) != payload_sha:
        raise SymbolicSubstrateExtensionError(
            "extended symbolic payload changed while writing"
        )

    prompts = [str(row["prompt"]) for row in support_rows]
    outputs = [str(row["response"]) for row in support_rows]
    record_ids = sorted(str(row["record_id"]) for row in support_rows)
    teacher_tokens = sum(int(row["teacher_tokens"]) for row in support_rows)
    source_models = sorted(
        {
            (str(row["source_model"]), str(row["source_model_revision"]))
            for row in support_rows
        }
    )
    source_ledger = bundle["ledger"]
    imported_information = {
        "passing_search_records_examined": len(support_rows),
        "record_id_set_sha256": hashlib.sha256(
            _canonical_json_bytes({"record_ids": record_ids})
        ).hexdigest(),
        "raw_source_prompt_utf8_bytes": _utf8_size(prompts, unique=False),
        "unique_source_prompt_utf8_bytes": _utf8_size(prompts, unique=True),
        "teacher_generated_output_utf8_bytes": _utf8_size(
            outputs, unique=False
        ),
        "unique_teacher_output_utf8_bytes": _utf8_size(
            outputs, unique=True
        ),
        "teacher_tokens": teacher_tokens,
        "teacher_token_counter": "authoritative_generated_token_ids",
        "logits_stored": 0,
        "logit_bytes_stored": 0,
        "hidden_activations_stored": 0,
        "hidden_activation_bytes_stored": 0,
        "source_parameters_copied": 0,
        "source_parameter_bytes_copied": 0,
        "final_imported_substrate_parameters": 0,
        "bridge_parameters_trained": 0,
        "teacher_text_bytes_retained": 0,
        "source_models": [
            {"model": model, "revision": revision}
            for model, revision in source_models
        ],
    }

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
        "new_handlers": list(NEW_HANDLERS),
        "parent_files_copied_byte_exact": copied_files,
        "training_bundle": {
            "path_at_extension": str(training_bundle_path),
            "archive_sha256": bundle["verification"]["archive_sha256"],
            "manifest_sha256": bundle["verification"]["manifest_sha256"],
            "budget_id": budget["budget_id"],
            "budget_record_count": int(budget["record_count"]),
            "budget_teacher_tokens": int(budget["teacher_tokens"]),
            "training_eligible": bundle["verification"]["training_eligible"],
            "domain_segregation_verified": bundle["verification"][
                "domain_segregation_verified"
            ],
        },
        "imported_information": imported_information,
        "one_time_source_extraction": {
            "external_hardware_used": source_ledger[
                "external_hardware_used"
            ],
            "external_hardware_description": source_ledger[
                "external_hardware_description"
            ],
            "source_extraction_devices": source_ledger[
                "source_extraction_devices"
            ],
            "source_model_inference_seconds": source_ledger[
                "source_model_inference_seconds"
            ],
            "source_parameter_count_read": source_ledger[
                "source_parameter_count_read"
            ],
            "source_weight_bytes_read": source_ledger[
                "source_weight_bytes_read"
            ],
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
        "This artifact extends the exact v54 teacher-free substrate using "
        "only passing, segregated search records. It retains no teacher text, "
        "changes no neural or tokenizer byte, and is not yet semantic or "
        "runtime certification."
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
    parser.add_argument("--training-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-index", type=int, default=-1)
    args = parser.parse_args(argv)
    metadata = extend_symbolic_substrate(
        parent_path=args.parent,
        training_bundle_path=args.training_bundle,
        output_path=args.output,
        budget_index=args.budget_index,
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
