"""Bind a nested train-only frequency reserve to a repaired dynamic host."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

from .layercake_host import _canonical_json_bytes
from .layercake_host_runtime import (
    RUNTIME_FORMAT,
    LayerCakeHostRuntimeError,
    _canonical_sha,
    _sha256_file,
)


FREQUENCY_FORMAT = "abi-layercake-frequency-output-vocabulary/1"
BUDGET_ADDITIONS = {
    "parent_frequency_128": 128,
    "parent_frequency_256": 256,
    "parent_frequency_512": 512,
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def derive_frequency_reserve_artifact(
    *,
    repaired_dynamic_artifact: str | Path,
    original_base_vocabulary_path: str | Path,
    parent_trajectory_evidence_path: str | Path,
    budget_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Select frequent parent-train IDs and retain the exact repaired graph."""

    repaired_dynamic_artifact = Path(
        repaired_dynamic_artifact
    ).resolve()
    original_base_vocabulary_path = Path(
        original_base_vocabulary_path
    ).resolve()
    parent_trajectory_evidence_path = Path(
        parent_trajectory_evidence_path
    ).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"frequency-reserve artifact is immutable: {output_path}"
        )
    if budget_id not in BUDGET_ADDITIONS:
        raise LayerCakeHostRuntimeError(
            f"unknown frequency budget: {budget_id}"
        )
    metadata = json.loads(
        (repaired_dynamic_artifact / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata.get("format") != RUNTIME_FORMAT:
        raise LayerCakeHostRuntimeError(
            "repaired dynamic source format changed"
        )
    source_contract = metadata["runtime"]["output_vocabulary"]
    if (
        source_contract.get("mode")
        != "train_base_union_prompt_tokens"
        or int(source_contract["selected_token_count"]) != 2511
    ):
        raise LayerCakeHostRuntimeError(
            "repaired all-unique dynamic source changed"
        )
    graph_source = (
        repaired_dynamic_artifact / metadata["runtime"]["graph"]
    )
    tokenizer_source = (
        repaired_dynamic_artifact / metadata["tokenizer"]["path"]
    )
    symbolic_source = (
        repaired_dynamic_artifact
        / metadata["symbolic_surface"]["path"]
    )
    source_vocabulary_path = (
        repaired_dynamic_artifact / source_contract["path"]
    )
    for path, expected in (
        (graph_source, metadata["runtime"]["graph_sha256"]),
        (tokenizer_source, metadata["tokenizer"]["sha256"]),
        (symbolic_source, metadata["symbolic_surface"]["sha256"]),
        (source_vocabulary_path, source_contract["sha256"]),
    ):
        if _sha256_file(path) != expected:
            raise LayerCakeHostRuntimeError(
                f"repaired frequency source changed: {path.name}"
            )
    base_vocabulary = json.loads(
        original_base_vocabulary_path.read_text(encoding="utf-8")
    )
    base_ids = [
        int(value) for value in base_vocabulary["global_token_ids"]
    ]
    if len(base_ids) != 1469 or base_ids != sorted(set(base_ids)):
        raise LayerCakeHostRuntimeError(
            "original 1,469-token base changed"
        )
    parent_evidence = json.loads(
        parent_trajectory_evidence_path.read_text(encoding="utf-8")
    )
    records = parent_evidence["generation"]["records"]
    if (
        parent_evidence.get("format")
        != "abi-layercake-parent-longform-curriculum/1"
        or len(records) != 128
        or parent_evidence["curriculum"][
            "validation_rows_seen_for_generation"
        ]
        != 0
    ):
        raise LayerCakeHostRuntimeError(
            "parent train-trajectory evidence changed"
        )
    counts = Counter(
        int(token_id)
        for record in records
        for token_id in record["authoritative_generated_token_ids"]
        if int(token_id) not in set(base_ids)
    )
    ordered = [
        token_id
        for token_id, _ in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    addition_count = BUDGET_ADDITIONS[budget_id]
    if len(ordered) < addition_count:
        raise LayerCakeHostRuntimeError(
            "parent trajectories lack the locked frequency depth"
        )
    selected_ids = sorted(set(base_ids) | set(ordered[:addition_count]))
    expected_count = len(base_ids) + addition_count
    if len(selected_ids) != expected_count:
        raise LayerCakeHostRuntimeError(
            "frequency reserve cardinality is not exact"
        )
    total_parent_tokens = sum(counts.values()) + sum(
        1
        for record in records
        for token_id in record["authoritative_generated_token_ids"]
        if int(token_id) in set(base_ids)
    )
    covered_parent_tokens = sum(
        1
        for record in records
        for token_id in record["authoritative_generated_token_ids"]
        if int(token_id) in set(selected_ids)
    )

    output_path.mkdir(parents=True, exist_ok=False)
    graph_path = output_path / "model-int8.onnx"
    tokenizer_path = output_path / "tokenizer.json"
    symbolic_path = output_path / "symbolic-surface.json"
    shutil.copyfile(graph_source, graph_path)
    shutil.copyfile(tokenizer_source, tokenizer_path)
    shutil.copyfile(symbolic_source, symbolic_path)

    vocabulary = json.loads(
        source_vocabulary_path.read_text(encoding="utf-8")
    )
    vocabulary["format"] = FREQUENCY_FORMAT
    vocabulary["status"] = (
        "DERIVED_PARENT_TRAIN_FREQUENCY_BASE_UNION_PROMPT_IDS"
    )
    vocabulary["budget_id"] = budget_id
    vocabulary["global_token_ids"] = selected_ids
    vocabulary["base_global_token_ids"] = selected_ids
    vocabulary["selected_token_count"] = len(selected_ids)
    vocabulary["base_token_count"] = len(selected_ids)
    vocabulary["frequency_selection"] = {
        "original_base_vocabulary_sha256": _sha256_file(
            original_base_vocabulary_path
        ),
        "original_base_token_count": len(base_ids),
        "parent_trajectory_evidence_sha256": parent_evidence[
            "evidence_sha256"
        ],
        "parent_trajectory_evidence_file_sha256": _sha256_file(
            parent_trajectory_evidence_path
        ),
        "added_token_count": addition_count,
        "ordering": "descending_frequency_then_ascending_global_id",
        "parent_token_observations": total_parent_tokens,
        "covered_parent_token_observations": covered_parent_tokens,
        "parent_token_observation_coverage": (
            covered_parent_tokens / total_parent_tokens
        ),
        "selected_ids_sha256": hashlib.sha256(
            _canonical_json_bytes(selected_ids)
        ).hexdigest(),
        "validation_outputs_seen": 0,
        "benchmark_outputs_seen": 0,
        "final_test_rows_seen": 0,
    }
    vocabulary["claim_boundary"] = (
        "This is a nested train-only frequency reserve. It is promoted only "
        "if the exact same repaired host passes every quality and speed gate."
    )
    vocabulary.pop("evidence_sha256", None)
    vocabulary["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(vocabulary)
    ).hexdigest()
    vocabulary_path = output_path / "output-vocabulary.json"
    _write_json(vocabulary_path, vocabulary)

    derived = json.loads(json.dumps(metadata))
    derived["status"] = "EXPORTED_NOT_YET_CERTIFIED"
    runtime = derived["runtime"]
    runtime["graph"] = graph_path.name
    runtime["graph_sha256"] = _sha256_file(graph_path)
    runtime["graph_bytes"] = graph_path.stat().st_size
    runtime["output_vocabulary"] = {
        "format": FREQUENCY_FORMAT,
        "mode": "train_base_union_prompt_tokens",
        "path": vocabulary_path.name,
        "sha256": _sha256_file(vocabulary_path),
        "budget_id": budget_id,
        "selected_token_count": len(selected_ids),
        "base_token_count": len(selected_ids),
        "full_token_count": 50_257,
        "byte_fallback_ids_complete": True,
        "global_runtime_token_ids": True,
        "allowed_output_ids_graph_input": "allowed_output_ids",
        "prompt_token_ids_added_at_runtime": True,
        "duplicate_output_projection_removed": True,
        "parent_token_observation_coverage": (
            covered_parent_tokens / total_parent_tokens
        ),
    }
    derived["tokenizer"]["path"] = tokenizer_path.name
    derived["tokenizer"]["sha256"] = _sha256_file(tokenizer_path)
    derived["symbolic_surface"]["path"] = symbolic_path.name
    derived["symbolic_surface"]["sha256"] = _sha256_file(symbolic_path)
    derived["evidence_sha256"] = _canonical_sha(derived)
    _write_json(output_path / "metadata.json", derived)
    return derived


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repaired-dynamic-artifact", required=True)
    parser.add_argument("--original-base-vocabulary", required=True)
    parser.add_argument("--parent-trajectory-evidence", required=True)
    parser.add_argument(
        "--budget-id", choices=tuple(BUDGET_ADDITIONS), required=True
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = derive_frequency_reserve_artifact(
        repaired_dynamic_artifact=args.repaired_dynamic_artifact,
        original_base_vocabulary_path=args.original_base_vocabulary,
        parent_trajectory_evidence_path=args.parent_trajectory_evidence,
        budget_id=args.budget_id,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
