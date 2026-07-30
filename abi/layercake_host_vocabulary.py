"""Derive a physically sparse LayerCake output vocabulary from train-only text.

The input tokenizer and embedding stay unchanged.  Only columns of the tied
quantized output projection are selected, and an immutable local-to-global
token-ID map restores authoritative runtime IDs after ONNX inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from tokenizers import Tokenizer

from .layercake_host import (
    LayerCakeHostError,
    _canonical_json_bytes,
    load_english_training_rows,
)
from .layercake_host_preservation import _load_general_rows
from .layercake_host_runtime import (
    RUNTIME_FORMAT,
    _canonical_sha,
    _sha256_file,
)


VOCABULARY_FORMAT = "abi-layercake-sparse-output-vocabulary/1"
EXPECTED_VOCAB_SIZE = 50_257
BYTE_FALLBACK_IDS = frozenset(range(256))
EOS_ID = 50_256
BUDGET_IDS = (
    "byte_fallback_only",
    "train_outputs",
    "all_train_text",
    "vocab_4096",
    "vocab_8192",
    "vocab_16384",
    "vocab_32768",
    "full_50257",
)


class SparseOutputVocabularyError(LayerCakeHostError):
    """Raised when a sparse-output artifact cannot be derived exactly."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _token_ids(
    tokenizer: Tokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    prompt_newline: bool = False,
) -> set[int]:
    selected: set[int] = set()
    for row in rows:
        text = str(row[field])
        if prompt_newline:
            text += "\n"
        selected.update(tokenizer.encode(text).ids)
    return selected


def build_nested_output_budgets(
    *,
    tokenizer: Tokenizer,
    abi_rows: Sequence[Mapping[str, Any]],
    general_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    """Build the preregistered train-only nested output-ID frontier."""

    if tokenizer.get_vocab_size(with_added_tokens=True) != EXPECTED_VOCAB_SIZE:
        raise SparseOutputVocabularyError(
            "sparse output protocol requires the locked GPT-2 vocabulary"
        )
    mandatory = set(BYTE_FALLBACK_IDS) | {EOS_ID}
    output_ids = set(mandatory)
    output_ids.update(_token_ids(tokenizer, abi_rows, field="response"))
    output_ids.update(_token_ids(tokenizer, general_rows, field="response"))
    all_train_ids = set(output_ids)
    all_train_ids.update(
        _token_ids(
            tokenizer, abi_rows, field="prompt", prompt_newline=True
        )
    )
    all_train_ids.update(
        _token_ids(
            tokenizer, general_rows, field="prompt", prompt_newline=True
        )
    )
    budgets: dict[str, list[int]] = {
        "byte_fallback_only": sorted(mandatory),
        "train_outputs": sorted(output_ids),
        "all_train_text": sorted(all_train_ids),
    }
    previous = set(all_train_ids)
    for limit in (4096, 8192, 16384, 32768):
        for token_id in range(EXPECTED_VOCAB_SIZE):
            if len(previous) >= limit:
                break
            previous.add(token_id)
        budgets[f"vocab_{limit}"] = sorted(previous)
    budgets["full_50257"] = list(range(EXPECTED_VOCAB_SIZE))
    counts = [len(budgets[name]) for name in BUDGET_IDS]
    if counts != sorted(counts) or any(
        not set(budgets[left]).issubset(budgets[right])
        for left, right in zip(BUDGET_IDS, BUDGET_IDS[1:])
    ):
        raise SparseOutputVocabularyError(
            "constructed output budgets are not nested"
        )
    return budgets


def _replace_initializer(document, name: str, array: np.ndarray) -> None:
    from onnx import numpy_helper

    for index, initializer in enumerate(document.graph.initializer):
        if initializer.name == name:
            document.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(array, name=name)
            )
            return
    raise SparseOutputVocabularyError(
        f"output projection initializer is absent: {name}"
    )


def derive_sparse_output_artifact(
    *,
    source_artifact: str | Path,
    training_bundle_path: str | Path,
    general_curriculum_path: str | Path,
    budget_index: int,
    vocabulary_budget_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Select exact int8 output columns and bind their global token IDs."""

    import onnx
    from onnx import numpy_helper

    source_artifact = Path(source_artifact).resolve()
    training_bundle_path = Path(training_bundle_path).resolve()
    general_curriculum_path = Path(general_curriculum_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise SparseOutputVocabularyError(
            f"sparse-output artifact is immutable: {output_path}"
        )
    if vocabulary_budget_id not in BUDGET_IDS:
        raise SparseOutputVocabularyError(
            f"unknown vocabulary budget: {vocabulary_budget_id}"
        )
    metadata_path = source_artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != RUNTIME_FORMAT:
        raise SparseOutputVocabularyError(
            "source artifact is not an ABI native host"
        )
    source_graph = source_artifact / metadata["runtime"]["graph"]
    if _sha256_file(source_graph) != metadata["runtime"]["graph_sha256"]:
        raise SparseOutputVocabularyError("source runtime graph changed")
    tokenizer_source = (
        source_artifact / metadata["tokenizer"]["path"]
    )
    symbolic_source = (
        source_artifact / metadata["symbolic_surface"]["path"]
    )
    if _sha256_file(tokenizer_source) != metadata["tokenizer"]["sha256"]:
        raise SparseOutputVocabularyError("source tokenizer changed")
    if _sha256_file(symbolic_source) != metadata["symbolic_surface"]["sha256"]:
        raise SparseOutputVocabularyError("source symbolic surface changed")

    tokenizer = Tokenizer.from_file(str(tokenizer_source))
    abi_rows, selected_budget, _ = load_english_training_rows(
        training_bundle_path, budget_index=budget_index
    )
    general_rows = _load_general_rows(
        general_curriculum_path, split="train"
    )
    budgets = build_nested_output_budgets(
        tokenizer=tokenizer,
        abi_rows=abi_rows,
        general_rows=general_rows,
    )
    selected_ids = budgets[vocabulary_budget_id]
    selected = np.asarray(selected_ids, dtype=np.int64)

    document = onnx.load(source_graph)
    arrays = {
        value.name: numpy_helper.to_array(value)
        for value in document.graph.initializer
    }
    projection_candidates = [
        (name, array)
        for name, array in arrays.items()
        if array.shape == (768, EXPECTED_VOCAB_SIZE)
        and array.dtype == np.int8
    ]
    if len(projection_candidates) != 1:
        raise SparseOutputVocabularyError(
            "expected exactly one quantized full output projection"
        )
    projection_name, projection = projection_candidates[0]
    if not projection_name.endswith("_quantized"):
        raise SparseOutputVocabularyError(
            "quantized output projection name is unrecognized"
        )
    base = projection_name[: -len("_quantized")]
    scale_name = base + "_scale"
    zero_name = base + "_zero_point"
    scale = arrays.get(scale_name)
    zero = arrays.get(zero_name)
    if (
        scale is None
        or zero is None
        or scale.shape != (EXPECTED_VOCAB_SIZE,)
        or zero.shape != (EXPECTED_VOCAB_SIZE,)
    ):
        raise SparseOutputVocabularyError(
            "per-channel output projection metadata changed"
        )
    sparse_projection = np.ascontiguousarray(projection[:, selected])
    sparse_scale = np.ascontiguousarray(scale[selected])
    sparse_zero = np.ascontiguousarray(zero[selected])
    _replace_initializer(document, projection_name, sparse_projection)
    _replace_initializer(document, scale_name, sparse_scale)
    _replace_initializer(document, zero_name, sparse_zero)
    logits_outputs = [
        value for value in document.graph.output if value.name == "logits"
    ]
    if len(logits_outputs) != 1:
        raise SparseOutputVocabularyError("graph logits output changed")
    logits_shape = logits_outputs[0].type.tensor_type.shape.dim
    if len(logits_shape) != 2:
        raise SparseOutputVocabularyError("graph logits rank changed")
    logits_shape[1].dim_value = len(selected_ids)

    output_path.mkdir(parents=True, exist_ok=False)
    graph_path = output_path / "model-int8.onnx"
    onnx.checker.check_model(document)
    onnx.save(document, graph_path)
    tokenizer_path = output_path / "tokenizer.json"
    symbolic_path = output_path / "symbolic-surface.json"
    shutil.copyfile(tokenizer_source, tokenizer_path)
    shutil.copyfile(symbolic_source, symbolic_path)

    vocabulary_document: dict[str, Any] = {
        "format": VOCABULARY_FORMAT,
        "status": "DERIVED_FROM_TRAIN_ONLY_TEXT",
        "budget_id": vocabulary_budget_id,
        "budget_index": BUDGET_IDS.index(vocabulary_budget_id),
        "global_token_ids": selected_ids,
        "selected_token_count": len(selected_ids),
        "full_token_count": EXPECTED_VOCAB_SIZE,
        "byte_fallback_ids_complete": (
            BYTE_FALLBACK_IDS.issubset(selected_ids)
        ),
        "eos_id_present": EOS_ID in selected_ids,
        "nested_budget_counts": {
            name: len(budgets[name]) for name in BUDGET_IDS
        },
        "selection_inputs": {
            "training_bundle_sha256": _sha256_file(
                training_bundle_path
            ),
            "english_budget_index": int(budget_index),
            "english_budget_record_count": len(
                selected_budget["record_ids"]
            ),
            "general_curriculum_sha256": _sha256_file(
                general_curriculum_path
            ),
            "general_train_row_count": len(general_rows),
            "validation_rows_seen": 0,
            "benchmark_rows_seen": 0,
            "final_test_rows_seen": 0,
        },
        "source_projection": {
            "initializer": projection_name,
            "shape": list(projection.shape),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(projection).tobytes()
            ).hexdigest(),
        },
        "selected_projection": {
            "shape": list(sparse_projection.shape),
            "sha256": hashlib.sha256(
                sparse_projection.tobytes()
            ).hexdigest(),
            "columns_byte_identical_to_source": bool(
                np.array_equal(
                    sparse_projection, projection[:, selected]
                )
            ),
        },
        "claim_boundary": (
            "This is a preregistered train-only nested vocabulary budget. "
            "Complete byte fallback proves realizability, not fluent quality."
        ),
        "final_test_accessed": False,
    }
    vocabulary_document["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(vocabulary_document)
    ).hexdigest()
    vocabulary_path = output_path / "output-vocabulary.json"
    _write_json(vocabulary_path, vocabulary_document)

    derived = json.loads(json.dumps(metadata))
    derived["status"] = "EXPORTED_NOT_YET_CERTIFIED"
    runtime = derived["runtime"]
    runtime["graph"] = graph_path.name
    runtime["graph_sha256"] = _sha256_file(graph_path)
    runtime["graph_bytes"] = graph_path.stat().st_size
    runtime["source_full_runtime_graph_sha256"] = metadata["runtime"][
        "graph_sha256"
    ]
    runtime["output_vocabulary"] = {
        "format": VOCABULARY_FORMAT,
        "path": vocabulary_path.name,
        "sha256": _sha256_file(vocabulary_path),
        "budget_id": vocabulary_budget_id,
        "selected_token_count": len(selected_ids),
        "full_token_count": EXPECTED_VOCAB_SIZE,
        "byte_fallback_ids_complete": True,
        "global_runtime_token_ids": True,
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
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--training-bundle", required=True)
    parser.add_argument("--general-curriculum", required=True)
    parser.add_argument("--english-budget-index", type=int, required=True)
    parser.add_argument(
        "--vocabulary-budget-id", choices=BUDGET_IDS, required=True
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = derive_sparse_output_artifact(
        source_artifact=args.source_artifact,
        training_bundle_path=args.training_bundle,
        general_curriculum_path=args.general_curriculum,
        budget_index=args.english_budget_index,
        vocabulary_budget_id=args.vocabulary_budget_id,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
