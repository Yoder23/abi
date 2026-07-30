"""Fuse prompt-token candidates into a compact train-only output head.

The derived graph removes the duplicate output-projection initializer.  It
gathers int8 rows and row scales from the already-present full input embedding
using a runtime ``allowed_output_ids`` input, then feeds those exact values
through the original MatMulInteger scaling path.
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

from .layercake_host import _canonical_json_bytes
from .layercake_host_runtime import (
    RUNTIME_FORMAT,
    LayerCakeHostRuntimeError,
    _canonical_sha,
    _sha256_file,
)


DYNAMIC_MODE = "train_base_union_prompt_tokens"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _replace_initializer(document, name: str, array: np.ndarray) -> None:
    from onnx import numpy_helper

    for index, initializer in enumerate(document.graph.initializer):
        if initializer.name == name:
            document.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(array, name=name)
            )
            return
    raise LayerCakeHostRuntimeError(
        f"dynamic vocabulary initializer is absent: {name}"
    )


def derive_dynamic_prompt_vocabulary_artifact(
    *,
    base_artifact: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Replace one static sparse head with an exact dynamic gathered head."""

    import onnx
    from onnx import helper, numpy_helper

    base_artifact = Path(base_artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"dynamic vocabulary artifact is immutable: {output_path}"
        )
    metadata = json.loads(
        (base_artifact / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("format") != RUNTIME_FORMAT:
        raise LayerCakeHostRuntimeError(
            "base artifact is not an ABI native host"
        )
    contract = metadata["runtime"].get("output_vocabulary")
    if (
        contract is None
        or contract.get("mode", "static") != "static"
        or int(contract["selected_token_count"]) >= 50_257
    ):
        raise LayerCakeHostRuntimeError(
            "dynamic derivation requires a static sparse-output base"
        )
    source_graph = base_artifact / metadata["runtime"]["graph"]
    vocabulary_source = base_artifact / contract["path"]
    tokenizer_source = base_artifact / metadata["tokenizer"]["path"]
    symbolic_source = (
        base_artifact / metadata["symbolic_surface"]["path"]
    )
    for path, expected in (
        (source_graph, metadata["runtime"]["graph_sha256"]),
        (vocabulary_source, contract["sha256"]),
        (tokenizer_source, metadata["tokenizer"]["sha256"]),
        (symbolic_source, metadata["symbolic_surface"]["sha256"]),
    ):
        if _sha256_file(path) != expected:
            raise LayerCakeHostRuntimeError(
                f"dynamic vocabulary source changed: {path.name}"
            )
    vocabulary = json.loads(
        vocabulary_source.read_text(encoding="utf-8")
    )
    base_ids = [int(value) for value in vocabulary["global_token_ids"]]
    base_count = len(base_ids)
    if (
        base_count != int(contract["selected_token_count"])
        or base_ids != sorted(set(base_ids))
    ):
        raise LayerCakeHostRuntimeError(
            "base vocabulary IDs are not canonical"
        )

    document = onnx.load(source_graph)
    arrays = {
        value.name: numpy_helper.to_array(value)
        for value in document.graph.initializer
    }
    embedding_candidates = [
        (name, array)
        for name, array in arrays.items()
        if array.shape == (50_257, 768) and array.dtype == np.int8
    ]
    head_candidates = [
        (name, array)
        for name, array in arrays.items()
        if array.shape == (768, base_count) and array.dtype == np.int8
    ]
    if len(embedding_candidates) != 1 or len(head_candidates) != 1:
        raise LayerCakeHostRuntimeError(
            "expected one full embedding and one static sparse head"
        )
    embedding_name, embedding = embedding_candidates[0]
    head_name, head = head_candidates[0]
    if not head_name.endswith("_quantized"):
        raise LayerCakeHostRuntimeError(
            "static sparse head name is unrecognized"
        )
    head_base = head_name[: -len("_quantized")]
    head_scale_name = head_base + "_scale"
    head_zero_name = head_base + "_zero_point"
    head_scale = arrays.get(head_scale_name)
    head_zero = arrays.get(head_zero_name)
    embedding_scale_candidates = [
        (name, array)
        for name, array in arrays.items()
        if array.shape == (50_257,)
        and array.dtype == np.float32
        and "row_scales" in name
    ]
    if (
        head_scale is None
        or head_zero is None
        or head_scale.shape != (base_count,)
        or head_zero.shape != (base_count,)
        or len(embedding_scale_candidates) != 1
    ):
        raise LayerCakeHostRuntimeError(
            "static sparse-head scale metadata changed"
        )
    embedding_scale_name, embedding_scales = (
        embedding_scale_candidates[0]
    )
    selected = np.asarray(base_ids, dtype=np.int64)
    if (
        not np.array_equal(head, embedding[selected].T)
        or not np.array_equal(head_scale, embedding_scales[selected])
        or np.unique(head_zero).tolist() != [0]
    ):
        raise LayerCakeHostRuntimeError(
            "input embedding cannot exactly replace sparse output columns"
        )

    allowed_name = "allowed_output_ids"
    document.graph.input.extend(
        [
            helper.make_tensor_value_info(
                allowed_name,
                onnx.TensorProto.INT64,
                ["allowed_output_tokens"],
            )
        ]
    )
    gathered_weight = "dynamic_output_embedding_rows_int8"
    transposed_weight = "dynamic_output_projection_int8"
    gathered_scales = "dynamic_output_projection_scales"
    dynamic_zero_name = "dynamic_output_projection_zero_point"
    gather_weight = helper.make_node(
        "Gather",
        [embedding_name, allowed_name],
        [gathered_weight],
        name="DynamicOutputWeightGather",
        axis=0,
    )
    transpose_weight = helper.make_node(
        "Transpose",
        [gathered_weight],
        [transposed_weight],
        name="DynamicOutputWeightTranspose",
        perm=[1, 0],
    )
    gather_scales = helper.make_node(
        "Gather",
        [embedding_scale_name, allowed_name],
        [gathered_scales],
        name="DynamicOutputScaleGather",
        axis=0,
    )
    document.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.asarray(0, dtype=np.int8),
                name=dynamic_zero_name,
            )
        ]
    )
    nodes = list(document.graph.node)
    scale_node = next(
        (
            node
            for node in nodes
            if node.op_type == "Mul"
            and head_scale_name in node.input
        ),
        None,
    )
    matmul_node = next(
        (
            node
            for node in nodes
            if node.op_type == "MatMulInteger"
            and head_name in node.input
        ),
        None,
    )
    if scale_node is None or matmul_node is None:
        raise LayerCakeHostRuntimeError(
            "static sparse-head execution nodes changed"
        )
    scale_node.input[
        list(scale_node.input).index(head_scale_name)
    ] = gathered_scales
    matmul_node.input[list(matmul_node.input).index(head_name)] = (
        transposed_weight
    )
    matmul_node.input[
        list(matmul_node.input).index(head_zero_name)
    ] = dynamic_zero_name
    insert_at = min(
        nodes.index(scale_node),
        nodes.index(matmul_node),
    )
    for offset, node in enumerate(
        (gather_weight, transpose_weight, gather_scales)
    ):
        document.graph.node.insert(insert_at + offset, node)
    for name in (head_name, head_scale_name, head_zero_name):
        for initializer in list(document.graph.initializer):
            if initializer.name == name:
                document.graph.initializer.remove(initializer)
                break
        else:
            raise LayerCakeHostRuntimeError(
                f"static head initializer was not removed: {name}"
            )
    logits = [
        value for value in document.graph.output if value.name == "logits"
    ]
    if len(logits) != 1:
        raise LayerCakeHostRuntimeError("graph logits output changed")
    width = logits[0].type.tensor_type.shape.dim[1]
    width.ClearField("dim_value")
    width.dim_param = "allowed_output_tokens"

    output_path.mkdir(parents=True, exist_ok=False)
    graph_path = output_path / "model-int8.onnx"
    onnx.checker.check_model(document)
    onnx.save(document, graph_path)
    tokenizer_path = output_path / "tokenizer.json"
    symbolic_path = output_path / "symbolic-surface.json"
    shutil.copyfile(tokenizer_source, tokenizer_path)
    shutil.copyfile(symbolic_source, symbolic_path)

    dynamic_vocabulary = json.loads(json.dumps(vocabulary))
    dynamic_vocabulary["status"] = (
        "DERIVED_TRAIN_BASE_UNION_RUNTIME_PROMPT_IDS"
    )
    dynamic_vocabulary["mode"] = DYNAMIC_MODE
    dynamic_vocabulary["base_global_token_ids"] = base_ids
    dynamic_vocabulary["base_token_count"] = base_count
    dynamic_vocabulary["allowed_output_ids_graph_input"] = allowed_name
    dynamic_vocabulary["prompt_token_ids_added_at_runtime"] = True
    dynamic_vocabulary["teacher_or_validation_ids_added_at_runtime"] = False
    dynamic_vocabulary["duplicate_output_projection_removed"] = True
    dynamic_vocabulary["exact_weight_reuse"] = {
        "input_embedding_initializer": embedding_name,
        "input_embedding_scale_initializer": embedding_scale_name,
        "static_head_int8_transpose_equal": True,
        "static_head_scales_equal": True,
        "static_head_zero_point": 0,
    }
    dynamic_vocabulary["claim_boundary"] = (
        "The runtime candidate set is the train-only base union exact "
        "token IDs already present in the current prompt. This proves no "
        "validation-output or teacher payload is injected at inference."
    )
    dynamic_vocabulary.pop("evidence_sha256", None)
    dynamic_vocabulary["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(dynamic_vocabulary)
    ).hexdigest()
    vocabulary_path = output_path / "output-vocabulary.json"
    _write_json(vocabulary_path, dynamic_vocabulary)

    derived = json.loads(json.dumps(metadata))
    derived["status"] = "EXPORTED_NOT_YET_CERTIFIED"
    runtime = derived["runtime"]
    runtime["graph"] = graph_path.name
    runtime["graph_sha256"] = _sha256_file(graph_path)
    runtime["graph_bytes"] = graph_path.stat().st_size
    runtime["source_static_sparse_runtime_graph_sha256"] = metadata[
        "runtime"
    ]["graph_sha256"]
    runtime["output_vocabulary"] = {
        "format": contract["format"],
        "mode": DYNAMIC_MODE,
        "path": vocabulary_path.name,
        "sha256": _sha256_file(vocabulary_path),
        "budget_id": contract["budget_id"],
        "selected_token_count": base_count,
        "base_token_count": base_count,
        "full_token_count": 50_257,
        "byte_fallback_ids_complete": True,
        "global_runtime_token_ids": True,
        "allowed_output_ids_graph_input": allowed_name,
        "prompt_token_ids_added_at_runtime": True,
        "duplicate_output_projection_removed": True,
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
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = derive_dynamic_prompt_vocabulary_artifact(
        base_artifact=args.base_artifact,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
