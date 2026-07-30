"""Create hash-bound mixed-precision variants of one native ABI host."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from .layercake_host_runtime import (
    LayerCakeHostRuntimeError,
    RUNTIME_FORMAT,
    _canonical_sha,
    _quantize_embedding_rows,
    _sha256_file,
)


LOCKED_CANDIDATES: dict[str, dict[str, Any]] = {
    "block0-fp32": {
        "nodes": [
            "/transformer/h.0/attn/c_attn/Gemm_MatMul",
            "/transformer/h.0/attn/c_proj/Gemm_MatMul",
            "/transformer/h.0/mlp/c_fc/Gemm_MatMul",
            "/transformer/h.0/mlp/c_proj/Gemm_MatMul",
        ],
        "embedding": "int8_per_row",
    },
    "block1-fp32": {
        "nodes": [
            "/transformer/h.1/attn/c_attn/Gemm_MatMul",
            "/transformer/h.1/attn/c_proj/Gemm_MatMul",
            "/transformer/h.1/mlp/c_fc/Gemm_MatMul",
            "/transformer/h.1/mlp/c_proj/Gemm_MatMul",
        ],
        "embedding": "int8_per_row",
    },
    "block2-fp32": {
        "nodes": [
            "/transformer/h.2/attn/c_attn/Gemm_MatMul",
            "/transformer/h.2/attn/c_proj/Gemm_MatMul",
            "/transformer/h.2/mlp/c_fc/Gemm_MatMul",
            "/transformer/h.2/mlp/c_proj/Gemm_MatMul",
        ],
        "embedding": "int8_per_row",
    },
    "classifier-fp32": {
        "nodes": ["/task_classifier/MatMul"],
        "embedding": "int8_per_row",
    },
    "output-head-fp32": {
        "nodes": ["/MatMul_2"],
        "embedding": "int8_per_row",
    },
    "input-embedding-fp16": {
        "nodes": [],
        "embedding": "fp16",
    },
    "input-embedding-global-int8": {
        "nodes": [],
        "embedding": "int8_global",
    },
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _replace_input_embedding(
    document,
    *,
    precision: str,
) -> None:
    import onnx
    from onnx import helper, numpy_helper

    initializers = {
        initializer.name: initializer
        for initializer in document.graph.initializer
    }
    embedding_name = None
    embedding_gather = None
    for node in document.graph.node:
        if node.op_type != "Gather" or not node.input:
            continue
        initializer = initializers.get(node.input[0])
        if initializer is None:
            continue
        array = numpy_helper.to_array(initializer)
        if array.shape == (50257, 768):
            embedding_name = initializer.name
            embedding_gather = node
            break
    if embedding_name is None or embedding_gather is None:
        raise LayerCakeHostRuntimeError(
            "could not locate exported token embedding Gather"
        )
    embedding = numpy_helper.to_array(
        initializers[embedding_name]
    ).astype(np.float32)
    original_output = embedding_gather.output[0]
    node_index = list(document.graph.node).index(embedding_gather)
    if precision == "int8_per_row":
        quantized, scales = _quantize_embedding_rows(embedding)
        quantized_name = embedding_name + "_runtime_int8"
        scales_name = embedding_name + "_runtime_row_scales"
        quantized_output = original_output + "_runtime_int8"
        cast_output = original_output + "_runtime_float"
        scale_output = original_output + "_runtime_scale"
        expanded_scale_output = scale_output + "_expanded"
        axes_name = embedding_name + "_runtime_scale_axes"
        document.graph.initializer.extend(
            (
                numpy_helper.from_array(quantized, name=quantized_name),
                numpy_helper.from_array(scales, name=scales_name),
                numpy_helper.from_array(
                    np.asarray([-1], dtype=np.int64), name=axes_name
                ),
            )
        )
        embedding_gather.input[0] = quantized_name
        embedding_gather.output[0] = quantized_output
        additions = (
            helper.make_node(
                "Gather",
                [scales_name, embedding_gather.input[1]],
                [scale_output],
                name="RuntimeEmbeddingScaleGather",
            ),
            helper.make_node(
                "Cast",
                [quantized_output],
                [cast_output],
                name="RuntimeEmbeddingCast",
                to=onnx.TensorProto.FLOAT,
            ),
            helper.make_node(
                "Unsqueeze",
                [scale_output, axes_name],
                [expanded_scale_output],
                name="RuntimeEmbeddingScaleExpand",
            ),
            helper.make_node(
                "Mul",
                [cast_output, expanded_scale_output],
                [original_output],
                name="RuntimeEmbeddingRowDequantize",
            ),
        )
    elif precision == "int8_global":
        scale = np.float32(
            max(float(np.abs(embedding).max()) / 127.0, 1.0e-8)
        )
        quantized = np.clip(
            np.rint(embedding / scale), -127, 127
        ).astype(np.int8)
        quantized_name = embedding_name + "_runtime_int8"
        scale_name = embedding_name + "_runtime_scale"
        zero_name = embedding_name + "_runtime_zero"
        quantized_output = original_output + "_runtime_int8"
        document.graph.initializer.extend(
            (
                numpy_helper.from_array(
                    quantized, name=quantized_name
                ),
                numpy_helper.from_array(
                    np.asarray(scale), name=scale_name
                ),
                numpy_helper.from_array(
                    np.asarray(0, dtype=np.int8), name=zero_name
                ),
            )
        )
        embedding_gather.input[0] = quantized_name
        embedding_gather.output[0] = quantized_output
        additions = (
            helper.make_node(
                "DequantizeLinear",
                [quantized_output, scale_name, zero_name],
                [original_output],
                name="RuntimeEmbeddingDequantize",
            ),
        )
    elif precision == "fp16":
        fp16_name = embedding_name + "_runtime_fp16"
        fp16_output = original_output + "_runtime_fp16"
        document.graph.initializer.extend(
            (
                numpy_helper.from_array(
                    embedding.astype(np.float16), name=fp16_name
                ),
            )
        )
        embedding_gather.input[0] = fp16_name
        embedding_gather.output[0] = fp16_output
        additions = (
            helper.make_node(
                "Cast",
                [fp16_output],
                [original_output],
                name="RuntimeEmbeddingFP16Cast",
                to=onnx.TensorProto.FLOAT,
            ),
        )
    else:
        raise LayerCakeHostRuntimeError(
            f"unsupported embedding precision: {precision}"
        )
    for offset, node in enumerate(additions, start=1):
        document.graph.node.insert(node_index + offset, node)
    remaining_inputs = {
        value for node in document.graph.node for value in node.input
    }
    if embedding_name not in remaining_inputs:
        document.graph.initializer.remove(initializers[embedding_name])


def create_precision_variant(
    *,
    base_artifact: str | Path,
    output_path: str | Path,
    candidate_id: str,
) -> dict[str, Any]:
    """Requantize the base FP32 graph with one locked precision change."""

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if candidate_id not in LOCKED_CANDIDATES:
        raise LayerCakeHostRuntimeError(
            f"candidate is not preregistered: {candidate_id}"
        )
    base_artifact = Path(base_artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"precision artifact is immutable: {output_path}"
        )
    metadata_path = base_artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != RUNTIME_FORMAT:
        raise LayerCakeHostRuntimeError(
            "base artifact is not a native ABI host"
        )
    fp32_path = base_artifact / "model-fp32.onnx"
    if not fp32_path.is_file():
        raise LayerCakeHostRuntimeError(
            "base artifact lacks its FP32 export graph"
        )
    candidate = LOCKED_CANDIDATES[candidate_id]
    output_path.mkdir(parents=True, exist_ok=False)
    intermediate = output_path / "model-int8-matmul.onnx"
    quantize_dynamic(
        fp32_path,
        intermediate,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_exclude=list(candidate["nodes"]),
    )
    document = onnx.load(intermediate)
    _replace_input_embedding(
        document, precision=str(candidate["embedding"])
    )
    graph_path = output_path / "model-int8.onnx"
    onnx.checker.check_model(document)
    onnx.save(document, graph_path)
    intermediate.unlink()
    for name in ("tokenizer.json", "symbolic-surface.json"):
        shutil.copyfile(base_artifact / name, output_path / name)

    updated = copy.deepcopy(metadata)
    updated["status"] = "PRECISION_VARIANT_NOT_YET_CERTIFIED"
    updated["runtime"]["graph"] = graph_path.name
    updated["runtime"]["graph_sha256"] = _sha256_file(graph_path)
    updated["runtime"]["graph_bytes"] = graph_path.stat().st_size
    updated["runtime"]["matrix_weight_quantization"] = (
        "dynamic signed int8 per channel with preregistered FP32 exclusions"
    )
    updated["runtime"]["embedding_quantization"] = (
        (
            "signed int8 per token row, gathered before dequantization"
            if candidate["embedding"] == "int8_per_row"
            else (
                "signed int8 per tensor, gathered before dequantization"
                if candidate["embedding"] == "int8_global"
                else "IEEE fp16 per token row, gathered then cast to fp32"
            )
        )
    )
    updated["precision_localization"] = {
        "format": "abi-layercake-native-precision-variant/1",
        "candidate_id": candidate_id,
        "base_artifact": str(base_artifact),
        "base_runtime_graph_sha256": metadata["runtime"][
            "graph_sha256"
        ],
        "matrix_nodes_left_fp32": list(candidate["nodes"]),
        "embedding_precision": candidate["embedding"],
        "validation_prompts_seen_during_export": 0,
        "final_test_accessed": False,
    }
    updated["final_test_accessed"] = False
    updated.pop("evidence_sha256", None)
    updated["evidence_sha256"] = _canonical_sha(updated)
    _write_json(output_path / "metadata.json", updated)
    return updated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate", required=True, choices=tuple(LOCKED_CANDIDATES)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = create_precision_variant(
        base_artifact=args.base_artifact,
        output_path=args.output,
        candidate_id=args.candidate,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_sha256": result["evidence_sha256"],
                "runtime": result["runtime"],
                "precision_localization": result[
                    "precision_localization"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
