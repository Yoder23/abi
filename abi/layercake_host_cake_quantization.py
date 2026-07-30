"""Quantize the two route-selected task-cake projections to int8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .layercake_host_runtime import (
    RUNTIME_FORMAT,
    LayerCakeHostRuntimeError,
    _canonical_sha,
    _sha256_file,
)


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
        f"cake initializer is absent: {name}"
    )


def _quantize_route_rows(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if weights.ndim != 3 or weights.shape[0] != 10:
        raise LayerCakeHostRuntimeError(
            "route cake weight must be a 10-route rank-3 tensor"
        )
    scales = np.maximum(
        np.max(np.abs(weights), axis=2) / np.float32(127.0),
        np.float32(1.0e-8),
    ).astype(np.float32)
    quantized = np.clip(
        np.rint(weights / scales[:, :, None]), -127, 127
    ).astype(np.int8)
    return quantized, scales


def derive_int8_cake_artifact(
    *,
    source_artifact: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Replace the two gathered FP32 cake MatMuls with int8 equivalents."""

    import onnx
    from onnx import helper, numpy_helper

    source_artifact = Path(source_artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"int8 cake artifact is immutable: {output_path}"
        )
    metadata = json.loads(
        (source_artifact / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("format") != RUNTIME_FORMAT:
        raise LayerCakeHostRuntimeError(
            "int8 cake source is not an ABI native host"
        )
    if metadata["runtime"].get("cake_activation_schedule") is not None:
        raise LayerCakeHostRuntimeError(
            "int8 cake fusion requires an always-active source cake"
        )
    graph_source = source_artifact / metadata["runtime"]["graph"]
    tokenizer_source = source_artifact / metadata["tokenizer"]["path"]
    symbolic_source = (
        source_artifact / metadata["symbolic_surface"]["path"]
    )
    components = [
        (graph_source, metadata["runtime"]["graph_sha256"]),
        (tokenizer_source, metadata["tokenizer"]["sha256"]),
        (symbolic_source, metadata["symbolic_surface"]["sha256"]),
    ]
    output_vocabulary = metadata["runtime"].get("output_vocabulary")
    vocabulary_source = None
    if output_vocabulary is not None:
        vocabulary_source = source_artifact / output_vocabulary["path"]
        components.append(
            (vocabulary_source, output_vocabulary["sha256"])
        )
    for path, expected in components:
        if _sha256_file(path) != expected:
            raise LayerCakeHostRuntimeError(
                f"int8 cake source changed: {path.name}"
            )

    document = onnx.load(graph_source)
    arrays = {
        value.name: numpy_helper.to_array(value)
        for value in document.graph.initializer
    }
    specifications = [
        {
            "id": "down",
            "weight": "onnx::Gather_1061",
            "gather": "/Gather_3",
            "transpose": "/Transpose",
            "matmul": "/MatMul",
            "expected_shape": (10, 64, 768),
        },
        {
            "id": "up",
            "weight": "onnx::Gather_1072",
            "gather": "/Gather_4",
            "transpose": "/Transpose_1",
            "matmul": "/MatMul_1",
            "expected_shape": (10, 768, 64),
        },
    ]
    node_by_name = {node.name: node for node in document.graph.node}
    zero_name = "TaskCakeInt8WeightZeroPoint"
    document.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.asarray(0, dtype=np.int8), name=zero_name
            )
        ]
    )
    quantization_evidence: dict[str, Any] = {}
    for specification in specifications:
        weight_name = specification["weight"]
        weights = arrays.get(weight_name)
        if (
            weights is None
            or weights.shape != specification["expected_shape"]
            or weights.dtype != np.float32
        ):
            raise LayerCakeHostRuntimeError(
                f"{specification['id']} cake weight changed"
            )
        gather = node_by_name.get(specification["gather"])
        transpose = node_by_name.get(specification["transpose"])
        matmul = node_by_name.get(specification["matmul"])
        if (
            gather is None
            or transpose is None
            or matmul is None
            or gather.op_type != "Gather"
            or list(gather.input) != [weight_name, "route"]
            or transpose.op_type != "Transpose"
            or list(transpose.input) != [gather.output[0]]
            or list(matmul.input)[1] != transpose.output[0]
        ):
            raise LayerCakeHostRuntimeError(
                f"{specification['id']} cake execution boundary changed"
            )
        quantized, scales = _quantize_route_rows(weights)
        _replace_initializer(document, weight_name, quantized)
        int8_intermediates = {
            gather.output[0],
            transpose.output[0],
        }
        for value_info in document.graph.value_info:
            if value_info.name in int8_intermediates:
                value_info.type.tensor_type.elem_type = (
                    onnx.TensorProto.INT8
                )
        scale_name = f"TaskCake{specification['id'].title()}Scales"
        selected_scale = (
            f"TaskCake{specification['id'].title()}SelectedScales"
        )
        document.graph.initializer.extend(
            [numpy_helper.from_array(scales, name=scale_name)]
        )
        prefix = f"TaskCake{specification['id'].title()}"
        activation_q = prefix + "ActivationQuantized"
        activation_scale = prefix + "ActivationScale"
        activation_zero = prefix + "ActivationZeroPoint"
        integer_output = prefix + "IntegerOutput"
        float_output = prefix + "FloatOutput"
        activation_scaled = prefix + "ActivationScaledOutput"
        replacement = [
            helper.make_node(
                "Gather",
                [scale_name, "route"],
                [selected_scale],
                name=prefix + "ScaleGather",
                axis=0,
            ),
            helper.make_node(
                "DynamicQuantizeLinear",
                [matmul.input[0]],
                [activation_q, activation_scale, activation_zero],
                name=prefix + "ActivationQuantize",
            ),
            helper.make_node(
                "MatMulInteger",
                [
                    activation_q,
                    matmul.input[1],
                    activation_zero,
                    zero_name,
                ],
                [integer_output],
                name=prefix + "MatMulInteger",
            ),
            helper.make_node(
                "Cast",
                [integer_output],
                [float_output],
                name=prefix + "IntegerCast",
                to=onnx.TensorProto.FLOAT,
            ),
            helper.make_node(
                "Mul",
                [float_output, activation_scale],
                [activation_scaled],
                name=prefix + "ActivationScaleApply",
            ),
            helper.make_node(
                "Mul",
                [activation_scaled, selected_scale],
                [matmul.output[0]],
                name=prefix + "WeightScaleApply",
            ),
        ]
        nodes = list(document.graph.node)
        matmul_index = nodes.index(matmul)
        document.graph.node.remove(matmul)
        for offset, node in enumerate(replacement):
            document.graph.node.insert(matmul_index + offset, node)
        quantization_evidence[specification["id"]] = {
            "source_shape": list(weights.shape),
            "quantized_dtype": "int8",
            "scale_shape": list(scales.shape),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
            "maximum_absolute_weight_error": float(
                np.max(
                    np.abs(
                        weights
                        - quantized.astype(np.float32)
                        * scales[:, :, None]
                    )
                )
            ),
        }

    onnx.checker.check_model(document)
    output_path.mkdir(parents=True, exist_ok=False)
    graph_path = output_path / "model-int8.onnx"
    onnx.save(document, graph_path)
    tokenizer_path = output_path / "tokenizer.json"
    symbolic_path = output_path / "symbolic-surface.json"
    shutil.copyfile(tokenizer_source, tokenizer_path)
    shutil.copyfile(symbolic_source, symbolic_path)
    if vocabulary_source is not None:
        shutil.copyfile(
            vocabulary_source,
            output_path / vocabulary_source.name,
        )

    derived = json.loads(json.dumps(metadata))
    derived["status"] = "EXPORTED_NOT_YET_CERTIFIED"
    runtime = derived["runtime"]
    runtime["graph"] = graph_path.name
    runtime["graph_sha256"] = _sha256_file(graph_path)
    runtime["graph_bytes"] = graph_path.stat().st_size
    runtime["source_fp32_cake_runtime_graph_sha256"] = metadata[
        "runtime"
    ]["graph_sha256"]
    runtime["task_cake_projection_quantization"] = {
        "format": "abi-layercake-route-cake-int8/1",
        "route_selection_before_matrix_execution": True,
        "weight_quantization": (
            "signed int8 per route and output channel"
        ),
        "activation_quantization": (
            "ONNX DynamicQuantizeLinear per active call"
        ),
        "matrix_execution": "MatMulInteger",
        "details": quantization_evidence,
        "weights_retrained": False,
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = derive_int8_cake_artifact(
        source_artifact=args.source_artifact,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
