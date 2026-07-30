"""Derive route-local FP32/int8 task-cake projection execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .layercake_host_cake_quantization import _quantize_route_rows
from .layercake_host_runtime import (
    RUNTIME_FORMAT,
    LayerCakeHostRuntimeError,
    _canonical_sha,
    _sha256_file,
)


FP32_ROUTE = 4


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def derive_route_local_precision_artifact(
    *,
    source_artifact: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Keep route 4 FP32 and execute all other cake routes through int8."""

    import onnx
    from onnx import helper, numpy_helper

    source_artifact = Path(source_artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"route-local precision artifact is immutable: {output_path}"
        )
    metadata = json.loads(
        (source_artifact / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("format") != RUNTIME_FORMAT:
        raise LayerCakeHostRuntimeError(
            "route-local precision source is not an ABI native host"
        )
    if metadata["runtime"].get("cake_activation_schedule") is not None:
        raise LayerCakeHostRuntimeError(
            "route-local precision requires an always-active source cake"
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
                f"route-local precision source changed: {path.name}"
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
            "transpose": "/Transpose",
            "matmul": "/MatMul",
            "shape": (10, 64, 768),
            "output_width": 64,
        },
        {
            "id": "up",
            "weight": "onnx::Gather_1072",
            "transpose": "/Transpose_1",
            "matmul": "/MatMul_1",
            "shape": (10, 768, 64),
            "output_width": 768,
        },
    ]
    node_by_name = {node.name: node for node in document.graph.node}
    axes_name = "TaskCakeRouteScalarAxes"
    fp32_route_name = "TaskCakeFp32Route"
    int8_zero_name = "TaskCakeHybridInt8WeightZeroPoint"
    document.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.asarray([0], dtype=np.int64), name=axes_name
            ),
            numpy_helper.from_array(
                np.asarray(FP32_ROUTE, dtype=np.int64),
                name=fp32_route_name,
            ),
            numpy_helper.from_array(
                np.asarray(0, dtype=np.int8), name=int8_zero_name
            ),
        ]
    )
    route_scalar = "TaskCakeRouteScalar"
    route_is_fp32 = "TaskCakeRouteUsesFp32"
    condition_nodes = [
        helper.make_node(
            "Squeeze",
            ["route", axes_name],
            [route_scalar],
            name="TaskCakeRouteSqueeze",
        ),
        helper.make_node(
            "Equal",
            [route_scalar, fp32_route_name],
            [route_is_fp32],
            name="TaskCakeRoutePrecisionCondition",
        ),
    ]
    first_matmul = node_by_name["/MatMul"]
    condition_index = list(document.graph.node).index(first_matmul)
    for offset, node in enumerate(condition_nodes):
        document.graph.node.insert(condition_index + offset, node)

    quantization_evidence: dict[str, Any] = {}
    for specification in specifications:
        weights = arrays.get(specification["weight"])
        transpose = node_by_name.get(specification["transpose"])
        matmul = node_by_name.get(specification["matmul"])
        if (
            weights is None
            or weights.shape != specification["shape"]
            or weights.dtype != np.float32
            or transpose is None
            or matmul is None
            or transpose.op_type != "Transpose"
            or matmul.op_type != "MatMul"
            or matmul.input[1] != transpose.output[0]
        ):
            raise LayerCakeHostRuntimeError(
                f"{specification['id']} hybrid cake boundary changed"
            )
        quantized, scales = _quantize_route_rows(weights)
        prefix = f"TaskCakeHybrid{specification['id'].title()}"
        quantized_weight = prefix + "Weights"
        scale_name = prefix + "Scales"
        selected_weight = prefix + "SelectedWeights"
        transposed_weight = prefix + "TransposedWeights"
        selected_scale = prefix + "SelectedScales"
        document.graph.initializer.extend(
            [
                numpy_helper.from_array(
                    quantized, name=quantized_weight
                ),
                numpy_helper.from_array(scales, name=scale_name),
            ]
        )
        outer_nodes = [
            helper.make_node(
                "Gather",
                [quantized_weight, "route"],
                [selected_weight],
                name=prefix + "WeightGather",
                axis=0,
            ),
            helper.make_node(
                "Transpose",
                [selected_weight],
                [transposed_weight],
                name=prefix + "WeightTranspose",
                perm=[0, 2, 1],
            ),
            helper.make_node(
                "Gather",
                [scale_name, "route"],
                [selected_scale],
                name=prefix + "ScaleGather",
                axis=0,
            ),
        ]
        fp32_output = prefix + "Fp32Output"
        fp32_branch = helper.make_graph(
            [
                helper.make_node(
                    "MatMul",
                    [matmul.input[0], matmul.input[1]],
                    [fp32_output],
                    name=prefix + "Fp32MatMul",
                )
            ],
            prefix + "Fp32Branch",
            [],
            [
                helper.make_tensor_value_info(
                    fp32_output,
                    onnx.TensorProto.FLOAT,
                    [1, 1, specification["output_width"]],
                )
            ],
        )
        activation_q = prefix + "ActivationQuantized"
        activation_scale = prefix + "ActivationScale"
        activation_zero = prefix + "ActivationZeroPoint"
        integer_output = prefix + "IntegerOutput"
        float_output = prefix + "FloatOutput"
        activation_scaled = prefix + "ActivationScaled"
        int8_output = prefix + "Int8Output"
        int8_branch = helper.make_graph(
            [
                helper.make_node(
                    "DynamicQuantizeLinear",
                    [matmul.input[0]],
                    [
                        activation_q,
                        activation_scale,
                        activation_zero,
                    ],
                    name=prefix + "ActivationQuantize",
                ),
                helper.make_node(
                    "MatMulInteger",
                    [
                        activation_q,
                        transposed_weight,
                        activation_zero,
                        int8_zero_name,
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
                    [int8_output],
                    name=prefix + "WeightScaleApply",
                ),
            ],
            prefix + "Int8Branch",
            [],
            [
                helper.make_tensor_value_info(
                    int8_output,
                    onnx.TensorProto.FLOAT,
                    [1, 1, specification["output_width"]],
                )
            ],
        )
        hybrid = helper.make_node(
            "If",
            [route_is_fp32],
            [matmul.output[0]],
            name=prefix + "PrecisionSwitch",
            then_branch=fp32_branch,
            else_branch=int8_branch,
        )
        nodes = list(document.graph.node)
        matmul_index = nodes.index(matmul)
        document.graph.node.remove(matmul)
        for offset, node in enumerate([*outer_nodes, hybrid]):
            document.graph.node.insert(matmul_index + offset, node)
        quantization_evidence[specification["id"]] = {
            "source_shape": list(weights.shape),
            "int8_scale_shape": list(scales.shape),
            "maximum_absolute_int8_weight_error": float(
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
    runtime["source_uniform_fp32_cake_graph_sha256"] = metadata[
        "runtime"
    ]["graph_sha256"]
    runtime["task_cake_route_local_precision"] = {
        "format": "abi-layercake-route-local-cake-precision/1",
        "fp32_routes": [FP32_ROUTE],
        "int8_routes": [
            route for route in range(10) if route != FP32_ROUTE
        ],
        "branch_input": "route",
        "fp32_execution": "original gathered weights and MatMul",
        "int8_execution": (
            "per-route/per-output-channel weights, "
            "DynamicQuantizeLinear, MatMulInteger"
        ),
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
    result = derive_route_local_precision_artifact(
        source_artifact=args.source_artifact,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
