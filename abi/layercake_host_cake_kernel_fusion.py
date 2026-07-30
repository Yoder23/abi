"""Fuse the two route-selected int8 cake projections into ORT kernels."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from .layercake_host_runtime import (
    LayerCakeHostRuntimeError,
    _canonical_sha,
    _sha256_file,
    _write_json,
)


def derive_fused_cake_kernel_artifact(
    *,
    source_artifact: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Fuse arithmetic only; preserve every tensor and public graph output."""

    import onnx
    from onnx import helper

    source_artifact = Path(source_artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"fused cake artifact is immutable: {output_path}"
        )
    metadata_path = source_artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    runtime = metadata["runtime"]
    quantization = runtime.get("task_cake_projection_quantization")
    if quantization is None:
        raise LayerCakeHostRuntimeError(
            "cake kernel fusion requires the route-cake int8 graph"
        )
    graph_source = source_artifact / runtime["graph"]
    tokenizer_source = (
        source_artifact / metadata["tokenizer"]["path"]
    )
    symbolic_source = (
        source_artifact / metadata["symbolic_surface"]["path"]
    )
    components = [
        (graph_source, runtime["graph_sha256"]),
        (tokenizer_source, metadata["tokenizer"]["sha256"]),
        (symbolic_source, metadata["symbolic_surface"]["sha256"]),
    ]
    output_vocabulary = runtime.get("output_vocabulary")
    vocabulary_source = None
    if output_vocabulary is not None:
        vocabulary_source = (
            source_artifact / output_vocabulary["path"]
        )
        components.append(
            (vocabulary_source, output_vocabulary["sha256"])
        )
    for path, expected in components:
        if _sha256_file(path) != expected:
            raise LayerCakeHostRuntimeError(
                f"cake kernel fusion source changed: {path.name}"
            )

    document = onnx.load(graph_source)
    node_by_name = {node.name: node for node in document.graph.node}
    fused_nodes = []
    for projection in ("Down", "Up"):
        prefix = f"TaskCake{projection}"
        names = {
            "gather": prefix + "ScaleGather",
            "quantize": prefix + "ActivationQuantize",
            "matmul": prefix + "MatMulInteger",
            "cast": prefix + "IntegerCast",
            "activation_scale": prefix + "ActivationScaleApply",
            "weight_scale": prefix + "WeightScaleApply",
        }
        nodes = {
            key: node_by_name.get(name)
            for key, name in names.items()
        }
        if any(node is None for node in nodes.values()):
            raise LayerCakeHostRuntimeError(
                f"{projection.lower()} cake quantized sequence changed"
            )
        gather = nodes["gather"]
        quantize = nodes["quantize"]
        matmul = nodes["matmul"]
        cast = nodes["cast"]
        activation_scale = nodes["activation_scale"]
        weight_scale = nodes["weight_scale"]
        if (
            list(matmul.input)[0] != quantize.output[0]
            or list(matmul.input)[2] != quantize.output[2]
            or list(cast.input) != [matmul.output[0]]
            or list(activation_scale.input)
            != [cast.output[0], quantize.output[1]]
            or list(weight_scale.input)
            != [activation_scale.output[0], gather.output[0]]
        ):
            raise LayerCakeHostRuntimeError(
                f"{projection.lower()} cake quantized dataflow changed"
            )
        fused = helper.make_node(
            "DynamicQuantizeMatMul",
            [
                quantize.input[0],
                matmul.input[1],
                gather.output[0],
                matmul.input[3],
            ],
            [weight_scale.output[0]],
            name=prefix + "DynamicQuantizeMatMul",
            domain="com.microsoft",
        )
        ordered = list(document.graph.node)
        insertion_index = min(ordered.index(node) for node in nodes.values())
        for key in (
            "quantize",
            "matmul",
            "cast",
            "activation_scale",
            "weight_scale",
        ):
            document.graph.node.remove(nodes[key])
        document.graph.node.insert(insertion_index + 1, fused)
        fused_nodes.append(fused.name)

    if not any(
        item.domain == "com.microsoft"
        for item in document.opset_import
    ):
        document.opset_import.extend(
            [helper.make_opsetid("com.microsoft", 1)]
        )
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
    derived_runtime = derived["runtime"]
    derived_runtime["graph"] = graph_path.name
    derived_runtime["graph_sha256"] = _sha256_file(graph_path)
    derived_runtime["graph_bytes"] = graph_path.stat().st_size
    derived_runtime["source_unfused_cake_runtime_graph_sha256"] = (
        runtime["graph_sha256"]
    )
    derived_runtime["task_cake_projection_quantization"][
        "kernel_fusion"
    ] = {
        "format": "abi-layercake-route-cake-kernel-fusion/1",
        "operator": "com.microsoft.DynamicQuantizeMatMul/1",
        "fused_nodes": fused_nodes,
        "unfused_nodes_removed": 10,
        "route_scale_gathers_preserved": 2,
        "weights_changed": False,
        "scales_changed": False,
    }
    derived["final_test_accessed"] = False
    derived.pop("evidence_sha256", None)
    derived["evidence_sha256"] = _canonical_sha(derived)
    _write_json(output_path / "metadata.json", derived)
    return derived


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = derive_fused_cake_kernel_artifact(
        source_artifact=args.source_artifact,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
