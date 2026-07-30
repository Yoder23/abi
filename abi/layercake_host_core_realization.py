"""Derive a capability-steering then English-core realization graph."""

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


SCHEDULE_FORMAT = "abi-layercake-core-realization-schedule/1"
ACTIVATION_INPUT = "cake_activation_scale"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def derive_core_realization_artifact(
    *,
    source_artifact: str | Path,
    output_path: str | Path,
    active_decode_steps: int = 0,
) -> dict[str, Any]:
    """Scale the task-cake residual according to a runtime-bound schedule."""

    import onnx
    from onnx import helper

    source_artifact = Path(source_artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"core-realization artifact is immutable: {output_path}"
        )
    if active_decode_steps < 0:
        raise LayerCakeHostRuntimeError(
            "active decode steps cannot be negative"
        )
    metadata = json.loads(
        (source_artifact / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("format") != RUNTIME_FORMAT:
        raise LayerCakeHostRuntimeError(
            "core-realization source is not a native host"
        )
    graph_source = source_artifact / metadata["runtime"]["graph"]
    tokenizer_source = (
        source_artifact / metadata["tokenizer"]["path"]
    )
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
        vocabulary_source = (
            source_artifact / output_vocabulary["path"]
        )
        components.append(
            (vocabulary_source, output_vocabulary["sha256"])
        )
    for path, expected in components:
        if _sha256_file(path) != expected:
            raise LayerCakeHostRuntimeError(
                f"core-realization source changed: {path.name}"
            )

    document = onnx.load(graph_source)
    residual_nodes = [
        node for node in document.graph.node
        if node.name == "/MatMul_1"
        and list(node.output) == ["/MatMul_1_output_0"]
    ]
    add_nodes = [
        node for node in document.graph.node
        if node.name == "/Add_3"
        and "/MatMul_1_output_0" in node.input
    ]
    if len(residual_nodes) != 1 or len(add_nodes) != 1:
        raise LayerCakeHostRuntimeError(
            "task-cake residual boundary changed"
        )
    document.graph.input.extend(
        [
            helper.make_tensor_value_info(
                ACTIVATION_INPUT,
                onnx.TensorProto.FLOAT,
                [1],
            )
        ]
    )
    scaled_name = "scheduled_task_cake_residual"
    scale_node = helper.make_node(
        "Mul",
        ["/MatMul_1_output_0", ACTIVATION_INPUT],
        [scaled_name],
        name="CoreRealizationCakeActivation",
    )
    add_node = add_nodes[0]
    add_index = list(document.graph.node).index(add_node)
    document.graph.node.insert(add_index, scale_node)
    add_node.input[
        list(add_node.input).index("/MatMul_1_output_0")
    ] = scaled_name

    output_path.mkdir(parents=True, exist_ok=False)
    graph_path = output_path / "model-int8.onnx"
    onnx.checker.check_model(document)
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
    runtime["cake_activation_schedule"] = {
        "format": SCHEDULE_FORMAT,
        "graph_input": ACTIVATION_INPUT,
        "prefill_activation": 1.0,
        "first_output_decision_activation": 1.0,
        "active_decode_steps": active_decode_steps,
        "continuation_activation": 0.0,
        "task_cake_weights_changed": False,
        "transformer_weights_changed": False,
        "public_runtime_interface_changed": False,
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
    parser.add_argument("--active-decode-steps", type=int, default=0)
    args = parser.parse_args(argv)
    result = derive_core_realization_artifact(
        source_artifact=args.source_artifact,
        output_path=args.output,
        active_decode_steps=args.active_decode_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
