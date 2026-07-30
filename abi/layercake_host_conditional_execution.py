"""Derive a physically conditional task-cake realization graph."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

from .layercake_host_runtime import (
    RUNTIME_FORMAT,
    LayerCakeHostRuntimeError,
    _canonical_sha,
    _sha256_file,
)


CONDITIONAL_SCHEDULE_FORMAT = (
    "abi-layercake-conditional-core-realization-schedule/1"
)
CONDITION_INPUT = "task_cake_active"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def derive_conditional_cake_artifact(
    *,
    source_artifact: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Move the exact task-cake residual into an ONNX If true branch."""

    import onnx
    from onnx import helper

    source_artifact = Path(source_artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"conditional cake artifact is immutable: {output_path}"
        )
    metadata = json.loads(
        (source_artifact / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("format") != RUNTIME_FORMAT:
        raise LayerCakeHostRuntimeError(
            "conditional source is not an ABI native host"
        )
    if metadata["runtime"].get("cake_activation_schedule") is not None:
        raise LayerCakeHostRuntimeError(
            "conditional derivation requires an unscheduled source graph"
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
                f"conditional source changed: {path.name}"
            )

    document = onnx.load(graph_source)
    nodes = list(document.graph.node)
    add_nodes = [
        node
        for node in nodes
        if node.name == "/Add_3"
        and list(node.output) == ["/Add_3_output_0"]
    ]
    residual_nodes = [
        node
        for node in nodes
        if node.name == "/MatMul_1"
        and list(node.output) == ["/MatMul_1_output_0"]
    ]
    if len(add_nodes) != 1 or len(residual_nodes) != 1:
        raise LayerCakeHostRuntimeError(
            "task-cake residual boundary changed"
        )
    add_node = add_nodes[0]
    residual_output = residual_nodes[0].output[0]
    if residual_output not in add_node.input:
        raise LayerCakeHostRuntimeError(
            "task-cake residual is not connected to its add"
        )
    hidden_inputs = [
        value for value in add_node.input if value != residual_output
    ]
    if len(hidden_inputs) != 1:
        raise LayerCakeHostRuntimeError(
            "task-cake core-hidden boundary changed"
        )
    hidden = hidden_inputs[0]

    producer = {
        output: node
        for node in nodes
        for output in node.output
    }
    boundaries = {hidden, "route"}
    cake_node_ids: set[int] = set()
    pending = [residual_output]
    while pending:
        value = pending.pop()
        if value in boundaries or value not in producer:
            continue
        node = producer[value]
        if id(node) in cake_node_ids:
            continue
        cake_node_ids.add(id(node))
        pending.extend(node.input)
    cake_nodes = [
        node for node in nodes if id(node) in cake_node_ids
    ]
    if (
        len(cake_nodes) != 26
        or residual_nodes[0] not in cake_nodes
        or any(node.op_type == "If" for node in cake_nodes)
    ):
        raise LayerCakeHostRuntimeError(
            "task-cake subgraph does not match the profiled 26-node boundary"
        )
    cake_outputs = {
        output for node in cake_nodes for output in node.output
    }
    permitted_external_consumer = add_node
    for node in nodes:
        if node in cake_nodes or node is permitted_external_consumer:
            continue
        if any(value in cake_outputs for value in node.input):
            raise LayerCakeHostRuntimeError(
                "task-cake intermediate escapes the conditional boundary"
            )

    then_nodes = [copy.deepcopy(node) for node in cake_nodes]
    then_output_name = "conditional_task_cake_adapted"
    then_nodes.append(
        helper.make_node(
            "Add",
            [hidden, residual_output],
            [then_output_name],
            name="ConditionalTaskCakeResidualAdd",
        )
    )
    then_branch = helper.make_graph(
        then_nodes,
        "task_cake_active_branch",
        [],
        [
            helper.make_tensor_value_info(
                then_output_name,
                onnx.TensorProto.FLOAT,
                [1, 1, 768],
            )
        ],
    )
    else_output_name = "conditional_english_core_hidden"
    else_branch = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                [hidden],
                [else_output_name],
                name="ConditionalEnglishCoreIdentity",
            )
        ],
        "english_core_continuation_branch",
        [],
        [
            helper.make_tensor_value_info(
                else_output_name,
                onnx.TensorProto.FLOAT,
                [1, 1, 768],
            )
        ],
    )
    conditional = helper.make_node(
        "If",
        [CONDITION_INPUT],
        [add_node.output[0]],
        name="PhysicalConditionalTaskCake",
        then_branch=then_branch,
        else_branch=else_branch,
    )
    removed_ids = {id(node) for node in [*cake_nodes, add_node]}
    first_index = sum(
        1
        for node in nodes[: nodes.index(add_node)]
        if id(node) not in removed_ids
    )
    rewritten = [node for node in nodes if id(node) not in removed_ids]
    rewritten.insert(first_index, conditional)
    del document.graph.node[:]
    document.graph.node.extend(rewritten)
    document.graph.input.extend(
        [
            helper.make_tensor_value_info(
                CONDITION_INPUT,
                onnx.TensorProto.BOOL,
                [],
            )
        ]
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
    runtime = derived["runtime"]
    runtime["graph"] = graph_path.name
    runtime["graph_sha256"] = _sha256_file(graph_path)
    runtime["graph_bytes"] = graph_path.stat().st_size
    runtime["source_unconditional_runtime_graph_sha256"] = metadata[
        "runtime"
    ]["graph_sha256"]
    runtime["cake_activation_schedule"] = {
        "format": CONDITIONAL_SCHEDULE_FORMAT,
        "graph_input": CONDITION_INPUT,
        "graph_input_type": "bool",
        "prompt_scan_activation": False,
        "first_output_decision_activation": True,
        "active_decode_steps": 0,
        "continuation_activation": False,
        "task_cake_nodes_in_true_branch": len(cake_nodes),
        "task_cake_nodes_in_false_branch": 0,
        "physical_conditional_execution": True,
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
    args = parser.parse_args(argv)
    result = derive_conditional_cake_artifact(
        source_artifact=args.source_artifact,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
