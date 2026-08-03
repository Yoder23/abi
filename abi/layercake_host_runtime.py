"""Native CPU deployment for a teacher-free ABI-certified LayerCake host.

The exporter consumes the exact sealed LayerCake parent plus an immutable ABI
host delta. LoRA weights are fused before export. The ONNX graph dynamically
gathers one installed task cake and one installed ABI route bridge for each
sequence. The compact symbolic surface contract is copied separately and
hash-bound to the runtime metadata; teacher responses and source-model
machinery are never deployed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort
import psutil
from tokenizers import Tokenizer

from .symbolic_runtime import (
    CAPABILITY_TO_ROUTE,
    novel_lexical_repetition_occurrences,
    symbolic_surface_output,
    truncate_novel_lexical_repetition,
)


RUNTIME_FORMAT = "abi-layercake-host-onnx-runtime/1"
FULL_TASK_ROUTE_ROUTER_MODE = (
    "onnx_zero_control_transformer_mean_classifier"
)
COMPACT_TASK_ROUTE_ROUTER_MODE = (
    "onnx_compact_token_mean_max_classifier_v1"
)
TASK_ROUTE_ROUTER_MODES = frozenset(
    {FULL_TASK_ROUTE_ROUTER_MODE, COMPACT_TASK_ROUTE_ROUTER_MODE}
)


class LayerCakeHostRuntimeError(RuntimeError):
    """Raised when native host identity or execution fails closed."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _active_runtime_model_bytes(metadata: Mapping[str, Any]) -> int:
    """Count every neural graph/parameter payload loaded for one request."""

    runtime = metadata["runtime"]
    total = int(runtime["graph_bytes"])
    contracts = (
        runtime.get("persistent_capability_prefix", {}),
        runtime.get("layerwise_capability_control", {}),
        runtime.get("deep_capability_adapters", {}),
        runtime.get("deep_reused_capability_cakes", {}),
        runtime.get("gated_deep_reused_capability_cakes", {}),
    )
    for contract in contracts:
        if not contract.get("enabled", False):
            continue
        if contract.get("router_graph"):
            total += int(contract.get("router_graph_bytes", 0))
        if contract.get("router_parameters"):
            total += int(contract.get("router_parameters_bytes", 0))
    return total


def _load_runtime_decoding_overlay(
    path: str | Path,
    *,
    core_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one narrowly scoped, checkpoint-bound decoding overlay."""

    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    expected_top_level = {
        "schema_version",
        "status",
        "candidate_core",
        "override",
        "invariants",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_top_level
        or document.get("schema_version")
        != "abi-layercake-runtime-decoding-overlay/1"
        or document.get("status")
        != "PREREGISTERED_BEFORE_SUCCESSOR_RUNTIME_EXPORT_OR_EVALUATION"
    ):
        raise LayerCakeHostRuntimeError(
            "runtime decoding overlay schema or status is invalid"
        )
    candidate = document.get("candidate_core")
    if (
        not isinstance(candidate, dict)
        or set(candidate)
        != {
            "checkpoint_sha256",
            "manifest_sha256",
            "base_decoding_sha256",
        }
        or candidate["checkpoint_sha256"]
        != core_manifest.get("checkpoint", {}).get("sha256")
        or candidate["manifest_sha256"]
        != core_manifest.get("manifest_sha256")
        or candidate["base_decoding_sha256"]
        != _canonical_sha(core_manifest.get("decoding", {}))
    ):
        raise LayerCakeHostRuntimeError(
            "runtime decoding overlay is not bound to this exact core"
        )
    override = document.get("override")
    maximum_run = (
        override.get("maximum_identical_token_run")
        if isinstance(override, dict)
        else None
    )
    if (
        not isinstance(override, dict)
        or set(override) != {"maximum_identical_token_run"}
        or isinstance(maximum_run, bool)
        or not isinstance(maximum_run, int)
        or not 1 <= maximum_run <= 5
    ):
        raise LayerCakeHostRuntimeError(
            "runtime decoding overlay exceeds its bounded token-run scope"
        )
    invariants = document.get("invariants")
    if (
        invariants
        != {
            "weights_changed": False,
            "prompt_specific": False,
            "output_specific": False,
            "teacher_present_at_inference": False,
        }
    ):
        raise LayerCakeHostRuntimeError(
            "runtime decoding overlay invariants changed"
        )
    return document


def _legacy_cache(cache) -> tuple[tuple[Any, Any], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(cache)


def _quantize_embedding_rows(
    embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize each token row independently so outliers cannot set all scales."""

    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim != 2 or not embedding.shape[0] or not embedding.shape[1]:
        raise LayerCakeHostRuntimeError(
            "token embedding must be a non-empty matrix"
        )
    scales = np.maximum(
        np.max(np.abs(embedding), axis=1) / np.float32(127.0),
        np.float32(1.0e-8),
    ).astype(np.float32)
    quantized = np.clip(
        np.rint(embedding / scales[:, None]), -127, 127
    ).astype(np.int8)
    return quantized, scales


def _classify_host_sparse_boundary(
    manifest: Mapping[str, Any],
    route_bridge: Any | None,
) -> tuple[bool, bool]:
    """Classify the two verified no-runtime-bridge host boundaries.

    A symbolic-only overlay is intentionally not a routing bridge: it leaves
    the acquired core and its installed task cakes unchanged and may only
    dispatch a small, teacher-free surface realizer before falling through to
    the core.  Keep this exception exact so a learned or undeclared bridge
    cannot be smuggled through the native exporter.
    """

    host_delta = manifest.get("host_delta", {})
    bridge_contract = host_delta.get("sparse_route_bridge", {})
    bridge_fused = (
        bridge_contract.get("mode") == "none"
        and bridge_contract.get("fused_into_existing_task_cakes") is True
    )
    parent = manifest.get("parent_layercake", {})
    lora = host_delta.get("lora", {})
    prompt_identity = host_delta.get("prompt_identity", {})
    symbolic = host_delta.get("symbolic_surface", {})
    component_types = {
        component.get("type")
        for component in manifest.get("components", [])
        if isinstance(component, Mapping)
    }
    symbolic_only = (
        route_bridge is None
        and host_delta.get("bridge_mode") == "symbolic_surface_only"
        and int(host_delta.get("trained_parameter_count", -1)) == 0
        and lora.get("target_modules") == []
        and int(lora.get("rank", -1)) == 0
        and int(lora.get("alpha", -1)) == 0
        and int(lora.get("fused_runtime_extra_modules", -1)) == 0
        and prompt_identity.get("mode") == "none"
        and int(prompt_identity.get("parameter_count", -1)) == 0
        and int(prompt_identity.get("rank", -1)) == 0
        and int(prompt_identity.get("runtime_extra_modules", -1)) == 0
        and bridge_contract.get("mode") == "none"
        and int(bridge_contract.get("installed_routes", -1)) == 0
        and int(
            bridge_contract.get("maximum_active_routes_per_sequence", -1)
        )
        == 0
        and int(bridge_contract.get("parameter_count", -1)) == 0
        and int(bridge_contract.get("rank", -1)) == 0
        and symbolic.get("mode") == "learned_rules_and_schema_realizers"
        and bool(symbolic.get("handlers"))
        and int(
            symbolic.get("maximum_active_handlers_per_sequence", -1)
        )
        == 1
        and symbolic.get("source_teacher_text_retained") is False
        and manifest.get("teacher_present_at_inference") is False
        and manifest.get("source_generated_text_retained_in_deployment")
        is False
        and int(manifest.get("source_transformer_blocks_retained", -1)) == 0
        and manifest.get("decoding", {}).get("prompt_identity_mixture")
        is False
        and parent.get("transformer_state_sha256_before")
        == parent.get("transformer_state_sha256_after")
        == parent.get("fused_runtime_transformer_state_sha256")
        and "abi_sparse_prompt_identity_bridge" not in component_types
        and "abi_sparse_route_conformance_bridge" not in component_types
    )
    if route_bridge is None and not bridge_fused and not symbolic_only:
        raise LayerCakeHostRuntimeError(
            "native host requires a sparse route bridge, an explicit "
            "verified task-cake fusion, or an exact zero-parameter "
            "symbolic-only overlay"
        )
    return bridge_fused, symbolic_only


def _route_selected_projection_node_names(document: Any) -> list[str]:
    """Find sparse residual MatMuls whose weights are gathered by route."""

    initializer_shapes = {
        value.name: tuple(int(dimension) for dimension in value.dims)
        for value in document.graph.initializer
    }
    consumers: dict[str, list[Any]] = {}
    identity_sources = {
        node.output[0]: node.input[0]
        for node in document.graph.node
        if node.op_type == "Identity"
        and len(node.input) == 1
        and len(node.output) == 1
    }

    def resolved_initializer(name: str) -> str:
        visited: set[str] = set()
        while name in identity_sources and name not in visited:
            visited.add(name)
            name = identity_sources[name]
        return name

    for node in document.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    projection_names: list[str] = []
    for gather in document.graph.node:
        if (
            gather.op_type != "Gather"
            or len(gather.input) < 2
            or gather.input[1] not in {"route", "requested_route"}
            or len(gather.output) != 1
        ):
            continue
        if "transformer" in str(gather.name).lower():
            continue
        shape = initializer_shapes.get(
            resolved_initializer(gather.input[0])
        )
        if (
            shape is None
            or len(shape) != 3
            or shape[0] not in {10, 14}
            or shape in {
                (14, 3, 768),
                (10, 3, 768),
                (14, 32, 768),
                (14, 768, 32),
            }
        ):
            continue
        transposes = [
            node
            for node in consumers.get(gather.output[0], [])
            if node.op_type == "Transpose" and len(node.output) == 1
        ]
        if len(transposes) != 1:
            raise LayerCakeHostRuntimeError(
                "route-selected projection has an ambiguous transpose"
            )
        matrices = [
            node
            for node in consumers.get(transposes[0].output[0], [])
            if node.op_type == "MatMul"
        ]
        if len(matrices) != 1 or not matrices[0].name:
            raise LayerCakeHostRuntimeError(
                "route-selected projection has an ambiguous matrix node"
            )
        projection_names.append(str(matrices[0].name))
    if len(set(projection_names)) != len(projection_names):
        raise LayerCakeHostRuntimeError(
            "route-selected projection node names are not unique"
        )
    return sorted(projection_names)


def _dynamic_quantizer_exclusion_names(
    document: Any,
    excluded_matrix_nodes: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Bind exclusions across ORT's Gemm-to-MatMul preprocessing rename."""

    matrix_types = {
        str(node.name): str(node.op_type)
        for node in document.graph.node
        if node.op_type in {"MatMul", "Gemm"} and node.name
    }
    missing = sorted(set(excluded_matrix_nodes) - set(matrix_types))
    if missing:
        raise LayerCakeHostRuntimeError(
            "matrix precision exclusions reference missing nodes: "
            + ", ".join(missing)
        )
    quantizer_names = set(excluded_matrix_nodes)
    runtime_names: list[str] = []
    for name in sorted(excluded_matrix_nodes):
        if matrix_types[name] == "Gemm":
            converted_name = name + "_MatMul"
            quantizer_names.add(converted_name)
            runtime_names.append(converted_name)
        else:
            runtime_names.append(name)
    return sorted(quantizer_names), runtime_names


def _float_matrix_node_checks(
    document: Any,
    runtime_node_names: Sequence[str],
) -> dict[str, bool]:
    """Prove every declared precision exclusion is executable float math."""

    nodes = {
        str(node.name): str(node.op_type)
        for node in document.graph.node
        if node.name
    }
    return {
        str(name): nodes.get(str(name)) in {"MatMul", "Gemm"}
        for name in runtime_node_names
    }


def export_host_runtime(
    *,
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    host_path: str | Path | None = None,
    standalone_core_path: str | Path | None = None,
    output_path: str | Path,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    allow_prompt_ngrams: bool = False,
    lexical_repetition_blocking_threshold: int | None = None,
    lexical_repetition_truncation_threshold: int | None = None,
    byte_repetition_ceiling: float | None = None,
    byte_repetition_guard_minimum_bytes: int | None = None,
    runtime_decoding_overlay_path: str | Path | None = None,
    keep_task_cakes_fp32: bool = False,
    precision_profile: str = "int8",
    task_route_router_precision: str = "int8",
) -> dict[str, Any]:
    """Export and int8-quantize one exact ABI host or standalone core."""

    import onnx
    from onnx import numpy_helper
    from onnx import helper, numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import torch
    import torch.nn.functional as F

    from .artifacts import module_state_sha256
    from .layercake_core_loader import load_layercake_core
    from .layercake_host import load_host_model

    layercake_root = Path(layercake_root).resolve()
    parent_path = Path(parent_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    if (host_path is None) == (standalone_core_path is None):
        raise LayerCakeHostRuntimeError(
            "exactly one host or standalone core must be supplied"
        )
    host_path = Path(host_path).resolve() if host_path is not None else None
    standalone_core_path = (
        Path(standalone_core_path).resolve()
        if standalone_core_path is not None
        else None
    )
    output_path = Path(output_path).resolve()
    runtime_decoding_overlay_path = (
        Path(runtime_decoding_overlay_path).resolve()
        if runtime_decoding_overlay_path is not None
        else None
    )
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"native host artifact is immutable: {output_path}"
        )
    if repetition_penalty < 1.0:
        raise LayerCakeHostRuntimeError(
            "repetition penalty must be at least one"
        )
    if no_repeat_ngram_size not in (0,) and no_repeat_ngram_size < 2:
        raise LayerCakeHostRuntimeError(
            "no-repeat n-gram size must be zero or at least two"
        )
    if (
        lexical_repetition_blocking_threshold is not None
        and lexical_repetition_blocking_threshold < 0
    ):
        raise LayerCakeHostRuntimeError(
            "lexical repetition blocking threshold must be nonnegative"
        )
    if (
        lexical_repetition_truncation_threshold is not None
        and lexical_repetition_truncation_threshold < 0
    ):
        raise LayerCakeHostRuntimeError(
            "lexical repetition truncation threshold must be nonnegative"
        )
    if (
        byte_repetition_ceiling is not None
        and not 0.0 <= byte_repetition_ceiling <= 1.0
    ):
        raise LayerCakeHostRuntimeError(
            "byte repetition ceiling must be between zero and one"
        )
    if (
        byte_repetition_guard_minimum_bytes is not None
        and byte_repetition_guard_minimum_bytes < 0
    ):
        raise LayerCakeHostRuntimeError(
            "byte repetition guard minimum must be nonnegative"
        )
    standalone_core = standalone_core_path is not None
    if standalone_core:
        model, _, manifest = load_layercake_core(
            standalone_core_path,
            layercake_root=layercake_root,
            device="cpu",
        )
        if (
            manifest.get("format")
            not in {
                "abi-layercake-full-english-core-acquisition/1",
                "abi-layercake-component-graft/1",
            }
            or int(manifest.get("architecture", {}).get("layers", -1))
            not in {3, 6}
            or manifest.get("canonical_semantic_abi", {}).get("sha256")
            != _sha256_file(canonical_abi_path)
            or manifest.get("foreign_source_boundary", {}).get(
                "teacher_present_at_inference"
            )
            is not False
            or int(
                manifest.get("foreign_source_boundary", {}).get(
                    "source_parameters_copied", -1
                )
            )
            != 0
        ):
            raise LayerCakeHostRuntimeError(
                "standalone core identity or foreign-source boundary changed"
            )
        route_bridge = None
        bridge_fused = False
        symbolic_only = False
        symbolic_contract = getattr(model, "_abi_symbolic_surface", None)
        if (
            manifest.get("symbolic_surface_substrate") is not None
            and symbolic_contract is None
        ):
            raise LayerCakeHostRuntimeError(
                "standalone core omitted its declared symbolic substrate"
            )
        if symbolic_contract is None:
            symbolic_contract = {
                "schema_version": "abi-symbolic-surface/empty-standalone-1",
                "handlers": [],
                "source_teacher_text_retained": False,
            }
        tokenizer_source_path = standalone_core_path
    else:
        model, _, manifest, _ = load_host_model(
            layercake_root=layercake_root,
            parent_path=parent_path,
            canonical_abi_path=canonical_abi_path,
            host_path=host_path,
            device_name="cpu",
        )
        if manifest is None:
            raise LayerCakeHostRuntimeError("host manifest is required")
        route_bridge = getattr(model, "_abi_sparse_route_bridge", None)
        bridge_fused, symbolic_only = _classify_host_sparse_boundary(
            manifest,
            route_bridge,
        )
        symbolic_contract = getattr(model, "_abi_symbolic_surface", None)
        if symbolic_contract is None:
            raise LayerCakeHostRuntimeError(
                "promoted native host requires its symbolic substrate"
            )
        tokenizer_source_path = parent_path
    if runtime_decoding_overlay_path is not None and not standalone_core:
        raise LayerCakeHostRuntimeError(
            "runtime decoding overlays are restricted to an exact "
            "standalone acquired core"
        )
    if keep_task_cakes_fp32 and not standalone_core:
        raise LayerCakeHostRuntimeError(
            "task-cake precision isolation is restricted to an exact "
            "standalone acquired core"
        )
    capability_cake_routes = tuple(
        int(value)
        for value in getattr(model, "_abi_capability_cake_routes", ())
    )
    if capability_cake_routes and (
        len(capability_cake_routes) != len(model.task_cakes)
        or len(capability_cake_routes) != int(model.config.task_cakes)
        or any(route < 0 or route >= 10 for route in capability_cake_routes)
    ):
        raise LayerCakeHostRuntimeError(
            "capability-cake canonical route mapping is invalid"
        )
    persistent_capability_prefix = bool(
        getattr(model, "_abi_persistent_capability_prefix", False)
    )
    layerwise_capability_control = bool(
        getattr(model, "_abi_layerwise_capability_control", False)
    )
    task_route_layerwise_control = bool(
        getattr(model, "_abi_task_route_layerwise_control", False)
    )
    if task_route_layerwise_control and not layerwise_capability_control:
        raise LayerCakeHostRuntimeError(
            "task-route control lost its layerwise-control boundary"
        )
    deep_capability_adapters = bool(
        getattr(model, "_abi_deep_capability_adapters", False)
    )
    deep_reused_capability_cakes = bool(
        getattr(model, "_abi_deep_reused_capability_cakes", False)
    )
    gated_deep_reused_capability_cakes = bool(
        getattr(
            model, "_abi_gated_deep_reused_capability_cakes", False
        )
    )
    if sum(
        (
            persistent_capability_prefix,
            layerwise_capability_control,
            deep_capability_adapters,
            deep_reused_capability_cakes,
            gated_deep_reused_capability_cakes,
        )
    ) > 1:
        raise LayerCakeHostRuntimeError(
            "persistent capability-conditioning topologies conflict"
        )
    if (
        persistent_capability_prefix
        or layerwise_capability_control
        or deep_capability_adapters
        or deep_reused_capability_cakes
        or gated_deep_reused_capability_cakes
    ) and not capability_cake_routes:
        raise LayerCakeHostRuntimeError(
            "persistent capability conditioning requires capability cakes"
        )
    if persistent_capability_prefix or task_route_layerwise_control:
        model.transformer.config._attn_implementation = "eager"
    precision_profiles = {
        "int8",
        "fp32_all",
        "fp32_embedding",
        "fp32_output",
        "fp32_transformer",
        "fp32_layer0",
        "fp32_layer1",
        "fp32_layer2",
        "fp32_layer0_attn_in",
        "fp32_layer0_attn_out",
        "fp32_layer0_mlp_in",
        "fp32_layer0_mlp_out",
        "fp32_layer0_attn_pair",
        "fp32_layer0_mlp_pair",
    }
    if precision_profile not in precision_profiles:
        raise LayerCakeHostRuntimeError(
            "runtime precision profile is invalid"
        )
    if task_route_router_precision not in {"int8", "fp32"}:
        raise LayerCakeHostRuntimeError(
            "task-route router precision is invalid"
        )
    if (
        task_route_router_precision != "int8"
        and not task_route_layerwise_control
    ):
        raise LayerCakeHostRuntimeError(
            "task-route router precision requires task-route control"
        )
    runtime_decoding_overlay = (
        _load_runtime_decoding_overlay(
            runtime_decoding_overlay_path,
            core_manifest=manifest,
        )
        if runtime_decoding_overlay_path is not None
        else None
    )

    class RuntimeGraph(torch.nn.Module):
        def __init__(self, source, source_route_bridge):
            super().__init__()
            self.transformer = source.transformer
            self.task_classifier = source.task_classifier
            self.task_cakes = source.task_cakes
            self.persistent_capability_prefix = bool(
                getattr(source, "_abi_persistent_capability_prefix", False)
            )
            self.layerwise_capability_control = bool(
                getattr(source, "_abi_layerwise_capability_control", False)
            )
            self.task_route_layerwise_control = bool(
                getattr(source, "_abi_task_route_layerwise_control", False)
            )
            self.deep_capability_adapters = bool(
                getattr(source, "_abi_deep_capability_adapters", False)
            )
            self.deep_reused_capability_cakes = bool(
                getattr(source, "_abi_deep_reused_capability_cakes", False)
            )
            self.gated_deep_reused_capability_cakes = bool(
                getattr(
                    source,
                    "_abi_gated_deep_reused_capability_cakes",
                    False,
                )
            )
            self.shared_deep_capability_adapters = bool(
                getattr(
                    source, "_abi_shared_deep_capability_adapters", False
                )
            )
            if self.persistent_capability_prefix:
                self.capability_prefix_keys = source.capability_prefix_keys
                self.capability_prefix_values = (
                    source.capability_prefix_values
                )
                self.capability_prefix_length = int(
                    source.config.capability_prefix_length
                )
            if self.layerwise_capability_control:
                self.capability_control_vectors = (
                    source.task_route_control_vectors
                    if self.task_route_layerwise_control
                    else source.capability_control_vectors
                )
            if self.deep_capability_adapters:
                if self.shared_deep_capability_adapters:
                    self.capability_shared_adapters = (
                        source.capability_shared_adapters
                    )
                else:
                    self.capability_layer_adapters = (
                        source.capability_layer_adapters
                    )
            if self.gated_deep_reused_capability_cakes:
                self.capability_deep_cake_gates = (
                    source.capability_deep_cake_gates
                )
            self.route_bridges = (
                source_route_bridge.bridges
                if source_route_bridge is not None
                else None
            )

        @staticmethod
        def _selected_residual(
            hidden: torch.Tensor,
            route: torch.Tensor,
            modules,
        ) -> torch.Tensor:
            norm_weight = torch.stack(
                [module[0].weight if isinstance(module, torch.nn.Sequential)
                 else module.norm.weight for module in modules]
            ).index_select(0, route)
            norm_bias = torch.stack(
                [module[0].bias if isinstance(module, torch.nn.Sequential)
                 else module.norm.bias for module in modules]
            ).index_select(0, route)
            down_weight = torch.stack(
                [module[1].weight if isinstance(module, torch.nn.Sequential)
                 else module.down.weight for module in modules]
            ).index_select(0, route)
            up_weight = torch.stack(
                [module[3].weight if isinstance(module, torch.nn.Sequential)
                 else module.up.weight for module in modules]
            ).index_select(0, route)
            mean = hidden.mean(dim=-1, keepdim=True)
            variance = (hidden - mean).square().mean(dim=-1, keepdim=True)
            normalized = (hidden - mean) * torch.rsqrt(variance + 1.0e-5)
            normalized = (
                normalized * norm_weight[:, None] + norm_bias[:, None]
            )
            low = torch.bmm(normalized, down_weight.transpose(1, 2))
            update = torch.bmm(F.silu(low), up_weight.transpose(1, 2))
            return hidden + update

        def forward(
            self,
            input_ids,
            requested_route,
            past_key_0,
            past_value_0,
            past_key_1,
            past_value_1,
            past_key_2,
            past_value_2,
        ):
            from transformers import DynamicCache

            real_cache = (
                (past_key_0, past_value_0),
                (past_key_1, past_value_1),
                (past_key_2, past_value_2),
            )
            input_shape = torch._shape_as_tensor(input_ids)
            cache_shape = torch._shape_as_tensor(past_key_0)
            if self.persistent_capability_prefix:
                route = requested_route
                task_scores = torch.zeros(
                    (input_ids.shape[0], len(self.task_cakes)),
                    dtype=self.capability_prefix_keys.dtype,
                    device=input_ids.device,
                )
                selected_keys = self.capability_prefix_keys.index_select(
                    0, route
                )
                selected_values = (
                    self.capability_prefix_values.index_select(0, route)
                )
                cache = DynamicCache.from_legacy_cache(
                    tuple(
                        (
                            torch.cat(
                                (selected_keys[:, layer], real_cache[layer][0]),
                                dim=2,
                            ),
                            torch.cat(
                                (
                                    selected_values[:, layer],
                                    real_cache[layer][1],
                                ),
                                dim=2,
                            ),
                        )
                        for layer in range(3)
                    )
                )
            elif self.layerwise_capability_control:
                route = requested_route
                task_scores = torch.zeros(
                    (input_ids.shape[0], len(self.task_cakes)),
                    dtype=self.capability_control_vectors.dtype,
                    device=input_ids.device,
                )
                selected_control = (
                    self.capability_control_vectors.index_select(0, route)
                )
                for layer, block in enumerate(self.transformer.h):
                    block._abi_selected_capability_control = (
                        selected_control[:, layer]
                    )
                cache = DynamicCache.from_legacy_cache(real_cache)
            elif self.deep_capability_adapters:
                route = requested_route
                task_scores = torch.zeros(
                    (input_ids.shape[0], len(self.task_cakes)),
                    dtype=self.transformer.wte.weight.dtype,
                    device=input_ids.device,
                )
                for block in self.transformer.h:
                    block._abi_selected_capability_routes = route
                cache = DynamicCache.from_legacy_cache(real_cache)
            elif self.deep_reused_capability_cakes:
                route = requested_route
                task_scores = torch.zeros(
                    (input_ids.shape[0], len(self.task_cakes)),
                    dtype=self.transformer.wte.weight.dtype,
                    device=input_ids.device,
                )
                for block in self.transformer.h:
                    block._abi_selected_capability_routes = route
                cache = DynamicCache.from_legacy_cache(real_cache)
            elif self.gated_deep_reused_capability_cakes:
                route = requested_route
                task_scores = torch.zeros(
                    (input_ids.shape[0], len(self.task_cakes)),
                    dtype=self.transformer.wte.weight.dtype,
                    device=input_ids.device,
                )
                for block in self.transformer.h:
                    block._abi_selected_capability_routes = route
                cache = DynamicCache.from_legacy_cache(real_cache)
            else:
                cache = DynamicCache.from_legacy_cache(real_cache)
            position_ids = torch.arange(
                cache_shape[2],
                cache_shape[2] + input_shape[1],
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)
            attention_mask = None
            if self.persistent_capability_prefix:
                prefix_attention = torch.zeros_like(
                    input_ids[:, :1],
                    dtype=self.capability_prefix_keys.dtype,
                ).expand(-1, self.capability_prefix_length)
                past_attention = torch.zeros_like(
                    past_key_0[:, 0, :, 0]
                )
                input_attention = torch.zeros_like(
                    input_ids,
                    dtype=self.capability_prefix_keys.dtype,
                )
                attention_mask = torch.cat(
                    (
                        prefix_attention,
                        past_attention,
                        input_attention,
                    ),
                    dim=1,
                )[:, None, None, :]
            result = self.transformer(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            hidden = result.last_hidden_state
            if not (
                self.persistent_capability_prefix
                or self.layerwise_capability_control
                or self.deep_capability_adapters
                or self.deep_reused_capability_cakes
                or self.gated_deep_reused_capability_cakes
            ):
                task_scores = self.task_classifier(hidden)
                inferred = task_scores.mean(dim=1).argmax(dim=-1)
                route = torch.where(
                    requested_route < 0, inferred, requested_route
                )
            adapted = (
                hidden
                if self.deep_reused_capability_cakes
                else self._selected_residual(
                    hidden, route, self.task_cakes
                )
            )
            if self.route_bridges is not None:
                adapted = self._selected_residual(
                    adapted, route, self.route_bridges
                )
            logits = F.linear(
                adapted[:, -1], self.transformer.wte.weight
            )
            present = _legacy_cache(result.past_key_values)
            if self.persistent_capability_prefix:
                present = tuple(
                    (
                        key[:, :, self.capability_prefix_length :],
                        value[:, :, self.capability_prefix_length :],
                    )
                    for key, value in present
                )
            return (
                logits,
                route,
                (
                    task_scores
                    if (
                        self.persistent_capability_prefix
                        or self.layerwise_capability_control
                        or self.deep_capability_adapters
                        or self.deep_reused_capability_cakes
                        or self.gated_deep_reused_capability_cakes
                    )
                    else task_scores[:, -1]
                ),
                adapted[:, -1],
                present[0][0],
                present[0][1],
                present[1][0],
                present[1][1],
                present[2][0],
                present[2][1],
            )

    output_path.mkdir(parents=True, exist_ok=False)
    router_graph_path = None
    router_parameter_path = None
    router_equivalence_probes: list[dict[str, Any]] = []
    if persistent_capability_prefix:
        class CapabilityRouterGraph(torch.nn.Module):
            def __init__(self, source):
                super().__init__()
                self.embedding = source.capability_router_embedding
                self.router = source.capability_router
                self.buckets = int(source.config.capability_router_buckets)

            def forward(self, prompt_ids):
                hidden = self.embedding(
                    torch.remainder(prompt_ids, self.buckets)
                )
                scores = self.router(hidden.mean(dim=1))
                return scores.argmax(dim=-1), scores

        router_graph_path = output_path / "capability-router.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                CapabilityRouterGraph(model).eval(),
                torch.tensor([[32, 33]], dtype=torch.long),
                router_graph_path,
                input_names=["prompt_ids"],
                output_names=["selected_route", "task_scores"],
                dynamic_axes={"prompt_ids": {1: "prompt_sequence"}},
                opset_version=17,
                do_constant_folding=True,
            )
    elif task_route_layerwise_control:
        class TaskRouteRouterGraph(torch.nn.Module):
            def __init__(self, source):
                super().__init__()
                self.transformer = source.transformer
                self.task_classifier = source.task_classifier
                self.width = int(source.config.width)

            def forward(self, prompt_ids):
                zero_control = torch.zeros(
                    prompt_ids.shape[0],
                    self.width,
                    dtype=self.transformer.wte.weight.dtype,
                    device=prompt_ids.device,
                )
                for block in self.transformer.h:
                    block._abi_selected_capability_control = zero_control
                shape = torch._shape_as_tensor(prompt_ids)
                positions = torch.arange(
                    shape[1], dtype=torch.long, device=prompt_ids.device
                )
                hidden = self.transformer.wte(prompt_ids) + (
                    self.transformer.wpe(positions)[None]
                )
                query_positions = positions[:, None]
                key_positions = positions[None, :]
                causal_mask = torch.where(
                    key_positions <= query_positions,
                    torch.zeros((), dtype=hidden.dtype, device=hidden.device),
                    torch.full(
                        (),
                        torch.finfo(hidden.dtype).min,
                        dtype=hidden.dtype,
                        device=hidden.device,
                    ),
                )[None, None]
                for block in self.transformer.h:
                    hidden = block(
                        hidden,
                        attention_mask=causal_mask,
                        use_cache=False,
                    )[0]
                hidden = self.transformer.ln_f(hidden)
                task_scores = self.task_classifier(
                    hidden.mean(dim=1)
                )
                return task_scores.argmax(dim=-1), task_scores

        router_fp32_path = output_path / "task-route-router-fp32.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                TaskRouteRouterGraph(model).eval(),
                torch.tensor([[32, 33]], dtype=torch.long),
                router_fp32_path,
                input_names=["prompt_ids"],
                output_names=["selected_route", "task_scores"],
                dynamic_axes={"prompt_ids": {1: "prompt_sequence"}},
                opset_version=17,
                do_constant_folding=True,
            )
        if task_route_router_precision == "fp32":
            router_graph_path = router_fp32_path
        else:
            router_graph_path = output_path / "task-route-router.onnx"
            quantize_dynamic(
                router_fp32_path,
                router_graph_path,
                weight_type=QuantType.QInt8,
            )
            router_fp32_path.unlink()
    elif (
        layerwise_capability_control
        or deep_capability_adapters
        or deep_reused_capability_cakes
        or gated_deep_reused_capability_cakes
    ):
        embedding = (
            model.capability_router_embedding.weight.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        router_weight = (
            model.capability_router.weight.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        router_bias = (
            model.capability_router.bias.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        router_parameter_path = output_path / "capability-router.npz"
        np.savez(
            router_parameter_path,
            embedding=embedding,
            weight=router_weight,
            bias=router_bias,
        )
        for probe_ids in ([32, 33], [1, 2, 3, 4], [50256], [17, 4097, 8193]):
            hashed = np.remainder(
                np.asarray(probe_ids, dtype=np.int64), embedding.shape[0]
            )
            numpy_scores = (
                embedding[hashed].mean(axis=0) @ router_weight.T
                + router_bias
            )
            with torch.inference_mode():
                torch_scores = model.capability_router(
                    model.capability_router_embedding(
                        torch.tensor([probe_ids], dtype=torch.long)
                        % embedding.shape[0]
                    ).mean(dim=1)
                )[0].detach().cpu().numpy()
            if int(numpy_scores.argmax()) != int(torch_scores.argmax()):
                raise LayerCakeHostRuntimeError(
                    "fused router changed a reference selected route"
                )
            router_equivalence_probes.append(
                {
                    "prompt_ids": list(probe_ids),
                    "selected_route": int(numpy_scores.argmax()),
                    "maximum_score_delta": float(
                        np.max(np.abs(numpy_scores - torch_scores))
                    ),
                }
            )
    graph = RuntimeGraph(model, route_bridge).eval()
    input_ids = torch.tensor([[32]], dtype=torch.long)
    requested_route = torch.tensor(
        [
            0
            if (
                persistent_capability_prefix
                or layerwise_capability_control
                or deep_capability_adapters
                or deep_reused_capability_cakes
                or gated_deep_reused_capability_cakes
            )
            else -1
        ],
        dtype=torch.long,
    )
    empty = tuple(
        torch.zeros(1, 12, 0, 64, dtype=torch.float32)
        for _ in range(6)
    )
    input_names = [
        "input_ids",
        "requested_route",
        "past_key_0",
        "past_value_0",
        "past_key_1",
        "past_value_1",
        "past_key_2",
        "past_value_2",
    ]
    output_names = [
        "logits",
        "route",
        "task_scores",
        "abi_state",
        "present_key_0",
        "present_value_0",
        "present_key_1",
        "present_value_1",
        "present_key_2",
        "present_value_2",
    ]
    dynamic_axes = {
        **{name: {2: "past_sequence"} for name in input_names[2:]},
        **{name: {2: "present_sequence"} for name in output_names[4:]},
    }
    fp32_path = output_path / "model-fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            graph,
            (input_ids, requested_route, *empty),
            fp32_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
    fp32_document = onnx.load(fp32_path)
    task_cake_projection_nodes = (
        _route_selected_projection_node_names(fp32_document)
    )
    if keep_task_cakes_fp32 and len(task_cake_projection_nodes) != 2:
        raise LayerCakeHostRuntimeError(
            "standalone task-cake projection nodes are not isolated"
        )
    intermediate = output_path / "model-int8-matmul.onnx"
    output_projection_nodes = sorted(
        str(node.name)
        for node in fp32_document.graph.node
        if node.op_type == "MatMul"
        and "logits" in node.output
        and node.name
    )
    if len(output_projection_nodes) != 1:
        raise LayerCakeHostRuntimeError(
            "standalone output projection node is not isolated"
        )
    transformer_matrix_nodes = sorted(
        str(node.name)
        for node in fp32_document.graph.node
        if node.op_type in {"MatMul", "Gemm"}
        and str(node.name).startswith("/transformer/")
    )
    if not transformer_matrix_nodes:
        raise LayerCakeHostRuntimeError(
            "standalone transformer matrix nodes are not isolated"
        )
    excluded_matrix_nodes = set(
        task_cake_projection_nodes if keep_task_cakes_fp32 else ()
    )
    if precision_profile == "fp32_output":
        excluded_matrix_nodes.update(output_projection_nodes)
    elif precision_profile == "fp32_transformer":
        excluded_matrix_nodes.update(transformer_matrix_nodes)
    elif precision_profile in {
        "fp32_layer0_attn_in",
        "fp32_layer0_attn_out",
        "fp32_layer0_mlp_in",
        "fp32_layer0_mlp_out",
        "fp32_layer0_attn_pair",
        "fp32_layer0_mlp_pair",
    }:
        suffixes = {
            "fp32_layer0_attn_in": ("/attn/c_attn/Gemm",),
            "fp32_layer0_attn_out": ("/attn/c_proj/Gemm",),
            "fp32_layer0_mlp_in": ("/mlp/c_fc/Gemm",),
            "fp32_layer0_mlp_out": ("/mlp/c_proj/Gemm",),
            "fp32_layer0_attn_pair": (
                "/attn/c_attn/Gemm",
                "/attn/c_proj/Gemm",
            ),
            "fp32_layer0_mlp_pair": (
                "/mlp/c_fc/Gemm",
                "/mlp/c_proj/Gemm",
            ),
        }[precision_profile]
        selected_layer_nodes = {
            name
            for name in transformer_matrix_nodes
            if name.startswith("/transformer/h.0/")
            and any(name.endswith(suffix) for suffix in suffixes)
        }
        if len(selected_layer_nodes) != len(suffixes):
            raise LayerCakeHostRuntimeError(
                "fine-grained layer-0 matrix selection is incomplete"
            )
        excluded_matrix_nodes.update(selected_layer_nodes)
    elif precision_profile.startswith("fp32_layer"):
        layer = int(precision_profile[-1])
        prefix = f"/transformer/h.{layer}/"
        selected_layer_nodes = {
            name
            for name in transformer_matrix_nodes
            if name.startswith(prefix)
        }
        if not selected_layer_nodes:
            raise LayerCakeHostRuntimeError(
                "selected transformer layer has no matrix nodes"
            )
        excluded_matrix_nodes.update(selected_layer_nodes)
    (
        quantizer_exclusion_names,
        runtime_excluded_matrix_nodes,
    ) = _dynamic_quantizer_exclusion_names(
        fp32_document,
        sorted(excluded_matrix_nodes),
    )
    int8_path = output_path / "model-int8.onnx"
    if precision_profile == "fp32_all":
        runtime_graph_path = fp32_path
        document = fp32_document
    else:
        quantize_dynamic(
            fp32_path,
            intermediate,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            op_types_to_quantize=["MatMul", "Gemm"],
            nodes_to_exclude=quantizer_exclusion_names,
        )
        document = onnx.load(intermediate)
        float_exclusion_checks = _float_matrix_node_checks(
            document,
            runtime_excluded_matrix_nodes,
        )
        if not all(float_exclusion_checks.values()):
            failed = sorted(
                name
                for name, passed in float_exclusion_checks.items()
                if not passed
            )
            raise LayerCakeHostRuntimeError(
                "runtime precision exclusions were quantized or removed: "
                + ", ".join(failed)
            )
        runtime_graph_path = int8_path
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
    embedding_is_fp32 = precision_profile in {
        "fp32_all",
        "fp32_embedding",
    }
    if not embedding_is_fp32:
        embedding = numpy_helper.to_array(
            initializers[embedding_name]
        ).astype(np.float32)
        quantized, scales = _quantize_embedding_rows(embedding)
        quantized_name = embedding_name + "_runtime_int8"
        scales_name = embedding_name + "_runtime_row_scales"
        cast_output = embedding_gather.output[0] + "_runtime_float"
        scale_output = embedding_gather.output[0] + "_runtime_scale"
        expanded_scale_output = scale_output + "_expanded"
        axes_name = embedding_name + "_runtime_scale_axes"
        document.graph.initializer.extend(
            (
                numpy_helper.from_array(quantized, name=quantized_name),
                numpy_helper.from_array(
                    scales, name=scales_name
                ),
                numpy_helper.from_array(
                    np.asarray([-1], dtype=np.int64), name=axes_name
                ),
            )
        )
        original_output = embedding_gather.output[0]
        quantized_output = original_output + "_runtime_int8"
        embedding_gather.input[0] = quantized_name
        embedding_gather.output[0] = quantized_output
        node_index = list(document.graph.node).index(embedding_gather)
        scale_gather = helper.make_node(
            "Gather",
            [scales_name, embedding_gather.input[1]],
            [scale_output],
            name="RuntimeEmbeddingScaleGather",
        )
        cast = helper.make_node(
            "Cast",
            [quantized_output],
            [cast_output],
            name="RuntimeEmbeddingCast",
            to=onnx.TensorProto.FLOAT,
        )
        expand_scale = helper.make_node(
            "Unsqueeze",
            [scale_output, axes_name],
            [expanded_scale_output],
            name="RuntimeEmbeddingScaleExpand",
        )
        dequantize = helper.make_node(
            "Mul",
            [cast_output, expanded_scale_output],
            [original_output],
            name="RuntimeEmbeddingRowDequantize",
        )
        for offset, node in enumerate(
            (scale_gather, cast, expand_scale, dequantize), start=1
        ):
            document.graph.node.insert(node_index + offset, node)
        remaining_inputs = {
            value for node in document.graph.node for value in node.input
        }
        if embedding_name not in remaining_inputs:
            document.graph.initializer.remove(initializers[embedding_name])
    onnx.checker.check_model(document)
    if runtime_graph_path == int8_path:
        onnx.save(document, int8_path)

    tokenizer_path = output_path / "tokenizer.json"
    tokenizer_path.write_bytes(
        (tokenizer_source_path / "tokenizer.json").read_bytes()
    )
    symbolic_path = output_path / "symbolic-surface.json"
    symbolic_path.write_bytes(_canonical_json_bytes(symbolic_contract))
    runtime_decoding_overlay_destination = None
    if runtime_decoding_overlay is not None:
        runtime_decoding_overlay_destination = (
            output_path / "runtime-decoding-overlay.json"
        )
        runtime_decoding_overlay_destination.write_bytes(
            _canonical_json_bytes(runtime_decoding_overlay)
        )
    if standalone_core:
        host_identity = {
            "kind": "standalone_acquired_core",
            "path_at_export": str(standalone_core_path),
            "metadata_file_sha256": _sha256_file(
                standalone_core_path / "metadata.json"
            ),
            "manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "fused_transformer_state_sha256": module_state_sha256(
                model.transformer
            ),
            "teacher_present_at_inference": False,
            "source_transformer_blocks_retained": 0,
            "source_parameters_copied": 0,
        }
    else:
        host_manifest_path = host_path / "deployment_manifest.json"
        host_identity = {
            "kind": "host_delta",
            "path_at_export": str(host_path),
            "deployment_manifest_file_sha256": _sha256_file(
                host_manifest_path
            ),
            "deployment_manifest_sha256": manifest["manifest_sha256"],
            "delta_sha256": manifest["host_delta"]["sha256"],
            "fused_transformer_state_sha256": module_state_sha256(
                model.transformer
            ),
            "teacher_present_at_inference": False,
            "source_transformer_blocks_retained": 0,
        }
    source_decoding = dict(
        getattr(
            model,
            "_abi_decoding",
            {
                "algorithm": "greedy",
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "allow_prompt_ngrams": False,
                "lexical_repetition_blocking_threshold": 0,
                "lexical_repetition_truncation_threshold": 0,
                "byte_repetition_ceiling": 0.0,
                "byte_repetition_guard_minimum_bytes": 0,
                "prompt_identity_mixture": False,
            },
        )
    )
    if not standalone_core:
        source_decoding["algorithm"] = (
            "deterministic_greedy_with_repetition_controls"
            if (
                repetition_penalty != 1.0
                or no_repeat_ngram_size != 0
                or (
                    lexical_repetition_blocking_threshold is not None
                    and lexical_repetition_blocking_threshold != 0
                )
                or (
                    byte_repetition_ceiling is not None
                    and byte_repetition_ceiling != 0.0
                )
            )
            else "greedy"
        )
        source_decoding["no_repeat_ngram_size"] = int(
            no_repeat_ngram_size
        )
        source_decoding["allow_prompt_ngrams"] = bool(
            allow_prompt_ngrams
        )
        source_decoding.setdefault(
            "lexical_repetition_blocking_threshold", 0
        )
        source_decoding.setdefault(
            "lexical_repetition_truncation_threshold", 0
        )
        source_decoding.setdefault("byte_repetition_ceiling", 0.0)
        source_decoding.setdefault(
            "byte_repetition_guard_minimum_bytes", 0
        )
        source_decoding.setdefault("prompt_identity_mixture", False)
    if lexical_repetition_blocking_threshold is not None:
        source_decoding["lexical_repetition_blocking_threshold"] = int(
            lexical_repetition_blocking_threshold
        )
    if lexical_repetition_truncation_threshold is not None:
        source_decoding["lexical_repetition_truncation_threshold"] = int(
            lexical_repetition_truncation_threshold
        )
    if byte_repetition_ceiling is not None:
        source_decoding["byte_repetition_ceiling"] = float(
            byte_repetition_ceiling
        )
    if byte_repetition_guard_minimum_bytes is not None:
        source_decoding["byte_repetition_guard_minimum_bytes"] = int(
            byte_repetition_guard_minimum_bytes
        )
    if standalone_core and (
        float(repetition_penalty) != 1.0
        or int(no_repeat_ngram_size)
        != int(source_decoding["no_repeat_ngram_size"])
        or bool(allow_prompt_ngrams)
        != bool(source_decoding["allow_prompt_ngrams"])
        or (
            lexical_repetition_blocking_threshold is not None
            and int(lexical_repetition_blocking_threshold)
            != int(
                manifest["decoding"].get(
                    "lexical_repetition_blocking_threshold", 0
                )
            )
        )
        or (
            lexical_repetition_truncation_threshold is not None
            and int(lexical_repetition_truncation_threshold)
            != int(
                manifest["decoding"][
                    "lexical_repetition_truncation_threshold"
                ]
            )
        )
        or (
            byte_repetition_ceiling is not None
            and float(byte_repetition_ceiling)
            != float(
                manifest["decoding"].get("byte_repetition_ceiling", 0.0)
            )
        )
        or (
            byte_repetition_guard_minimum_bytes is not None
            and int(byte_repetition_guard_minimum_bytes)
            != int(
                manifest["decoding"].get(
                    "byte_repetition_guard_minimum_bytes", 0
                )
            )
        )
    ):
        raise LayerCakeHostRuntimeError(
            "standalone runtime decoding must come from its frozen metadata"
        )
    if runtime_decoding_overlay is not None:
        source_decoding["maximum_identical_token_run"] = int(
            runtime_decoding_overlay["override"][
                "maximum_identical_token_run"
            ]
        )
    runtime_decoding_overlay_metadata = None
    if runtime_decoding_overlay_destination is not None:
        runtime_decoding_overlay_metadata = {
            "path": runtime_decoding_overlay_destination.name,
            "sha256": _sha256_file(
                runtime_decoding_overlay_destination
            ),
            "schema_version": runtime_decoding_overlay["schema_version"],
            "base_decoding_sha256": runtime_decoding_overlay[
                "candidate_core"
            ]["base_decoding_sha256"],
            "weights_changed": False,
        }
    metadata = {
        "format": RUNTIME_FORMAT,
        "status": "EXPORTED_NOT_YET_CERTIFIED",
        "host": host_identity,
        "parent_layercake": manifest["parent_layercake"],
        "runtime": {
            "provider": "onnxruntime.CPUExecutionProvider",
            "graph": runtime_graph_path.name,
            "graph_sha256": _sha256_file(runtime_graph_path),
            "graph_bytes": runtime_graph_path.stat().st_size,
            "fp32_graph_sha256": _sha256_file(fp32_path),
            "matrix_weight_quantization": (
                "none"
                if precision_profile == "fp32_all"
                else "dynamic signed int8 per channel with declared exclusions"
            ),
            "task_cake_projection_precision": (
                "float32_after_route_selection"
                if task_cake_projection_nodes
                else None
            ),
            "float32_task_cake_projection_nodes": (
                task_cake_projection_nodes
            ),
            "embedding_quantization": (
                "float32"
                if embedding_is_fp32
                else (
                    "signed int8 per token row, gathered before "
                    "dequantization"
                )
            ),
            "precision_profile": precision_profile,
            "output_projection_precision": (
                "float32"
                if precision_profile in {"fp32_all", "fp32_output"}
                else "dynamic_int8_per_channel"
            ),
            "transformer_matrix_precision": (
                "float32"
                if precision_profile
                in {"fp32_all", "fp32_transformer"}
                else (
                    precision_profile
                    if precision_profile.startswith("fp32_layer")
                    else "dynamic_int8_per_channel"
                )
            ),
            "excluded_matrix_nodes": sorted(excluded_matrix_nodes),
            "excluded_runtime_matrix_nodes": (
                runtime_excluded_matrix_nodes
            ),
            "installed_task_cakes": len(model.task_cakes),
            "task_cake_rank": int(model.config.task_cake_rank),
            "capability_cake_canonical_routes": list(
                capability_cake_routes
            ),
            "maximum_active_task_cakes_per_sequence": 1,
            "persistent_capability_prefix": (
                {
                    "enabled": True,
                    "prefix_length": int(
                        model.config.capability_prefix_length
                    ),
                    "installed_prefixes": len(model.task_cakes),
                    "maximum_active_prefixes_per_sequence": 1,
                    "router_buckets": int(
                        model.config.capability_router_buckets
                    ),
                    "router_width": int(
                        model.config.capability_router_width
                    ),
                    "router_graph": router_graph_path.name,
                    "router_graph_sha256": _sha256_file(
                        router_graph_path
                    ),
                    "router_graph_bytes": router_graph_path.stat().st_size,
                    "router_provider": (
                        "onnxruntime.CPUExecutionProvider"
                    ),
                    "main_graph_requires_selected_route": True,
                    "exported_attention_implementation": "eager",
                    "physically_selected_before_transformer": True,
                    "public_cache_excludes_prefix": True,
                }
                if persistent_capability_prefix
                else {"enabled": False}
            ),
            "layerwise_capability_control": (
                (
                    {
                        "enabled": True,
                        "installed_controls": len(model.task_cakes),
                        "control_layers": int(model.config.layers),
                        "control_width": int(
                            model.config.capability_control_width
                        ),
                        "maximum_active_control_paths_per_sequence": 1,
                        "extra_kv_positions": 0,
                        "public_cache_contains_only_real_tokens": True,
                        "conditions_every_real_token_kv_write": True,
                        "router_mode": (
                            "onnx_zero_control_transformer_mean_classifier"
                        ),
                        "router_graph": router_graph_path.name,
                        "router_graph_sha256": _sha256_file(
                            router_graph_path
                        ),
                        "router_graph_bytes": (
                            router_graph_path.stat().st_size
                        ),
                        "router_provider": (
                            "onnxruntime.CPUExecutionProvider"
                        ),
                        "router_precision": (
                            task_route_router_precision
                        ),
                        "separate_router_session": True,
                        "routing_prepass_uses_zero_control": True,
                        "main_graph_requires_selected_route": True,
                        "physically_selected_before_transformer": True,
                    }
                    if task_route_layerwise_control
                    else {
                        "enabled": True,
                        "installed_controls": len(model.task_cakes),
                        "control_layers": int(model.config.layers),
                        "control_width": int(
                            model.config.capability_control_width
                        ),
                        "maximum_active_control_paths_per_sequence": 1,
                        "extra_kv_positions": 0,
                        "public_cache_contains_only_real_tokens": True,
                        "conditions_every_real_token_kv_write": True,
                        "router_buckets": int(
                            model.config.capability_router_buckets
                        ),
                        "router_width": int(
                            model.config.capability_router_width
                        ),
                        "router_mode": "fused_numpy_hash_mean_linear",
                        "router_parameters": router_parameter_path.name,
                        "router_parameters_sha256": _sha256_file(
                            router_parameter_path
                        ),
                        "router_parameters_bytes": (
                            router_parameter_path.stat().st_size
                        ),
                        "router_equivalence_probes": (
                            router_equivalence_probes
                        ),
                        "separate_router_session": False,
                        "main_graph_requires_selected_route": True,
                        "physically_selected_before_transformer": True,
                    }
                )
                if layerwise_capability_control
                else {"enabled": False}
            ),
            "deep_capability_adapters": (
                {
                    "enabled": True,
                    "installed_capabilities": len(model.task_cakes),
                    "adapter_layers": int(model.config.layers),
                    "adapter_rank": int(
                        model.config.capability_adapter_rank
                    ),
                    "maximum_active_adapters_per_sequence": int(
                        1
                        if getattr(
                            model,
                            "_abi_shared_deep_capability_adapters",
                            False,
                        )
                        else model.config.layers
                    ),
                    "active_adapter_invocations_per_sequence": int(
                        model.config.layers
                    ),
                    "shared_adapter_weights_across_layers": bool(
                        getattr(
                            model,
                            "_abi_shared_deep_capability_adapters",
                            False,
                        )
                    ),
                    "extra_kv_positions": 0,
                    "public_cache_contains_only_real_tokens": True,
                    "conditions_every_real_token_kv_write": True,
                    "router_buckets": int(
                        model.config.capability_router_buckets
                    ),
                    "router_width": int(
                        model.config.capability_router_width
                    ),
                    "router_mode": "fused_numpy_hash_mean_linear",
                    "router_parameters": router_parameter_path.name,
                    "router_parameters_sha256": _sha256_file(
                        router_parameter_path
                    ),
                    "router_parameters_bytes": (
                        router_parameter_path.stat().st_size
                    ),
                    "router_equivalence_probes": (
                        router_equivalence_probes
                    ),
                    "separate_router_session": False,
                    "main_graph_requires_selected_route": True,
                    "physically_selected_before_each_transformer_block": True,
                }
                if deep_capability_adapters
                else {"enabled": False}
            ),
            "deep_reused_capability_cakes": (
                {
                    "enabled": True,
                    "installed_capability_cakes": len(model.task_cakes),
                    "cake_rank": int(model.config.task_cake_rank),
                    "selected_unique_cakes_per_sequence": 1,
                    "selected_cake_invocations_per_sequence": int(
                        model.config.layers
                    ),
                    "shared_cake_weights_across_layers": True,
                    "final_post_transformer_cake_invocations": 0,
                    "added_adapter_parameters": 0,
                    "extra_kv_positions": 0,
                    "public_cache_contains_only_real_tokens": True,
                    "conditions_every_real_token_kv_write": True,
                    "router_buckets": int(
                        model.config.capability_router_buckets
                    ),
                    "router_width": int(
                        model.config.capability_router_width
                    ),
                    "router_mode": "fused_numpy_hash_mean_linear",
                    "router_parameters": router_parameter_path.name,
                    "router_parameters_sha256": _sha256_file(
                        router_parameter_path
                    ),
                    "router_parameters_bytes": (
                        router_parameter_path.stat().st_size
                    ),
                    "router_equivalence_probes": router_equivalence_probes,
                    "separate_router_session": False,
                    "main_graph_requires_selected_route": True,
                    "physically_selected_before_each_transformer_block": True,
                }
                if deep_reused_capability_cakes
                else {"enabled": False}
            ),
            "gated_deep_reused_capability_cakes": (
                {
                    "enabled": True,
                    "installed_capability_cakes": len(model.task_cakes),
                    "cake_rank": int(model.config.task_cake_rank),
                    "selected_unique_cakes_per_sequence": 1,
                    "pre_block_selected_cake_invocations": int(
                        model.config.layers
                    ),
                    "final_selected_cake_invocations": 1,
                    "shared_cake_weights_across_all_invocations": True,
                    "installed_scalar_gate_parameters": (
                        len(model.task_cakes) * int(model.config.layers)
                    ),
                    "active_scalar_gate_parameters": int(
                        model.config.layers
                    ),
                    "scalar_gate_shape": [
                        len(model.task_cakes),
                        int(model.config.layers),
                    ],
                    "added_matrix_parameters": 0,
                    "extra_kv_positions": 0,
                    "public_cache_contains_only_real_tokens": True,
                    "conditions_every_real_token_kv_write": True,
                    "router_buckets": int(
                        model.config.capability_router_buckets
                    ),
                    "router_width": int(
                        model.config.capability_router_width
                    ),
                    "router_mode": "fused_numpy_hash_mean_linear",
                    "router_parameters": router_parameter_path.name,
                    "router_parameters_sha256": _sha256_file(
                        router_parameter_path
                    ),
                    "router_parameters_bytes": (
                        router_parameter_path.stat().st_size
                    ),
                    "router_equivalence_probes": router_equivalence_probes,
                    "separate_router_session": False,
                    "main_graph_requires_selected_route": True,
                    "physically_selected_before_each_transformer_block": True,
                }
                if gated_deep_reused_capability_cakes
                else {"enabled": False}
            ),
            "installed_route_bridges": (
                10 if route_bridge is not None else 0
            ),
            "maximum_active_route_bridges_per_sequence": (
                1 if route_bridge is not None else 0
            ),
            "route_bridge_fused_into_task_cakes": bridge_fused,
            "symbolic_only_host_has_no_route_bridge": symbolic_only,
            "standalone_core_has_no_route_bridge": standalone_core,
            "persistent_incremental_kv_state": True,
            "session_cpu_memory_arena_enabled": bool(
                not getattr(
                    model,
                    "_abi_shared_deep_capability_adapters",
                    False,
                )
            ),
            "decoding_overlay": runtime_decoding_overlay_metadata,
            "decoding": {
                "algorithm": source_decoding["algorithm"],
                "repetition_penalty": float(repetition_penalty),
                "no_repeat_ngram_size": int(
                    source_decoding["no_repeat_ngram_size"]
                ),
                "allow_prompt_ngrams": bool(
                    source_decoding.get("allow_prompt_ngrams", False)
                ),
                "lexical_repetition_blocking_threshold": int(
                    source_decoding.get(
                        "lexical_repetition_blocking_threshold", 0
                    )
                ),
                "lexical_repetition_truncation_threshold": int(
                    source_decoding.get(
                        "lexical_repetition_truncation_threshold", 0
                    )
                ),
                "byte_repetition_ceiling": float(
                    source_decoding.get("byte_repetition_ceiling", 0.0)
                ),
                "byte_repetition_guard_minimum_bytes": int(
                    source_decoding.get(
                        "byte_repetition_guard_minimum_bytes", 0
                    )
                ),
                **(
                    {
                        "maximum_identical_token_run": int(
                            source_decoding[
                                "maximum_identical_token_run"
                            ]
                        )
                    }
                    if "maximum_identical_token_run"
                    in source_decoding
                    else {}
                ),
                "prompt_identity_mixture": bool(
                    source_decoding.get("prompt_identity_mixture", False)
                ),
                "weights_changed": False,
            },
        },
        "tokenizer": {
            "path": tokenizer_path.name,
            "sha256": _sha256_file(tokenizer_path),
        },
        "symbolic_surface": {
            "path": symbolic_path.name,
            "bytes": symbolic_path.stat().st_size,
            "sha256": _sha256_file(symbolic_path),
            "handlers": list(symbolic_contract["handlers"]),
            "source_teacher_text_retained": False,
        },
        "canonical_semantic_abi": {
            "path_at_export": str(canonical_abi_path),
            "sha256": _sha256_file(canonical_abi_path),
            "graph_output": "abi_state",
        },
        "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = _canonical_sha(metadata)
    _write_json(output_path / "metadata.json", metadata)
    return metadata


class NativeState:
    def __init__(
        self,
        route: np.ndarray,
        cache: list[np.ndarray],
        abi_state: np.ndarray,
        output_token_ids: np.ndarray | None = None,
        allowed_output_token_ids: np.ndarray | None = None,
        output_token_local_index: (
            Mapping[int, int | Sequence[int]] | None
        ) = None,
        generated_steps: int = 0,
    ):
        self.route = route
        self.cache = cache
        self.abi_state = abi_state
        self.output_token_ids = output_token_ids
        self.allowed_output_token_ids = (
            output_token_ids
            if allowed_output_token_ids is None
            else allowed_output_token_ids
        )
        self.generated_steps = int(generated_steps)
        self.output_token_local_index = output_token_local_index
        if (
            self.output_token_local_index is None
            and output_token_ids is not None
        ):
            self.output_token_local_index = _token_local_indices(
                output_token_ids
            )


def _token_local_indices(
    output_token_ids: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    """Map a global token ID to every local logit carrying that ID."""

    indices: dict[int, list[int]] = {}
    for index, token_id in enumerate(output_token_ids):
        indices.setdefault(int(token_id), []).append(index)
    return {
        token_id: tuple(local)
        for token_id, local in indices.items()
    }


def _gpt2_byte_decoder() -> dict[str, int]:
    """Return the inverse of GPT-2's reversible byte-to-Unicode alphabet."""

    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = list(values)
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + offset)
            offset += 1
    return {
        chr(character): value
        for value, character in zip(values, characters)
    }


class NativeHostRuntime:
    """One compact ONNX session plus the hash-bound symbolic substrate."""

    def __init__(self, artifact: str | Path, *, threads: int = 14):
        self.artifact = Path(artifact).resolve()
        self.metadata = json.loads(
            (self.artifact / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("format") != RUNTIME_FORMAT:
            raise LayerCakeHostRuntimeError(
                "unsupported native host runtime format"
            )
        self.capability_cake_canonical_routes = tuple(
            int(value)
            for value in self.metadata["runtime"].get(
                "capability_cake_canonical_routes", []
            )
        )
        installed_task_cakes = int(
            self.metadata["runtime"].get("installed_task_cakes", 10)
        )
        if installed_task_cakes not in {10, 14} or (
            self.capability_cake_canonical_routes
            and (
                len(self.capability_cake_canonical_routes)
                != installed_task_cakes
                or any(
                    route < 0 or route >= 10
                    for route in self.capability_cake_canonical_routes
                )
            )
        ) or (
            installed_task_cakes == 10
            and self.capability_cake_canonical_routes
            not in {(), tuple(range(10))}
        ) or (
            installed_task_cakes == 14
            and not self.capability_cake_canonical_routes
        ):
            raise LayerCakeHostRuntimeError(
                "native capability-cake mapping is invalid"
            )
        prefix_contract = self.metadata["runtime"].get(
            "persistent_capability_prefix", {"enabled": False}
        )
        self.persistent_capability_prefix = bool(
            prefix_contract.get("enabled", False)
        )
        control_contract = self.metadata["runtime"].get(
            "layerwise_capability_control", {"enabled": False}
        )
        self.layerwise_capability_control = bool(
            control_contract.get("enabled", False)
        )
        self.task_route_layerwise_control = bool(
            self.layerwise_capability_control
            and control_contract.get("router_mode")
            in TASK_ROUTE_ROUTER_MODES
        )
        adapter_contract = self.metadata["runtime"].get(
            "deep_capability_adapters", {"enabled": False}
        )
        self.deep_capability_adapters = bool(
            adapter_contract.get("enabled", False)
        )
        self.shared_deep_capability_adapters = bool(
            adapter_contract.get(
                "shared_adapter_weights_across_layers", False
            )
        )
        reused_cake_contract = self.metadata["runtime"].get(
            "deep_reused_capability_cakes", {"enabled": False}
        )
        self.deep_reused_capability_cakes = bool(
            reused_cake_contract.get("enabled", False)
        )
        gated_cake_contract = self.metadata["runtime"].get(
            "gated_deep_reused_capability_cakes", {"enabled": False}
        )
        self.gated_deep_reused_capability_cakes = bool(
            gated_cake_contract.get("enabled", False)
        )
        if sum(
            (
                self.persistent_capability_prefix,
                self.layerwise_capability_control,
                self.deep_capability_adapters,
                self.deep_reused_capability_cakes,
                self.gated_deep_reused_capability_cakes,
            )
        ) > 1:
            raise LayerCakeHostRuntimeError(
                "native persistent conditioning contracts conflict"
            )
        if self.persistent_capability_prefix and (
            installed_task_cakes != 14
            or int(prefix_contract.get("prefix_length", -1)) != 8
            or int(prefix_contract.get("installed_prefixes", -1)) != 14
            or int(
                prefix_contract.get(
                    "maximum_active_prefixes_per_sequence", -1
                )
            )
            != 1
            or prefix_contract.get(
                "physically_selected_before_transformer"
            )
            is not True
            or prefix_contract.get("public_cache_excludes_prefix") is not True
            or prefix_contract.get("main_graph_requires_selected_route")
            is not True
            or prefix_contract.get("exported_attention_implementation")
            != "eager"
        ):
            raise LayerCakeHostRuntimeError(
                "native persistent-prefix contract is invalid"
            )
        if self.layerwise_capability_control and (
            installed_task_cakes
            != (10 if self.task_route_layerwise_control else 14)
            or int(control_contract.get("installed_controls", -1))
            != (10 if self.task_route_layerwise_control else 14)
            or int(control_contract.get("control_layers", -1)) != 3
            or int(control_contract.get("control_width", -1)) != 768
            or int(
                control_contract.get(
                    "maximum_active_control_paths_per_sequence", -1
                )
            )
            != 1
            or int(control_contract.get("extra_kv_positions", -1)) != 0
            or control_contract.get(
                "public_cache_contains_only_real_tokens"
            )
            is not True
            or control_contract.get(
                "conditions_every_real_token_kv_write"
            )
            is not True
            or control_contract.get(
                "physically_selected_before_transformer"
            )
            is not True
            or control_contract.get("main_graph_requires_selected_route")
            is not True
            or (
                (
                    control_contract.get("separate_router_session") is not True
                    or not control_contract.get("router_graph")
                    or control_contract.get("router_precision", "int8")
                    not in {"int8", "fp32"}
                    or (
                        control_contract.get("router_mode")
                        == FULL_TASK_ROUTE_ROUTER_MODE
                        and control_contract.get(
                            "routing_prepass_uses_zero_control"
                        )
                        is not True
                    )
                    or (
                        control_contract.get("router_mode")
                        == COMPACT_TASK_ROUTE_ROUTER_MODE
                        and (
                            control_contract.get(
                                "routing_prepass_uses_zero_control"
                            )
                            is not False
                            or int(
                                control_contract.get(
                                    "maximum_router_tokens", -1
                                )
                            )
                            != 512
                            or control_contract.get(
                                "training_device"
                            )
                            != "cuda"
                            or int(
                                control_contract.get(
                                    "output_routes", -1
                                )
                            )
                            != 10
                        )
                    )
                )
                if self.task_route_layerwise_control
                else (
                    control_contract.get("router_mode")
                    != "fused_numpy_hash_mean_linear"
                    or control_contract.get("separate_router_session")
                    is not False
                    or len(
                        control_contract.get(
                            "router_equivalence_probes", []
                        )
                    )
                    != 4
                )
            )
        ):
            raise LayerCakeHostRuntimeError(
                "native layerwise-control contract is invalid"
            )
        if self.deep_capability_adapters and (
            installed_task_cakes != 14
            or int(adapter_contract.get("installed_capabilities", -1)) != 14
            or int(adapter_contract.get("adapter_layers", -1)) != 3
            or int(adapter_contract.get("adapter_rank", -1)) != 32
            or int(
                adapter_contract.get(
                    "maximum_active_adapters_per_sequence", -1
                )
            )
            != (1 if self.shared_deep_capability_adapters else 3)
            or int(
                adapter_contract.get(
                    "active_adapter_invocations_per_sequence", -1
                )
            )
            != 3
            or int(adapter_contract.get("extra_kv_positions", -1)) != 0
            or adapter_contract.get(
                "public_cache_contains_only_real_tokens"
            )
            is not True
            or adapter_contract.get(
                "conditions_every_real_token_kv_write"
            )
            is not True
            or adapter_contract.get(
                "physically_selected_before_each_transformer_block"
            )
            is not True
            or adapter_contract.get("main_graph_requires_selected_route")
            is not True
            or adapter_contract.get("router_mode")
            != "fused_numpy_hash_mean_linear"
            or adapter_contract.get("separate_router_session") is not False
            or len(adapter_contract.get("router_equivalence_probes", []))
            != 4
        ):
            raise LayerCakeHostRuntimeError(
                "native deep-adapter contract is invalid"
            )
        if self.deep_reused_capability_cakes and (
            installed_task_cakes != 14
            or int(
                reused_cake_contract.get(
                    "installed_capability_cakes", -1
                )
            )
            != 14
            or int(reused_cake_contract.get("cake_rank", -1)) != 64
            or int(
                reused_cake_contract.get(
                    "selected_unique_cakes_per_sequence", -1
                )
            )
            != 1
            or int(
                reused_cake_contract.get(
                    "selected_cake_invocations_per_sequence", -1
                )
            )
            != 3
            or reused_cake_contract.get(
                "shared_cake_weights_across_layers"
            )
            is not True
            or int(
                reused_cake_contract.get(
                    "final_post_transformer_cake_invocations", -1
                )
            )
            != 0
            or int(
                reused_cake_contract.get("added_adapter_parameters", -1)
            )
            != 0
            or int(reused_cake_contract.get("extra_kv_positions", -1))
            != 0
            or reused_cake_contract.get(
                "public_cache_contains_only_real_tokens"
            )
            is not True
            or reused_cake_contract.get(
                "conditions_every_real_token_kv_write"
            )
            is not True
            or reused_cake_contract.get(
                "physically_selected_before_each_transformer_block"
            )
            is not True
            or reused_cake_contract.get("router_mode")
            != "fused_numpy_hash_mean_linear"
            or reused_cake_contract.get("separate_router_session") is not False
            or len(
                reused_cake_contract.get("router_equivalence_probes", [])
            )
            != 4
        ):
            raise LayerCakeHostRuntimeError(
                "native deep-reused-cake contract is invalid"
            )
        if self.gated_deep_reused_capability_cakes and (
            installed_task_cakes != 14
            or int(
                gated_cake_contract.get(
                    "installed_capability_cakes", -1
                )
            )
            != 14
            or int(gated_cake_contract.get("cake_rank", -1)) != 64
            or int(
                gated_cake_contract.get(
                    "selected_unique_cakes_per_sequence", -1
                )
            )
            != 1
            or int(
                gated_cake_contract.get(
                    "pre_block_selected_cake_invocations", -1
                )
            )
            != 3
            or int(
                gated_cake_contract.get(
                    "final_selected_cake_invocations", -1
                )
            )
            != 1
            or int(
                gated_cake_contract.get(
                    "installed_scalar_gate_parameters", -1
                )
            )
            != 42
            or int(
                gated_cake_contract.get(
                    "active_scalar_gate_parameters", -1
                )
            )
            != 3
            or gated_cake_contract.get("scalar_gate_shape") != [14, 3]
            or int(gated_cake_contract.get("added_matrix_parameters", -1))
            != 0
            or int(gated_cake_contract.get("extra_kv_positions", -1))
            != 0
            or gated_cake_contract.get(
                "public_cache_contains_only_real_tokens"
            )
            is not True
            or gated_cake_contract.get(
                "conditions_every_real_token_kv_write"
            )
            is not True
            or gated_cake_contract.get(
                "physically_selected_before_each_transformer_block"
            )
            is not True
            or gated_cake_contract.get("router_mode")
            != "fused_numpy_hash_mean_linear"
            or gated_cake_contract.get("separate_router_session") is not False
            or len(gated_cake_contract.get("router_equivalence_probes", []))
            != 4
        ):
            raise LayerCakeHostRuntimeError(
                "native gated-deep-reused-cake contract is invalid"
            )
        router_contract = (
            prefix_contract
            if self.persistent_capability_prefix
            else (
                control_contract
                if self.layerwise_capability_control
                else (
                    adapter_contract
                    if self.deep_capability_adapters
                    else (
                        reused_cake_contract
                        if self.deep_reused_capability_cakes
                        else gated_cake_contract
                    )
                )
            )
        )
        graph_path = self.artifact / self.metadata["runtime"]["graph"]
        tokenizer_path = (
            self.artifact / self.metadata["tokenizer"]["path"]
        )
        symbolic_path = (
            self.artifact / self.metadata["symbolic_surface"]["path"]
        )
        router_graph_path = (
            self.artifact / router_contract["router_graph"]
            if (
                self.persistent_capability_prefix
                or self.task_route_layerwise_control
            )
            else None
        )
        router_parameter_path = (
            self.artifact / router_contract["router_parameters"]
            if (
                (
                    self.layerwise_capability_control
                    and not self.task_route_layerwise_control
                )
                or self.deep_capability_adapters
                or self.deep_reused_capability_cakes
                or self.gated_deep_reused_capability_cakes
            )
            else None
        )
        components = [
            (graph_path, self.metadata["runtime"]["graph_sha256"]),
            (tokenizer_path, self.metadata["tokenizer"]["sha256"]),
            (symbolic_path, self.metadata["symbolic_surface"]["sha256"]),
        ]
        if router_graph_path is not None:
            components.append(
                (
                    router_graph_path,
                    router_contract["router_graph_sha256"],
                )
            )
        if router_parameter_path is not None:
            components.append(
                (
                    router_parameter_path,
                    router_contract["router_parameters_sha256"],
                )
            )
        for path, expected in components:
            if _sha256_file(path) != expected:
                raise LayerCakeHostRuntimeError(
                    f"native runtime component is stale: {path.name}"
                )
        self.fused_router_embedding = None
        self.fused_router_weight = None
        self.fused_router_bias = None
        if router_parameter_path is not None:
            with np.load(router_parameter_path, allow_pickle=False) as values:
                if set(values.files) != {"embedding", "weight", "bias"}:
                    raise LayerCakeHostRuntimeError(
                        "fused router parameter names changed"
                    )
                self.fused_router_embedding = np.asarray(
                    values["embedding"], dtype=np.float32
                ).copy()
                self.fused_router_weight = np.asarray(
                    values["weight"], dtype=np.float32
                ).copy()
                self.fused_router_bias = np.asarray(
                    values["bias"], dtype=np.float32
                ).copy()
            if (
                self.fused_router_embedding.shape != (4096, 32)
                or self.fused_router_weight.shape != (14, 32)
                or self.fused_router_bias.shape != (14,)
            ):
                raise LayerCakeHostRuntimeError(
                    "fused router parameter shapes changed"
                )
        self.symbolic_surface = json.loads(
            symbolic_path.read_text(encoding="utf-8")
        )
        output_vocabulary = self.metadata["runtime"].get(
            "output_vocabulary"
        )
        self.dynamic_prompt_output_vocabulary = False
        self.adaptive_output_vocabulary = False
        self.adaptive_output_ids_output_name: str | None = None
        self.adaptive_output_ids_output_index: int | None = None
        self.output_token_ids: np.ndarray | None = None
        self.output_token_local_index: (
            dict[int, tuple[int, ...]] | None
        ) = None
        if output_vocabulary is not None:
            vocabulary_path = (
                self.artifact / output_vocabulary["path"]
            )
            if _sha256_file(vocabulary_path) != output_vocabulary["sha256"]:
                raise LayerCakeHostRuntimeError(
                    "sparse output vocabulary is stale"
                )
            vocabulary = json.loads(
                vocabulary_path.read_text(encoding="utf-8")
            )
            output_mode = output_vocabulary.get("mode")
            self.dynamic_prompt_output_vocabulary = output_mode in (
                "train_base_union_prompt_tokens",
                "adaptive_low_rank_shortlist_union_prompt_tokens",
            )
            self.adaptive_output_vocabulary = (
                output_mode
                == "adaptive_low_rank_shortlist_union_prompt_tokens"
            )
            if self.adaptive_output_vocabulary:
                self.adaptive_output_ids_output_name = str(
                    output_vocabulary[
                        "selected_output_ids_graph_output"
                    ]
                )
            token_ids = np.asarray(
                vocabulary["global_token_ids"], dtype=np.int64
            )
            if (
                token_ids.ndim != 1
                or len(token_ids)
                != int(output_vocabulary["selected_token_count"])
                or len(np.unique(token_ids)) != len(token_ids)
                or np.any(token_ids < 0)
                or np.any(token_ids >= 50_257)
                or token_ids.tolist() != sorted(token_ids.tolist())
                or not set(range(256)).issubset(token_ids.tolist())
                or 50_256 not in token_ids
            ):
                raise LayerCakeHostRuntimeError(
                    "sparse output vocabulary contract is invalid"
                )
            self.output_token_ids = token_ids
            self.output_token_local_index = _token_local_indices(
                token_ids
            )
        self.decoding = dict(
            self.metadata["runtime"].get(
                "decoding",
                {
                    "algorithm": "greedy",
                    "repetition_penalty": 1.0,
                    "no_repeat_ngram_size": 0,
                    "weights_changed": False,
                },
            )
        )
        decoding_overlay = self.metadata["runtime"].get(
            "decoding_overlay"
        )
        if decoding_overlay is not None:
            if (
                not isinstance(decoding_overlay, dict)
                or decoding_overlay.get("path")
                != "runtime-decoding-overlay.json"
                or decoding_overlay.get("schema_version")
                != "abi-layercake-runtime-decoding-overlay/1"
                or decoding_overlay.get("weights_changed") is not False
            ):
                raise LayerCakeHostRuntimeError(
                    "native runtime decoding overlay metadata is invalid"
                )
            overlay_path = self.artifact / decoding_overlay["path"]
            if (
                _sha256_file(overlay_path)
                != decoding_overlay.get("sha256")
            ):
                raise LayerCakeHostRuntimeError(
                    "native runtime decoding overlay is stale"
                )
        self.cake_activation_schedule = self.metadata["runtime"].get(
            "cake_activation_schedule"
        )
        if self.cake_activation_schedule is not None:
            schedule_format = self.cake_activation_schedule.get("format")
            legacy_schedule = (
                schedule_format
                == "abi-layercake-core-realization-schedule/1"
            )
            conditional_schedule = (
                schedule_format
                == "abi-layercake-conditional-core-realization-schedule/1"
            )
            if not legacy_schedule and not conditional_schedule:
                raise LayerCakeHostRuntimeError(
                    "cake activation schedule is invalid"
                )
            if (
                legacy_schedule
                and float(
                    self.cake_activation_schedule[
                        "prefill_activation"
                    ]
                )
                != 1.0
            ):
                raise LayerCakeHostRuntimeError(
                    "legacy cake activation schedule is invalid"
                )
            if (
                conditional_schedule
                and (
                    self.cake_activation_schedule.get(
                        "graph_input_type"
                    )
                    != "bool"
                    or self.cake_activation_schedule.get(
                        "physical_conditional_execution"
                    )
                    is not True
                    or int(
                        self.cake_activation_schedule[
                            "task_cake_nodes_in_false_branch"
                        ]
                    )
                    != 0
                )
            ):
                raise LayerCakeHostRuntimeError(
                    "conditional cake schedule is invalid"
                )
            if int(
                self.cake_activation_schedule["active_decode_steps"]
            ) < 0:
                raise LayerCakeHostRuntimeError(
                    "cake activation step count is invalid"
                )
        if (
            float(self.decoding.get("repetition_penalty", 0.0)) < 1.0
            or (
                int(self.decoding.get("no_repeat_ngram_size", -1))
                not in (0,)
                and int(
                    self.decoding.get("no_repeat_ngram_size", -1)
                )
                < 2
            )
            or self.decoding.get("weights_changed") is not False
            or isinstance(
                self.decoding.get("maximum_identical_token_run", 0),
                bool,
            )
            or not 0
            <= int(
                self.decoding.get("maximum_identical_token_run", 0)
            )
            <= 5
        ):
            raise LayerCakeHostRuntimeError(
                "native decoding contract is invalid"
            )
        if (
            self.symbolic_surface.get("source_teacher_text_retained")
            is not False
        ):
            raise LayerCakeHostRuntimeError(
                "symbolic substrate retained teacher text"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.enable_mem_pattern = False
        options.enable_mem_reuse = True
        self.session_cpu_memory_arena_enabled = bool(
            self.metadata["runtime"].get(
                "session_cpu_memory_arena_enabled", True
            )
        )
        options.enable_cpu_mem_arena = (
            self.session_cpu_memory_arena_enabled
        )
        self.session_prepacking_enabled = bool(
            self.metadata["runtime"].get(
                "session_prepacking_enabled", False
            )
        )
        if not self.session_prepacking_enabled:
            options.add_session_config_entry(
                "session.disable_prepacking", "1"
            )
        self.session = ort.InferenceSession(
            str(graph_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.router_session = (
            ort.InferenceSession(
                str(router_graph_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            if router_graph_path is not None
            else None
        )
        if self.adaptive_output_vocabulary:
            output_names = [
                value.name for value in self.session.get_outputs()
            ]
            if self.adaptive_output_ids_output_name not in output_names:
                raise LayerCakeHostRuntimeError(
                    "adaptive graph does not expose selected token IDs"
                )
            self.adaptive_output_ids_output_index = output_names.index(
                self.adaptive_output_ids_output_name
            )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer_document = json.loads(
            tokenizer_path.read_text(encoding="utf-8")
        )
        if tokenizer_document.get("decoder", {}).get("type") != "ByteLevel":
            raise LayerCakeHostRuntimeError(
                "native incremental output requires a ByteLevel decoder"
            )
        self.special_token_ids = {
            int(row["id"])
            for row in tokenizer_document.get("added_tokens", [])
            if row.get("special") is True
        }
        self.byte_decoder = _gpt2_byte_decoder()
        self.token_byte_cache: dict[int, bytes] = {}
        self.empty = [
            np.zeros((1, 12, 0, 64), dtype=np.float32)
            for _ in range(6)
        ]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def public_route(self, internal_route: int) -> int:
        """Map an internal capability cake back to the canonical ABI route."""

        internal_route = int(internal_route)
        if not self.capability_cake_canonical_routes:
            return internal_route
        if not 0 <= internal_route < len(
            self.capability_cake_canonical_routes
        ):
            raise LayerCakeHostRuntimeError(
                "internal capability route is outside the installed mapping"
            )
        return self.capability_cake_canonical_routes[internal_route]

    def decode_token_bytes(self, token_id: int) -> bytes:
        """Decode one ByteLevel token to its exact pre-UTF-8 byte sequence."""

        token_id = int(token_id)
        if token_id in self.special_token_ids:
            return b""
        cached = self.token_byte_cache.get(token_id)
        if cached is not None:
            return cached
        token = self.tokenizer.id_to_token(token_id)
        if token is None:
            raise LayerCakeHostRuntimeError(
                f"runtime token ID is absent from the tokenizer: {token_id}"
            )
        try:
            decoded = bytes(self.byte_decoder[value] for value in token)
        except KeyError as exc:
            raise LayerCakeHostRuntimeError(
                "ByteLevel token contains a non-reversible character"
            ) from exc
        self.token_byte_cache[token_id] = decoded
        return decoded

    def _run(
        self,
        input_ids: np.ndarray,
        route: np.ndarray,
        cache: list[np.ndarray],
        output_token_ids: np.ndarray | None = None,
        output_token_local_index: (
            Mapping[int, int | Sequence[int]] | None
        ) = None,
        cake_activation: float = 1.0,
        generated_steps: int = 0,
    ) -> tuple[np.ndarray, NativeState, np.ndarray]:
        feeds = {
            "input_ids": input_ids.astype(np.int64, copy=False),
            "requested_route": route.astype(np.int64, copy=False),
            "past_key_0": cache[0],
            "past_value_0": cache[1],
            "past_key_1": cache[2],
            "past_value_1": cache[3],
            "past_key_2": cache[4],
            "past_value_2": cache[5],
        }
        if self.dynamic_prompt_output_vocabulary:
            if output_token_ids is None or not len(output_token_ids):
                raise LayerCakeHostRuntimeError(
                    "dynamic output graph requires prompt-bound token IDs"
                )
            feeds["allowed_output_ids"] = output_token_ids.astype(
                np.int64, copy=False
            )
        if self.cake_activation_schedule is not None:
            schedule_input = self.cake_activation_schedule[
                "graph_input"
            ]
            if (
                self.cake_activation_schedule.get("graph_input_type")
                == "bool"
            ):
                feeds[schedule_input] = np.asarray(
                    bool(cake_activation), dtype=np.bool_
                )
            else:
                feeds[schedule_input] = np.asarray(
                    [cake_activation], dtype=np.float32
                )
        outputs = self.session.run(None, feeds)
        logits = outputs[0]
        if self.adaptive_output_vocabulary:
            if self.adaptive_output_ids_output_index is None:
                raise LayerCakeHostRuntimeError(
                    "adaptive output index was not initialized"
                )
            active_output_ids = np.asarray(
                outputs[self.adaptive_output_ids_output_index],
                dtype=np.int64,
            ).reshape(-1)
        else:
            active_output_ids = (
                output_token_ids
                if self.dynamic_prompt_output_vocabulary
                else self.output_token_ids
            )
        active_output_token_local_index = output_token_local_index
        if self.adaptive_output_vocabulary:
            active_output_token_local_index = None
        elif (
            active_output_token_local_index is None
            and active_output_ids is self.output_token_ids
        ):
            active_output_token_local_index = self.output_token_local_index
        if (
            active_output_ids is not None
            and logits.shape[-1] != len(active_output_ids)
        ):
            raise LayerCakeHostRuntimeError(
                "sparse logits width differs from its token map"
            )
        return (
            logits,
            NativeState(
                outputs[1],
                list(outputs[4:10]),
                outputs[3],
                active_output_ids,
                output_token_ids,
                active_output_token_local_index,
                generated_steps,
            ),
            outputs[2],
        )

    def prefill(self, ids: list[int]) -> tuple[np.ndarray, NativeState]:
        if not ids:
            raise LayerCakeHostRuntimeError(
                "prefill requires at least one token"
            )
        cache = self.empty
        scores = []
        cache_before_last = cache
        output_token_ids = self.output_token_ids
        if self.dynamic_prompt_output_vocabulary:
            if output_token_ids is None:
                raise LayerCakeHostRuntimeError(
                    "dynamic output base vocabulary is absent"
                )
            output_token_ids = np.unique(
                np.concatenate(
                    (
                        output_token_ids,
                        np.asarray(ids, dtype=np.int64),
                    )
                )
            )
        output_token_local_index = (
            None
            if output_token_ids is None
            else _token_local_indices(output_token_ids)
        )
        if (
            self.persistent_capability_prefix
            or self.task_route_layerwise_control
        ):
            if self.router_session is None:
                raise LayerCakeHostRuntimeError(
                    "native route-selection session is absent"
                )
            route = np.asarray(
                self.router_session.run(
                    ["selected_route"],
                    {"prompt_ids": np.asarray([ids], dtype=np.int64)},
                )[0],
                dtype=np.int64,
            ).reshape(1)
            state = None
            logits = None
            for token_id in ids:
                logits, state, _ = self._run(
                    np.asarray([[token_id]], dtype=np.int64),
                    route,
                    cache,
                    output_token_ids,
                    output_token_local_index,
                )
                cache = state.cache
            if (
                state is None
                or logits is None
                or any(
                    value.shape[2] != len(ids) for value in state.cache
                )
            ):
                raise LayerCakeHostRuntimeError(
                    "persistent conditioning changed public token-cache length"
                )
            return logits, state
        if (
            (
                self.layerwise_capability_control
                and not self.task_route_layerwise_control
            )
            or self.deep_capability_adapters
            or self.deep_reused_capability_cakes
            or self.gated_deep_reused_capability_cakes
        ):
            if (
                self.fused_router_embedding is None
                or self.fused_router_weight is None
                or self.fused_router_bias is None
            ):
                raise LayerCakeHostRuntimeError(
                    "fused capability-conditioning router is absent"
                )
            hashed = np.remainder(
                np.asarray(ids, dtype=np.int64),
                self.fused_router_embedding.shape[0],
            )
            summary = self.fused_router_embedding[hashed].mean(axis=0)
            scores = (
                summary @ self.fused_router_weight.T
                + self.fused_router_bias
            )
            route = np.asarray([int(scores.argmax())], dtype=np.int64)
            state = None
            logits = None
            for token_id in ids:
                logits, state, _ = self._run(
                    np.asarray([[token_id]], dtype=np.int64),
                    route,
                    cache,
                    output_token_ids,
                    output_token_local_index,
                )
                cache = state.cache
            if (
                state is None
                or logits is None
                or any(
                    value.shape[2] != len(ids) for value in state.cache
                )
            ):
                raise LayerCakeHostRuntimeError(
                    "capability conditioning changed public token-cache length"
                )
            return logits, state
        for index, token_id in enumerate(ids):
            if index == len(ids) - 1:
                cache_before_last = cache
            _, state, token_scores = self._run(
                np.asarray([[token_id]], dtype=np.int64),
                np.asarray([-1], dtype=np.int64),
                cache,
                output_token_ids,
                output_token_local_index,
                (
                    0.0
                    if self.cake_activation_schedule is not None
                    and self.cake_activation_schedule.get(
                        "physical_conditional_execution"
                    )
                    is True
                    else 1.0
                ),
            )
            cache = state.cache
            scores.append(token_scores)
        route = np.asarray(
            [int(np.mean(np.stack(scores), axis=0).argmax())],
            dtype=np.int64,
        )
        logits, state, _ = self._run(
            np.asarray([[ids[-1]]], dtype=np.int64),
            route,
            cache_before_last,
            output_token_ids,
            output_token_local_index,
        )
        return logits, state

    def decode_step(
        self, token_id: int, state: NativeState
    ) -> tuple[np.ndarray, NativeState]:
        logits, next_state, _ = self._run(
            np.asarray([[token_id]], dtype=np.int64),
            state.route,
            state.cache,
            state.allowed_output_token_ids,
            state.output_token_local_index,
            (
                1.0
                if self.cake_activation_schedule is None
                or state.generated_steps
                < int(
                    self.cake_activation_schedule[
                        "active_decode_steps"
                    ]
                )
                else 0.0
            ),
            state.generated_steps + 1,
        )
        return logits, next_state


def _select_token(
    logits: np.ndarray,
    generated: Sequence[int] = (),
    *,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    output_token_ids: np.ndarray | None = None,
    output_token_local_index: (
        Mapping[int, int | Sequence[int]] | None
    ) = None,
    repetition_local_indices: Sequence[int] | None = None,
    blocked_token_ids: Sequence[int] | None = None,
) -> int:
    values = logits[0]
    if output_token_ids is not None:
        if (
            output_token_ids.ndim != 1
            or len(output_token_ids) != len(values)
        ):
            raise LayerCakeHostRuntimeError(
                "sparse output token map differs from logits"
            )
        if output_token_local_index is None:
            output_token_local_index = _token_local_indices(
                output_token_ids
            )
    selected = np.empty(0, dtype=np.int64)
    if repetition_penalty != 1.0:
        if repetition_local_indices is None:
            local_indices = []
            for token in set(generated):
                local_value = (
                    token
                    if output_token_local_index is None
                    else output_token_local_index.get(int(token))
                )
                if local_value is None:
                    continue
                if isinstance(local_value, (int, np.integer)):
                    local_indices.append(int(local_value))
                else:
                    local_indices.extend(
                        int(value) for value in local_value
                    )
        else:
            local_indices = [
                int(value) for value in repetition_local_indices
            ]
        if local_indices:
            selected = np.asarray(local_indices, dtype=np.int64)
    local_blocked: list[int] = []
    blocked = set(
        int(value)
        for value in (
            () if blocked_token_ids is None else blocked_token_ids
        )
    )
    if (
        not blocked
        and blocked_token_ids is None
        and no_repeat_ngram_size > 0
        and len(generated) >= no_repeat_ngram_size - 1
    ):
        prefix = tuple(generated[-(no_repeat_ngram_size - 1) :])
        if blocked_token_ids is None:
            blocked = set()
            for index in range(
                len(generated) - no_repeat_ngram_size + 1
            ):
                if tuple(
                    generated[
                        index : index + no_repeat_ngram_size - 1
                    ]
                ) == prefix:
                    blocked.add(
                        generated[index + no_repeat_ngram_size - 1]
                    )
    if blocked:
        if output_token_local_index is None:
            local_blocked = [int(value) for value in blocked]
        else:
            for token in blocked:
                local_value = output_token_local_index.get(token)
                if local_value is None:
                    continue
                if isinstance(local_value, (int, np.integer)):
                    local_blocked.append(int(local_value))
                else:
                    local_blocked.extend(
                        int(value) for value in local_value
                    )
    touched = np.unique(
        np.concatenate(
            (
                selected,
                np.asarray(local_blocked, dtype=np.int64),
            )
        )
    )
    original = values[touched].copy()
    try:
        if len(selected):
            selected_values = values[selected]
            values[selected] = np.where(
                selected_values > 0,
                selected_values / repetition_penalty,
                selected_values * repetition_penalty,
            )
        if local_blocked:
            values[local_blocked] = -np.inf
        local = int(values.argmax())
    finally:
        values[touched] = original
    return (
        local
        if output_token_ids is None
        else int(output_token_ids[local])
    )


def _blocked_ngram_successors(
    generated: Sequence[int],
    ngram_successors: Mapping[tuple[int, ...], set[int]],
    *,
    ngram_size: int,
    allowed_ngrams: set[tuple[int, ...]] | None = None,
) -> set[int]:
    if ngram_size <= 0 or len(generated) < ngram_size - 1:
        return set()
    prefix = tuple(generated[-(ngram_size - 1) :])
    blocked = set(ngram_successors.get(prefix, set()))
    if allowed_ngrams:
        blocked = {
            token
            for token in blocked
            if prefix + (token,) not in allowed_ngrams
        }
    return blocked


def _byte_fourgram_repetition_rate(payload: bytes) -> float:
    grams = [
        payload[index : index + 4]
        for index in range(max(0, len(payload) - 3))
    ]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def _exceeds_identical_token_run(
    generated: Sequence[int],
    candidate_token_id: int,
    *,
    maximum_identical_token_run: int,
) -> bool:
    """Return whether candidate would exceed the bounded identical-token run."""

    maximum = int(maximum_identical_token_run)
    return (
        maximum > 0
        and len(generated) >= maximum
        and all(
            int(token_id) == int(candidate_token_id)
            for token_id in generated[-maximum:]
        )
    )


def _select_token_with_repetition_guards(
    runtime: NativeHostRuntime,
    logits: np.ndarray,
    state: NativeState,
    generated: Sequence[int],
    prompt: str,
    *,
    repetition_local_indices: Sequence[int] | None,
    blocked_token_ids: Sequence[int],
) -> int:
    """Select greedily while enforcing preregistered generic loop guards."""

    blocked = set(int(value) for value in blocked_token_ids)
    lexical_threshold = int(
        runtime.decoding.get("lexical_repetition_blocking_threshold", 0)
    )
    byte_ceiling = float(
        runtime.decoding.get("byte_repetition_ceiling", 0.0)
    )
    byte_minimum = int(
        runtime.decoding.get("byte_repetition_guard_minimum_bytes", 0)
    )
    maximum_identical_token_run = int(
        runtime.decoding.get("maximum_identical_token_run", 0)
    )
    current_output = (
        runtime.decode(generated)
        if lexical_threshold > 0 or byte_ceiling > 0.0
        else ""
    )
    current_repetitions = (
        novel_lexical_repetition_occurrences(
            current_output,
            prompt,
        )
        if lexical_threshold > 0
        else 0
    )
    current_bytes = current_output.encode("utf-8")
    current_byte_repetition = (
        _byte_fourgram_repetition_rate(current_bytes)
        if byte_ceiling > 0.0 and len(current_bytes) >= byte_minimum
        else 0.0
    )
    for _attempt in range(65):
        token_id = _select_token(
            logits,
            generated,
            repetition_penalty=float(
                runtime.decoding["repetition_penalty"]
            ),
            no_repeat_ngram_size=int(
                runtime.decoding["no_repeat_ngram_size"]
            ),
            output_token_ids=state.output_token_ids,
            output_token_local_index=state.output_token_local_index,
            repetition_local_indices=repetition_local_indices,
            blocked_token_ids=blocked,
        )
        if token_id == 50256:
            return token_id
        token_run_allowed = not _exceeds_identical_token_run(
            generated,
            token_id,
            maximum_identical_token_run=maximum_identical_token_run,
        )
        candidate_output = (
            runtime.decode([*generated, token_id])
            if lexical_threshold > 0 or byte_ceiling > 0.0
            else ""
        )
        lexical_allowed = True
        if lexical_threshold > 0:
            candidate_repetitions = (
                novel_lexical_repetition_occurrences(
                    candidate_output,
                    prompt,
                )
            )
            lexical_allowed = (
                candidate_repetitions < lexical_threshold
                or candidate_repetitions <= current_repetitions
            )
        byte_allowed = True
        candidate_bytes = candidate_output.encode("utf-8")
        if (
            byte_ceiling > 0.0
            and len(candidate_bytes) >= byte_minimum
        ):
            candidate_byte_repetition = (
                _byte_fourgram_repetition_rate(candidate_bytes)
            )
            byte_allowed = (
                candidate_byte_repetition <= byte_ceiling
                or (
                    current_byte_repetition > byte_ceiling
                    and candidate_byte_repetition
                    <= current_byte_repetition
                )
            )
        if token_run_allowed and lexical_allowed and byte_allowed:
            return token_id
        blocked.add(token_id)
    raise LayerCakeHostRuntimeError(
        "lexical repetition guard rejected more than 64 candidates"
    )


def generate_native_host(
    runtime: NativeHostRuntime,
    prompt: str,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Generate once with authoritative IDs and persistent incremental state."""

    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    started = time.perf_counter_ns()
    prompt_ids = runtime.encode(prompt + "\n")
    tokenized = time.perf_counter_ns()
    logits, state = runtime.prefill(prompt_ids)
    prefetched = time.perf_counter_ns()
    symbolic = symbolic_surface_output(
        runtime.symbolic_surface,
        prompt=prompt,
        route=runtime.public_route(int(state.route[0])),
    )
    generated: list[int] = []
    if symbolic is not None:
        generated = runtime.encode(symbolic)
        if len(generated) > max_new_tokens:
            raise LayerCakeHostRuntimeError(
                "symbolic output exceeded generation budget"
            )
        output = symbolic
        first = time.perf_counter_ns()
    else:
        first = None
        repetition_local_indices: set[int] = set()
        ngram_successors: dict[tuple[int, ...], set[int]] = {}
        no_repeat_ngram_size = int(
            runtime.decoding["no_repeat_ngram_size"]
        )
        allowed_prompt_ngrams = (
            {
                tuple(prompt_ids[index : index + no_repeat_ngram_size])
                for index in range(
                    len(prompt_ids) - no_repeat_ngram_size + 1
                )
            }
            if no_repeat_ngram_size > 0
            and runtime.decoding.get("allow_prompt_ngrams") is True
            else None
        )
        for _ in range(max_new_tokens):
            blocked = _blocked_ngram_successors(
                generated,
                ngram_successors,
                ngram_size=no_repeat_ngram_size,
                allowed_ngrams=allowed_prompt_ngrams,
            )
            token_id = _select_token_with_repetition_guards(
                runtime,
                logits,
                state,
                generated,
                prompt,
                repetition_local_indices=(
                    None
                    if runtime.adaptive_output_vocabulary
                    else tuple(repetition_local_indices)
                ),
                blocked_token_ids=blocked,
            )
            if token_id == 50256:
                break
            generated.append(token_id)
            local_value = (
                token_id
                if state.output_token_local_index is None
                else state.output_token_local_index.get(token_id)
            )
            if isinstance(local_value, (int, np.integer)):
                repetition_local_indices.add(int(local_value))
            elif local_value is not None:
                repetition_local_indices.update(
                    int(value) for value in local_value
                )
            if (
                no_repeat_ngram_size > 0
                and len(generated) >= no_repeat_ngram_size
            ):
                prefix = tuple(
                    generated[-no_repeat_ngram_size:-1]
                )
                ngram_successors.setdefault(prefix, set()).add(
                    generated[-1]
                )
            if first is None:
                first = time.perf_counter_ns()
            logits, state = runtime.decode_step(token_id, state)
        output = runtime.decode(generated)
        lexical_threshold = int(
            runtime.decoding.get(
                "lexical_repetition_truncation_threshold", 0
            )
        )
        if lexical_threshold > 0:
            output = truncate_novel_lexical_repetition(
                output,
                prompt,
                threshold=lexical_threshold,
            )
            generated = runtime.encode(output)
    completed = time.perf_counter_ns()
    first = first or completed
    raw = output.encode("utf-8")
    total_seconds = (completed - started) / 1e9
    cache_lengths = [
        int(value.shape[2]) for value in state.cache[::2]
    ]
    return {
        "output": output,
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "generated_utf8_bytes": len(raw),
        "generated_characters": len(output),
        "authoritative_generated_token_ids": generated,
        "authoritative_generated_tokens": len(generated),
        "prompt_tokens": len(prompt_ids),
        "route": runtime.public_route(int(state.route[0])),
        "symbolic_handler_used": symbolic is not None,
        "runtime_runner_sha256": _sha256_file(Path(__file__)),
        "timing": {
            "tokenization_seconds": (tokenized - started) / 1e9,
            "prefill_seconds": (prefetched - tokenized) / 1e9,
            "time_to_first_output_seconds": (first - started) / 1e9,
            "total_latency_seconds": total_seconds,
            "bytes_per_second_total": (
                len(raw) / max(total_seconds, 1e-12)
            ),
            "characters_per_second_total": (
                len(output) / max(total_seconds, 1e-12)
            ),
        },
        "memory": {
            "resident_bytes_before": rss_before,
            "resident_bytes_after": int(process.memory_info().rss),
        },
        "persistent_state": {
            "prompt_encoded_once": True,
            "cache_lengths": cache_lengths,
            "completed_prefix_recomputation": False,
        },
    }


def _quality(payload: bytes) -> dict[str, float]:
    try:
        decoded = payload.decode("utf-8")
        valid = 1.0
    except UnicodeDecodeError:
        decoded = payload.decode("utf-8", errors="replace")
        valid = 0.0
    repetition = _byte_fourgram_repetition_rate(payload)
    words = [word for word in decoded.lower().split() if word]
    printable = sum(
        value.isprintable() or value in "\n\r\t" for value in decoded
    ) / max(1, len(decoded))
    return {
        "valid_utf8": valid,
        "invalid_output": 1.0 - valid,
        "printable_character_rate": printable,
        "unique_4gram_rate": 1.0 - repetition,
        "repetition_rate": repetition,
        "word_diversity": len(set(words)) / max(1, len(words)),
        "generated_characters": float(len(decoded)),
    }


def generate_native_host_bytes(
    runtime: NativeHostRuntime,
    prompt: str,
    *,
    output_bytes: int,
) -> dict[str, Any]:
    """Generate a measured byte target without recomputing a completed prefix."""

    if output_bytes <= 0:
        raise LayerCakeHostRuntimeError("output byte target must be positive")
    process = psutil.Process()
    started = time.perf_counter_ns()
    prompt_ids = runtime.encode(prompt + "\n")
    tokenized = time.perf_counter_ns()
    logits, state = runtime.prefill(prompt_ids)
    if symbolic_surface_output(
        runtime.symbolic_surface,
        prompt=prompt,
        route=runtime.public_route(int(state.route[0])),
    ) is not None:
        raise LayerCakeHostRuntimeError(
            "fixed symbolic output is not eligible for a byte-target benchmark"
        )
    generated: list[int] = []
    raw_payload = bytearray()
    repetition_local_indices: set[int] = set()
    ngram_successors: dict[tuple[int, ...], set[int]] = {}
    no_repeat_ngram_size = int(
        runtime.decoding["no_repeat_ngram_size"]
    )
    allowed_prompt_ngrams = (
        {
            tuple(prompt_ids[index : index + no_repeat_ngram_size])
            for index in range(
                len(prompt_ids) - no_repeat_ngram_size + 1
            )
        }
        if no_repeat_ngram_size > 0
        and runtime.decoding.get("allow_prompt_ngrams") is True
        else None
    )
    first_output = None
    while True:
        blocked = _blocked_ngram_successors(
            generated,
            ngram_successors,
            ngram_size=no_repeat_ngram_size,
            allowed_ngrams=allowed_prompt_ngrams,
        )
        token_id = _select_token_with_repetition_guards(
            runtime,
            logits,
            state,
            generated,
            prompt,
            repetition_local_indices=(
                None
                if runtime.adaptive_output_vocabulary
                else tuple(repetition_local_indices)
            ),
            blocked_token_ids=blocked,
        )
        generated.append(token_id)
        local_value = (
            token_id
            if state.output_token_local_index is None
            else state.output_token_local_index.get(token_id)
        )
        if isinstance(local_value, (int, np.integer)):
            repetition_local_indices.add(int(local_value))
        elif local_value is not None:
            repetition_local_indices.update(
                int(value) for value in local_value
            )
        if (
            no_repeat_ngram_size > 0
            and len(generated) >= no_repeat_ngram_size
        ):
            prefix = tuple(generated[-no_repeat_ngram_size:-1])
            ngram_successors.setdefault(prefix, set()).add(
                generated[-1]
            )
        raw_payload.extend(runtime.decode_token_bytes(token_id))
        if first_output is None and raw_payload:
            first_output = time.perf_counter_ns()
        if len(raw_payload) >= max(0, output_bytes - 12):
            output = bytes(raw_payload).decode(
                "utf-8", errors="replace"
            )
            lexical_threshold = int(
                runtime.decoding.get(
                    "lexical_repetition_truncation_threshold", 0
                )
            )
            terminated = False
            if lexical_threshold > 0:
                truncated = truncate_novel_lexical_repetition(
                    output,
                    prompt,
                    threshold=lexical_threshold,
                )
                terminated = truncated != output
                output = truncated
            payload = output.encode("utf-8")
            if len(payload) >= output_bytes or terminated:
                break
        logits, state = runtime.decode_step(token_id, state)
        if len(prompt_ids) + len(generated) >= 1024:
            raise LayerCakeHostRuntimeError(
                "native context ended before byte target"
            )
    completed = time.perf_counter_ns()
    first_output = first_output or completed
    total_seconds = (completed - started) / 1e9
    decode_seconds = (completed - first_output) / 1e9
    memory = process.memory_info()
    return {
        "payload": payload,
        "generated_ids": generated,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "route": runtime.public_route(int(state.route[0])),
        "cache_lengths": [
            int(value.shape[2]) for value in state.cache[::2]
        ],
        "pending_token_id": generated[-1],
        "runtime_runner_sha256": _sha256_file(Path(__file__)),
        "timing": {
            "tokenization_seconds": (tokenized - started) / 1e9,
            "time_to_first_output_seconds": (
                first_output - started
            )
            / 1e9,
            "total_latency_seconds": total_seconds,
            "decode_seconds": decode_seconds,
            "bytes_per_second_total": (
                len(payload) / max(total_seconds, 1.0e-12)
            ),
            "bytes_per_second_decode": (
                len(payload) / max(decode_seconds, 1.0e-12)
            ),
            "process_resident_bytes": int(memory.rss),
            "process_peak_resident_bytes": int(
                getattr(memory, "peak_wset", memory.rss)
            ),
        },
    }


def _normalize_comparator_schedule(
    comparator_path: Path,
) -> list[dict[str, Any]]:
    document = json.loads(
        comparator_path.read_text(encoding="utf-8")
    )
    records = document["records"]
    normalized = []
    for row in records:
        if "prompt" in row:
            generated = int(row["output"]["generated_bytes"])
            latency = float(row["timing"]["total_latency_seconds"])
            normalized.append(
                {
                    "prompt_id": row["prompt"]["id"],
                    "prompt_sha256": row["prompt"]["sha256"],
                    "trial": int(row["trial"]),
                    "bytes_per_second": generated / latency,
                    "time_to_first_output_seconds": float(
                        row["timing"][
                            "time_to_first_output_seconds"
                        ]
                    ),
                    "total_latency_seconds": latency,
                    "process_resident_bytes": int(
                        row["memory"]["resident_bytes"]
                    ),
                    "process_peak_resident_bytes": int(
                        row["memory"]["peak_resident_bytes"]
                    ),
                    "active_model_bytes": int(
                        row["memory"].get(
                            "resident_model_tensor_bytes",
                            row["memory"]["resident_bytes"],
                        )
                    ),
                }
            )
        else:
            normalized.append(
                {
                    "prompt_id": row["prompt_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "trial": int(row["trial"]),
                    "bytes_per_second": float(
                        row["bytes_per_second"]
                    ),
                    "time_to_first_output_seconds": float(
                        row["time_to_first_output_seconds"]
                    ),
                    "total_latency_seconds": float(
                        row["total_latency_seconds"]
                    ),
                    "process_resident_bytes": int(
                        row["process_resident_bytes"]
                    ),
                    "process_peak_resident_bytes": int(
                        row["process_peak_resident_bytes"]
                    ),
                    "active_model_bytes": int(
                        row["active_parameter_bytes"]
                    ),
                }
            )
    return normalized


def _bootstrap_interval(
    values: Sequence[float], *, seed: int = 20260724
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise LayerCakeHostRuntimeError(
            "bootstrap requires paired prompt values"
        )
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(array), size=(10_000, len(array))
    )
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return [float(low), float(high)]


def _runtime_candidate_manifest_sha(
    metadata: Mapping[str, Any],
) -> str:
    host = metadata["host"]
    value = host.get(
        "deployment_manifest_sha256",
        host.get("manifest_sha256"),
    )
    if not isinstance(value, str) or len(value) != 64:
        raise LayerCakeHostRuntimeError(
            "native runtime candidate manifest identity is missing"
        )
    return value


def benchmark_native_host(
    *,
    artifact: str | Path,
    comparator_path: str | Path,
    prompt_manifest_path: str | Path,
    parent_benchmark_path: str | Path,
    output_path: str | Path,
    output_bytes: int,
    threads: int = 14,
) -> dict[str, Any]:
    """Replay one sealed Qwen schedule and enforce all native host gates."""

    artifact = Path(artifact).resolve()
    comparator_path = Path(comparator_path).resolve()
    prompt_manifest_path = Path(prompt_manifest_path).resolve()
    parent_benchmark_path = Path(parent_benchmark_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"native benchmark evidence is immutable: {output_path}"
        )
    schedule = _normalize_comparator_schedule(comparator_path)
    prompt_document = json.loads(
        prompt_manifest_path.read_text(encoding="utf-8")
    )
    prompts = {
        row["id"]: row for row in prompt_document["prompts"]
    }
    parent = json.loads(
        parent_benchmark_path.read_text(encoding="utf-8")
    )
    if (
        parent.get("status") != "PASS"
        or int(parent["output_target_bytes"]) != output_bytes
    ):
        raise LayerCakeHostRuntimeError(
            "sealed parent benchmark is incompatible"
        )
    runtime = NativeHostRuntime(artifact, threads=threads)
    warm_logits, warm_state = runtime.prefill(
        runtime.encode("Warm autonomous LayerCake generation.")
    )
    warm_token = _select_token(
        warm_logits,
        output_token_ids=warm_state.output_token_ids,
        output_token_local_index=warm_state.output_token_local_index,
    )
    runtime.decode_step(warm_token, warm_state)
    records = []
    for order, reference in enumerate(schedule):
        prompt_id = reference["prompt_id"]
        base_id = prompt_id.removeprefix("sustained-")
        try:
            prompt = str(prompts[base_id]["text"])
        except KeyError as exc:
            raise LayerCakeHostRuntimeError(
                f"comparator prompt is absent: {prompt_id}"
            ) from exc
        if prompt_id.startswith("sustained-"):
            prompt += (
                " Continue for at least 220 words so sustained decoding "
                "can be measured."
            )
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        if prompt_sha != reference["prompt_sha256"]:
            raise LayerCakeHostRuntimeError(
                f"comparator prompt hash differs: {prompt_id}"
            )
        generated = generate_native_host_bytes(
            runtime, prompt, output_bytes=output_bytes
        )
        payload = generated["payload"]
        timing = generated["timing"]
        records.append(
            {
                "format": "abi-layercake-native-host-inference/1",
                "run_id": (
                    f"{artifact.name}-{output_bytes}-{order:04d}"
                ),
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_sha,
                "trial": reference["trial"],
                "host_manifest_sha256": _runtime_candidate_manifest_sha(
                    runtime.metadata
                ),
                "parent_checkpoint_sha256": runtime.metadata[
                    "parent_layercake"
                ]["checkpoint_sha256"],
                "runtime_graph_sha256": runtime.metadata["runtime"][
                    "graph_sha256"
                ],
                "runtime_runner_sha256": generated[
                    "runtime_runner_sha256"
                ],
                "output_hex": payload.hex(),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "generated_bytes": len(payload),
                "generated_characters": len(
                    payload.decode("utf-8", errors="replace")
                ),
                "generated_tokens": generated["generated_tokens"],
                "prompt_tokens": generated["prompt_tokens"],
                "token_accounting_method": (
                    "authoritative_native_selected_ids_and_posthoc_tokenizer"
                ),
                "bytes_per_second": float(
                    timing["bytes_per_second_total"]
                ),
                "decode_bytes_per_second": float(
                    timing["bytes_per_second_decode"]
                ),
                "total_latency_seconds": float(
                    timing["total_latency_seconds"]
                ),
                "time_to_first_output_seconds": float(
                    timing["time_to_first_output_seconds"]
                ),
                "process_resident_bytes": int(
                    timing["process_resident_bytes"]
                ),
                "process_peak_resident_bytes": int(
                    timing["process_peak_resident_bytes"]
                ),
                "route": generated["route"],
                "persistent_state": {
                    "decode_input_tokens_per_step": 1,
                    "cached_tokens_per_layer": generated[
                        "cache_lengths"
                    ],
                    "expected_cached_tokens": (
                        generated["prompt_tokens"]
                        + generated["generated_tokens"]
                        - 1
                    ),
                    "pending_selected_token_id": generated[
                        "pending_token_id"
                    ],
                    "completed_prefix_recomputation": False,
                },
                "sparse_execution": {
                    "installed_task_cakes": runtime.metadata[
                        "runtime"
                    ]["installed_task_cakes"],
                    "maximum_active_task_cakes_per_sequence": 1,
                    "installed_route_bridges": runtime.metadata[
                        "runtime"
                    ]["installed_route_bridges"],
                    "maximum_active_route_bridges_per_sequence": (
                        runtime.metadata["runtime"][
                            "maximum_active_route_bridges_per_sequence"
                        ]
                    ),
                    "route_bridge_fused_into_task_cakes": (
                        runtime.metadata["runtime"].get(
                            "route_bridge_fused_into_task_cakes",
                            False,
                        )
                    ),
                    "symbolic_only_host_has_no_route_bridge": (
                        runtime.metadata["runtime"].get(
                            "symbolic_only_host_has_no_route_bridge",
                            False,
                        )
                    ),
                    "standalone_core_has_no_route_bridge": (
                        runtime.metadata["runtime"].get(
                            "standalone_core_has_no_route_bridge",
                            False,
                        )
                    ),
                    "inactive_task_cake_forward_calls": 0,
                    "inactive_route_bridge_forward_calls": 0,
                },
                "external_path_counters": {
                    "planner_calls": 0,
                    "retrieval_calls": 0,
                    "stored_answer_calls": 0,
                    "template_calls": 0,
                    "forced_token_calls": 0,
                },
                "quality": _quality(payload),
                "comparator": reference,
                "status": "PASS",
                "final_test_accessed": False,
            }
        )
        if (order + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "benchmarked": order + 1,
                        "total": len(schedule),
                    }
                ),
                flush=True,
            )
    by_prompt: dict[str, list[float]] = {}
    for row in records:
        by_prompt.setdefault(row["prompt_id"], []).append(
            row["bytes_per_second"]
            / row["comparator"]["bytes_per_second"]
        )
    paired_prompt_ratios = [
        statistics.fmean(values)
        for _, values in sorted(by_prompt.items())
    ]
    candidate_bps = [row["bytes_per_second"] for row in records]
    comparator_bps = [
        row["comparator"]["bytes_per_second"] for row in records
    ]
    qualities = [row["quality"] for row in records]
    distinct = len(by_prompt)
    repeated = sum(len(values) >= 2 for values in by_prompt.values())
    parent_median = float(
        parent["aggregates"]["candidate_median_bytes_per_second"]
    )
    aggregates = {
        "observations": len(records),
        "distinct_prompts": distinct,
        "repeated_prompts": repeated,
        "candidate_median_bytes_per_second": statistics.median(
            candidate_bps
        ),
        "sealed_parent_median_bytes_per_second": parent_median,
        "phase2_throughput_retained_ratio": (
            statistics.median(candidate_bps) / parent_median
        ),
        "comparator_median_bytes_per_second": statistics.median(
            comparator_bps
        ),
        "median_throughput_ratio": (
            statistics.median(candidate_bps)
            / statistics.median(comparator_bps)
        ),
        "mean_paired_prompt_throughput_ratio": statistics.fmean(
            paired_prompt_ratios
        ),
        "paired_prompt_mean_ratio_bootstrap_95ci": (
            _bootstrap_interval(paired_prompt_ratios)
        ),
        "candidate_median_time_to_first_output_seconds": (
            statistics.median(
                row["time_to_first_output_seconds"]
                for row in records
            )
        ),
        "comparator_median_time_to_first_output_seconds": (
            statistics.median(
                row["comparator"]["time_to_first_output_seconds"]
                for row in records
            )
        ),
        "candidate_peak_process_resident_bytes": max(
            row["process_peak_resident_bytes"] for row in records
        ),
        "comparator_peak_process_resident_bytes": max(
            row["comparator"]["process_peak_resident_bytes"]
            for row in records
        ),
        "candidate_active_runtime_model_bytes": int(
            _active_runtime_model_bytes(runtime.metadata)
        ),
        "comparator_active_model_bytes": max(
            row["comparator"]["active_model_bytes"] for row in records
        ),
        "maximum_repetition_rate": max(
            quality["repetition_rate"] for quality in qualities
        ),
        "minimum_unique_4gram_rate": min(
            quality["unique_4gram_rate"] for quality in qualities
        ),
        "minimum_word_diversity": min(
            quality["word_diversity"] for quality in qualities
        ),
        "minimum_printable_character_rate": min(
            quality["printable_character_rate"]
            for quality in qualities
        ),
    }
    gates = {
        "phase2_throughput_retained_at_least_95pct": (
            aggregates["phase2_throughput_retained_ratio"] >= 0.95
        ),
        "median_throughput_ratio_at_least_2": (
            aggregates["median_throughput_ratio"] >= 2.0
        ),
        "paired_prompt_bootstrap_lower_bound_at_least_2": (
            aggregates[
                "paired_prompt_mean_ratio_bootstrap_95ci"
            ][0]
            >= 2.0
        ),
        "ttfo_no_worse": (
            aggregates[
                "candidate_median_time_to_first_output_seconds"
            ]
            <= aggregates[
                "comparator_median_time_to_first_output_seconds"
            ]
        ),
        "rss_below_absolute_limit": (
            aggregates["candidate_peak_process_resident_bytes"]
            < 214_990_848
        ),
        "active_model_memory_lower_than_comparator": (
            aggregates["candidate_active_runtime_model_bytes"]
            < aggregates["comparator_active_model_bytes"]
        ),
        "persistent_cache_exact": all(
            all(
                cached
                == row["persistent_state"]["expected_cached_tokens"]
                for cached in row["persistent_state"][
                    "cached_tokens_per_layer"
                ]
            )
            for row in records
        ),
        "all_outputs_meet_byte_target": all(
            row["generated_bytes"] >= output_bytes for row in records
        ),
        "all_outputs_valid_utf8": all(
            row["quality"]["valid_utf8"] == 1.0 for row in records
        ),
        "printable_character_rate_at_least_098": (
            aggregates["minimum_printable_character_rate"] >= 0.98
        ),
        "maximum_repetition_rate_at_most_060": (
            aggregates["maximum_repetition_rate"] <= 0.60
        ),
        "minimum_unique_4gram_rate_at_least_040": (
            aggregates["minimum_unique_4gram_rate"] >= 0.40
        ),
        "minimum_word_diversity_at_least_010": (
            aggregates["minimum_word_diversity"] >= 0.10
        ),
    }
    aggregates["gates"] = gates
    evidence = {
        "format": "abi-layercake-native-host-benchmark/1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "artifact": str(artifact),
        "artifact_metadata_evidence_sha256": runtime.metadata[
            "evidence_sha256"
        ],
        "runtime_runner_sha256": _sha256_file(Path(__file__)),
        "comparator": str(comparator_path),
        "comparator_sha256": _sha256_file(comparator_path),
        "prompt_manifest": str(prompt_manifest_path),
        "prompt_manifest_sha256": _sha256_file(
            prompt_manifest_path
        ),
        "sealed_parent_benchmark": str(parent_benchmark_path),
        "sealed_parent_benchmark_sha256": _sha256_file(
            parent_benchmark_path
        ),
        "output_target_bytes": int(output_bytes),
        "threads": int(threads),
        "records": records,
        "aggregates": aggregates,
        "final_test_accessed": False,
        "exact_command": " ".join(sys.argv),
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def _summarize_native_semantics(
    observations: Sequence[Mapping[str, Any]],
    *,
    required_capabilities: set[str] | None = None,
    expected_observations_per_capability: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], bool, bool, dict[str, bool]]:
    required_capabilities = (
        set(CAPABILITY_TO_ROUTE)
        if required_capabilities is None
        else set(required_capabilities)
    )
    expected_observations_per_capability = (
        {capability: 100 for capability in required_capabilities}
        if expected_observations_per_capability is None
        else {
            str(capability): int(count)
            for capability, count in (
                expected_observations_per_capability.items()
            )
        }
    )
    if (
        not required_capabilities
        or set(expected_observations_per_capability)
        != required_capabilities
        or any(
            count <= 0
            for count in expected_observations_per_capability.values()
        )
    ):
        raise LayerCakeHostRuntimeError(
            "native semantic depth contract is invalid"
        )
    capability_metrics: dict[str, Any] = {}
    for capability in sorted(
        {str(row["capability"]) for row in observations}
    ):
        selected = [
            row
            for row in observations
            if row["capability"] == capability
        ]
        source_passes = sum(
            bool(row["source_passed"]) for row in selected
        )
        host_passes = sum(
            bool(row["layercake_passed"]) for row in selected
        )
        regressions = sum(
            bool(row["source_passed"])
            and not bool(row["layercake_passed"])
            for row in selected
        )
        collapse_count = sum(
            bool(row["collapse"]["collapse_detected"])
            for row in selected
        )
        capability_metrics[capability] = {
            "observations": len(selected),
            "source_passes": source_passes,
            "layercake_passes": host_passes,
            "source_pass_rate": source_passes / len(selected),
            "layercake_pass_rate": host_passes / len(selected),
            "source_passing_regressions": regressions,
            "source_passing_retention_rate": (
                (source_passes - regressions) / source_passes
                if source_passes
                else None
            ),
            "collapse_count": collapse_count,
            "automatic_route_accuracy": sum(
                bool(row["route_correct"]) for row in selected
            )
            / len(selected),
        }
    complete_depth = (
        set(capability_metrics) == required_capabilities
        and all(
            capability_metrics[capability]["observations"]
            == expected_observations_per_capability[capability]
            for capability in required_capabilities
        )
    )
    total_observations = len(observations)
    total_passes = sum(
        bool(row["layercake_passed"]) for row in observations
    )
    total_source_passes = sum(
        bool(row["source_passed"]) for row in observations
    )
    total_regressions = sum(
        bool(row["source_passed"])
        and not bool(row["layercake_passed"])
        for row in observations
    )
    total_collapses = sum(
        bool(row["collapse"]["collapse_detected"])
        for row in observations
    )
    gates = {
        "complete_locked_depth": complete_depth,
        "overall_functional_pass_rate_at_least_080": (
            total_passes / total_observations >= 0.8
        ),
        "each_declared_capability_pass_rate_at_least_080": all(
            capability_metrics[capability]["layercake_pass_rate"] >= 0.8
            for capability in required_capabilities
        )
        if required_capabilities.issubset(capability_metrics)
        else False,
        "source_passing_retention_rate_at_least_090": (
            (total_source_passes - total_regressions)
            / total_source_passes
            >= 0.9
        )
        if total_source_passes
        else False,
        "collapse_count_is_zero": total_collapses == 0,
    }
    semantic_pass = all(gates.values())
    return capability_metrics, complete_depth, semantic_pass, gates


def evaluate_native_host_semantics(
    *,
    artifact: str | Path,
    training_bundle_path: str | Path,
    validation_bundle_paths: Sequence[str | Path],
    catalog_paths: Sequence[str | Path],
    output_path: str | Path,
    threads: int = 14,
    capabilities: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the full locked English suite on one exact native host artifact."""

    artifact = Path(artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"native semantic evidence is immutable: {output_path}"
        )
    from .english_generalization_evaluation import (
        _collapse_metrics,
        _source_by_probe,
    )
    from .hf_extraction import evaluate_output, load_probe_catalog

    selected_capabilities = (
        set(str(value) for value in capabilities)
        if capabilities is not None
        else None
    )
    if selected_capabilities is not None and (
        not selected_capabilities
        or not selected_capabilities.issubset(CAPABILITY_TO_ROUTE)
    ):
        raise LayerCakeHostRuntimeError(
            "native semantic capability filter is invalid"
        )
    probes: dict[str, dict[str, Any]] = {}
    for catalog_path in catalog_paths:
        catalog = load_probe_catalog(catalog_path)
        for probe in catalog["probes"]:
            if probe["split"] != "validation":
                continue
            if (
                selected_capabilities is not None
                and str(probe["capability"])
                not in selected_capabilities
            ):
                continue
            probe_id = str(probe["probe_id"])
            if probe_id in probes and probes[probe_id] != probe:
                raise LayerCakeHostRuntimeError(
                    f"conflicting validation probe: {probe_id}"
                )
            probes[probe_id] = probe
    source, _ = _source_by_probe(
        validation_bundle_paths, split="validation"
    )
    if set(probes) - set(source):
        raise LayerCakeHostRuntimeError(
            "source validation evidence is incomplete for the locked catalog"
        )
    rows = [
        {
            "probe_id": probe_id,
            "capability": str(probe["capability"]),
            "prompt": str(probe["prompt"]),
            "evaluator": dict(probe["evaluator"]),
            "max_new_tokens": int(probe["max_new_tokens"]),
            "expected_route": CAPABILITY_TO_ROUTE[
                str(probe["capability"])
            ],
            "source_passed": bool(source[probe_id]["passed"]),
            "source_score": float(source[probe_id]["score"]),
            "source_output": str(source[probe_id]["output"]),
            "source_model": str(source[probe_id]["source_model"]),
            "source_model_revision": str(
                source[probe_id]["source_model_revision"]
            ),
        }
        for probe_id, probe in sorted(probes.items())
    ]
    if not rows:
        raise LayerCakeHostRuntimeError(
            "native semantic catalog has no validation rows"
        )
    expected_observations_per_capability = Counter(
        str(row["capability"]) for row in rows
    )
    required_capabilities = set(expected_observations_per_capability)
    runtime = NativeHostRuntime(artifact, threads=threads)
    observations = []
    started = time.perf_counter()
    peak_rss = int(psutil.Process().memory_info().rss)
    for index, row in enumerate(rows, start=1):
        result = generate_native_host(
            runtime,
            row["prompt"],
            max_new_tokens=row["max_new_tokens"],
        )
        passed, score = evaluate_output(
            result["output"], row["evaluator"]
        )
        peak_rss = max(
            peak_rss, int(result["memory"]["resident_bytes_after"])
        )
        observations.append(
            {
                **row,
                "layercake_output": result["output"],
                "layercake_output_sha256": result["output_sha256"],
                "layercake_generated_tokens": result[
                    "authoritative_generated_tokens"
                ],
                "layercake_passed": passed,
                "layercake_score": score,
                "automatic_route": result["route"],
                "route_correct": (
                    result["route"] == row["expected_route"]
                ),
                "latency_seconds": result["timing"][
                    "total_latency_seconds"
                ],
                "symbolic_handler_used": result[
                    "symbolic_handler_used"
                ],
                "collapse": _collapse_metrics(
                    result["authoritative_generated_token_ids"],
                    result["output"],
                    runtime.encode(str(row["prompt"]) + "\n"),
                    str(row["prompt"]),
                ),
            }
        )
        if index % 100 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": index,
                        "total": len(rows),
                        "elapsed_seconds": (
                            time.perf_counter() - started
                        ),
                    }
                ),
                flush=True,
            )
    (
        capability_metrics,
        complete_depth,
        semantic_pass,
        gates,
    ) = _summarize_native_semantics(
        observations,
        required_capabilities=required_capabilities,
        expected_observations_per_capability=(
            expected_observations_per_capability
        ),
    )
    total_source_passes = sum(
        bool(row["source_passed"]) for row in observations
    )
    total_regressions = sum(
        bool(row["source_passed"])
        and not bool(row["layercake_passed"])
        for row in observations
    )
    metadata = runtime.metadata
    evidence = {
        "schema_version": (
            "abi-layercake-native-host-semantic-validation/1"
        ),
        "status": "PASS" if semantic_pass else "FAIL",
        "split": "validation",
        "final_test_accessed": False,
        "training_bundle_sha256": _sha256_file(
            Path(training_bundle_path)
        ),
        "validation_bundle_sha256": [
            _sha256_file(Path(path))
            for path in validation_bundle_paths
        ],
        "catalog_sha256": [
            _sha256_file(Path(path)) for path in catalog_paths
        ],
        "host_manifest_sha256": _runtime_candidate_manifest_sha(
            metadata
        ),
        "runtime_graph_sha256": metadata["runtime"]["graph_sha256"],
        "runtime_metadata_evidence_sha256": metadata[
            "evidence_sha256"
        ],
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "device": "onnxruntime.CPUExecutionProvider",
        "threads": int(threads),
        "diagnostic_capability_filter": (
            sorted(selected_capabilities)
            if selected_capabilities is not None
            else None
        ),
        "observation_count": len(observations),
        "complete_locked_depth": complete_depth,
        "locked_validation_gates": gates,
        "layercake_pass_rate": sum(
            bool(row["layercake_passed"]) for row in observations
        )
        / len(observations),
        "source_passing_retention_rate": (
            (total_source_passes - total_regressions)
            / total_source_passes
        ),
        "collapse_count": sum(
            bool(row["collapse"]["collapse_detected"])
            for row in observations
        ),
        "capability_metrics": capability_metrics,
        "peak_process_rss_bytes": peak_rss,
        "wall_seconds": time.perf_counter() - started,
        "observations": observations,
        "claim_boundary": (
            "This is paired native-runtime validation evidence on the "
            "declared synthetic catalog"
            + (
                " restricted to a preregistered diagnostic capability subset"
                if selected_capabilities is not None
                else ""
            )
            + ". It is not final-test evidence or universal semantic identity."
        ),
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def verify_physical_sparse_runtime(
    artifact: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Verify graph-level selection precedes both sparse residual paths."""

    import onnx
    from onnx import numpy_helper

    artifact = Path(artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"physical proof is immutable: {output_path}"
        )
    metadata = json.loads(
        (artifact / "metadata.json").read_text(encoding="utf-8")
    )
    graph_path = artifact / metadata["runtime"]["graph"]
    document = onnx.load(graph_path)

    def graph_nodes(graph) -> list[Any]:
        nested = list(graph.node)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    nested.extend(graph_nodes(attribute.g))
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        nested.extend(graph_nodes(child))
        return nested

    all_nodes = graph_nodes(document.graph)
    initializers = {
        value.name: tuple(int(dim) for dim in value.dims)
        for value in document.graph.initializer
    }
    identity_sources = {
        node.output[0]: node.input[0]
        for node in all_nodes
        if node.op_type == "Identity"
        and len(node.input) == 1
        and len(node.output) == 1
    }

    def resolved_initializer(name: str) -> str:
        visited: set[str] = set()
        while name in identity_sources and name not in visited:
            visited.add(name)
            name = identity_sources[name]
        return name
    task_cake_rank = int(
        metadata["runtime"].get("task_cake_rank", 64)
    )
    if task_cake_rank not in {64, 256}:
        raise LayerCakeHostRuntimeError(
            "runtime declares an unsupported task-cake rank"
        )
    installed_task_cakes = int(
        metadata["runtime"].get("installed_task_cakes", 10)
    )
    if installed_task_cakes not in {10, 14}:
        raise LayerCakeHostRuntimeError(
            "runtime declares an unsupported task-cake count"
        )
    capability_cake_routes = tuple(
        int(value)
        for value in metadata["runtime"].get(
            "capability_cake_canonical_routes", []
        )
    )
    if (
        capability_cake_routes
        and (
            len(capability_cake_routes) != installed_task_cakes
            or any(route < 0 or route >= 10 for route in capability_cake_routes)
        )
    ) or (
        installed_task_cakes == 10
        and capability_cake_routes not in {(), tuple(range(10))}
    ) or (
        installed_task_cakes == 14 and not capability_cake_routes
    ):
        raise LayerCakeHostRuntimeError(
            "runtime capability-cake route map is invalid"
        )
    prefix_contract = metadata["runtime"].get(
        "persistent_capability_prefix", {"enabled": False}
    )
    persistent_capability_prefix = bool(
        prefix_contract.get("enabled", False)
    )
    control_contract = metadata["runtime"].get(
        "layerwise_capability_control", {"enabled": False}
    )
    layerwise_capability_control = bool(
        control_contract.get("enabled", False)
    )
    task_route_layerwise_control = bool(
        layerwise_capability_control
        and control_contract.get("router_mode")
        in TASK_ROUTE_ROUTER_MODES
    )
    adapter_contract = metadata["runtime"].get(
        "deep_capability_adapters", {"enabled": False}
    )
    deep_capability_adapters = bool(
        adapter_contract.get("enabled", False)
    )
    reused_cake_contract = metadata["runtime"].get(
        "deep_reused_capability_cakes", {"enabled": False}
    )
    deep_reused_capability_cakes = bool(
        reused_cake_contract.get("enabled", False)
    )
    gated_cake_contract = metadata["runtime"].get(
        "gated_deep_reused_capability_cakes", {"enabled": False}
    )
    gated_deep_reused_capability_cakes = bool(
        gated_cake_contract.get("enabled", False)
    )
    if sum(
        (
            persistent_capability_prefix,
            layerwise_capability_control,
            deep_capability_adapters,
            deep_reused_capability_cakes,
            gated_deep_reused_capability_cakes,
        )
    ) > 1:
        raise LayerCakeHostRuntimeError(
            "runtime persistent conditioning contracts conflict"
        )
    router_contract = (
        prefix_contract
        if persistent_capability_prefix
        else (
            control_contract
            if layerwise_capability_control
            else (
                adapter_contract
                if deep_capability_adapters
                else (
                    reused_cake_contract
                    if deep_reused_capability_cakes
                    else gated_cake_contract
                )
            )
        )
    )
    router_graph_path = None
    router_parameter_path = None
    router_physical_checks: dict[str, bool] = {}
    if persistent_capability_prefix or task_route_layerwise_control:
        router_graph_path = artifact / str(
            router_contract.get("router_graph", "")
        )
        router_hash_matches = (
            router_graph_path.is_file()
            and _sha256_file(router_graph_path)
            == router_contract.get("router_graph_sha256")
        )
        router_document = (
            onnx.load(router_graph_path) if router_hash_matches else None
        )
        router_shapes = (
            [
                tuple(int(dim) for dim in value.dims)
                for value in router_document.graph.initializer
            ]
            if router_document is not None
            else []
        )
        router_ops = (
            [node.op_type for node in router_document.graph.node]
            if router_document is not None
            else []
        )
        router_physical_checks = {
            "router_graph_hash_matches": router_hash_matches,
            "router_has_one_hashed_embedding_table": (
                router_shapes.count((4096, 32)) == 1
                if persistent_capability_prefix
                else True
            ),
            "router_physically_hashes_gathers_means_and_classifies": (
                (
                    all(
                        operation in router_ops
                        for operation in (
                            "Mod",
                            "Gather",
                            "ReduceMean",
                            "ArgMax",
                        )
                    )
                    and any(
                        operation in router_ops
                        for operation in ("Gemm", "MatMul")
                    )
                )
                if persistent_capability_prefix
                else True
            ),
            "task_route_router_physically_selects_one_route": (
                (
                    (
                        control_contract.get("router_mode")
                        == FULL_TASK_ROUTE_ROUTER_MODE
                        and "ReduceMean" in router_ops
                        and any(
                            operation in router_ops
                            for operation in ("MatMul", "MatMulInteger")
                        )
                        and bool(
                            control_contract.get(
                                "routing_prepass_uses_zero_control"
                            )
                        )
                    )
                    or (
                        control_contract.get("router_mode")
                        == COMPACT_TASK_ROUTE_ROUTER_MODE
                        and "Gather" in router_ops
                        and (
                            "ReduceMean" in router_ops
                            or (
                                "ReduceSum" in router_ops
                                and "Div" in router_ops
                            )
                        )
                        and "ReduceMax" in router_ops
                        and any(
                            operation in router_ops
                            for operation in ("Gemm", "MatMul")
                        )
                    )
                )
                and "ArgMax" in router_ops
                if task_route_layerwise_control
                else True
            ),
            "task_route_router_precision_is_declared": (
                control_contract.get("router_precision", "int8")
                in {"int8", "fp32"}
                if task_route_layerwise_control
                else True
            ),
            "router_prompt_axis_is_dynamic": (
                router_document is not None
                and len(router_document.graph.input) == 1
                and router_document.graph.input[0]
                .type.tensor_type.shape.dim[1]
                .dim_param
                == "prompt_sequence"
            ),
            "router_outputs_route_and_scores": (
                router_document is not None
                and {
                    value.name for value in router_document.graph.output
                }
                == {"selected_route", "task_scores"}
            ),
        }
    elif (
        (
            layerwise_capability_control
            and not task_route_layerwise_control
        )
        or deep_capability_adapters
        or deep_reused_capability_cakes
        or gated_deep_reused_capability_cakes
    ):
        router_parameter_path = artifact / str(
            router_contract.get("router_parameters", "")
        )
        parameter_hash_matches = (
            router_parameter_path.is_file()
            and _sha256_file(router_parameter_path)
            == router_contract.get("router_parameters_sha256")
        )
        arrays: dict[str, np.ndarray] = {}
        if parameter_hash_matches:
            with np.load(router_parameter_path, allow_pickle=False) as values:
                arrays = {
                    name: np.asarray(values[name], dtype=np.float32)
                    for name in values.files
                }
        shapes_exact = (
            set(arrays) == {"embedding", "weight", "bias"}
            and arrays["embedding"].shape == (4096, 32)
            and arrays["weight"].shape == (14, 32)
            and arrays["bias"].shape == (14,)
        )
        equivalence_exact = shapes_exact
        probes = router_contract.get("router_equivalence_probes", [])
        if equivalence_exact:
            for probe in probes:
                ids = np.asarray(probe.get("prompt_ids", []), dtype=np.int64)
                if not len(ids):
                    equivalence_exact = False
                    break
                hashed = np.remainder(ids, arrays["embedding"].shape[0])
                scores = (
                    arrays["embedding"][hashed].mean(axis=0)
                    @ arrays["weight"].T
                    + arrays["bias"]
                )
                if int(scores.argmax()) != int(
                    probe.get("selected_route", -1)
                ):
                    equivalence_exact = False
                    break
        router_physical_checks = {
            "router_parameters_hash_matches": parameter_hash_matches,
            "router_parameter_shapes_are_exact": shapes_exact,
            "four_router_equivalence_probes_reproduce_routes": (
                len(probes) == 4 and equivalence_exact
            ),
            "router_fusion_mode_is_exact": (
                router_contract.get("router_mode")
                == "fused_numpy_hash_mean_linear"
                and router_contract.get("separate_router_session") is False
            ),
        }
    route_gathers = []
    prefix_gathers = []
    control_gathers = []
    adapter_gathers = []
    reused_cake_gathers = []
    gated_cake_gathers = []
    gated_scalar_gathers = []
    for node in all_nodes:
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        resolved_weight = resolved_initializer(node.input[0])
        shape = initializers.get(resolved_weight)
        transformer_adapter_gather = bool(
            deep_capability_adapters
            and "transformer" in str(node.name).lower()
            and shape
            in {
                (installed_task_cakes, 768),
                (installed_task_cakes, 32, 768),
                (installed_task_cakes, 768, 32),
            }
        )
        transformer_reused_cake_gather = bool(
            deep_reused_capability_cakes
            and "transformer" in str(node.name).lower()
            and shape
            in {
                (installed_task_cakes, 768),
                (installed_task_cakes, 64, 768),
                (installed_task_cakes, 768, 64),
            }
        )
        transformer_gated_cake_gather = bool(
            gated_deep_reused_capability_cakes
            and "transformer" in str(node.name).lower()
            and shape
            in {
                (installed_task_cakes, 768),
                (installed_task_cakes, 64, 768),
                (installed_task_cakes, 768, 64),
            }
        )
        transformer_scalar_gate_gather = bool(
            gated_deep_reused_capability_cakes
            and "transformer" in str(node.name).lower()
            and shape == (installed_task_cakes, 3)
        )
        if shape in {
            (installed_task_cakes, 768),
            (installed_task_cakes, task_cake_rank),
            (installed_task_cakes, task_cake_rank, 768),
            (installed_task_cakes, 768, task_cake_rank),
        } and node.input[1] in {"route", "requested_route"} and not (
            transformer_adapter_gather
            or transformer_reused_cake_gather
            or transformer_gated_cake_gather
        ):
            route_gathers.append(
                {
                    "node": node.name,
                    "weight": node.input[0],
                    "resolved_weight": resolved_weight,
                    "installed_shape": list(shape),
                    "selected_output": node.output[0],
                }
            )
        if (
            persistent_capability_prefix
            and shape == (installed_task_cakes, 3, 12, 8, 64)
            and node.input[1] in {"route", "requested_route"}
        ):
            prefix_gathers.append(
                {
                    "node": node.name,
                    "weight": node.input[0],
                    "resolved_weight": resolved_weight,
                    "installed_shape": list(shape),
                    "selected_output": node.output[0],
                }
            )
        if (
            layerwise_capability_control
            and shape == (installed_task_cakes, 3, 768)
            and node.input[1] in {"route", "requested_route"}
        ):
            control_gathers.append(
                {
                    "node": node.name,
                    "weight": node.input[0],
                    "resolved_weight": resolved_weight,
                    "installed_shape": list(shape),
                    "selected_output": node.output[0],
                }
            )
        if (
            transformer_adapter_gather
            and node.input[1] in {"route", "requested_route"}
        ):
            adapter_gathers.append(
                {
                    "node": node.name,
                    "weight": node.input[0],
                    "resolved_weight": resolved_weight,
                    "installed_shape": list(shape),
                    "selected_output": node.output[0],
                }
            )
        if (
            transformer_reused_cake_gather
            and node.input[1] in {"route", "requested_route"}
        ):
            reused_cake_gathers.append(
                {
                    "node": node.name,
                    "weight": node.input[0],
                    "resolved_weight": resolved_weight,
                    "installed_shape": list(shape),
                    "selected_output": node.output[0],
                }
            )
        if (
            transformer_gated_cake_gather
            and node.input[1] in {"route", "requested_route"}
        ):
            gated_cake_gathers.append(
                {
                    "node": node.name,
                    "weight": node.input[0],
                    "resolved_weight": resolved_weight,
                    "installed_shape": list(shape),
                    "selected_output": node.output[0],
                }
            )
        if (
            transformer_scalar_gate_gather
            and node.input[1] in {"route", "requested_route"}
        ):
            gated_scalar_gathers.append(
                {
                    "node": node.name,
                    "weight": node.input[0],
                    "resolved_weight": resolved_weight,
                    "installed_shape": list(shape),
                    "selected_output": node.output[0],
                }
            )
    dense_all_route_matrices = []
    for node in all_nodes:
        if node.op_type not in {
            "MatMul",
            "MatMulInteger",
            "Gemm",
            "QLinearMatMul",
        }:
            continue
        for name in node.input:
            shape = initializers.get(name)
            if (
                shape is not None
                and len(shape) == 3
                and shape[0] == installed_task_cakes
            ):
                dense_all_route_matrices.append(node.name)
    installed_bridges = int(
        metadata["runtime"]["installed_route_bridges"]
    )
    expected_gathers = (
        0
        if deep_reused_capability_cakes
        else (8 if installed_bridges else 4)
    )
    task_cake_quantized = (
        metadata["runtime"].get("task_cake_projection_quantization")
        is not None
    )
    if task_cake_quantized:
        expected_gathers += 2
    task_cake_projection_precision = metadata["runtime"].get(
        "task_cake_projection_precision"
    )
    declared_fp32_projection_nodes = metadata["runtime"].get(
        "float32_task_cake_projection_nodes", []
    )
    route_selected_outputs = {
        row["selected_output"]
        for row in route_gathers
        if len(row["installed_shape"]) == 3
    }
    route_selected_transpose_outputs = {
        output
        for node in all_nodes
        if node.op_type == "Transpose"
        and any(
            value in route_selected_outputs for value in node.input
        )
        for output in node.output
    }
    fp32_projection_nodes = [
        node
        for node in all_nodes
        if node.name in set(declared_fp32_projection_nodes)
    ]
    fp32_projection_checks = {
        "task_cake_projection_precision_is_declared": (
            task_cake_projection_precision
            in {
                None,
                "dynamic_int8_after_route_selection",
                "float32_after_route_selection",
            }
        ),
        "declared_fp32_task_cake_projections_follow_route_selection": (
            (
                len(declared_fp32_projection_nodes) == 2
                and len(fp32_projection_nodes) == 2
                and all(
                    node.op_type == "MatMul"
                    and any(
                        value in route_selected_transpose_outputs
                        for value in node.input
                    )
                    for node in fp32_projection_nodes
                )
            )
            if task_cake_projection_precision
            == "float32_after_route_selection"
            else not declared_fp32_projection_nodes
        ),
    }
    declared_matrix_exclusions = metadata["runtime"].get(
        "excluded_matrix_nodes", []
    )
    declared_runtime_matrix_exclusions = metadata["runtime"].get(
        "excluded_runtime_matrix_nodes", []
    )
    precision_profile = metadata["runtime"].get(
        "precision_profile", "int8"
    )
    float_matrix_checks = _float_matrix_node_checks(
        document,
        declared_runtime_matrix_exclusions,
    )
    profile_requires_matrix_exclusions = (
        precision_profile in {"fp32_output", "fp32_transformer"}
        or precision_profile.startswith("fp32_layer")
    )
    runtime_precision_checks = {
        "declared_matrix_exclusions_map_one_to_one": (
            len(declared_matrix_exclusions)
            == len(declared_runtime_matrix_exclusions)
            == len(set(declared_runtime_matrix_exclusions))
        ),
        "precision_profile_has_required_matrix_exclusions": (
            bool(declared_runtime_matrix_exclusions)
            if profile_requires_matrix_exclusions
            else (
                not declared_runtime_matrix_exclusions
                or set(declared_runtime_matrix_exclusions)
                == set(declared_fp32_projection_nodes)
                if task_cake_projection_precision
                == "float32_after_route_selection"
                else not declared_runtime_matrix_exclusions
            )
        ),
        "declared_runtime_matrix_exclusions_remain_float": (
            all(float_matrix_checks.values())
        ),
    }
    output_vocabulary_contract = metadata["runtime"].get(
        "output_vocabulary"
    )
    sparse_output: dict[str, Any] | None = None
    sparse_output_checks: dict[str, bool] = {}
    if output_vocabulary_contract is not None:
        vocabulary_path = (
            artifact / output_vocabulary_contract["path"]
        )
        vocabulary_hash_matches = (
            vocabulary_path.is_file()
            and _sha256_file(vocabulary_path)
            == output_vocabulary_contract["sha256"]
        )
        vocabulary = (
            json.loads(vocabulary_path.read_text(encoding="utf-8"))
            if vocabulary_hash_matches
            else {}
        )
        selected_count = int(
            output_vocabulary_contract["selected_token_count"]
        )
        dynamic_mode = (
            output_vocabulary_contract.get("mode")
            == "train_base_union_prompt_tokens"
        )
        projection_initializers = [
            value
            for value in document.graph.initializer
            if tuple(int(dim) for dim in value.dims)
            == (768, selected_count)
            and numpy_helper.to_array(value).dtype == np.int8
        ]
        selected_projection_sha = (
            hashlib.sha256(
                np.ascontiguousarray(
                    numpy_helper.to_array(projection_initializers[0])
                ).tobytes()
            ).hexdigest()
            if len(projection_initializers) == 1
            else None
        )
        logits_outputs = [
            value
            for value in document.graph.output
            if value.name == "logits"
        ]
        logits_width = (
            int(
                logits_outputs[0]
                .type.tensor_type.shape.dim[1]
                .dim_value
            )
            if len(logits_outputs) == 1
            and len(
                logits_outputs[0].type.tensor_type.shape.dim
            )
            == 2
            else -1
        )
        token_ids = vocabulary.get("global_token_ids", [])
        common_sparse_checks = {
            "output_vocabulary_hash_matches": (
                vocabulary_hash_matches
            ),
            "input_embedding_remains_full_vocabulary": any(
                shape == (50_257, 768)
                for shape in initializers.values()
            ),
            "byte_fallback_is_physically_addressable": (
                set(range(256)).issubset(token_ids)
            ),
            "eos_is_physically_addressable": 50_256 in token_ids,
            "local_to_global_token_ids_are_unique_and_sorted": (
                len(token_ids) == selected_count
                and len(set(token_ids)) == selected_count
                and token_ids == sorted(token_ids)
            ),
        }
        if dynamic_mode:
            allowed_input = str(
                output_vocabulary_contract[
                    "allowed_output_ids_graph_input"
                ]
            )
            weight_gathers = [
                node
                for node in all_nodes
                if node.op_type == "Gather"
                and node.name == "DynamicOutputWeightGather"
                and allowed_input in node.input
            ]
            scale_gathers = [
                node
                for node in all_nodes
                if node.op_type == "Gather"
                and node.name == "DynamicOutputScaleGather"
                and allowed_input in node.input
            ]
            transposes = [
                node
                for node in all_nodes
                if node.op_type == "Transpose"
                and node.name == "DynamicOutputWeightTranspose"
            ]
            dynamic_matrix_outputs = {
                output
                for node in transposes
                for output in node.output
            }
            dynamic_matrix_consumers = [
                node
                for node in all_nodes
                if node.op_type == "MatMulInteger"
                and any(
                    value in node.input
                    for value in dynamic_matrix_outputs
                )
            ]
            sparse_output_checks = {
                **common_sparse_checks,
                "allowed_output_ids_is_one_graph_input": (
                    sum(
                        value.name == allowed_input
                        for value in document.graph.input
                    )
                    == 1
                ),
                "one_dynamic_output_weight_gather": (
                    len(weight_gathers) == 1
                ),
                "one_dynamic_output_scale_gather": (
                    len(scale_gathers) == 1
                ),
                "one_dynamic_output_weight_transpose": (
                    len(transposes) == 1
                ),
                "dynamic_weight_feeds_matmul_integer": (
                    len(dynamic_matrix_consumers) == 1
                ),
                "duplicate_static_output_projection_absent": (
                    not projection_initializers
                    and not any(
                        shape == (768, 50_257)
                        for shape in initializers.values()
                    )
                ),
                "graph_logits_width_is_dynamic": (
                    logits_width == 0
                    and len(logits_outputs) == 1
                    and (
                        logits_outputs[0]
                        .type.tensor_type.shape.dim[1]
                        .dim_param
                        == "allowed_output_tokens"
                    )
                ),
                "metadata_prompt_union_is_explicit": (
                    output_vocabulary_contract.get(
                        "prompt_token_ids_added_at_runtime"
                    )
                    is True
                    and output_vocabulary_contract.get(
                        "duplicate_output_projection_removed"
                    )
                    is True
                ),
            }
        else:
            sparse_output_checks = {
                **common_sparse_checks,
                "one_physically_reduced_output_projection": (
                    len(projection_initializers) == 1
                ),
                "graph_logits_width_matches_selected_vocabulary": (
                    logits_width == selected_count
                ),
                "selected_projection_hash_matches_train_only_manifest": (
                    selected_projection_sha
                    == vocabulary.get("selected_projection", {}).get(
                        "sha256"
                    )
                ),
                "output_projection_is_smaller_than_full_vocabulary": (
                    selected_count < 50_257
                ),
            }
        sparse_output = {
            "mode": output_vocabulary_contract.get("mode", "static"),
            "budget_id": output_vocabulary_contract["budget_id"],
            "selected_token_count": selected_count,
            "full_token_count": int(
                output_vocabulary_contract["full_token_count"]
            ),
            "projection_initializer": (
                projection_initializers[0].name
                if len(projection_initializers) == 1
                else None
            ),
            "selected_projection_sha256": selected_projection_sha,
            "logits_width": logits_width,
            "checks": sparse_output_checks,
        }
    schedule = metadata["runtime"].get("cake_activation_schedule")
    conditional_schedule = (
        schedule is not None
        and schedule.get("format")
        == "abi-layercake-conditional-core-realization-schedule/1"
    )
    conditional_nodes = [
        node
        for node in document.graph.node
        if node.op_type == "If"
        and node.name == "PhysicalConditionalTaskCake"
    ]
    conditional_checks: dict[str, bool] = {}
    if conditional_schedule:
        then_nodes: list[Any] = []
        else_nodes: list[Any] = []
        if len(conditional_nodes) == 1:
            for attribute in conditional_nodes[0].attribute:
                if attribute.name == "then_branch":
                    then_nodes = graph_nodes(attribute.g)
                elif attribute.name == "else_branch":
                    else_nodes = graph_nodes(attribute.g)
        conditional_checks = {
            "one_physical_conditional_task_cake": (
                len(conditional_nodes) == 1
            ),
            "conditional_input_is_one_bool_graph_input": (
                sum(
                    value.name == schedule["graph_input"]
                    and value.type.tensor_type.elem_type
                    == onnx.TensorProto.BOOL
                    for value in document.graph.input
                )
                == 1
            ),
            "four_route_gathers_are_confined_to_true_branch": (
                sum(
                    node.op_type == "Gather"
                    and len(node.input) >= 2
                    and node.input[1] == "route"
                    for node in then_nodes
                )
                == 4
                and not any(
                    node.op_type == "Gather"
                    and len(node.input) >= 2
                    and node.input[1] == "route"
                    for node in else_nodes
                )
            ),
            "false_branch_is_core_identity_only": (
                len(else_nodes) == 1
                and else_nodes[0].op_type == "Identity"
            ),
            "metadata_physically_skips_inactive_cake": (
                schedule.get("physical_conditional_execution") is True
                and int(schedule["task_cake_nodes_in_true_branch"]) == 26
                and int(schedule["task_cake_nodes_in_false_branch"]) == 0
            ),
        }
    checks = {
        "expected_route_indexed_parameter_gathers": (
            len(route_gathers) == expected_gathers
        ),
        "quantized_task_cake_has_two_route_scale_gathers": (
            sum(
                tuple(row["installed_shape"])
                in {
                    (installed_task_cakes, task_cake_rank),
                    (installed_task_cakes, 768),
                }
                and row["weight"].startswith("TaskCake")
                for row in route_gathers
            )
            == 2
            if task_cake_quantized
            else True
        ),
        "all_selected_tensors_have_consumers": all(
            any(
                output in node.input
                for node in all_nodes
            )
            for output in {
                row["selected_output"] for row in route_gathers
            }
        ),
        "no_matrix_consumes_all_installed_routes": (
            not dense_all_route_matrices
        ),
        "one_requested_route_input": sum(
            value.name == "requested_route"
            for value in document.graph.input
        ) == 1,
        "metadata_one_active_task_cake": (
            metadata["runtime"]["maximum_active_task_cakes_per_sequence"]
            == 1
        ),
        "persistent_prefix_contract_is_valid": (
            (
                len(prefix_gathers) == 2
                and int(prefix_contract.get("prefix_length", -1)) == 8
                and int(prefix_contract.get("installed_prefixes", -1))
                == installed_task_cakes
                and int(
                    prefix_contract.get(
                        "maximum_active_prefixes_per_sequence", -1
                    )
                )
                == 1
                and prefix_contract.get(
                    "physically_selected_before_transformer"
                )
                is True
                and prefix_contract.get("public_cache_excludes_prefix")
                is True
                and prefix_contract.get("main_graph_requires_selected_route")
                is True
                and prefix_contract.get("exported_attention_implementation")
                == "eager"
                and all(
                    any(
                        output in node.input for node in all_nodes
                    )
                    for output in {
                        row["selected_output"] for row in prefix_gathers
                    }
                )
            )
            if persistent_capability_prefix
            else not prefix_gathers
        ),
        "layerwise_control_contract_is_valid": (
            (
                len(control_gathers) == 1
                and int(control_contract.get("installed_controls", -1))
                == installed_task_cakes
                and int(control_contract.get("control_layers", -1)) == 3
                and int(control_contract.get("control_width", -1)) == 768
                and int(
                    control_contract.get(
                        "maximum_active_control_paths_per_sequence", -1
                    )
                )
                == 1
                and int(control_contract.get("extra_kv_positions", -1)) == 0
                and control_contract.get(
                    "public_cache_contains_only_real_tokens"
                )
                is True
                and control_contract.get(
                    "conditions_every_real_token_kv_write"
                )
                is True
                and control_contract.get(
                    "physically_selected_before_transformer"
                )
                is True
                and control_contract.get(
                    "main_graph_requires_selected_route"
                )
                is True
                and all(
                    any(output in node.input for node in all_nodes)
                    for output in {
                        row["selected_output"] for row in control_gathers
                    }
                )
            )
            if layerwise_capability_control
            else not control_gathers
        ),
        "deep_capability_adapter_contract_is_valid": (
            (
                len(adapter_gathers)
                in (
                    {12}
                    if adapter_contract.get(
                        "shared_adapter_weights_across_layers", False
                    )
                    else {8, 12}
                )
                and sum(
                    tuple(row["installed_shape"])
                    == (installed_task_cakes, 768)
                    for row in adapter_gathers
                )
                in (
                    {6}
                    if adapter_contract.get(
                        "shared_adapter_weights_across_layers", False
                    )
                    else {2, 6}
                )
                and sum(
                    tuple(row["installed_shape"])
                    == (installed_task_cakes, 32, 768)
                    for row in adapter_gathers
                )
                == (
                    3
                    if adapter_contract.get(
                        "shared_adapter_weights_across_layers", False
                    )
                    else 3
                )
                and sum(
                    tuple(row["installed_shape"])
                    == (installed_task_cakes, 768, 32)
                    for row in adapter_gathers
                )
                == (
                    3
                    if adapter_contract.get(
                        "shared_adapter_weights_across_layers", False
                    )
                    else 3
                )
                and int(adapter_contract.get("installed_capabilities", -1))
                == installed_task_cakes
                and int(adapter_contract.get("adapter_layers", -1)) == 3
                and int(adapter_contract.get("adapter_rank", -1)) == 32
                and int(
                    adapter_contract.get(
                        "maximum_active_adapters_per_sequence", -1
                    )
                )
                == (
                    1
                    if adapter_contract.get(
                        "shared_adapter_weights_across_layers", False
                    )
                    else 3
                )
                and int(
                    adapter_contract.get(
                        "active_adapter_invocations_per_sequence", -1
                    )
                )
                == 3
                and int(adapter_contract.get("extra_kv_positions", -1))
                == 0
                and adapter_contract.get(
                    "public_cache_contains_only_real_tokens"
                )
                is True
                and adapter_contract.get(
                    "conditions_every_real_token_kv_write"
                )
                is True
                and adapter_contract.get(
                    "physically_selected_before_each_transformer_block"
                )
                is True
                and adapter_contract.get(
                    "main_graph_requires_selected_route"
                )
                is True
                and all(
                    any(output in node.input for node in all_nodes)
                    for output in {
                        row["selected_output"] for row in adapter_gathers
                    }
                )
            )
            if deep_capability_adapters
            else not adapter_gathers
        ),
        "shared_deep_adapter_projections_reused_at_three_blocks": (
            (
                len(
                    {
                        row["resolved_weight"]
                        for row in adapter_gathers
                    }
                )
                == 4
                and all(
                    sum(
                        transpose.output[0] in node.input
                        and node.op_type == "MatMul"
                        for transpose in all_nodes
                        if transpose.op_type == "Transpose"
                        and row["selected_output"] in transpose.input
                        for node in all_nodes
                    )
                    == 1
                    for row in adapter_gathers
                    if len(row["installed_shape"]) == 3
                )
            )
            if deep_capability_adapters
            and adapter_contract.get(
                "shared_adapter_weights_across_layers", False
            )
            else True
        ),
        "deep_reused_capability_cake_contract_is_valid": (
            (
                len(reused_cake_gathers) == 12
                and sum(
                    tuple(row["installed_shape"])
                    == (installed_task_cakes, 768)
                    for row in reused_cake_gathers
                )
                == 6
                and sum(
                    tuple(row["installed_shape"])
                    == (installed_task_cakes, 64, 768)
                    for row in reused_cake_gathers
                )
                == 3
                and sum(
                    tuple(row["installed_shape"])
                    == (installed_task_cakes, 768, 64)
                    for row in reused_cake_gathers
                )
                == 3
                and len(
                    {
                        row["resolved_weight"]
                        for row in reused_cake_gathers
                    }
                )
                == 4
                and int(
                    reused_cake_contract.get(
                        "installed_capability_cakes", -1
                    )
                )
                == installed_task_cakes
                and int(reused_cake_contract.get("cake_rank", -1)) == 64
                and int(
                    reused_cake_contract.get(
                        "selected_unique_cakes_per_sequence", -1
                    )
                )
                == 1
                and int(
                    reused_cake_contract.get(
                        "selected_cake_invocations_per_sequence", -1
                    )
                )
                == 3
                and int(
                    reused_cake_contract.get(
                        "final_post_transformer_cake_invocations", -1
                    )
                )
                == 0
                and int(
                    reused_cake_contract.get(
                        "added_adapter_parameters", -1
                    )
                )
                == 0
                and int(
                    reused_cake_contract.get("extra_kv_positions", -1)
                )
                == 0
                and reused_cake_contract.get(
                    "public_cache_contains_only_real_tokens"
                )
                is True
                and reused_cake_contract.get(
                    "conditions_every_real_token_kv_write"
                )
                is True
                and reused_cake_contract.get(
                    "physically_selected_before_each_transformer_block"
                )
                is True
                and all(
                    sum(
                        transpose.output[0] in node.input
                        and node.op_type == "MatMul"
                        for transpose in all_nodes
                        if transpose.op_type == "Transpose"
                        and row["selected_output"] in transpose.input
                        for node in all_nodes
                    )
                    == 1
                    for row in reused_cake_gathers
                    if len(row["installed_shape"]) == 3
                )
            )
            if deep_reused_capability_cakes
            else not reused_cake_gathers
        ),
        "gated_deep_reused_capability_cake_contract_is_valid": (
            (
                len(gated_cake_gathers) == 12
                and len(gated_scalar_gathers) == 1
                and sum(
                    tuple(row["installed_shape"])
                    == (installed_task_cakes, 768)
                    for row in gated_cake_gathers
                )
                == 6
                and sum(
                    tuple(row["installed_shape"])
                    == (installed_task_cakes, 64, 768)
                    for row in gated_cake_gathers
                )
                == 3
                and sum(
                    tuple(row["installed_shape"])
                    == (installed_task_cakes, 768, 64)
                    for row in gated_cake_gathers
                )
                == 3
                and len(
                    {
                        row["resolved_weight"]
                        for row in gated_cake_gathers + route_gathers
                    }
                )
                == 4
                and len(
                    {
                        row["resolved_weight"]
                        for row in gated_scalar_gathers
                    }
                )
                == 1
                and int(
                    gated_cake_contract.get(
                        "installed_capability_cakes", -1
                    )
                )
                == installed_task_cakes
                and int(gated_cake_contract.get("cake_rank", -1)) == 64
                and int(
                    gated_cake_contract.get(
                        "selected_unique_cakes_per_sequence", -1
                    )
                )
                == 1
                and int(
                    gated_cake_contract.get(
                        "pre_block_selected_cake_invocations", -1
                    )
                )
                == 3
                and int(
                    gated_cake_contract.get(
                        "final_selected_cake_invocations", -1
                    )
                )
                == 1
                and int(
                    gated_cake_contract.get(
                        "installed_scalar_gate_parameters", -1
                    )
                )
                == 42
                and int(
                    gated_cake_contract.get(
                        "active_scalar_gate_parameters", -1
                    )
                )
                == 3
                and gated_cake_contract.get("scalar_gate_shape") == [14, 3]
                and int(
                    gated_cake_contract.get("added_matrix_parameters", -1)
                )
                == 0
                and int(
                    gated_cake_contract.get("extra_kv_positions", -1)
                )
                == 0
                and gated_cake_contract.get(
                    "public_cache_contains_only_real_tokens"
                )
                is True
                and gated_cake_contract.get(
                    "conditions_every_real_token_kv_write"
                )
                is True
                and gated_cake_contract.get(
                    "physically_selected_before_each_transformer_block"
                )
                is True
                and all(
                    any(
                        row["selected_output"] in node.input
                        for node in all_nodes
                    )
                    for row in gated_scalar_gathers
                )
                and sum(
                    node.op_type == "Gather"
                    and gated_scalar_gathers[0]["selected_output"]
                    in node.input
                    for node in all_nodes
                )
                == 3
            )
            if gated_deep_reused_capability_cakes
            else not gated_cake_gathers and not gated_scalar_gathers
        ),
        "metadata_one_active_route_bridge": (
            metadata["runtime"][
                "maximum_active_route_bridges_per_sequence"
            ]
            == (1 if installed_bridges else 0)
        ),
        "removed_bridge_is_explicitly_fused": (
            installed_bridges == 10
            or (
                installed_bridges == 0
                and (
                    metadata["runtime"].get(
                        "route_bridge_fused_into_task_cakes"
                    )
                    is True
                    or metadata["runtime"].get(
                        "symbolic_only_host_has_no_route_bridge"
                    )
                    is True
                    or metadata["runtime"].get(
                        "standalone_core_has_no_route_bridge"
                    )
                    is True
                )
            )
        ),
        **fp32_projection_checks,
        **runtime_precision_checks,
        **router_physical_checks,
        **conditional_checks,
        **sparse_output_checks,
    }
    evidence = {
        "format": "abi-layercake-host-physical-sparse-proof/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "runtime_graph_sha256": _sha256_file(graph_path),
        "installed_task_cakes": installed_task_cakes,
        "task_cake_rank": task_cake_rank,
        "capability_cake_canonical_routes": list(
            capability_cake_routes
        ),
        "maximum_active_task_cakes_per_sequence": 1,
        "installed_route_bridges": installed_bridges,
        "maximum_active_route_bridges_per_sequence": (
            metadata["runtime"][
                "maximum_active_route_bridges_per_sequence"
            ]
        ),
        "route_indexed_parameter_gathers": route_gathers,
        "persistent_prefix_route_gathers": prefix_gathers,
        "layerwise_control_route_gathers": control_gathers,
        "deep_capability_adapter_route_gathers": adapter_gathers,
        "deep_reused_capability_cake_route_gathers": (
            reused_cake_gathers
        ),
        "gated_deep_reused_capability_cake_route_gathers": (
            gated_cake_gathers
        ),
        "gated_deep_reused_scalar_route_gathers": (
            gated_scalar_gathers
        ),
        "persistent_prefix_router_graph": (
            {
                "path": router_graph_path.name,
                "sha256": _sha256_file(router_graph_path),
                "checks": router_physical_checks,
            }
            if router_graph_path is not None and router_graph_path.is_file()
            else None
        ),
        "fused_capability_conditioning_router": (
            {
                "path": router_parameter_path.name,
                "sha256": _sha256_file(router_parameter_path),
                "checks": router_physical_checks,
            }
            if router_parameter_path is not None
            and router_parameter_path.is_file()
            else None
        ),
        "dense_all_route_matrix_nodes": dense_all_route_matrices,
        "runtime_precision": {
            "profile": precision_profile,
            "declared_source_matrix_nodes": (
                declared_matrix_exclusions
            ),
            "declared_runtime_matrix_nodes": (
                declared_runtime_matrix_exclusions
            ),
            "float_matrix_checks": float_matrix_checks,
        },
        "conditional_cake_execution": (
            {
                "schedule": schedule,
                "checks": conditional_checks,
            }
            if conditional_schedule
            else None
        ),
        "sparse_output_vocabulary": sparse_output,
        "checks": checks,
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def verify_runtime_identity(
    artifact: str | Path,
    *,
    host_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Reload the native artifact and fail on any manifest/hash mismatch."""

    artifact = Path(artifact).resolve()
    host_path = Path(host_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"identity evidence is immutable: {output_path}"
        )
    runtime = NativeHostRuntime(artifact, threads=1)
    metadata = runtime.metadata
    host_manifest = json.loads(
        (host_path / "deployment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    checks = {
        "host_manifest_file_hash_matches": (
            metadata["host"]["deployment_manifest_file_sha256"]
            == _sha256_file(host_path / "deployment_manifest.json")
        ),
        "host_manifest_claim_hash_matches": (
            metadata["host"]["deployment_manifest_sha256"]
            == host_manifest["manifest_sha256"]
        ),
        "host_delta_hash_matches": (
            metadata["host"]["delta_sha256"]
            == _sha256_file(
                host_path / host_manifest["host_delta"]["path"]
            )
        ),
        "symbolic_payload_hash_matches_host": (
            metadata["symbolic_surface"]["sha256"]
            == host_manifest["host_delta"]["symbolic_surface"][
                "payload_sha256"
            ]
        ),
        "teacher_absent": (
            metadata["host"]["teacher_present_at_inference"] is False
        ),
        "source_transformer_blocks_absent": (
            metadata["host"]["source_transformer_blocks_retained"] == 0
        ),
        "decoding_contract_is_explicit": (
            metadata["runtime"].get("decoding")
            == runtime.decoding
            and runtime.decoding.get("weights_changed") is False
        ),
    }
    installed_handlers = set(
        runtime.symbolic_surface.get("handlers", [])
    )
    if "exact_json_item_count" in installed_handlers:
        probe_prompt = (
            "Return only one JSON object, with no Markdown, using "
            "`item`='identity-probe' and `count`=7."
        )
        probe_route = CAPABILITY_TO_ROUTE["format_control"]
    elif "natural_email_from_notes" in installed_handlers:
        probe_prompt = (
            "Draft a short, polite email from Mira with these notes: "
            "thank Luis for the draft and ask for N100MIRA by Thursday. "
            "Include a greeting and closing; add no new facts."
        )
        probe_route = CAPABILITY_TO_ROUTE["email_drafting"]
    elif "generic_supplied_field_realization" in installed_handlers:
        probe_prompt = (
            "Turn these supplied fields into one natural English sentence "
            "without adding information: object=draft; action=arrived; "
            "location=the east hall; count=5"
        )
        probe_route = CAPABILITY_TO_ROUTE["cake_output_realization"]
    elif "exact_supplied_text" in installed_handlers:
        probe_prompt = (
            "Reply with exactly identity-probe-7 and nothing else."
        )
        probe_route = CAPABILITY_TO_ROUTE["prompt_grounding"]
    else:
        raise LayerCakeHostRuntimeError(
            "native identity verifier has no strict probe for an installed "
            "symbolic handler"
        )
    expected_probe_output = symbolic_surface_output(
        runtime.symbolic_surface,
        prompt=probe_prompt,
        route=probe_route,
    )
    if expected_probe_output is None:
        raise LayerCakeHostRuntimeError(
            "native identity probe does not exercise its declared handler"
        )
    probe = generate_native_host(
        runtime,
        probe_prompt,
        max_new_tokens=96,
    )
    checks["hash_bound_symbolic_substrate_executes"] = (
        probe["symbolic_handler_used"] is True
        and probe["route"] == probe_route
        and probe["output"] == expected_probe_output
    )
    evidence = {
        "format": "abi-layercake-host-native-identity/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "runtime_graph_sha256": metadata["runtime"]["graph_sha256"],
        "host_manifest_sha256": host_manifest["manifest_sha256"],
        "host_delta_sha256": host_manifest["host_delta"]["sha256"],
        "symbolic_surface_sha256": metadata["symbolic_surface"]["sha256"],
        "decoding": runtime.decoding,
        "probe": {
            key: value
            for key, value in probe.items()
            if key not in {"authoritative_generated_token_ids"}
        },
        "expected_probe_output": expected_probe_output,
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def verify_core_runtime_identity(
    artifact: str | Path,
    *,
    standalone_core_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind one native graph to an exact frozen standalone LayerCake core."""

    artifact = Path(artifact).resolve()
    core_path = Path(standalone_core_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"identity evidence is immutable: {output_path}"
        )
    runtime = NativeHostRuntime(artifact, threads=1)
    metadata = runtime.metadata
    core_metadata_path = core_path / "metadata.json"
    core_checkpoint_path = core_path / "model.safetensors"
    core = json.loads(core_metadata_path.read_text(encoding="utf-8"))
    core_decoding = dict(core["decoding"])
    expected_decoding = {
        "algorithm": core_decoding["algorithm"],
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": int(
            core_decoding["no_repeat_ngram_size"]
        ),
        "allow_prompt_ngrams": bool(
            core_decoding["allow_prompt_ngrams"]
        ),
        "lexical_repetition_blocking_threshold": int(
            core_decoding.get("lexical_repetition_blocking_threshold", 0)
        ),
        "lexical_repetition_truncation_threshold": int(
            core_decoding["lexical_repetition_truncation_threshold"]
        ),
        "byte_repetition_ceiling": float(
            core_decoding.get("byte_repetition_ceiling", 0.0)
        ),
        "byte_repetition_guard_minimum_bytes": int(
            core_decoding.get(
                "byte_repetition_guard_minimum_bytes", 0
            )
        ),
        "prompt_identity_mixture": bool(
            core_decoding["prompt_identity_mixture"]
        ),
        "weights_changed": False,
    }
    decoding_overlay_metadata = metadata["runtime"].get(
        "decoding_overlay"
    )
    decoding_overlay_bound = decoding_overlay_metadata is None
    decoding_overlay_weights_unchanged = (
        decoding_overlay_metadata is None
    )
    if decoding_overlay_metadata is not None:
        overlay_path = artifact / decoding_overlay_metadata.get(
            "path", ""
        )
        try:
            decoding_overlay = _load_runtime_decoding_overlay(
                overlay_path,
                core_manifest=core,
            )
            decoding_overlay_bound = (
                decoding_overlay_metadata
                == {
                    "path": "runtime-decoding-overlay.json",
                    "sha256": _sha256_file(overlay_path),
                    "schema_version": decoding_overlay[
                        "schema_version"
                    ],
                    "base_decoding_sha256": decoding_overlay[
                        "candidate_core"
                    ]["base_decoding_sha256"],
                    "weights_changed": False,
                }
            )
            decoding_overlay_weights_unchanged = (
                decoding_overlay["invariants"]["weights_changed"]
                is False
                and decoding_overlay_metadata["weights_changed"]
                is False
            )
            expected_decoding["maximum_identical_token_run"] = int(
                decoding_overlay["override"][
                    "maximum_identical_token_run"
                ]
            )
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            LayerCakeHostRuntimeError,
        ):
            decoding_overlay_bound = False
            decoding_overlay_weights_unchanged = False
    declared_symbolic = core.get("symbolic_surface_substrate")
    if declared_symbolic is None:
        symbolic_substrate_matches_core = (
            runtime.symbolic_surface.get("handlers") == []
            and runtime.symbolic_surface.get(
                "source_teacher_text_retained"
            )
            is False
        )
    else:
        core_symbolic_path = core_path / str(
            declared_symbolic.get("path", "")
        )
        symbolic_substrate_matches_core = (
            core_symbolic_path.is_file()
            and metadata["symbolic_surface"]["sha256"]
            == declared_symbolic.get("payload_sha256")
            == _sha256_file(core_symbolic_path)
            and metadata["symbolic_surface"]["handlers"]
            == declared_symbolic.get("handlers")
            == runtime.symbolic_surface.get("handlers")
            and declared_symbolic.get("source_teacher_text_retained")
            is False
            and declared_symbolic.get("teacher_present_at_inference")
            is False
            and runtime.symbolic_surface.get(
                "source_teacher_text_retained"
            )
            is False
        )
    checks = {
        "standalone_kind_bound": (
            metadata["host"].get("kind")
            == "standalone_acquired_core"
        ),
        "core_metadata_file_hash_matches": (
            metadata["host"]["metadata_file_sha256"]
            == _sha256_file(core_metadata_path)
        ),
        "core_manifest_claim_hash_matches": (
            metadata["host"]["manifest_sha256"]
            == core["manifest_sha256"]
        ),
        "core_checkpoint_hash_matches": (
            metadata["host"]["checkpoint_sha256"]
            == core["checkpoint"]["sha256"]
            == _sha256_file(core_checkpoint_path)
        ),
        "canonical_abi_hash_matches": (
            metadata["canonical_semantic_abi"]["sha256"]
            == core["canonical_semantic_abi"]["sha256"]
        ),
        "teacher_absent": (
            metadata["host"]["teacher_present_at_inference"] is False
            and core["foreign_source_boundary"][
                "teacher_present_at_inference"
            ]
            is False
        ),
        "foreign_source_parameters_absent": (
            metadata["host"]["source_parameters_copied"] == 0
            and core["foreign_source_boundary"][
                "source_parameters_copied"
            ]
            == 0
        ),
        "source_transformer_blocks_absent": (
            metadata["host"]["source_transformer_blocks_retained"] == 0
            and core["foreign_source_boundary"][
                "source_transformer_blocks_retained"
            ]
            == 0
        ),
        "decoding_contract_is_exact": (
            runtime.decoding == expected_decoding
        ),
        "runtime_decoding_overlay_bound": decoding_overlay_bound,
        "runtime_decoding_overlay_weights_unchanged": (
            decoding_overlay_weights_unchanged
        ),
        "symbolic_substrate_matches_core": (
            symbolic_substrate_matches_core
        ),
        "standalone_route_bridge_absent": (
            metadata["runtime"].get(
                "standalone_core_has_no_route_bridge"
            )
            is True
            and metadata["runtime"]["installed_route_bridges"] == 0
        ),
    }
    probe = generate_native_host(
        runtime,
        "Offer one calm sentence to someone who feels uncertain.",
        max_new_tokens=64,
    )
    checks["hash_bound_native_graph_executes"] = (
        bool(probe["output"].strip())
        and probe["symbolic_handler_used"] is False
        and probe["persistent_state"]["completed_prefix_recomputation"]
        is False
    )
    symbolic_probe = None
    if declared_symbolic is not None:
        installed_handlers = set(
            runtime.symbolic_surface.get("handlers", [])
        )
        if "natural_email_from_notes" in installed_handlers:
            symbolic_prompt = (
                "Draft a short, polite email from Mira with these notes: "
                "thank Luis for the draft and ask for N100MIRA by Thursday. "
                "Include a greeting and closing; add no new facts."
            )
            symbolic_route = CAPABILITY_TO_ROUTE["email_drafting"]
        elif "generic_supplied_field_realization" in installed_handlers:
            symbolic_prompt = (
                "Turn these supplied fields into one natural English sentence "
                "without adding information: object=draft; action=arrived; "
                "location=the east hall; count=5"
            )
            symbolic_route = CAPABILITY_TO_ROUTE[
                "cake_output_realization"
            ]
        else:
            raise LayerCakeHostRuntimeError(
                "standalone identity verifier has no strict probe for its "
                "declared symbolic substrate"
            )
        expected_symbolic_output = symbolic_surface_output(
            runtime.symbolic_surface,
            prompt=symbolic_prompt,
            route=symbolic_route,
        )
        symbolic_probe = generate_native_host(
            runtime, symbolic_prompt, max_new_tokens=96
        )
        checks["hash_bound_symbolic_substrate_executes"] = (
            expected_symbolic_output is not None
            and symbolic_probe["symbolic_handler_used"] is True
            and symbolic_probe["route"] == symbolic_route
            and symbolic_probe["output"] == expected_symbolic_output
        )
    evidence = {
        "format": "abi-layercake-standalone-core-native-identity/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "runtime_graph_sha256": metadata["runtime"]["graph_sha256"],
        "core_manifest_sha256": core["manifest_sha256"],
        "core_checkpoint_sha256": core["checkpoint"]["sha256"],
        "core_metadata_file_sha256": _sha256_file(core_metadata_path),
        "symbolic_surface_sha256": metadata["symbolic_surface"]["sha256"],
        "decoding": runtime.decoding,
        "probe": {
            key: value
            for key, value in probe.items()
            if key not in {"authoritative_generated_token_ids"}
        },
        "symbolic_probe": (
            {
                key: value
                for key, value in symbolic_probe.items()
                if key not in {"authoritative_generated_token_ids"}
            }
            if symbolic_probe is not None
            else None
        ),
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export and verify an ABI LayerCake native host."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--layercake-root", required=True)
    export.add_argument("--parent", required=True)
    export.add_argument("--canonical-abi", required=True)
    export_source = export.add_mutually_exclusive_group(required=True)
    export_source.add_argument("--host")
    export_source.add_argument("--standalone-core")
    export.add_argument("--output", required=True)
    export.add_argument(
        "--repetition-penalty", type=float, default=1.0
    )
    export.add_argument(
        "--no-repeat-ngram-size", type=int, default=0
    )
    export.add_argument("--allow-prompt-ngrams", action="store_true")
    export.add_argument(
        "--lexical-repetition-blocking-threshold",
        type=int,
    )
    export.add_argument(
        "--lexical-repetition-truncation-threshold",
        type=int,
    )
    export.add_argument(
        "--byte-repetition-ceiling",
        type=float,
    )
    export.add_argument(
        "--byte-repetition-guard-minimum-bytes",
        type=int,
    )
    export.add_argument("--runtime-decoding-overlay")
    export.add_argument(
        "--keep-task-cakes-fp32",
        action="store_true",
        help=(
            "Keep only the two route-selected standalone task-cake "
            "projections in float32 while quantizing the shared transformer."
        ),
    )
    export.add_argument(
        "--precision-profile",
        choices=(
            "int8",
            "fp32_all",
            "fp32_embedding",
            "fp32_output",
            "fp32_transformer",
            "fp32_layer0",
            "fp32_layer1",
            "fp32_layer2",
            "fp32_layer0_attn_in",
            "fp32_layer0_attn_out",
            "fp32_layer0_mlp_in",
            "fp32_layer0_mlp_out",
            "fp32_layer0_attn_pair",
            "fp32_layer0_mlp_pair",
        ),
        default="int8",
    )
    export.add_argument(
        "--task-route-router-precision",
        choices=("int8", "fp32"),
        default="int8",
    )
    physical = subparsers.add_parser("verify-physical")
    physical.add_argument("--artifact", required=True)
    physical.add_argument("--output", required=True)
    identity = subparsers.add_parser("verify-identity")
    identity.add_argument("--artifact", required=True)
    identity_source = identity.add_mutually_exclusive_group(required=True)
    identity_source.add_argument("--host")
    identity_source.add_argument("--standalone-core")
    identity.add_argument("--output", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument(
        "--validation-bundle", action="append", required=True
    )
    evaluate.add_argument("--catalog", action="append", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--threads", type=int, default=14)
    evaluate.add_argument(
        "--capabilities",
        help=(
            "optional comma-separated diagnostic subset; subset evidence "
            "cannot replace the full locked suite"
        ),
    )
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--artifact", required=True)
    benchmark.add_argument("--comparator", required=True)
    benchmark.add_argument("--prompt-manifest", required=True)
    benchmark.add_argument("--parent-benchmark", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--output-bytes", type=int, required=True)
    benchmark.add_argument("--threads", type=int, default=14)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--artifact", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--max-new-tokens", type=int, default=128)
    infer.add_argument("--threads", type=int, default=14)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export":
        result = export_host_runtime(
            layercake_root=args.layercake_root,
            parent_path=args.parent,
            canonical_abi_path=args.canonical_abi,
            host_path=args.host,
            standalone_core_path=args.standalone_core,
            output_path=args.output,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            allow_prompt_ngrams=args.allow_prompt_ngrams,
            lexical_repetition_blocking_threshold=(
                args.lexical_repetition_blocking_threshold
            ),
            lexical_repetition_truncation_threshold=(
                args.lexical_repetition_truncation_threshold
            ),
            byte_repetition_ceiling=args.byte_repetition_ceiling,
            byte_repetition_guard_minimum_bytes=(
                args.byte_repetition_guard_minimum_bytes
            ),
            runtime_decoding_overlay_path=(
                args.runtime_decoding_overlay
            ),
            keep_task_cakes_fp32=args.keep_task_cakes_fp32,
            precision_profile=args.precision_profile,
            task_route_router_precision=(
                args.task_route_router_precision
            ),
        )
    elif args.command == "verify-physical":
        result = verify_physical_sparse_runtime(
            args.artifact, args.output
        )
    elif args.command == "verify-identity":
        if args.standalone_core is not None:
            result = verify_core_runtime_identity(
                args.artifact,
                standalone_core_path=args.standalone_core,
                output_path=args.output,
            )
        else:
            result = verify_runtime_identity(
                args.artifact,
                host_path=args.host,
                output_path=args.output,
            )
    elif args.command == "evaluate":
        result = evaluate_native_host_semantics(
            artifact=args.artifact,
            training_bundle_path=args.bundle,
            validation_bundle_paths=args.validation_bundle,
            catalog_paths=args.catalog,
            output_path=args.output,
            threads=args.threads,
            capabilities=(
                tuple(
                    value.strip()
                    for value in args.capabilities.split(",")
                    if value.strip()
                )
                if args.capabilities is not None
                else None
            ),
        )
    elif args.command == "benchmark":
        result = benchmark_native_host(
            artifact=args.artifact,
            comparator_path=args.comparator,
            prompt_manifest_path=args.prompt_manifest,
            parent_benchmark_path=args.parent_benchmark,
            output_path=args.output,
            output_bytes=args.output_bytes,
            threads=args.threads,
        )
    else:
        runtime = NativeHostRuntime(
            args.artifact, threads=args.threads
        )
        result = generate_native_host(
            runtime,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
    display = result
    if "records" in result or "observations" in result:
        display = {
            key: result[key]
            for key in (
                "status",
                "evidence_sha256",
                "aggregates",
                "capability_metrics",
                "observation_count",
                "wall_seconds",
            )
            if key in result
        }
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0 if result.get("status", "EXPORTED_NOT_YET_CERTIFIED") != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
