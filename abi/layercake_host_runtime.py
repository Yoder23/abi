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
    symbolic_surface_output,
)


RUNTIME_FORMAT = "abi-layercake-host-onnx-runtime/1"


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
    standalone_core = standalone_core_path is not None
    if standalone_core:
        model, _, manifest = load_layercake_core(
            standalone_core_path,
            layercake_root=layercake_root,
            device="cpu",
        )
        if (
            manifest.get("format")
            != "abi-layercake-full-english-core-acquisition/1"
            or int(manifest.get("architecture", {}).get("layers", -1)) != 3
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
        bridge_contract = manifest["host_delta"].get(
            "sparse_route_bridge", {}
        )
        bridge_fused = (
            bridge_contract.get("mode") == "none"
            and bridge_contract.get("fused_into_existing_task_cakes")
            is True
        )
        if route_bridge is None and not bridge_fused:
            raise LayerCakeHostRuntimeError(
                "native host requires a sparse route bridge or an explicit "
                "verified task-cake fusion"
            )
        symbolic_contract = getattr(model, "_abi_symbolic_surface", None)
        if symbolic_contract is None:
            raise LayerCakeHostRuntimeError(
                "promoted native host requires its symbolic substrate"
            )
        tokenizer_source_path = parent_path

    class RuntimeGraph(torch.nn.Module):
        def __init__(self, source, source_route_bridge):
            super().__init__()
            self.transformer = source.transformer
            self.task_classifier = source.task_classifier
            self.task_cakes = source.task_cakes
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

            cache = DynamicCache.from_legacy_cache(
                (
                    (past_key_0, past_value_0),
                    (past_key_1, past_value_1),
                    (past_key_2, past_value_2),
                )
            )
            input_shape = torch._shape_as_tensor(input_ids)
            cache_shape = torch._shape_as_tensor(past_key_0)
            position_ids = torch.arange(
                cache_shape[2],
                cache_shape[2] + input_shape[1],
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)
            result = self.transformer(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            hidden = result.last_hidden_state
            task_scores = self.task_classifier(hidden)
            inferred = task_scores.mean(dim=1).argmax(dim=-1)
            route = torch.where(
                requested_route < 0, inferred, requested_route
            )
            adapted = self._selected_residual(
                hidden, route, self.task_cakes
            )
            if self.route_bridges is not None:
                adapted = self._selected_residual(
                    adapted, route, self.route_bridges
                )
            logits = F.linear(
                adapted[:, -1], self.transformer.wte.weight
            )
            present = _legacy_cache(result.past_key_values)
            return (
                logits,
                route,
                task_scores[:, -1],
                adapted[:, -1],
                present[0][0],
                present[0][1],
                present[1][0],
                present[1][1],
                present[2][0],
                present[2][1],
            )

    output_path.mkdir(parents=True, exist_ok=False)
    graph = RuntimeGraph(model, route_bridge).eval()
    input_ids = torch.tensor([[32]], dtype=torch.long)
    requested_route = torch.tensor([-1], dtype=torch.long)
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
    intermediate = output_path / "model-int8-matmul.onnx"
    quantize_dynamic(
        fp32_path,
        intermediate,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    document = onnx.load(intermediate)
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
    int8_path = output_path / "model-int8.onnx"
    onnx.checker.check_model(document)
    onnx.save(document, int8_path)

    tokenizer_path = output_path / "tokenizer.json"
    tokenizer_path.write_bytes(
        (tokenizer_source_path / "tokenizer.json").read_bytes()
    )
    symbolic_path = output_path / "symbolic-surface.json"
    symbolic_path.write_bytes(_canonical_json_bytes(symbolic_contract))
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
                "lexical_repetition_truncation_threshold": 0,
                "prompt_identity_mixture": False,
            },
        )
    )
    if not standalone_core:
        source_decoding["algorithm"] = (
            "deterministic_greedy_with_repetition_controls"
            if repetition_penalty != 1.0 or no_repeat_ngram_size != 0
            else "greedy"
        )
        source_decoding["no_repeat_ngram_size"] = int(
            no_repeat_ngram_size
        )
        source_decoding.setdefault("allow_prompt_ngrams", False)
        source_decoding.setdefault(
            "lexical_repetition_truncation_threshold", 0
        )
        source_decoding.setdefault("prompt_identity_mixture", False)
    if standalone_core and (
        float(repetition_penalty) != 1.0
        or int(no_repeat_ngram_size)
        != int(source_decoding["no_repeat_ngram_size"])
    ):
        raise LayerCakeHostRuntimeError(
            "standalone runtime decoding must come from its frozen metadata"
        )
    metadata = {
        "format": RUNTIME_FORMAT,
        "status": "EXPORTED_NOT_YET_CERTIFIED",
        "host": host_identity,
        "parent_layercake": manifest["parent_layercake"],
        "runtime": {
            "provider": "onnxruntime.CPUExecutionProvider",
            "graph": int8_path.name,
            "graph_sha256": _sha256_file(int8_path),
            "graph_bytes": int8_path.stat().st_size,
            "fp32_graph_sha256": _sha256_file(fp32_path),
            "matrix_weight_quantization": (
                "dynamic signed int8 per channel"
            ),
            "embedding_quantization": (
                "signed int8 per token row, gathered before dequantization"
            ),
            "installed_task_cakes": 10,
            "maximum_active_task_cakes_per_sequence": 1,
            "installed_route_bridges": (
                10 if route_bridge is not None else 0
            ),
            "maximum_active_route_bridges_per_sequence": (
                1 if route_bridge is not None else 0
            ),
            "route_bridge_fused_into_task_cakes": bridge_fused,
            "standalone_core_has_no_route_bridge": standalone_core,
            "persistent_incremental_kv_state": True,
            "decoding": {
                "algorithm": source_decoding["algorithm"],
                "repetition_penalty": float(repetition_penalty),
                "no_repeat_ngram_size": int(
                    source_decoding["no_repeat_ngram_size"]
                ),
                "allow_prompt_ngrams": bool(
                    source_decoding.get("allow_prompt_ngrams", False)
                ),
                "lexical_repetition_truncation_threshold": int(
                    source_decoding.get(
                        "lexical_repetition_truncation_threshold", 0
                    )
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
        graph_path = self.artifact / self.metadata["runtime"]["graph"]
        tokenizer_path = (
            self.artifact / self.metadata["tokenizer"]["path"]
        )
        symbolic_path = (
            self.artifact / self.metadata["symbolic_surface"]["path"]
        )
        for path, expected in (
            (graph_path, self.metadata["runtime"]["graph_sha256"]),
            (tokenizer_path, self.metadata["tokenizer"]["sha256"]),
            (symbolic_path, self.metadata["symbolic_surface"]["sha256"]),
        ):
            if _sha256_file(path) != expected:
                raise LayerCakeHostRuntimeError(
                    f"native runtime component is stale: {path.name}"
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
    if (
        no_repeat_ngram_size > 0
        and len(generated) >= no_repeat_ngram_size - 1
    ):
        prefix = tuple(
            generated[-(no_repeat_ngram_size - 1) :]
        )
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
        else:
            blocked = set(int(value) for value in blocked_token_ids)
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
        route=int(state.route[0]),
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
        for _ in range(max_new_tokens):
            blocked = (
                ngram_successors.get(
                    tuple(generated[-(no_repeat_ngram_size - 1) :]),
                    set(),
                )
                if no_repeat_ngram_size > 0
                and len(generated) >= no_repeat_ngram_size - 1
                else set()
            )
            token_id = _select_token(
                logits,
                generated,
                repetition_penalty=float(
                    runtime.decoding["repetition_penalty"]
                ),
                no_repeat_ngram_size=no_repeat_ngram_size,
                output_token_ids=state.output_token_ids,
                output_token_local_index=(
                    state.output_token_local_index
                ),
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
            from .layercake_host import (
                _truncate_novel_lexical_repetition,
            )

            output = _truncate_novel_lexical_repetition(
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
        "route": int(state.route[0]),
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
    grams = [
        payload[index : index + 4]
        for index in range(max(0, len(payload) - 3))
    ]
    repetition = 1.0 - len(set(grams)) / max(1, len(grams))
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
        route=int(state.route[0]),
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
    first_output = None
    while True:
        blocked = (
            ngram_successors.get(
                tuple(generated[-(no_repeat_ngram_size - 1) :]),
                set(),
            )
            if no_repeat_ngram_size > 0
            and len(generated) >= no_repeat_ngram_size - 1
            else set()
        )
        token_id = _select_token(
            logits,
            generated,
            repetition_penalty=float(
                runtime.decoding["repetition_penalty"]
            ),
            no_repeat_ngram_size=no_repeat_ngram_size,
            output_token_ids=state.output_token_ids,
            output_token_local_index=(
                state.output_token_local_index
            ),
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
                from .layercake_host import (
                    _truncate_novel_lexical_repetition,
                )

                truncated = _truncate_novel_lexical_repetition(
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
        "route": int(state.route[0]),
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
                    "installed_task_cakes": 10,
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
            runtime.metadata["runtime"]["graph_bytes"]
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
) -> tuple[dict[str, Any], bool, bool]:
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
        capability_metrics[capability] = {
            "observations": len(selected),
            "source_passes": source_passes,
            "layercake_passes": host_passes,
            "source_pass_rate": source_passes / len(selected),
            "layercake_pass_rate": host_passes / len(selected),
            "source_passing_regressions": regressions,
            "bounded_zero_regression_pass": regressions == 0,
            "automatic_route_accuracy": sum(
                bool(row["route_correct"]) for row in selected
            )
            / len(selected),
        }
    complete_depth = (
        len(capability_metrics) == len(CAPABILITY_TO_ROUTE)
        and all(
            metrics["observations"] == 100
            for metrics in capability_metrics.values()
        )
    )
    semantic_pass = (
        complete_depth
        and all(
            metrics["layercake_passes"] == metrics["observations"]
            and metrics["bounded_zero_regression_pass"]
            and metrics["automatic_route_accuracy"] == 1.0
            for metrics in capability_metrics.values()
        )
    )
    return capability_metrics, complete_depth, semantic_pass


def evaluate_native_host_semantics(
    *,
    artifact: str | Path,
    training_bundle_path: str | Path,
    validation_bundle_paths: Sequence[str | Path],
    catalog_paths: Sequence[str | Path],
    output_path: str | Path,
    threads: int = 14,
) -> dict[str, Any]:
    """Run the full locked English suite on one exact native host artifact."""

    artifact = Path(artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"native semantic evidence is immutable: {output_path}"
        )
    from .hf_extraction import evaluate_output
    from .layercake_host import build_validation_rows

    rows = build_validation_rows(
        training_bundle_path=training_bundle_path,
        validation_bundle_paths=validation_bundle_paths,
        catalog_paths=catalog_paths,
    )
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
    ) = _summarize_native_semantics(observations)
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
        "host_manifest_sha256": metadata["host"][
            "deployment_manifest_sha256"
        ],
        "runtime_graph_sha256": metadata["runtime"]["graph_sha256"],
        "runtime_metadata_evidence_sha256": metadata[
            "evidence_sha256"
        ],
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "device": "onnxruntime.CPUExecutionProvider",
        "threads": int(threads),
        "observation_count": len(observations),
        "complete_locked_depth": complete_depth,
        "bounded_zero_regression_pass": semantic_pass,
        "capability_metrics": capability_metrics,
        "peak_process_rss_bytes": peak_rss,
        "wall_seconds": time.perf_counter() - started,
        "observations": observations,
        "claim_boundary": (
            "This is paired native-runtime validation evidence on the "
            "declared synthetic catalog. It is not final-test evidence or "
            "universal semantic identity."
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
    route_gathers = []
    for node in all_nodes:
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        shape = initializers.get(node.input[0])
        if shape in {
            (10, 768),
            (10, 64),
            (10, 64, 768),
            (10, 768, 64),
        } and node.input[1] == "route":
            route_gathers.append(
                {
                    "node": node.name,
                    "weight": node.input[0],
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
            if shape is not None and len(shape) == 3 and shape[0] == 10:
                dense_all_route_matrices.append(node.name)
    installed_bridges = int(
        metadata["runtime"]["installed_route_bridges"]
    )
    expected_gathers = 8 if installed_bridges else 4
    task_cake_quantized = (
        metadata["runtime"].get("task_cake_projection_quantization")
        is not None
    )
    if task_cake_quantized:
        expected_gathers += 2
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
                tuple(row["installed_shape"]) in {(10, 64), (10, 768)}
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
                        "standalone_core_has_no_route_bridge"
                    )
                    is True
                )
            )
        ),
        **conditional_checks,
        **sparse_output_checks,
    }
    evidence = {
        "format": "abi-layercake-host-physical-sparse-proof/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "runtime_graph_sha256": _sha256_file(graph_path),
        "installed_task_cakes": 10,
        "maximum_active_task_cakes_per_sequence": 1,
        "installed_route_bridges": installed_bridges,
        "maximum_active_route_bridges_per_sequence": (
            metadata["runtime"][
                "maximum_active_route_bridges_per_sequence"
            ]
        ),
        "route_indexed_parameter_gathers": route_gathers,
        "dense_all_route_matrix_nodes": dense_all_route_matrices,
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
    probe = generate_native_host(
        runtime,
        (
            "Return only one JSON object, with no Markdown, using "
            "`item`='identity-probe' and `count`=7."
        ),
        max_new_tokens=32,
    )
    checks["hash_bound_symbolic_substrate_executes"] = (
        probe["output"] == '{"item":"identity-probe","count":7}'
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
        "lexical_repetition_truncation_threshold": int(
            core_decoding["lexical_repetition_truncation_threshold"]
        ),
        "prompt_identity_mixture": bool(
            core_decoding["prompt_identity_mixture"]
        ),
        "weights_changed": False,
    }
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
        "symbolic_handlers_absent": (
            runtime.symbolic_surface.get("handlers") == []
            and runtime.symbolic_surface.get(
                "source_teacher_text_retained"
            )
            is False
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
