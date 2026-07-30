"""Derive and audit an adaptive low-rank LayerCake output shortlist.

Approximate low-rank scores retrieve candidates across the full tokenizer.
Only the candidate rows are then scored through the original exact int8
MatMulInteger path.  The graph returns the global candidate IDs used for each
step so runtime token accounting and repetition controls remain authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .layercake_host import _canonical_json_bytes
from .layercake_host_preservation import _load_general_rows
from .layercake_host_runtime import (
    NativeHostRuntime,
    RUNTIME_FORMAT,
    _canonical_sha,
    _select_token,
    _sha256_file,
)
from .layercake_host_train_calibrated_vocabulary import (
    CALIBRATION_FORMAT,
    CALIBRATION_SUFFIX,
)


FACTOR_FORMAT = "abi-layercake-output-shortlist-factorization/1"
TRACE_FORMAT = "abi-layercake-output-shortlist-train-trace/1"
VOCABULARY_FORMAT = "abi-layercake-adaptive-output-vocabulary/1"
ADAPTIVE_MODE = "adaptive_low_rank_shortlist_union_prompt_tokens"
EXPECTED_EMBEDDING_SHAPE = (50_257, 768)
LOCKED_CONFIGURATIONS = (
    ("rank32_top64", 32, 64),
    ("rank64_top128", 64, 128),
    ("rank128_top256", 128, 256),
)


class AdaptiveShortlistError(RuntimeError):
    """Raised when factorization, trace audit, or derivation is invalid."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _embedding_arrays(
    graph_path: Path,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    import onnx
    from onnx import numpy_helper

    document = onnx.load(graph_path)
    arrays = {
        value.name: numpy_helper.to_array(value)
        for value in document.graph.initializer
    }
    embeddings = [
        (name, array)
        for name, array in arrays.items()
        if array.shape == EXPECTED_EMBEDDING_SHAPE
        and array.dtype == np.int8
    ]
    scales = [
        (name, array)
        for name, array in arrays.items()
        if array.shape == (EXPECTED_EMBEDDING_SHAPE[0],)
        and array.dtype == np.float32
        and "row_scales" in name
    ]
    if len(embeddings) != 1 or len(scales) != 1:
        raise AdaptiveShortlistError(
            "expected one exact int8 embedding and one row-scale vector"
        )
    embedding_name, embedding = embeddings[0]
    scale_name, scale = scales[0]
    return embedding, scale, embedding_name, scale_name


def factorize_output_embedding(
    *,
    source_artifact: str | Path,
    factor_output_path: str | Path,
    evidence_output_path: str | Path,
    maximum_rank: int = 128,
    seed: int = 9824,
    power_iterations: int = 5,
) -> dict[str, Any]:
    """Compute one deterministic uncentered randomized SVD factorization."""

    from sklearn.utils.extmath import randomized_svd

    source_artifact = Path(source_artifact).resolve()
    factor_output_path = Path(factor_output_path).resolve()
    evidence_output_path = Path(evidence_output_path).resolve()
    if factor_output_path.exists() or evidence_output_path.exists():
        raise AdaptiveShortlistError(
            "factorization outputs are immutable"
        )
    metadata_path = source_artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != RUNTIME_FORMAT:
        raise AdaptiveShortlistError(
            "factorization source is not a native host"
        )
    graph_path = source_artifact / metadata["runtime"]["graph"]
    if _sha256_file(graph_path) != metadata["runtime"]["graph_sha256"]:
        raise AdaptiveShortlistError("factorization source graph changed")
    embedding, scales, embedding_name, scale_name = _embedding_arrays(
        graph_path
    )
    started = time.perf_counter()
    dequantized = (
        embedding.astype(np.float32)
        * scales.astype(np.float32)[:, None]
    )
    left, singular, right_t = randomized_svd(
        dequantized,
        n_components=maximum_rank,
        n_iter=power_iterations,
        random_state=seed,
        flip_sign=True,
    )
    projection = np.ascontiguousarray(
        right_t[:maximum_rank].T.astype(np.float32)
    )
    vocabulary_factor = np.ascontiguousarray(
        (
            singular[:maximum_rank, None]
            * left[:, :maximum_rank].T
        ).astype(np.float32)
    )
    reconstruction_energy = np.square(singular, dtype=np.float64)
    frobenius_energy = float(
        np.square(dequantized, dtype=np.float64).sum()
    )
    del dequantized
    factor_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        factor_output_path,
        projection=projection,
        vocabulary_factor=vocabulary_factor,
    )
    evidence: dict[str, Any] = {
        "format": FACTOR_FORMAT,
        "status": "FACTORIZED_FROM_DEPLOYED_LAYERCAKE_WEIGHTS",
        "source": {
            "artifact": str(source_artifact),
            "metadata_sha256": _sha256_file(metadata_path),
            "runtime_graph_sha256": _sha256_file(graph_path),
            "embedding_initializer": embedding_name,
            "embedding_int8_sha256": hashlib.sha256(
                np.ascontiguousarray(embedding).tobytes()
            ).hexdigest(),
            "row_scale_initializer": scale_name,
            "row_scales_sha256": hashlib.sha256(
                np.ascontiguousarray(scales).tobytes()
            ).hexdigest(),
        },
        "method": {
            "algorithm": "sklearn_randomized_svd_uncentered",
            "maximum_rank": maximum_rank,
            "seed": seed,
            "power_iterations": power_iterations,
            "source_shape": list(embedding.shape),
            "projection_shape": list(projection.shape),
            "vocabulary_factor_shape": list(
                vocabulary_factor.shape
            ),
            "projection_sha256": hashlib.sha256(
                projection.tobytes()
            ).hexdigest(),
            "vocabulary_factor_sha256": hashlib.sha256(
                vocabulary_factor.tobytes()
            ).hexdigest(),
            "singular_values": [
                float(value) for value in singular
            ],
            "retained_frobenius_energy_ratio": float(
                reconstruction_energy.sum()
                / max(frobenius_energy, 1.0e-30)
            ),
            "wall_seconds": time.perf_counter() - started,
        },
        "factor_file": {
            "path_at_creation": str(factor_output_path),
            "sha256": _sha256_file(factor_output_path),
            "bytes": factor_output_path.stat().st_size,
        },
        "imported_information_accounting": {
            "source_teacher_parameters": 0,
            "source_teacher_tokens": 0,
            "source_teacher_activations": 0,
            "layercake_parameters_changed": 0,
            "derived_projection_parameters": int(projection.size),
            "derived_vocabulary_factor_parameters": int(
                vocabulary_factor.size
            ),
        },
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _payload_hash(evidence)
    _write_json(evidence_output_path, evidence)
    return evidence


def evaluate_train_trace_recall(
    *,
    full_vocabulary_artifact: str | Path,
    factor_path: str | Path,
    factor_evidence_path: str | Path,
    calibration_evidence_path: str | Path,
    general_curriculum_path: str | Path,
    output_path: str | Path,
    threads: int = 16,
    trace_prompts: int = 32,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Audit selected-token retrieval recall on train-only trajectories."""

    import torch

    full_vocabulary_artifact = Path(
        full_vocabulary_artifact
    ).resolve()
    factor_path = Path(factor_path).resolve()
    factor_evidence_path = Path(factor_evidence_path).resolve()
    calibration_evidence_path = Path(
        calibration_evidence_path
    ).resolve()
    general_curriculum_path = Path(
        general_curriculum_path
    ).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise AdaptiveShortlistError(
            f"train-trace evidence is immutable: {output_path}"
        )
    factor_evidence = json.loads(
        factor_evidence_path.read_text(encoding="utf-8")
    )
    if (
        factor_evidence.get("format") != FACTOR_FORMAT
        or factor_evidence["factor_file"]["sha256"]
        != _sha256_file(factor_path)
    ):
        raise AdaptiveShortlistError(
            "factor evidence is invalid or stale"
        )
    calibration = json.loads(
        calibration_evidence_path.read_text(encoding="utf-8")
    )
    if (
        calibration.get("format") != CALIBRATION_FORMAT
        or trace_prompts <= 0
        or trace_prompts > len(calibration["records"])
    ):
        raise AdaptiveShortlistError(
            "calibration evidence cannot supply the locked trace"
        )
    factor_arrays = np.load(factor_path)
    projection = np.asarray(
        factor_arrays["projection"], dtype=np.float32
    )
    vocabulary_factor = np.asarray(
        factor_arrays["vocabulary_factor"], dtype=np.float32
    )
    if (
        projection.shape != (768, 128)
        or vocabulary_factor.shape != (128, 50_257)
    ):
        raise AdaptiveShortlistError("factor shapes changed")

    rows = _load_general_rows(general_curriculum_path, split="train")
    by_hash = {str(row["prompt_sha256"]): row for row in rows}
    runtime = NativeHostRuntime(
        full_vocabulary_artifact, threads=threads
    )
    hidden_states: list[np.ndarray] = []
    targets: list[int] = []
    allowed_sets: list[set[int]] = []
    trajectory_checks = 0
    started = time.perf_counter()
    for record in calibration["records"][:trace_prompts]:
        if record["source_split"] != "train":
            raise AdaptiveShortlistError(
                "trace record is not from the train split"
            )
        row = by_hash.get(str(record["source_prompt_sha256"]))
        if row is None:
            raise AdaptiveShortlistError(
                "trace train prompt is absent from the locked curriculum"
            )
        prompt = str(row["prompt"]) + CALIBRATION_SUFFIX
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != record[
            "transformed_prompt_sha256"
        ]:
            raise AdaptiveShortlistError(
                "trace transformed prompt hash changed"
            )
        prompt_ids = runtime.encode(prompt + "\n")
        allowed = set(range(256)) | {50_256} | set(prompt_ids)
        logits, state = runtime.prefill(prompt_ids)
        generated: list[int] = []
        for expected in record["authoritative_generated_token_ids"]:
            selected = _select_token(
                logits,
                generated,
                repetition_penalty=float(
                    runtime.decoding["repetition_penalty"]
                ),
                no_repeat_ngram_size=int(
                    runtime.decoding["no_repeat_ngram_size"]
                ),
            )
            if selected != int(expected):
                raise AdaptiveShortlistError(
                    "full-head train trajectory is not reproducible"
                )
            hidden_states.append(
                np.asarray(state.abi_state, dtype=np.float32).reshape(768)
            )
            targets.append(selected)
            allowed_sets.append(allowed)
            generated.append(selected)
            logits, state = runtime.decode_step(selected, state)
            trajectory_checks += 1

    hidden = np.stack(hidden_states)
    target_array = np.asarray(targets, dtype=np.int64)
    device = torch.device(
        device_name
        if device_name != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    hidden_tensor = torch.from_numpy(hidden).to(device)
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for config_id, rank, top_k in LOCKED_CONFIGURATIONS:
            projected = hidden_tensor @ torch.from_numpy(
                projection[:, :rank]
            ).to(device)
            factor = torch.from_numpy(
                vocabulary_factor[:rank]
            ).to(device)
            retrieved = []
            batch_size = 256
            for offset in range(0, len(hidden), batch_size):
                scores = projected[offset : offset + batch_size] @ factor
                retrieved.append(
                    torch.topk(
                        scores, k=top_k, dim=1, largest=True, sorted=False
                    ).indices.cpu().numpy()
                )
            top_ids = np.concatenate(retrieved, axis=0)
            hits = np.asarray(
                [
                    int(target_array[index]) in allowed_sets[index]
                    or bool(
                        np.any(
                            top_ids[index] == target_array[index]
                        )
                    )
                    for index in range(len(target_array))
                ],
                dtype=np.bool_,
            )
            results[config_id] = {
                "rank": rank,
                "top_k": top_k,
                "selected_token_observations": len(hits),
                "selected_token_hits": int(hits.sum()),
                "selected_token_recall": float(hits.mean()),
                "selected_token_misses": int((~hits).sum()),
            }
            del projected, factor
    del hidden_tensor, hidden_states, hidden
    if device.type == "cuda":
        torch.cuda.empty_cache()

    evidence: dict[str, Any] = {
        "format": TRACE_FORMAT,
        "status": "TRAIN_TRACE_COMPLETE",
        "sources": {
            "full_vocabulary_artifact": str(
                full_vocabulary_artifact
            ),
            "full_vocabulary_metadata_sha256": _sha256_file(
                full_vocabulary_artifact / "metadata.json"
            ),
            "factor_file_sha256": _sha256_file(factor_path),
            "factor_evidence_sha256": factor_evidence[
                "evidence_sha256"
            ],
            "calibration_evidence_sha256": calibration[
                "evidence_sha256"
            ],
            "general_curriculum_sha256": _sha256_file(
                general_curriculum_path
            ),
        },
        "trace": {
            "source_split": "train",
            "prompt_count": trace_prompts,
            "trajectory_token_observations": trajectory_checks,
            "full_head_trajectory_mismatches": 0,
            "validation_rows_seen": 0,
            "benchmark_rows_seen": 0,
            "final_test_rows_seen": 0,
            "hidden_activations_persisted": 0,
            "evaluation_device": str(device),
            "wall_seconds": time.perf_counter() - started,
        },
        "configurations": results,
        "claim_boundary": (
            "This is train-only shortlist recall, not validation quality. "
            "Approximate retrieval is promoted only after exact-reranked "
            "runtime quality and speed gates pass."
        ),
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _payload_hash(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def derive_adaptive_shortlist_artifact(
    *,
    base_dynamic_artifact: str | Path,
    factor_path: str | Path,
    factor_evidence_path: str | Path,
    trace_evidence_path: str | Path,
    configuration_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Insert low-rank TopK retrieval before the exact dynamic head."""

    import onnx
    from onnx import helper, numpy_helper

    configurations = {
        config_id: (rank, top_k)
        for config_id, rank, top_k in LOCKED_CONFIGURATIONS
    }
    if configuration_id not in configurations:
        raise AdaptiveShortlistError(
            f"unknown shortlist configuration: {configuration_id}"
        )
    rank, top_k = configurations[configuration_id]
    base_dynamic_artifact = Path(base_dynamic_artifact).resolve()
    factor_path = Path(factor_path).resolve()
    factor_evidence_path = Path(factor_evidence_path).resolve()
    trace_evidence_path = Path(trace_evidence_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise AdaptiveShortlistError(
            f"adaptive artifact is immutable: {output_path}"
        )
    factor_evidence = json.loads(
        factor_evidence_path.read_text(encoding="utf-8")
    )
    trace_evidence = json.loads(
        trace_evidence_path.read_text(encoding="utf-8")
    )
    if (
        factor_evidence.get("format") != FACTOR_FORMAT
        or factor_evidence["factor_file"]["sha256"]
        != _sha256_file(factor_path)
        or trace_evidence.get("format") != TRACE_FORMAT
        or configuration_id
        not in trace_evidence["configurations"]
    ):
        raise AdaptiveShortlistError(
            "adaptive factor or trace evidence is invalid"
        )
    factor_arrays = np.load(factor_path)
    projection = np.ascontiguousarray(
        np.asarray(factor_arrays["projection"], dtype=np.float32)[
            :, :rank
        ]
    )
    vocabulary_factor = np.ascontiguousarray(
        np.asarray(
            factor_arrays["vocabulary_factor"], dtype=np.float32
        )[:rank]
    )

    metadata = json.loads(
        (base_dynamic_artifact / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata.get("format") != RUNTIME_FORMAT:
        raise AdaptiveShortlistError(
            "adaptive base is not a native host"
        )
    contract = metadata["runtime"]["output_vocabulary"]
    if (
        contract.get("mode")
        != "train_base_union_prompt_tokens"
        or int(contract["selected_token_count"]) != 257
    ):
        raise AdaptiveShortlistError(
            "adaptive derivation requires the byte/EOS dynamic base"
        )
    graph_source = base_dynamic_artifact / metadata["runtime"]["graph"]
    vocabulary_source = base_dynamic_artifact / contract["path"]
    tokenizer_source = (
        base_dynamic_artifact / metadata["tokenizer"]["path"]
    )
    symbolic_source = (
        base_dynamic_artifact / metadata["symbolic_surface"]["path"]
    )
    for path, expected in (
        (graph_source, metadata["runtime"]["graph_sha256"]),
        (vocabulary_source, contract["sha256"]),
        (tokenizer_source, metadata["tokenizer"]["sha256"]),
        (symbolic_source, metadata["symbolic_surface"]["sha256"]),
    ):
        if _sha256_file(path) != expected:
            raise AdaptiveShortlistError(
                f"adaptive source changed: {path.name}"
            )
    vocabulary = json.loads(
        vocabulary_source.read_text(encoding="utf-8")
    )
    base_ids = [
        int(value) for value in vocabulary["global_token_ids"]
    ]
    if (
        base_ids != sorted(set(base_ids))
        or set(base_ids) != set(range(256)) | {50_256}
    ):
        raise AdaptiveShortlistError(
            "byte/EOS adaptive base IDs changed"
        )

    document = onnx.load(graph_source)
    projection_name = "adaptive_shortlist_projection"
    factor_name = "adaptive_shortlist_vocabulary_factor"
    top_k_name = "adaptive_shortlist_k"
    flat_shape_name = "adaptive_shortlist_flat_shape"
    projected_name = "adaptive_shortlist_projected_state"
    scores_name = "adaptive_shortlist_scores"
    top_values_name = "adaptive_shortlist_top_values"
    top_indices_2d_name = "adaptive_shortlist_top_indices_2d"
    top_indices_name = "adaptive_shortlist_top_indices"
    selected_ids_name = "adaptive_selected_output_ids"
    document.graph.initializer.extend(
        [
            numpy_helper.from_array(
                projection, name=projection_name
            ),
            numpy_helper.from_array(
                vocabulary_factor, name=factor_name
            ),
            numpy_helper.from_array(
                np.asarray([top_k], dtype=np.int64),
                name=top_k_name,
            ),
            numpy_helper.from_array(
                np.asarray([-1], dtype=np.int64),
                name=flat_shape_name,
            ),
        ]
    )
    adaptive_nodes = [
        helper.make_node(
            "MatMul",
            ["/Gather_5_output_0", projection_name],
            [projected_name],
            name="AdaptiveShortlistProjectState",
        ),
        helper.make_node(
            "MatMul",
            [projected_name, factor_name],
            [scores_name],
            name="AdaptiveShortlistScoreVocabulary",
        ),
        helper.make_node(
            "TopK",
            [scores_name, top_k_name],
            [top_values_name, top_indices_2d_name],
            name="AdaptiveShortlistTopK",
            axis=1,
            largest=1,
            sorted=0,
        ),
        helper.make_node(
            "Reshape",
            [top_indices_2d_name, flat_shape_name],
            [top_indices_name],
            name="AdaptiveShortlistFlattenIDs",
        ),
        helper.make_node(
            "Concat",
            ["allowed_output_ids", top_indices_name],
            [selected_ids_name],
            name="AdaptiveShortlistUnionCandidates",
            axis=0,
        ),
    ]
    nodes = list(document.graph.node)
    gather_indices = [
        index
        for index, node in enumerate(nodes)
        if node.name
        in ("DynamicOutputWeightGather", "DynamicOutputScaleGather")
    ]
    if len(gather_indices) != 2:
        raise AdaptiveShortlistError(
            "exact dynamic gather nodes changed"
        )
    insert_at = min(gather_indices)
    for offset, node in enumerate(adaptive_nodes):
        document.graph.node.insert(insert_at + offset, node)
    for node in document.graph.node:
        if node.name in (
            "DynamicOutputWeightGather",
            "DynamicOutputScaleGather",
        ):
            node.input[
                list(node.input).index("allowed_output_ids")
            ] = selected_ids_name
    logits = [
        value for value in document.graph.output
        if value.name == "logits"
    ]
    if len(logits) != 1:
        raise AdaptiveShortlistError("adaptive logits output changed")
    width = logits[0].type.tensor_type.shape.dim[1]
    width.ClearField("dim_param")
    width.dim_param = "adaptive_output_tokens"
    document.graph.output.extend(
        [
            helper.make_tensor_value_info(
                selected_ids_name,
                onnx.TensorProto.INT64,
                ["adaptive_output_tokens"],
            )
        ]
    )

    output_path.mkdir(parents=True, exist_ok=False)
    graph_path = output_path / "model-int8.onnx"
    onnx.checker.check_model(document)
    onnx.save(document, graph_path)
    tokenizer_path = output_path / "tokenizer.json"
    symbolic_path = output_path / "symbolic-surface.json"
    shutil.copyfile(tokenizer_source, tokenizer_path)
    shutil.copyfile(symbolic_source, symbolic_path)

    adaptive_vocabulary = json.loads(json.dumps(vocabulary))
    adaptive_vocabulary["format"] = VOCABULARY_FORMAT
    adaptive_vocabulary["status"] = (
        "DERIVED_ADAPTIVE_SHORTLIST_UNION_PROMPT_IDS"
    )
    adaptive_vocabulary["mode"] = ADAPTIVE_MODE
    adaptive_vocabulary["base_global_token_ids"] = base_ids
    adaptive_vocabulary["base_token_count"] = len(base_ids)
    adaptive_vocabulary["allowed_output_ids_graph_input"] = (
        "allowed_output_ids"
    )
    adaptive_vocabulary["selected_output_ids_graph_output"] = (
        selected_ids_name
    )
    adaptive_vocabulary["adaptive_shortlist"] = {
        "configuration_id": configuration_id,
        "rank": rank,
        "top_k": top_k,
        "projection_shape": list(projection.shape),
        "projection_sha256": hashlib.sha256(
            projection.tobytes()
        ).hexdigest(),
        "vocabulary_factor_shape": list(
            vocabulary_factor.shape
        ),
        "vocabulary_factor_sha256": hashlib.sha256(
            vocabulary_factor.tobytes()
        ).hexdigest(),
        "factor_evidence_sha256": factor_evidence[
            "evidence_sha256"
        ],
        "train_trace_evidence_sha256": trace_evidence[
            "evidence_sha256"
        ],
        "train_trace_selected_token_recall": trace_evidence[
            "configurations"
        ][configuration_id]["selected_token_recall"],
        "exact_rerank_uses_original_int8_rows_and_scales": True,
        "approximate_retrieval_is_lossless_claim": False,
    }
    adaptive_vocabulary["claim_boundary"] = (
        "Low-rank retrieval is approximate. Candidate reranking reuses the "
        "original exact int8 rows and scales. Only the complete bounded "
        "quality and speed suites can promote this artifact."
    )
    adaptive_vocabulary.pop("evidence_sha256", None)
    adaptive_vocabulary["evidence_sha256"] = _payload_hash(
        adaptive_vocabulary
    )
    vocabulary_path = output_path / "output-vocabulary.json"
    _write_json(vocabulary_path, adaptive_vocabulary)

    derived = json.loads(json.dumps(metadata))
    derived["status"] = "EXPORTED_NOT_YET_CERTIFIED"
    runtime = derived["runtime"]
    runtime["graph"] = graph_path.name
    runtime["graph_sha256"] = _sha256_file(graph_path)
    runtime["graph_bytes"] = graph_path.stat().st_size
    runtime["output_vocabulary"] = {
        "format": VOCABULARY_FORMAT,
        "mode": ADAPTIVE_MODE,
        "path": vocabulary_path.name,
        "sha256": _sha256_file(vocabulary_path),
        "budget_id": "byte_fallback_only",
        "selected_token_count": len(base_ids),
        "base_token_count": len(base_ids),
        "full_token_count": 50_257,
        "byte_fallback_ids_complete": True,
        "global_runtime_token_ids": True,
        "allowed_output_ids_graph_input": "allowed_output_ids",
        "selected_output_ids_graph_output": selected_ids_name,
        "prompt_token_ids_added_at_runtime": True,
        "duplicate_output_projection_removed": True,
        "adaptive_shortlist_configuration": configuration_id,
        "adaptive_shortlist_rank": rank,
        "adaptive_shortlist_top_k": top_k,
        "exact_rerank_uses_original_int8_rows_and_scales": True,
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
    commands = parser.add_subparsers(dest="command", required=True)

    factorize = commands.add_parser("factorize")
    factorize.add_argument("--source-artifact", required=True)
    factorize.add_argument("--factor-output", required=True)
    factorize.add_argument("--evidence-output", required=True)
    factorize.add_argument("--maximum-rank", type=int, default=128)
    factorize.add_argument("--seed", type=int, default=9824)
    factorize.add_argument("--power-iterations", type=int, default=5)

    trace = commands.add_parser("trace")
    trace.add_argument("--full-vocabulary-artifact", required=True)
    trace.add_argument("--factor", required=True)
    trace.add_argument("--factor-evidence", required=True)
    trace.add_argument("--calibration-evidence", required=True)
    trace.add_argument("--general-curriculum", required=True)
    trace.add_argument("--output", required=True)
    trace.add_argument("--threads", type=int, default=16)
    trace.add_argument("--trace-prompts", type=int, default=32)
    trace.add_argument("--device", default="cuda")

    derive = commands.add_parser("derive")
    derive.add_argument("--base-dynamic-artifact", required=True)
    derive.add_argument("--factor", required=True)
    derive.add_argument("--factor-evidence", required=True)
    derive.add_argument("--trace-evidence", required=True)
    derive.add_argument(
        "--configuration-id",
        choices=tuple(row[0] for row in LOCKED_CONFIGURATIONS),
        required=True,
    )
    derive.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "factorize":
        result = factorize_output_embedding(
            source_artifact=args.source_artifact,
            factor_output_path=args.factor_output,
            evidence_output_path=args.evidence_output,
            maximum_rank=args.maximum_rank,
            seed=args.seed,
            power_iterations=args.power_iterations,
        )
    elif args.command == "trace":
        result = evaluate_train_trace_recall(
            full_vocabulary_artifact=args.full_vocabulary_artifact,
            factor_path=args.factor,
            factor_evidence_path=args.factor_evidence,
            calibration_evidence_path=args.calibration_evidence,
            general_curriculum_path=args.general_curriculum,
            output_path=args.output,
            threads=args.threads,
            trace_prompts=args.trace_prompts,
            device_name=args.device,
        )
    else:
        result = derive_adaptive_shortlist_artifact(
            base_dynamic_artifact=args.base_dynamic_artifact,
            factor_path=args.factor,
            factor_evidence_path=args.factor_evidence,
            trace_evidence_path=args.trace_evidence,
            configuration_id=args.configuration_id,
            output_path=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
