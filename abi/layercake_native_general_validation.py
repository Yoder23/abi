"""Held-out general-English preservation on exact native LayerCake graphs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .layercake_host import LayerCakeHostError, _canonical_json_bytes
from .layercake_host_preservation import _load_general_rows
from .layercake_host_runtime import (
    RUNTIME_FORMAT,
    _sha256_file,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class _State:
    def __init__(
        self,
        route: np.ndarray,
        cache: list[np.ndarray],
        output_token_ids: np.ndarray | None = None,
    ):
        self.route = route
        self.cache = cache
        self.output_token_ids = output_token_ids


class _TeacherForcedNativeRuntime:
    """Minimal common runner for sealed-parent and ABI-host ONNX graphs."""

    def __init__(
        self,
        *,
        graph_path: Path,
        tokenizer_path: Path,
        threads: int,
        output_token_ids: Sequence[int] | None = None,
        dynamic_prompt_union: bool = False,
        cake_activation_schedule: Mapping[str, Any] | None = None,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.enable_mem_pattern = False
        options.enable_mem_reuse = True
        options.add_session_config_entry(
            "session.disable_prepacking", "1"
        )
        self.session = ort.InferenceSession(
            str(graph_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.output_token_ids = (
            None
            if output_token_ids is None
            else np.asarray(output_token_ids, dtype=np.int64)
        )
        self.dynamic_prompt_union = bool(dynamic_prompt_union)
        self.cake_activation_schedule = (
            None
            if cake_activation_schedule is None
            else dict(cake_activation_schedule)
        )
        self.empty = [
            np.zeros((1, 12, 0, 64), dtype=np.float32)
            for _ in range(6)
        ]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def _run(
        self,
        token_id: int,
        route: np.ndarray,
        cache: list[np.ndarray],
        output_token_ids: np.ndarray | None = None,
        cake_active: bool = False,
    ) -> tuple[np.ndarray, _State, np.ndarray]:
        feeds = {
            "input_ids": np.asarray([[token_id]], dtype=np.int64),
            "requested_route": route.astype(np.int64, copy=False),
            "past_key_0": cache[0],
            "past_value_0": cache[1],
            "past_key_1": cache[2],
            "past_value_1": cache[3],
            "past_key_2": cache[4],
            "past_value_2": cache[5],
        }
        if self.dynamic_prompt_union:
            if output_token_ids is None or not len(output_token_ids):
                raise LayerCakeHostError(
                    "dynamic candidate requires prompt-bound token IDs"
                )
            feeds["allowed_output_ids"] = output_token_ids.astype(
                np.int64, copy=False
            )
        if self.cake_activation_schedule is not None:
            if (
                self.cake_activation_schedule.get("format")
                != "abi-layercake-conditional-core-realization-schedule/1"
                or self.cake_activation_schedule.get(
                    "graph_input_type"
                )
                != "bool"
            ):
                raise LayerCakeHostError(
                    "native general runner requires the conditional schedule"
                )
            feeds[
                self.cake_activation_schedule["graph_input"]
            ] = np.asarray(bool(cake_active), dtype=np.bool_)
        outputs = self.session.run(None, feeds)
        return (
            outputs[0],
            _State(
                outputs[1],
                list(outputs[4:]),
                (
                    output_token_ids
                    if self.dynamic_prompt_union
                    else self.output_token_ids
                ),
            ),
            outputs[2],
        )

    def _top_global_id(
        self,
        logits: np.ndarray,
        output_token_ids: np.ndarray | None = None,
    ) -> int:
        local = int(logits[0].argmax())
        active = (
            output_token_ids
            if self.dynamic_prompt_union
            else self.output_token_ids
        )
        if active is None:
            return local
        if logits.shape[-1] != len(active):
            raise LayerCakeHostError(
                "candidate sparse logits width differs from token map"
            )
        return int(active[local])

    def prefill(self, ids: Sequence[int]) -> tuple[np.ndarray, _State]:
        if not ids:
            raise LayerCakeHostError(
                "native general validation prompt encoded empty"
            )
        cache = self.empty
        scores = []
        cache_before_last = cache
        output_token_ids = self.output_token_ids
        if self.dynamic_prompt_union:
            if output_token_ids is None:
                raise LayerCakeHostError(
                    "dynamic candidate base vocabulary is absent"
                )
            output_token_ids = np.unique(
                np.concatenate(
                    (
                        output_token_ids,
                        np.asarray(ids, dtype=np.int64),
                    )
                )
            )
        for index, token_id in enumerate(ids):
            if index == len(ids) - 1:
                cache_before_last = cache
            _, state, token_scores = self._run(
                int(token_id),
                np.asarray([-1], dtype=np.int64),
                cache,
                output_token_ids,
                False,
            )
            cache = state.cache
            scores.append(token_scores)
        route = np.asarray(
            [int(np.mean(np.stack(scores), axis=0).argmax())],
            dtype=np.int64,
        )
        logits, state, _ = self._run(
            int(ids[-1]),
            route,
            cache_before_last,
            output_token_ids,
            True,
        )
        return logits, state

    def decode_step(
        self, token_id: int, state: _State
    ) -> tuple[np.ndarray, _State]:
        logits, next_state, _ = self._run(
            int(token_id),
            state.route,
            state.cache,
            state.output_token_ids,
            False,
        )
        return logits, next_state

    def teacher_forced_predictions(
        self,
        prompt: str,
        response: str,
        *,
        max_tokens: int,
    ) -> tuple[list[int], list[int], int]:
        prompt_ids = self.encode(prompt + "\n")
        if len(prompt_ids) >= max_tokens:
            raise LayerCakeHostError(
                "native general validation prompt exceeds context cap"
            )
        response_ids = self.encode(response)[
            : max_tokens - len(prompt_ids)
        ]
        if not response_ids:
            raise LayerCakeHostError(
                "native general validation response encoded empty"
            )
        logits, state = self.prefill(prompt_ids)
        predictions = []
        for index, target in enumerate(response_ids):
            predictions.append(
                self._top_global_id(logits, state.output_token_ids)
            )
            if index + 1 < len(response_ids):
                logits, state = self.decode_step(target, state)
        return predictions, response_ids, int(state.route[0])


def _parent_paths(parent_artifact: Path) -> tuple[Path, Path, dict[str, Any]]:
    metadata_path = parent_artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != (
        "layercake-shallow-sparse-onnx-runtime/2"
    ):
        raise LayerCakeHostError(
            "parent artifact is not the sealed native runtime format"
        )
    graph_path = parent_artifact / "model-int8.onnx"
    tokenizer_path = parent_artifact / "tokenizer.json"
    if _sha256_file(graph_path) != metadata["runtime"]["graph_sha256"]:
        raise LayerCakeHostError("sealed parent graph hash changed")
    if _sha256_file(tokenizer_path) != metadata["tokenizer"]["sha256"]:
        raise LayerCakeHostError("sealed parent tokenizer hash changed")
    return graph_path, tokenizer_path, metadata


def _candidate_paths(
    candidate_artifact: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    metadata_path = candidate_artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != RUNTIME_FORMAT:
        raise LayerCakeHostError(
            "candidate artifact is not an ABI native runtime"
        )
    graph_path = candidate_artifact / metadata["runtime"]["graph"]
    tokenizer_path = (
        candidate_artifact / metadata["tokenizer"]["path"]
    )
    if _sha256_file(graph_path) != metadata["runtime"]["graph_sha256"]:
        raise LayerCakeHostError("candidate native graph hash changed")
    if _sha256_file(tokenizer_path) != metadata["tokenizer"]["sha256"]:
        raise LayerCakeHostError(
            "candidate native tokenizer hash changed"
        )
    return graph_path, tokenizer_path, metadata


def evaluate_native_general_preservation(
    *,
    curriculum_path: str | Path,
    parent_artifact: str | Path,
    candidate_artifact: str | Path,
    output_path: str | Path,
    threads: int = 14,
    max_tokens: int = 192,
) -> dict[str, Any]:
    """Compare authoritative native top-1 IDs over every response token."""

    curriculum_path = Path(curriculum_path).resolve()
    parent_artifact = Path(parent_artifact).resolve()
    candidate_artifact = Path(candidate_artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(
            f"native general evidence is immutable: {output_path}"
        )
    if threads <= 0 or max_tokens <= 0:
        raise LayerCakeHostError(
            "native general threads and token cap must be positive"
        )
    rows = _load_general_rows(
        curriculum_path, split="instruction_validation"
    )
    if len(rows) != 420:
        raise LayerCakeHostError(
            "locked general instruction-validation depth changed"
        )
    parent_graph, parent_tokenizer, parent_metadata = _parent_paths(
        parent_artifact
    )
    (
        candidate_graph,
        candidate_tokenizer,
        candidate_metadata,
    ) = _candidate_paths(candidate_artifact)
    if (
        _sha256_file(parent_tokenizer)
        != _sha256_file(candidate_tokenizer)
    ):
        raise LayerCakeHostError(
            "parent and candidate tokenizers are not identical"
        )

    started = time.perf_counter()
    parent = _TeacherForcedNativeRuntime(
        graph_path=parent_graph,
        tokenizer_path=parent_tokenizer,
        threads=threads,
    )
    parent_results: dict[str, tuple[list[int], list[int], int]] = {}
    for index, row in enumerate(rows, start=1):
        parent_results[str(row["id"])] = (
            parent.teacher_forced_predictions(
                str(row["prompt"]),
                str(row["response"]),
                max_tokens=max_tokens,
            )
        )
        if index % 100 == 0:
            print(
                json.dumps(
                    {
                        "runtime": "sealed_parent",
                        "evaluated": index,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
    del parent
    gc.collect()

    candidate = _TeacherForcedNativeRuntime(
        graph_path=candidate_graph,
        tokenizer_path=candidate_tokenizer,
        threads=threads,
        output_token_ids=(
            json.loads(
                (
                    candidate_artifact
                    / candidate_metadata["runtime"][
                        "output_vocabulary"
                    ]["path"]
                ).read_text(encoding="utf-8")
            )["global_token_ids"]
            if candidate_metadata["runtime"].get(
                "output_vocabulary"
            )
            else None
        ),
        dynamic_prompt_union=(
            candidate_metadata["runtime"]
            .get("output_vocabulary", {})
            .get("mode")
            == "train_base_union_prompt_tokens"
        ),
        cake_activation_schedule=candidate_metadata["runtime"].get(
            "cake_activation_schedule"
        ),
    )
    observations = []
    total_compared = 0
    total_agreed = 0
    candidate_correct = 0
    parent_correct = 0
    route_equal_rows = 0
    for index, row in enumerate(rows, start=1):
        candidate_predictions, targets, candidate_route = (
            candidate.teacher_forced_predictions(
                str(row["prompt"]),
                str(row["response"]),
                max_tokens=max_tokens,
            )
        )
        parent_predictions, parent_targets, parent_route = (
            parent_results[str(row["id"])]
        )
        if targets != parent_targets:
            raise LayerCakeHostError(
                "native teacher-forced target IDs changed"
            )
        if len(candidate_predictions) != len(parent_predictions):
            raise LayerCakeHostError(
                "native prediction lengths changed"
            )
        compared = len(targets)
        agreed = sum(
            left == right
            for left, right in zip(
                candidate_predictions, parent_predictions
            )
        )
        candidate_hits = sum(
            predicted == target
            for predicted, target in zip(
                candidate_predictions, targets
            )
        )
        parent_hits = sum(
            predicted == target
            for predicted, target in zip(
                parent_predictions, targets
            )
        )
        route_equal = candidate_route == parent_route
        total_compared += compared
        total_agreed += agreed
        candidate_correct += candidate_hits
        parent_correct += parent_hits
        route_equal_rows += int(route_equal)
        observations.append(
            {
                "row_id": row["id"],
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
                "compared_response_tokens": compared,
                "parent_top1_agreed_tokens": agreed,
                "parent_top1_agreement": agreed / compared,
                "candidate_target_top1_tokens": candidate_hits,
                "parent_target_top1_tokens": parent_hits,
                "candidate_route": candidate_route,
                "parent_route": parent_route,
                "route_equal": route_equal,
            }
        )
        if index % 100 == 0:
            print(
                json.dumps(
                    {
                        "runtime": "candidate",
                        "evaluated": index,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
    agreement = total_agreed / max(1, total_compared)
    checks = {
        "locked_420_row_depth": len(observations) == 420,
        "locked_43089_response_token_depth": (
            total_compared == 43089
        ),
        "parent_top1_agreement_at_least_095": agreement >= 0.95,
        "validation_only": True,
        "final_test_unopened": True,
        "identical_tokenizers": True,
        "persistent_incremental_state_used": True,
    }
    evidence = {
        "format": (
            "abi-layercake-native-general-preservation-validation/1"
        ),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "split": "instruction_validation",
        "final_test_accessed": False,
        "curriculum_sha256": _sha256_file(curriculum_path),
        "candidate_runtime_graph_sha256": candidate_metadata[
            "runtime"
        ]["graph_sha256"],
        "candidate_metadata_evidence_sha256": candidate_metadata[
            "evidence_sha256"
        ],
        "validation_runner_sha256": _sha256_file(Path(__file__)),
        "parent_runtime_graph_sha256": parent_metadata["runtime"][
            "graph_sha256"
        ],
        "parent_checkpoint_sha256": parent_metadata[
            "source_checkpoint_sha256"
        ],
        "threads": threads,
        "max_tokens": max_tokens,
        "observation_count": len(observations),
        "compared_response_tokens": total_compared,
        "parent_top1_agreed_tokens": total_agreed,
        "parent_top1_agreement": agreement,
        "candidate_target_top1_accuracy": (
            candidate_correct / max(1, total_compared)
        ),
        "parent_target_top1_accuracy": (
            parent_correct / max(1, total_compared)
        ),
        "route_agreement": route_equal_rows / len(observations),
        "checks": checks,
        "wall_seconds": time.perf_counter() - started,
        "observations": observations,
        "claim_boundary": (
            "This is held-out teacher-forced agreement between exact native "
            "graphs. It is not autonomous-generation or final-test evidence."
        ),
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--max-tokens", type=int, default=192)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_native_general_preservation(
        curriculum_path=args.curriculum,
        parent_artifact=args.parent_artifact,
        candidate_artifact=args.candidate_artifact,
        output_path=args.output,
        threads=args.threads,
        max_tokens=args.max_tokens,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "evidence_sha256",
                    "observation_count",
                    "compared_response_tokens",
                    "parent_top1_agreement",
                    "candidate_target_top1_accuracy",
                    "parent_target_top1_accuracy",
                    "route_agreement",
                    "checks",
                    "wall_seconds",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
