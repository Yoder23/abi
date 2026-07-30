"""Create a train-only long-form curriculum from the sealed LayerCake parent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .layercake_host import _canonical_json_bytes
from .layercake_host_runtime import _select_token, _sha256_file
from .layercake_host_train_calibrated_vocabulary import (
    CALIBRATION_FORMAT,
)


CURRICULUM_FORMAT = "abi-layercake-parent-longform-curriculum/1"


class ParentCurriculumError(RuntimeError):
    """Raised when parent generation or curriculum isolation fails."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _ParentRuntime:
    def __init__(self, artifact: Path, *, threads: int):
        metadata = json.loads(
            (artifact / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("format") != (
            "layercake-shallow-sparse-onnx-runtime/2"
        ):
            raise ParentCurriculumError(
                "sealed parent runtime format changed"
            )
        graph = artifact / "model-int8.onnx"
        tokenizer = artifact / "tokenizer.json"
        if (
            _sha256_file(graph)
            != metadata["runtime"]["graph_sha256"]
            or _sha256_file(tokenizer)
            != metadata["tokenizer"]["sha256"]
        ):
            raise ParentCurriculumError(
                "sealed parent runtime component changed"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
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
            str(graph),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.empty = [
            np.zeros((1, 12, 0, 64), dtype=np.float32)
            for _ in range(6)
        ]
        self.metadata = metadata

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=True
        )

    def _run(
        self,
        token_id: int,
        route: np.ndarray,
        cache: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
        outputs = self.session.run(
            None,
            {
                "input_ids": np.asarray(
                    [[token_id]], dtype=np.int64
                ),
                "requested_route": route.astype(
                    np.int64, copy=False
                ),
                "past_key_0": cache[0],
                "past_value_0": cache[1],
                "past_key_1": cache[2],
                "past_value_1": cache[3],
                "past_key_2": cache[4],
                "past_value_2": cache[5],
            },
        )
        return outputs[0], outputs[1], list(outputs[4:10]), outputs[2]

    def prefill(
        self, ids: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        cache = self.empty
        scores = []
        cache_before_last = cache
        for index, token_id in enumerate(ids):
            if index == len(ids) - 1:
                cache_before_last = cache
            _, _, cache, token_scores = self._run(
                int(token_id),
                np.asarray([-1], dtype=np.int64),
                cache,
            )
            scores.append(token_scores)
        route = np.asarray(
            [int(np.mean(np.stack(scores), axis=0).argmax())],
            dtype=np.int64,
        )
        logits, route, cache, _ = self._run(
            int(ids[-1]), route, cache_before_last
        )
        return logits, route, cache

    def generate(
        self, prompt: str, *, output_bytes: int
    ) -> dict[str, Any]:
        prompt_ids = self.encode(prompt + "\n")
        logits, route, cache = self.prefill(prompt_ids)
        generated: list[int] = []
        started = time.perf_counter()
        while True:
            token_id = _select_token(
                logits,
                generated,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
            )
            generated.append(token_id)
            output = self.decode(generated)
            payload = output.encode("utf-8")
            if len(payload) >= output_bytes:
                break
            logits, route, cache, _ = self._run(
                token_id, route, cache
            )
            if len(prompt_ids) + len(generated) >= 1024:
                raise ParentCurriculumError(
                    "sealed parent context ended before byte target"
                )
        return {
            "output": output,
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "output_utf8_bytes": len(payload),
            "authoritative_generated_token_ids": generated,
            "authoritative_generated_tokens": len(generated),
            "route": int(route[0]),
            "prompt_tokens": len(prompt_ids),
            "generation_seconds": time.perf_counter() - started,
        }


def build_parent_longform_curriculum(
    *,
    parent_artifact: str | Path,
    source_curriculum_path: str | Path,
    calibration_evidence_path: str | Path,
    curriculum_output_path: str | Path,
    evidence_output_path: str | Path,
    prompt_count: int = 128,
    output_bytes: int = 1024,
    threads: int = 16,
) -> dict[str, Any]:
    """Replace selected train rows with sealed-parent long continuations."""

    parent_artifact = Path(parent_artifact).resolve()
    source_curriculum_path = Path(
        source_curriculum_path
    ).resolve()
    calibration_evidence_path = Path(
        calibration_evidence_path
    ).resolve()
    curriculum_output_path = Path(curriculum_output_path).resolve()
    evidence_output_path = Path(evidence_output_path).resolve()
    if curriculum_output_path.exists() or evidence_output_path.exists():
        raise ParentCurriculumError(
            "parent curriculum outputs are immutable"
        )
    calibration = json.loads(
        calibration_evidence_path.read_text(encoding="utf-8")
    )
    if (
        calibration.get("format") != CALIBRATION_FORMAT
        or prompt_count <= 0
        or prompt_count > len(calibration["records"])
    ):
        raise ParentCurriculumError(
            "train-only calibration evidence is invalid"
        )
    rows = [
        json.loads(line)
        for line in source_curriculum_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [
        row for row in rows
        if row["split"] == "instruction_validation"
    ]
    if len(train_rows) != 2100 or len(validation_rows) != 420:
        raise ParentCurriculumError(
            "locked curriculum depth changed"
        )
    index_by_hash = {
        str(row["prompt_sha256"]): index
        for index, row in enumerate(rows)
        if row["split"] == "train"
    }
    validation_hash_before = hashlib.sha256(
        _canonical_json_bytes(validation_rows)
    ).hexdigest()
    runtime = _ParentRuntime(parent_artifact, threads=threads)
    generated_records = []
    for ordinal, record in enumerate(
        calibration["records"][:prompt_count]
    ):
        source_hash = str(record["source_prompt_sha256"])
        try:
            row_index = index_by_hash[source_hash]
        except KeyError as exc:
            raise ParentCurriculumError(
                "calibration train prompt is absent"
            ) from exc
        transformed_prompt = next(
            (
                str(row["prompt"])
                for row in train_rows
                if str(row["prompt_sha256"]) == source_hash
            ),
            None,
        )
        if transformed_prompt is None:
            raise ParentCurriculumError(
                "source train text is absent"
            )
        suffix = calibration["selection"]["prompt_suffix"]
        transformed_prompt += suffix
        if _sha_text(transformed_prompt) != record[
            "transformed_prompt_sha256"
        ]:
            raise ParentCurriculumError(
                "transformed train prompt hash changed"
            )
        generated = runtime.generate(
            transformed_prompt, output_bytes=output_bytes
        )
        output = generated.pop("output")
        source_row = rows[row_index]
        rows[row_index] = {
            **source_row,
            "id": str(source_row["id"]) + "-parent-longform",
            "prompt": transformed_prompt,
            "prompt_sha256": _sha_text(transformed_prompt),
            "response": output,
            "response_sha256": _sha_text(output),
            "supervision": (
                "sealed LayerCake parent autonomous long-form trajectory"
            ),
            "teacher_terminal_eval_count": None,
            "teacher_tokens": generated[
                "authoritative_generated_tokens"
            ],
        }
        generated_records.append(
            {
                "ordinal": ordinal,
                "source_row_id": str(source_row["id"]),
                "derived_row_id": rows[row_index]["id"],
                "source_prompt_sha256": source_hash,
                "transformed_prompt_sha256": _sha_text(
                    transformed_prompt
                ),
                **generated,
            }
        )
        if (ordinal + 1) % 16 == 0:
            print(
                json.dumps(
                    {
                        "generated": ordinal + 1,
                        "total": prompt_count,
                    }
                ),
                flush=True,
            )
    validation_after = [
        row for row in rows
        if row["split"] == "instruction_validation"
    ]
    validation_hash_after = hashlib.sha256(
        _canonical_json_bytes(validation_after)
    ).hexdigest()
    if validation_hash_after != validation_hash_before:
        raise ParentCurriculumError(
            "instruction-validation rows changed"
        )

    curriculum_output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    curriculum_output_path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    evidence: dict[str, Any] = {
        "format": CURRICULUM_FORMAT,
        "status": "TRAIN_ONLY_PARENT_TRAJECTORIES_GENERATED",
        "sources": {
            "parent_artifact": str(parent_artifact),
            "parent_metadata_sha256": _sha256_file(
                parent_artifact / "metadata.json"
            ),
            "parent_runtime_graph_sha256": runtime.metadata["runtime"][
                "graph_sha256"
            ],
            "parent_checkpoint_sha256": runtime.metadata[
                "source_checkpoint_sha256"
            ],
            "source_curriculum_sha256": _sha256_file(
                source_curriculum_path
            ),
            "calibration_evidence_sha256": calibration[
                "evidence_sha256"
            ],
        },
        "curriculum": {
            "path_at_creation": str(curriculum_output_path),
            "sha256": _sha256_file(curriculum_output_path),
            "bytes": curriculum_output_path.stat().st_size,
            "train_rows": 2100,
            "replaced_train_rows": prompt_count,
            "unchanged_train_rows": 2100 - prompt_count,
            "instruction_validation_rows": 420,
            "instruction_validation_canonical_sha256_before": (
                validation_hash_before
            ),
            "instruction_validation_canonical_sha256_after": (
                validation_hash_after
            ),
            "validation_rows_seen_for_generation": 0,
            "benchmark_rows_seen_for_generation": 0,
            "final_test_rows_seen": 0,
        },
        "generation": {
            "output_target_bytes": output_bytes,
            "threads": threads,
            "records": generated_records,
            "total_generated_tokens": sum(
                row["authoritative_generated_tokens"]
                for row in generated_records
            ),
            "total_generated_utf8_bytes": sum(
                row["output_utf8_bytes"]
                for row in generated_records
            ),
        },
        "imported_information_accounting": {
            "external_source_teacher_tokens": 0,
            "external_source_teacher_bytes": 0,
            "sealed_layercake_parent_generated_tokens": sum(
                row["authoritative_generated_tokens"]
                for row in generated_records
            ),
            "sealed_layercake_parent_generated_bytes": sum(
                row["output_utf8_bytes"]
                for row in generated_records
            ),
            "logits_stored": 0,
            "hidden_activations_stored": 0,
        },
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    _write_json(evidence_output_path, evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--source-curriculum", required=True)
    parser.add_argument("--calibration-evidence", required=True)
    parser.add_argument("--curriculum-output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--prompt-count", type=int, default=128)
    parser.add_argument("--output-bytes", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(argv)
    result = build_parent_longform_curriculum(
        parent_artifact=args.parent_artifact,
        source_curriculum_path=args.source_curriculum,
        calibration_evidence_path=args.calibration_evidence,
        curriculum_output_path=args.curriculum_output,
        evidence_output_path=args.evidence_output,
        prompt_count=args.prompt_count,
        output_bytes=args.output_bytes,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
