"""Bounded train-only logit diagnostic for one native runtime derivation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .layercake_host_preservation import _load_general_rows
from .layercake_host_runtime import (
    NativeHostRuntime,
    _canonical_sha,
    _sha256_file,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def compare_train_logits(
    *,
    source_artifact: str | Path,
    candidate_artifact: str | Path,
    curriculum_path: str | Path,
    output_path: str | Path,
    rows: int = 100,
    threads: int = 16,
) -> dict[str, Any]:
    source_artifact = Path(source_artifact).resolve()
    candidate_artifact = Path(candidate_artifact).resolve()
    curriculum_path = Path(curriculum_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise RuntimeError(
            f"quantization diagnostic is immutable: {output_path}"
        )
    train_rows = _load_general_rows(curriculum_path, split="train")
    selected = train_rows[:rows]
    if len(selected) != rows:
        raise RuntimeError("train-only diagnostic depth changed")
    source = NativeHostRuntime(source_artifact, threads=threads)
    candidate = NativeHostRuntime(candidate_artifact, threads=threads)
    observations = []
    for row in selected:
        prompt = str(row["prompt"])
        source_logits, source_state = source.prefill(
            source.encode(prompt + "\n")
        )
        candidate_logits, candidate_state = candidate.prefill(
            candidate.encode(prompt + "\n")
        )
        if not np.array_equal(
            source_state.output_token_ids,
            candidate_state.output_token_ids,
        ):
            raise RuntimeError("diagnostic active token maps differ")
        token_ids = source_state.output_token_ids
        source_local = int(source_logits[0].argmax())
        candidate_local = int(candidate_logits[0].argmax())
        source_top = (
            source_local
            if token_ids is None
            else int(token_ids[source_local])
        )
        candidate_top = (
            candidate_local
            if token_ids is None
            else int(token_ids[candidate_local])
        )
        difference = np.abs(
            source_logits.astype(np.float64)
            - candidate_logits.astype(np.float64)
        )
        observations.append(
            {
                "row_id": row["id"],
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "source_route": int(source_state.route[0]),
                "candidate_route": int(candidate_state.route[0]),
                "source_top_token_id": source_top,
                "candidate_top_token_id": candidate_top,
                "top1_equal": source_top == candidate_top,
                "maximum_absolute_logit_difference": float(
                    difference.max()
                ),
                "mean_absolute_logit_difference": float(
                    difference.mean()
                ),
            }
        )
    top1_agreement = sum(
        row["top1_equal"] for row in observations
    ) / len(observations)
    route_agreement = sum(
        row["source_route"] == row["candidate_route"]
        for row in observations
    ) / len(observations)
    evidence: dict[str, Any] = {
        "format": "abi-layercake-train-only-quantization-diagnostic/1",
        "status": (
            "PASS"
            if top1_agreement >= 0.99 and route_agreement == 1.0
            else "FAIL"
        ),
        "split": "train",
        "row_selection": "first N immutable train rows in file order",
        "observation_count": len(observations),
        "source_artifact_metadata_sha256": source.metadata[
            "evidence_sha256"
        ],
        "candidate_artifact_metadata_sha256": candidate.metadata[
            "evidence_sha256"
        ],
        "curriculum_sha256": _sha256_file(curriculum_path),
        "runtime_runner_sha256": _sha256_file(
            Path(__file__).with_name("layercake_host_runtime.py")
        ),
        "top1_agreement": top1_agreement,
        "route_agreement": route_agreement,
        "maximum_absolute_logit_difference": max(
            row["maximum_absolute_logit_difference"]
            for row in observations
        ),
        "mean_of_mean_absolute_logit_differences": float(
            np.mean(
                [
                    row["mean_absolute_logit_difference"]
                    for row in observations
                ]
            )
        ),
        "observations": observations,
        "claim_boundary": (
            "Train-only first-decision diagnostic. It is neither held-out "
            "functional certification nor autonomous-generation evidence."
        ),
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(argv)
    result = compare_train_logits(
        source_artifact=args.source_artifact,
        candidate_artifact=args.candidate_artifact,
        curriculum_path=args.curriculum,
        output_path=args.output,
        rows=args.rows,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "observation_count": result["observation_count"],
                "top1_agreement": result["top1_agreement"],
                "route_agreement": result["route_agreement"],
                "maximum_absolute_logit_difference": result[
                    "maximum_absolute_logit_difference"
                ],
                "evidence_sha256": result["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
