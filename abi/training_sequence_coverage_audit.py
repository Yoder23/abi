"""Audit how much immutable teacher supervision a training ceiling truncates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tokenizers import Tokenizer

from .layercake_host import _canonical_json_bytes, _sha256_file
from .layercake_host_v3 import load_english_training_rows


class TrainingSequenceCoverageAuditError(RuntimeError):
    """Raised when sequence-coverage evidence cannot be bound exactly."""


def _summarize_lengths(
    rows: Sequence[Mapping[str, Any]],
    *,
    sequence_ceiling: int,
) -> dict[str, Any]:
    if sequence_ceiling <= 0 or not rows:
        raise TrainingSequenceCoverageAuditError(
            "a positive ceiling and non-empty rows are required"
        )
    by_capability: dict[str, dict[str, int]] = {}
    total_lengths: list[int] = []
    prompt_at_or_above_ceiling = 0
    truncated_rows = 0
    omitted_teacher_targets = 0
    maximum_row: Mapping[str, Any] | None = None
    for row in rows:
        prompt_tokens = int(row["prompt_tokens"])
        response_tokens = int(row["response_tokens"])
        full_tokens = prompt_tokens + response_tokens
        omitted = max(0, full_tokens - sequence_ceiling)
        capability = str(row["capability"])
        summary = by_capability.setdefault(
            capability,
            {
                "rows": 0,
                "truncated_rows": 0,
                "omitted_teacher_target_tokens": 0,
            },
        )
        summary["rows"] += 1
        summary["truncated_rows"] += int(omitted > 0)
        summary["omitted_teacher_target_tokens"] += omitted
        prompt_at_or_above_ceiling += int(
            prompt_tokens >= sequence_ceiling
        )
        truncated_rows += int(omitted > 0)
        omitted_teacher_targets += omitted
        total_lengths.append(full_tokens)
        if (
            maximum_row is None
            or full_tokens > int(maximum_row["full_sequence_tokens"])
        ):
            maximum_row = {
                "record_id": str(row["record_id"]),
                "capability": capability,
                "prompt_tokens": prompt_tokens,
                "response_tokens_including_eos": response_tokens,
                "full_sequence_tokens": full_tokens,
                "omitted_teacher_target_tokens": omitted,
            }
    ordered = sorted(total_lengths)

    def quantile(fraction: float) -> int:
        index = min(
            len(ordered) - 1,
            int(fraction * (len(ordered) - 1)),
        )
        return ordered[index]

    return {
        "rows": len(rows),
        "sequence_ceiling": sequence_ceiling,
        "prompt_at_or_above_ceiling": prompt_at_or_above_ceiling,
        "full_sequence_above_ceiling": truncated_rows,
        "full_sequence_above_ceiling_rate": (
            truncated_rows / len(rows)
        ),
        "omitted_teacher_target_tokens": omitted_teacher_targets,
        "full_sequence_token_quantiles": {
            "p50": quantile(0.50),
            "p90": quantile(0.90),
            "p95": quantile(0.95),
            "p99": quantile(0.99),
            "maximum": ordered[-1],
        },
        "by_capability": dict(sorted(by_capability.items())),
        "maximum_row": maximum_row,
    }


def audit_training_sequence_coverage(
    *,
    bundle_path: str | Path,
    tokenizer_path: str | Path,
    budget_index: int,
    sequence_ceiling: int,
    successor_ceiling: int,
    output_path: str | Path,
) -> dict[str, Any]:
    bundle_path = Path(bundle_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise TrainingSequenceCoverageAuditError(
            f"coverage evidence is immutable: {output_path}"
        )
    if successor_ceiling < sequence_ceiling:
        raise TrainingSequenceCoverageAuditError(
            "successor ceiling may not be lower than the audited ceiling"
        )
    rows, budget, _bundle = load_english_training_rows(
        bundle_path,
        budget_index=budget_index,
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    lengths = []
    for row in rows:
        prompt_tokens = len(
            tokenizer.encode(str(row["prompt"]) + "\n").ids
        )
        response_tokens = len(
            tokenizer.encode(str(row["response"])).ids
        ) + 1
        lengths.append(
            {
                "record_id": str(row["record_id"]),
                "capability": str(row["capability"]),
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
            }
        )
    current = _summarize_lengths(
        lengths,
        sequence_ceiling=sequence_ceiling,
    )
    successor = _summarize_lengths(
        lengths,
        sequence_ceiling=successor_ceiling,
    )
    evidence: dict[str, Any] = {
        "schema_version": "abi-training-sequence-coverage-audit/1",
        "status": "MEASURED_TRAINING_SUPERVISION_TRUNCATION",
        "bundle": {
            "path_at_audit": str(bundle_path),
            "sha256": _sha256_file(bundle_path),
            "budget_index": budget_index,
            "budget_id": budget["budget_id"],
            "records": len(rows),
        },
        "tokenizer": {
            "path_at_audit": str(tokenizer_path),
            "sha256": _sha256_file(tokenizer_path),
            "counter": "authoritative_layercake_tokenizer_plus_one_eos",
        },
        "current": current,
        "successor": successor,
        "runner_sha256": _sha256_file(Path(__file__)),
        "interpretation": (
            "The current ceiling discards target suffix tokens after "
            "teacher forcing. The successor ceiling changes acquisition "
            "coverage only; it does not change deployed context length, "
            "parameters, graph topology, or runtime execution."
        ),
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--budget-index", type=int, required=True)
    parser.add_argument("--sequence-ceiling", type=int, required=True)
    parser.add_argument("--successor-ceiling", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    evidence = audit_training_sequence_coverage(
        bundle_path=args.bundle,
        tokenizer_path=args.tokenizer,
        budget_index=args.budget_index,
        sequence_ceiling=args.sequence_ceiling,
        successor_ceiling=args.successor_ceiling,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "current": evidence["current"],
                "successor": evidence["successor"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
