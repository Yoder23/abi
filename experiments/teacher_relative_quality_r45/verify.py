"""Fail-closed raw recomputation for the R45 paired teacher comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31.acquire import MODEL_ID, REVISION, SYSTEMS
from experiments.english_sufficiency_r31.cascade_v3 import _runtime_score
from experiments.factual_semantic_r16.public_qualification import _render_chat
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    sha256_file,
    write_json_once,
)
from experiments.teacher_relative_quality_r45 import run_v1 as campaign


class VerificationError(RuntimeError):
    """Raised when R45 cannot be independently recomputed."""


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required JSON missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"required JSON unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"required JSON is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VerificationError(f"required JSONL missing: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"required JSONL unreadable: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise VerificationError(f"required JSONL has a non-object row: {path}")
    return rows


def verify(
    run_dir: Path,
    r44_result_path: Path,
    fixture_path: Path,
    evaluation_path: Path,
    candidate_package: Path,
) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    rows_path = run_dir / "paired_quality.jsonl"
    result = _json(result_path)
    unsigned = dict(result)
    stored_evidence = unsigned.pop("evidence_sha256", None)
    if not isinstance(stored_evidence, str) or evidence_hash(unsigned) != stored_evidence:
        raise VerificationError("R45 evidence digest changed")
    if (
        result.get("format") != "abi-r45-teacher-relative-quality/1"
        or result.get("claim_ceiling")
        != "SAME_PROMPT_BOUNDED_FUNCTIONAL_TEACHER_COMPARISON_NOT_HUMAN_OR_UNRESTRICTED_ENGLISH_PARITY"
        or result.get("full_abi_moonshot") != "OPEN"
    ):
        raise VerificationError("R45 result scope changed")

    frozen = (
        (r44_result_path, campaign.EXPECTED_R44, "r44_result_sha256"),
        (fixture_path, campaign.EXPECTED_FIXTURE, "fixture_sha256"),
        (evaluation_path, campaign.EXPECTED_EVALUATION, "evaluation_sha256"),
        (candidate_package, campaign.EXPECTED_PACKAGE, "candidate_package_sha256"),
    )
    for path, digest, input_name in frozen:
        if not path.is_file() or sha256_file(path) != digest:
            raise VerificationError(f"R45 frozen input changed: {path}")
        if result.get("inputs", {}).get(input_name) != digest:
            raise VerificationError(f"R45 frozen input receipt changed: {input_name}")
    system = SYSTEMS[0]
    if (
        hashlib.sha256(system.encode()).hexdigest() != campaign.EXPECTED_SYSTEM
        or result.get("inputs", {}).get("system_sha256") != campaign.EXPECTED_SYSTEM
    ):
        raise VerificationError("R45 system prompt changed")
    source = result.get("source", {})
    if (
        source.get("model_id") != MODEL_ID
        or source.get("revision") != REVISION
        or source.get("generation") != "greedy"
        or source.get("attempts_per_prompt") != 1
        or source.get("maximum_new_tokens") != campaign.MAXIMUM_NEW_TOKENS
        or not isinstance(source.get("parameters"), int)
        or source["parameters"] <= 0
    ):
        raise VerificationError("R45 source identity or generation contract changed")
    snapshot = Path(str(source.get("snapshot", "")))
    if snapshot.name != REVISION or not (snapshot / "config.json").is_file():
        raise VerificationError("R45 pinned source snapshot is absent")

    artifact = result.get("artifacts", {}).get("paired_quality", {})
    if (
        set(result.get("artifacts", {})) != {"paired_quality"}
        or artifact.get("path") != rows_path.name
        or not rows_path.is_file()
        or artifact.get("sha256") != sha256_file(rows_path)
        or artifact.get("bytes") != rows_path.stat().st_size
    ):
        raise VerificationError("R45 raw paired artifact changed")
    fixture = _jsonl(fixture_path)
    evaluation = _jsonl(evaluation_path)
    rows = _jsonl(rows_path)
    candidates = {
        row.get("record_id"): row
        for row in evaluation
        if row.get("device") == "cpu" and row.get("condition") == "candidate"
    }
    if (
        len(fixture) != 144
        or len(candidates) != 144
        or len(rows) != 144
        or len({row.get("record_id") for row in rows}) != 144
        or {row.get("record_id") for row in fixture} != set(candidates) == {
            row.get("record_id") for row in rows
        }
    ):
        raise VerificationError("R45 paired matrix cardinality or identity changed")

    # Use the already validated content-addressed snapshot path directly. Some
    # Transformers releases perform an API lookup for a model ID even when
    # local_files_only=True, which would make an offline verifier non-hermetic.
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    fixture_by_id = {row["record_id"]: row for row in fixture}
    recomputed = []
    for row in rows:
        record_id = row["record_id"]
        prompt_row = fixture_by_id[record_id]
        candidate = candidates[record_id]["output"]
        teacher = row.get("teacher_output")
        if not isinstance(teacher, str) or not isinstance(candidate, str):
            raise VerificationError(f"R45 output missing: {record_id}")
        teacher.encode("utf-8", errors="strict")
        candidate.encode("utf-8", errors="strict")
        rendered = _render_chat(tokenizer, system, prompt_row["prompt"])
        teacher_score = _runtime_score(prompt_row["prompt"], teacher)[1]
        candidate_score = _runtime_score(prompt_row["prompt"], candidate)[1]
        expected = {
            "record_id": record_id,
            "oracle_task": prompt_row["oracle_task"],
            "prompt_sha256": hashlib.sha256(prompt_row["prompt"].encode()).hexdigest(),
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "input_tokens": len(tokenizer.encode(rendered, add_special_tokens=False)),
            "teacher_output": teacher,
            "teacher_output_sha256": hashlib.sha256(teacher.encode()).hexdigest(),
            # _generate records generated IDs before skip-special-token decode.
            # Greedy rows shorter than the cap therefore contain one terminal
            # EOS ID that is absent from the decoded text; capped rows do not.
            "teacher_output_tokens": (
                len(tokenizer.encode(teacher, add_special_tokens=False))
                + int(
                    len(tokenizer.encode(teacher, add_special_tokens=False))
                    < campaign.MAXIMUM_NEW_TOKENS
                )
            ),
            "teacher_latency_seconds": row.get("teacher_latency_seconds"),
            "teacher_score": teacher_score,
            "candidate_output": candidate,
            "candidate_output_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
            "candidate_score": candidate_score,
            "candidate_improvement": bool(
                candidate_score["functional"] and not teacher_score["functional"]
            ),
            "candidate_regression": bool(
                teacher_score["functional"] and not candidate_score["functional"]
            ),
        }
        if (
            not isinstance(expected["teacher_latency_seconds"], (int, float))
            or expected["teacher_latency_seconds"] <= 0
            or row != expected
        ):
            raise VerificationError(f"R45 paired row changed: {record_id}")
        recomputed.append(expected)

    by_task = {
        task: {
            "rows": sum(row["oracle_task"] == task for row in recomputed),
            "teacher_functional": sum(
                row["oracle_task"] == task and row["teacher_score"]["functional"]
                for row in recomputed
            ),
            "candidate_functional": sum(
                row["oracle_task"] == task and row["candidate_score"]["functional"]
                for row in recomputed
            ),
        }
        for task in TASKS
    }
    metrics = {
        "paired_rows": len(recomputed),
        "teacher_functional": sum(row["teacher_score"]["functional"] for row in recomputed),
        "candidate_functional": sum(row["candidate_score"]["functional"] for row in recomputed),
        "candidate_improvements": sum(row["candidate_improvement"] for row in recomputed),
        "candidate_regressions": sum(row["candidate_regression"] for row in recomputed),
        "teacher_noncollapsed": sum(row["teacher_score"]["noncollapsed"] for row in recomputed),
        "candidate_noncollapsed": sum(row["candidate_score"]["noncollapsed"] for row in recomputed),
        "by_task": by_task,
    }
    accounting = result.get("information_accounting", {})
    exact_accounting = {
        "teacher_calls": len(recomputed),
        "rendered_input_tokens": sum(row["input_tokens"] for row in recomputed),
        "teacher_generated_tokens": sum(row["teacher_output_tokens"] for row in recomputed),
        "raw_prompt_bytes": sum(len(row["prompt"].encode()) for row in fixture),
        "teacher_output_bytes": sum(len(row["teacher_output"].encode()) for row in recomputed),
        "candidate_output_bytes": sum(len(row["candidate_output"].encode()) for row in recomputed),
    }
    if any(accounting.get(key) != value for key, value in exact_accounting.items()):
        raise VerificationError("R45 information accounting changed")
    if (
        accounting.get("candidate_regenerated") is not False
        or accounting.get("candidate_training_steps") != 0
        or accounting.get("external_hardware_used") is not False
        or accounting.get("source_model_deleted_after_capture") is not True
        or any(
            not isinstance(accounting.get(key), (int, float)) or accounting[key] < 0
            for key in (
                "source_load_seconds",
                "source_generation_seconds",
                "elapsed_seconds",
                "peak_cpu_rss_bytes",
                "peak_cuda_memory_bytes",
                "cuda_allocated_after_cleanup",
            )
        )
    ):
        raise VerificationError("R45 runtime accounting is incomplete")
    if result.get("metrics") != metrics:
        raise VerificationError("R45 published metrics differ from raw recomputation")
    if not (
        metrics["candidate_functional"] == 141
        and metrics["candidate_functional"] >= metrics["teacher_functional"]
        and metrics["candidate_improvements"] >= metrics["candidate_regressions"]
        and metrics["candidate_regressions"] <= 3
        and all(value["candidate_functional"] >= 10 for value in by_task.values())
        and by_task["abstention"]["candidate_functional"] == 12
        and metrics["teacher_noncollapsed"] == metrics["candidate_noncollapsed"] == 144
    ):
        raise VerificationError("R45 recomputed scientific gate failed")
    return {
        "format": "abi-r45-strict-verification/1",
        "verdict": "PASS_R45_STRICT_VERIFICATION",
        "result_sha256": sha256_file(result_path),
        "evidence_sha256": stored_evidence,
        "paired_rows_recomputed": len(recomputed),
        "teacher_outputs_retokenized": len(recomputed),
        "candidate_outputs_bound_to_r44": len(recomputed),
        "teacher_functional": metrics["teacher_functional"],
        "candidate_functional": metrics["candidate_functional"],
        "full_abi_moonshot": "OPEN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--r44-result", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(
        args.run_dir.resolve(),
        args.r44_result.resolve(),
        args.fixture.resolve(),
        args.evaluation.resolve(),
        args.candidate_package.resolve(),
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
