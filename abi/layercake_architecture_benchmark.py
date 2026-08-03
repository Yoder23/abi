"""Measure LayerCake graph cost on an identical forced token schedule.

This diagnostic deliberately cannot certify autonomous throughput or quality.
It exists to separate runtime graph cost from a checkpoint's generated
bytes-per-token distribution before an expensive GPU acquisition run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import psutil

from .layercake_host_runtime import (
    NativeHostRuntime,
    _bootstrap_interval,
    _canonical_sha,
    _sha256_file,
)


EVIDENCE_FORMAT = "abi-layercake-fixed-token-architecture-benchmark/1"
COMPARISON_FORMAT = "abi-layercake-fixed-token-architecture-comparison/1"


class ArchitectureBenchmarkError(RuntimeError):
    """Raised when a fixed-token architecture comparison is not exact."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prompt_texts(path: Path) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    prompts = document.get("prompts", [])
    values = {str(row["id"]): str(row["text"]) for row in prompts}
    if len(values) < 100:
        raise ArchitectureBenchmarkError(
            "locked prompt manifest lacks its required depth"
        )
    return values


def _fixed_schedules(
    runtime: NativeHostRuntime,
    *,
    parent_benchmark_path: Path,
    prompt_manifest_path: Path,
    forced_tokens: int,
) -> list[dict[str, Any]]:
    parent = json.loads(parent_benchmark_path.read_text(encoding="utf-8"))
    prompts = _prompt_texts(prompt_manifest_path)
    unique: dict[str, dict[str, Any]] = {}
    for row in parent.get("records", []):
        prompt_id = str(row["prompt_id"])
        if prompt_id in unique:
            continue
        manifest_id = prompt_id.removeprefix("sustained-")
        prompt = prompts.get(manifest_id)
        if prompt is None:
            raise ArchitectureBenchmarkError(
                f"parent benchmark prompt is unbound: {prompt_id}"
            )
        payload = bytes.fromhex(str(row["output_hex"]))
        try:
            output = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArchitectureBenchmarkError(
                "parent fixed-token source is not valid UTF-8"
            ) from exc
        token_ids = runtime.encode(output)
        if len(token_ids) < forced_tokens:
            raise ArchitectureBenchmarkError(
                "parent output lacks the locked forced-token depth"
            )
        unique[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "forced_token_ids": token_ids[:forced_tokens],
            "forced_token_ids_sha256": hashlib.sha256(
                json.dumps(
                    token_ids[:forced_tokens], separators=(",", ":")
                ).encode("ascii")
            ).hexdigest(),
        }
    if len(unique) != 20:
        raise ArchitectureBenchmarkError(
            "parent benchmark must bind exactly 20 distinct prompts"
        )
    return [unique[key] for key in sorted(unique)]


def benchmark_artifact(
    *,
    artifact: str | Path,
    parent_benchmark_path: str | Path,
    prompt_manifest_path: str | Path,
    output_path: str | Path,
    repeats: int,
    forced_tokens: int,
    threads: int,
) -> dict[str, Any]:
    """Run one artifact on the immutable forced-token schedule."""

    if repeats != 20 or forced_tokens != 64 or threads != 14:
        raise ArchitectureBenchmarkError(
            "fixed-token benchmark depth or thread lock changed"
        )
    artifact = Path(artifact).resolve()
    parent_benchmark_path = Path(parent_benchmark_path).resolve()
    prompt_manifest_path = Path(prompt_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ArchitectureBenchmarkError(
            f"fixed-token evidence is immutable: {output_path}"
        )
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    runtime = NativeHostRuntime(artifact, threads=threads)
    schedules = _fixed_schedules(
        runtime,
        parent_benchmark_path=parent_benchmark_path,
        prompt_manifest_path=prompt_manifest_path,
        forced_tokens=forced_tokens,
    )
    work = [
        (prompt_index, trial)
        for trial in range(repeats)
        for prompt_index in range(len(schedules))
    ]
    random.Random(20260802).shuffle(work)
    records: list[dict[str, Any]] = []
    for order, (prompt_index, trial) in enumerate(work):
        schedule = schedules[prompt_index]
        prompt_ids = runtime.encode(schedule["prompt"] + "\n")
        prefill_started = time.perf_counter_ns()
        _, state = runtime.prefill(prompt_ids)
        prefill_completed = time.perf_counter_ns()
        decode_started = prefill_completed
        for token_id in schedule["forced_token_ids"]:
            _, state = runtime.decode_step(int(token_id), state)
        decode_completed = time.perf_counter_ns()
        decode_seconds = (decode_completed - decode_started) / 1.0e9
        expected_cache = len(prompt_ids) + forced_tokens
        cache_lengths = [int(value.shape[2]) for value in state.cache[::2]]
        if any(value != expected_cache for value in cache_lengths):
            raise ArchitectureBenchmarkError(
                "fixed-token decode did not preserve its exact cache"
            )
        records.append(
            {
                "order": order,
                "prompt_id": schedule["prompt_id"],
                "prompt_sha256": schedule["prompt_sha256"],
                "trial": trial + 1,
                "prompt_tokens": len(prompt_ids),
                "forced_tokens": forced_tokens,
                "forced_token_ids_sha256": schedule[
                    "forced_token_ids_sha256"
                ],
                "prefill_seconds": (
                    prefill_completed - prefill_started
                )
                / 1.0e9,
                "decode_seconds": decode_seconds,
                "decode_tokens_per_second": (
                    forced_tokens / max(decode_seconds, 1.0e-12)
                ),
                "cache_lengths": cache_lengths,
                "expected_cache_length": expected_cache,
                "internal_cake_route": int(state.route[0]),
                "canonical_route": runtime.public_route(int(state.route[0])),
            }
        )
        if (order + 1) % 100 == 0:
            print(
                json.dumps(
                    {"benchmarked": order + 1, "total": len(work)},
                    sort_keys=True,
                ),
                flush=True,
            )
    process_memory = process.memory_info()
    rates = [float(row["decode_tokens_per_second"]) for row in records]
    evidence: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "status": "MEASURED_ARCHITECTURE_ONLY_NOT_PRODUCT_EVIDENCE",
        "artifact": str(artifact),
        "artifact_graph_sha256": runtime.metadata["runtime"]["graph_sha256"],
        "artifact_metadata_evidence_sha256": runtime.metadata[
            "evidence_sha256"
        ],
        "parent_benchmark": str(parent_benchmark_path),
        "parent_benchmark_sha256": _sha256_file(parent_benchmark_path),
        "prompt_manifest": str(prompt_manifest_path),
        "prompt_manifest_sha256": _sha256_file(prompt_manifest_path),
        "threads": threads,
        "distinct_prompts": len(schedules),
        "repeats_per_prompt": repeats,
        "observations": len(records),
        "forced_decode_tokens_per_observation": forced_tokens,
        "aggregates": {
            "median_decode_tokens_per_second": statistics.median(rates),
            "mean_decode_tokens_per_second": statistics.fmean(rates),
            "median_prefill_seconds": statistics.median(
                float(row["prefill_seconds"]) for row in records
            ),
            "resident_bytes_before_session": rss_before,
            "resident_bytes_after_benchmark": int(process_memory.rss),
            "peak_process_resident_bytes": int(
                getattr(process_memory, "peak_wset", process_memory.rss)
            ),
            "all_cache_lengths_exact": True,
        },
        "records": records,
        "claim_boundary": (
            "Identical forced GPT-2 IDs isolate graph decode cost. This is "
            "not autonomous bytes/second, quality, or product evidence."
        ),
        "final_test_accessed": False,
        "exact_command": " ".join(sys.argv),
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def compare_artifacts(
    *,
    parent_evidence_path: str | Path,
    candidate_evidence_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Pair fixed-token records and enforce the preregistered overhead gate."""

    parent_evidence_path = Path(parent_evidence_path).resolve()
    candidate_evidence_path = Path(candidate_evidence_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ArchitectureBenchmarkError(
            f"fixed-token comparison is immutable: {output_path}"
        )
    parent = json.loads(parent_evidence_path.read_text(encoding="utf-8"))
    candidate = json.loads(
        candidate_evidence_path.read_text(encoding="utf-8")
    )
    if (
        parent.get("format") != EVIDENCE_FORMAT
        or candidate.get("format") != EVIDENCE_FORMAT
        or parent["threads"] != 14
        or candidate["threads"] != 14
        or parent["distinct_prompts"] != 20
        or candidate["distinct_prompts"] != 20
        or parent["repeats_per_prompt"] != 20
        or candidate["repeats_per_prompt"] != 20
    ):
        raise ArchitectureBenchmarkError(
            "fixed-token comparison inputs differ from the lock"
        )
    parent_rows = {
        (str(row["prompt_id"]), int(row["trial"])): row
        for row in parent["records"]
    }
    candidate_rows = {
        (str(row["prompt_id"]), int(row["trial"])): row
        for row in candidate["records"]
    }
    if parent_rows.keys() != candidate_rows.keys() or len(parent_rows) != 400:
        raise ArchitectureBenchmarkError(
            "fixed-token records do not pair one-to-one"
        )
    paired = []
    prompt_ratios: dict[str, list[float]] = {}
    for key in sorted(parent_rows):
        left = parent_rows[key]
        right = candidate_rows[key]
        if (
            left["forced_token_ids_sha256"]
            != right["forced_token_ids_sha256"]
            or left["prompt_sha256"] != right["prompt_sha256"]
        ):
            raise ArchitectureBenchmarkError(
                "fixed-token paired schedules changed"
            )
        rate_ratio = float(right["decode_tokens_per_second"]) / float(
            left["decode_tokens_per_second"]
        )
        latency_ratio = float(right["decode_seconds"]) / float(
            left["decode_seconds"]
        )
        prompt_ratios.setdefault(key[0], []).append(rate_ratio)
        paired.append(
            {
                "prompt_id": key[0],
                "trial": key[1],
                "candidate_to_parent_token_rate_ratio": rate_ratio,
                "candidate_to_parent_latency_ratio": latency_ratio,
            }
        )
    per_prompt = [statistics.fmean(values) for values in prompt_ratios.values()]
    rate_ci = _bootstrap_interval(per_prompt)
    latency_per_prompt = [
        statistics.fmean(
            row["candidate_to_parent_latency_ratio"]
            for row in paired
            if row["prompt_id"] == prompt_id
        )
        for prompt_id in sorted(prompt_ratios)
    ]
    latency_ci = _bootstrap_interval(latency_per_prompt)
    gates = {
        "candidate_median_token_rate_retains_parent_at_least_95pct": (
            statistics.median(per_prompt) >= 0.95
        ),
        "paired_prompt_token_rate_bootstrap_lower_at_least_95pct": (
            rate_ci[0] >= 0.95
        ),
        "paired_prompt_latency_bootstrap_upper_at_most_1_05": (
            latency_ci[1] <= 1.05
        ),
        "candidate_peak_rss_below_absolute_limit": (
            int(candidate["aggregates"]["peak_process_resident_bytes"])
            < 214_990_848
        ),
        "both_caches_exact": (
            parent["aggregates"]["all_cache_lengths_exact"] is True
            and candidate["aggregates"]["all_cache_lengths_exact"] is True
        ),
    }
    evidence: dict[str, Any] = {
        "format": COMPARISON_FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "parent_evidence": str(parent_evidence_path),
        "parent_evidence_sha256": _sha256_file(parent_evidence_path),
        "candidate_evidence": str(candidate_evidence_path),
        "candidate_evidence_sha256": _sha256_file(candidate_evidence_path),
        "distinct_prompts": 20,
        "repeats_per_prompt": 20,
        "observations_per_artifact": 400,
        "aggregates": {
            "candidate_to_parent_median_prompt_token_rate_ratio": (
                statistics.median(per_prompt)
            ),
            "candidate_to_parent_mean_prompt_token_rate_ratio": (
                statistics.fmean(per_prompt)
            ),
            "paired_prompt_token_rate_ratio_bootstrap_95ci": rate_ci,
            "paired_prompt_latency_ratio_bootstrap_95ci": latency_ci,
            "parent_median_decode_tokens_per_second": parent["aggregates"][
                "median_decode_tokens_per_second"
            ],
            "candidate_median_decode_tokens_per_second": candidate[
                "aggregates"
            ]["median_decode_tokens_per_second"],
            "candidate_peak_process_resident_bytes": candidate[
                "aggregates"
            ]["peak_process_resident_bytes"],
            "gates": gates,
        },
        "paired_records": paired,
        "claim_boundary": (
            "This pass can authorize GPU acquisition only. It cannot replace "
            "the final autonomous byte/character throughput or quality gates."
        ),
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--artifact", required=True)
    run.add_argument("--parent-benchmark", required=True)
    run.add_argument("--prompt-manifest", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--repeats", type=int, default=20)
    run.add_argument("--forced-tokens", type=int, default=64)
    run.add_argument("--threads", type=int, default=14)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--parent-evidence", required=True)
    compare.add_argument("--candidate-evidence", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = benchmark_artifact(
            artifact=args.artifact,
            parent_benchmark_path=args.parent_benchmark,
            prompt_manifest_path=args.prompt_manifest,
            output_path=args.output,
            repeats=args.repeats,
            forced_tokens=args.forced_tokens,
            threads=args.threads,
        )
    else:
        result = compare_artifacts(
            parent_evidence_path=args.parent_evidence,
            candidate_evidence_path=args.candidate_evidence,
            output_path=args.output,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_sha256": result["evidence_sha256"],
                "aggregates": result.get("aggregates"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
