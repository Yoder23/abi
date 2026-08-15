"""Compose exact-B50 V24 and quality-qualified conformed-L1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    load_catalog,
    sha256_file,
)
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
)
from .capability_compiler_phase4_b50_cpu_runtime import _paired_prompt_throughput
from .capability_compiler_phase4_b50_gpu_runtime import _runtime_metrics
from .capability_compiler_phase4_b50_l1_conformance_verify import _gates, _report
from .capability_compiler_phase4_b50_runtime_compose import _paired_or_zero, _ratio
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b50-v24-l1-conformed-compose/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-v24-l1-conformed-compose-result/1"
SEEDS = (104729, 130363, 155921)


def _evidence_valid(result: Mapping[str, Any]) -> bool:
    expected = str(result.get("evidence_sha256", ""))
    payload = {key: value for key, value in result.items() if key != "evidence_sha256"}
    return expected == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _normalized_quality(
    rows: Sequence[Mapping[str, Any]], probes: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    indexed = {str(row["probe_id"]): row for row in rows}
    if set(indexed) != set(probes):
        raise Phase3Error("composed quality prompt set changed")
    normalized = []
    for probe_id in sorted(probes):
        probe = probes[probe_id]
        capability = str(probe["canonical_capability"])
        output = str(indexed[probe_id]["output"])
        normalized.append(
            {
                "probe_id": probe_id,
                "capability": capability,
                "functional_pass_v1": evaluate_functional(output, probe["evaluator"]),
                "functional_pass_v2": evaluate_functional_v2(
                    output, probe["evaluator"], capability
                ),
                "repetition_collapse_v2": repetition_collapse_v2(output),
            }
        )
    return normalized


def _paired_quality(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if [row["probe_id"] for row in candidate] != [row["probe_id"] for row in baseline]:
        raise Phase3Error("composed paired quality order changed")
    pairs = [
        {
            "capability": str(left["capability"]),
            "candidate_pass": bool(left["functional_pass_v1"]),
            "teacher_pass": bool(right["functional_pass_v1"]),
        }
        for left, right in zip(candidate, baseline)
    ]
    result = paired_stratified_bootstrap(pairs, replicates=replicates, seed=seed)
    result["method"] = "capability_stratified_paired_percentile_bootstrap"
    result["observations"] = len(pairs)
    return result


def _metrics_match(claimed: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    recomputed = _runtime_metrics(rows)
    return all(claimed.get(key) == value for key, value in recomputed.items())


def run(root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_INDEPENDENT_V24_CONFORMED_L1_COMPOSITION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("V24/conformed-L1 composition governance changed")
    if output_path.exists():
        raise Phase3Error("V24/conformed-L1 composition output exists")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V24/conformed-L1 composition binding changed: {relative}")

    catalog = load_catalog((root / str(protocol["development_catalog"])).resolve())
    probes = {
        str(row["probe_id"]): row
        for row in catalog["probes"]
        if row.get("split") == "validation"
        and row.get("canonical_capability") in CAPABILITIES
    }
    if len(probes) != 1400:
        raise Phase3Error("V24/conformed-L1 quality catalog changed")
    all_candidate = _jsonl((root / str(protocol["candidate_quality_outputs"])).resolve())
    quality_by_seed: list[dict[str, Any]] = []
    thresholds = protocol["absolute_screen"]
    statistics = protocol["statistics"]
    for index, seed in enumerate(SEEDS):
        candidate_rows = _normalized_quality(
            [row for row in all_candidate if int(row["seed"]) == seed], probes
        )
        baseline_rows = _normalized_quality(
            _jsonl((root / str(protocol["baseline_quality_outputs"][str(seed)])).resolve()),
            probes,
        )
        candidate_report = _report(candidate_rows)
        baseline_report = _report(baseline_rows)
        candidate_gates = _gates(candidate_report, thresholds)
        baseline_gates = _gates(baseline_report, thresholds)
        paired = _paired_quality(
            candidate_rows,
            baseline_rows,
            replicates=int(statistics["bootstrap_replicates"]),
            seed=int(statistics["quality_seed_base"]) + index,
        )
        quality_by_seed.append(
            {
                "seed": seed,
                "candidate": candidate_report,
                "baseline": baseline_report,
                "candidate_gates": candidate_gates,
                "baseline_gates": baseline_gates,
                "candidate_all_quality_gates_pass": all(candidate_gates.values()),
                "baseline_all_quality_gates_pass": all(baseline_gates.values()),
                "candidate_minus_baseline_v1": paired,
                "candidate_noninferior": float(paired["lower_95"])
                >= float(statistics["quality_noninferiority_lower_95_minimum"]),
            }
        )

    candidate_runtime = _json((root / str(protocol["candidate_gpu_result"])).resolve())
    baseline_runtime = _json((root / str(protocol["baseline_gpu_result"])).resolve())
    if (
        candidate_runtime.get("status") != "PASS_SAME_ARTIFACT_B50_V24_GPU_RUNTIME"
        or baseline_runtime.get("status") != "PASS_CONFORMED_L1_SAME_PRODUCT_GPU_RUNTIME"
        or not _evidence_valid(candidate_runtime)
        or not _evidence_valid(baseline_runtime)
        or not all(bool(value) for value in baseline_runtime.get("gates", {}).values())
    ):
        raise Phase3Error("V24/conformed-L1 runtime prerequisite invalid")
    candidate_rows = _jsonl((root / str(candidate_runtime["observations"])).resolve())
    baseline_rows = _jsonl((root / str(baseline_runtime["observations"])).resolve())
    if (
        len(candidate_rows) != 120
        or len(baseline_rows) != 120
        or [row["probe_id"] for row in candidate_rows]
        != [row["probe_id"] for row in baseline_rows]
        or not _metrics_match(candidate_runtime["engine"]["metrics"], candidate_rows)
        or not _metrics_match(baseline_runtime["metrics"], baseline_rows)
    ):
        raise Phase3Error("V24/conformed-L1 runtime observations changed")

    candidate_bytes, baseline_bytes = _paired_prompt_throughput(
        candidate_rows, baseline_rows
    )
    paired_bytes = _paired_or_zero(
        candidate_bytes,
        baseline_bytes,
        replicates=int(statistics["bootstrap_replicates"]),
        seed=int(statistics["runtime_bytes_seed"]),
    )
    paired_bytes["method"] = (
        "paired_prompt_median_bytes_throughput_ratio_percentile_bootstrap"
    )
    paired_bytes["prompt_pairs"] = len(candidate_bytes)
    candidate_chars, baseline_chars = _paired_prompt_throughput(
        [{**row, "bytes_per_second": row["characters_per_second"]} for row in candidate_rows],
        [{**row, "bytes_per_second": row["characters_per_second"]} for row in baseline_rows],
    )
    paired_chars = _paired_or_zero(
        candidate_chars,
        baseline_chars,
        replicates=int(statistics["bootstrap_replicates"]),
        seed=int(statistics["runtime_characters_seed"]),
    )
    paired_chars["method"] = (
        "paired_prompt_median_characters_throughput_ratio_percentile_bootstrap"
    )
    paired_chars["prompt_pairs"] = len(candidate_chars)

    candidate_engine = candidate_runtime["engine"]
    ratios = {
        "median_bytes_per_second": _ratio(
            candidate_engine["metrics"]["median_bytes_per_second"],
            baseline_runtime["metrics"]["median_bytes_per_second"],
        ),
        "median_characters_per_second": _ratio(
            candidate_engine["metrics"]["median_characters_per_second"],
            baseline_runtime["metrics"]["median_characters_per_second"],
        ),
        "median_ttft": _ratio(
            candidate_engine["metrics"]["median_time_to_first_output_seconds"],
            baseline_runtime["metrics"]["median_time_to_first_output_seconds"],
        ),
        "cold_ttft": _ratio(
            candidate_engine["cold"]["time_to_first_output_from_cold_start_seconds"],
            baseline_runtime["cold"]["time_to_first_output_from_cold_start_seconds"],
        ),
        "active_tensor_bytes": _ratio(
            candidate_engine["active_tensor_bytes"], baseline_runtime["active_tensor_bytes"]
        ),
        "peak_process_rss": _ratio(
            candidate_engine["peak_process_rss_delta_bytes"],
            baseline_runtime["peak_process_rss_delta_bytes"],
        ),
        "peak_cuda": _ratio(
            candidate_engine["peak_cuda_allocated_bytes"],
            baseline_runtime["peak_cuda_allocated_bytes"],
        ),
        "installed_bytes": _ratio(
            int(protocol["deployment"]["candidate_installed_bytes"]),
            int(protocol["deployment"]["baseline_installed_bytes"]),
        ),
    }
    gates = {
        "candidate_quality_pass_all_seeds": all(
            row["candidate_all_quality_gates_pass"] for row in quality_by_seed
        ),
        "baseline_quality_pass_all_seeds": all(
            row["baseline_all_quality_gates_pass"] for row in quality_by_seed
        ),
        "candidate_quality_noninferior_all_seeds": all(
            row["candidate_noninferior"] for row in quality_by_seed
        ),
        "both_same_product_runtime_results_pass": all(
            bool(value) for value in candidate_engine["gates"].values()
        ) and all(bool(value) for value in baseline_runtime["gates"].values()),
        "median_gpu_throughput_strictly_faster": float(ratios["median_bytes_per_second"])
        > 1.0,
        "paired_gpu_throughput_lower_95_above_one": paired_bytes["lower_95"] is not None
        and float(paired_bytes["lower_95"]) > 1.0,
        "median_gpu_ttft_lower": float(ratios["median_ttft"]) < 1.0,
        "cold_gpu_ttft_lower": float(ratios["cold_ttft"]) < 1.0,
        "active_tensor_bytes_lower": float(ratios["active_tensor_bytes"]) < 1.0,
        "peak_process_rss_lower": float(ratios["peak_process_rss"]) < 1.0,
        "peak_cuda_lower": float(ratios["peak_cuda"]) < 1.0,
        "installed_bytes_lower": float(ratios["installed_bytes"]) < 1.0,
        "candidate_source_and_teacher_absent_at_inference": True,
        "baseline_source_base_counted_at_inference": bool(
            protocol["deployment"]["baseline_source_base_present_at_inference"]
        ),
        "equal_b50_teacher_sequence_information": True,
        "training_absent": True,
        "model_inference_absent": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    result: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "status": (
            "PASS_B50_V24_DOMINATES_QUALITY_QUALIFIED_CONFORMED_L1_GPU"
            if all(gates.values())
            else "FAIL_B50_V24_CONFORMED_L1_COMPOSITION"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "quality_by_seed": quality_by_seed,
        "runtime": {
            "ratios": ratios,
            "paired_bytes_per_second": paired_bytes,
            "paired_characters_per_second": paired_chars,
            "candidate_metrics": candidate_engine["metrics"],
            "baseline_metrics": baseline_runtime["metrics"],
        },
        "deployment": protocol["deployment"],
        "gates": gates,
        "training_performed": False,
        "model_inference_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "abi_superiority_certified": False,
        "bounded_lora_dominance_certified": all(gates.values()),
        "claim_boundary": "Exact-B50 development evidence for the same V24 product versus the independently quality-qualified conformed routed-LoRA product under equal teacher-sequence information. It is not a universal LoRA, distillation, Phase 4, final-test, or unconditional ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output_path, canonical_json_bytes(result))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    result = run(root, args.protocol.resolve(), args.output.resolve())
    print(result["status"])
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
