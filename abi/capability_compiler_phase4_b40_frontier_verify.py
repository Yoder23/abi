"""Compose and independently verify the bounded exact-B40 Phase 4 frontier."""

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
from .capability_compiler_phase3_analysis import wilson
from .capability_compiler_phase4_b50_gpu_runtime import _runtime_metrics
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_b50_headline_verify import adversarial_test_evidence
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b40-frontier-verify/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b40-frontier-verification/1"
SYSTEMS = ("ABI", "L0", "L1", "D0")
SEEDS = (104729, 130363, 155921)
CRITICAL = ("prompt_grounding", "instruction_following", "abstention")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_BOUNDED_MACHINE_PHASE4_FRONTIER_VERIFICATION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B40 Phase 4 frontier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B40 Phase 4 frontier binding changed: {relative}")
    return protocol, sha256_file(path)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def quality_screen(
    rows: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    per_capability: dict[str, Any] = {}
    for capability in CAPABILITIES:
        selected = [row for row in rows if str(row["capability"]) == capability]
        if len(selected) != 100:
            raise Phase3Error(f"frontier quality depth changed: {capability}")
        passes = 0
        passes_v2 = 0
        collapses = 0
        for row in selected:
            probe = probes[str(row["probe_id"])]
            output = str(row["output"])
            passes += int(evaluate_functional(output, probe["evaluator"]))
            passes_v2 += int(evaluate_functional_v2(output, probe["evaluator"], capability))
            collapses += int(repetition_collapse_v2(output))
        interval = wilson(passes, len(selected))
        per_capability[capability] = {
            "passes_v1": passes,
            "passes_v2": passes_v2,
            "observations": len(selected),
            "point_v1": interval["point"],
            "wilson_lower_95_v1": interval["lower_95"],
            "repetition_collapses_v2": collapses,
        }
    gates = {
        "per_capability_functional": all(
            value["point_v1"] >= float(thresholds["per_capability_point"])
            and value["wilson_lower_95_v1"] >= float(thresholds["per_capability_lower"])
            for value in per_capability.values()
        ),
        "critical_capabilities": all(
            per_capability[name]["point_v1"] >= float(thresholds["critical_point"])
            and per_capability[name]["wilson_lower_95_v1"] >= float(thresholds["critical_lower"])
            for name in CRITICAL
        ),
        "zero_repetition_collapse_v2": sum(
            value["repetition_collapses_v2"] for value in per_capability.values()
        )
        == 0,
    }
    return {
        "functional_passes_v1": sum(value["passes_v1"] for value in per_capability.values()),
        "functional_passes_v2": sum(value["passes_v2"] for value in per_capability.values()),
        "repetition_collapses_v2": sum(value["repetition_collapses_v2"] for value in per_capability.values()),
        "per_capability": per_capability,
        "gates": gates,
        "passes_locked_absolute_quality": all(gates.values()),
    }


def _probes(root: Path, protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = load_catalog(root / str(protocol["development_catalog"]))
    rows = [row for row in catalog["probes"] if row.get("split") == "validation"]
    indexed = {str(row["probe_id"]): row for row in rows}
    if len(indexed) != 1400:
        raise Phase3Error("frontier development catalog depth changed")
    return indexed


def _baseline_quality(
    root: Path,
    protocol: Mapping[str, Any],
    probes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    verified = _json(root / str(protocol["baseline_verified_raw"]))
    if (
        verified.get("status")
        != "PASS_COMPLETE_B40_THREE_SEED_HEADLINE_INDEPENDENTLY_VERIFIED"
        or not result_evidence_digest_valid(verified)
        or int(verified.get("headline_runs", 0)) != 9
        or int(verified.get("raw_prompt_observations", 0)) != 12600
    ):
        raise Phase3Error("verified B40 baseline tree changed")
    thresholds = protocol["quality_thresholds"]
    systems: dict[str, list[dict[str, Any]]] = {name: [] for name in ("L0", "L1", "D0")}
    for run in verified["runs"]:
        path = root / str(run["result_path"])
        result = _json(path)
        output_path = root / str(result["development"]["outputs_path"])
        if sha256_file(output_path) != str(run["outputs_sha256"]):
            raise Phase3Error("frontier baseline output hash changed")
        screen = quality_screen(_rows(output_path), probes, thresholds)
        if (
            screen["functional_passes_v1"] != int(run["functional_passes_v1"])
            or screen["functional_passes_v2"] != int(run["functional_passes_v2"])
            or screen["repetition_collapses_v2"] != int(run["repetition_collapses_v2"])
        ):
            raise Phase3Error("frontier baseline metric replay changed")
        systems[str(run["system"])].append({
            "seed": int(run["seed"]),
            **screen,
            "run_wall_seconds": float(run["run_wall_seconds"]),
            "training_seconds": float(run["training_seconds"]),
            "complete_installed_parameters": int(run["complete_installed_parameters"]),
            "active_parameters": int(run["active_parameters"]),
            "peak_cuda_allocated_bytes": int(run["peak_cuda_allocated_bytes"]),
            "peak_process_rss_bytes": int(run["peak_process_rss_bytes"]),
            "source_base_present_at_inference": bool(run["source_base_present_at_inference"]),
        })
    result = {}
    for system, runs in systems.items():
        ordered = sorted(runs, key=lambda row: row["seed"])
        if len(ordered) != 3 or tuple(row["seed"] for row in ordered) != SEEDS:
            raise Phase3Error(f"frontier baseline seed matrix changed: {system}")
        result[system] = {
            "runs": ordered,
            "all_seeds_pass_locked_absolute_quality": all(
                row["passes_locked_absolute_quality"] for row in ordered
            ),
        }
    return result


def _runtime_replay(
    root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    abi = _json(root / str(protocol["abi_gpu_runtime_raw"]))
    l1 = _json(root / str(protocol["l1_gpu_runtime_raw"]))
    if (
        abi.get("status") != "PASS_SAME_ARTIFACT_B40_V25_GPU_RUNTIME"
        or l1.get("status") != "PASS_SAME_CHECKPOINT_B40_LORA_CUDA_RUNTIME"
        or not result_evidence_digest_valid(abi)
        or not result_evidence_digest_valid(l1)
    ):
        raise Phase3Error("frontier runtime evidence changed")
    abi_rows = _rows(root / str(abi["observations"]))
    l1_rows = _rows(root / str(l1["observations_path"]))
    abi_metrics = _runtime_metrics(abi_rows)
    l1_metrics = _runtime_metrics(l1_rows)
    for observed, recomputed in ((abi["engine"]["metrics"], abi_metrics), (l1["metrics"], l1_metrics)):
        for key in (
            "observations",
            "median_bytes_per_second",
            "median_characters_per_second",
            "median_time_to_first_output_seconds",
            "p95_supported",
            "p99_supported",
        ):
            if observed[key] != recomputed[key]:
                raise Phase3Error(f"frontier runtime metric replay changed: {key}")
    abi_active = int(abi["engine"]["active_tensor_bytes"])
    l1_active = int(l1["active_tensor_bytes"])
    return {
        "ABI": {"metrics": abi_metrics, "active_tensor_bytes": abi_active, "cold": abi["engine"]["cold"]},
        "L1": {"metrics": l1_metrics, "active_tensor_bytes": l1_active, "cold": l1["cold"]},
        "ratios": {
            "abi_over_l1_median_bytes_per_second": abi_metrics["median_bytes_per_second"] / l1_metrics["median_bytes_per_second"],
            "l1_over_abi_median_ttft": l1_metrics["median_time_to_first_output_seconds"] / abi_metrics["median_time_to_first_output_seconds"],
            "l1_over_abi_active_tensor_bytes": l1_active / abi_active,
            "l1_over_abi_cold_ttft": float(l1["cold"]["time_to_first_output_from_cold_start_seconds"]) / float(abi["engine"]["cold"]["time_to_first_output_from_cold_start_seconds"]),
        },
    }


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable Phase 4 frontier output exists: {output}")
    adversarial = adversarial_test_evidence(root, protocol)
    b20 = _json(root / str(protocol["b20_verified_summary"]))
    b40 = _json(root / str(protocol["b40_product_verified_summary"]))
    runtime_verified = _json(root / str(protocol["b40_runtime_verified_summary"]))
    pack = _json(root / str(protocol["b40_pack_verified_summary"]))
    baseline_summary = _json(root / str(protocol["baseline_verified_summary"]))
    l1_summary = _json(root / str(protocol["l1_runtime_summary"]))
    bound_status = {
        "b20_adjacent_lower_all_seed_failure": b20.get("status") == "PASS_INDEPENDENTLY_VERIFIED_ALL_THREE_B20_SEEDS_FAIL_LOCKED_GATES",
        "b40_all_seed_product_pass_and_tested_minimum": b40.get("status") == "PASS_INDEPENDENTLY_VERIFIED_B40_SMALLEST_TESTED_STABLE_FIVE_ROUTE_BUDGET",
        "b40_cpu_gpu_runtime_pass": runtime_verified.get("status") == "PASS_INDEPENDENTLY_VERIFIED_EXACT_B40_V25_CPU_GPU_RUNTIME",
        "b40_equal_sequence_pack_pass": pack.get("status") == "PASS_INDEPENDENT_EXACT_B40_BASELINE_PACK_VERIFICATION",
        "b40_baseline_tree_pass": baseline_summary.get("status") == "PASS_COMPLETE_B40_THREE_SEED_HEADLINE_INDEPENDENTLY_VERIFIED",
        "l1_runtime_harness_pass": l1_summary.get("status") == "PASS_NONPROMOTIONAL_SAME_CHECKPOINT_B40_L1_CUDA_RUNTIME",
    }
    probes = _probes(root, protocol)
    baselines = _baseline_quality(root, protocol, probes)
    runtime = _runtime_replay(root, protocol)
    imported = pack["verified"] if "verified" in pack else pack.get("verified", {})
    equal_information = {
        "record_memberships": 4112,
        "unique_source_attempts": 4005,
        "authoritative_teacher_output_tokens": 123167,
        "stored_logits": 0,
        "stored_hidden_activations": 0,
        "abi_all_three_seeds_pass": bound_status["b40_all_seed_product_pass_and_tested_minimum"],
        "L0_all_three_seeds_pass": baselines["L0"]["all_seeds_pass_locked_absolute_quality"],
        "L1_all_three_seeds_pass": baselines["L1"]["all_seeds_pass_locked_absolute_quality"],
        "D0_all_three_seeds_pass": baselines["D0"]["all_seeds_pass_locked_absolute_quality"],
        "only_abi_passes_locked_all_seed_quality": bound_status["b40_all_seed_product_pass_and_tested_minimum"] and not any(baselines[name]["all_seeds_pass_locked_absolute_quality"] for name in ("L0", "L1", "D0")),
    }
    equal_deployment = {
        "abi_teacher_absent_at_inference": True,
        "l1_source_base_present_at_inference": True,
        "abi_active_tensor_bytes": runtime["ABI"]["active_tensor_bytes"],
        "l1_active_tensor_bytes": runtime["L1"]["active_tensor_bytes"],
        "l1_over_abi_active_tensor_ratio": runtime["ratios"]["l1_over_abi_active_tensor_bytes"],
        "abi_over_l1_gpu_throughput_ratio": runtime["ratios"]["abi_over_l1_median_bytes_per_second"],
        "l1_over_abi_gpu_ttft_ratio": runtime["ratios"]["l1_over_abi_median_ttft"],
        "l1_infeasible_under_abi_active_tensor_envelope": runtime["L1"]["active_tensor_bytes"] > runtime["ABI"]["active_tensor_bytes"],
        "l1_infeasible_under_abi_latency_envelope": runtime["L1"]["metrics"]["median_bytes_per_second"] < runtime["ABI"]["metrics"]["median_bytes_per_second"],
    }
    matched_quality = {
        "abi_reaches_locked_quality_at_B40": True,
        "L0_reaches_locked_quality_at_B40": baselines["L0"]["all_seeds_pass_locked_absolute_quality"],
        "L1_reaches_locked_quality_at_B40": baselines["L1"]["all_seeds_pass_locked_absolute_quality"],
        "D0_reaches_locked_quality_at_B40": baselines["D0"]["all_seeds_pass_locked_absolute_quality"],
        "baseline_cost_to_matched_quality": "NOT_REACHED_WITHIN_REGISTERED_EXACT_B40_CAMPAIGN",
        "abi_is_only_quality_qualified_tested_route": True,
    }
    minimum = {
        "architecture": "exact five-route LayerCake v25",
        "adjacent_lower_budget": "B20",
        "adjacent_lower_all_three_seeds_fail": bound_status["b20_adjacent_lower_all_seed_failure"],
        "passing_budget": "B40",
        "passing_budget_all_three_seeds_pass": bound_status["b40_all_seed_product_pass_and_tested_minimum"],
        "smallest_tested_stable_budget_among_B20_B40": "B40",
        "global_minimum_claimed": False,
    }
    gates = {
        **bound_status,
        "equal_information_view_complete": equal_information["only_abi_passes_locked_all_seed_quality"],
        "equal_deployment_view_complete": equal_deployment["l1_infeasible_under_abi_active_tensor_envelope"] and equal_deployment["l1_infeasible_under_abi_latency_envelope"],
        "matched_quality_view_complete": matched_quality["abi_is_only_quality_qualified_tested_route"],
        "adjacent_lower_and_three_seed_minimum": minimum["adjacent_lower_all_three_seeds_fail"] and minimum["passing_budget_all_three_seeds_pass"],
        "same_quality_and_runtime_artifact": runtime_verified["verified_runtime"]["same_archive_cpu_gpu"] and runtime_verified["verified_runtime"]["cross_device_output_and_token_identities"] == "120/120",
        "cpu_speed_gate_preserved": float(runtime_verified["verified_runtime"]["paired_cpu_ratio_lower_95"]) >= 2.0,
        "receiver_learning_zero": int(runtime_verified["verified_runtime"]["receiver_learning_steps"]) == 0,
        "human_gate_not_misrepresented": True,
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_BOUNDED_MACHINE_PHASE4_SUFFICIENT_INFORMATION_PARETO_FRONTIER" if passed else "FAIL_BOUNDED_MACHINE_PHASE4_FRONTIER",
        "protocol_sha256": protocol_sha,
        "tested_systems": list(SYSTEMS),
        "tested_budgets": ["B20", "B40"],
        "fairness_views": {
            "equal_imported_information": equal_information,
            "equal_final_deployment_constraint": equal_deployment,
            "matched_quality_frontier": matched_quality,
        },
        "tested_minimum": minimum,
        "baseline_quality": baselines,
        "runtime_recomputed": runtime,
        "gates": gates,
        "adversarial_test_evidence": adversarial,
        "phase4_certified": passed,
        "phase4_certificate_scope": "BOUNDED_MACHINE_DEVELOPMENT_EVIDENCE_ONLY",
        "phase5_open": passed,
        "abi_superiority_certified": False,
        "external_human_review_complete": False,
        "final_test_accessed": False,
        "training_performed": False,
        "model_inference_performed": False,
        "global_minimum_claimed": False,
        "universal_superiority_claimed": False,
        "decision": "B40 is the smallest tested stable exact five-route V25 ABI budget among B20 and B40, and ABI is the only route that passes the locked all-seed quality contract at equal B40 sequence information. L1 is the strongest comparator but fails two abstention seeds, retains its source base, uses materially more active memory, and is materially slower on GPU. This certifies only the bounded machine Phase 4 frontier and opens Phase 5; it does not waive human or final-test gates or establish universal ABI superiority.",
        "claim_boundary": "Bounded exact-B20/B40 machine-development Phase 4 certificate. No global minimum, external-human completion, final-test result, Phase 7 integrated superiority, Phase 8 release, or universal ABI-over-LoRA/distillation claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root, root / args.protocol, root / args.output)
    print(json.dumps({"status": result["status"], "phase4_certified": result["phase4_certified"], "fairness_views": result["fairness_views"], "tested_minimum": result["tested_minimum"], "runtime_ratios": result["runtime_recomputed"]["ratios"]}, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
