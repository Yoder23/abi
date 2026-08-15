"""Compare verified B50 ABI and matched baselines without new inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
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
from .capability_compiler_phase2_prepare import _verified_snapshot
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
    wilson,
)
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b50-matched-quality/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-matched-quality-result/1"
SYSTEMS = ("L0", "L1", "D0", "D1", "D2")
SEEDS = (104729, 130363, 155921)
CRITICAL_CAPABILITIES = ("prompt_grounding", "instruction_following", "abstention")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_MATCHED_B50_COMPARISON"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("runtime_profiling_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("matched B50 quality governance changed")
    if tuple(int(seed) for seed in protocol.get("seeds", ())) != SEEDS:
        raise Phase3Error("matched B50 paired seeds changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"matched B50 quality binding changed: {relative}")
    return protocol, sha256_file(path)


def _probes(root: Path, protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = load_catalog((root / str(protocol["development_catalog"])).resolve())
    rows = {
        str(row["probe_id"]): row
        for row in catalog["probes"]
        if row.get("split") == "validation"
        and row.get("canonical_capability") in CAPABILITIES
    }
    if len(rows) != 1400 or any(
        sum(row["canonical_capability"] == capability for row in rows.values()) != 100
        for capability in CAPABILITIES
    ):
        raise Phase3Error("matched B50 development suite changed")
    return rows


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _normalized_rows(
    path: Path,
    probes: Mapping[str, Mapping[str, Any]],
    *,
    candidate: bool,
) -> list[dict[str, Any]]:
    source = _jsonl(path)
    if len(source) != 1400 or {str(row.get("probe_id")) for row in source} != set(probes):
        raise Phase3Error(f"matched B50 raw prompt set changed: {path}")
    output: list[dict[str, Any]] = []
    for row in source:
        probe_id = str(row["probe_id"])
        probe = probes[probe_id]
        capability = str(probe["canonical_capability"])
        text = str(row["output"])
        v1 = evaluate_functional(text, probe["evaluator"])
        v2 = evaluate_functional_v2(text, probe["evaluator"], capability)
        collapse = repetition_collapse_v2(text)
        keys = (
            ("functional_pass_v1", "functional_pass_v2", "repetition_collapse_v2")
            if candidate
            else ("functional_pass", "functional_pass_v2", "repetition_collapse_v2")
        )
        if (
            row.get("capability") != capability
            or row.get(keys[0]) != v1
            or row.get(keys[1]) != v2
            or row.get(keys[2]) != collapse
        ):
            raise Phase3Error(f"matched B50 raw metric changed: {path}/{probe_id}")
        output.append(
            {
                "probe_id": probe_id,
                "capability": capability,
                "functional_pass_v1": v1,
                "functional_pass_v2": v2,
                "repetition_collapse_v2": collapse,
            }
        )
    return sorted(output, key=lambda row: row["probe_id"])


def evaluation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per = {}
    for capability in CAPABILITIES:
        subset = [row for row in rows if row["capability"] == capability]
        passed = sum(bool(row["functional_pass_v1"]) for row in subset)
        per[capability] = {
            "observations": len(subset),
            "functional_passes_v1": passed,
            "functional_passes_v2": sum(
                bool(row["functional_pass_v2"]) for row in subset
            ),
            "repetition_collapses_v2": sum(
                bool(row["repetition_collapse_v2"]) for row in subset
            ),
            "wilson_v1": wilson(passed, len(subset)),
        }
    return {
        "observations": len(rows),
        "functional_passes_v1": sum(bool(row["functional_pass_v1"]) for row in rows),
        "functional_passes_v2": sum(bool(row["functional_pass_v2"]) for row in rows),
        "repetition_collapses_v2": sum(
            bool(row["repetition_collapse_v2"]) for row in rows
        ),
        "per_capability": per,
    }


def absolute_gates(
    report: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    per = report["per_capability"]
    return {
        "per_capability_functional": all(
            float(value["wilson_v1"]["point"])
            >= float(thresholds["per_capability_functional_point_estimate_minimum"])
            and float(value["wilson_v1"]["lower_95"])
            >= float(thresholds["per_capability_functional_wilson_lower_minimum"])
            for value in per.values()
        ),
        "critical_capabilities": all(
            float(per[name]["wilson_v1"]["point"])
            >= float(thresholds["critical_point_minimum"])
            and float(per[name]["wilson_v1"]["lower_95"])
            >= float(thresholds["critical_wilson_lower_minimum"])
            for name in CRITICAL_CAPABILITIES
        ),
        "zero_repetition_collapse": int(report["repetition_collapses_v2"])
        <= int(thresholds["repetition_collapse_v2_count_maximum"]),
    }


def paired_quality(
    candidate: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if [row["probe_id"] for row in candidate] != [row["probe_id"] for row in comparator]:
        raise Phase3Error("matched B50 paired prompt order changed")
    pairs = [
        {
            "capability": str(left["capability"]),
            "candidate_pass": bool(left["functional_pass_v1"]),
            "teacher_pass": bool(right["functional_pass_v1"]),
        }
        for left, right in zip(candidate, comparator)
    ]
    return paired_stratified_bootstrap(pairs, replicates=replicates, seed=seed)


def _teacher_rows(
    root: Path,
    protocol: Mapping[str, Any],
    probes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _jsonl((root / str(protocol["teacher_reference"])).resolve())
    indexed = {str(row["probe_id"]): row for row in rows}
    if set(indexed) != set(probes):
        raise Phase3Error("matched B50 teacher reference changed")
    return [
        {
            "probe_id": probe_id,
            "capability": str(probes[probe_id]["canonical_capability"]),
            "functional_pass_v1": evaluate_functional(
                str(indexed[probe_id]["output"]), probes[probe_id]["evaluator"]
            ),
            "functional_pass_v2": evaluate_functional_v2(
                str(indexed[probe_id]["output"]),
                probes[probe_id]["evaluator"],
                str(probes[probe_id]["canonical_capability"]),
            ),
            "repetition_collapse_v2": repetition_collapse_v2(
                str(indexed[probe_id]["output"])
            ),
        }
        for probe_id in sorted(probes)
    ]


def _candidate_cost(
    root: Path,
    lineage_path: Path,
    *,
    expected_seed: int,
    final_parameters: int,
    interrupted_attempt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lineage = _json(lineage_path)
    if (
        int(lineage.get("seed", -1)) != expected_seed
        or lineage.get("budget", {}).get("id") != "B50"
        or int(lineage["budget"]["authoritative_teacher_output_tokens"]) != 152266
        or int(lineage["budget"]["unique_source_attempts"]) != 4953
    ):
        raise Phase3Error(f"matched B50 candidate cost lineage changed: {expected_seed}")
    stage_seconds = 0.0
    peak_cuda = 0
    peak_rss = 0
    stage_rows = []
    base = lineage_path.parent
    for stage, expected_hash in lineage["stage_metadata_sha256"].items():
        metadata_path = base / stage / "metadata.json"
        if not metadata_path.is_file() or sha256_file(metadata_path) != expected_hash:
            raise Phase3Error(f"matched B50 stage metadata changed: {expected_seed}/{stage}")
        metadata = _json(metadata_path)
        training = metadata.get("training", {})
        seconds = float(training.get("wall_seconds", 0.0))
        stage_seconds += seconds
        peak_cuda = max(peak_cuda, int(training.get("peak_cuda_allocated_bytes", 0)))
        peak_rss = max(peak_rss, int(training.get("peak_process_rss_bytes", 0)))
        stage_rows.append(
            {
                "stage": stage,
                "training_seconds": seconds,
                "steps": int(training.get("steps", 0)),
                "metadata_sha256": expected_hash,
            }
        )
    successful_wall = float(lineage["wall_seconds"])
    interrupted_training_seconds = 0.0
    interrupted_rows = []
    terminal_stage_unmeasured = False
    if interrupted_attempt is not None:
        interrupted_root = root / str(interrupted_attempt["root"])
        for stage, expected_hash in interrupted_attempt[
            "stage_metadata_sha256"
        ].items():
            metadata_path = interrupted_root / stage / "metadata.json"
            if not metadata_path.is_file() or sha256_file(metadata_path) != expected_hash:
                raise Phase3Error(
                    f"matched B50 interrupted cost lineage changed: {expected_seed}/{stage}"
                )
            training = _json(metadata_path).get("training", {})
            seconds = float(training.get("wall_seconds", 0.0))
            interrupted_training_seconds += seconds
            interrupted_rows.append(
                {
                    "stage": stage,
                    "recorded_training_seconds": seconds,
                    "metadata_sha256": expected_hash,
                }
            )
        terminal_stage_unmeasured = (
            interrupted_attempt.get("terminal_stage_cost_status")
            == "NOT_RECOVERABLE_NO_METADATA"
        )
        if not terminal_stage_unmeasured:
            raise Phase3Error("matched B50 interrupted terminal cost status changed")
    return {
        "seed": expected_seed,
        "successful_lineage_wall_seconds": successful_wall,
        "recorded_interrupted_training_seconds": interrupted_training_seconds,
        "total_consumed_wall_seconds_lower_bound": successful_wall
        + interrupted_training_seconds,
        "interrupted_terminal_stage_cost_unmeasured": terminal_stage_unmeasured,
        "interrupted_cost_completeness": "LOWER_BOUND_ONLY"
        if terminal_stage_unmeasured
        else "COMPLETE_NO_INTERRUPTED_ATTEMPT",
        "training_seconds": stage_seconds,
        "active_parameter_seconds_conservative_upper_bound": final_parameters
        * stage_seconds,
        "active_parameter_seconds_method": "complete final installed parameter count multiplied by every recorded stage training second; conservative because not every final component is active in every stage",
        "peak_cuda_allocated_bytes": peak_cuda,
        "peak_process_rss_bytes": peak_rss,
        "stages": sorted(stage_rows, key=lambda row: row["stage"]),
        "interrupted_recorded_stages": sorted(
            interrupted_rows, key=lambda row: row["stage"]
        ),
        "lineage_path": lineage_path.relative_to(root).as_posix(),
        "lineage_sha256": sha256_file(lineage_path),
    }


def _source_artifact_costs(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    evidence = protocol["source_artifact_evidence"]
    phase1 = _json(root / str(evidence["phase1_certificate"]))
    targeted = _json(root / str(evidence["targeted_extraction_result"]))
    targeted_summary_path = root / str(evidence["targeted_source_summary"])
    targeted_summary = _json(targeted_summary_path)
    abstention = _json(root / str(evidence["abstention_source_summary"]))
    combined = _json(root / str(evidence["targeted_combined_ir_result"]))
    host = _json(root / str(evidence["host_supervision_result"]))
    host_verify = _json(root / str(evidence["host_supervision_verification"]))
    top64 = _json(root / str(evidence["top64_result"]))
    phase1_cost = phase1["source"]
    targeted_cost = targeted["accounting"]
    abstention_cost = abstention["accounting"]
    targeted_counts = targeted.get("eligible_by_capability", {})
    non_abstention_counts = {
        key: int(value)
        for key, value in targeted_counts.items()
        if key != "abstention"
    }
    if (
        phase1.get("status") != "PASS"
        or targeted.get("status") != "FAIL_SOURCE_EVIDENCE_INADEQUATE"
        or targeted.get("summary", {}).get("sha256")
        != sha256_file(targeted_summary_path)
        or targeted_summary.get("status") != "FAIL_SOURCE_EVIDENCE_INADEQUATE"
        or len(non_abstention_counts) != 13
        or not all(value >= 500 for value in non_abstention_counts.values())
        or int(targeted_counts.get("abstention", 0)) >= 500
        or abstention.get("status") != "PASS_SOURCE_EVIDENCE_READY_FOR_NORMALIZATION"
        or int(abstention.get("selection", {}).get("selected_counts", {}).get("abstention", 0))
        < 500
        or combined.get("status") != "PASS_BALANCED_IR_CONSTRUCTION"
        or combined.get("source_policy", {}).get("v135_non_abstention_capabilities")
        != 13
        or combined.get("source_policy", {}).get("v119_abstention_capabilities")
        != 1
        or host.get("controls", {}).get("teacher_model_loaded") is not False
        or host_verify.get("status")
        != "PASS_INDEPENDENT_HOSTILE_ARTIFACT_VERIFICATION"
        or host_verify.get("artifact", {}).get("sha256")
        != host.get("artifact", {}).get("sha256")
        or host_verify.get("teacher_model_loaded") is not False
        or top64.get("status") != "PASS_EXACT_B50_TOP64_CACHE_READY"
    ):
        raise Phase3Error("matched B50 source-artifact accounting changed")
    phase1_seconds = float(phase1_cost["source_inference_seconds"])
    targeted_seconds = float(targeted_cost["source_inference_seconds"])
    abstention_seconds = float(
        abstention_cost["source_inference_seconds_this_process"]
    )
    common_seconds = phase1_seconds + targeted_seconds + abstention_seconds
    source_load_seconds = (
        float(phase1_cost["source_load_seconds"])
        + float(targeted_cost["source_load_seconds"])
        + float(abstention_cost["source_load_seconds"])
    )
    source_wall_seconds = (
        float(phase1_cost["wall_seconds"])
        + float(targeted_summary["accounting"]["wall_seconds_this_process"])
        + float(abstention_cost["wall_seconds_this_process"])
    )
    teacher_input_tokens = (
        int(phase1_cost["teacher_input_tokens_all_attempts"])
        + int(targeted_cost["teacher_input_tokens"])
        + int(abstention_cost["teacher_input_tokens"])
    )
    teacher_output_tokens = (
        int(phase1_cost["authoritative_teacher_tokens_all_attempts"])
        + int(targeted_cost["authoritative_teacher_tokens"])
        + int(abstention_cost["authoritative_teacher_tokens"])
    )
    return {
        "accounting_scope": "conservative complete reusable source artifacts from which B50 is selected; not fractionally allocated to the B50 subset",
        "common_sequence_artifacts": {
            "phase1_source_inference_seconds": phase1_seconds,
            "v135_targeted_source_inference_seconds": targeted_seconds,
            "v135_status_preserved": targeted["status"],
            "v135_usable_non_abstention_capabilities": 13,
            "v119_abstention_source_inference_seconds": abstention_seconds,
            "host_supervision_additional_teacher_inference_seconds": 0.0,
            "host_supervision_cost_relation": "V480 selects cached V135 attempts and therefore adds no teacher inference cost",
            "total_source_inference_seconds": common_seconds,
            "total_source_inference_hours": common_seconds / 3600.0,
            "total_source_load_seconds": source_load_seconds,
            "total_source_wall_seconds": source_wall_seconds,
            "teacher_input_tokens_all_reusable_source_runs": teacher_input_tokens,
            "teacher_output_tokens_all_reusable_source_runs": teacher_output_tokens,
            "raw_generation_prompt_bytes": {
                "phase1": "NOT_REPORTED_FOR_ALL_ATTEMPTS_IN_CERTIFICATE",
                "v135": int(
                    targeted_summary["accounting"]["raw_generation_prompt_bytes"]
                ),
                "v119": int(abstention_cost["raw_generation_prompt_bytes"]),
            },
            "raw_teacher_output_bytes": {
                "phase1": "NOT_REPORTED_FOR_ALL_ATTEMPTS_IN_CERTIFICATE",
                "v135": int(
                    targeted_summary["accounting"]["raw_teacher_output_bytes"]
                ),
                "v119": int(abstention_cost["raw_teacher_output_bytes"]),
            },
            "stored_logits": 0,
            "stored_hidden_activations": 0,
            "copied_source_parameters": 0,
            "external_hardware": "CUDA",
        },
        "richer_top64_control_addition": {
            "source_inference_seconds": float(top64["source_inference_seconds"]),
            "source_inference_hours": float(top64["source_inference_seconds"])
            / 3600.0,
            "source_load_seconds": float(top64["source_load_seconds"]),
            "source_wall_seconds": float(top64["wall_seconds"]),
            "stored_logit_values": int(top64["stored_logit_values"]),
            "stored_logit_value_bytes": int(top64["stored_logit_value_bytes"]),
            "stored_logit_index_bytes": int(top64["stored_logit_index_bytes"]),
            "peak_cuda_allocated_bytes": int(top64["peak_cuda_allocated_bytes"]),
            "peak_process_rss_bytes": int(top64["peak_process_rss_bytes"]),
            "applies_only_to": ["D1", "D2"],
        },
        "common_cost_shared_by_equal_information_systems": ["ABI", "L0", "L1", "D0"],
        "teacher_present_at_final_abi_inference": False,
    }


def _verified_baseline_pack(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    pack_path = root / str(protocol["baseline_pack_result"])
    verification_path = root / str(protocol["baseline_pack_verification"])
    pack = _json(pack_path)
    verification = _json(verification_path)
    if (
        pack.get("status") != "PASS_EXACT_B50_BASELINE_SEQUENCE_PACK_READY"
        or not result_evidence_digest_valid(pack)
        or verification.get("status")
        != "PASS_INDEPENDENT_EXACT_B50_BASELINE_PACK_VERIFICATION"
        or not result_evidence_digest_valid(verification)
        or verification.get("result_under_test_sha256") != sha256_file(pack_path)
        or verification.get("selection_sha256")
        != pack.get("budget", {}).get("selection_sha256")
        or verification.get("imported_information")
        != pack.get("imported_information")
        or not all(bool(value) for value in verification.get("gates", {}).values())
        or not all(bool(value) for value in verification.get("attacks", {}).values())
        or verification.get("training_performed") is not False
        or verification.get("model_inference_performed") is not False
        or verification.get("teacher_model_loaded") is not False
        or verification.get("final_test_accessed") is not False
    ):
        raise Phase3Error("matched B50 independently verified baseline pack changed")
    return pack, verification


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    baseline_verify = _json(root / str(protocol["baseline_headline_verification"]))
    candidate_verify = _json(root / str(protocol["candidate_verification"]))
    candidate_screen = _json(root / str(protocol["candidate_screen_result"]))
    baseline_pack, baseline_pack_verification = _verified_baseline_pack(root, protocol)
    if (
        baseline_verify.get("status")
        != "PASS_COMPLETE_THREE_SEED_HEADLINE_INDEPENDENTLY_VERIFIED"
        or baseline_verify.get("raw_prompt_observations") != 21000
        or not result_evidence_digest_valid(baseline_verify)
        or candidate_verify.get("status")
        != "PASS_INDEPENDENTLY_VERIFIED_STABLE_B50_V22_DEVELOPMENT_CANDIDATE"
        or not result_evidence_digest_valid(candidate_verify)
        or candidate_screen.get("status") != "PASS_STABLE_B50_V22_DEVELOPMENT_CANDIDATE"
        or not result_evidence_digest_valid(candidate_screen)
    ):
        raise Phase3Error("matched B50 verified source result changed")
    probes = _probes(root, protocol)
    teacher = _teacher_rows(root, protocol, probes)
    thresholds = protocol["absolute_screen"]
    replicates = int(protocol["statistics"]["bootstrap_replicates"])
    lower_minimum = float(protocol["statistics"]["noninferiority_lower_95_minimum"])

    candidate_specs = {int(row["seed"]): row for row in candidate_screen["systems"]}
    if set(candidate_specs) != set(SEEDS):
        raise Phase3Error("matched B50 candidate seed matrix changed")
    candidate_rows: dict[int, list[dict[str, Any]]] = {}
    candidate_reports: dict[int, dict[str, Any]] = {}
    candidate_costs: dict[int, dict[str, Any]] = {}
    source_artifact_costs = _source_artifact_costs(root, protocol)
    lineages = protocol["candidate_lineages"]
    for seed in SEEDS:
        spec = candidate_specs[seed]
        raw_path = root / str(spec["outputs"]["path"])
        if sha256_file(raw_path) != spec["outputs"]["sha256"]:
            raise Phase3Error(f"matched B50 candidate raw output changed: {seed}")
        rows = _normalized_rows(raw_path, probes, candidate=True)
        report = evaluation(rows)
        gates = absolute_gates(report, thresholds)
        teacher_comparison = paired_quality(
            rows,
            teacher,
            replicates=replicates,
            seed=int(protocol["statistics"]["candidate_teacher_seed_base"]) + seed,
        )
        gates["teacher_noninferior"] = teacher_comparison["lower_95"] >= lower_minimum
        candidate_rows[seed] = rows
        candidate_reports[seed] = {
            "seed": seed,
            "evaluation": report,
            "absolute_gates": gates,
            "teacher_comparison_v1": teacher_comparison,
            "all_quality_gates_pass": all(gates.values()),
            "package": spec["package"],
            "outputs_path": raw_path.relative_to(root).as_posix(),
            "outputs_sha256": sha256_file(raw_path),
        }
        lineage_spec = lineages[str(seed)]
        candidate_costs[seed] = _candidate_cost(
            root,
            root / str(lineage_spec["result"]),
            expected_seed=seed,
            final_parameters=int(spec["package"]["total_parameters"]),
            interrupted_attempt=lineage_spec.get("interrupted_attempt"),
        )

    baseline_runs = {
        (str(row["system"]), int(row["seed"])): row
        for row in baseline_verify["runs"]
    }
    if set(baseline_runs) != {(system, seed) for system in SYSTEMS for seed in SEEDS}:
        raise Phase3Error("matched B50 baseline seed matrix changed")
    baseline_reports: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    comparisons: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    source_snapshot_bytes = int(protocol["deployment_accounting"]["lora_source_snapshot_bytes"])
    snapshot = _verified_snapshot(root)
    observed_snapshot_bytes = sum(
        path.stat().st_size for path in snapshot.rglob("*") if path.is_file()
    )
    if observed_snapshot_bytes != source_snapshot_bytes:
        raise Phase3Error("matched B50 LoRA source snapshot byte accounting changed")
    for system_index, system in enumerate(SYSTEMS):
        for seed in SEEDS:
            source = baseline_runs[(system, seed)]
            raw_path = root / str(source["result_path"])
            result = _json(raw_path)
            output_path = root / str(result["development"]["outputs_path"])
            if (
                sha256_file(output_path) != source["outputs_sha256"]
                or sha256_file(root / str(source["checkpoint_path"]))
                != source["checkpoint_sha256"]
            ):
                raise Phase3Error(f"matched B50 baseline raw binding changed: {system}/{seed}")
            rows = _normalized_rows(output_path, probes, candidate=False)
            report = evaluation(rows)
            gates = absolute_gates(report, thresholds)
            teacher_comparison = paired_quality(
                rows,
                teacher,
                replicates=replicates,
                seed=int(protocol["statistics"]["baseline_teacher_seed_base"])
                + system_index * 1_000_000
                + seed,
            )
            gates["teacher_noninferior"] = teacher_comparison["lower_95"] >= lower_minimum
            deployed_bytes = int(source["checkpoint_bytes"])
            if system in {"L0", "L1"}:
                deployed_bytes += source_snapshot_bytes
            baseline_reports[system].append(
                {
                    "seed": seed,
                    "evaluation": report,
                    "absolute_gates": gates,
                    "teacher_comparison_v1": teacher_comparison,
                    "all_quality_gates_pass": all(gates.values()),
                    "teacher_present_at_inference": bool(
                        source["teacher_present_at_inference"]
                    ),
                    "imported_information": {
                        **source["imported_information"],
                        "frozen_source_parameters_copied": int(
                            source["frozen_source_parameters_copied"]
                        ),
                        "frozen_source_snapshot_bytes_retained": source_snapshot_bytes
                        if system in {"L0", "L1"}
                        else 0,
                    },
                    "complete_installed_parameters": int(
                        source["complete_installed_parameters"]
                    ),
                    "active_parameters": int(source["active_parameters"]),
                    "deployed_artifact_bytes": deployed_bytes,
                    "training_seconds": float(source["training_seconds"]),
                    "run_wall_seconds": float(source["run_wall_seconds"]),
                    "active_parameter_seconds": int(source["active_parameters"])
                    * float(source["training_seconds"]),
                    "peak_cuda_allocated_bytes": int(
                        source["peak_cuda_allocated_bytes"]
                    ),
                    "peak_process_rss_bytes": int(source["peak_process_rss_bytes"]),
                    "checkpoint_path": source["checkpoint_path"],
                    "checkpoint_sha256": source["checkpoint_sha256"],
                    "outputs_path": output_path.relative_to(root).as_posix(),
                    "outputs_sha256": sha256_file(output_path),
                }
            )
            comparison = paired_quality(
                candidate_rows[seed],
                rows,
                replicates=replicates,
                seed=int(protocol["statistics"]["candidate_baseline_seed_base"])
                + system_index * 1_000_000
                + seed,
            )
            comparison["candidate_noninferior"] = (
                comparison["lower_95"] >= lower_minimum
            )
            comparisons[system].append({"seed": seed, **comparison})

    system_decisions = {}
    candidate_all_seed = all(
        candidate_reports[seed]["all_quality_gates_pass"] for seed in SEEDS
    )
    for system in SYSTEMS:
        baseline_all_seed = all(
            row["all_quality_gates_pass"] for row in baseline_reports[system]
        )
        noninferior = all(row["candidate_noninferior"] for row in comparisons[system])
        package_ratios = [
            int(candidate_reports[seed]["package"]["archive_bytes"])
            / int(next(row for row in baseline_reports[system] if row["seed"] == seed)["deployed_artifact_bytes"])
            for seed in SEEDS
        ]
        training_ratios = [
            candidate_costs[seed]["training_seconds"]
            / next(row for row in baseline_reports[system] if row["seed"] == seed)["training_seconds"]
            for seed in SEEDS
        ]
        parameter_second_ratios = [
            candidate_costs[seed]["active_parameter_seconds_conservative_upper_bound"]
            / next(row for row in baseline_reports[system] if row["seed"] == seed)["active_parameter_seconds"]
            for seed in SEEDS
        ]
        consumed_wall_lower_ratios = [
            candidate_costs[seed]["total_consumed_wall_seconds_lower_bound"]
            / next(row for row in baseline_reports[system] if row["seed"] == seed)[
                "run_wall_seconds"
            ]
            for seed in SEEDS
        ]
        candidate_cost_complete = all(
            candidate_costs[seed]["interrupted_cost_completeness"]
            == "COMPLETE_NO_INTERRUPTED_ATTEMPT"
            for seed in SEEDS
        )
        system_decisions[system] = {
            "equal_teacher_sequence_information": system in {"L0", "L1", "D0"},
            "equal_total_imported_information": system == "D0",
            "richer_information_control": system in {"L0", "L1", "D1", "D2"},
            "retains_frozen_source_parameters": system in {"L0", "L1"},
            "candidate_all_seed_quality_pass": candidate_all_seed,
            "baseline_all_seed_quality_pass": baseline_all_seed,
            "candidate_noninferior_all_seed": noninferior,
            "baseline_teacher_present_at_inference": system in {"L0", "L1"},
            "candidate_teacher_present_at_inference": False,
            "candidate_package_bytes_ratio_by_seed": package_ratios,
            "candidate_package_bytes_ratio_maximum": max(package_ratios),
            "candidate_to_baseline_training_seconds_ratio_by_seed": training_ratios,
            "candidate_to_baseline_training_seconds_ratio_median": statistics.median(
                training_ratios
            ),
            "successful_lineage_training_ratio_excludes_interrupted_attempts": True,
            "candidate_to_baseline_active_parameter_seconds_upper_ratio_by_seed": parameter_second_ratios,
            "candidate_to_baseline_active_parameter_seconds_upper_ratio_median": statistics.median(
                parameter_second_ratios
            ),
            "candidate_to_baseline_consumed_wall_lower_bound_ratio_by_seed": consumed_wall_lower_ratios,
            "candidate_to_baseline_consumed_wall_lower_bound_ratio_median": statistics.median(
                consumed_wall_lower_ratios
            ),
            "candidate_consumed_wall_cost_complete_all_seeds": candidate_cost_complete,
            "matched_quality_endpoint_reached_by_baseline": baseline_all_seed,
            "bounded_functional_quality_advantage": candidate_all_seed
            and not baseline_all_seed,
            "bounded_deployed_size_and_teacher_absence_advantage": candidate_all_seed
            and noninferior
            and max(package_ratios)
            <= float(protocol["deployment_accounting"]["versus_lora_total_deployed_bytes_ratio_maximum"])
            and system in {"L0", "L1"},
            "bounded_composite_advantage": False,
            "composite_decision_deferred_to_runtime_and_contract_compose": True,
            "matched_quality_training_efficiency_endpoint_available": baseline_all_seed
            and candidate_cost_complete,
        }

    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_READ_ONLY_MATCHED_B50_QUALITY_AND_COST_RECOMPUTED",
        "protocol_sha256": protocol_sha,
        "candidate": [candidate_reports[seed] for seed in SEEDS],
        "candidate_costs": [candidate_costs[seed] for seed in SEEDS],
        "source_artifact_costs": source_artifact_costs,
        "exact_b50_imported_information": baseline_pack["imported_information"],
        "exact_b50_budget": baseline_pack["budget"],
        "exact_b50_pack_independent_verification": {
            "status": baseline_pack_verification["status"],
            "result_under_test_sha256": baseline_pack_verification[
                "result_under_test_sha256"
            ],
            "evidence_sha256": baseline_pack_verification["evidence_sha256"],
        },
        "baselines": baseline_reports,
        "paired_candidate_minus_baseline_v1": comparisons,
        "system_decisions": system_decisions,
        "raw_prompt_observations_recomputed": 3 * 1400 + 15 * 1400,
        "bootstrap_comparisons": 3 + 15 + 15,
        "bootstrap_replicates_each": replicates,
        "candidate_all_seed_quality_pass": candidate_all_seed,
        "lora_source_snapshot": {
            "revision_path": snapshot.as_posix(),
            "bytes": observed_snapshot_bytes,
            "accounting": "all files in the verified frozen source revision snapshot",
        },
        "training_performed": False,
        "model_inference_performed": False,
        "runtime_profiling_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "abi_superiority_certified": False,
        "next_action": "Seal same-checkpoint CPU/GPU runtime and complete acquisition-accounting verification. The mixed B40 adjacent-lower topology still prevents a Phase 4 minimum claim.",
        "claim_boundary": "Read-only exact-B50 development quality, deployed-size, training-time, and conservative active-parameter-second comparison. No runtime, final-test, stable-minimum, Phase 4, or unconditional ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root, (root / args.protocol).resolve(), (root / args.output).resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_all_seed_quality_pass": result[
                    "candidate_all_seed_quality_pass"
                ],
                "systems": result["system_decisions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
