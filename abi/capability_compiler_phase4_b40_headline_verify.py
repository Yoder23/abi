"""Independently verify the exact-B40 three-seed matched-baseline tree."""

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
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase2_lora import route_prompt
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b40_baselines import (
    load_exact_records,
    train_exact_b40_router,
)
from .capability_compiler_phase4_b50_grid_verify import (
    _loss,
    result_evidence_digest_valid,
)
from .capability_compiler_phase4_b50_headline_verify import (
    adversarial_test_evidence,
    checkpoint_filename,
    development_probes,
    expected_filenames,
    generation_attention_mask_audit,
    headline_tree_sha256,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b40-headline-verify/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b40-headline-verification/1"
RUN_RESULT_FORMAT = "abi-capability-compiler-phase4-b40-baseline-run-result/1"
SYSTEMS = ("L0", "L1", "D0")
SEEDS = (104729, 130363, 155921)
PROMPTS_PER_CAPABILITY = 100
EXPECTED_OBSERVATIONS_PER_RUN = 1400


def expected_configurations() -> dict[str, tuple[int | None, float, int]]:
    return {
        "L0": (16, 1e-4, 4),
        "L1": (8, 1e-4, 4),
        "D0": (None, 3e-5, 4),
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_COMPLETE_B40_HEADLINE_INDEPENDENT_VERIFICATION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("runtime_profiling_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B40 headline verification governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B40 headline verification binding changed: {relative}")
    return protocol, sha256_file(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_capability = {}
    for capability in CAPABILITIES:
        subset = [row for row in rows if row["capability"] == capability]
        per_capability[capability] = {
            "observations": len(subset),
            "functional_passes": sum(bool(row["functional_pass"]) for row in subset),
            "functional_passes_v2": sum(bool(row["functional_pass_v2"]) for row in subset),
            "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in subset),
            "repetition_collapses_v2": sum(bool(row["repetition_collapse_v2"]) for row in subset),
        }
    return {
        "observations": len(rows),
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "functional_passes_v2": sum(bool(row["functional_pass_v2"]) for row in rows),
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "repetition_collapses_v2": sum(bool(row["repetition_collapse_v2"]) for row in rows),
        "per_capability": per_capability,
    }


def _verify_run(
    root: Path,
    directory: Path,
    probes: Mapping[str, Mapping[str, Any]],
    *,
    source_protocol_sha256: str,
    router_centroids: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = directory / "phase4_result.json"
    receipt_path = directory / "receipt.json"
    outputs_path = directory / "development_outputs.jsonl"
    result = _json(result_path)
    receipt = _json(receipt_path)
    system = str(result.get("system"))
    if system not in SYSTEMS or {path.name for path in directory.iterdir()} != expected_filenames(system):
        raise Phase3Error(f"B40 headline directory structure changed: {directory.name}")
    if (
        result.get("format") != RUN_RESULT_FORMAT
        or result.get("status") != "PASS_EXACT_B40_BASELINE_RUN_COMPLETE"
        or result.get("stage") != "headline"
        or result.get("protocol_sha256") != source_protocol_sha256
        or result.get("candidate_training_performed") is not True
        or result.get("teacher_query_generation_performed") is not False
        or result.get("final_test_accessed") is not False
        or result.get("phase4_certified") is not False
        or not result_evidence_digest_valid(result)
    ):
        raise Phase3Error(f"B40 headline run governance changed: {directory.name}")
    if result["receipt"] != {
        "path": receipt_path.relative_to(root).as_posix(),
        "sha256": sha256_file(receipt_path),
    }:
        raise Phase3Error(f"B40 headline receipt binding changed: {directory.name}")
    if receipt["development"]["outputs_sha256"] != sha256_file(outputs_path):
        raise Phase3Error(f"B40 headline output binding changed: {directory.name}")
    checkpoint_path = root / str(result["checkpoint"]["path"])
    if (
        checkpoint_path.parent.resolve() != directory.resolve()
        or checkpoint_path.name != checkpoint_filename(system)
        or not checkpoint_path.is_file()
        or result["checkpoint"]["sha256"] != sha256_file(checkpoint_path)
        or receipt.get("checkpoint_path") != checkpoint_path.relative_to(root).as_posix()
        or receipt.get("checkpoint_sha256") != result["checkpoint"]["sha256"]
    ):
        raise Phase3Error(f"B40 headline checkpoint binding changed: {directory.name}")
    rows = _read_jsonl(outputs_path)
    if len(rows) != len(probes) or {str(row["probe_id"]) for row in rows} != set(probes):
        raise Phase3Error(f"B40 headline prompt set changed: {directory.name}")
    router_correct = 0
    for row in rows:
        probe = probes[str(row["probe_id"])]
        capability = str(probe["canonical_capability"])
        output = str(row["output"])
        expected = {
            "functional_pass": evaluate_functional(output, probe["evaluator"]),
            "functional_pass_v2": evaluate_functional_v2(output, probe["evaluator"], capability),
            "repetition_collapse": repetition_collapse(output),
            "repetition_collapse_v2": repetition_collapse_v2(output),
        }
        if row.get("capability") != capability or any(row.get(key) != value for key, value in expected.items()):
            raise Phase3Error(f"B40 headline prompt metric changed: {directory.name}/{row.get('probe_id')}")
        if system in {"L0", "L1"}:
            routed = capability if system == "L0" else route_prompt(str(probe["prompt"]), router_centroids)
            if row.get("routed_capability") != routed:
                raise Phase3Error(f"B40 headline route changed: {directory.name}/{row.get('probe_id')}")
            router_correct += int(routed == capability)
    metrics = _metrics(rows)
    for key in (
        "observations",
        "functional_passes",
        "functional_passes_v2",
        "repetition_collapses",
        "repetition_collapses_v2",
        "per_capability",
    ):
        if result["development"][key] != metrics[key] or receipt["development"][key] != metrics[key]:
            raise Phase3Error(f"B40 headline aggregate changed: {directory.name}/{key}")
    config = result["configuration"]
    identity = (config.get("rank"), float(config["learning_rate"]), int(config["exposures"]))
    seed = int(config["seed"])
    if identity != expected_configurations()[system] or seed not in SEEDS:
        raise Phase3Error(f"B40 headline configuration changed: {directory.name}")
    exposures = int(config["exposures"])
    if (
        int(config["development_per_capability"]) != PROMPTS_PER_CAPABILITY
        or int(result["training"]["optimizer_steps"]) != 693 * exposures
        or int(result["training"]["response_tokens_seen"]) != 130885 * exposures
    ):
        raise Phase3Error(f"B40 headline training depth changed: {directory.name}")
    receipt_method = receipt.get("system") if system in {"L0", "L1"} else receipt.get("method")
    if (
        receipt.get("status") != "PASS"
        or receipt_method != system
        or int(receipt["seed"]) != seed
        or float(receipt["learning_rate"]) != identity[1]
        or int(receipt["target_token_exposures"]) != exposures
        or int(receipt["successful_optimizer_steps"]) != int(result["training"]["optimizer_steps"])
        or int(receipt["response_tokens_seen"]) != int(result["training"]["response_tokens_seen"])
        or bool(receipt["final_prompts_accessed"]) is not False
    ):
        raise Phase3Error(f"B40 headline training receipt changed: {directory.name}")
    if system in {"L0", "L1"} and (
        int(receipt["rank"]) != identity[0]
        or int(result["development"].get("router_correct", -1)) != router_correct
        or int(receipt["development"].get("router_correct", -1)) != router_correct
    ):
        raise Phase3Error(f"B40 headline LoRA accounting changed: {directory.name}")
    imported = result["imported_information"]
    if (
        result.get("information_role") != "EXACT_B40_SEQUENCE_EQUAL_INFORMATION"
        or int(imported["record_memberships"]) != 4112
        or int(imported["unique_source_attempts"]) != 4005
        or int(imported["authoritative_teacher_output_tokens"]) != 123167
        or int(imported["stored_top64_values"]) != 0
    ):
        raise Phase3Error(f"B40 imported-information accounting changed: {directory.name}")
    expected_source_present = system in {"L0", "L1"}
    base_parameters = int(receipt.get("base_parameters", 0))
    adapter_parameters = int(receipt.get("adapter_parameters_per_capability", 0))
    router_parameters = int(receipt.get("router_parameters", 0))
    if (
        bool(result["source_base_present_at_inference"]) is not expected_source_present
        or bool(result["deployment"]["source_base_present_at_inference"]) is not expected_source_present
        or bool(result["deployment"]["all_fourteen_lora_adapters_counted"]) is not True
        or (
            expected_source_present
            and (
                base_parameters <= 0
                or adapter_parameters <= 0
                or int(result["deployment"]["complete_installed_parameters"])
                != base_parameters + adapter_parameters * 14 + router_parameters
                or int(result["deployment"]["active_parameters"])
                != base_parameters + adapter_parameters + router_parameters
            )
        )
    ):
        raise Phase3Error(f"B40 deployment accounting changed: {directory.name}")
    macro_v1 = statistics.fmean(
        metrics["per_capability"][capability]["functional_passes"] / PROMPTS_PER_CAPABILITY
        for capability in CAPABILITIES
    )
    macro_v2 = statistics.fmean(
        metrics["per_capability"][capability]["functional_passes_v2"] / PROMPTS_PER_CAPABILITY
        for capability in CAPABILITIES
    )
    return {
        "system": system,
        "seed": seed,
        "configuration": {"rank": config.get("rank"), "learning_rate": identity[1], "exposures": exposures},
        "observations": metrics["observations"],
        "functional_passes_v1": metrics["functional_passes"],
        "functional_passes_v2": metrics["functional_passes_v2"],
        "macro_functional_rate_v1": macro_v1,
        "macro_functional_rate_v2": macro_v2,
        "repetition_collapses_v1": metrics["repetition_collapses"],
        "repetition_collapses_v2": metrics["repetition_collapses_v2"],
        "per_capability": metrics["per_capability"],
        "response_loss": _loss(receipt, system),
        "training_seconds": float(result["training"]["training_seconds"]),
        "run_wall_seconds": float(result["training"]["run_wall_seconds"]),
        "complete_installed_parameters": int(result["deployment"]["complete_installed_parameters"]),
        "active_parameters": int(result["deployment"]["active_parameters"]),
        "peak_cuda_allocated_bytes": int(result["training"]["peak_cuda_allocated_bytes"]),
        "peak_process_rss_bytes": int(result["training"]["peak_process_rss_bytes"]),
        "source_base_present_at_inference": expected_source_present,
        "frozen_source_parameters_copied": base_parameters if expected_source_present else 0,
        "adapter_parameters_per_capability": adapter_parameters if expected_source_present else 0,
        "router_parameters": router_parameters if expected_source_present else 0,
        "imported_information": imported,
        "router_correct": router_correct if expected_source_present else None,
        "result_path": result_path.relative_to(root).as_posix(),
        "result_sha256": sha256_file(result_path),
        "receipt_sha256": sha256_file(receipt_path),
        "outputs_sha256": sha256_file(outputs_path),
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: item["seed"])
    per_capability = {
        capability: {
            "functional_passes_v1": [int(row["per_capability"][capability]["functional_passes"]) for row in ordered],
            "functional_passes_v2": [int(row["per_capability"][capability]["functional_passes_v2"]) for row in ordered],
            "repetition_collapses_v1": [int(row["per_capability"][capability]["repetition_collapses"]) for row in ordered],
            "repetition_collapses_v2": [int(row["per_capability"][capability]["repetition_collapses_v2"]) for row in ordered],
        }
        for capability in CAPABILITIES
    }
    return {
        "seeds": [int(row["seed"]) for row in ordered],
        "functional_passes_v1": [int(row["functional_passes_v1"]) for row in ordered],
        "functional_passes_v2": [int(row["functional_passes_v2"]) for row in ordered],
        "repetition_collapses_v1": [int(row["repetition_collapses_v1"]) for row in ordered],
        "repetition_collapses_v2": [int(row["repetition_collapses_v2"]) for row in ordered],
        "per_capability": per_capability,
        "mean_macro_functional_rate_v1": statistics.fmean(float(row["macro_functional_rate_v1"]) for row in rows),
        "mean_macro_functional_rate_v2": statistics.fmean(float(row["macro_functional_rate_v2"]) for row in rows),
        "mean_response_loss": statistics.fmean(float(row["response_loss"]) for row in rows),
        "median_training_seconds": statistics.median(float(row["training_seconds"]) for row in rows),
        "median_run_wall_seconds": statistics.median(float(row["run_wall_seconds"]) for row in rows),
        "maximum_peak_cuda_allocated_bytes": max(int(row["peak_cuda_allocated_bytes"]) for row in rows),
        "maximum_peak_process_rss_bytes": max(int(row["peak_process_rss_bytes"]) for row in rows),
        "checkpoint_sha256": [str(row["checkpoint_sha256"]) for row in ordered],
    }


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    adversarial = adversarial_test_evidence(root, protocol)
    source_protocol_sha = str(protocol["source_protocol_sha256"])
    source_protocol_path = (root / str(protocol["source_protocol"])).resolve()
    if sha256_file(source_protocol_path) != source_protocol_sha:
        raise Phase3Error("B40 headline source protocol lineage changed")
    headline_root = (root / str(protocol["headline_root"])).resolve()
    tree_sha, files = headline_tree_sha256(headline_root)
    if tree_sha != protocol["headline_tree_sha256"] or len(files) != 36:
        raise Phase3Error("complete B40 headline evidence tree changed")
    directories = sorted({path.parent for path in files})
    if len(directories) != 9:
        raise Phase3Error("complete B40 headline run count changed")
    probes = development_probes(root, protocol)
    mask_audit = generation_attention_mask_audit(probes, _tokenizer(_verified_snapshot(root)))
    if not mask_audit["implicit_all_ones_mask_semantically_correct"]:
        raise Phase3Error("B40 generation attention-mask safety changed")
    source_protocol = _json(source_protocol_path)
    records = load_exact_records(root / str(source_protocol["records_archive"]))
    router_centroids = train_exact_b40_router([
        dict(row, normalized_acquisition_prompt=row["normalized_generation_prompt"])
        for row in records
    ])
    rows = [
        _verify_run(root, directory, probes, source_protocol_sha256=source_protocol_sha, router_centroids=router_centroids)
        for directory in directories
    ]
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        if len(system_rows) != 3 or {row["seed"] for row in system_rows} != set(SEEDS):
            raise Phase3Error(f"B40 headline paired-seed matrix changed: {system}")
        if len({row["checkpoint_sha256"] for row in system_rows}) != 3:
            raise Phase3Error(f"B40 headline seed-distinct checkpoints changed: {system}")
    aggregates = {system: _aggregate([row for row in rows if row["system"] == system]) for system in SYSTEMS}
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_COMPLETE_B40_THREE_SEED_HEADLINE_INDEPENDENTLY_VERIFIED",
        "protocol_sha256": protocol_sha,
        "source_protocol_sha256": source_protocol_sha,
        "headline_tree_sha256": tree_sha,
        "headline_files": len(files),
        "file_manifest": [{"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files],
        "headline_runs": len(rows),
        "raw_prompt_observations": sum(int(row["observations"]) for row in rows),
        "recomputed_from_raw_outputs": True,
        "generation_attention_mask_audit": mask_audit,
        "paired_seeds": list(SEEDS),
        "runs": sorted(rows, key=lambda row: (row["system"], row["seed"])),
        "aggregates": aggregates,
        "all_checkpoints_content_addressed": True,
        "all_checkpoints_seed_distinct_within_system": True,
        "fail_closed_rejection_controls": {
            "missing_or_extra_file_rejected": True,
            "tree_content_mutation_rejected": True,
            "result_digest_mutation_rejected": True,
            "receipt_hash_mutation_rejected": True,
            "checkpoint_hash_mutation_rejected": True,
            "prompt_output_mutation_rejected": True,
            "metric_mutation_rejected": True,
            "configuration_or_seed_mutation_rejected": True,
            "protocol_lineage_mutation_rejected": True,
            "information_budget_mutation_rejected": True,
        },
        "adversarial_test_evidence": adversarial,
        "training_performed": False,
        "model_inference_performed": False,
        "runtime_profiling_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "next_action": "Preregister matched runtime for these exact checkpoints and compare them with the exact B40 LayerCake-hosted ABI products.",
        "claim_boundary": "Independent exact-B40 baseline headline verification only. No runtime, final-test, Phase 4, or ABI-superiority claim.",
    }
    if result["raw_prompt_observations"] != 12600:
        raise Phase3Error("B40 headline raw observation count changed")
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
    print(json.dumps({"status": result["status"], "runs": result["headline_runs"], "observations": result["raw_prompt_observations"], "aggregates": result["aggregates"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
