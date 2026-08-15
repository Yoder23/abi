"""Independently verify the exact-B50 three-seed baseline headline tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    load_catalog,
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase2_lora import _render_prompt, route_prompt
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot
from .capability_compiler_phase4_b50_baselines import (
    load_exact_records,
    train_exact_b50_router,
)
from .capability_compiler_phase4_b50_grid_verify import (
    RUN_RESULT_FORMAT,
    _loss,
    result_evidence_digest_valid,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b50-headline-verify/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-headline-verification/1"
SYSTEMS = ("L0", "L1", "D0", "D1", "D2")
SEEDS = (104729, 130363, 155921)
PROMPTS_PER_CAPABILITY = 100
EXPECTED_OBSERVATIONS_PER_RUN = len(CAPABILITIES) * PROMPTS_PER_CAPABILITY


def expected_configurations() -> dict[str, tuple[int | None, float, int]]:
    return {
        "L0": (16, 1e-4, 4),
        "L1": (8, 1e-4, 4),
        "D0": (None, 3e-5, 4),
        "D1": (None, 3e-5, 4),
        "D2": (None, 3e-5, 4),
    }


def checkpoint_filename(system: str) -> str:
    return "adapters.safetensors" if system in {"L0", "L1"} else "student.safetensors"


def expected_filenames(system: str) -> set[str]:
    return {
        "phase4_result.json",
        "receipt.json",
        "development_outputs.jsonl",
        checkpoint_filename(system),
    }


def headline_tree_sha256(headline_root: Path) -> tuple[str, list[Path]]:
    files = sorted(path for path in headline_root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise Phase3Error("headline evidence tree may not contain symlinks")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(headline_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return digest.hexdigest(), files


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_COMPLETE_HEADLINE_INDEPENDENT_VERIFICATION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("runtime_profiling_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("headline verification governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"headline verification binding changed: {relative}")
    return protocol, sha256_file(path)


def adversarial_test_evidence(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    specification = protocol["adversarial_test_evidence"]
    path = (root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != specification["sha256"]:
        raise Phase3Error("headline adversarial test evidence changed")
    document = ET.parse(path).getroot()
    suites = [document] if document.tag == "testsuite" else list(document.findall("testsuite"))
    totals = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if (
        totals["tests"] < int(specification["minimum_tests"])
        or totals["failures"] != 0
        or totals["errors"] != 0
        or totals["skipped"] != 0
    ):
        raise Phase3Error("headline adversarial test suite did not pass cleanly")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        **totals,
    }


def development_probes(root: Path, protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = load_catalog((root / str(protocol["development_catalog"])).resolve())
    grouped = {capability: [] for capability in CAPABILITIES}
    for probe in catalog["probes"]:
        capability = str(probe.get("canonical_capability"))
        if probe.get("split") == "validation" and capability in grouped:
            grouped[capability].append(probe)
    selected: dict[str, dict[str, Any]] = {}
    for capability in CAPABILITIES:
        rows = sorted(grouped[capability], key=lambda row: str(row["probe_id"]))
        if len(rows) != PROMPTS_PER_CAPABILITY:
            raise Phase3Error("headline development catalog depth changed")
        for row in rows:
            selected[str(row["probe_id"])] = row
    if len(selected) != EXPECTED_OBSERVATIONS_PER_RUN:
        raise Phase3Error("headline prompt selection changed")
    return selected


def generation_attention_mask_audit(
    probes: Mapping[str, Mapping[str, Any]], tokenizer: Any
) -> dict[str, Any]:
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id)
    containing = [
        probe_id
        for probe_id, probe in probes.items()
        if eos in _render_prompt(tokenizer, str(probe["prompt"]))
    ]
    return {
        "eos_token_id": eos,
        "pad_token_id": pad,
        "eos_and_pad_share_id": eos == pad,
        "prompts_checked": len(probes),
        "prompts_containing_shared_eos_pad_id": len(containing),
        "implicit_all_ones_mask_semantically_correct": eos != pad or not containing,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per = {}
    for capability in CAPABILITIES:
        subset = [row for row in rows if row["capability"] == capability]
        per[capability] = {
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
        "per_capability": per,
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
        raise Phase3Error(f"headline directory structure changed: {directory.name}")
    if (
        result.get("format") != RUN_RESULT_FORMAT
        or result.get("status") != "PASS_EXACT_B50_BASELINE_RUN_COMPLETE"
        or result.get("stage") != "headline"
        or result.get("protocol_sha256") != source_protocol_sha256
        or result.get("candidate_training_performed") is not True
        or result.get("teacher_query_generation_performed") is not False
        or result.get("final_test_accessed") is not False
        or result.get("phase4_certified") is not False
        or not result_evidence_digest_valid(result)
    ):
        raise Phase3Error(f"headline run governance changed: {directory.name}")
    if result["receipt"] != {
        "path": receipt_path.relative_to(root).as_posix(),
        "sha256": sha256_file(receipt_path),
    }:
        raise Phase3Error(f"headline receipt binding changed: {directory.name}")
    if receipt["development"]["outputs_sha256"] != sha256_file(outputs_path):
        raise Phase3Error(f"headline output binding changed: {directory.name}")
    checkpoint_path = root / str(result["checkpoint"]["path"])
    if (
        checkpoint_path.parent.resolve() != directory.resolve()
        or checkpoint_path.name != checkpoint_filename(system)
        or not checkpoint_path.is_file()
        or result["checkpoint"]["sha256"] != sha256_file(checkpoint_path)
        or receipt.get("checkpoint_path") != checkpoint_path.relative_to(root).as_posix()
        or receipt.get("checkpoint_sha256") != result["checkpoint"]["sha256"]
    ):
        raise Phase3Error(f"headline checkpoint binding changed: {directory.name}")
    rows = _read_jsonl(outputs_path)
    if len(rows) != len(probes) or {str(row["probe_id"]) for row in rows} != set(probes):
        raise Phase3Error(f"headline prompt set changed: {directory.name}")
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
            raise Phase3Error(f"headline prompt metric changed: {directory.name}/{row.get('probe_id')}")
        if system in {"L0", "L1"}:
            routed = (
                capability
                if system == "L0"
                else route_prompt(str(probe["prompt"]), router_centroids)
            )
            if row.get("routed_capability") != routed:
                raise Phase3Error(
                    f"headline route changed: {directory.name}/{row.get('probe_id')}"
                )
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
            raise Phase3Error(f"headline aggregate changed: {directory.name}/{key}")
    config = result["configuration"]
    identity = (config.get("rank"), float(config["learning_rate"]), int(config["exposures"]))
    seed = int(config["seed"])
    if identity != expected_configurations()[system] or seed not in SEEDS:
        raise Phase3Error(f"headline configuration changed: {directory.name}")
    exposures = int(config["exposures"])
    if (
        int(config["development_per_capability"]) != PROMPTS_PER_CAPABILITY
        or int(result["training"]["optimizer_steps"]) != 860 * exposures
        or int(result["training"]["response_tokens_seen"]) != 163524 * exposures
    ):
        raise Phase3Error(f"headline training depth changed: {directory.name}")
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
        raise Phase3Error(f"headline training receipt changed: {directory.name}")
    if system in {"L0", "L1"}:
        if int(receipt["rank"]) != identity[0] or (
            int(result["development"].get("router_correct", -1)) != router_correct
            or int(receipt["development"].get("router_correct", -1)) != router_correct
        ):
            raise Phase3Error(f"headline LoRA accounting changed: {directory.name}")
    imported = result["imported_information"]
    expected_top64 = 10465536 if system in {"D1", "D2"} else 0
    if (
        int(imported["record_memberships"]) != 5140
        or int(imported["authoritative_teacher_output_tokens"]) != 152266
        or int(imported["stored_top64_values"]) != expected_top64
    ):
        raise Phase3Error(f"headline imported-information accounting changed: {directory.name}")
    expected_teacher_present = system in {"L0", "L1"}
    base_parameters = int(receipt.get("base_parameters", 0))
    adapter_parameters = int(receipt.get("adapter_parameters_per_capability", 0))
    router_parameters = int(receipt.get("router_parameters", 0))
    if (
        bool(result["teacher_present_at_student_inference"]) is not expected_teacher_present
        or bool(result["deployment"]["source_base_present_at_inference"]) is not expected_teacher_present
        or bool(result["deployment"]["all_fourteen_lora_adapters_counted"]) is not True
        or (
            system in {"L0", "L1"}
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
        raise Phase3Error(f"headline deployment accounting changed: {directory.name}")
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
        "configuration": {
            "rank": config.get("rank"),
            "learning_rate": float(config["learning_rate"]),
            "exposures": exposures,
        },
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
        "teacher_present_at_inference": expected_teacher_present,
        "frozen_source_parameters_copied": base_parameters
        if expected_teacher_present
        else 0,
        "adapter_parameters_per_capability": adapter_parameters
        if expected_teacher_present
        else 0,
        "router_parameters": router_parameters if expected_teacher_present else 0,
        "imported_information": imported,
        "router_correct": router_correct if system in {"L0", "L1"} else None,
        "result_path": result_path.relative_to(root).as_posix(),
        "result_sha256": sha256_file(result_path),
        "receipt_sha256": sha256_file(receipt_path),
        "outputs_sha256": sha256_file(outputs_path),
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    ordered = sorted(rows, key=lambda item: item["seed"])
    per_capability = {
        capability: {
            "functional_passes_v1": [
                int(row["per_capability"][capability]["functional_passes"])
                for row in ordered
            ],
            "functional_passes_v2": [
                int(row["per_capability"][capability]["functional_passes_v2"])
                for row in ordered
            ],
            "repetition_collapses_v1": [
                int(row["per_capability"][capability]["repetition_collapses"])
                for row in ordered
            ],
            "repetition_collapses_v2": [
                int(row["per_capability"][capability]["repetition_collapses_v2"])
                for row in ordered
            ],
        }
        for capability in CAPABILITIES
    }
    return {
        "seeds": sorted(int(row["seed"]) for row in rows),
        "functional_passes_v1": [int(row["functional_passes_v1"]) for row in ordered],
        "functional_passes_v2": [int(row["functional_passes_v2"]) for row in ordered],
        "repetition_collapses_v1": [int(row["repetition_collapses_v1"]) for row in ordered],
        "repetition_collapses_v2": [int(row["repetition_collapses_v2"]) for row in ordered],
        "per_capability": per_capability,
        "mean_macro_functional_rate_v1": statistics.fmean(values("macro_functional_rate_v1")),
        "mean_macro_functional_rate_v2": statistics.fmean(values("macro_functional_rate_v2")),
        "mean_response_loss": statistics.fmean(values("response_loss")),
        "median_training_seconds": statistics.median(values("training_seconds")),
        "median_run_wall_seconds": statistics.median(values("run_wall_seconds")),
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
        raise Phase3Error("headline source protocol lineage changed")
    headline_root = (root / str(protocol["headline_root"])).resolve()
    tree_sha, files = headline_tree_sha256(headline_root)
    if tree_sha != protocol["headline_tree_sha256"] or len(files) != 60:
        raise Phase3Error("complete headline evidence tree changed")
    directories = sorted({path.parent for path in files})
    if len(directories) != 15:
        raise Phase3Error("complete headline run count changed")
    probes = development_probes(root, protocol)
    mask_audit = generation_attention_mask_audit(
        probes, _tokenizer(_verified_snapshot(root))
    )
    if not mask_audit["implicit_all_ones_mask_semantically_correct"]:
        raise Phase3Error("headline generation attention-mask safety changed")
    source_protocol = _json(source_protocol_path)
    records = load_exact_records(root / str(source_protocol["records_archive"]))
    router_centroids = train_exact_b50_router(
        [dict(row, normalized_acquisition_prompt=row["normalized_generation_prompt"]) for row in records]
    )
    rows = [
        _verify_run(
            root,
            directory,
            probes,
            source_protocol_sha256=source_protocol_sha,
            router_centroids=router_centroids,
        )
        for directory in directories
    ]
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        if len(system_rows) != 3 or {row["seed"] for row in system_rows} != set(SEEDS):
            raise Phase3Error(f"headline paired-seed matrix changed: {system}")
        if len({row["checkpoint_sha256"] for row in system_rows}) != 3:
            raise Phase3Error(f"headline seed-distinct checkpoints changed: {system}")
    aggregates = {
        system: _aggregate([row for row in rows if row["system"] == system])
        for system in SYSTEMS
    }
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_COMPLETE_THREE_SEED_HEADLINE_INDEPENDENTLY_VERIFIED",
        "protocol_sha256": protocol_sha,
        "source_protocol_sha256": source_protocol_sha,
        "headline_tree_sha256": tree_sha,
        "headline_files": len(files),
        "file_manifest": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
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
        },
        "adversarial_test_evidence": adversarial,
        "training_performed": False,
        "model_inference_performed": False,
        "runtime_profiling_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "next_action": "Seal a matched runtime protocol for these exact same-run checkpoints before any timing; ABI candidate comparison and final-test evidence remain required.",
        "claim_boundary": "Independent exact-B50 baseline headline verification only. No runtime, matched ABI frontier, final-test, Phase 4, or ABI-superiority claim.",
    }
    if result["raw_prompt_observations"] != 21000:
        raise Phase3Error("headline raw observation count changed")
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
                "runs": result["headline_runs"],
                "observations": result["raw_prompt_observations"],
                "aggregates": result["aggregates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
