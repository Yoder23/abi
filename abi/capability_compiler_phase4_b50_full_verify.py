"""Independently verify exact-B50 D2 screening and full development."""

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
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b50_grid_verify import (
    RUN_RESULT_FORMAT,
    _loss,
    configuration_identity,
    grid_tree_sha256,
    rank_rows,
    result_evidence_digest_valid,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b50-full-verify/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-full-selection/1"
SYSTEMS = ("L0", "L1", "D0", "D1", "D2")


def expected_full_configurations() -> dict[str, set[tuple[int | None, float, int]]]:
    return {
        "L0": {(16, 1e-4, 1), (16, 1e-4, 4)},
        "L1": {(8, 1e-4, 4), (8, 3e-4, 4)},
        "D0": {(None, 3e-5, 2), (None, 3e-5, 4)},
        "D1": {(None, 3e-5, 2), (None, 3e-5, 4)},
        "D2": {(None, 3e-5, 2), (None, 3e-5, 4)},
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_D2_AND_FULL_INDEPENDENT_SELECTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("full-development verification governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"full-development verification binding changed: {relative}")
    return protocol, sha256_file(path)


def development_probes(root: Path, protocol: Mapping[str, Any], depth: int) -> dict[str, dict[str, Any]]:
    catalog = load_catalog((root / str(protocol["development_catalog"])).resolve())
    grouped = {capability: [] for capability in CAPABILITIES}
    for probe in catalog["probes"]:
        capability = str(probe.get("canonical_capability"))
        if probe.get("split") == "validation" and capability in grouped:
            grouped[capability].append(probe)
    selected = {}
    for capability in CAPABILITIES:
        rows = sorted(grouped[capability], key=lambda row: str(row["probe_id"]))
        if len(rows) != 100 or depth not in {10, 100}:
            raise Phase3Error("full-development catalog depth changed")
        for row in rows[:depth]:
            selected[str(row["probe_id"])] = row
    if len(selected) != len(CAPABILITIES) * depth:
        raise Phase3Error("full-development prompt selection changed")
    return selected


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


def verify_run(
    root: Path,
    directory: Path,
    probes: Mapping[str, Mapping[str, Any]],
    *,
    stage: str,
    protocol_sha: str,
) -> dict[str, Any]:
    result_path = directory / "phase4_result.json"
    receipt_path = directory / "receipt.json"
    outputs_path = directory / "development_outputs.jsonl"
    result = _json(result_path)
    receipt = _json(receipt_path)
    system = str(result.get("system"))
    if (
        result.get("format") != RUN_RESULT_FORMAT
        or result.get("status") != "PASS_EXACT_B50_BASELINE_RUN_COMPLETE"
        or system not in SYSTEMS
        or result.get("stage") != stage
        or result.get("protocol_sha256") != protocol_sha
        or result.get("candidate_training_performed") is not True
        or result.get("teacher_query_generation_performed") is not False
        or result.get("final_test_accessed") is not False
        or result.get("phase4_certified") is not False
        or not result_evidence_digest_valid(result)
    ):
        raise Phase3Error(f"full-development run governance changed: {directory.name}")
    if result["receipt"] != {"path": receipt_path.relative_to(root).as_posix(), "sha256": sha256_file(receipt_path)}:
        raise Phase3Error(f"full-development receipt binding changed: {directory.name}")
    if receipt["development"]["outputs_sha256"] != sha256_file(outputs_path):
        raise Phase3Error(f"full-development output binding changed: {directory.name}")
    rows = [json.loads(line) for line in outputs_path.read_bytes().splitlines() if line.strip()]
    if len(rows) != len(probes) or {str(row["probe_id"]) for row in rows} != set(probes):
        raise Phase3Error(f"full-development prompt set changed: {directory.name}")
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
            raise Phase3Error(f"full-development prompt metric changed: {directory.name}/{row.get('probe_id')}")
    metrics = _metrics(rows)
    for key in ("observations", "functional_passes", "functional_passes_v2", "repetition_collapses", "repetition_collapses_v2", "per_capability"):
        if result["development"][key] != metrics[key] or receipt["development"][key] != metrics[key]:
            raise Phase3Error(f"full-development aggregate changed: {directory.name}/{key}")
    config = result["configuration"]
    identity = (config.get("rank"), float(config["learning_rate"]), int(config["exposures"]))
    if identity not in expected_full_configurations()[system] or int(config["seed"]) != 104729:
        raise Phase3Error(f"full-development configuration changed: {directory.name}")
    exposures = int(config["exposures"])
    if int(result["training"]["optimizer_steps"]) != 860 * exposures or int(result["training"]["response_tokens_seen"]) != 163524 * exposures:
        raise Phase3Error(f"full-development training depth changed: {directory.name}")
    depth = len(probes) // len(CAPABILITIES)
    macro = sum(float(metrics["per_capability"][cap]["functional_passes"]) / depth for cap in CAPABILITIES) / len(CAPABILITIES)
    imported = int(result["imported_information"]["authoritative_teacher_output_tokens"])
    if system in {"D1", "D2"}:
        imported += int(result["imported_information"]["stored_top64_values"])
    return {
        "system": system,
        "stage": stage,
        "configuration": {"rank": config.get("rank"), "learning_rate": float(config["learning_rate"]), "exposures": exposures, "seed": 104729},
        "macro_functional_rate_v1": macro,
        "functional_passes_v1": metrics["functional_passes"],
        "functional_passes_v2": metrics["functional_passes_v2"],
        "repetition_collapses_v1": metrics["repetition_collapses"],
        "repetition_collapses_v2": metrics["repetition_collapses_v2"],
        "router_correct": int(result["development"].get("router_correct", 0)) if system == "L1" else None,
        "response_loss": _loss(receipt, system),
        "imported_information_scalars": imported,
        "training_seconds": float(result["training"]["training_seconds"]),
        "run_wall_seconds": float(result["training"]["run_wall_seconds"]),
        "complete_installed_parameters": int(result["deployment"]["complete_installed_parameters"]),
        "active_parameters": int(result["deployment"]["active_parameters"]),
        "peak_cuda_allocated_bytes": int(result["training"]["peak_cuda_allocated_bytes"]),
        "peak_process_rss_bytes": int(result["training"]["peak_process_rss_bytes"]),
        "result_path": result_path.relative_to(root).as_posix(),
        "result_sha256": sha256_file(result_path),
        "receipt_sha256": sha256_file(receipt_path),
        "outputs_sha256": sha256_file(outputs_path),
    }


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_hash = load_protocol(root, protocol_path)
    source_protocol_hash = str(protocol["source_protocol_sha256"])
    d2_root = (root / str(protocol["d2_grid_root"])).resolve()
    full_root = (root / str(protocol["full_root"])).resolve()
    d2_tree, d2_files = grid_tree_sha256(d2_root)
    full_tree, full_files = grid_tree_sha256(full_root)
    if d2_tree != protocol["d2_grid_tree_sha256"] or len(d2_files) != 6:
        raise Phase3Error("D2 grid tree changed")
    if full_tree != protocol["full_tree_sha256"] or len(full_files) != 30:
        raise Phase3Error("full-development tree changed")
    d2_dirs = sorted({path.parent for path in d2_files})
    full_dirs = sorted({path.parent for path in full_files})
    d2_probes = development_probes(root, protocol, 10)
    full_probes = development_probes(root, protocol, 100)
    d2_rows = [verify_run(root, directory, d2_probes, stage="d2_grid", protocol_sha=source_protocol_hash) for directory in d2_dirs]
    full_rows = [verify_run(root, directory, full_probes, stage="full", protocol_sha=source_protocol_hash) for directory in full_dirs]
    if len(d2_rows) != 2 or {row["system"] for row in d2_rows} != {"D2"}:
        raise Phase3Error("D2 grid matrix changed")
    for system in SYSTEMS:
        observed = {(row["configuration"]["rank"], row["configuration"]["learning_rate"], row["configuration"]["exposures"]) for row in full_rows if row["system"] == system}
        if observed != expected_full_configurations()[system]:
            raise Phase3Error(f"full-development matrix changed: {system}")
    selected = {system: list(rank_rows([row for row in full_rows if row["system"] == system]))[0] for system in SYSTEMS}
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_D2_AND_FULL_DEVELOPMENT_VERIFIED_HEADLINE_SELECTION_DERIVED",
        "protocol_sha256": protocol_hash,
        "source_protocol_sha256": source_protocol_hash,
        "d2_grid_tree_sha256": d2_tree,
        "full_tree_sha256": full_tree,
        "d2_grid_runs": d2_rows,
        "full_runs": sorted(full_rows, key=lambda row: (row["system"], configuration_identity(row))),
        "raw_prompt_observations": len(d2_rows) * 140 + len(full_rows) * 1400,
        "recomputed_from_raw_outputs": True,
        "selection_metric_version": "historical V1",
        "prospective_v2_used_for_selection": False,
        "selected_headline": selected,
        "training_performed": False,
        "model_inference_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "next_action": "Seal a headline protocol for the five selected configurations across seeds 104729, 130363, and 155921 with checkpoints; do not profile runtime or access final data yet.",
        "claim_boundary": "D2 screening and seed-104729 full-development verification plus deterministic headline selection only. No three-seed headline, runtime, final-test, Phase 4, or ABI-superiority claim."
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
    print(json.dumps({"status": result["status"], "observations": result["raw_prompt_observations"], "selected": {system: configuration_identity(row) for system, row in result["selected_headline"].items()}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
