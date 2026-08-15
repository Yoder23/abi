"""Independently verify and select the exact-B50 fixed baseline grid."""

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
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b50-grid-verify/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-grid-selection/1"
RUN_RESULT_FORMAT = "abi-capability-compiler-phase4-b50-baseline-run-result/1"
SYSTEMS = ("L0", "L1", "D0", "D1")


def expected_configurations() -> dict[str, set[tuple[int | None, float, int]]]:
    return {
        "L0": {(rank, lr, exp) for rank in (16, 64) for lr in (1e-4, 3e-4) for exp in (1, 4)},
        "L1": {(rank, lr, exp) for rank in (8, 32) for lr in (1e-4, 3e-4) for exp in (1, 4)},
        "D0": {(None, lr, exp) for lr in (1e-5, 3e-5) for exp in (1, 2, 4)},
        "D1": {(None, lr, exp) for lr in (1e-5, 3e-5) for exp in (1, 2, 4)},
    }


def grid_tree_sha256(grid_root: Path) -> tuple[str, list[Path]]:
    files = sorted(
        path
        for path in grid_root.glob("*/*")
        if path.is_file()
        and path.name in {"phase4_result.json", "receipt.json", "development_outputs.jsonl"}
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(grid_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return digest.hexdigest(), files


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_COMPLETE_GRID_INDEPENDENT_SELECTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("grid verification governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"grid verification binding changed: {relative}")
    return protocol, sha256_file(path)


def selected_probes(root: Path, protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = load_catalog((root / str(protocol["development_catalog"])).resolve())
    grouped = {capability: [] for capability in CAPABILITIES}
    for probe in catalog["probes"]:
        capability = str(probe.get("canonical_capability"))
        if probe.get("split") == "validation" and capability in grouped:
            grouped[capability].append(probe)
    selected: dict[str, dict[str, Any]] = {}
    for capability in CAPABILITIES:
        rows = sorted(grouped[capability], key=lambda row: str(row["probe_id"]))
        if len(rows) != 100:
            raise Phase3Error("grid development catalog depth changed")
        for row in rows[:10]:
            selected[str(row["probe_id"])] = row
    if len(selected) != 140:
        raise Phase3Error("grid development selection depth changed")
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def result_evidence_digest_valid(result: Mapping[str, Any]) -> bool:
    evidence = dict(result)
    observed = evidence.pop("evidence_sha256", None)
    return isinstance(observed, str) and hashlib.sha256(
        canonical_json_bytes(evidence)
    ).hexdigest() == observed


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


def _loss(receipt: Mapping[str, Any], system: str) -> float:
    if system in {"D0", "D1", "D2"}:
        return float(receipt["loss"]["total"])
    values = receipt["per_capability_training"].values()
    steps = sum(int(value["successful_optimizer_steps"]) for value in values)
    if steps <= 0:
        raise Phase3Error("LoRA response-loss step count changed")
    return sum(
        float(value["mean_loss"]) * int(value["successful_optimizer_steps"])
        for value in values
    ) / steps


def configuration_identity(row: Mapping[str, Any]) -> str:
    config = row["configuration"]
    rank = config.get("rank")
    return f"{row['system']}:r{rank if rank is not None else 'none'}:lr{float(config['learning_rate']):.8g}:e{int(config['exposures'])}"


def ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["macro_functional_rate_v1"]),
        0 if int(row["repetition_collapses_v1"]) == 0 else 1,
        -int(row["functional_passes_v1"]),
        float(row["response_loss"]),
        int(row["imported_information_scalars"]),
        float(row["training_seconds"]),
        configuration_identity(row),
    )


def rank_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=ranking_key)


def _verify_one(
    root: Path,
    directory: Path,
    probes: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
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
        or result.get("stage") != "grid"
        or result.get("candidate_training_performed") is not True
        or result.get("teacher_query_generation_performed") is not False
        or result.get("final_test_accessed") is not False
        or result.get("phase4_certified") is not False
    ):
        raise Phase3Error(f"grid result governance changed: {directory.name}")
    if not result_evidence_digest_valid(result):
        raise Phase3Error(f"grid result evidence digest changed: {directory.name}")
    expected_protocol = (
        protocol["source_protocol_sha256"]["lora"]
        if system in {"L0", "L1"}
        else protocol["source_protocol_sha256"]["student"]
    )
    if result.get("protocol_sha256") != expected_protocol:
        raise Phase3Error(f"grid source protocol changed: {directory.name}")
    if result["receipt"] != {
        "path": receipt_path.relative_to(root).as_posix(),
        "sha256": sha256_file(receipt_path),
    }:
        raise Phase3Error(f"grid receipt binding changed: {directory.name}")
    if receipt["development"]["outputs_sha256"] != sha256_file(outputs_path):
        raise Phase3Error(f"grid output binding changed: {directory.name}")
    rows = _read_jsonl(outputs_path)
    if len(rows) != 140 or {str(row["probe_id"]) for row in rows} != set(probes):
        raise Phase3Error(f"grid prompt set changed: {directory.name}")
    recomputed = []
    for row in rows:
        probe = probes[str(row["probe_id"])]
        capability = str(probe["canonical_capability"])
        output = str(row["output"])
        observed = dict(row)
        expected = {
            "functional_pass": evaluate_functional(output, probe["evaluator"]),
            "functional_pass_v2": evaluate_functional_v2(output, probe["evaluator"], capability),
            "repetition_collapse": repetition_collapse(output),
            "repetition_collapse_v2": repetition_collapse_v2(output),
        }
        if row.get("capability") != capability or any(observed.get(key) != value for key, value in expected.items()):
            raise Phase3Error(f"grid prompt metric changed: {directory.name}/{row.get('probe_id')}")
        recomputed.append(observed)
    metrics = _metrics(recomputed)
    for key in ("observations", "functional_passes", "functional_passes_v2", "repetition_collapses", "repetition_collapses_v2", "per_capability"):
        if result["development"][key] != metrics[key] or receipt["development"][key] != metrics[key]:
            raise Phase3Error(f"grid aggregate changed: {directory.name}/{key}")
    config = result["configuration"]
    identity = (config.get("rank"), float(config["learning_rate"]), int(config["exposures"]))
    if identity not in expected_configurations()[system] or int(config["seed"]) != 104729:
        raise Phase3Error(f"grid configuration changed: {directory.name}")
    exposures = int(config["exposures"])
    if (
        int(result["training"]["optimizer_steps"]) != 860 * exposures
        or int(result["training"]["response_tokens_seen"]) != 163524 * exposures
    ):
        raise Phase3Error(f"grid training depth changed: {directory.name}")
    per = metrics["per_capability"]
    macro = sum(float(per[cap]["functional_passes"]) / 10.0 for cap in CAPABILITIES) / len(CAPABILITIES)
    imported_scalars = int(result["imported_information"]["authoritative_teacher_output_tokens"])
    if system == "D1":
        imported_scalars += int(result["imported_information"]["stored_top64_values"])
    return {
        "system": system,
        "configuration": {
            "rank": config.get("rank"),
            "learning_rate": float(config["learning_rate"]),
            "exposures": exposures,
            "seed": 104729,
        },
        "macro_functional_rate_v1": macro,
        "functional_passes_v1": metrics["functional_passes"],
        "functional_passes_v2": metrics["functional_passes_v2"],
        "repetition_collapses_v1": metrics["repetition_collapses"],
        "repetition_collapses_v2": metrics["repetition_collapses_v2"],
        "router_correct": int(result["development"].get("router_correct", 0)) if system == "L1" else None,
        "response_loss": _loss(receipt, system),
        "imported_information_scalars": imported_scalars,
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
    protocol, protocol_sha = load_protocol(root, protocol_path)
    grid_root = (root / str(protocol["grid_root"])).resolve()
    tree_sha, files = grid_tree_sha256(grid_root)
    if tree_sha != protocol["grid_tree_sha256"] or len(files) != 84:
        raise Phase3Error("complete grid evidence tree changed")
    probes = selected_probes(root, protocol)
    directories = sorted({path.parent for path in files})
    if len(directories) != 28 or any(len(list(directory.iterdir())) != 3 for directory in directories):
        raise Phase3Error("complete grid directory structure changed")
    rows = [_verify_one(root, directory, probes, protocol) for directory in directories]
    for system in SYSTEMS:
        observed = {
            (row["configuration"]["rank"], row["configuration"]["learning_rate"], row["configuration"]["exposures"])
            for row in rows if row["system"] == system
        }
        if observed != expected_configurations()[system]:
            raise Phase3Error(f"complete grid matrix changed: {system}")
    ranked = {system: list(rank_rows([row for row in rows if row["system"] == system])) for system in SYSTEMS}
    selected = {system: values[:2] for system, values in ranked.items()}
    d2_configs = sorted({
        (float(row["configuration"]["learning_rate"]), int(row["configuration"]["exposures"]))
        for system in ("D0", "D1") for row in selected[system]
    })
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_COMPLETE_GRID_VERIFIED_SELECTION_DERIVED",
        "protocol_sha256": protocol_sha,
        "grid_tree_sha256": tree_sha,
        "grid_files": len(files),
        "grid_runs": len(rows),
        "grid_prompt_observations": len(rows) * 140,
        "recomputed_from_raw_outputs": True,
        "selection_metric_version": "historical V1",
        "prospective_v2_used_for_selection": False,
        "ranking_rule": protocol["ranking_rule"],
        "runs": sorted(rows, key=lambda row: (row["system"], ranking_key(row))),
        "selected_top_two": selected,
        "authorized_d2_grid_configurations": [
            {"learning_rate": lr, "exposures": exposures} for lr, exposures in d2_configs
        ],
        "adversarial_checks": {
            "missing_run_rejected": True,
            "extra_run_rejected": True,
            "result_digest_mutation_rejected": True,
            "receipt_hash_mutation_rejected": True,
            "prompt_output_mutation_rejected": True,
            "metric_mutation_rejected": True,
            "configuration_mutation_rejected": True,
            "protocol_lineage_mutation_rejected": True
        },
        "training_performed": False,
        "model_inference_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "next_action": "Seal a derived protocol for the two D2 grid controls and the selected full-development replays; no headline, runtime, or final access yet.",
        "claim_boundary": "Complete exact-B50 seed-104729 grid verification and deterministic development selection only. No full-depth, headline, runtime, matched frontier, Phase 4, or ABI-superiority claim."
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
    print(json.dumps({"status": result["status"], "grid_runs": result["grid_runs"], "selected": {key: [configuration_identity(row) for row in value] for key, value in result["selected_top_two"].items()}, "d2": result["authorized_d2_grid_configurations"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
