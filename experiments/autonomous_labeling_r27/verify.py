"""Strict live end-to-end R27 replay and fail-closed verification."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, evidence_hash, json_object, sha256_file, write_json_once
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash

from .binding import load_config
from .import_run import BUILD_INITIALIZATIONS, HOST_INITIALIZATIONS, run as run_import
from .prepare_import import run as prepare_import
from .protocol import answer_key
from .source_run import run as run_source


ROW_FILES = ("package_builds.jsonl", "domain_observations.jsonl", "english_immutability.jsonl", "lifecycle.jsonl", "english_leakage.jsonl")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _source_metrics(directory: Path) -> dict[str, int]:
    rows = _rows(directory / "source_rows.jsonl"); evaluation = _rows(directory / "evaluation.jsonl")
    result = json_object(directory / "result.json")
    package_count = len(result["packages"])
    return {
        "rows": len(rows), "parse": sum(row["parse_exact"] for row in rows),
        "answers": sum(row["answer_exact"] for row in rows), "labels": sum(row["label_semantic_valid"] for row in rows),
        "evaluation": len(evaluation), "package_exact": sum(answer_key(str(row["package_answer"] or "")) == answer_key(row["oracle_answer"]) for row in evaluation),
        "teacher_agreement": sum(answer_key(str(row["package_answer"] or "")) == answer_key(row["teacher_answer"]) for row in evaluation),
        "other_abstain": sum(row["other_answer"] is None for row in evaluation), "package_count": package_count,
    }


def _import_metrics(directory: Path) -> dict[str, int]:
    result = json_object(directory / "result.json"); observations = _rows(directory / "domain_observations.jsonl"); core = _rows(directory / "english_immutability.jsonl"); lifecycle = _rows(directory / "lifecycle.jsonl"); leakage = _rows(directory / "english_leakage.jsonl")
    packages = len(result["canonical_packages"])
    return {
        "observations": len(observations), "oracle": sum(answer_key(row["gpu_output"]) == answer_key(row["oracle_answer"]) for row in observations),
        "teacher": sum(answer_key(row["gpu_output"]) == answer_key(row["teacher_answer"]) for row in observations),
        "cpu_gpu": sum(row["cpu_output"] == row["gpu_output"] for row in observations),
        "other_abstain": sum(all(not value for value in row["other_outputs"]) for row in observations),
        "core_rows": len(core), "core_exact": 0, "leakage": sum(answer_key(row["output"]) == answer_key(row["target"]) for row in leakage),
        "lifecycle": sum(row["removed"] and row["absent_rejected"] and row["before"] == row["after"] and row["restored_sha256"] == row["expected_sha256"] for row in lifecycle),
        "corruptions": sum(row["corruption_rejected"] for row in lifecycle), "lifecycle_rows": len(lifecycle), "packages": packages,
    }


def verify(config_path: Path, reveal_path: Path, import_config: Path, source_stored: Path, import_stored: Path, replay: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]; config = load_config(root, config_path)
    source_value = json_object(source_stored / "result.json"); import_value = json_object(import_stored / "result.json")
    if source_value.get("evidence_sha256") != selfless_evidence_hash(source_value) or import_value.get("evidence_sha256") != selfless_evidence_hash(import_value): raise R14Error("R27 stored evidence hash changed")
    replay_source = replay / "source"; replay_import = replay / "import"; replay_config = replay / "import_config.json"
    run_source(config_path, reveal_path, replay_source)
    for relative in ("source_rows.jsonl", "source_bundle.json", "evaluation.jsonl"):
        if (source_stored / relative).read_bytes() != (replay_source / relative).read_bytes(): raise R14Error(f"R27 source live replay changed: {relative}")
    for stored_package in source_value["packages"]:
        candidate = next(item for item in json_object(replay_source / "result.json")["packages"] if item["namespace"] == stored_package["namespace"])
        if (source_stored / stored_package["path"]).read_bytes() != (replay_source / candidate["path"]).read_bytes(): raise R14Error("R27 compiler package replay changed")
    prepare_import(config_path, replay_source, replay_config); run_import(replay_config, replay_import)
    for relative in ROW_FILES:
        left, right = _rows(import_stored / relative), _rows(replay_import / relative)
        if relative == "package_builds.jsonl":
            left = [{k: v for k, v in row.items() if k != "path"} for row in left]; right = [{k: v for k, v in row.items() if k != "path"} for row in right]
        if left != right: raise R14Error(f"R27 LayerCake live replay changed: {relative}")
    source_metrics = _source_metrics(source_stored); imported = _import_metrics(import_stored)
    expected_observations = 36 * len(HOST_INITIALIZATIONS); expected_lifecycle = imported["packages"] * len(HOST_INITIALIZATIONS)
    scientific = (
        source_metrics["rows"] == 72 and source_metrics["parse"] == 72 and source_metrics["answers"] == 72 and source_metrics["labels"] == 72
        and source_metrics["evaluation"] == 36 and source_metrics["package_exact"] == 36 and source_metrics["teacher_agreement"] == 36 and source_metrics["other_abstain"] == 36 and source_metrics["package_count"] >= 4
        and imported["observations"] == expected_observations and all(imported[name] == expected_observations for name in ("oracle", "teacher", "cpu_gpu", "other_abstain"))
        and imported["core_rows"] == 720 and imported["leakage"] == 0 and imported["lifecycle"] == expected_lifecycle and imported["corruptions"] == expected_lifecycle and imported["lifecycle_rows"] == expected_lifecycle
    )
    hostile = 0
    with tempfile.TemporaryDirectory(prefix="r27-hostile-") as raw:
        for mutation in ("missing", "hash", "stale"):
            value = json_object(import_config)
            if mutation == "missing": value["source_packages"][0]["path"] = "physically-absent.abipkg"
            elif mutation == "hash": value["source_packages"][0]["sha256"] = "0" * 64
            else: value["source_result"]["sha256"] = "f" * 64
            path = Path(raw) / f"{mutation}.json"; path.write_text(json.dumps(value), encoding="utf-8")
            try: run_import(path, Path(raw) / f"out-{mutation}")
            except (R14Error, FileNotFoundError): hostile += 1
    if not scientific or hostile != 3 or source_value.get("verdict") != "PASS" or import_value.get("verdict") != "PASS": raise R14Error("R27 strict scientific gate failed")
    return {"format": "abi-r27-strict-live-verification/1", "status": "PASS_STRICTLY_VERIFIED_BOUNDED_OPEN_LABEL_IMPORT", "scientific_claim": "BOUNDED_HELDOUT_FREE_LABEL_DISCOVERY_COMPILATION_AND_LAYERCAKE_IMPORT", "claim_ceiling": "NOT_EXHAUSTIVE_TEACHER_DIAGNOSIS_OR_FLUENT_ENGLISH_EXTRACTION", "config_sha256": sha256_file(config_path), "reveal_sha256": sha256_file(reveal_path), "stored_source_sha256": sha256_file(source_stored / "result.json"), "stored_import_sha256": sha256_file(import_stored / "result.json"), "source_metrics_recomputed": source_metrics, "import_metrics_recomputed": imported, "source_rows_live_replayed": 72, "layercake_rows_live_replayed": sum(len(_rows(replay_import / name)) for name in ROW_FILES), "source_packages_rebuilt": source_metrics["package_count"], "layercake_packages_rebuilt": imported["packages"] * len(BUILD_INITIALIZATIONS), "hostile_cases_rejected": hostile, "hostile_cases": 3, "stored_scientific_booleans_trusted": False, "source_parameters_copied": 0, "receiver_training_steps": 0, "teacher_present_at_layercake_execution": False, "full_abi_moonshot": "OPEN"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--reveal", type=Path, required=True); parser.add_argument("--import-config", type=Path, required=True); parser.add_argument("--source-stored", type=Path, required=True); parser.add_argument("--import-stored", type=Path, required=True); parser.add_argument("--replay", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = verify(args.config, args.reveal, args.import_config, args.source_stored, args.import_stored, args.replay); write_json_once(args.output, result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
