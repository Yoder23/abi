"""Strict fresh live verification for R28 acquisition and LayerCake import."""

from __future__ import annotations

import argparse, json, tempfile
from pathlib import Path

from experiments.autonomous_labeling_r27.import_run import BUILD_INITIALIZATIONS, HOST_INITIALIZATIONS, run as run_import
from experiments.autonomous_labeling_r27.prepare_import import run as prepare_import
from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file, write_json_once
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash
from .binding import load_config
from .protocol import answer_key, unique_quorum
from .source_run import run as run_source


IMPORT_ROWS = ("package_builds.jsonl", "domain_observations.jsonl", "english_immutability.jsonl", "lifecycle.jsonl", "english_leakage.jsonl")
def rows(path): return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_metrics(directory):
    raw = rows(directory / "source_rows.jsonl"); evaluation = rows(directory / "evaluation.jsonl"); result = json_object(directory / "result.json"); extraction = json_object(directory / "extraction/result.json"); by_fact = {}
    for row in raw:
        if row["split"] == "extraction": by_fact.setdefault(row["fact_id"], []).append(row)
    return {"rows": len(raw), "parse": sum(row["parse_exact"] for row in raw), "answers": sum(row["answer_exact"] for row in raw), "labels": sum(row["label_semantic_valid"] for row in raw), "answer_quorums": sum(unique_quorum([row["teacher_answer"] for row in group], answer_key) is not None for group in by_fact.values()), "label_quorums": sum(unique_quorum([row["free_label"] for row in group]) is not None for group in by_fact.values()), "compiled_facts": sum(item["facts"] for item in result["packages"]), "packages": len(result["packages"]), "evaluation": len(evaluation), "package_exact": sum(answer_key(str(row["package_answer"] or "")) == answer_key(row["oracle_answer"]) for row in evaluation), "teacher_agreement": sum(answer_key(str(row["package_answer"] or "")) == answer_key(row["teacher_answer"]) for row in evaluation), "other_abstain": sum(row["other_answer"] is None for row in evaluation), "control_exact": sum(answer_key(str(row["control_answer"] or "")) == answer_key(row["oracle_answer"]) for row in evaluation), "oracle_fields_consumed": extraction["oracle_fields_consumed"], "accepted_labels_consumed": extraction["accepted_labels_consumed"]}


def import_metrics(root, host_config, directory):
    result = json_object(directory / "result.json"); observations = rows(directory / "domain_observations.jsonl"); core = rows(directory / "english_immutability.jsonl"); lifecycle = rows(directory / "lifecycle.jsonl"); leakage = rows(directory / "english_leakage.jsonl"); builds = rows(directory / "package_builds.jsonl")
    stored = rows(root / host_config["r23_hidden_rows"]["path"]); expected = {row["record_id"]: row["output"] for row in stored if row.get("method") == "abi_factorized" and row.get("seed") == 21022}; packages = len(result["canonical_packages"])
    return {"packages": packages, "builds": len(builds), "reproducible_builds": sum(row["byte_identical"] for row in builds), "observations": len(observations), "oracle": sum(answer_key(row["gpu_output"]) == answer_key(row["oracle_answer"]) for row in observations), "teacher": sum(answer_key(row["gpu_output"]) == answer_key(row["teacher_answer"]) for row in observations), "cpu_gpu": sum(row["cpu_output"] == row["gpu_output"] for row in observations), "other_abstain": sum(all(not value for value in row["other_outputs"]) for row in observations), "core_rows": len(core), "core_exact": sum(row["output"] == expected[row["record_id"]] for row in core), "leakage": sum(answer_key(row["output"]) == answer_key(row["target"]) for row in leakage), "lifecycle_rows": len(lifecycle), "lifecycle": sum(row["removed"] and row["absent_rejected"] and row["before"] == row["after"] and row["restored_sha256"] == row["expected_sha256"] for row in lifecycle), "corruptions": sum(row["corruption_rejected"] for row in lifecycle)}


def verify(config_path, reveal_path, source_stored, import_config, import_stored, replay):
    root = Path(__file__).resolve().parents[2]; config = load_config(root, config_path); source_value = json_object(source_stored / "result.json"); import_value = json_object(import_stored / "result.json")
    if source_value.get("evidence_sha256") != selfless_evidence_hash(source_value) or import_value.get("evidence_sha256") != selfless_evidence_hash(import_value): raise R14Error("R28 stored evidence hash changed")
    replay_source = replay / "source"; replay_import_config = replay / "import_config.json"; replay_import = replay / "import"; run_source(config_path, reveal_path, replay_source)
    for relative in ("source_rows.jsonl", "source_bundle.json", "evaluation.jsonl"):
        if (source_stored / relative).read_bytes() != (replay_source / relative).read_bytes(): raise R14Error(f"R28 source replay changed: {relative}")
    replay_source_value = json_object(replay_source / "result.json")
    for item in source_value["packages"]:
        candidate = next(value for value in replay_source_value["packages"] if value["namespace"] == item["namespace"])
        if (source_stored / item["path"]).read_bytes() != (replay_source / candidate["path"]).read_bytes(): raise R14Error("R28 source package replay changed")
    host_config_path = root / config["r27_host_config"]["path"]; prepare_import(host_config_path, replay_source, replay_import_config); run_import(replay_import_config, replay_import)
    for relative in IMPORT_ROWS:
        left, right = rows(import_stored / relative), rows(replay_import / relative)
        if relative == "package_builds.jsonl": left = [{k: v for k, v in row.items() if k != "path"} for row in left]; right = [{k: v for k, v in row.items() if k != "path"} for row in right]
        if left != right: raise R14Error(f"R28 host replay changed: {relative}")
    scientific_source = source_metrics(source_stored); host_config = json_object(host_config_path); scientific_import = import_metrics(root, host_config, import_stored); expected_observations = 36 * len(HOST_INITIALIZATIONS); expected_lifecycle = scientific_import["packages"] * len(HOST_INITIALIZATIONS)
    source_pass = scientific_source["rows"] == 72 and all(scientific_source[name] >= 68 for name in ("parse", "answers", "labels")) and scientific_source["answer_quorums"] == 12 and scientific_source["label_quorums"] == 12 and scientific_source["compiled_facts"] == 12 and scientific_source["packages"] >= 4 and scientific_source["evaluation"] == 36 and scientific_source["package_exact"] == 36 and scientific_source["teacher_agreement"] >= 32 and scientific_source["other_abstain"] == 36 and scientific_source["control_exact"] / 36 <= 0.1 and scientific_source["oracle_fields_consumed"] == 0 and scientific_source["accepted_labels_consumed"] == 0
    import_pass = scientific_import["builds"] == scientific_import["packages"] * len(BUILD_INITIALIZATIONS) and scientific_import["reproducible_builds"] == scientific_import["builds"] and scientific_import["observations"] == expected_observations and all(scientific_import[name] == expected_observations for name in ("oracle", "teacher", "cpu_gpu", "other_abstain")) and scientific_import["core_rows"] == 720 and scientific_import["core_exact"] == 720 and scientific_import["leakage"] == 0 and scientific_import["lifecycle_rows"] == expected_lifecycle and scientific_import["lifecycle"] == expected_lifecycle and scientific_import["corruptions"] == expected_lifecycle
    hostile = 0
    with tempfile.TemporaryDirectory(prefix="r28-hostile-") as raw:
        for mode in ("missing", "hash", "stale"):
            value = json_object(import_config)
            if mode == "missing": value["source_packages"][0]["path"] = "absent.abipkg"
            elif mode == "hash": value["source_packages"][0]["sha256"] = "0" * 64
            else: value["source_result"]["sha256"] = "f" * 64
            path = Path(raw) / f"{mode}.json"; path.write_text(json.dumps(value), encoding="utf-8")
            try: run_import(path, Path(raw) / f"out-{mode}")
            except (R14Error, FileNotFoundError): hostile += 1
    if not source_pass or not import_pass or hostile != 3 or source_value.get("verdict") != "PASS" or import_value.get("verdict") != "PASS": raise R14Error("R28 strict scientific gate failed")
    return {"format": "abi-r28-strict-live-verification/1", "status": "PASS_STRICTLY_VERIFIED_ERROR_CORRECTING_OPEN_LABEL_IMPORT", "scientific_claim": "BOUNDED_HELDOUT_ERROR_CORRECTING_FREE_LABEL_DISCOVERY_AND_LAYERCAKE_IMPORT", "claim_ceiling": "NOT_EXHAUSTIVE_TEACHER_DIAGNOSIS_OR_FLUENT_ENGLISH_EXTRACTION", "config_sha256": sha256_file(config_path), "reveal_sha256": sha256_file(reveal_path), "stored_source_sha256": sha256_file(source_stored / "result.json"), "stored_import_sha256": sha256_file(import_stored / "result.json"), "source_metrics_recomputed": scientific_source, "import_metrics_recomputed": scientific_import, "source_rows_live_replayed": 72, "layercake_rows_live_replayed": sum(len(rows(replay_import / name)) for name in IMPORT_ROWS), "source_packages_rebuilt": scientific_source["packages"], "layercake_packages_rebuilt": scientific_import["packages"] * len(BUILD_INITIALIZATIONS), "hostile_cases_rejected": hostile, "hostile_cases": 3, "stored_scientific_booleans_trusted": False, "source_parameters_copied": 0, "receiver_training_steps": 0, "teacher_present_at_layercake_execution": False, "full_abi_moonshot": "OPEN"}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--reveal", type=Path, required=True); parser.add_argument("--source-stored", type=Path, required=True); parser.add_argument("--import-config", type=Path, required=True); parser.add_argument("--import-stored", type=Path, required=True); parser.add_argument("--replay", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); result = verify(args.config, args.reveal, args.source_stored, args.import_config, args.import_stored, args.replay); result["evidence_sha256"] = selfless_evidence_hash(result); write_json_once(args.output, result); print(json.dumps(result, indent=2))
if __name__ == "__main__": main()

