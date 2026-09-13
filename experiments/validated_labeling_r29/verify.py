"""Strict live end-to-end verifier for R29."""

from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path

from experiments.autonomous_labeling_r27.import_run import BUILD_INITIALIZATIONS, HOST_INITIALIZATIONS
from experiments.autonomous_labeling_r27.prepare_import import run as prepare_import
from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file, write_json_once
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash
from .binding import load_config
from .import_run import run as run_import
from .protocol import answer_key
from .source_run import run as run_source


IMPORT_ROWS = ("package_builds.jsonl", "domain_observations.jsonl", "english_immutability.jsonl", "lifecycle.jsonl", "english_leakage.jsonl")
def rows(path): return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
def source_metrics(directory):
    raw = rows(directory / "source_rows.jsonl"); evaluation = rows(directory / "evaluation.jsonl"); extraction = json_object(directory / "extraction/result.json"); validated = [row for row in evaluation if row["validator_used"]]; unvalidated = [row for row in evaluation if not row["validator_used"]]
    import hashlib
    validator_by_subject = {row["subject_sha256"]: row["validator_used"] for row in extraction["validation"]}
    unvalidated_raw = [row for row in raw if not validator_by_subject[hashlib.sha256(row["subject"].encode()).hexdigest()]]
    return {"source_rows": len(raw), "parse": sum(row["parse_exact"] for row in raw), "raw_answers": sum(row["answer_exact"] for row in raw), "unvalidated_source_rows": len(unvalidated_raw), "unvalidated_source_exact": sum(row["answer_exact"] for row in unvalidated_raw), "label_syntax": sum(row["label_syntax_valid"] for row in raw), "unique_facts": extraction["unique_facts"], "package_fact_records": extraction["package_fact_records"], "validator_uses": extraction["validator_uses"], "packages": len(extraction["packages"]), "evaluation_rows": len(evaluation), "oracle_coverage": sum(row["oracle_domain"] in row["compiled_tags"] for row in evaluation), "tag_purity": sum(set(row["compiled_tags"]).issubset(set(row["allowed_tags"])) for row in evaluation), "package_exact": sum(answer_key(str(row["package_answer"] or "")) == answer_key(row["oracle_answer"]) for row in evaluation), "source_exact": sum(answer_key(row["teacher_answer"]) == answer_key(row["oracle_answer"]) for row in evaluation), "unauthorized_abstain": sum(row["unauthorized_answer"] is None for row in evaluation), "control_validated_exact": sum(answer_key(str(row["control_answer"] or "")) == answer_key(row["oracle_answer"]) for row in validated), "control_unvalidated_exact": sum(answer_key(str(row["control_answer"] or "")) == answer_key(row["oracle_answer"]) for row in unvalidated), "validated_rows": len(validated), "unvalidated_rows": len(unvalidated), "oracle_fields_consumed": extraction["oracle_fields_consumed"]}
def import_metrics(root, host_config, directory):
    result = json_object(directory / "result.json"); observations = rows(directory / "domain_observations.jsonl"); core = rows(directory / "english_immutability.jsonl"); lifecycle = rows(directory / "lifecycle.jsonl"); leakage = rows(directory / "english_leakage.jsonl"); builds = rows(directory / "package_builds.jsonl"); stored = rows(root / host_config["r23_hidden_rows"]["path"]); expected_english = {row["record_id"]: row["output"] for row in stored if row.get("method") == "abi_factorized" and row.get("seed") == 21022}; packages = len(result["canonical_packages"])
    return {"packages": packages, "builds": len(builds), "reproducible": sum(row["byte_identical"] for row in builds), "observations": len(observations), "oracle": sum(answer_key(row["gpu_output"]) == answer_key(row["oracle_answer"]) for row in observations), "teacher": sum(answer_key(row["gpu_output"]) == answer_key(row["teacher_answer"]) for row in observations), "cpu_gpu": sum(row["cpu_output"] == row["gpu_output"] for row in observations), "unauthorized_abstain": sum(all(not text for namespace, text in row["other_outputs"].items() if namespace.split("/", 1)[1] not in row["allowed_tags"]) for row in observations), "core_rows": len(core), "core_exact": sum(row["output"] == expected_english[row["record_id"]] for row in core), "leakage": sum(answer_key(row["output"]) == answer_key(row["target"]) for row in leakage), "lifecycle_rows": len(lifecycle), "lifecycle": sum(row["removed"] and row["absent_rejected"] and row["before"] == row["after"] and row["restored_sha256"] == row["expected_sha256"] for row in lifecycle), "corruptions": sum(row["corruption_rejected"] for row in lifecycle)}
def verify(config_path, reveal_path, source_stored, import_config, import_stored, replay):
    root = Path(__file__).resolve().parents[2]; config = load_config(root, config_path); source_result = json_object(source_stored / "result.json"); import_result = json_object(import_stored / "result.json")
    if source_result.get("evidence_sha256") != selfless_evidence_hash(source_result) or import_result.get("evidence_sha256") != selfless_evidence_hash(import_result): raise R14Error("R29 stored evidence hash changed")
    replay_source = replay / "source"; replay_import_config = replay / "import_config.json"; replay_import = replay / "import"; run_source(config_path, reveal_path, replay_source)
    for relative in ("source_rows.jsonl", "source_bundle.json", "evaluation.jsonl"):
        if (source_stored / relative).read_bytes() != (replay_source / relative).read_bytes(): raise R14Error(f"R29 source replay changed: {relative}")
    replay_source_result = json_object(replay_source / "result.json")
    for item in source_result["packages"]:
        candidate = next(value for value in replay_source_result["packages"] if value["namespace"] == item["namespace"])
        if (source_stored / item["path"]).read_bytes() != (replay_source / candidate["path"]).read_bytes(): raise R14Error("R29 source package replay changed")
    host_path = root / config["r27_host_config"]["path"]; prepare_import(host_path, replay_source, replay_import_config); run_import(replay_import_config, replay_import)
    for relative in IMPORT_ROWS:
        left, right = rows(import_stored / relative), rows(replay_import / relative)
        if relative == "package_builds.jsonl": left = [{key: value for key, value in row.items() if key != "path"} for row in left]; right = [{key: value for key, value in row.items() if key != "path"} for row in right]
        if left != right: raise R14Error(f"R29 LayerCake replay changed: {relative}")
    source = source_metrics(source_stored); host_config = json_object(host_path); imported = import_metrics(root, host_config, import_stored); expected_observations = 36 * len(HOST_INITIALIZATIONS); expected_lifecycle = imported["packages"] * len(HOST_INITIALIZATIONS)
    source_pass = source["source_rows"] == 72 and source["parse"] >= 68 and source["unvalidated_source_rows"] == 36 and source["unvalidated_source_exact"] >= 32 and source["label_syntax"] >= 68 and source["unique_facts"] == 12 and source["package_fact_records"] >= 12 and source["validator_uses"] == 6 and source["packages"] >= 4 and source["evaluation_rows"] == 36 and source["oracle_coverage"] == 36 and source["tag_purity"] == 36 and source["package_exact"] == 36 and source["package_exact"] >= source["source_exact"] and source["unauthorized_abstain"] == 36 and source["control_validated_exact"] == source["validated_rows"] and source["control_unvalidated_exact"] <= 2 and source["oracle_fields_consumed"] == 0
    import_pass = imported["builds"] == imported["packages"] * len(BUILD_INITIALIZATIONS) and imported["reproducible"] == imported["builds"] and imported["observations"] == expected_observations and imported["oracle"] == expected_observations and imported["teacher"] >= 32 * len(HOST_INITIALIZATIONS) and imported["cpu_gpu"] == expected_observations and imported["unauthorized_abstain"] == expected_observations and imported["core_rows"] == imported["core_exact"] == 720 and imported["leakage"] == 0 and imported["lifecycle_rows"] == imported["lifecycle"] == imported["corruptions"] == expected_lifecycle
    hostile = 0
    with tempfile.TemporaryDirectory(prefix="r29-hostile-") as raw:
        for mode in ("missing", "hash", "stale"):
            value = json_object(import_config)
            if mode == "missing": value["source_packages"][0]["path"] = "absent.abipkg"
            elif mode == "hash": value["source_packages"][0]["sha256"] = "0" * 64
            else: value["source_result"]["sha256"] = "f" * 64
            path = Path(raw) / f"{mode}.json"; path.write_text(json.dumps(value), encoding="utf-8")
            try: run_import(path, Path(raw) / f"output-{mode}")
            except (R14Error, FileNotFoundError): hostile += 1
    if not source_pass or not import_pass or hostile != 3 or source_result.get("verdict") != "PASS" or import_result.get("verdict") != "PASS": raise R14Error("R29 strict scientific gate failed")
    return {"format": "abi-r29-strict-live-verification/1", "status": "PASS_STRICTLY_VERIFIED_VALIDATOR_BACKED_MULTITAG_IMPORT", "scientific_claim": "BOUNDED_HELDOUT_VALIDATED_LABELING_AND_LAYERCAKE_IMPORT", "claim_ceiling": "NOT_EXHAUSTIVE_DIAGNOSIS_OR_FLUENT_ENGLISH_EXTRACTION", "config_sha256": sha256_file(config_path), "reveal_sha256": sha256_file(reveal_path), "stored_source_sha256": sha256_file(source_stored / "result.json"), "stored_import_sha256": sha256_file(import_stored / "result.json"), "source_metrics_recomputed": source, "import_metrics_recomputed": imported, "source_rows_live_replayed": 72, "layercake_rows_live_replayed": sum(len(rows(replay_import / relative)) for relative in IMPORT_ROWS), "source_packages_rebuilt": source["packages"], "layercake_packages_rebuilt": imported["packages"] * len(BUILD_INITIALIZATIONS), "hostile_cases_rejected": hostile, "hostile_cases": 3, "stored_scientific_booleans_trusted": False, "source_parameters_copied": 0, "receiver_training_steps": 0, "teacher_present_at_layercake_execution": False, "full_abi_moonshot": "OPEN"}
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--reveal", type=Path, required=True); parser.add_argument("--source-stored", type=Path, required=True); parser.add_argument("--import-config", type=Path, required=True); parser.add_argument("--import-stored", type=Path, required=True); parser.add_argument("--replay", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); result = verify(args.config, args.reveal, args.source_stored, args.import_config, args.import_stored, args.replay); result["evidence_sha256"] = selfless_evidence_hash(result); write_json_once(args.output, result); print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
