"""Disclosed R29 replay on the preserved R28 source failure."""

from __future__ import annotations

import argparse, json
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once
from experiments.factual_semantic_r16.package import load_package
from experiments.robust_labeling_r28.facts import HIDDEN_FACTS
from .isolation import run_isolated
from .protocol import answer_key, answer_packages


def run(output):
    root = Path(__file__).resolve().parents[2]
    if output.exists(): raise ValueError(f"immutable output exists: {output}")
    output.mkdir(parents=True); prior = root / "results/validated_labeling_r29/heldout_v2_source"; rows = [json.loads(line) for line in (prior / "source_rows.jsonl").read_text(encoding="utf-8").splitlines() if line]; extraction_rows = [row for row in rows if row["split"] == "extraction"]
    bundle_rows = [{"subject": row["subject"], "question": row["question"], "view": row["view"], "answer": row["teacher_answer"], "label": row["free_label"]} for row in extraction_rows]; bundle = {"format": "abi-r27-anonymous-free-label-observations/1", "records": bundle_rows}; bundle["evidence_sha256"] = evidence_hash(bundle); bundle_path = output / "source_bundle.json"; write_json_once(bundle_path, bundle); extraction = run_isolated(root, bundle_path, output / "extraction")
    packages = [load_package(output / "extraction" / item["path"]) for item in extraction["result"]["packages"]]; evaluation = []
    for row in (item for item in rows if item["split"] == "evaluation"):
        namespace = f"teacher/{row['oracle_domain']}"; selected = [item for item in packages if item["namespace"] == namespace]; prediction = answer_packages(selected, row["question"]); evaluation.append({"fact_id": row["fact_id"], "question": row["question"], "oracle_answer": row["oracle_answer"], "teacher_answer": row["teacher_answer"], "package_answer": prediction})
    evaluation_path = output / "evaluation.jsonl"; evaluation_path.write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in evaluation))
    validator_by_subject = {row["subject_sha256"]: row["validator_used"] for row in extraction["result"]["validation"]}
    unvalidated = [row for row in rows if not validator_by_subject[__import__("hashlib").sha256(row["subject"].encode()).hexdigest()]]
    metrics = {"source_records": len(bundle_rows), "all_source_rows": len(rows), "unvalidated_source_rows": len(unvalidated), "unvalidated_source_exact": sum(row["answer_exact"] for row in unvalidated), "unique_facts": extraction["result"]["unique_facts"], "package_fact_records": extraction["result"]["package_fact_records"], "validator_uses": extraction["result"]["validator_uses"], "packages": len(packages), "evaluation_rows": len(evaluation), "teacher_oracle_exact": sum(answer_key(row["teacher_answer"]) == answer_key(row["oracle_answer"]) for row in evaluation), "package_oracle_exact": sum(answer_key(str(row["package_answer"] or "")) == answer_key(row["oracle_answer"]) for row in evaluation), "oracle_fields_consumed": extraction["result"]["oracle_fields_consumed"]}
    passed = metrics["source_records"] == 36 and metrics["all_source_rows"] == 72 and metrics["unvalidated_source_rows"] == 36 and metrics["unvalidated_source_exact"] >= 32 and metrics["unique_facts"] == 12 and metrics["package_fact_records"] >= 12 and metrics["validator_uses"] == 6 and metrics["packages"] == 4 and metrics["evaluation_rows"] == 36 and metrics["teacher_oracle_exact"] < metrics["package_oracle_exact"] and metrics["package_oracle_exact"] == 36 and metrics["oracle_fields_consumed"] == 0
    result = {"format": "abi-r29-public-stratified-qualification/3", "verdict": "PASS" if passed else "FAIL", "claim": "DISCLOSED_STRATIFIED_PROVENANCE_SCORER", "claim_ceiling": "POSTHOC_ON_V2_ROWS_NOT_FRESH_HELDOUT_OR_LAYERCAKE_IMPORT", "r29_v2_source_rows_sha256": sha256_file(prior / "source_rows.jsonl"), "metrics": metrics, "packages": extraction["result"]["packages"], "artifacts": {"evaluation": {"path": evaluation_path.name, "sha256": sha256_file(evaluation_path)}}, "full_abi_moonshot": "OPEN"}; result["evidence_sha256"] = evidence_hash(result); write_json_once(output / "result.json", result); return result
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); print(json.dumps(run(args.output), indent=2))
if __name__ == "__main__": main()
