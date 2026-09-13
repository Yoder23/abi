"""Replay disclosed R27 rows through the physically isolated R28 compiler."""

from __future__ import annotations

import argparse, json, tempfile
from pathlib import Path

from experiments.autonomous_labeling_r27.facts import PUBLIC_FACTS
from experiments.factual_semantic_r16.package import load_package
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once
from .isolation import run_isolated
from .protocol import answer_key, answer_packages


def run(output: Path):
    root = Path(__file__).resolve().parents[2]
    if output.exists(): raise ValueError(f"immutable output exists: {output}")
    output.mkdir(parents=True)
    rows_path = root / "results/autonomous_labeling_r27/public_v2/source_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    facts = {fact.fact_id: fact for fact in PUBLIC_FACTS}; bundle_rows = []
    subjects = {
        "chem-carbon-number": "carbon", "chem-gold-symbol": "gold",
        "geo-japan-capital": "Japan", "geo-kenya-continent": "Kenya",
        "math-square-root": "144", "math-triangle-angles": "triang",
        "python-list-length": "len", "python-uppercase-method": "method",
    }
    for row in rows:
        fact = facts[row["fact_id"]]; bundle_rows.append({"subject": subjects[fact.fact_id], "question": row["question"], "view": row["view"], "answer": row["answer"], "label": row["free_label"]})
    bundle = {"format": "abi-r27-anonymous-free-label-observations/1", "records": bundle_rows}; bundle["evidence_sha256"] = evidence_hash(bundle); bundle_path = output / "source_bundle.json"; write_json_once(bundle_path, bundle)
    extraction = run_isolated(root, bundle_path, output / "extraction")
    packages = [load_package(output / "extraction" / item["path"]) for item in extraction["result"]["packages"]]
    exact = sum(answer_packages(packages, question) is not None and answer_key(answer_packages(packages, question)) == answer_key(fact.answer) for fact in PUBLIC_FACTS for question in fact.evaluation_questions)
    metrics = {"records": len(rows), "facts_accepted": extraction["result"]["facts_accepted_by_quorum"], "facts_rejected": extraction["result"]["facts_rejected_by_quorum"], "packages": len(packages), "evaluation_exact": exact, "evaluation_rows": len(PUBLIC_FACTS) * 3, "oracle_fields_consumed": extraction["result"]["oracle_fields_consumed"], "accepted_labels_consumed": extraction["result"]["accepted_labels_consumed"]}
    passed = metrics == {"records": 24, "facts_accepted": 8, "facts_rejected": 0, "packages": 5, "evaluation_exact": 24, "evaluation_rows": 24, "oracle_fields_consumed": 0, "accepted_labels_consumed": 0}
    result = {"format": "abi-r28-public-quorum-qualification/1", "verdict": "PASS" if passed else "FAIL", "claim": "DISCLOSED_ERROR_CORRECTING_COMPILER_PREREQUISITE", "claim_ceiling": "NOT_HELDOUT_OR_LAYERCAKE_IMPORT", "source_rows_sha256": sha256_file(rows_path), "metrics": metrics, "packages": extraction["result"]["packages"], "full_abi_moonshot": "OPEN"}; result["evidence_sha256"] = evidence_hash(result); write_json_once(output / "result.json", result); return result
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); print(json.dumps(run(args.output), indent=2))
if __name__ == "__main__": main()
