"""Seal the R28 hidden quorum failure from immutable partial source rows."""

from __future__ import annotations

import argparse, json
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once
from .protocol import answer_key, unique_quorum


def run(config, reveal, source, output):
    raw = [json.loads(line) for line in (source / "source_rows.jsonl").read_text(encoding="utf-8").splitlines() if line]; bundle = json.loads((source / "source_bundle.json").read_text(encoding="utf-8"))["records"]; by_subject = {}
    for row in bundle: by_subject.setdefault(row["subject"], []).append(row)
    groups = []
    for subject, rows in sorted(by_subject.items()):
        answer = unique_quorum([row["answer"] for row in rows], answer_key); label = unique_quorum([row["label"] for row in rows]); oracle = next(row["oracle_answer"] for row in raw if row["subject"] == subject)
        groups.append({"subject": subject, "answers": [row["answer"] for row in rows], "labels": [row["label"] for row in rows], "answer_quorum": answer, "label_quorum": label, "answer_quorum_oracle_exact": answer == answer_key(oracle)})
    metrics = {"source_rows": len(raw), "extraction_rows": len(bundle), "strict_json": sum(row["parse_exact"] for row in raw), "oracle_answer_exact": sum(row["answer_exact"] for row in raw), "semantic_label_valid": sum(row["label_semantic_valid"] for row in raw), "answer_quorums": sum(row["answer_quorum"] is not None for row in groups), "oracle_correct_answer_quorums": sum(row["answer_quorum_oracle_exact"] for row in groups), "label_quorums": sum(row["label_quorum"] is not None for row in groups)}
    failures = [row for row in groups if row["label_quorum"] is None or not row["answer_quorum_oracle_exact"]]
    result = {"format": "abi-r28-heldout-source-failure/1", "verdict": "FAIL_CLOSED_BEFORE_PACKAGE_EMISSION", "claim": "R28_QUORUM_ONLY_ARCHITECTURE_FAILED", "config_sha256": sha256_file(config), "reveal_sha256": sha256_file(reveal), "source_rows_sha256": sha256_file(source / "source_rows.jsonl"), "source_bundle_sha256": sha256_file(source / "source_bundle.json"), "metrics": metrics, "failing_groups": failures, "layercake_invoked": False, "packages_emitted": 0, "interpretation": "Unique majority voting cannot correct a confidently wrong answer and forced single labels cannot represent a math/Python-overlapping fact.", "next_architecture": "multi-label taxonomy plus independent validator-backed quarantine", "full_abi_moonshot": "OPEN"}; result["evidence_sha256"] = evidence_hash(result); write_json_once(output, result); return result
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--reveal", type=Path, required=True); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); print(json.dumps(run(args.config, args.reveal, args.source, args.output), indent=2))
if __name__ == "__main__": main()
