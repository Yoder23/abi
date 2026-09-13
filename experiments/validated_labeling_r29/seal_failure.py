"""Seal the R29 v1 protocol-conformance failure from raw rows."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once


def run(config, reveal, source, output):
    rows = [json.loads(line) for line in (source / "source_rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
    invalid = [{"split": row["split"], "fact_id": row["fact_id"], "view": row["view"], "completion": row["completion"]} for row in rows if not row["parse_exact"]]
    result = {"format": "abi-r29-heldout-v1-conformance-failure/1", "verdict": "FAIL_CLOSED_BEFORE_PACKAGE_EMISSION", "config_sha256": sha256_file(config), "reveal_sha256": sha256_file(reveal), "source_rows_sha256": sha256_file(source / "source_rows.jsonl"), "source_bundle_sha256": sha256_file(source / "source_bundle.json"), "metrics": {"source_rows": len(rows), "parse_exact": sum(row["parse_exact"] for row in rows), "answer_exact": sum(row["answer_exact"] for row in rows), "label_syntax_valid": sum(row["label_syntax_valid"] for row in rows), "invalid_rows": len(invalid), "registered_minimum_parse": 68}, "invalid_rows": invalid, "cause": "The worker required all three rows to be valid although the frozen protocol allowed up to four parse failures and specified a 2-of-3 quorum.", "scientific_interpretation": "Implementation/protocol mismatch; no capability package or LayerCake evidence was produced.", "layercake_invoked": False, "packages_emitted": 0, "full_abi_moonshot": "OPEN"}; result["evidence_sha256"] = evidence_hash(result); write_json_once(output, result); return result
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--reveal", type=Path, required=True); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); print(json.dumps(run(args.config, args.reveal, args.source, args.output), indent=2))
if __name__ == "__main__": main()
