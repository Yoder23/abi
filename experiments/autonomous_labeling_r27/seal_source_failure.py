"""Seal the fail-closed R27 v1 source/compiler outcome from partial raw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once

from .protocol import answer_key


def run(config: Path, reveal: Path, source: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in (source / "source_rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
    bundle = json.loads((source / "source_bundle.json").read_text(encoding="utf-8"))["records"]
    groups = []
    for subject in sorted({row["subject"] for row in bundle}):
        selected = [row for row in bundle if row["subject"] == subject]
        answers = sorted({answer_key(row["answer"]) for row in selected})
        labels = sorted({row["label"] for row in selected})
        if len(answers) != 1 or len(labels) != 1:
            groups.append({"subject": subject, "answer_keys": answers, "labels": labels, "answer_consensus": len(answers) == 1, "label_consensus": len(labels) == 1})
    metrics = {
        "source_rows": len(rows), "extraction_rows": len(bundle),
        "strict_json": sum(row["parse_exact"] for row in rows),
        "oracle_answer_exact": sum(row["answer_exact"] for row in rows),
        "semantic_label_valid": sum(row["label_semantic_valid"] for row in rows),
        "inconsistent_extraction_facts": len(groups),
    }
    result = {
        "format": "abi-r27-heldout-source-failure/1", "verdict": "FAIL_CLOSED_BEFORE_PACKAGE_EMISSION",
        "claim": "R27_V1_OPEN_LABEL_COMPILER_PREREQUISITE_FAILED",
        "config_sha256": sha256_file(config), "reveal_sha256": sha256_file(reveal),
        "source_rows_sha256": sha256_file(source / "source_rows.jsonl"), "source_bundle_sha256": sha256_file(source / "source_bundle.json"),
        "metrics": metrics, "inconsistent_groups": groups,
        "layercake_invoked": False, "packages_emitted": 0,
        "interpretation": "Teacher view inconsistency in an answer and a free label prevented fail-closed isolated compilation; this is an ABI acquisition/labeling failure, not a LayerCake host result.",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result); write_json_once(output, result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--reveal", type=Path, required=True); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(run(args.config, args.reveal, args.source, args.output), indent=2))


if __name__ == "__main__": main()
