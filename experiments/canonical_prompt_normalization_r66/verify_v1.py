"""Fail-closed stored-evidence verifier for the R66 development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from abi.capability_compiler_phase2_common import sha256_file
from abi.hf_extraction import evaluate_output, load_probe_catalog
from experiments.canonical_prompt_normalization_r66.screen_v1 import (
    CANDIDATE_SHA256, CATALOG_SHA256, METADATA_SHA256,
    NORMALIZATION_RECEIPT_SHA256, SOURCE_OUTPUTS_SHA256,
)
from experiments.evaluator_lineage_audit_r64.audit_v1 import repaired_evaluator


def _rows(path: Path) -> list[dict]:
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit(f"FAIL: missing raw evidence {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"FAIL: unreadable raw evidence {path}: {error}")
    if any(not isinstance(row, dict) for row in rows):
        raise SystemExit("FAIL: non-object raw evidence")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    candidate, catalog, source, evidence = map(
        Path.resolve, (args.candidate, args.catalog, args.source, args.evidence)
    )
    required = (
        (candidate / "model.safetensors", CANDIDATE_SHA256),
        (candidate / "metadata.json", METADATA_SHA256),
        (candidate / "normalization_receipt.json", NORMALIZATION_RECEIPT_SHA256),
        (catalog, CATALOG_SHA256), (source, SOURCE_OUTPUTS_SHA256),
    )
    for path, digest in required:
        if not path.is_file() or sha256_file(path) != digest:
            raise SystemExit(f"FAIL: frozen input absent or changed: {path}")
    result_path, raw_path = evidence / "result.json", evidence / "evaluation.jsonl"
    if not result_path.is_file() or not raw_path.is_file():
        raise SystemExit("FAIL: R66 result or raw rows missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = _rows(raw_path)
    probes = {str(row["probe_id"]): row for row in load_probe_catalog(catalog)["probes"]}
    source_rows = {str(row["probe_id"]): row for row in _rows(source)}
    if len(rows) != 1400 or len(probes) != 1400 or len(source_rows) != 1400:
        raise SystemExit("FAIL: R66 matrix is incomplete")
    if len({str(row["probe_id"]) for row in rows}) != 1400:
        raise SystemExit("FAIL: duplicate R66 row identity")
    recomputed = Counter()
    per_capability: dict[str, Counter] = {}
    for row in rows:
        probe_id = str(row["probe_id"])
        probe = probes.get(probe_id)
        source_row = source_rows.get(probe_id)
        if probe is None or source_row is None:
            raise SystemExit("FAIL: R66 row is outside frozen matrix")
        prompt = str(probe["prompt"])
        classification, evaluator, reason = repaired_evaluator(probe)
        if (
            row.get("capability") != probe["capability"]
            or row.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or row.get("classification") != classification
            or row.get("classification_reason") != reason
            or row.get("evaluator") != evaluator
            or row.get("source_output_sha256") != source_row["output_sha256"]
            or row.get("output_sha256") != hashlib.sha256(str(row["output"]).encode("utf-8")).hexdigest()
        ):
            raise SystemExit(f"FAIL: R66 row binding changed: {probe_id}")
        if evaluator is not None:
            candidate_pass, candidate_score = evaluate_output(str(row["output"]), evaluator)
            source_pass, source_score = evaluate_output(str(source_row["output"]), evaluator)
            if (
                row.get("functional_pass") is not candidate_pass
                or row.get("functional_score") != candidate_score
                or row.get("source_pass") is not source_pass
                or row.get("source_score") != source_score
            ):
                raise SystemExit(f"FAIL: R66 score is not recomputable: {probe_id}")
            recomputed["adjudicable"] += 1
            recomputed["candidate"] += int(candidate_pass)
            recomputed["source"] += int(source_pass)
            recomputed["regressions"] += int(source_pass and not candidate_pass)
        else:
            if any(row.get(name) is not None for name in ("functional_pass", "functional_score", "source_pass", "source_score")):
                raise SystemExit(f"FAIL: censored R66 row was scored: {probe_id}")
        recomputed["collapses"] += int(row["final_collapse"]["collapse_detected"])
        recomputed["physical"] += int(row["physical_sparse"])
        cap = str(row["capability"])
        per_capability.setdefault(cap, Counter())
        per_capability[cap]["candidate"] += int(row.get("functional_pass") is True)
    metrics = result.get("metrics", {})
    if (
        sha256_file(raw_path) != result.get("artifacts", {}).get("evaluation", {}).get("sha256")
        or recomputed["adjudicable"] != metrics.get("adjudicable_rows")
        or recomputed["candidate"] != metrics.get("adjudicable_candidate_passes")
        or recomputed["source"] != metrics.get("adjudicable_source_passes")
        or recomputed["regressions"] != metrics.get("source_passing_regressions")
        or recomputed["collapses"] != metrics.get("final_collapses")
        or recomputed["physical"] != metrics.get("physical_sparse_rows")
        or result.get("promotion_eligible") is not False
        or result.get("full_abi_moonshot") != "OPEN"
        or result.get("verdict") != "FAIL_R66_CLOSE_NORMALIZATION_BRANCH"
    ):
        raise SystemExit("FAIL: R66 aggregate or claim boundary changed")
    print("PASS: R66 stored evidence recomputes and the failed branch remains closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
