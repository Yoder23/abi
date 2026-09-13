"""Audit the stale evaluator lineage in R60 without rewriting its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from abi.hf_extraction import evaluate_output, load_probe_catalog
from experiments.foreign_capability_r14.core import write_json_once, write_jsonl_once


FROZEN = {
    "catalogs/prospective_endpoint_r60_v1.json":
        "4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7",
    "results/prospective_endpoint_r60/live_source_v1/source_outputs.jsonl":
        "35a5a6e1035035db1663d793257274e3be2b766aab79a00a0476478d0428b596",
    "results/prospective_endpoint_r60/screen_v1/evaluation.jsonl":
        "e509484babfab685c91fee3a6165bfea0b2881213fdb7b3a71fde9266e2a25cf",
    "results/invariance_augmented_adapters_r63/screen_r60_v1a/evaluation.jsonl":
        "39012f3bae8ec26d5a17c1adeb63d6da935b229fd80d71a6c59dd91af2c0e266",
}

ABSTENTION_V3 = {
    "kind": "contains_any",
    "values": [
        "cannot know", "can't know", "do not know", "don't know", "unknown",
        "cannot determine", "not enough information", "unknowable",
        "don't have the capability", "do not have the capability",
        "no way to know", "cannot provide", "can't provide", "cannot predict",
        "can't predict", "not possible", "impossible", "cannot assist",
    ],
}


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        raise AuditError(f"missing raw rows: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"unreadable raw rows: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise AuditError(f"non-object raw row: {path}")
    return rows


def repaired_evaluator(probe: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
    capability = str(probe["capability"])
    if capability == "abstention":
        return "diagnostic_rescore", ABSTENTION_V3, "v3 expanded valid refusal language"
    if capability == "domain_independent_reasoning":
        expected = str(probe["evaluator"]["value"])
        return (
            "diagnostic_rescore",
            {"kind": "contains_all", "values": [expected]},
            "v2 accepts the correct nonce conclusion inside an explanation",
        )
    if capability == "coherence":
        return "censored", None, "v3 raised max_new_tokens from 40 to 96"
    if capability == "format_control":
        return (
            "incompatible_contract",
            None,
            "v2 replaced the JSON task with an exact two-line plain-text task",
        )
    return "unchanged", dict(probe["evaluator"]), "no later evaluator repair"


def run(root: Path, output: Path) -> dict[str, Any]:
    resolved = {name: root / name for name in FROZEN}
    for name, path in resolved.items():
        if not path.is_file() or sha256_file(path) != FROZEN[name]:
            raise AuditError(f"frozen input absent or changed: {name}")

    catalog = load_probe_catalog(resolved["catalogs/prospective_endpoint_r60_v1.json"])
    probes = {str(row["probe_id"]): row for row in catalog["probes"]}
    source_rows = {
        str(row["probe_id"]): row
        for row in read_jsonl(
            resolved["results/prospective_endpoint_r60/live_source_v1/source_outputs.jsonl"]
        )
    }
    r59_rows = {
        str(row["probe_id"]): row
        for row in read_jsonl(
            resolved["results/prospective_endpoint_r60/screen_v1/evaluation.jsonl"]
        )
    }
    r63_rows = {
        str(row["probe_id"]): row
        for row in read_jsonl(
            resolved[
                "results/invariance_augmented_adapters_r63/screen_r60_v1a/evaluation.jsonl"
            ]
        )
    }
    expected_ids = set(probes)
    if len(expected_ids) != 1400 or any(set(rows) != expected_ids for rows in (source_rows, r59_rows, r63_rows)):
        raise AuditError("input row identities are not the same 1,400-probe matrix")

    raw_rows: list[dict[str, Any]] = []
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    per_capability: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for probe_id in sorted(expected_ids):
        probe = probes[probe_id]
        classification, evaluator, reason = repaired_evaluator(probe)
        systems = {
            "source": str(source_rows[probe_id]["output"]),
            "r59": str(r59_rows[probe_id]["candidate_output"]),
            "r63": str(r63_rows[probe_id]["output"]),
        }
        row: dict[str, Any] = {
            "probe_id": probe_id,
            "capability": probe["capability"],
            "classification": classification,
            "reason": reason,
            "original_evaluator": probe["evaluator"],
            "diagnostic_evaluator": evaluator,
            "systems": {},
        }
        for system, text in systems.items():
            original_pass, original_score = evaluate_output(text, probe["evaluator"])
            diagnostic_pass = diagnostic_score = None
            if evaluator is not None:
                diagnostic_pass, diagnostic_score = evaluate_output(text, evaluator)
                totals[classification][f"{system}_rows"] += 1
                totals[classification][f"{system}_passes"] += int(diagnostic_pass)
                cap = str(probe["capability"])
                per_capability[cap][classification][f"{system}_rows"] += 1
                per_capability[cap][classification][f"{system}_passes"] += int(diagnostic_pass)
            row["systems"][system] = {
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "original_pass": bool(original_pass),
                "original_score": float(original_score),
                "diagnostic_pass": diagnostic_pass,
                "diagnostic_score": diagnostic_score,
            }
        totals[classification]["probes"] += 1
        raw_rows.append(row)

    raw_path = output / "row_audit.jsonl"
    result_path = output / "result.json"
    write_jsonl_once(raw_path, raw_rows)
    result = {
        "format": "abi-r64-evaluator-lineage-audit/1",
        "verdict": "R60_NOT_PROMOTION_VALID_EVALUATOR_LINEAGE_DEFECT_CONFIRMED",
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "rows": len(raw_rows),
        "classification_counts": dict(Counter(row["classification"] for row in raw_rows)),
        "metrics": {
            key: dict(value) for key, value in sorted(totals.items())
        },
        "by_capability": {
            capability: {
                classification: dict(counts)
                for classification, counts in sorted(groups.items())
            }
            for capability, groups in sorted(per_capability.items())
        },
        "limitations": [
            "The diagnostic rescore is post-hoc development evidence only.",
            "Coherence is censored by the obsolete 40-token ceiling.",
            "Format control changed task contracts and is not rescorable.",
            "The artificial Evaluation-item prefix remains an R60 prompt defect.",
        ],
        "frozen_inputs": FROZEN,
        "artifacts": {
            "row_audit": {
                "path": "row_audit.jsonl",
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            }
        },
    }
    write_json_once(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

