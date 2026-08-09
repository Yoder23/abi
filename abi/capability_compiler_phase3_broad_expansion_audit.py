"""Audit the preregistered broad-English Phase 3 expansion catalog.

This is deliberately a no-model audit.  It proves source/split identity,
English-core segregation, balance, and exact prompt novelty before any teacher
generation is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .hf_extraction import load_probe_catalog
from .natural_english_catalog import BUILDERS


RESULT_SCHEMA = "abi-phase3-broad-expansion-catalog-audit/1"


class BroadExpansionAuditError(RuntimeError):
    """Raised when a catalog violates the preregistered expansion contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_catalog(
    candidate_path: Path,
    historical_path: Path,
    *,
    expected_story_counts: dict[str, int],
    expected_train_sha256: str,
    expected_validation_sha256: str,
) -> dict[str, Any]:
    candidate = load_probe_catalog(candidate_path)
    historical = load_probe_catalog(historical_path)
    generation = candidate.get("generation", {})
    manifest = candidate.get("raw_corpus", {})

    failures: list[str] = []
    if generation.get("story_counts") != expected_story_counts:
        failures.append("story_counts_mismatch")
    if manifest.get("train_arrow", {}).get("sha256") != expected_train_sha256:
        failures.append("train_corpus_sha256_mismatch")
    if (
        manifest.get("validation_arrow", {}).get("sha256")
        != expected_validation_sha256
    ):
        failures.append("validation_corpus_sha256_mismatch")
    if manifest.get("closed_book_fact_prompts") != 0:
        failures.append("closed_book_fact_prompt_claim_nonzero")
    if generation.get("final_test_used_for_selection") is not False:
        failures.append("final_test_selection_boundary_missing")

    probes = candidate["probes"]
    counts = Counter((row["split"], row["capability"]) for row in probes)
    expected_counts = {
        (split, capability): count
        for split, count in expected_story_counts.items()
        for capability in BUILDERS
    }
    if counts != Counter(expected_counts):
        failures.append("capability_balance_mismatch")

    segregation_failures = 0
    label_hash_failures = 0
    for row in probes:
        if not (
            row.get("destination_scope") == "english_core"
            and row.get("domain") == "domain_independent"
            and row.get("domain_labels") == []
            and row.get("domain_claims") == []
            and row.get("output_introduces_unsupplied_facts") is False
        ):
            segregation_failures += 1
        if not isinstance(row.get("label_evidence_sha256"), str):
            label_hash_failures += 1
    if segregation_failures:
        failures.append("english_core_segregation_failure")
    if label_hash_failures:
        failures.append("label_evidence_missing")

    contexts = generation.get("context_sha256_by_split", {})
    context_sets = {name: set(values) for name, values in contexts.items()}
    split_names = sorted(expected_story_counts)
    context_overlap: dict[str, int] = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            key = f"{left}__{right}"
            context_overlap[key] = len(context_sets.get(left, set()) & context_sets.get(right, set()))
    if any(context_overlap.values()):
        failures.append("source_context_split_overlap")

    historical_prompts = {row["prompt"].strip() for row in historical["probes"]}
    candidate_prompts = {row["prompt"].strip() for row in probes}
    duplicate_candidate_prompts = len(probes) - len(candidate_prompts)
    if duplicate_candidate_prompts:
        failures.append("duplicate_candidate_prompts")
    exact_historical_prompt_overlap = len(candidate_prompts & historical_prompts)
    if exact_historical_prompt_overlap:
        failures.append("historical_prompt_overlap")

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "candidate_catalog": {
            "path": candidate_path.as_posix(),
            "sha256": _sha256(candidate_path),
            "probe_count": len(probes),
            "unique_prompt_count": len(candidate_prompts),
            "duplicate_prompt_count": duplicate_candidate_prompts,
        },
        "historical_catalog": {
            "path": historical_path.as_posix(),
            "sha256": _sha256(historical_path),
            "probe_count": len(historical["probes"]),
        },
        "story_counts": generation.get("story_counts"),
        "capability_split_counts": {
            f"{split}:{capability}": counts[(split, capability)]
            for split in ("search", "validation", "final_test")
            for capability in BUILDERS
        },
        "segregation_failures": segregation_failures,
        "label_hash_failures": label_hash_failures,
        "context_overlap_by_split_pair": context_overlap,
        "exact_historical_prompt_overlap": exact_historical_prompt_overlap,
        "failures": failures,
        "claim_boundary": (
            "Catalog construction audit only. PASS does not authorize teacher "
            "loading, neural training, final-test access, or Phase 3 promotion."
        ),
    }
    if failures:
        raise BroadExpansionAuditError(",".join(failures))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--search-stories", type=int, required=True)
    parser.add_argument("--validation-stories", type=int, required=True)
    parser.add_argument("--final-stories", type=int, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--validation-sha256", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"result is immutable: {output}")
    result = audit_catalog(
        Path(args.candidate).resolve(),
        Path(args.historical).resolve(),
        expected_story_counts={
            "search": args.search_stories,
            "validation": args.validation_stories,
            "final_test": args.final_stories,
        },
        expected_train_sha256=args.train_sha256.casefold(),
        expected_validation_sha256=args.validation_sha256.casefold(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
