"""Build a fresh broad-context abstention supplement after evaluator failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .broad_english_catalog import (
    DEFAULT_CORE_EXCLUSION_MARKERS,
    DEFAULT_MAX_CONTEXT_CHARACTERS,
    _arrow_texts,
    _corpus_manifest,
    _minhash_sample,
    _prompt,
    _story_excerpt,
)
from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256
from .natural_abstention_catalog import ABSTENTION_MARKERS


CATALOG_FORMAT = "abi-phase3-broad-abstention-supplement/1"
DEFAULT_SEED = 79_824


class BroadAbstentionError(RuntimeError):
    """Raised when the fresh abstention supplement is not independent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_texts(texts: Iterable[str], excluded_context_hashes: set[str]) -> Iterable[str]:
    for text in texts:
        try:
            excerpt = _story_excerpt(text, DEFAULT_MAX_CONTEXT_CHARACTERS)
        except Exception:
            continue
        if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() not in excluded_context_hashes:
            yield text


def _evaluator() -> dict[str, Any]:
    return {
        "kind": "all_of",
        "rules": [
            {"kind": "nonempty", "minimum_characters": 8},
            {"kind": "maximum_characters", "value": 600},
            {"kind": "contains_none", "values": list(DEFAULT_CORE_EXCLUSION_MARKERS)},
            {"kind": "contains_any", "values": list(ABSTENTION_MARKERS)},
        ],
    }


def build_catalog(
    stories: list[str],
    *,
    corpus_manifest: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    probes = []
    context_hashes = []
    for index, story in enumerate(stories):
        excerpt = _story_excerpt(story, DEFAULT_MAX_CONTEXT_CHARACTERS)
        context_sha = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        context_hashes.append(context_sha)
        prompt, _ = _prompt("abstention", excerpt)
        probe: dict[str, Any] = {
            "probe_id": f"broad-abs-v2-search-{index:04d}",
            "destination_scope": "english_core",
            "capability": "abstention",
            "canonical_capability": "abstention",
            "domain": "domain_independent",
            "split": "search",
            "prompt": prompt,
            "max_new_tokens": 96,
            "temperature": 0,
            "seed": seed + index,
            "evaluator": _evaluator(),
            "record_schema": SEGREGATED_RECORD_SCHEMA,
            "knowledge_class": LINGUISTIC_FORM,
            "content_basis": "domain_free_instruction",
            "domain_labels": [],
            "domain_claims": [],
            "label_method": "preregistered_catalog",
            "output_introduces_unsupplied_facts": False,
            "raw_context_sha256": context_sha,
            "phase3_template_family": "fresh_broad_abstention_v2",
        }
        probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
        probes.append(probe)
    prompts = {row["prompt"] for row in probes}
    if len(prompts) != len(probes) or len(set(context_hashes)) != len(probes):
        raise BroadAbstentionError("fresh supplement contains duplicate prompt or context")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-phase3-fresh-broad-abstention-v2",
        "catalog_contract_schema": CATALOG_FORMAT,
        "status": "PREREGISTERED_FRESH_SEARCH_ONLY_AFTER_V112_EVALUATOR_FAILURE",
        "claim_boundary": "Fresh abstention teacher queries only. V112 failures remain failed and are not reclassified.",
        "generation": {
            "generator": "abi.capability_compiler_phase3_broad_abstention",
            "seed": seed,
            "search_records": len(probes),
            "validation_records": 0,
            "final_records": 0,
            "repair_rounds": 0,
            "excluded_prior_contexts": True,
            "context_sha256": sorted(context_hashes),
        },
        "observed_v112_constructions_covered_by_established_evaluator": list(ABSTENTION_MARKERS),
        "raw_corpus": corpus_manifest,
        "probes": probes,
    }


def build_from_arrow(
    *,
    train_path: Path,
    validation_path: Path,
    dataset_info_path: Path,
    prior_catalog_path: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    prior = load_probe_catalog(prior_catalog_path)
    excluded = {
        str(row["raw_context_sha256"])
        for row in prior["probes"]
        if isinstance(row.get("raw_context_sha256"), str)
    }
    stories = _minhash_sample(
        _fresh_texts(_arrow_texts(train_path), excluded),
        count=count,
        seed=seed,
        exclusion_markers=DEFAULT_CORE_EXCLUSION_MARKERS,
    )
    catalog = build_catalog(
        stories,
        corpus_manifest=_corpus_manifest(
            train_path=train_path,
            validation_path=validation_path,
            dataset_info_path=dataset_info_path,
        ),
        seed=seed,
    )
    if set(catalog["generation"]["context_sha256"]) & excluded:
        raise BroadAbstentionError("prior context crossed fresh supplement")
    prior_prompts = {row["prompt"] for row in prior["probes"]}
    if prior_prompts & {row["prompt"] for row in catalog["probes"]}:
        raise BroadAbstentionError("prior prompt crossed fresh supplement")
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-arrow", required=True)
    parser.add_argument("--validation-arrow", required=True)
    parser.add_argument("--dataset-info", required=True)
    parser.add_argument("--prior-catalog", required=True)
    parser.add_argument("--count", type=int, default=700)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    catalog = build_from_arrow(
        train_path=Path(args.train_arrow).resolve(),
        validation_path=Path(args.validation_arrow).resolve(),
        dataset_info_path=Path(args.dataset_info).resolve(),
        prior_catalog_path=Path(args.prior_catalog).resolve(),
        count=args.count,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(output)
    print(json.dumps({"output": str(output), "records": len(catalog["probes"]), "sha256": _sha256(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
