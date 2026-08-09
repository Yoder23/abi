"""Build one source-disjoint Phase 3 task-family-matched search catalog.

This successor reuses the Phase 1 builders that existed before development
results were observed.  It changes only the source-index range, immutable
record namespace, and search wrappers.  It never reads teacher outputs or
final-test material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .capability_compiler_phase1_catalog import CAPABILITY_ALIASES, DEFAULT_SEED
from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import PROBE_CATALOG_SCHEMA, probe_label_evidence_sha256
from .natural_english_catalog import BUILDERS, _v2_probe, _v3_probe


CATALOG_FORMAT = "abi-capability-compiler-phase3-targeted-catalog/1"
SEARCH_PER_CAPABILITY = 700
SOURCE_OFFSET = 700_000

_WRAPPERS = (
    lambda body, ref: f"Complete this new bounded English practice task. Reference {ref}.\n{body}",
    lambda body, ref: f"Use only the material supplied in search item {ref}.\n{body}",
    lambda body, ref: f"Follow the exact requested wording and format for item {ref}.\n{body}",
    lambda body, ref: f"Respond directly to this independent English search task ({ref}).\n{body}",
    lambda body, ref: f"Work only from the text below. Search case {ref}.\n{body}",
    lambda body, ref: f"Give only the requested answer for new exercise {ref}.\n{body}",
    lambda body, ref: f"Read the supplied context, then answer search ticket {ref}.\n{body}",
    lambda body, ref: f"Handle this language task without outside facts. Search ref {ref}.\n{body}",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _probe(capability: str, capability_index: int, local_index: int) -> dict[str, Any]:
    if not 0 <= local_index < SEARCH_PER_CAPABILITY:
        raise ValueError("targeted local index outside fixed depth")
    source_index = SOURCE_OFFSET + capability_index * 10_000 + local_index
    family = local_index % 4
    body, evaluator, maximum = BUILDERS[capability](source_index, family)
    reference = f"P3T-{capability_index:02d}-{local_index:04d}"
    wrapper_index = local_index % len(_WRAPPERS)
    row: dict[str, Any] = {
        "probe_id": f"phase3-targeted-search-{capability}-{local_index:04d}-v1",
        "destination_scope": "english_core",
        "capability": capability,
        "canonical_capability": CAPABILITY_ALIASES[capability],
        "domain": "domain_independent",
        "split": "search",
        "prompt": body,
        "max_new_tokens": max(int(maximum), 192),
        "temperature": 0,
        "seed": DEFAULT_SEED + source_index,
        "evaluator": evaluator,
        "record_schema": SEGREGATED_RECORD_SCHEMA,
        "knowledge_class": LINGUISTIC_FORM,
        "content_basis": "supplied_non_domain_context",
        "domain_labels": [],
        "domain_claims": [],
        "label_method": "preregistered_catalog",
        "output_introduces_unsupplied_facts": False,
        "phase3_targeted_template_family": f"targeted:{capability}:builder-{family}:wrapper-{wrapper_index}",
        "source_index": source_index,
    }
    row = _v3_probe(row) if capability in {"abstention", "coherence"} else _v2_probe(row)
    row["prompt"] = _WRAPPERS[wrapper_index](str(row["prompt"]), reference)
    row["label_evidence_sha256"] = probe_label_evidence_sha256(row)
    return row


def build_catalog(*, excluded_prompt_hashes: set[str] | None = None) -> dict[str, Any]:
    probes = [
        _probe(capability, capability_index, local_index)
        for capability_index, capability in enumerate(BUILDERS)
        for local_index in range(SEARCH_PER_CAPABILITY)
    ]
    prompt_hashes = [_sha256_text(str(row["prompt"])) for row in probes]
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise RuntimeError("targeted catalog contains duplicate prompts")
    excluded = excluded_prompt_hashes or set()
    overlap = sorted(set(prompt_hashes) & excluded)
    if overlap:
        raise RuntimeError("targeted catalog overlaps a bound prior prompt")
    if any(row["domain_labels"] or row["domain_claims"] or row["domain"] != "domain_independent" for row in probes):
        raise RuntimeError("specialist material crossed the English catalog boundary")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "phase3_targeted_catalog_format": CATALOG_FORMAT,
        "catalog_id": "abi-capability-compiler-phase3-targeted-search-v132",
        "status": "PREREGISTERED_NO_TEACHER_CATALOG",
        "claim_boundary": "Search-only task-family-matched acquisition surface; not evidence of model quality, minimum information, or Phase 3 certification.",
        "generation": {
            "generator": "abi.capability_compiler_phase3_targeted_catalog",
            "source_offset": SOURCE_OFFSET,
            "search_per_english_capability": SEARCH_PER_CAPABILITY,
            "total_search_probes": len(probes),
            "teacher_outputs_read_during_construction": False,
            "validation_prompts_read_during_construction": False,
            "final_prompts_read_during_construction": False,
            "source_builders_frozen_before_phase3_results": True,
        },
        "capability_aliases": CAPABILITY_ALIASES,
        "probes": probes,
        "domain_isolation_probes": [],
        "adversarial_probes": [],
    }


def _prior_hashes(paths: Sequence[Path]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        values.update(_sha256_text(str(row["prompt"])) for row in document["probes"])
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-catalog", action="append", default=[])
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    excluded = _prior_hashes([Path(value).resolve() for value in args.exclude_catalog])
    catalog = build_catalog(excluded_prompt_hashes=excluded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "probes": len(catalog["probes"]), "excluded_prompt_hashes": len(excluded), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
