"""Build the preregistered R69 answer-stem reasoning catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256


ROWS = 2_000
PREMISES = (
    "Every {a} is a {b}. Every {b} is a {c}. {s} is a {a}.",
    "All {a} are {b}; all {b} are {c}; {s} belongs to {a}.",
    "If something is {a}, it is {b}. If it is {b}, it is {c}. {s} is {a}.",
    "Class {a} is inside {b}, and class {b} is inside {c}. {s} is in {a}.",
    "A {a} always counts as a {b}. A {b} always counts as a {c}. {s} counts as a {a}.",
    "Membership in {a} implies {b}, and membership in {b} implies {c}. {s} has membership in {a}.",
    "The fictional sets nest {a} within {b} and {b} within {c}; {s} starts in {a}.",
    "Rule one maps {a} to {b}. Rule two maps {b} to {c}. The invented item {s} maps to {a}.",
)
STEMS = (
    "Use only those rules. Complete with the category code only: The terminal category for {s} is",
    "Infer transitively. Add only the missing code: Therefore {s} must belong to",
    "No explanation or outside facts. Finish this answer: The last supported class is",
    "Follow the chain. Supply exactly the final class code after the colon: Answer:",
)


def build_catalog() -> dict:
    probes = []
    for index in range(ROWS):
        values = {letter.lower(): f"{letter}{index:04d}" for letter in "ABCS"}
        prompt = (
            PREMISES[index % len(PREMISES)].format(**values)
            + " "
            + STEMS[(index // len(PREMISES)) % len(STEMS)].format(**values)
        )
        probe = {
            "probe_id": f"r69-reasoning-search-{index:04d}",
            "destination_scope": "english_core",
            "capability": "domain_independent_reasoning",
            "domain": "domain_independent", "split": "search",
            "prompt": prompt, "max_new_tokens": 24, "temperature": 0,
            "seed": 69_000_000 + index,
            "evaluator": {"kind": "contains_all", "values": [values["c"]]},
            "record_schema": SEGREGATED_RECORD_SCHEMA,
            "knowledge_class": LINGUISTIC_FORM,
            "content_basis": "abstract_or_nonce_content",
            "domain_labels": [], "domain_claims": [],
            "label_method": "preregistered_catalog",
            "output_introduces_unsupplied_facts": False,
        }
        probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
        probes.append(probe)
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-r69-answer-stem-reasoning-search-v1",
        "status": "PREREGISTERED_SEARCH_TRAINING_ONLY",
        "claim_boundary": "Three-hop nonce transitivity acquisition; not broad reasoning or host certification.",
        "generation": {
            "generator": "experiments.answer_stem_reasoning_r69.build_catalog_v1",
            "rows": ROWS, "premise_templates": len(PREMISES),
            "answer_stems": len(STEMS), "max_new_tokens": 24,
            "real_world_facts": 0, "candidate_outputs_observed": 0,
            "teacher_outputs_observed": 0,
        },
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"catalog is immutable: {args.output}")
    value = build_catalog()
    prompts = [row["prompt"] for row in value["probes"]]
    if len(prompts) != ROWS or len(set(prompts)) != ROWS:
        parser.error("R69 prompts are not exactly 2,000 unique rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
