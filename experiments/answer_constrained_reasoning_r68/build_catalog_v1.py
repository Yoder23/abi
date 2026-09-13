"""Build the preregistered R68 answer-constrained reasoning catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256


ROWS = 2_000
TEMPLATES = (
    "Every {a} is a {b}. Every {b} is a {c}. {s} is a {a}.",
    "All {a} are {b}; all {b} are {c}; {s} is {a}.",
    "If a thing is {a}, it is {b}. If it is {b}, it is {c}. {s} is {a}.",
    "Class {a} is inside {b}, and {b} is inside {c}. {s} belongs to {a}.",
    "A {a} counts as a {b}. A {b} counts as a {c}. {s} counts as a {a}.",
    "Every {a} maps to {b}; every {b} maps to {c}; every {c} maps to {d}; {s} starts as {a}.",
    "Given {a} implies {b}, {b} implies {c}, and {c} implies {d}, {s} is {a}.",
    "The sets nest {a} inside {b}, {b} inside {c}, and {c} inside {d}; {s} is in {a}.",
)
REQUESTS = (
    "Using only these invented rules, answer with the terminal class for {s}. Be concise.",
    "Name the furthest class entailed for {s}; do not add outside facts.",
    "Follow the fictional chain and return the final supported category for {s}.",
    "Derive the transitive conclusion and state only the last class containing {s}.",
)


def _code(letter: str, index: int) -> str:
    return f"{letter}{index:04d}"


def build_catalog() -> dict:
    probes = []
    for index in range(ROWS):
        template_index = index % len(TEMPLATES)
        values = {letter.lower(): _code(letter, index) for letter in "ABCDS"}
        conclusion = values["d"] if template_index >= 5 else values["c"]
        prompt = (
            TEMPLATES[template_index].format(**values)
            + " "
            + REQUESTS[(index // len(TEMPLATES)) % len(REQUESTS)].format(**values)
        )
        probe = {
            "probe_id": f"r68-reasoning-search-{index:04d}",
            "destination_scope": "english_core",
            "capability": "domain_independent_reasoning",
            "domain": "domain_independent", "split": "search",
            "prompt": prompt, "max_new_tokens": 96, "temperature": 0,
            "seed": 68_000_000 + index,
            "evaluator": {"kind": "contains_all", "values": [conclusion]},
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
        "catalog_id": "abi-r68-answer-constrained-reasoning-search-v1",
        "status": "PREREGISTERED_SEARCH_TRAINING_ONLY",
        "claim_boundary": "Domain-free transitive-rule acquisition; not broad reasoning or host certification.",
        "generation": {
            "generator": "experiments.answer_constrained_reasoning_r68.build_catalog_v1",
            "rows": ROWS, "premise_templates": len(TEMPLATES),
            "request_templates": len(REQUESTS), "max_new_tokens": 96,
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
        parser.error("R68 prompts are not exactly 2,000 unique rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
