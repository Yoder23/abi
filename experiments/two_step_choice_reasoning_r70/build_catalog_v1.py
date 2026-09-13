"""Build the preregistered R70 explicit two-step reasoning catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256


ROWS = 2_000
PREMISES = (
    "Rule 1: Every {a} is a {b}. Rule 2: Every {b} is a {c}. Start: {s} is a {a}.",
    "First, all {a} become {b}. Second, all {b} become {c}. Initially {s} is {a}.",
    "If a thing is {a}, rule 1 maps it to {b}; rule 2 maps {b} to {c}. {s} begins as {a}.",
    "Class {a} enters {b} under the first rule, then {b} enters {c} under the second. {s} starts in {a}.",
    "The first invented relation is {a} -> {b}. The second is {b} -> {c}. The initial label for {s} is {a}.",
    "Step-one membership: {a} implies {b}. Step-two membership: {b} implies {c}. Given {s} has {a} membership.",
    "Fictional transition one sends {a} to {b}; transition two sends {b} to {c}; {s} begins at {a}.",
    "Apply the ordered rules {a} to {b}, followed by {b} to {c}. The starting class of {s} is {a}.",
)
INSTRUCTIONS = (
    "Apply both rules exactly once in order. Which class is reached after step two? Choose only one code: {a}, {b}, or {c}.",
    "Starting at {a}, follow step one and then step two. Return only the final code from [{a}, {b}, {c}].",
    "Do not stop after rule one. After both supplied transitions, output exactly the destination class code among {a}/{b}/{c}.",
    "Use no outside facts. Execute the two stated mappings in sequence and name only the code reached after the second mapping: {a}, {b}, or {c}.",
)


def build_catalog() -> dict:
    probes = []
    for index in range(ROWS):
        values = {letter.lower(): f"{letter}{index:04d}" for letter in "ABCS"}
        prompt = (
            PREMISES[index % len(PREMISES)].format(**values)
            + " "
            + INSTRUCTIONS[(index // len(PREMISES)) % len(INSTRUCTIONS)].format(**values)
        )
        probe = {
            "probe_id": f"r70-reasoning-search-{index:04d}",
            "destination_scope": "english_core",
            "capability": "domain_independent_reasoning",
            "domain": "domain_independent", "split": "search",
            "prompt": prompt, "max_new_tokens": 16, "temperature": 0,
            "seed": 70_000_000 + index,
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
        "catalog_id": "abi-r70-two-step-choice-reasoning-search-v1",
        "status": "PREREGISTERED_SEARCH_TRAINING_ONLY",
        "claim_boundary": "Two-step nonce transitivity acquisition; not broad reasoning or host certification.",
        "generation": {
            "generator": "experiments.two_step_choice_reasoning_r70.build_catalog_v1",
            "rows": ROWS, "premise_templates": len(PREMISES),
            "instruction_templates": len(INSTRUCTIONS), "max_new_tokens": 16,
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
        parser.error("R70 prompts are not exactly 2,000 unique rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
