"""Build the disjoint R72 robust-choice reasoning catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256


ROWS_PER_FAMILY = 300
PREMISES = (
    "Rule 1: Every {a} is a {b}. Rule 2: Every {b} is a {c}. Start: {s} is a {a}.",
    "First, all {a} become {b}. Second, all {b} become {c}. Initially {s} is {a}.",
    "If a thing is {a}, rule 1 maps it to {b}; rule 2 maps {b} to {c}. {s} begins as {a}.",
    "Class {a} enters {b} under the first rule, then {b} enters {c} under the second. {s} starts in {a}.",
    "The first invented relation is {a} -> {b}. The second is {b} -> {c}. The initial label for {s} is {a}.",
    "Step-one membership: {a} implies {b}. Step-two membership: {b} implies {c}. Given {s} has {a} membership.",
    "Fictional transition one sends {a} to {b}; transition two sends {b} to {c}; {s} begins at {a}.",
)
INSTRUCTION = (
    "Do not stop after rule one. After both supplied transitions, output exactly "
    "the destination class code among {a}/{b}/{c}."
)


def build_catalog() -> dict:
    probes = []
    for family, premise in enumerate(PREMISES):
        for local_index in range(ROWS_PER_FAMILY):
            numeric = 10_000 + family * ROWS_PER_FAMILY + local_index
            values = {letter.lower(): f"{letter}{numeric:05d}" for letter in "ABCS"}
            prompt = premise.format(**values) + " " + INSTRUCTION.format(**values)
            probe = {
                "probe_id": f"r72-reasoning-f{family}-{local_index:03d}",
                "destination_scope": "english_core",
                "capability": "domain_independent_reasoning",
                "domain": "domain_independent", "split": "search",
                "prompt": prompt, "max_new_tokens": 16, "temperature": 0,
                "seed": 72_000_000 + family * ROWS_PER_FAMILY + local_index,
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
        "catalog_id": "abi-r72-robust-choice-reasoning-search-v1",
        "status": "PREREGISTERED_DISCLOSED_DEVELOPMENT_SEARCH_ONLY",
        "claim_boundary": "R71-selected robust source interface; not independent validation or host certification.",
        "generation": {
            "generator": "experiments.robust_choice_substrate_r72.build_catalog_v1",
            "rows": len(PREMISES) * ROWS_PER_FAMILY,
            "premise_families": len(PREMISES), "rows_per_family": ROWS_PER_FAMILY,
            "instruction_family": "r71_family_2_disclosed_selection",
            "nonce_index_offset": 10_000, "real_world_facts": 0,
            "candidate_outputs_observed": 0, "teacher_outputs_observed": 0,
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
    if len(prompts) != 2_100 or len(set(prompts)) != 2_100:
        parser.error("R72 prompts are not exactly 2,100 unique rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

