"""Build the preregistered R67 domain-free reasoning search catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256


ROWS = 2_000
TEMPLATES = (
    "Every {a} is a {b}. Every {b} is a {c}. {subject} is a {a}.",
    "All members of {a} belong to {b}; anything in {b} belongs to {c}; {subject} belongs to {a}.",
    "If something is {a}, then it is {b}. If it is {b}, then it is {c}. {subject} is {a}.",
    "The invented class {a} sits inside {b}, and {b} sits inside {c}. Place {subject} in {a}.",
    "A {a} always counts as a {b}. A {b} always counts as a {c}. The fictional item {subject} is a {a}.",
    "Every {a} maps to {b}. Every {b} maps to {c}. Every {c} maps to {d}. {subject} starts in {a}.",
    "Given {a} implies {b}, {b} implies {c}, and {c} implies {d}, suppose {subject} is {a}.",
    "The made-up sets nest as {a} within {b}, {b} within {c}, and {c} within {d}; {subject} is in {a}.",
)
REQUESTS = (
    "Reason only from these invented statements. State the furthest supported class for {subject}.",
    "Use no outside facts. Give the final category entailed for {subject} in one concise sentence.",
    "Follow the fictional chain and identify the last class that must contain {subject}.",
    "Derive the transitive conclusion from the supplied rules and name the terminal class for {subject}.",
)
SYLLABLES = (
    "bav", "cer", "dun", "fal", "gix", "hel", "jor", "kav", "lum", "mep",
    "nor", "pex", "qir", "rov", "sul", "tir", "vex", "wun", "yap", "zod",
)


def _symbol(index: int, offset: int) -> str:
    first = SYLLABLES[(index * 7 + offset * 3) % len(SYLLABLES)]
    second = SYLLABLES[(index * 11 + offset * 5 + 1) % len(SYLLABLES)]
    return f"{first}{second}-{index:04d}-{offset}"


def build_catalog() -> dict:
    probes = []
    for index in range(ROWS):
        template_index = index % len(TEMPLATES)
        depth_four = template_index >= 5
        values = {
            "a": _symbol(index, 0), "b": _symbol(index, 1),
            "c": _symbol(index, 2), "d": _symbol(index, 3),
            "subject": f"item-{_symbol(index, 4)}",
        }
        premises = TEMPLATES[template_index].format(**values)
        request = REQUESTS[(index // len(TEMPLATES)) % len(REQUESTS)].format(**values)
        conclusion = values["d"] if depth_four else values["c"]
        prompt = f"{premises} {request}"
        probe = {
            "probe_id": f"r67-reasoning-search-{index:04d}",
            "destination_scope": "english_core",
            "capability": "domain_independent_reasoning",
            "domain": "domain_independent",
            "split": "search",
            "prompt": prompt,
            "max_new_tokens": 64,
            "temperature": 0,
            "seed": 67_000_000 + index,
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
        "catalog_id": "abi-r67-domain-free-reasoning-search-v1",
        "status": "PREREGISTERED_SEARCH_TRAINING_ONLY",
        "claim_boundary": "Domain-free transitive-rule acquisition coverage; not broad reasoning or host certification.",
        "generation": {
            "generator": "experiments.reasoning_coverage_acquisition_r67.build_catalog_v1",
            "rows": ROWS, "premise_templates": len(TEMPLATES),
            "request_templates": len(REQUESTS), "real_world_facts": 0,
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
    if len(prompts) != ROWS or len(set(prompts)) != ROWS:
        parser.error("R67 prompts are not exactly 2,000 unique rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

