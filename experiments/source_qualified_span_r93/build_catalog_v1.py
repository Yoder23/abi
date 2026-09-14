"""Build fresh R93 rows from the source-qualified R92 syntax families."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256


ROWS_PER_FAMILY = 350
TRIPLES = (
    ("CUZ", "DIB", "FEN"), ("GAK", "HOV", "JUR"), ("KEM", "LUD", "MOX"),
    ("NAB", "PIR", "QUS"), ("ROV", "SEK", "TAD"), ("UFI", "VOG", "WAP"),
    ("XEN", "YUB", "ZIC"),
)
PERMUTATIONS = tuple(itertools.permutations(range(3)))
WRAPPERS = (
    lambda body, number: f"Fictional taxonomy exercise {number}. {body}",
    lambda body, number: f"Use the stated hierarchy and nothing else. {body}",
    lambda body, number: f"Complete both category steps in this made-up system. {body}",
    lambda body, number: f"Answer this closed-world label question. {body}",
    lambda body, number: f"Follow the two supplied inclusions before responding. {body}",
)


def _premise(family: int, a: str, b: str, c: str, subject: str) -> str:
    # These are the four R92 forms selected solely by frozen source evidence.
    forms = (
        f"{subject} bears {a}; bearing {a} entails {b}; bearing {b} entails {c}.",
        f"In a nested hierarchy, {subject} lies in {a}, {a} is a subset of {b}, and {b} is a subset of {c}.",
        f"Membership ledger: {subject} has {a}. The {a} ledger points to {b}. The {b} ledger points to {c}.",
        f"Trace this classification: {subject} begins under {a}; the next broader type is {b}; the type broader than {b} is {c}.",
    )
    return forms[family]


def build() -> dict:
    probes = []
    role_counts = [0, 0, 0]
    display_counts = [0, 0, 0]
    placement_counts = [0, 0, 0, 0]
    for family in range(4):
        for local in range(ROWS_PER_FAMILY):
            global_index = family * ROWS_PER_FAMILY + local
            numeric = 4_000_000 + global_index
            triple = TRIPLES[(global_index * 5 + family) % len(TRIPLES)]
            rotation = (global_index + family) % 3
            roles = triple[rotation:] + triple[:rotation]
            a, b, c = [f"{stem}{numeric:07d}" for stem in roles]
            subject = f"FOCUS{numeric:07d}"
            role_counts[rotation] += 1
            permutation = PERMUTATIONS[(global_index * 7 + family) % len(PERMUTATIONS)]
            semantic = [a, b, c]
            displayed = [semantic[index] for index in permutation]
            display_counts[displayed.index(c)] += 1
            placement = global_index % 4
            placement_counts[placement] += 1
            relation = _premise(family, a, b, c, subject)
            choices = f"Available labels in arbitrary order: {' | '.join(displayed)}."
            question = f"Return only the label reached for {subject} after both links."
            if placement == 0:
                body = f"{relation} {question}"
            elif placement == 1:
                body = f"{choices} {relation} {question}"
            elif placement == 2:
                body = f"{relation} {choices} {question}"
            else:
                body = f"{relation} {question} {choices}"
            prompt = WRAPPERS[(global_index * 2 + family) % len(WRAPPERS)](body, global_index + 1)
            probe = {
                "probe_id": f"r93-reasoning-f{family}-{local:04d}",
                "destination_scope": "english_core", "capability": "domain_independent_reasoning",
                "domain": "domain_independent", "split": "validation",
                "prompt": prompt, "max_new_tokens": 16, "temperature": 0,
                "seed": 93_000_000 + global_index,
                "evaluator": {"kind": "exact", "value": c, "case_sensitive": True},
                "conditional_candidates": displayed, "semantic_role_codes": semantic,
                "destination_display_index": displayed.index(c),
                "structural_family": family, "r92_source_selected_family": (1, 3, 5, 6)[family],
                "candidate_list_placement": placement,
                "record_schema": SEGREGATED_RECORD_SCHEMA, "knowledge_class": LINGUISTIC_FORM,
                "content_basis": "abstract_or_nonce_content", "domain_labels": [], "domain_claims": [],
                "label_method": "preregistered_catalog", "output_introduces_unsupplied_facts": False,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-r93-source-qualified-structural-span-v1",
        "status": "PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY",
        "claim_boundary": "Bounded teacher-qualified two-hop reasoning interface only.",
        "generation": {
            "generator": "experiments.source_qualified_span_r93.build_catalog_v1",
            "rows": len(probes), "premise_families": 4, "rows_per_family": ROWS_PER_FAMILY,
            "r92_selected_families": [1, 3, 5, 6], "r92_selection_threshold_per_200": 189,
            "outer_interfaces": len(WRAPPERS), "candidate_list_placements": 4,
            "lexical_triples": [list(row) for row in TRIPLES], "subject_prefix": "FOCUS",
            "numeric_offset": 4_000_000, "semantic_destination_stem_rotation_counts": role_counts,
            "destination_display_position_counts": display_counts,
            "candidate_list_placement_counts": placement_counts,
            "real_world_facts": 0, "candidate_outputs_observed": 0, "teacher_outputs_observed": 0,
        },
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R93 catalog exists: {args.output}")
    value = build()
    if len(value["probes"]) != 1_400 or len({row["prompt"] for row in value["probes"]}) != 1_400:
        parser.error("R93 catalog coverage changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
