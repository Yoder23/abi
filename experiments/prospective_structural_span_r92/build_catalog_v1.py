"""Build the unseen R92 structural reasoning catalog."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256


ROWS_PER_FAMILY = 200
TRIPLES = (
    ("CEV", "DAX", "FOP"), ("GUB", "HIZ", "JEK"), ("KOR", "LAV", "MIP"),
    ("NUQ", "PEX", "RAB"), ("SIV", "TOG", "UWL"), ("VOX", "WEM", "XAD"),
    ("YIF", "ZOL", "BEQ"),
)
PERMUTATIONS = tuple(itertools.permutations(range(3)))
WRAPPERS = (
    lambda body, number: f"New reasoning case {number}: {body}",
    lambda body, number: f"Work only from this fictional record. {body}",
    lambda body, number: f"Resolve the invented classification below without outside facts. {body}",
    lambda body, number: f"For this self-contained item, follow both links. {body}",
    lambda body, number: f"Analyze the supplied category chain and answer concisely. {body}",
)


def _premise(family: int, a: str, b: str, c: str, subject: str) -> str:
    forms = (
        f"Provided {a} counts as {b}, and {b} counts as {c}. {subject} is categorized as {a}.",
        f"{subject} bears {a}; bearing {a} entails {b}; bearing {b} entails {c}.",
        f"The starting category of {subject} is {a}. Reclassify each {a} as {b}, then each {b} as {c}.",
        f"In a nested hierarchy, {subject} lies in {a}, {a} is a subset of {b}, and {b} is a subset of {c}.",
        f"Objects marked {a} are relabeled {b}; objects marked {b} are relabeled {c}; {subject} is marked {a}.",
        f"Membership ledger: {subject} has {a}. The {a} ledger points to {b}. The {b} ledger points to {c}.",
        f"Trace this classification: {subject} begins under {a}; the next broader type is {b}; the type broader than {b} is {c}.",
    )
    return forms[family]


def build() -> dict:
    probes = []
    role_counts = [0, 0, 0]
    display_counts = [0, 0, 0]
    placement_counts = [0, 0, 0, 0]
    for family in range(7):
        for local in range(ROWS_PER_FAMILY):
            global_index = family * ROWS_PER_FAMILY + local
            numeric = 2_000_000 + global_index
            triple = TRIPLES[(global_index * 3 + family) % len(TRIPLES)]
            rotation = (global_index + family * 2) % 3
            roles = triple[rotation:] + triple[:rotation]
            a, b, c = [f"{stem}{numeric:07d}" for stem in roles]
            subject = f"TARGET{numeric:07d}"
            role_counts[rotation] += 1
            permutation = PERMUTATIONS[(global_index * 5 + family) % len(PERMUTATIONS)]
            semantic = [a, b, c]
            displayed = [semantic[index] for index in permutation]
            display_counts[displayed.index(c)] += 1
            placement = global_index % 4
            placement_counts[placement] += 1
            relation = _premise(family, a, b, c, subject)
            candidates = f"Candidate labels: {' | '.join(displayed)}."
            question = f"After both relations, return only the final class for {subject}."
            if placement == 0:
                body = f"{relation} {question}"
            elif placement == 1:
                body = f"{candidates} {relation} {question}"
            elif placement == 2:
                body = f"{relation} {candidates} {question}"
            else:
                body = f"{relation} {question} {candidates}"
            prompt = WRAPPERS[(global_index + family) % len(WRAPPERS)](body, global_index + 1)
            probe = {
                "probe_id": f"r92-reasoning-f{family}-{local:04d}",
                "destination_scope": "english_core",
                "capability": "domain_independent_reasoning",
                "domain": "domain_independent", "split": "validation",
                "prompt": prompt, "max_new_tokens": 16, "temperature": 0,
                "seed": 92_000_000 + global_index,
                "evaluator": {"kind": "exact", "value": c, "case_sensitive": True},
                "conditional_candidates": displayed,
                "semantic_role_codes": semantic,
                "destination_display_index": displayed.index(c),
                "structural_family": family,
                "candidate_list_placement": placement,
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
        "catalog_id": "abi-r92-prospective-structural-span-v1",
        "status": "PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY",
        "claim_boundary": "Bounded unseen two-hop reasoning surface only; not broad English or the ABI moonshot.",
        "generation": {
            "generator": "experiments.prospective_structural_span_r92.build_catalog_v1",
            "rows": len(probes), "premise_families": 7,
            "rows_per_family": ROWS_PER_FAMILY,
            "outer_interfaces": len(WRAPPERS), "candidate_list_placements": 4,
            "lexical_triples": [list(row) for row in TRIPLES],
            "subject_prefix": "TARGET", "numeric_offset": 2_000_000,
            "semantic_destination_stem_rotation_counts": role_counts,
            "destination_display_position_counts": display_counts,
            "candidate_list_placement_counts": placement_counts,
            "real_world_facts": 0, "candidate_outputs_observed": 0,
            "teacher_outputs_observed": 0,
        },
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R92 catalog exists: {args.output}")
    value = build()
    probes = value["probes"]
    if len(probes) != 1_400 or len({row["prompt"] for row in probes}) != 1_400:
        parser.error("R92 catalog coverage changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
