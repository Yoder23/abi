"""Build R97 after the R96 candidate was frozen."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256


ROWS_PER_FAMILY = 350
TRIPLES = (
    ("ALFA", "BENO", "CIRA"), ("DOVI", "ELMA", "FARO"), ("GILI", "HAPO", "IRVA"),
    ("JESO", "KUNI", "LARO"), ("MIVA", "NOLI", "OPRA"), ("PESO", "QUVA", "RILO"),
    ("SANO", "TUVI", "WERA"),
)
PERMUTATIONS = tuple(itertools.permutations(range(3)))
WRAPPERS = (
    lambda body, n: f"Invented ontology case {n}. {body}",
    lambda body, n: f"Treat every name as a nonce code. {body}",
    lambda body, n: f"Using only this closed record, resolve item {n}. {body}",
    lambda body, n: f"No real-world information applies to this hierarchy. {body}",
    lambda body, n: f"Complete the supplied classification path. {body}",
    lambda body, n: f"Read all links before returning one code. {body}",
)


def _relation(family: int, a: str, b: str, c: str, subject: str) -> str:
    return (
        f"The register lists {subject} with {a}. It expands {a} to {b}, then expands {b} to {c}.",
        f"For {subject}, the narrow type is {a}; its middle type is {b}; its broad type is {c}.",
        f"A first portal carries {subject} into {a}, a second carries {a} into {b}, and a third carries {b} into {c}.",
        f"Tagging facts say {subject} has {a}; having {a} guarantees {b}; having {b} guarantees {c}.",
    )[family]


def build() -> dict:
    probes = []; roles = [0] * 3; displays = [0] * 3; placements = [0] * 4
    for family in range(4):
        for local in range(ROWS_PER_FAMILY):
            index = family * ROWS_PER_FAMILY + local; numeric = 10_000_000 + index
            triple = TRIPLES[(index * 4 + family) % len(TRIPLES)]; rotation = (index + family * 2) % 3
            rotated = triple[rotation:] + triple[:rotation]
            a, b, c = [f"{stem}{numeric:08d}" for stem in rotated]; subject = f"ITEM{numeric:08d}"
            roles[rotation] += 1
            permutation = PERMUTATIONS[(index * 7 + family * 5) % len(PERMUTATIONS)]
            semantic = [a, b, c]; displayed = [semantic[item] for item in permutation]
            displays[displayed.index(c)] += 1; placement = (index * 3 + family) % 4; placements[placement] += 1
            relation = _relation(family, a, b, c, subject)
            choices = f"Unordered answer codes: {' | '.join(displayed)}."
            question = f"Return only the terminal code associated with {subject}."
            body = (f"{relation} {question}" if placement == 0 else f"{choices} {relation} {question}" if placement == 1 else f"{relation} {choices} {question}" if placement == 2 else f"{relation} {question} {choices}")
            prompt = WRAPPERS[(index * 5 + family) % len(WRAPPERS)](body, index + 1)
            row = {
                "probe_id": f"r97-reasoning-f{family}-{local:04d}", "destination_scope": "english_core",
                "capability": "domain_independent_reasoning", "domain": "domain_independent", "split": "validation",
                "prompt": prompt, "max_new_tokens": 16, "temperature": 0, "seed": 97_000_000 + index,
                "evaluator": {"kind": "exact", "value": c, "case_sensitive": True},
                "conditional_candidates": displayed, "semantic_role_codes": semantic,
                "destination_display_index": displayed.index(c), "structural_family": family,
                "candidate_list_placement": placement, "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM, "content_basis": "abstract_or_nonce_content",
                "domain_labels": [], "domain_claims": [], "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
            }
            row["label_evidence_sha256"] = probe_label_evidence_sha256(row); probes.append(row)
    return {"schema_version": PROBE_CATALOG_SCHEMA, "catalog_id": "abi-r97-prospective-length-span-v1", "status": "PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY", "claim_boundary": "Bounded prospective R96 two-hop reasoning only.", "generation": {"generator": "experiments.prospective_length_span_r97.build_catalog_v1", "rows": len(probes), "premise_families": 4, "rows_per_family": ROWS_PER_FAMILY, "outer_interfaces": len(WRAPPERS), "candidate_list_placements": 4, "lexical_triples": [list(row) for row in TRIPLES], "subject_prefix": "ITEM", "numeric_offset": 10_000_000, "role_rotation_counts": roles, "destination_display_position_counts": displays, "candidate_list_placement_counts": placements, "real_world_facts": 0, "candidate_outputs_observed": 0, "teacher_outputs_observed": 0}, "probes": probes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", required=True, type=Path); args = parser.parse_args()
    if args.output.exists(): parser.error("immutable R97 catalog exists")
    value = build()
    if len(value["probes"]) != 1_400 or len({row["prompt"] for row in value["probes"]}) != 1_400: parser.error("R97 catalog coverage changed")
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode()); load_probe_catalog(args.output); print(args.output); return 0


if __name__ == "__main__": raise SystemExit(main())
