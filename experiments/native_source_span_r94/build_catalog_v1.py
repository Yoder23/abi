"""Build the fresh R94 native-source-likelihood catalog."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256
from experiments.source_qualified_span_r93.build_catalog_v1 import _premise


ROWS_PER_FAMILY = 350
TRIPLES = (
    ("CUM", "DEK", "FAS"), ("GIV", "HUR", "JOP"), ("KAB", "LIX", "MOR"),
    ("NUV", "PEM", "QAD"), ("RIF", "SOK", "TUG"), ("UZE", "VAN", "WIB"),
    ("XOR", "YEK", "ZAP"),
)
PERMUTATIONS = tuple(itertools.permutations(range(3)))
WRAPPERS = (
    lambda body, n: f"Synthetic relation query {n}. {body}",
    lambda body, n: f"Reason within this invented label system only. {body}",
    lambda body, n: f"Apply the complete fictional hierarchy. {body}",
    lambda body, n: f"Determine the terminal category in this closed record. {body}",
    lambda body, n: f"Use both declared classification links. {body}",
)


def build() -> dict:
    probes = []
    role_counts, display_counts, placement_counts = [0] * 3, [0] * 3, [0] * 4
    for family in range(4):
        for local in range(ROWS_PER_FAMILY):
            index = family * ROWS_PER_FAMILY + local
            numeric = 6_000_000 + index
            triple = TRIPLES[(index * 2 + family) % len(TRIPLES)]
            rotation = (index + 2 * family) % 3
            roles = triple[rotation:] + triple[:rotation]
            a, b, c = [f"{stem}{numeric:07d}" for stem in roles]
            subject = f"PIVOT{numeric:07d}"
            role_counts[rotation] += 1
            permutation = PERMUTATIONS[(index * 11 + family) % len(PERMUTATIONS)]
            semantic = [a, b, c]
            displayed = [semantic[item] for item in permutation]
            display_counts[displayed.index(c)] += 1
            placement = index % 4
            placement_counts[placement] += 1
            relation = _premise(family, a, b, c, subject)
            choices = f"Possible class labels, unsorted: {' | '.join(displayed)}."
            question = f"Give only the class reached by {subject} after traversing both links."
            body = (
                f"{relation} {question}" if placement == 0 else
                f"{choices} {relation} {question}" if placement == 1 else
                f"{relation} {choices} {question}" if placement == 2 else
                f"{relation} {question} {choices}"
            )
            prompt = WRAPPERS[(index * 3 + family) % len(WRAPPERS)](body, index + 1)
            row = {
                "probe_id": f"r94-reasoning-f{family}-{local:04d}",
                "destination_scope": "english_core", "capability": "domain_independent_reasoning",
                "domain": "domain_independent", "split": "validation",
                "prompt": prompt, "max_new_tokens": 16, "temperature": 0,
                "seed": 94_000_000 + index,
                "evaluator": {"kind": "exact", "value": c, "case_sensitive": True},
                "conditional_candidates": displayed, "semantic_role_codes": semantic,
                "destination_display_index": displayed.index(c), "structural_family": family,
                "candidate_list_placement": placement,
                "record_schema": SEGREGATED_RECORD_SCHEMA, "knowledge_class": LINGUISTIC_FORM,
                "content_basis": "abstract_or_nonce_content", "domain_labels": [], "domain_claims": [],
                "label_method": "preregistered_catalog", "output_introduces_unsupplied_facts": False,
            }
            row["label_evidence_sha256"] = probe_label_evidence_sha256(row)
            probes.append(row)
    return {
        "schema_version": PROBE_CATALOG_SCHEMA, "catalog_id": "abi-r94-native-source-span-v1",
        "status": "PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY",
        "claim_boundary": "Bounded native-source-qualified two-hop reasoning only.",
        "generation": {
            "generator": "experiments.native_source_span_r94.build_catalog_v1",
            "rows": len(probes), "premise_families": 4, "rows_per_family": ROWS_PER_FAMILY,
            "outer_interfaces": len(WRAPPERS), "candidate_list_placements": 4,
            "lexical_triples": [list(row) for row in TRIPLES], "subject_prefix": "PIVOT",
            "numeric_offset": 6_000_000, "role_rotation_counts": role_counts,
            "destination_display_position_counts": display_counts,
            "candidate_list_placement_counts": placement_counts,
            "real_world_facts": 0, "candidate_outputs_observed": 0, "teacher_outputs_observed": 0,
        }, "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R94 catalog exists: {args.output}")
    value = build()
    if len(value["probes"]) != 1_400 or len({row["prompt"] for row in value["probes"]}) != 1_400:
        parser.error("R94 catalog coverage changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
