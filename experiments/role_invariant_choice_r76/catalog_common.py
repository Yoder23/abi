"""Deterministic nonce catalogs for the R76/R77 role-invariance test."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, probe_label_evidence_sha256
from experiments.robust_choice_substrate_r72.build_catalog_v1 import PREMISES


TRAINING_TRIPLES = (
    ("KAV", "LEM", "NUR"),
    ("PAX", "RIV", "SOM"),
    ("TAL", "VEX", "WUN"),
    ("YOR", "ZEM", "BIC"),
    ("GAQ", "HUP", "JEN"),
    ("LOV", "MAZ", "NIP"),
    ("QET", "RUK", "SAV"),
    ("TIX", "VOM", "WAZ"),
    ("YEP", "ZUN", "BAF"),
    ("GIR", "HOK", "JUV"),
    ("KEX", "LUP", "MOR"),
    ("NAP", "QIV", "ROZ"),
)
VALIDATION_TRIPLES = (
    ("DOR", "EVI", "FUM"),
    ("CEX", "IBO", "OJU"),
    ("APR", "UDO", "EKS"),
    ("FIZ", "CAV", "JOT"),
    ("DUM", "EPR", "FOQ"),
    ("CUN", "IXA", "OBE"),
    ("AJO", "UVI", "EZO"),
)
PERMUTATIONS = tuple(itertools.permutations(range(3)))


def build_catalog(
    *,
    campaign: str,
    split: str,
    rows_per_family: int,
    numeric_offset: int,
    triples: Sequence[tuple[str, str, str]],
    subject_prefix: str,
    seed_offset: int,
    status: str,
    generator: str,
) -> dict:
    """Build a catalog whose destination surface and display position vary."""

    probes = []
    role_counts = [0, 0, 0]
    display_counts = [0, 0, 0]
    for family, premise in enumerate(PREMISES):
        for local_index in range(rows_per_family):
            global_index = family * rows_per_family + local_index
            numeric = numeric_offset + global_index
            triple = triples[(global_index * 5 + family) % len(triples)]

            # Rotate which lexical stem occupies each semantic role. Therefore
            # the two-hop destination is not learnable as one fixed prefix.
            role_rotation = (global_index + 2 * family) % 3
            role_stems = triple[role_rotation:] + triple[:role_rotation]
            values = {
                "a": f"{role_stems[0]}{numeric:06d}",
                "b": f"{role_stems[1]}{numeric:06d}",
                "c": f"{role_stems[2]}{numeric:06d}",
                "s": f"{subject_prefix}{numeric:06d}",
            }
            role_counts[role_rotation] += 1

            permutation = PERMUTATIONS[(global_index * 7 + family) % len(PERMUTATIONS)]
            semantic_codes = [values["a"], values["b"], values["c"]]
            displayed = [semantic_codes[index] for index in permutation]
            display_counts[displayed.index(values["c"])] += 1
            instruction = (
                "Apply exactly the two supplied transitions to the start item. "
                f"Candidate codes in random order: {' | '.join(displayed)}. "
                "Return only the final destination code, with no explanation."
            )
            prompt = premise.format(**values) + " " + instruction
            probe = {
                "probe_id": f"{campaign}-reasoning-f{family}-{local_index:04d}",
                "destination_scope": "english_core",
                "capability": "domain_independent_reasoning",
                "domain": "domain_independent",
                "split": split,
                "prompt": prompt,
                "max_new_tokens": 16,
                "temperature": 0,
                "seed": seed_offset + global_index,
                "evaluator": {
                    "kind": "exact",
                    "value": values["c"],
                    "case_sensitive": True,
                },
                "conditional_candidates": displayed,
                "semantic_role_codes": semantic_codes,
                "destination_display_index": displayed.index(values["c"]),
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": "abstract_or_nonce_content",
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)

    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": f"abi-{campaign}-role-invariant-conditional-choice-{split}-v1",
        "status": status,
        "claim_boundary": (
            "Bounded nonce two-hop relation/copy acquisition only; not unrestricted "
            "English, domain transfer, minimality, or ABI moonshot certification."
        ),
        "generation": {
            "generator": generator,
            "rows": len(probes),
            "premise_families": len(PREMISES),
            "rows_per_family": rows_per_family,
            "lexical_triples": [list(row) for row in triples],
            "subject_prefix": subject_prefix,
            "numeric_offset": numeric_offset,
            "semantic_destination_stem_rotation_counts": role_counts,
            "destination_display_position_counts": display_counts,
            "real_world_facts": 0,
            "candidate_outputs_observed": 0,
            "teacher_outputs_observed": 0,
        },
        "probes": probes,
    }

