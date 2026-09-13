"""Build the disjoint R75 validation matrix before R74 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256
from experiments.robust_choice_substrate_r72.build_catalog_v1 import PREMISES


ROWS_PER_FAMILY = 100
INSTRUCTION = (
    "Do not stop after rule one. After both supplied transitions, output exactly "
    "the destination class code among {a}/{b}/{c}."
)


def build_catalog() -> dict:
    probes = []
    for family, premise in enumerate(PREMISES):
        for local_index in range(ROWS_PER_FAMILY):
            numeric = 50_000 + family * ROWS_PER_FAMILY + local_index
            values = {"a": f"D{numeric:05d}", "b": f"E{numeric:05d}", "c": f"F{numeric:05d}", "s": f"T{numeric:05d}"}
            prompt = premise.format(**values) + " " + INSTRUCTION.format(**values)
            probe = {
                "probe_id": f"r75-reasoning-f{family}-{local_index:03d}",
                "destination_scope": "english_core",
                "capability": "domain_independent_reasoning",
                "domain": "domain_independent",
                "split": "validation",
                "prompt": prompt,
                "max_new_tokens": 16,
                "temperature": 0,
                "seed": 75_000_000 + family * ROWS_PER_FAMILY + local_index,
                "evaluator": {"kind": "contains_all", "values": [values["c"]]},
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
        "catalog_id": "abi-r75-disjoint-conditional-choice-reasoning-validation-v1",
        "status": "PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY",
        "claim_boundary": "Bounded reasoning transfer only; not unrestricted English or promotion.",
        "generation": {
            "generator": "experiments.conditional_choice_host_r74.build_validation_catalog_v1",
            "rows": len(probes),
            "premise_families": len(PREMISES),
            "rows_per_family": ROWS_PER_FAMILY,
            "training_code_prefixes": ["A", "B", "C"],
            "validation_code_prefixes": ["D", "E", "F"],
            "nonce_index_offset": 50_000,
            "real_world_facts": 0,
            "candidate_outputs_observed": 0,
            "teacher_outputs_observed": 0,
        },
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R75 catalog exists: {args.output}")
    value = build_catalog()
    prompts = [row["prompt"] for row in value["probes"]]
    if len(prompts) != 700 or len(set(prompts)) != 700:
        parser.error("R75 catalog is not exactly 700 unique prompts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
