"""Materialize the single post-R59 prospective English catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
)
from abi.natural_english_catalog import BUILDERS, CONTENT_BASIS


ROWS_PER_CAPABILITY = 100
INDEX_OFFSET = 20_003
WRAPPERS = (
    lambda body: f"Please solve this fresh request and follow every constraint: {body}",
    lambda body: f"For this turn, carry out the following instruction exactly: {body}",
    lambda body: f"Read the request below, then respond accordingly: {body}",
    lambda body: f"Complete this user request carefully and directly: {body}",
)


def build_catalog() -> dict:
    probes = []
    for capability_index, (capability, builder) in enumerate(BUILDERS.items()):
        for local_index in range(ROWS_PER_CAPABILITY):
            family = local_index % len(WRAPPERS)
            content_index = INDEX_OFFSET + capability_index * 257 + local_index
            body, evaluator, maximum = builder(content_index, family)
            probe = {
                "probe_id": f"r60-{capability}-prospective-{local_index:03d}",
                "destination_scope": "english_core",
                "capability": capability,
                "domain": "domain_independent",
                "split": "final_test",
                "prompt": WRAPPERS[family](body),
                "max_new_tokens": maximum,
                "temperature": 0,
                "seed": 60_000_000 + capability_index * ROWS_PER_CAPABILITY + local_index,
                "evaluator": evaluator,
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": CONTENT_BASIS.get(
                    capability, "supplied_non_domain_context"
                ),
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-r60-prospective-endpoint-v1",
        "status": "PREREGISTERED_POST_R59_PROSPECTIVE_ONLY",
        "claim_boundary": (
            "A post-endpoint-freeze bounded functional English surface; not an "
            "exhaustive definition of English fluency or human preference."
        ),
        "generation": {
            "generator": "experiments.prospective_endpoint_r60.build_catalog_v1",
            "endpoint_seal": "8526fbd",
            "capabilities": list(BUILDERS),
            "probes_per_capability": ROWS_PER_CAPABILITY,
            "prompt_wrappers": len(WRAPPERS),
            "content_index_offset": INDEX_OFFSET,
            "specialist_or_closed_book_prompts": 0,
            "candidate_outputs_seen_before_materialization": 0,
            "source_outputs_seen_before_materialization": 0,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

