"""Build and verify one English record and one specialist-domain record."""

from __future__ import annotations

import json

from abi.capability_segregation import (
    LINGUISTIC_FORM,
    SPECIALIST_KNOWLEDGE,
    build_core_domain_segregation_manifest,
    build_domain_ontology,
    build_segregated_extraction_record,
    validate_core_domain_segregation_manifest,
)


def main() -> None:
    ontology = build_domain_ontology(
        [
            {
                "domain_id": "chemistry",
                "description": "Chemical elements and reactions.",
                "capabilities": ["periodic_table"],
                "core_exclusion_markers": ["atomic number", "periodic table"],
                "label_evidence_sha256": "a" * 64,
            }
        ],
        ontology_id="example-domains-v1",
    )
    common = {
        "split": "validation",
        "source_model": "organization/teacher",
        "source_model_revision": "1" * 40,
        "teacher_token_counter": "authoritative_generated_token_ids",
    }
    english = build_segregated_extraction_record(
        **common,
        destination_scope="english_core",
        capability="rewriting",
        domain="domain_independent",
        provenance="example:english-1",
        prompt="Rewrite politely: Send the file.",
        output="Could you please send the file?",
        teacher_tokens=7,
        knowledge_class=LINGUISTIC_FORM,
        content_basis="domain_free_instruction",
        domain_labels=[],
        domain_claims=[],
        label_method="human_review",
        label_evidence_sha256="b" * 64,
        output_introduces_unsupplied_facts=False,
    )
    chemistry = build_segregated_extraction_record(
        **common,
        destination_scope="domain_cake",
        capability="periodic_table",
        domain="chemistry",
        provenance="example:chemistry-1",
        prompt="Name the element with atomic number 1.",
        output="Atomic number 1 is hydrogen.",
        teacher_tokens=8,
        knowledge_class=SPECIALIST_KNOWLEDGE,
        content_basis="specialist_fact",
        domain_labels=["chemistry"],
        domain_claims=["periodic_table:atomic_number_1=hydrogen"],
        label_method="human_review",
        label_evidence_sha256="c" * 64,
        output_introduces_unsupplied_facts=True,
    )
    records = [english, chemistry]
    manifest = build_core_domain_segregation_manifest(
        records,
        domain_ontology=ontology,
    )
    validate_core_domain_segregation_manifest(
        manifest,
        records,
        domain_ontology=ontology,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
