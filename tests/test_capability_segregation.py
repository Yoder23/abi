from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from abi.capability_segregation import (
    LINGUISTIC_FORM,
    QUARANTINED,
    SPECIALIST_KNOWLEDGE,
    CapabilitySegregationError,
    build_core_domain_segregation_manifest,
    build_domain_ontology,
    build_segregated_extraction_record,
    validate_core_domain_segregation_manifest,
    validate_domain_ontology,
    validate_segregated_extraction_record,
)
from abi.layercake_acquisition import validate_labeled_extraction_record

ROOT = Path(__file__).resolve().parents[1]


def _ontology():
    return build_domain_ontology(
        [
            {
                "domain_id": "chemistry",
                "description": "Chemical elements and reactions.",
                "capabilities": ["periodic_table"],
                "core_exclusion_markers": [
                    "atomic number",
                    "hydrogen",
                    "periodic table",
                ],
                "label_evidence_sha256": "a" * 64,
            },
            {
                "domain_id": "python",
                "description": "Python language code and APIs.",
                "capabilities": ["python_generation"],
                "core_exclusion_markers": [
                    "def ",
                    "python",
                    "return a + b",
                ],
                "label_evidence_sha256": "b" * 64,
            },
        ],
        ontology_id="unit-test-domains-v1",
    )


def _english(*, prompt="Rewrite this politely: Send the file."):
    return build_segregated_extraction_record(
        destination_scope="english_core",
        capability="rewriting",
        domain="domain_independent",
        provenance="pure-english-catalog:rewrite-1",
        split="search",
        source_model="teacher/model",
        source_model_revision="1" * 40,
        prompt=prompt,
        output="Could you please send the file?",
        teacher_tokens=8,
        teacher_token_counter="authoritative_generated_token_ids",
        knowledge_class=LINGUISTIC_FORM,
        content_basis="domain_free_instruction",
        domain_labels=[],
        domain_claims=[],
        label_method="human_review",
        label_evidence_sha256="c" * 64,
        output_introduces_unsupplied_facts=False,
    )


def _chemistry(*, output="Atomic number 1 is hydrogen."):
    return build_segregated_extraction_record(
        destination_scope="domain_cake",
        capability="periodic_table",
        domain="chemistry",
        provenance="chemistry-catalog:element-1",
        split="search",
        source_model="teacher/model",
        source_model_revision="1" * 40,
        prompt="Name the element with atomic number 1.",
        output=output,
        teacher_tokens=9,
        teacher_token_counter="authoritative_generated_token_ids",
        knowledge_class=SPECIALIST_KNOWLEDGE,
        content_basis="specialist_fact",
        domain_labels=["chemistry"],
        domain_claims=["periodic_table:atomic_number_1=hydrogen"],
        label_method="preregistered_catalog",
        label_evidence_sha256="d" * 64,
        output_introduces_unsupplied_facts=True,
    )


def test_manifest_certifies_bounded_core_domain_separation() -> None:
    ontology = _ontology()
    records = [_english(), _chemistry()]
    manifest = build_core_domain_segregation_manifest(
        records, domain_ontology=ontology
    )
    assert manifest["status"] == "PASS"
    assert manifest["selected_domains"] == ["chemistry"]
    assert manifest["unselected_domain_records_in_bundle"] == 0
    assert manifest["absolute_zero_world_knowledge_claimed"] is False
    assert all(manifest["gates"].values())
    validate_core_domain_segregation_manifest(
        manifest, records, domain_ontology=ontology
    )
    for record in records:
        validate_segregated_extraction_record(record)
        validate_labeled_extraction_record(record)


def test_english_core_cannot_carry_domain_labels_or_claims() -> None:
    kwargs = {
        "destination_scope": "english_core",
        "capability": "grammar",
        "domain": "domain_independent",
        "provenance": "bad",
        "split": "search",
        "source_model": "teacher/model",
        "source_model_revision": "1" * 40,
        "prompt": "Correct this sentence.",
        "output": "This sentence is correct.",
        "teacher_tokens": 5,
        "teacher_token_counter": "authoritative_generated_token_ids",
        "knowledge_class": LINGUISTIC_FORM,
        "content_basis": "domain_free_instruction",
        "domain_labels": ["chemistry"],
        "domain_claims": ["periodic_table:hydrogen"],
        "label_method": "human_review",
        "label_evidence_sha256": "e" * 64,
        "output_introduces_unsupplied_facts": False,
    }
    with pytest.raises(
        CapabilitySegregationError, match="domain labels or claims"
    ):
        build_segregated_extraction_record(**kwargs)


def test_known_domain_marker_in_english_fails_closed() -> None:
    record = _english(
        prompt="Rewrite clearly: Hydrogen has atomic number one."
    )
    with pytest.raises(
        CapabilitySegregationError, match="known domain marker"
    ):
        build_core_domain_segregation_manifest(
            [record], domain_ontology=_ontology()
        )


def test_normalized_content_cannot_exist_in_core_and_domain() -> None:
    english = _english()
    domain = _chemistry(output=english["output"].upper())
    with pytest.raises(
        CapabilitySegregationError, match="content overlap"
    ):
        build_core_domain_segregation_manifest(
            [english, domain], domain_ontology=_ontology()
        )


def test_domain_record_requires_exact_label_and_atomic_claim() -> None:
    with pytest.raises(
        CapabilitySegregationError, match="exactly its destination"
    ):
        build_segregated_extraction_record(
            destination_scope="domain_cake",
            capability="periodic_table",
            domain="chemistry",
            provenance="bad",
            split="search",
            source_model="teacher/model",
            source_model_revision="1" * 40,
            prompt="Name element one.",
            output="Hydrogen.",
            teacher_tokens=2,
            teacher_token_counter="authoritative_generated_token_ids",
            knowledge_class=SPECIALIST_KNOWLEDGE,
            content_basis="specialist_fact",
            domain_labels=["chemistry", "physics"],
            domain_claims=["periodic_table:atomic_number_1=hydrogen"],
            label_method="human_review",
            label_evidence_sha256="f" * 64,
            output_introduces_unsupplied_facts=True,
        )


def test_ambiguous_record_is_preserved_but_cannot_enter_training() -> None:
    ambiguous = build_segregated_extraction_record(
        destination_scope="domain_cake",
        capability="unresolved_fact",
        domain="quarantine",
        provenance="triage:ambiguous-1",
        split="search",
        source_model="teacher/model",
        source_model_revision="1" * 40,
        prompt="Explain the overlapping concept.",
        output="The response spans chemistry and mathematics.",
        teacher_tokens=9,
        teacher_token_counter="authoritative_generated_token_ids",
        knowledge_class=QUARANTINED,
        content_basis="cross_domain_conflict",
        domain_labels=["chemistry", "mathematics"],
        domain_claims=[],
        label_method="human_review",
        label_evidence_sha256="f" * 64,
        output_introduces_unsupplied_facts=True,
    )
    validate_segregated_extraction_record(ambiguous)
    with pytest.raises(
        CapabilitySegregationError, match="cannot enter a training bundle"
    ):
        build_core_domain_segregation_manifest(
            [_english(), ambiguous],
            domain_ontology=_ontology(),
        )


def test_tampered_semantic_label_invalidates_record_and_manifest() -> None:
    ontology = _ontology()
    records = [_english(), _chemistry()]
    manifest = build_core_domain_segregation_manifest(
        records, domain_ontology=ontology
    )
    changed_record = copy.deepcopy(records[0])
    changed_record["content_basis"] = "specialist_fact"
    with pytest.raises(CapabilitySegregationError):
        validate_segregated_extraction_record(changed_record)

    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["selected_domains"] = ["chemistry", "python"]
    with pytest.raises(CapabilitySegregationError, match="manifest"):
        validate_core_domain_segregation_manifest(
            changed_manifest, records, domain_ontology=ontology
        )


def test_ontology_rejects_ambiguous_markers() -> None:
    with pytest.raises(
        CapabilitySegregationError, match="ambiguous core exclusion marker"
    ):
        build_domain_ontology(
            [
                {
                    "domain_id": "one",
                    "description": "First.",
                    "capabilities": ["a"],
                    "core_exclusion_markers": ["shared marker"],
                    "label_evidence_sha256": "a" * 64,
                },
                {
                    "domain_id": "two",
                    "description": "Second.",
                    "capabilities": ["b"],
                    "core_exclusion_markers": ["shared marker"],
                    "label_evidence_sha256": "b" * 64,
                },
            ],
            ontology_id="ambiguous",
        )


def test_checked_in_contract_and_ontology_lock_correct_claim_boundary() -> None:
    contract = json.loads(
        (
            ROOT / "ABI_ENGLISH_CORE_DOMAIN_SEGREGATION_CONTRACT_V2.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["format"] == (
        "abi-english-core-domain-segregation-contract/2"
    )
    assert (
        contract["claim_vocabulary"]["foreign_teacher_to_layercake"]
        .casefold()
        .find("never called lossless")
        >= 0
    )
    assert contract["claim_vocabulary"][
        "layercake_to_layercake_package_transfer"
    ].startswith("Lossless only")
    assert contract["domain_artifact_scope"][
        "unselected_domains_absent_from_training_bundle"
    ] is True
    assert contract["segregation_gates"]["english_domain_label_count"] == 0
    assert contract["segregation_gates"]["english_domain_claim_count"] == 0
    assert contract["current_result"]["product_moonshot_passed"] is False

    ontology = json.loads(
        (ROOT / "catalogs" / "domain_ontology_v1.json").read_text(
            encoding="utf-8"
        )
    )
    validate_domain_ontology(ontology)
    assert ontology["discovery_exhaustive_claimed"] is False
    catalog_value = json.loads(
        (
            ROOT
            / "catalogs"
            / "english_and_first_domains_certification_v6.json"
        ).read_text(encoding="utf-8")
    )
    catalog_sha = hashlib.sha256(
        json.dumps(
            catalog_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert {
        row["label_evidence_sha256"] for row in ontology["domains"]
    } == {catalog_sha}


def test_implementation_certificate_binds_code_contract_and_catalog() -> None:
    historical_path = (
        ROOT / "ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE.json"
    )
    assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == (
        "60ca048704c17e36e331afd8f6694783f56d1e06ba223d2d8ebfb9f80a1b7700"
    )
    historical_v2_path = (
        ROOT / "ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE_V2.json"
    )
    assert hashlib.sha256(historical_v2_path.read_bytes()).hexdigest() == (
        "4922120a3066c6a93e2146b3015610ef98a661d42a610f0bd7cb53720415974b"
    )
    historical_v3_path = (
        ROOT / "ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE_V3.json"
    )
    assert hashlib.sha256(historical_v3_path.read_bytes()).hexdigest() == (
        "2639dcfc1048838fc181afeb36ee556f80302df9bb61a7e4c7fad9f8176ca0ab"
    )
    historical_v4_path = (
        ROOT / "ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE_V4.json"
    )
    assert hashlib.sha256(historical_v4_path.read_bytes()).hexdigest() == (
        "44c5b1ba6b27897e5ef530bf0c454adb12befcd4e92a9debfc6e41f9aa3217d2"
    )
    certificate = json.loads(
        (
            ROOT
            / "ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE_V5.json"
        ).read_text(encoding="utf-8")
    )
    assert certificate["status"] == "PASS_IMPLEMENTATION_AND_CONTRACT_GATES"
    assert certificate["current_product_state"][
        "broad_english_moonshot_complete"
    ] is False
    bound_paths = {
        certificate["controlling_contract"]["path"]: certificate[
            "controlling_contract"
        ]["sha256"],
        certificate["catalog"]["path"]: certificate["catalog"]["sha256"],
        certificate["domain_ontology"]["path"]: certificate[
            "domain_ontology"
        ]["file_sha256"],
        **certificate["implementation_files"],
    }
    for relative_path, expected_sha in bound_paths.items():
        assert hashlib.sha256(
            (ROOT / relative_path).read_bytes()
        ).hexdigest() == expected_sha
