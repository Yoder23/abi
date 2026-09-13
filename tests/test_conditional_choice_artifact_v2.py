from __future__ import annotations

import pytest

from abi.conditional_choice_artifact_v2 import (
    DESCRIPTOR_SCHEMA,
    ConditionalChoiceArtifactV2Error,
    _canonical_sha,
    validate_descriptor,
)


def _descriptor() -> dict:
    value = {
        "schema_version": DESCRIPTOR_SCHEMA,
        "campaign": "unit",
        "catalog_sha256": "1" * 64,
        "result_file_sha256": "2" * 64,
        "raw_scores_sha256": "3" * 64,
        "result_evidence_sha256": "4" * 64,
        "source_archive_sha256": "5" * 64,
        "source_manifest_sha256": "6" * 64,
        "source_rows": 700,
        "minimum_packaged_rows": 630,
        "minimum_packaged_rows_per_family": 90,
        "result_format": "unit/1",
        "accepted_result_verdicts": ["PASS", "FAIL_SELECTIVE"],
        "allow_selective_packaging": True,
    }
    value["descriptor_sha256"] = _canonical_sha(value)
    return value


def test_descriptor_validates_exactly() -> None:
    validate_descriptor(_descriptor())


def test_descriptor_fails_closed_after_mutation() -> None:
    descriptor = _descriptor()
    descriptor["minimum_packaged_rows"] -= 1
    with pytest.raises(ConditionalChoiceArtifactV2Error, match="self-hash"):
        validate_descriptor(descriptor)


def test_descriptor_rejects_impossible_family_floor() -> None:
    descriptor = _descriptor()
    descriptor["minimum_packaged_rows_per_family"] = 101
    descriptor["descriptor_sha256"] = _canonical_sha(
        {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
    )
    with pytest.raises(ConditionalChoiceArtifactV2Error, match="impossible"):
        validate_descriptor(descriptor)
