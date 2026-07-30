from __future__ import annotations

import json
from pathlib import Path

import pytest

from abi.moonshot_release import (
    CERTIFICATE_FORMAT,
    CERTIFICATE_FORMATS,
    MoonshotReleaseError,
    _canonical_sha,
    _claim_hash,
    _relative_file,
)
from abi.hf_extraction import load_probe_catalog


def test_release_certificate_versions_are_explicit() -> None:
    assert CERTIFICATE_FORMAT.endswith("/2")
    assert "abi-layercake-moonshot-release-certificate/1" in (
        CERTIFICATE_FORMATS
    )
    assert CERTIFICATE_FORMAT in CERTIFICATE_FORMATS


def test_claim_hash_fails_closed_after_tampering() -> None:
    value = {"format": CERTIFICATE_FORMAT, "status": "PASS"}
    value["evidence_sha256"] = _canonical_sha(value)
    assert _claim_hash(value) == value["evidence_sha256"]
    value["status"] = "FAIL"
    with pytest.raises(MoonshotReleaseError, match="hash mismatch"):
        _claim_hash(value)


def test_release_paths_cannot_escape_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.json"
    outside.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    with pytest.raises(MoonshotReleaseError, match="escapes"):
        _relative_file(tmp_path.resolve(), "../outside.json")


def test_postcert_audit_preserves_bounded_pass_and_blocks_broad_claim() -> None:
    root = Path(__file__).resolve().parents[1]
    catalog = load_probe_catalog(
        root / "catalogs/postcert_novel_english_audit_v1.json"
    )
    assert len(catalog["probes"]) == 28
    assert len({row["capability"] for row in catalog["probes"]}) == 14

    decision = json.loads(
        (root / "ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = decision.pop("evidence_sha256")
    assert claimed == _canonical_sha(decision)
    assert decision["status"] == "FAIL_BROAD_ENGLISH_MOONSHOT_REMAINS_OPEN"
    assert decision["locked_layercake"]["passes"] == 0
    assert decision["frozen_source"]["passes"] == 19
    assert decision["comparison"]["source_passing_regressions"] == 19
    assert decision["effect_on_prior_evidence"][
        "v2_locked_suite_claim_remains_valid"
    ]
    assert not decision["effect_on_prior_evidence"][
        "broad_product_moonshot_complete"
    ]
