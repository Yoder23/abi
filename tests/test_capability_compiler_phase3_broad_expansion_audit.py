import json
from pathlib import Path

import pytest

from abi.broad_english_catalog import build_catalog
from abi.capability_compiler_phase3_broad_expansion_audit import (
    BroadExpansionAuditError,
    audit_catalog,
)
from abi.hf_extraction import PROBE_CATALOG_SCHEMA, probe_label_evidence_sha256


def _story(name: str) -> str:
    return (
        f"{name} found a quiet blue basket beside the garden path. "
        f"Then {name} carried the basket toward a friendly neighbor. "
        "The neighbor smiled and opened the wooden gate. "
        "Together they placed the basket safely on a small table."
    )


def _catalog(tmp_path: Path) -> Path:
    catalog = build_catalog(
        search_stories=[_story("Lina")],
        validation_stories=[_story("Mara")],
        final_stories=[_story("Nora")],
        corpus_manifest={
            "train_arrow": {"sha256": "a" * 64},
            "validation_arrow": {"sha256": "b" * 64},
            "closed_book_fact_prompts": 0,
        },
    )
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def _historical(tmp_path: Path) -> Path:
    path = tmp_path / "historical.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": PROBE_CATALOG_SCHEMA,
                "catalog_id": "historical-test-v1",
                "probes": [
                    {
                        "probe_id": "historical-1",
                        "destination_scope": "english_core",
                        "domain": "domain_independent",
                        "capability": "grammar",
                        "split": "search",
                        "prompt": "Correct this unrelated historical sentence.",
                        "max_new_tokens": 16,
                        "seed": 1,
                        "evaluator": {"kind": "nonempty", "minimum_characters": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_audit_accepts_balanced_disjoint_catalog(tmp_path: Path) -> None:
    result = audit_catalog(
        _catalog(tmp_path),
        _historical(tmp_path),
        expected_story_counts={"search": 1, "validation": 1, "final_test": 1},
        expected_train_sha256="a" * 64,
        expected_validation_sha256="b" * 64,
    )
    assert result["status"] == "PASS"
    assert result["exact_historical_prompt_overlap"] == 0
    assert result["segregation_failures"] == 0


def test_audit_rejects_wrong_source_identity(tmp_path: Path) -> None:
    with pytest.raises(BroadExpansionAuditError, match="train_corpus_sha256_mismatch"):
        audit_catalog(
            _catalog(tmp_path),
            _historical(tmp_path),
            expected_story_counts={"search": 1, "validation": 1, "final_test": 1},
            expected_train_sha256="c" * 64,
            expected_validation_sha256="b" * 64,
        )


def test_audit_rejects_duplicate_prompts(tmp_path: Path) -> None:
    candidate = _catalog(tmp_path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["probes"][1]["prompt"] = payload["probes"][0]["prompt"]
    payload["probes"][1]["label_evidence_sha256"] = probe_label_evidence_sha256(
        payload["probes"][1]
    )
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BroadExpansionAuditError, match="duplicate_candidate_prompts"):
        audit_catalog(
            candidate,
            _historical(tmp_path),
            expected_story_counts={"search": 1, "validation": 1, "final_test": 1},
            expected_train_sha256="a" * 64,
            expected_validation_sha256="b" * 64,
        )
