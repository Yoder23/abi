from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import load_probe_catalog, probe_label_evidence_sha256
from abi.natural_prompt_grounding_expansion import (
    NaturalPromptGroundingExpansionError,
    build_expansion_catalogs,
)


CAPABILITIES = (
    "coherence",
    "conversation",
    "email_drafting",
    "format_control",
    "grammar",
    "instruction_following",
    "prompt_grounding",
    "rewriting",
    "summarization",
    "tone_control",
)


def _parent(tmp_path: Path, rows_per_capability: int = 70) -> tuple[dict, Path]:
    probes = []
    for capability in CAPABILITIES:
        for index in range(rows_per_capability):
            prompt = f"Natural {capability} request {index} with supplied words."
            probe = {
                "probe_id": f"parent-{capability}-{index}",
                "destination_scope": "english_core",
                "capability": capability,
                "domain": "domain_independent",
                "split": "search",
                "prompt": prompt,
                "max_new_tokens": 128,
                "temperature": 0,
                "seed": index,
                "evaluator": {"kind": "nonempty", "minimum_characters": 1},
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": "domain_free_instruction",
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
                "natural_prompt_sha256": "a" * 64,
                "source_prompt_corpus": "fixture",
                "source_shard_sha256": "b" * 64,
                "source_row_index": index,
                "corpus_assistant_messages_imported": 0,
                "corpus_reference_answers_imported": 0,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)
    parent = {
        "schema_version": "abi-capability-probe-catalog/1",
        "catalog_id": "fixture-parent",
        "probes": probes,
    }
    path = tmp_path / "parent.json"
    import json

    path.write_text(json.dumps(parent), encoding="utf-8")
    return parent, path


def test_builds_disjoint_full_and_preflight_catalogs(tmp_path: Path) -> None:
    parent, path = _parent(tmp_path)
    full, preflight = build_expansion_catalogs(
        parent_catalog=parent,
        parent_path=path,
        validation_per_capability=64,
        preflight_per_capability=2,
    )
    assert len(full["probes"]) == 9 * 70
    assert len(preflight["probes"]) == 9 * 2
    assert {row["capability"] for row in full["probes"]} == set(CAPABILITIES) - {
        "grammar"
    }
    for capability in set(CAPABILITIES) - {"grammar"}:
        rows = [row for row in full["probes"] if row["capability"] == capability]
        assert sum(row["split"] == "validation" for row in rows) == 64
        assert sum(row["split"] == "search" for row in rows) == 6
    assert all(row["split"] == "search" for row in preflight["probes"])
    assert all(row["max_new_tokens"] == 256 for row in full["probes"])
    assert all(row["domain_labels"] == [] for row in full["probes"])

    full_path = tmp_path / "full.json"
    preflight_path = tmp_path / "preflight.json"
    import json

    full_path.write_text(json.dumps(full), encoding="utf-8")
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    load_probe_catalog(full_path)
    load_probe_catalog(preflight_path)


def test_rejects_insufficient_split_depth(tmp_path: Path) -> None:
    parent, path = _parent(tmp_path, rows_per_capability=65)
    with pytest.raises(NaturalPromptGroundingExpansionError):
        build_expansion_catalogs(
            parent_catalog=parent,
            parent_path=path,
            validation_per_capability=64,
            preflight_per_capability=2,
        )


def test_catalog_is_deterministic(tmp_path: Path) -> None:
    parent, path = _parent(tmp_path)
    first = build_expansion_catalogs(
        parent_catalog=deepcopy(parent),
        parent_path=path,
        validation_per_capability=64,
        preflight_per_capability=2,
    )
    second = build_expansion_catalogs(
        parent_catalog=deepcopy(parent),
        parent_path=path,
        validation_per_capability=64,
        preflight_per_capability=2,
    )
    assert first == second
