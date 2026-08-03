from __future__ import annotations

import json
from pathlib import Path

import pytest

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.hf_extraction import load_probe_catalog, probe_label_evidence_sha256
from abi.supplied_context_expansion import (
    INCLUDED_CAPABILITIES,
    SuppliedContextExpansionError,
    build_supplied_context_catalog,
)


def _parent(tmp_path: Path, rows: int = 145) -> tuple[dict, Path]:
    probes = []
    for capability in sorted(INCLUDED_CAPABILITIES | {"grammar"}):
        for index in range(rows):
            prompt = (
                f"Rewrite this {capability} text.\n\n<supplied_text>"
                f"Aster {index} sent a calm note.</supplied_text>"
            )
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
                "content_basis": "supplied_non_domain_context",
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
                "natural_prompt_sha256": f"{index:064x}"[-64:],
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)
    parent = {
        "schema_version": "abi-capability-probe-catalog/1",
        "catalog_id": "fixture-natural-parent",
        "generation": {
            "capability_counts": {
                capability: rows
                for capability in sorted(INCLUDED_CAPABILITIES | {"grammar"})
            }
        },
        "probes": probes,
    }
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    return parent, path


def test_builds_disjoint_supplied_context_candidates(tmp_path: Path) -> None:
    parent, path = _parent(tmp_path)
    catalog = build_supplied_context_catalog(
        parent_catalog=parent,
        parent_path=path,
        validation_per_capability=40,
    )
    assert len(catalog["probes"]) == 145 * len(INCLUDED_CAPABILITIES)
    assert {row["capability"] for row in catalog["probes"]} == set(
        INCLUDED_CAPABILITIES
    )
    for capability in INCLUDED_CAPABILITIES:
        rows = [row for row in catalog["probes"] if row["capability"] == capability]
        assert sum(row["split"] == "validation" for row in rows) == 40
        assert sum(row["split"] == "search" for row in rows) == 105
    assert all("<supplied_text>" in row["prompt"] for row in catalog["probes"])
    assert all(row["max_new_tokens"] == 192 for row in catalog["probes"])
    assert all(
        row["prompt_domain_qualification_required"] is True
        for row in catalog["probes"]
    )
    output = tmp_path / "catalog.json"
    output.write_text(json.dumps(catalog), encoding="utf-8")
    load_probe_catalog(output)


def test_rejects_capability_without_search_depth(tmp_path: Path) -> None:
    parent, path = _parent(tmp_path, rows=139)
    with pytest.raises(SuppliedContextExpansionError):
        build_supplied_context_catalog(
            parent_catalog=parent,
            parent_path=path,
            validation_per_capability=40,
        )
