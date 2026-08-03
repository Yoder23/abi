from __future__ import annotations

import json
from pathlib import Path

from abi.capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from abi.deterministic_prompt_filter import build_deterministic_qualified_catalog
from abi.hf_extraction import load_probe_catalog, probe_label_evidence_sha256


def _probe(capability: str, split: str, index: int, text: str) -> dict:
    prompt = f"Rewrite this. <supplied_text>{text}</supplied_text>"
    value = {
        "probe_id": f"{capability}-{split}-{index}",
        "destination_scope": "english_core",
        "capability": capability,
        "domain": "domain_independent",
        "split": split,
        "prompt": prompt,
        "max_new_tokens": 64,
        "seed": index,
        "evaluator": {"kind": "nonempty", "minimum_characters": 1},
        "record_schema": SEGREGATED_RECORD_SCHEMA,
        "knowledge_class": LINGUISTIC_FORM,
        "content_basis": "supplied_non_domain_context",
        "domain_labels": [],
        "domain_claims": [],
        "label_method": "preregistered_catalog",
        "output_introduces_unsupplied_facts": False,
    }
    value["label_evidence_sha256"] = probe_label_evidence_sha256(value)
    return value


def test_filter_quarantines_markers_and_keeps_only_deep_capabilities(
    tmp_path: Path,
) -> None:
    probes = []
    for capability, search_count in (("rewriting", 4), ("conversation", 2)):
        probes.extend(
            _probe(capability, "search", index, "Aster sent a calm note.")
            for index in range(search_count)
        )
        probes.extend(
            _probe(capability, "validation", index, "Velin read the blue note.")
            for index in range(2)
        )
    probes.append(
        _probe("rewriting", "search", 99, "A catalyst changed activation energy.")
    )
    parent = {
        "schema_version": "abi-capability-probe-catalog/1",
        "catalog_id": "parent",
        "probes": probes,
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    output = build_deterministic_qualified_catalog(
        parent_catalog=parent,
        parent_path=parent_path,
        minimum_search_rows=3,
        minimum_validation_rows=2,
    )
    assert output["generation"]["available_capabilities"] == ["rewriting"]
    assert output["generation"]["quarantined_probes"] == 1
    assert all(
        row["capability"] == "rewriting" for row in output["probes"]
    )
    output_path = tmp_path / "output.json"
    output_path.write_text(json.dumps(output), encoding="utf-8")
    load_probe_catalog(output_path)
