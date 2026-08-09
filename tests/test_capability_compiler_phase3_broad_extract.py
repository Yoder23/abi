import hashlib
import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3_broad_extract import (
    BroadExtractionError,
    PROTOCOL_FORMAT,
    _group_by_generation_budget,
    _repair_candidates,
    verify_extraction_protocol,
)
from abi.hf_extraction import PROBE_CATALOG_SCHEMA


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_extraction_protocol_binds_catalog_and_files(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": PROBE_CATALOG_SCHEMA,
                "catalog_id": "test",
                "probes": [{
                    "probe_id": "p1", "destination_scope": "english_core",
                    "domain": "domain_independent", "capability": "grammar",
                    "split": "search", "prompt": "Fix this.", "max_new_tokens": 8,
                    "seed": 1, "evaluator": {"kind": "nonempty", "minimum_characters": 1}
                }],
            }
        ), encoding="utf-8"
    )
    bound = tmp_path / "bound.txt"
    bound.write_text("fixed", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "format": PROTOCOL_FORMAT,
        "catalog": {"path": "catalog.json", "sha256": _sha(catalog), "search_probe_count": 1},
        "bindings": {"bound.txt": _sha(bound)},
    }), encoding="utf-8")
    assert verify_extraction_protocol(protocol)["search_probe_count"] == 1
    bound.write_text("changed", encoding="utf-8")
    with pytest.raises(BroadExtractionError, match="binding mismatch"):
        verify_extraction_protocol(protocol)


def test_generation_groups_never_mix_token_budgets() -> None:
    rows = [
        {"probe_id": "a", "max_new_tokens": 80},
        {"probe_id": "b", "max_new_tokens": 64},
        {"probe_id": "c", "max_new_tokens": 80},
    ]
    groups = _group_by_generation_budget(rows)
    assert [maximum for maximum, _ in groups] == [64, 80]
    assert all(
        {row["max_new_tokens"] for row in group} == {maximum}
        for maximum, group in groups
    )


def test_zero_repair_rounds_produces_no_candidates() -> None:
    probe = {"probe_id": "p1", "max_new_tokens": 64}
    assert _repair_candidates(
        ["p1"], {"p1": probe}, {}, maximum_rounds=0
    ) == []
