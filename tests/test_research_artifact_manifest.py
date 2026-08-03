import hashlib
import json
from pathlib import Path

import pytest

from abi.research_artifact_manifest import build_manifest, inventory_large_catalogs, write_manifest


def test_inventory_is_deterministic_relative_and_hash_bound(tmp_path: Path):
    catalogs = tmp_path / "catalogs"
    catalogs.mkdir()
    (catalogs / "small.json").write_text("{}", encoding="utf-8")
    payload = b"bounded-large-catalog"
    (catalogs / "large.json").write_bytes(payload)

    first = inventory_large_catalogs(
        tmp_path,
        threshold_bytes=8,
        tracked_paths={"catalogs/large.json"},
    )
    second = inventory_large_catalogs(
        tmp_path,
        threshold_bytes=8,
        tracked_paths={"catalogs/large.json"},
    )

    assert first == second
    assert first == [
        {
            "path": "catalogs/large.json",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "classification": "historical_generated_catalog",
            "git_state": "TRACKED",
            "retention": "PRESERVED_AND_CONTENT_ADDRESSED",
        }
    ]


def test_manifest_records_policy_counts_and_round_trips(tmp_path: Path):
    catalogs = tmp_path / "catalogs"
    catalogs.mkdir()
    (catalogs / "one.json").write_bytes(b"12345678")
    (catalogs / "two.json").write_bytes(b"abcdefghij")

    manifest = build_manifest(
        tmp_path,
        status_date="2026-08-03",
        threshold_bytes=8,
        tracked_paths={"catalogs/two.json"},
    )
    output = tmp_path / "manifest.json"
    write_manifest(output, manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert manifest["artifact_count"] == 2
    assert manifest["total_bytes"] == 18
    assert manifest["tracked_artifact_count"] == 1
    assert manifest["local_only_artifact_count"] == 1
    assert manifest["git_policy"]["deletion_authorized"] is False


def test_inventory_rejects_invalid_threshold(tmp_path: Path):
    (tmp_path / "catalogs").mkdir()
    with pytest.raises(ValueError, match="positive"):
        inventory_large_catalogs(tmp_path, threshold_bytes=0)
