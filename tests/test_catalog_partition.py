from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from abi.catalog_partition import partition_catalog, write_partition
from abi.natural_conversation_catalog import (
    _ultrachat_candidates,
    build_catalog,
)


SHA = "a" * 64


def _catalog() -> dict:
    rows = []
    for index in range(12):
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Write a friendly apology using the unique word "
                            f"marker{chr(97 + index)} while remaining concise."
                        ),
                    }
                ]
            }
        )
    candidates = list(_ultrachat_candidates(rows, shard_sha256=SHA))
    return build_catalog(
        ultrachat_candidates=candidates,
        oasst_candidates=[],
        source_corpora=[],
        maximum_per_capability=20,
        minimum_per_capability=1,
        seed=7,
    )


def test_partition_is_exact_disjoint_and_does_not_change_probes() -> None:
    parent = _catalog()
    original_by_id = {
        probe["probe_id"]: deepcopy(probe) for probe in parent["probes"]
    }
    shards = partition_catalog(parent, parent_sha256=SHA, shard_count=4)
    observed = {}
    for shard in shards:
        for probe in shard["probes"]:
            assert probe["probe_id"] not in observed
            observed[probe["probe_id"]] = probe
    assert observed == original_by_id


def test_written_partition_reloads_and_binds_parent(tmp_path: Path) -> None:
    parent = _catalog()
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    output = tmp_path / "shards"
    manifest = write_partition(
        parent_path=parent_path,
        output_directory=output,
        shard_count=3,
    )
    assert manifest["status"] == "PASS_EXACT_DISJOINT_PARTITION"
    assert manifest["union_probe_count"] == len(parent["probes"])
    assert manifest["intersection_probe_count"] == 0
    assert manifest["probe_payload_changes"] == 0
    assert len(manifest["shards"]) == 3
