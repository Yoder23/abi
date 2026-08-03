"""Partition a validated search catalog into disjoint stratified shards.

This is an operational transport mechanism only.  It does not change probe
content, labels, generation settings, or source identity.  Every original
probe appears in exactly one shard and keeps its original evidence binding.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hf_extraction import load_probe_catalog


FORMAT = "abi-probe-catalog-partition/1"


class CatalogPartitionError(RuntimeError):
    """Raised when an exact disjoint partition cannot be produced."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_catalog(
    catalog: Mapping[str, Any],
    *,
    parent_sha256: str,
    shard_count: int,
) -> list[dict[str, Any]]:
    """Return capability-stratified catalogs with an exact probe partition."""

    if shard_count < 2:
        raise CatalogPartitionError("shard_count must be at least two")
    probes = catalog.get("probes")
    if not isinstance(probes, list) or not probes:
        raise CatalogPartitionError("parent catalog has no probes")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_ids: set[str] = set()
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise CatalogPartitionError("parent probe is not an object")
        probe_id = str(probe["probe_id"])
        if probe_id in original_ids:
            raise CatalogPartitionError("parent probe IDs are not unique")
        original_ids.add(probe_id)
        grouped[str(probe["capability"])].append(dict(probe))

    shard_probes: list[list[dict[str, Any]]] = [
        [] for _ in range(shard_count)
    ]
    for capability in sorted(grouped):
        ordered = sorted(grouped[capability], key=lambda row: row["probe_id"])
        for position, probe in enumerate(ordered):
            shard_probes[position % shard_count].append(probe)

    partitions: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for shard_index, selected in enumerate(shard_probes):
        selected = sorted(selected, key=lambda row: row["probe_id"])
        ids = [str(probe["probe_id"]) for probe in selected]
        overlap = observed_ids.intersection(ids)
        if overlap:
            raise CatalogPartitionError("probe crossed shard boundary")
        observed_ids.update(ids)
        capability_counts = Counter(
            str(probe["capability"]) for probe in selected
        )
        source_counts = Counter(
            str(probe.get("source_prompt_corpus", "undeclared"))
            for probe in selected
        )
        partition = dict(catalog)
        generation = dict(catalog.get("generation", {}))
        generation.update(
            {
                "total_probes": len(selected),
                "capability_counts": dict(sorted(capability_counts.items())),
                "source_counts": dict(sorted(source_counts.items())),
                "partition_parent_catalog_sha256": parent_sha256,
                "partition_shard_index": shard_index,
                "partition_shard_count": shard_count,
                "partition_probe_id_set_sha256": hashlib.sha256(
                    "\n".join(ids).encode("utf-8")
                ).hexdigest(),
                "partition_changes_to_probe_payloads": 0,
            }
        )
        partition["catalog_id"] = (
            f"{catalog['catalog_id']}-shard-{shard_index + 1:02d}-"
            f"of-{shard_count:02d}"
        )
        partition["status"] = "EXACT_DISJOINT_SEARCH_TRANSPORT_SHARD"
        partition["generation"] = generation
        partition["probes"] = selected
        partitions.append(partition)
    if observed_ids != original_ids:
        raise CatalogPartitionError("partition union differs from parent")
    return partitions


def write_partition(
    *,
    parent_path: Path,
    output_directory: Path,
    shard_count: int,
) -> dict[str, Any]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise CatalogPartitionError(
            f"partition output directory is not empty: {output_directory}"
        )
    parent = load_probe_catalog(parent_path)
    parent_sha256 = _sha256_file(parent_path)
    partitions = partition_catalog(
        parent,
        parent_sha256=parent_sha256,
        shard_count=shard_count,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    shard_receipts = []
    union_ids: set[str] = set()
    file_stem = parent_path.stem
    for index, partition in enumerate(partitions):
        path = output_directory / (
            f"{file_stem}_shard_{index + 1:02d}_of_{shard_count:02d}.json"
        )
        path.write_text(
            json.dumps(
                partition,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        loaded = load_probe_catalog(path)
        ids = {str(probe["probe_id"]) for probe in loaded["probes"]}
        if union_ids & ids:
            raise CatalogPartitionError("written shards overlap")
        union_ids.update(ids)
        shard_receipts.append(
            {
                "index": index,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "probes": len(ids),
                "capability_counts": partition["generation"][
                    "capability_counts"
                ],
                "source_counts": partition["generation"]["source_counts"],
            }
        )
    parent_ids = {str(probe["probe_id"]) for probe in parent["probes"]}
    if union_ids != parent_ids:
        raise CatalogPartitionError("written shard union differs from parent")
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS_EXACT_DISJOINT_PARTITION",
        "parent_catalog": {
            "path": str(parent_path),
            "sha256": parent_sha256,
            "probes": len(parent_ids),
        },
        "shard_count": shard_count,
        "union_probe_count": len(union_ids),
        "intersection_probe_count": 0,
        "missing_probe_count": 0,
        "extra_probe_count": 0,
        "probe_payload_changes": 0,
        "shards": shard_receipts,
    }
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    manifest["evidence_sha256"] = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    manifest_path = output_directory / "partition_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--shards", type=int, default=16)
    args = parser.parse_args(argv)
    result = write_partition(
        parent_path=Path(args.catalog).resolve(),
        output_directory=Path(args.output_directory).resolve(),
        shard_count=args.shards,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
