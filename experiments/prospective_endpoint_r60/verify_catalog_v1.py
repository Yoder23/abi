"""Fail-closed shape and disjointness verifier for the R60 catalog."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from abi.capability_compiler_phase2_common import sha256_file
from abi.hf_extraction import load_probe_catalog, probe_label_evidence_sha256
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once
from experiments.prospective_endpoint_r60.build_catalog_v1 import (
    ROWS_PER_CAPABILITY,
    build_catalog,
)


class CatalogError(RuntimeError):
    pass


def _prompts(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "prompt" and isinstance(child, str):
                found.add(child)
            else:
                found.update(_prompts(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_prompts(child))
    return found


def verify(root: Path, catalog_path: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    catalog_path = catalog_path.resolve()
    if output.exists():
        raise CatalogError(f"immutable receipt exists: {output}")
    catalog = load_probe_catalog(catalog_path)
    if catalog != build_catalog():
        raise CatalogError("materialized catalog differs from frozen builder")
    probes = catalog["probes"]
    counts = Counter(str(probe["capability"]) for probe in probes)
    prompt_set = {str(probe["prompt"]) for probe in probes}
    probe_ids = {str(probe["probe_id"]) for probe in probes}
    if (
        catalog.get("status") != "PREREGISTERED_POST_R59_PROSPECTIVE_ONLY"
        or catalog.get("generation", {}).get("endpoint_seal") != "8526fbd"
        or len(probes) != 1_400
        or len(counts) != 14
        or set(counts.values()) != {ROWS_PER_CAPABILITY}
        or len(prompt_set) != 1_400
        or len(probe_ids) != 1_400
        or any(probe["split"] != "final_test" for probe in probes)
        or any(probe["domain"] != "domain_independent" for probe in probes)
        or any(probe["destination_scope"] != "english_core" for probe in probes)
        or any(probe["domain_labels"] or probe["domain_claims"] for probe in probes)
        or any(
            probe["label_evidence_sha256"] != probe_label_evidence_sha256(probe)
            for probe in probes
        )
    ):
        raise CatalogError("prospective catalog shape or labels changed")

    prior_catalogs = []
    prior_prompts: set[str] = set()
    for path in sorted((root / "catalogs").glob("*.json")):
        if path.resolve() == catalog_path:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CatalogError(f"prior catalog unreadable: {path}") from error
        prompts = _prompts(value)
        prior_prompts.update(prompts)
        prior_catalogs.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "prompt_strings": len(prompts),
            }
        )
    overlap = prompt_set & prior_prompts
    if overlap:
        raise CatalogError(f"prospective catalog overlaps {len(overlap)} prior prompts")

    result = {
        "format": "abi-r60-prospective-catalog-verification/1",
        "verdict": "PASS_R60_CATALOG_SHAPE_AND_DISJOINTNESS",
        "catalog": {
            "path": catalog_path.relative_to(root).as_posix(),
            "sha256": sha256_file(catalog_path),
            "rows": len(probes),
            "unique_prompts": len(prompt_set),
            "unique_probe_ids": len(probe_ids),
            "capability_counts": dict(sorted(counts.items())),
        },
        "prior_catalogs": prior_catalogs,
        "prior_prompt_strings": len(prior_prompts),
        "exact_prompt_overlap": 0,
        "candidate_loaded": False,
        "teacher_loaded": False,
        "model_outputs_observed": 0,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(Path.cwd(), args.catalog, args.output)


if __name__ == "__main__":
    main()

