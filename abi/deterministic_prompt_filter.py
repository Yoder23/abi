"""Fail-closed deterministic filtering for supplied-context candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hf_extraction import load_probe_catalog
from .layercake_host import _canonical_json_bytes, _sha256_file
from .prompt_domain_qualification import _deterministic_rejection_reasons


CATALOG_ID = "abi-supplied-context-deterministic-qualified-search-validation-v78"


class DeterministicPromptFilterError(RuntimeError):
    """Raised when the deterministic prompt filter cannot be reproduced."""


def build_deterministic_qualified_catalog(
    *,
    parent_catalog: Mapping[str, Any],
    parent_path: Path,
    minimum_search_rows: int = 100,
    minimum_validation_rows: int = 32,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    quarantined = []
    reason_counts: Counter[str] = Counter()
    for probe in parent_catalog.get("probes", []):
        reasons = _deterministic_rejection_reasons(str(probe["prompt"]))
        if reasons:
            reason_counts.update(reasons)
            quarantined.append(
                {
                    "probe_id": probe["probe_id"],
                    "prompt_sha256": hashlib.sha256(
                        str(probe["prompt"]).encode("utf-8")
                    ).hexdigest(),
                    "reasons": reasons,
                }
            )
            continue
        grouped[str(probe["capability"])][str(probe["split"])].append(probe)

    available = sorted(
        capability
        for capability, splits in grouped.items()
        if len(splits.get("search", [])) >= minimum_search_rows
        and len(splits.get("validation", [])) >= minimum_validation_rows
    )
    if not available:
        raise DeterministicPromptFilterError(
            "no capability retains the required deterministic safe depth"
        )
    selected = []
    for capability in available:
        for split in ("search", "validation"):
            for probe in grouped[capability][split]:
                row = dict(probe)
                row["deterministic_prompt_filter"] = {
                    "status": "PASS",
                    "filter_source_sha256": _sha256_file(
                        Path(__file__).resolve()
                    ),
                    "rejection_reasons": [],
                }
                selected.append(row)
    counts = Counter((row["capability"], row["split"]) for row in selected)
    evidence_basis = {
        "parent_catalog_sha256": _sha256_file(parent_path),
        "minimum_search_rows": minimum_search_rows,
        "minimum_validation_rows": minimum_validation_rows,
        "available_capabilities": available,
        "selected_probe_ids": sorted(str(row["probe_id"]) for row in selected),
        "quarantined": quarantined,
    }
    evidence_sha = hashlib.sha256(_canonical_json_bytes(evidence_basis)).hexdigest()
    for row in selected:
        row["deterministic_prompt_filter"]["evidence_sha256"] = evidence_sha
    return {
        **{
            key: value
            for key, value in parent_catalog.items()
            if key not in {"catalog_id", "status", "generation", "probes"}
        },
        "catalog_id": CATALOG_ID,
        "status": "DETERMINISTIC_QUALIFICATION_PASS_AWAITING_MODEL_QUALIFICATION",
        "parent_candidate_catalog": {
            "path": parent_path.as_posix(),
            "sha256": _sha256_file(parent_path),
            "catalog_id": parent_catalog["catalog_id"],
            "probes": len(parent_catalog["probes"]),
        },
        "generation": {
            "generator": "abi.deterministic_prompt_filter",
            "filter_source_sha256": _sha256_file(Path(__file__).resolve()),
            "minimum_search_rows": minimum_search_rows,
            "minimum_validation_rows": minimum_validation_rows,
            "available_capabilities": available,
            "unavailable_capabilities": sorted(
                set(grouped) - set(available)
            ),
            "capability_split_counts": {
                capability: {
                    split: counts[(capability, split)]
                    for split in ("search", "validation")
                }
                for capability in available
            },
            "selected_probes": len(selected),
            "search_probes": sum(row["split"] == "search" for row in selected),
            "validation_probes": sum(
                row["split"] == "validation" for row in selected
            ),
            "quarantined_probes": len(quarantined),
            "quarantine_reason_counts": dict(sorted(reason_counts.items())),
            "deterministic_evidence_sha256": evidence_sha,
            "final_test_probes": 0,
        },
        "quarantine": {
            "records": quarantined,
            "record_count": len(quarantined),
            "training_eligible": False,
        },
        "probes": selected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-search-rows", type=int, default=100)
    parser.add_argument("--minimum-validation-rows", type=int, default=32)
    args = parser.parse_args(argv)
    parent_path = Path(args.parent_catalog).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        parser.error(f"catalog is immutable: {output_path}")
    parent = load_probe_catalog(parent_path)
    catalog = build_deterministic_qualified_catalog(
        parent_catalog=parent,
        parent_path=parent_path,
        minimum_search_rows=args.minimum_search_rows,
        minimum_validation_rows=args.minimum_validation_rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    load_probe_catalog(output_path)
    print(
        json.dumps(
            {
                "path": str(output_path),
                "sha256": _sha256_file(output_path),
                "generation": catalog["generation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
