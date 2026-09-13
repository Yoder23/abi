"""Verify the frozen R75 matrix and exact non-overlap with prior catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from abi.hf_extraction import load_probe_catalog, probe_label_evidence_sha256


EXPECTED_SHA256 = "e8cfc9199d05b796c7625803706a62b245b6537d2962924123fdf8d9c607f086"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--prior-root", default=Path("catalogs"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    catalog_path = args.catalog.resolve()
    catalog = load_probe_catalog(catalog_path)
    probes = list(catalog["probes"])
    families = Counter(str(row["probe_id"]).split("-")[2] for row in probes)
    errors = []
    if (
        hashlib.sha256(catalog_path.read_bytes()).hexdigest() != EXPECTED_SHA256
        or len(probes) != 700
        or len({row["probe_id"] for row in probes}) != 700
        or len({row["prompt"] for row in probes}) != 700
        or families != {f"f{index}": 100 for index in range(7)}
    ):
        errors.append("matrix_identity")
    for row in probes:
        expected = str(row.get("evaluator", {}).get("values", [""])[0])
        if (
            row.get("split") != "validation"
            or row.get("capability") != "domain_independent_reasoning"
            or row.get("destination_scope") != "english_core"
            or row.get("knowledge_class") != "english_linguistic_form"
            or row.get("content_basis") != "abstract_or_nonce_content"
            or row.get("domain_labels") != []
            or row.get("domain_claims") != []
            or row.get("output_introduces_unsupplied_facts") is not False
            or not expected.startswith("F5")
            or row.get("label_evidence_sha256") != probe_label_evidence_sha256(row)
        ):
            errors.append(str(row.get("probe_id")))
    current = {str(row["prompt"]) for row in probes}
    prior: set[str] = set()
    for path in sorted(args.prior_root.resolve().glob("*.json")):
        if path == catalog_path:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for row in value.get("probes", []) if isinstance(value, dict) else []:
            if isinstance(row, dict) and isinstance(row.get("prompt"), str):
                prior.add(row["prompt"])
    result = {
        "format": "abi-r75-prospective-validation-catalog-verification/1",
        "verdict": "PASS" if not errors and not current.intersection(prior) else "FAIL",
        "catalog_sha256": EXPECTED_SHA256,
        "rows": len(probes),
        "unique_prompts": len(current),
        "family_counts": dict(sorted(families.items())),
        "prior_prompt_strings": len(prior),
        "exact_prior_overlap": len(current.intersection(prior)),
        "validation_errors": errors,
        "candidate_loaded": False,
        "teacher_loaded": False,
        "full_abi_moonshot": "OPEN",
    }
    if result["verdict"] != "PASS" or args.output.exists():
        raise SystemExit(json.dumps(result, indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
