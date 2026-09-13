"""Fail-closed R67 catalog shape and historical-overlap verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog, probe_label_evidence_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, default=Path("catalogs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog_path = args.catalog.resolve()
    catalog = load_probe_catalog(catalog_path)
    probes = list(catalog["probes"])
    errors = []
    if len(probes) != 2_000 or len({row["probe_id"] for row in probes}) != 2_000 or len({row["prompt"] for row in probes}) != 2_000:
        errors.append("matrix_identity")
    for row in probes:
        if (
            row.get("split") != "search"
            or row.get("capability") != "domain_independent_reasoning"
            or row.get("destination_scope") != "english_core"
            or row.get("domain") != "domain_independent"
            or row.get("knowledge_class") != "english_linguistic_form"
            or row.get("domain_labels") != [] or row.get("domain_claims") != []
            or row.get("output_introduces_unsupplied_facts") is not False
            or row.get("label_evidence_sha256") != probe_label_evidence_sha256(row)
        ):
            errors.append(str(row.get("probe_id")))
    current = {str(row["prompt"]) for row in probes}
    prior = set()
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
        "format": "abi-r67-reasoning-catalog-verification/1",
        "verdict": "PASS" if not errors and not (current & prior) else "FAIL",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "rows": len(probes), "unique_prompts": len(current),
        "historical_prompt_strings": len(prior),
        "exact_historical_overlap": len(current & prior),
        "validation_errors": errors,
        "candidate_loaded": False, "teacher_loaded": False,
        "promotion_eligible": False, "full_abi_moonshot": "OPEN",
    }
    if result["verdict"] != "PASS":
        raise SystemExit(json.dumps(result, indent=2, sort_keys=True))
    if args.output.exists():
        raise SystemExit("R67 verification output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
