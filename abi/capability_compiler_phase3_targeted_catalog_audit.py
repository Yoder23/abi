"""Hostile no-model audit of the targeted Phase 3 search catalog."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .capability_compiler_phase1_catalog import CAPABILITY_ALIASES
from .capability_compiler_phase3_targeted_catalog import SEARCH_PER_CAPABILITY, SOURCE_OFFSET
from .hf_extraction import load_probe_catalog, probe_label_evidence_sha256
from .natural_english_catalog import BUILDERS


FORMAT = "abi-capability-compiler-phase3-targeted-catalog-audit/1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(candidate_path: Path, prior_paths: Sequence[Path]) -> dict[str, Any]:
    candidate = load_probe_catalog(candidate_path)
    probes = list(candidate["probes"])
    failures: Counter[str] = Counter()
    if candidate.get("phase3_targeted_catalog_format") != "abi-capability-compiler-phase3-targeted-catalog/1":
        failures["format_mismatch"] += 1
    if candidate.get("catalog_id") != "abi-capability-compiler-phase3-targeted-search-v132":
        failures["catalog_id_mismatch"] += 1
    counts = Counter(str(row.get("canonical_capability")) for row in probes)
    expected = Counter({value: SEARCH_PER_CAPABILITY for value in CAPABILITY_ALIASES.values()})
    if counts != expected:
        failures["capability_balance_mismatch"] += 1
    candidate_hashes = [hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest() for row in probes]
    if len(candidate_hashes) != len(set(candidate_hashes)):
        failures["duplicate_candidate_prompts"] += len(candidate_hashes) - len(set(candidate_hashes))
    prior_hashes: set[str] = set()
    prior_records = 0
    for path in prior_paths:
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_probes = prior.get("probes")
        if not isinstance(prior_probes, list):
            raise ValueError("prior catalog lacks a probe list")
        prior_records += len(prior_probes)
        prior_hashes.update(hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest() for row in prior_probes)
    overlap = len(set(candidate_hashes) & prior_hashes)
    if overlap:
        failures["prior_prompt_overlap"] += overlap
    expected_capabilities = list(BUILDERS)
    for row in probes:
        capability = str(row.get("capability"))
        try:
            capability_index = expected_capabilities.index(capability)
        except ValueError:
            failures["unknown_capability"] += 1
            continue
        local = int(str(row.get("probe_id")).split("-")[-2])
        expected_index = SOURCE_OFFSET + capability_index * 10_000 + local
        if row.get("source_index") != expected_index:
            failures["source_index_mismatch"] += 1
        if not (
            row.get("split") == "search"
            and row.get("destination_scope") == "english_core"
            and row.get("domain") == "domain_independent"
            and row.get("domain_labels") == []
            and row.get("domain_claims") == []
            and row.get("output_introduces_unsupplied_facts") is False
        ):
            failures["segregation_mismatch"] += 1
        if row.get("label_evidence_sha256") != probe_label_evidence_sha256(row):
            failures["label_hash_mismatch"] += 1
    return {
        "format": FORMAT,
        "status": "PASS_TARGETED_CATALOG_AUDIT" if not failures else "FAIL_TARGETED_CATALOG_AUDIT",
        "candidate": {"path": candidate_path.as_posix(), "sha256": _sha256_file(candidate_path), "records": len(probes), "unique_prompts": len(set(candidate_hashes))},
        "prior_catalogs": [{"path": path.as_posix(), "sha256": _sha256_file(path)} for path in prior_paths],
        "prior_records": prior_records,
        "exact_prior_prompt_overlap": overlap,
        "counts": dict(sorted(counts.items())),
        "failures": dict(sorted(failures.items())),
        "teacher_model_loaded": False,
        "validation_or_final_outputs_read": False,
        "neural_training_performed": False,
        "claim_boundary": "Catalog construction audit only; no teacher extraction, model quality, or Phase 3 certification is claimed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prior", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("targeted catalog audit output exists")
    result = audit(Path(args.candidate).resolve(), [Path(value).resolve() for value in args.prior])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
