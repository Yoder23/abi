"""Fail-closed verifier for prospective R81 and prior-catalog disjointness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.capability_compiler_phase2_common import sha256_file
from abi.hf_extraction import load_probe_catalog
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once
from experiments.role_invariant_choice_r76.verify_catalogs_v1 import _verify_rows


R81_SHA256 = "0d2b24417bf1ea925b97747d54ac7053ac7a38f517966431e8befb8f893d2c54"
PRIOR = {
    "catalogs/role_invariant_reasoning_search_r76_v1.json": "b7bf96a48d2c51fc08948ff1f92a5211c22fd24e0f83d5749a61523c22ee2387",
    "catalogs/role_invariant_reasoning_validation_r77_v1.json": "b89a473da82b66297f0cd9f7796b341f873d1bb3d1b2c7f82e94255b5b61330a",
    "catalogs/role_invariant_reasoning_validation_r80_v1.json": "5c2975520630fd6cc1b57e473c50ed89e43540fb5e810ade91fba70c9b98b894",
}


class VerificationError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable verification exists: {args.output}")
    if sha256_file(args.catalog) != R81_SHA256:
        raise VerificationError("R81 catalog hash changed")
    catalog = load_probe_catalog(args.catalog)
    checks = _verify_rows(catalog, campaign="r81", split="validation", rows=1_400)
    prompts = {str(row["prompt"]) for row in catalog["probes"]}
    stems = {stem for triple in catalog["generation"]["lexical_triples"] for stem in triple}
    prior_prompt_overlap = 0
    prior_stem_overlap = 0
    for relative, expected in PRIOR.items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise VerificationError(f"prior catalog missing or changed: {relative}")
        prior = load_probe_catalog(path)
        prior_prompt_overlap += len(prompts & {str(row["prompt"]) for row in prior["probes"]})
        prior_stems = {stem for triple in prior["generation"]["lexical_triples"] for stem in triple}
        prior_stem_overlap += len(stems & prior_stems)
    if prior_prompt_overlap or prior_stem_overlap:
        raise VerificationError("R81 overlaps R76/R77/R80 prompts or stems")
    result = {
        "format": "abi-r81-prospective-catalog-verification/1",
        "verdict": "PASS_R81_FROZEN_BEFORE_SOURCE_OR_CANDIDATE_ACCESS",
        "catalog_sha256": R81_SHA256,
        "checks": checks,
        "prior_exact_prompt_overlap": prior_prompt_overlap,
        "prior_lexical_stem_overlap": prior_stem_overlap,
        "source_accessed": False,
        "candidate_accessed": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
