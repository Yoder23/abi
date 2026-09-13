"""Fail-closed verifier for the frozen R76/R77 catalog pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from abi.capability_compiler_phase2_common import sha256_file
from abi.hf_extraction import load_probe_catalog, probe_label_evidence_sha256
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once


R76_SHA256 = "b7bf96a48d2c51fc08948ff1f92a5211c22fd24e0f83d5749a61523c22ee2387"
R77_SHA256 = "b89a473da82b66297f0cd9f7796b341f873d1bb3d1b2c7f82e94255b5b61330a"
PRIOR = {
    "catalogs/robust_choice_reasoning_search_r72_v1.json": "d179445c92a649f5ab6587c1aa71e00e7c622c326f8ea84438497542c00c2b47",
    "catalogs/conditional_choice_reasoning_validation_r75_v1.json": "e8cfc9199d05b796c7625803706a62b245b6537d2962924123fdf8d9c607f086",
}


class CatalogVerificationError(RuntimeError):
    pass


def _verify_rows(catalog: dict, *, campaign: str, split: str, rows: int) -> dict:
    probes = list(catalog["probes"])
    if len(probes) != rows or len({row["probe_id"] for row in probes}) != rows:
        raise CatalogVerificationError(f"{campaign} row identity changed")
    prompts = [str(row["prompt"]) for row in probes]
    if len(set(prompts)) != rows:
        raise CatalogVerificationError(f"{campaign} prompts are duplicated")
    families: Counter[str] = Counter()
    display: Counter[int] = Counter()
    expected_stems: Counter[str] = Counter()
    for row in probes:
        probe_id = str(row["probe_id"])
        if not probe_id.startswith(f"{campaign}-reasoning-f") or row["split"] != split:
            raise CatalogVerificationError(f"{campaign} ID or split changed")
        family = probe_id.split("-")[2]
        families[family] += 1
        candidates = row.get("conditional_candidates")
        semantic = row.get("semantic_role_codes")
        evaluator = row.get("evaluator")
        expected = evaluator.get("value") if isinstance(evaluator, dict) else None
        if (
            not isinstance(candidates, list)
            or len(candidates) != 3
            or len(set(candidates)) != 3
            or not isinstance(semantic, list)
            or len(semantic) != 3
            or set(candidates) != set(semantic)
            or expected != semantic[2]
            or evaluator.get("kind") != "exact"
            or evaluator.get("case_sensitive") is not True
            or row.get("destination_display_index") != candidates.index(expected)
            or any(code not in row["prompt"] for code in candidates)
            or row.get("label_evidence_sha256") != probe_label_evidence_sha256(row)
            or row.get("domain_labels") != []
            or row.get("domain_claims") != []
            or row.get("output_introduces_unsupplied_facts") is not False
        ):
            raise CatalogVerificationError(f"{campaign} semantic row contract changed")
        display[int(row["destination_display_index"])] += 1
        expected_stems["".join(character for character in expected if character.isalpha())] += 1
    if len(families) != 7 or max(families.values()) != min(families.values()):
        raise CatalogVerificationError(f"{campaign} family balance changed")
    if max(display.values()) - min(display.values()) > max(1, rows // 100):
        raise CatalogVerificationError(f"{campaign} display positions are not balanced")
    return {
        "rows": rows,
        "unique_prompts": len(set(prompts)),
        "prompt_set_sha256": hashlib.sha256("\n".join(sorted(prompts)).encode()).hexdigest(),
        "families": dict(sorted(families.items())),
        "destination_display_positions": dict(sorted(display.items())),
        "expected_stems": dict(sorted(expected_stems.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r76", required=True, type=Path)
    parser.add_argument("--r77", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable verification exists: {args.output}")
    if sha256_file(args.r76) != R76_SHA256 or sha256_file(args.r77) != R77_SHA256:
        raise CatalogVerificationError("R76/R77 catalog hash changed")
    r76, r77 = load_probe_catalog(args.r76), load_probe_catalog(args.r77)
    checks = {
        "r76": _verify_rows(r76, campaign="r76", split="search", rows=8_400),
        "r77": _verify_rows(r77, campaign="r77", split="validation", rows=1_400),
    }
    prompt_sets = [set(row["prompt"] for row in value["probes"]) for value in (r76, r77)]
    if prompt_sets[0] & prompt_sets[1]:
        raise CatalogVerificationError("R76/R77 prompts overlap")
    prior_prompts: set[str] = set()
    for relative, expected_hash in PRIOR.items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise CatalogVerificationError(f"prior catalog missing or changed: {relative}")
        prior_prompts.update(str(row["prompt"]) for row in load_probe_catalog(path)["probes"])
    if prior_prompts & prompt_sets[0] or prior_prompts & prompt_sets[1]:
        raise CatalogVerificationError("new catalogs overlap prior conditional-choice prompts")
    train_stems = {stem for triple in r76["generation"]["lexical_triples"] for stem in triple}
    validation_stems = {stem for triple in r77["generation"]["lexical_triples"] for stem in triple}
    if train_stems & validation_stems:
        raise CatalogVerificationError("R77 lexical stems leak into R76")
    result = {
        "format": "abi-r76-r77-role-invariant-catalog-verification/1",
        "verdict": "PASS_CATALOGS_FROZEN_BEFORE_SOURCE_OR_CANDIDATE_ACCESS",
        "catalog_sha256": {"r76": R76_SHA256, "r77": R77_SHA256},
        "checks": checks,
        "r76_r77_exact_prompt_overlap": 0,
        "prior_exact_prompt_overlap": 0,
        "lexical_stem_overlap": 0,
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
