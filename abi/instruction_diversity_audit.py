"""Compare a successor natural catalog with a frozen ABI training artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .hf_extraction import load_probe_catalog
from .layercake_host_v3 import load_english_training_rows
from .layercake_host import _canonical_json_bytes, _sha256_file


AUDIT_FORMAT = "abi-instruction-diversity-audit/1"
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


class InstructionDiversityAuditError(RuntimeError):
    """Raised when the diversity comparison cannot be reproduced."""


def _metrics(prompts: Sequence[str]) -> dict[str, Any]:
    unique_prompts = sorted(set(prompts))
    tokenized = [_TOKEN.findall(prompt.casefold()) for prompt in unique_prompts]
    unigram_counts: Counter[str] = Counter()
    bigrams: set[tuple[str, str]] = set()
    trigrams: set[tuple[str, str, str]] = set()
    lead_four: set[tuple[str, ...]] = set()
    for tokens in tokenized:
        unigram_counts.update(tokens)
        bigrams.update(zip(tokens, tokens[1:]))
        trigrams.update(zip(tokens, tokens[1:], tokens[2:]))
        lead_four.add(tuple(tokens[:4]))
    total_tokens = sum(unigram_counts.values())
    return {
        "prompt_instances": len(prompts),
        "unique_prompts": len(unique_prompts),
        "duplicate_prompt_instances": len(prompts) - len(unique_prompts),
        "unique_prompt_utf8_bytes": sum(
            len(prompt.encode("utf-8")) for prompt in unique_prompts
        ),
        "word_tokens": total_tokens,
        "unique_word_types": len(unigram_counts),
        "distinct_word_bigrams": len(bigrams),
        "distinct_word_trigrams": len(trigrams),
        "distinct_leading_four_grams": len(lead_four),
        "word_type_token_ratio": (
            len(unigram_counts) / total_tokens if total_tokens else 0.0
        ),
        "prompt_set_sha256": hashlib.sha256(
            "\n".join(unique_prompts).encode("utf-8")
        ).hexdigest(),
    }


def audit_instruction_diversity(
    *,
    baseline_bundle_path: Path,
    baseline_budget_index: int,
    candidate_catalog_path: Path,
    output_path: Path,
    minimum_unique_prompt_ratio: float,
    minimum_unique_byte_ratio: float,
    minimum_trigram_ratio: float,
    minimum_lead_four_ratio: float,
    minimum_candidate_capabilities: int,
) -> dict[str, Any]:
    if output_path.exists():
        raise InstructionDiversityAuditError(
            f"diversity evidence is immutable: {output_path}"
        )
    baseline_rows, baseline_budget, _ = load_english_training_rows(
        baseline_bundle_path, budget_index=baseline_budget_index
    )
    catalog = load_probe_catalog(candidate_catalog_path)
    candidate_probes = [
        probe for probe in catalog["probes"] if probe["split"] == "search"
    ]
    baseline = _metrics([str(row["prompt"]) for row in baseline_rows])
    candidate = _metrics([str(probe["prompt"]) for probe in candidate_probes])

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    ratios = {
        "unique_prompt_ratio": ratio(
            candidate["unique_prompts"], baseline["unique_prompts"]
        ),
        "unique_prompt_utf8_byte_ratio": ratio(
            candidate["unique_prompt_utf8_bytes"],
            baseline["unique_prompt_utf8_bytes"],
        ),
        "distinct_word_trigram_ratio": ratio(
            candidate["distinct_word_trigrams"],
            baseline["distinct_word_trigrams"],
        ),
        "distinct_leading_four_gram_ratio": ratio(
            candidate["distinct_leading_four_grams"],
            baseline["distinct_leading_four_grams"],
        ),
    }
    capability_count = len(
        {str(probe["capability"]) for probe in candidate_probes}
    )
    prompt_overlap = len(
        {str(row["prompt"]) for row in baseline_rows}
        & {str(probe["prompt"]) for probe in candidate_probes}
    )
    checks = {
        "candidate_search_prompts_are_unique": (
            candidate["duplicate_prompt_instances"] == 0
        ),
        "no_exact_prompt_overlap_with_baseline": prompt_overlap == 0,
        "minimum_candidate_capabilities": (
            capability_count >= minimum_candidate_capabilities
        ),
        "minimum_unique_prompt_ratio": (
            ratios["unique_prompt_ratio"] >= minimum_unique_prompt_ratio
        ),
        "minimum_unique_prompt_utf8_byte_ratio": (
            ratios["unique_prompt_utf8_byte_ratio"] >= minimum_unique_byte_ratio
        ),
        "minimum_distinct_word_trigram_ratio": (
            ratios["distinct_word_trigram_ratio"] >= minimum_trigram_ratio
        ),
        "minimum_distinct_leading_four_gram_ratio": (
            ratios["distinct_leading_four_gram_ratio"]
            >= minimum_lead_four_ratio
        ),
        "final_test_not_present": all(
            probe["split"] != "final_test" for probe in catalog["probes"]
        ),
    }
    evidence: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "baseline": {
            "path": str(baseline_bundle_path),
            "sha256": _sha256_file(baseline_bundle_path),
            "budget_index": baseline_budget_index,
            "budget_id": baseline_budget["budget_id"],
            "metrics": baseline,
        },
        "candidate": {
            "path": str(candidate_catalog_path),
            "sha256": _sha256_file(candidate_catalog_path),
            "catalog_id": catalog["catalog_id"],
            "search_capabilities": capability_count,
            "metrics": candidate,
        },
        "thresholds": {
            "minimum_unique_prompt_ratio": minimum_unique_prompt_ratio,
            "minimum_unique_prompt_utf8_byte_ratio": minimum_unique_byte_ratio,
            "minimum_distinct_word_trigram_ratio": minimum_trigram_ratio,
            "minimum_distinct_leading_four_gram_ratio": minimum_lead_four_ratio,
            "minimum_candidate_capabilities": minimum_candidate_capabilities,
        },
        "ratios": ratios,
        "exact_prompt_overlap": prompt_overlap,
        "checks": checks,
        "final_test_accessed": False,
        "layercake_invoked": False,
        "abi_transfer_proven": False,
        "moonshot_complete": False,
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bundle", required=True)
    parser.add_argument("--baseline-budget-index", type=int, required=True)
    parser.add_argument("--candidate-catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-unique-prompt-ratio", type=float, required=True)
    parser.add_argument("--minimum-unique-byte-ratio", type=float, required=True)
    parser.add_argument("--minimum-trigram-ratio", type=float, required=True)
    parser.add_argument("--minimum-lead-four-ratio", type=float, required=True)
    parser.add_argument("--minimum-candidate-capabilities", type=int, required=True)
    args = parser.parse_args(argv)
    evidence = audit_instruction_diversity(
        baseline_bundle_path=Path(args.baseline_bundle).resolve(),
        baseline_budget_index=args.baseline_budget_index,
        candidate_catalog_path=Path(args.candidate_catalog).resolve(),
        output_path=Path(args.output).resolve(),
        minimum_unique_prompt_ratio=args.minimum_unique_prompt_ratio,
        minimum_unique_byte_ratio=args.minimum_unique_byte_ratio,
        minimum_trigram_ratio=args.minimum_trigram_ratio,
        minimum_lead_four_ratio=args.minimum_lead_four_ratio,
        minimum_candidate_capabilities=args.minimum_candidate_capabilities,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "ratios": evidence["ratios"],
                "checks": evidence["checks"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if evidence["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
