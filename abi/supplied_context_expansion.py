"""Build the V77 supplied-context-only natural instruction candidates.

The frozen parent was produced from Alpaca prompt surfaces while discarding all
corpus answers.  This successor retains only capabilities with enough rows for
at least 100 search and 40 validation candidates, assigns those splits before
any prompt-domain judgment, and adds a bounded response-length instruction.
The result is still a *candidate* catalog: an independent open-weight prompt
classifier must approve every row before source answer generation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
    prompt_contract_sha256,
)
from .layercake_host import _canonical_json_bytes, _sha256_file


CATALOG_FORMAT = "abi-supplied-context-natural-expansion-catalog/1"
CATALOG_ID = "abi-supplied-context-natural-candidates-search-validation-v77"
DEFAULT_SEED = 99_824
DEFAULT_VALIDATION_PER_CAPABILITY = 40
INCLUDED_CAPABILITIES = frozenset(
    {
        "conversation",
        "format_control",
        "instruction_following",
        "prompt_grounding",
        "rewriting",
        "summarization",
        "tone_control",
    }
)
OUTPUT_CONTRACT = (
    "Keep the complete answer under eighty words. Use only the supplied text; "
    "do not add facts, names, numbers, products, procedures, or claims that are "
    "not already present there."
)


class SuppliedContextExpansionError(RuntimeError):
    """Raised when the supplied-context candidate set cannot be derived."""


def _rank(*, seed: int, capability: str, probe_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:split:{capability}:{probe_id}".encode("utf-8")
    ).hexdigest()


def build_supplied_context_catalog(
    *,
    parent_catalog: Mapping[str, Any],
    parent_path: Path,
    validation_per_capability: int = DEFAULT_VALIDATION_PER_CAPABILITY,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if validation_per_capability < 40:
        raise SuppliedContextExpansionError(
            "validation_per_capability must be at least 40"
        )
    grouped: dict[str, list[Mapping[str, Any]]] = {
        capability: [] for capability in INCLUDED_CAPABILITIES
    }
    for probe in parent_catalog.get("probes", []):
        capability = str(probe["capability"])
        if capability not in INCLUDED_CAPABILITIES:
            continue
        prompt = str(probe["prompt"])
        if "<supplied_text>" not in prompt or "</supplied_text>" not in prompt:
            raise SuppliedContextExpansionError(
                f"candidate lacks explicit supplied text: {probe['probe_id']}"
            )
        if probe.get("split") != "search":
            raise SuppliedContextExpansionError(
                "parent natural-instruction catalog must remain search-only"
            )
        grouped[capability].append(probe)
    if set(grouped) != set(INCLUDED_CAPABILITIES):
        raise SuppliedContextExpansionError("included capability set drifted")

    probes: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    parent_hashes: list[str] = []
    for capability_index, capability in enumerate(sorted(grouped)):
        ranked = sorted(
            grouped[capability],
            key=lambda row: _rank(
                seed=seed,
                capability=capability,
                probe_id=str(row["probe_id"]),
            ),
        )
        if len(ranked) - validation_per_capability < 100:
            raise SuppliedContextExpansionError(
                f"{capability} has fewer than 100 disjoint search candidates"
            )
        validation_ids = {
            str(row["probe_id"])
            for row in ranked[:validation_per_capability]
        }
        ordinals: Counter[str] = Counter()
        for parent_probe in ranked:
            split = (
                "validation"
                if str(parent_probe["probe_id"]) in validation_ids
                else "search"
            )
            ordinal = ordinals[split]
            ordinals[split] += 1
            prompt = f"{parent_probe['prompt']}\n\n{OUTPUT_CONTRACT}"
            evaluator = {
                "kind": "all_of",
                "prompt_contract_sha256": prompt_contract_sha256(prompt),
                "rules": [
                    {"kind": "nonempty", "minimum_characters": 8},
                    {"kind": "maximum_characters", "value": 800},
                ],
            }
            probe = {
                **dict(parent_probe),
                "probe_id": (
                    f"supplied-natural-{capability}-{split}-{ordinal:05d}-v77"
                ),
                "split": split,
                "prompt": prompt,
                "max_new_tokens": 192,
                "temperature": 0,
                "seed": seed + capability_index * 100_000 + ordinal,
                "evaluator": evaluator,
                "parent_catalog_id": parent_catalog["catalog_id"],
                "parent_probe_id": parent_probe["probe_id"],
                "parent_probe_sha256": hashlib.sha256(
                    _canonical_json_bytes(parent_probe)
                ).hexdigest(),
                "prompt_domain_qualification_required": True,
                "corpus_reference_answers_imported": 0,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)
            counts[(capability, split)] += 1
            parent_hashes.append(str(parent_probe["natural_prompt_sha256"]))

    prompt_hashes = [prompt_contract_sha256(str(row["prompt"])) for row in probes]
    if len(set(prompt_hashes)) != len(probes):
        raise SuppliedContextExpansionError("candidate prompts are not unique")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": CATALOG_ID,
        "catalog_contract_schema": CATALOG_FORMAT,
        "status": "PREREGISTERED_CANDIDATES_AWAITING_PROMPT_DOMAIN_QUALIFICATION",
        "claim_boundary": (
            "Rows contain explicit supplied text and no corpus answers, but are "
            "not source-training material until an independent prompt-domain "
            "qualification certifies that the exact request can be completed "
            "without outside or specialist knowledge."
        ),
        "parent_catalog": {
            "path": parent_path.as_posix(),
            "sha256": _sha256_file(parent_path),
            "catalog_id": parent_catalog["catalog_id"],
            "probes": len(parent_catalog["probes"]),
        },
        "generation": {
            "generator": "abi.supplied_context_expansion",
            "seed": seed,
            "included_capabilities": sorted(INCLUDED_CAPABILITIES),
            "excluded_capabilities": sorted(
                set(parent_catalog["generation"]["capability_counts"])
                - set(INCLUDED_CAPABILITIES)
            ),
            "validation_per_capability": validation_per_capability,
            "capability_split_counts": {
                capability: {
                    split: counts[(capability, split)]
                    for split in ("search", "validation")
                }
                for capability in sorted(INCLUDED_CAPABILITIES)
            },
            "total_probes": len(probes),
            "search_probes": sum(
                count for (capability, split), count in counts.items()
                if split == "search"
            ),
            "validation_probes": sum(
                count for (capability, split), count in counts.items()
                if split == "validation"
            ),
            "prompt_sha256_set_sha256": hashlib.sha256(
                "\n".join(sorted(prompt_hashes)).encode("ascii")
            ).hexdigest(),
            "parent_natural_prompt_sha256_set_sha256": hashlib.sha256(
                "\n".join(sorted(parent_hashes)).encode("ascii")
            ).hexdigest(),
            "corpus_reference_answers_imported": 0,
            "specialist_domain_rows_claimed_before_independent_qualification": None,
            "final_test_probes": 0,
        },
        "source_prompt_corpus": parent_catalog.get("source_prompt_corpus"),
        "core_exclusion_markers": parent_catalog.get("core_exclusion_markers", []),
        "probes": probes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validation-per-capability",
        type=int,
        default=DEFAULT_VALIDATION_PER_CAPABILITY,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    parent_path = Path(args.parent_catalog).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        parser.error(f"catalog is immutable: {output_path}")
    parent = load_probe_catalog(parent_path)
    catalog = build_supplied_context_catalog(
        parent_catalog=parent,
        parent_path=parent_path,
        validation_per_capability=args.validation_per_capability,
        seed=args.seed,
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
                "probes": len(catalog["probes"]),
                "generation": catalog["generation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
