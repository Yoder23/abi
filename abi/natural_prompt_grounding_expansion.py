"""Build the V76 natural prompt-grounding expansion catalogs.

The input is the frozen V3 human-prompt-surface catalog.  This successor does
not import any corpus answer.  It deterministically assigns the human prompt
surfaces to disjoint search and validation splits, adds one bounded
knowledge-minimizing response contract, and emits a small preflight catalog
from the exact full-catalog search population.

Grammar is intentionally excluded.  The frozen V71 artifact already supplies
grammar from a separately qualified counterbalanced source method, while the
natural-grammar source branches failed.  Reopening that branch here would mix
an unrelated source weakness into the measured prompt-breadth experiment.
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


CATALOG_FORMAT = "abi-natural-prompt-grounding-expansion-catalog/1"
FULL_CATALOG_ID = "abi-natural-prompt-grounding-expansion-search-validation-v76"
PREFLIGHT_CATALOG_ID = f"{FULL_CATALOG_ID}-gpu-preflight"
DEFAULT_SEED = 89_824
DEFAULT_VALIDATION_PER_CAPABILITY = 80
DEFAULT_PREFLIGHT_PER_CAPABILITY = 20
EXCLUDED_CAPABILITIES = frozenset({"grammar"})
CONCISION_CONTRACT = (
    "Keep the complete answer concise and under one hundred twenty words. "
    "Do not mention these acquisition instructions."
)


class NaturalPromptGroundingExpansionError(RuntimeError):
    """Raised when the immutable natural expansion cannot be derived."""


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise NaturalPromptGroundingExpansionError(
            f"catalog is immutable: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _rank(*, seed: int, purpose: str, capability: str, probe_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:{purpose}:{capability}:{probe_id}".encode("utf-8")
    ).hexdigest()


def _response_evaluator(prompt: str) -> dict[str, Any]:
    return {
        "kind": "all_of",
        "prompt_contract_sha256": prompt_contract_sha256(prompt),
        "rules": [
            {"kind": "nonempty", "minimum_characters": 12},
            {"kind": "maximum_characters", "value": 1_500},
        ],
    }


def build_expansion_catalogs(
    *,
    parent_catalog: Mapping[str, Any],
    parent_path: Path,
    validation_per_capability: int = DEFAULT_VALIDATION_PER_CAPABILITY,
    preflight_per_capability: int = DEFAULT_PREFLIGHT_PER_CAPABILITY,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the exact V76 full and preflight catalogs."""

    if validation_per_capability < 64:
        raise NaturalPromptGroundingExpansionError(
            "validation_per_capability must be at least 64"
        )
    if preflight_per_capability < 2:
        raise NaturalPromptGroundingExpansionError(
            "preflight_per_capability must be at least 2"
        )
    probes = parent_catalog.get("probes")
    if not isinstance(probes, list) or not probes:
        raise NaturalPromptGroundingExpansionError("parent catalog is empty")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for probe in probes:
        capability = str(probe["capability"])
        if capability in EXCLUDED_CAPABILITIES:
            continue
        if probe.get("split") != "search":
            raise NaturalPromptGroundingExpansionError(
                "parent natural catalog must remain search-only"
            )
        grouped.setdefault(capability, []).append(probe)
    if len(grouped) != 9:
        raise NaturalPromptGroundingExpansionError(
            f"expected nine natural capabilities after exclusions, got {len(grouped)}"
        )

    full_probes: list[dict[str, Any]] = []
    split_counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[str] = Counter()
    for capability_index, capability in enumerate(sorted(grouped)):
        ranked = sorted(
            grouped[capability],
            key=lambda row: _rank(
                seed=seed,
                purpose="validation-split",
                capability=capability,
                probe_id=str(row["probe_id"]),
            ),
        )
        if len(ranked) < validation_per_capability + preflight_per_capability:
            raise NaturalPromptGroundingExpansionError(
                f"{capability} lacks enough disjoint natural prompts"
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
            prompt = f"{parent_probe['prompt']}\n\n{CONCISION_CONTRACT}"
            probe = {
                **dict(parent_probe),
                "probe_id": (
                    f"natural-expansion-{capability}-{split}-{ordinal:05d}-v76"
                ),
                "split": split,
                "prompt": prompt,
                "max_new_tokens": 256,
                "temperature": 0,
                "seed": seed + capability_index * 100_000 + ordinal,
                "evaluator": _response_evaluator(prompt),
                "parent_catalog_id": parent_catalog["catalog_id"],
                "parent_probe_id": parent_probe["probe_id"],
                "parent_probe_sha256": hashlib.sha256(
                    _canonical_json_bytes(parent_probe)
                ).hexdigest(),
                "corpus_assistant_messages_imported": 0,
                "corpus_reference_answers_imported": 0,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            full_probes.append(probe)
            split_counts[(capability, split)] += 1
            source_counts[str(probe.get("source_prompt_corpus", "unknown"))] += 1

    prompt_hashes = [
        prompt_contract_sha256(str(probe["prompt"])) for probe in full_probes
    ]
    if len(set(prompt_hashes)) != len(full_probes):
        raise NaturalPromptGroundingExpansionError(
            "natural expansion contains duplicate prompts"
        )
    full_catalog: dict[str, Any] = {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": FULL_CATALOG_ID,
        "catalog_contract_schema": CATALOG_FORMAT,
        "status": "PREREGISTERED_NATURAL_SEARCH_VALIDATION_EXPANSION",
        "claim_boundary": (
            "This catalog contributes only frozen human-written prompt surfaces "
            "and zero corpus answers. Deterministic filtering and finite lexical "
            "audits do not prove literal absence of latent world knowledge, source "
            "quality, LayerCake fluency, or ABI transfer."
        ),
        "parent_catalog": {
            "path": parent_path.as_posix(),
            "sha256": _sha256_file(parent_path),
            "catalog_id": parent_catalog["catalog_id"],
            "probes": len(probes),
        },
        "generation": {
            "generator": "abi.natural_prompt_grounding_expansion",
            "seed": seed,
            "validation_per_capability": validation_per_capability,
            "preflight_per_capability": preflight_per_capability,
            "included_capabilities": sorted(grouped),
            "excluded_capabilities": sorted(EXCLUDED_CAPABILITIES),
            "total_probes": len(full_probes),
            "search_probes": sum(
                count
                for (capability, split), count in split_counts.items()
                if split == "search"
            ),
            "validation_probes": sum(
                count
                for (capability, split), count in split_counts.items()
                if split == "validation"
            ),
            "capability_split_counts": {
                capability: {
                    split: split_counts[(capability, split)]
                    for split in ("search", "validation")
                }
                for capability in sorted(grouped)
            },
            "source_counts": dict(sorted(source_counts.items())),
            "prompt_sha256_set_sha256": hashlib.sha256(
                "\n".join(sorted(prompt_hashes)).encode("ascii")
            ).hexdigest(),
            "corpus_assistant_messages_imported": 0,
            "corpus_reference_answers_imported": 0,
            "specialist_domain_rows_admitted": 0,
            "final_test_probes": 0,
        },
        "source_prompt_corpora": parent_catalog.get("source_prompt_corpora", []),
        "core_exclusion_markers": parent_catalog.get(
            "core_exclusion_markers", []
        ),
        "probes": full_probes,
    }

    preflight_probes: list[dict[str, Any]] = []
    for capability in sorted(grouped):
        candidates = sorted(
            [
                probe
                for probe in full_probes
                if probe["capability"] == capability and probe["split"] == "search"
            ],
            key=lambda probe: _rank(
                seed=seed,
                purpose="gpu-preflight",
                capability=capability,
                probe_id=str(probe["probe_id"]),
            ),
        )[:preflight_per_capability]
        for row in candidates:
            probe = dict(row)
            probe["probe_id"] = str(row["probe_id"]).replace(
                "-v76", "-gpu-preflight-v76"
            )
            probe["parent_full_probe_id"] = row["probe_id"]
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            preflight_probes.append(probe)
    preflight_catalog: dict[str, Any] = {
        **{key: value for key, value in full_catalog.items() if key != "probes"},
        "catalog_id": PREFLIGHT_CATALOG_ID,
        "status": "PREREGISTERED_GPU_PREFLIGHT_SUBSET",
        "parent_full_catalog_sha256": hashlib.sha256(
            _canonical_json_bytes(full_catalog)
        ).hexdigest(),
        "generation": {
            **dict(full_catalog["generation"]),
            "total_probes": len(preflight_probes),
            "search_probes": len(preflight_probes),
            "validation_probes": 0,
            "preflight_per_capability": preflight_per_capability,
            "capability_split_counts": {
                capability: {"search": preflight_per_capability, "validation": 0}
                for capability in sorted(grouped)
            },
            "final_test_probes": 0,
        },
        "probes": preflight_probes,
    }
    return full_catalog, preflight_catalog


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-output", required=True)
    parser.add_argument(
        "--validation-per-capability",
        type=int,
        default=DEFAULT_VALIDATION_PER_CAPABILITY,
    )
    parser.add_argument(
        "--preflight-per-capability",
        type=int,
        default=DEFAULT_PREFLIGHT_PER_CAPABILITY,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    parent_path = Path(args.parent_catalog).resolve()
    output_path = Path(args.output).resolve()
    preflight_path = Path(args.preflight_output).resolve()
    parent = load_probe_catalog(parent_path)
    full, preflight = build_expansion_catalogs(
        parent_catalog=parent,
        parent_path=parent_path,
        validation_per_capability=args.validation_per_capability,
        preflight_per_capability=args.preflight_per_capability,
        seed=args.seed,
    )
    _write_immutable(output_path, full)
    _write_immutable(preflight_path, preflight)
    load_probe_catalog(output_path)
    load_probe_catalog(preflight_path)
    result = {
        "full": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
            "probes": len(full["probes"]),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": _sha256_file(preflight_path),
            "probes": len(preflight["probes"]),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
