"""Audit a broad natural acquisition catalog without reading corpus answers."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Sequence

from .capability_segregation import LINGUISTIC_FORM
from .hf_extraction import load_probe_catalog
from .natural_conversation_catalog import _safe_prompt_surface


FORMAT = "abi-broad-natural-catalog-audit/1"
_POLICY_SUFFIX = (
    "\n\nRespond in English. This is English-form acquisition: follow the "
    "request, but do not introduce outside factual claims, named entities, "
    "numerical facts, or specialist knowledge. If the request depends on "
    "missing or specialist information, state that it is not supplied or ask "
    "one concise clarification question."
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def audit_catalog(path: Path) -> dict[str, Any]:
    catalog = load_probe_catalog(path)
    probes = catalog["probes"]
    failures: Counter[str] = Counter()
    probe_ids: set[str] = set()
    natural_hashes: set[str] = set()
    prompts: set[str] = set()
    source_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    lengths: list[int] = []
    word_trigrams: set[tuple[str, str, str]] = set()
    total_word_trigrams = 0
    for probe in probes:
        probe_id = str(probe["probe_id"])
        prompt = str(probe["prompt"])
        natural_hash = str(probe["natural_prompt_sha256"])
        if probe_id in probe_ids:
            failures["duplicate_probe_id"] += 1
        probe_ids.add(probe_id)
        if prompt in prompts:
            failures["duplicate_teacher_prompt"] += 1
        prompts.add(prompt)
        if natural_hash in natural_hashes:
            failures["duplicate_natural_prompt_hash"] += 1
        natural_hashes.add(natural_hash)
        if not prompt.endswith(_POLICY_SUFFIX):
            failures["policy_suffix_mismatch"] += 1
            continue
        surface = prompt[: -len(_POLICY_SUFFIX)]
        safe = _safe_prompt_surface(surface)
        if safe is None:
            failures["unsafe_surface"] += 1
        else:
            normalized, capability = safe
            if capability != probe["capability"]:
                failures["capability_drift"] += 1
            if (
                hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                != natural_hash
            ):
                failures["surface_hash_mismatch"] += 1
        if probe["split"] != "search":
            failures["non_search_probe"] += 1
        if probe["destination_scope"] != "english_core":
            failures["non_core_destination"] += 1
        if probe["domain"] != "domain_independent":
            failures["non_independent_domain"] += 1
        if probe["knowledge_class"] != LINGUISTIC_FORM:
            failures["non_linguistic_knowledge_class"] += 1
        if probe["domain_labels"] or probe["domain_claims"]:
            failures["domain_metadata_present"] += 1
        if probe["output_introduces_unsupplied_facts"] is not False:
            failures["unsupplied_facts_allowed"] += 1
        if probe["corpus_assistant_messages_imported"] != 0:
            failures["assistant_messages_imported"] += 1
        if probe["corpus_reference_answers_imported"] != 0:
            failures["reference_answers_imported"] += 1
        if probe["temperature"] != 0:
            failures["nondeterministic_temperature"] += 1
        source_counts[str(probe["source_prompt_corpus"])] += 1
        capability_counts[str(probe["capability"])] += 1
        lengths.append(len(surface))
        words = [word.casefold() for word in _WORD.findall(surface)]
        trigrams = list(zip(words, words[1:], words[2:]))
        total_word_trigrams += len(trigrams)
        word_trigrams.update(trigrams)

    minimum_capability_count = min(capability_counts.values(), default=0)
    if minimum_capability_count < 100:
        failures["capability_below_preregistered_minimum"] += 1
    source_manifest_counts = Counter(
        {
            key: int(value)
            for key, value in catalog["generation"]["source_counts"].items()
        }
    )
    if source_counts != source_manifest_counts:
        failures["source_count_manifest_mismatch"] += 1
    declared_capability_counts = Counter(
        {
            key: int(value)
            for key, value in catalog["generation"][
                "capability_counts"
            ].items()
        }
    )
    if capability_counts != declared_capability_counts:
        failures["capability_count_manifest_mismatch"] += 1
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS" if not failures else "FAIL",
        "catalog": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "catalog_id": catalog["catalog_id"],
        },
        "observations": {
            "probes": len(probes),
            "unique_probe_ids": len(probe_ids),
            "unique_teacher_prompts": len(prompts),
            "unique_natural_prompt_hashes": len(natural_hashes),
            "capability_counts": dict(sorted(capability_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "minimum_capability_count": minimum_capability_count,
            "natural_prompt_characters": {
                "minimum": min(lengths, default=0),
                "median": median(lengths) if lengths else 0,
                "p95": _quantile(lengths, 0.95) if lengths else 0,
                "maximum": max(lengths, default=0),
            },
            "word_trigrams_total": total_word_trigrams,
            "word_trigrams_unique": len(word_trigrams),
            "word_trigram_unique_ratio": (
                len(word_trigrams) / total_word_trigrams
                if total_word_trigrams
                else 0
            ),
            "corpus_assistant_messages_imported": 0,
            "corpus_reference_answers_imported": 0,
            "domain_labels_present": 0,
            "domain_claims_present": 0,
            "non_search_probes": 0,
        },
        "failures": dict(sorted(failures.items())),
        "claim_boundary": (
            "This machine audit proves the declared deterministic catalog "
            "invariants. It does not prove teacher-output quality, zero latent "
            "world knowledge, or LayerCake generalization."
        ),
    }
    evidence_payload = dict(result)
    result["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            evidence_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"audit output is immutable: {output}")
    result = audit_catalog(Path(args.catalog).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
