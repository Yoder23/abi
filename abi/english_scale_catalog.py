"""Build the nested, domain-clean English scale acquisition catalog.

The first corpus-grounded campaign repeated fourteen tasks over only 400
search contexts.  This successor uses one long-form transformation per
context so that teacher-token growth also means growth in distinct linguistic
contexts.  Every prompt is grounded in supplied fiction, carries no domain
label or claim, and is routed only to the English core.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .broad_english_catalog import (
    BroadEnglishCatalogError,
    DEFAULT_CORE_EXCLUSION_MARKERS,
    _arrow_texts,
    _corpus_manifest,
    _evaluator,
    _is_safe_story,
    _marker_present,
    _normalized,
    _sha256_file,
    _story_excerpt,
)
from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
)


CATALOG_SCHEMA = "abi-english-distinct-context-scale-catalog/1"
DEFAULT_SEED = 19_824
DEFAULT_SEARCH_CONTEXTS = 40_000
DEFAULT_VALIDATION_CONTEXTS = 1_000
DEFAULT_FINAL_CONTEXTS = 1_000
DEFAULT_MAXIMUM_CONTEXT_CHARACTERS = 720
SCALE_CAPABILITIES = (
    "rewriting",
    "coherence",
    "tone_control",
    "cake_output_realization",
)


def _unique_context_sample(
    texts: Iterable[str],
    *,
    count: int,
    seed: int,
    exclusion_markers: Sequence[str],
    excluded_context_sha256: frozenset[bytes] = frozenset(),
    maximum_context_characters: int = DEFAULT_MAXIMUM_CONTEXT_CHARACTERS,
) -> list[str]:
    """Take a stable min-hash sample of exact, unique bounded contexts."""

    if count <= 0:
        raise BroadEnglishCatalogError("scale sample count must be positive")
    heap: list[tuple[int, str]] = []
    observed_contexts: set[bytes] = set()
    prefix = f"{seed}:".encode("utf-8")
    for text in texts:
        if not isinstance(text, str) or not _is_safe_story(
            text, exclusion_markers=exclusion_markers
        ):
            continue
        normalized = _normalized(text)
        try:
            excerpt = _story_excerpt(
                normalized, maximum_characters=maximum_context_characters
            )
        except BroadEnglishCatalogError:
            # A source row can contain enough sentence delimiters overall but
            # still have an oversized first sentence that leaves fewer than
            # three sentences inside the bounded acquisition context.
            continue
        context_sha = hashlib.sha256(excerpt.encode("utf-8")).digest()
        if (
            context_sha in excluded_context_sha256
            or context_sha in observed_contexts
        ):
            continue
        observed_contexts.add(context_sha)
        score = int.from_bytes(
            hashlib.sha256(prefix + context_sha).digest(),
            "big",
        )
        item = (-score, normalized)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) != count:
        raise BroadEnglishCatalogError(
            f"only {len(heap)} unique safe contexts available for {count} rows"
        )
    return [
        text
        for _, text in sorted(
            [(-negative_score, text) for negative_score, text in heap]
        )
    ]


def _context_hashes(
    stories: Sequence[str],
    *,
    maximum_context_characters: int = DEFAULT_MAXIMUM_CONTEXT_CHARACTERS,
) -> frozenset[bytes]:
    return frozenset(
        hashlib.sha256(
            _story_excerpt(
                story,
                maximum_characters=maximum_context_characters,
            ).encode("utf-8")
        ).digest()
        for story in stories
    )


def _sentences(excerpt: str) -> list[str]:
    values = [
        value.strip()
        for value in excerpt.replace("!", ".").replace("?", ".").split(".")
        if value.strip()
    ]
    if len(values) < 4:
        raise BroadEnglishCatalogError(
            "scale context cannot supply four complete events"
        )
    return [f"{value}." for value in values]


def _scale_prompt(capability: str, excerpt: str) -> tuple[str, int]:
    context = f"<fictional_context>\n{excerpt}\n</fictional_context>"
    events = _sentences(excerpt)
    if capability == "rewriting":
        return (
            f"{context}\nRewrite the supplied fictional passage in polished, "
            "natural English for an adult reader. Use five to seven complete "
            "sentences, preserve the events and relationships, and introduce "
            "no outside information.",
            192,
        )
    if capability == "coherence":
        reordered = events[2:] + events[:2]
        numbered = "\n".join(
            f"{index + 1}. {event}" for index, event in enumerate(reordered)
        )
        return (
            "The following fictional events were rotated out of order. "
            "Restore a coherent narrative in five to seven complete, natural "
            "English sentences. Use every supplied event and add no outside "
            f"information.\n<events>\n{numbered}\n</events>",
            192,
        )
    if capability == "tone_control":
        return (
            f"{context}\nRetell only the supplied fictional events in a calm, "
            "reassuring tone. Write five to seven connected, complete English "
            "sentences and add no outside information.",
            192,
        )
    if capability == "cake_output_realization":
        fields = "\n".join(
            f"event_{index + 1}={event}"
            for index, event in enumerate(events)
        )
        return (
            "Realize these supplied fictional fields as one fluent narrative "
            "of five to seven complete English sentences. Preserve every "
            "field, connect the events naturally, and add no new information."
            f"\n<fields>\n{fields}\n</fields>",
            192,
        )
    raise BroadEnglishCatalogError(
        f"unsupported scale capability: {capability}"
    )


def build_catalog(
    *,
    search_stories: Sequence[str],
    validation_stories: Sequence[str],
    final_stories: Sequence[str],
    corpus_manifest: dict[str, Any],
    seed: int = DEFAULT_SEED,
    maximum_context_characters: int = DEFAULT_MAXIMUM_CONTEXT_CHARACTERS,
    exclusion_markers: Sequence[str] = DEFAULT_CORE_EXCLUSION_MARKERS,
) -> dict[str, Any]:
    splits = {
        "search": list(search_stories),
        "validation": list(validation_stories),
        "final_test": list(final_stories),
    }
    normalized = {
        split: {_normalized(value) for value in stories}
        for split, stories in splits.items()
    }
    if any(not values for values in normalized.values()):
        raise BroadEnglishCatalogError("every scale split must be nonempty")
    split_names = list(normalized)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if normalized[left] & normalized[right]:
                raise BroadEnglishCatalogError(
                    f"scale story overlap between {left} and {right}"
                )

    probes: list[dict[str, Any]] = []
    context_hashes: dict[str, list[str]] = {}
    capability_counts: dict[str, dict[str, int]] = {}
    for split_index, (split, stories) in enumerate(splits.items()):
        split_hashes: list[str] = []
        split_counts = {capability: 0 for capability in SCALE_CAPABILITIES}
        for story_index, story in enumerate(stories):
            excerpt = _story_excerpt(
                story, maximum_characters=maximum_context_characters
            )
            excerpt_sha = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
            split_hashes.append(excerpt_sha)
            capability = SCALE_CAPABILITIES[story_index % len(SCALE_CAPABILITIES)]
            split_counts[capability] += 1
            prompt, maximum = _scale_prompt(capability, excerpt)
            if any(
                _marker_present(prompt, marker)
                for marker in exclusion_markers
            ):
                raise BroadEnglishCatalogError(
                    "core exclusion marker crossed scale prompt construction"
                )
            probe: dict[str, Any] = {
                "probe_id": (
                    f"english-scale-{split}-{story_index:05d}-"
                    f"{capability}-v1"
                ),
                "destination_scope": "english_core",
                "capability": capability,
                "domain": "domain_independent",
                "split": split,
                "prompt": prompt,
                "max_new_tokens": maximum,
                "temperature": 0,
                "seed": seed + split_index * 1_000_000 + story_index,
                "evaluator": _evaluator(
                    capability,
                    exclusion_markers=exclusion_markers,
                ),
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": "supplied_non_domain_context",
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
                "raw_context_sha256": excerpt_sha,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)
        context_hashes[split] = sorted(split_hashes)
        capability_counts[split] = split_counts

    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-english-distinct-context-scale-v1",
        "catalog_contract_schema": CATALOG_SCHEMA,
        "status": "PREREGISTERED_NESTED_ENGLISH_SCALE_ACQUISITION",
        "claim_boundary": (
            "This catalog measures whether more distinct, domain-clean "
            "teacher sequences can teach the fixed fast LayerCake graph. It "
            "does not itself establish fluency, teacher equivalence, a global "
            "minimum payload, or absolute absence of latent world knowledge."
        ),
        "generation": {
            "generator": "abi.english_scale_catalog",
            "seed": seed,
            "capabilities": list(SCALE_CAPABILITIES),
            "story_counts": {
                split: len(stories) for split, stories in splits.items()
            },
            "probe_counts": {
                split: len(stories) for split, stories in splits.items()
            },
            "capability_counts": capability_counts,
            "maximum_context_characters": maximum_context_characters,
            "one_probe_per_distinct_context": True,
            "closed_book_fact_prompts": 0,
            "specialist_domain_prompts": 0,
            "final_test_used_for_selection": False,
            "context_sha256_by_split": context_hashes,
        },
        "core_exclusion_markers": list(exclusion_markers),
        "raw_corpus": corpus_manifest,
        "probes": probes,
    }


def build_catalog_from_arrow(
    *,
    train_path: Path,
    validation_path: Path,
    dataset_info_path: Path | None,
    search_context_count: int,
    validation_context_count: int,
    final_context_count: int,
    seed: int,
) -> dict[str, Any]:
    search = _unique_context_sample(
        _arrow_texts(train_path),
        count=search_context_count,
        seed=seed,
        exclusion_markers=DEFAULT_CORE_EXCLUSION_MARKERS,
    )
    held_out = _unique_context_sample(
        _arrow_texts(validation_path),
        count=validation_context_count + final_context_count,
        seed=seed + 1,
        exclusion_markers=DEFAULT_CORE_EXCLUSION_MARKERS,
        excluded_context_sha256=_context_hashes(search),
    )
    return build_catalog(
        search_stories=search,
        validation_stories=held_out[:validation_context_count],
        final_stories=held_out[validation_context_count:],
        corpus_manifest=_corpus_manifest(
            train_path=train_path,
            validation_path=validation_path,
            dataset_info_path=dataset_info_path,
        ),
        seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-arrow", required=True)
    parser.add_argument("--validation-arrow", required=True)
    parser.add_argument("--dataset-info")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--search-contexts", type=int, default=DEFAULT_SEARCH_CONTEXTS
    )
    parser.add_argument(
        "--validation-contexts",
        type=int,
        default=DEFAULT_VALIDATION_CONTEXTS,
    )
    parser.add_argument(
        "--final-contexts", type=int, default=DEFAULT_FINAL_CONTEXTS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    catalog = build_catalog_from_arrow(
        train_path=Path(args.train_arrow).resolve(),
        validation_path=Path(args.validation_arrow).resolve(),
        dataset_info_path=(
            Path(args.dataset_info).resolve()
            if args.dataset_info is not None
            else None
        ),
        search_context_count=args.search_contexts,
        validation_context_count=args.validation_contexts,
        final_context_count=args.final_contexts,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            catalog,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    load_probe_catalog(output)
    print(
        json.dumps(
            {
                "catalog_id": catalog["catalog_id"],
                "output": str(output),
                "probes": len(catalog["probes"]),
                "sha256": _sha256_file(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
