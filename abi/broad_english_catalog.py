"""Build a broad, corpus-grounded English acquisition catalog.

The historical natural-English catalog varied synthetic slots but still gave
the teacher only a narrow surface distribution. This successor derives prompts
from disjoint TinyStories train and validation rows. The source text is treated
only as fictional or supplied context: no prompt asks the teacher for
closed-book facts, and every output is destined for the English core with zero
domain labels or claims.

The builder records the local corpus identity and keeps source-model extraction
separate. It does not claim that TinyStories is an English teacher, that the
corpus license is known when the cache metadata omits it, or that the resulting
LayerCake is fluent before held-out evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Sequence

from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
)
from .natural_english_catalog import BUILDERS


CATALOG_SCHEMA = "abi-corpus-grounded-english-catalog/1"
DEFAULT_SEED = 9_824
DEFAULT_SEARCH_STORIES = 400
DEFAULT_VALIDATION_STORIES = 100
DEFAULT_FINAL_STORIES = 100
DEFAULT_MAX_CONTEXT_CHARACTERS = 560

# These are the exact bounded exclusions in the currently locked starter
# ontology. They are recorded in every catalog and also enforced against
# teacher output by the schema-closed ``contains_none`` evaluator.
DEFAULT_CORE_EXCLUSION_MARKERS = (
    "atomic number",
    "chemical element",
    "periodic table",
    "independence day",
    "national holiday",
    "united states independence",
    "arithmetic",
    "calculate",
    "equation",
    "def",
    "import",
    "python",
)

_SPACE = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")


class BroadEnglishCatalogError(RuntimeError):
    """Raised when a corpus-grounded catalog cannot be built safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _marker_present(text: str, marker: str) -> bool:
    return re.search(
        r"(?<!\w)" + re.escape(marker) + r"(?!\w)",
        text.casefold(),
    ) is not None


def _is_safe_story(
    text: str,
    *,
    exclusion_markers: Sequence[str],
) -> bool:
    normalized = _normalized(text)
    if len(normalized) < 180:
        return False
    if any(_marker_present(normalized, marker) for marker in exclusion_markers):
        return False
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE.split(normalized)
        if sentence.strip()
    ]
    return len(sentences) >= 4


def _story_excerpt(text: str, maximum_characters: int) -> str:
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE.split(_normalized(text))
        if sentence.strip()
    ]
    selected: list[str] = []
    length = 0
    for sentence in sentences:
        addition = len(sentence) + (1 if selected else 0)
        if selected and length + addition > maximum_characters:
            break
        selected.append(sentence)
        length += addition
        if len(selected) >= 6:
            break
    if len(selected) < 3:
        raise BroadEnglishCatalogError(
            "story cannot supply three bounded context sentences"
        )
    return " ".join(selected)


def _story_fields(excerpt: str) -> dict[str, Any]:
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE.split(excerpt)
        if sentence.strip()
    ]
    distinctive = []
    seen = set()
    for word in _WORD.findall(excerpt):
        folded = word.casefold()
        if folded in seen or len(folded) < 4:
            continue
        seen.add(folded)
        distinctive.append(word)
    if len(distinctive) < 3:
        raise BroadEnglishCatalogError(
            "story lacks enough distinct linguistic anchors"
        )
    return {
        "first": sentences[0],
        "second": sentences[1],
        "third": sentences[2],
        "anchor": distinctive[0],
    }


def _corrupt_sentence(sentence: str) -> str:
    words = sentence.rstrip(".!?").split()
    if len(words) < 5:
        return sentence.casefold()
    return " ".join(words).casefold()


def _prompt(
    capability: str,
    excerpt: str,
) -> tuple[str, int]:
    fields = _story_fields(excerpt)
    context = f"<fictional_context>\n{excerpt}\n</fictional_context>"
    if capability == "grammar":
        return (
            f"{context}\nEdit the supplied rough sentence into one fluent, grammatical "
            "English sentence. Preserve its meaning and add no facts.\n"
            f"Rough sentence: {_corrupt_sentence(fields['first'])}",
            80,
        )
    if capability == "coherence":
        reordered = " ".join(
            [fields["third"], fields["first"], fields["second"]]
        )
        return (
            "Restore these fictional events to a coherent order and express "
            "them as a short paragraph. Use only the supplied events.\n"
            f"Events out of order: {reordered}",
            112,
        )
    if capability == "prompt_grounding":
        return (
            f"{context}\nAnswer in one or two sentences: What happened in "
            "this passage? Use only the fictional context and do not add "
            "background facts.",
            96,
        )
    if capability == "instruction_following":
        return (
            f"{context}\nWrite exactly two concise bullet points that capture "
            "the supplied passage. Begin each line with '- ' and add no "
            "heading or outside information.",
            96,
        )
    if capability == "conversation":
        return (
            f"{context}\nImagine that the main character has just told you "
            "this experience. Reply naturally and supportively in two short "
            "sentences, then ask one relevant follow-up question. Do not "
            "pretend to know anything beyond the passage.",
            112,
        )
    if capability == "summarization":
        return (
            f"{context}\nSummarize the passage in one clear sentence, using "
            "only information explicitly supplied.",
            80,
        )
    if capability == "rewriting":
        return (
            f"{context}\nRewrite the passage as a concise paragraph for an "
            "adult reader. Preserve the events, improve flow, and introduce "
            "no new facts.",
            128,
        )
    if capability == "email_drafting":
        return (
            f"{context}\nTreat the passage as fictional notes. Draft a short "
            "email from one character to another about the events. Include a "
            "greeting and closing, and add no facts not present in the notes.",
            144,
        )
    if capability == "tone_control":
        return (
            f"{context}\nRetell the supplied events in a calm, reassuring "
            "tone in no more than three sentences. Do not add information.",
            112,
        )
    if capability == "format_control":
        return (
            f"{context}\nReturn only a JSON object with string fields "
            "'summary' and 'mood'. Base both values solely on the fictional "
            "context. Do not use a Markdown code fence.",
            112,
        )
    if capability == "clarification":
        return (
            f"{context}\n"
            "A speaker then gives this ambiguous request: "
            f"'Please take care of the {fields['anchor']} part soon.' "
            "Ask one concise clarification question instead of assuming what "
            "they mean.",
            64,
        )
    if capability == "abstention":
        return (
            f"{context}\nWhat is the exact serial number of the "
            f"{fields['anchor']} mentioned here? The passage does not supply "
            "one. Say briefly that it cannot be known from the given "
            "information, without inventing a number.",
            64,
        )
    if capability == "domain_independent_reasoning":
        return (
            f"{context}\nUsing only the sequence of events in the fictional "
            "passage, explain one likely cause-and-effect connection. If the "
            "cause is not explicit, state that limitation instead of adding "
            "outside knowledge.",
            112,
        )
    if capability == "cake_output_realization":
        return (
            f"{context}\nUse the context only to disambiguate the fields. "
            "Turn the following supplied fictional fields into two connected, "
            "natural English sentences. Preserve each field and add no new "
            "facts.\n"
            f"event_one={fields['first']}\n"
            f"event_two={fields['second']}",
            112,
        )
    raise BroadEnglishCatalogError(f"unsupported capability: {capability}")


def _evaluator(
    capability: str,
    *,
    exclusion_markers: Sequence[str],
) -> dict[str, Any]:
    rules: list[dict[str, Any]] = [
        {"kind": "nonempty", "minimum_characters": 12},
        {"kind": "maximum_characters", "value": 1_200},
        {"kind": "contains_none", "values": list(exclusion_markers)},
    ]
    if capability == "clarification":
        rules.append({"kind": "contains_all", "values": ["?"]})
    elif capability == "abstention":
        rules.append(
            {
                "kind": "contains_any",
                "values": [
                    "cannot know",
                    "can't know",
                    "cannot determine",
                    "not provided",
                    "not given",
                    "not specified",
                    "unknown",
                ],
            }
        )
    return {"kind": "all_of", "rules": rules}


def _minhash_sample(
    texts: Iterable[str],
    *,
    count: int,
    seed: int,
    exclusion_markers: Sequence[str],
) -> list[str]:
    """Select the lowest stable hashes without loading a corpus into memory."""

    if count <= 0:
        raise BroadEnglishCatalogError("sample count must be positive")
    heap: list[tuple[int, str]] = []
    seen_excerpt_hashes: set[bytes] = set()
    prefix = f"{seed}:".encode("utf-8")
    for text in texts:
        if not isinstance(text, str) or not _is_safe_story(
            text, exclusion_markers=exclusion_markers
        ):
            continue
        normalized = _normalized(text)
        try:
            excerpt = _story_excerpt(
                normalized, DEFAULT_MAX_CONTEXT_CHARACTERS
            )
        except BroadEnglishCatalogError:
            continue
        excerpt_hash = hashlib.sha256(excerpt.encode("utf-8")).digest()
        if excerpt_hash in seen_excerpt_hashes:
            continue
        seen_excerpt_hashes.add(excerpt_hash)
        score = int.from_bytes(
            hashlib.sha256(prefix + normalized.encode("utf-8")).digest(),
            "big",
        )
        item = (-score, normalized)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) != count:
        raise BroadEnglishCatalogError(
            f"only {len(heap)} safe stories available for {count} rows"
        )
    return [
        text
        for _, text in sorted(
            [(-negative_score, text) for negative_score, text in heap]
        )
    ]


def _arrow_texts(path: Path) -> Iterator[str]:
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise BroadEnglishCatalogError(
            "pyarrow is required to read the cached corpus"
        ) from exc
    with path.open("rb") as handle:
        reader = ipc.open_stream(handle)
        if "text" not in reader.schema.names:
            raise BroadEnglishCatalogError("Arrow corpus has no text column")
        for batch in reader:
            for value in batch.column("text").to_pylist():
                if isinstance(value, str):
                    yield value


def _corpus_manifest(
    *,
    train_path: Path,
    validation_path: Path,
    dataset_info_path: Path | None,
) -> dict[str, Any]:
    info_sha = (
        _sha256_file(dataset_info_path)
        if dataset_info_path is not None and dataset_info_path.is_file()
        else None
    )
    declared_license = None
    if dataset_info_path is not None and dataset_info_path.is_file():
        info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        declared_license = str(info.get("license") or "").strip() or None
    return {
        "schema_version": "abi-raw-language-corpus-manifest/1",
        "dataset_id": "roneneldan/TinyStories",
        "dataset_revision": (
            "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
        ),
        "role": "fictional_supplied_context_only",
        "train_arrow": {
            "file_name": train_path.name,
            "sha256": _sha256_file(train_path),
            "bytes": train_path.stat().st_size,
        },
        "validation_arrow": {
            "file_name": validation_path.name,
            "sha256": _sha256_file(validation_path),
            "bytes": validation_path.stat().st_size,
        },
        "cached_dataset_info_sha256": info_sha,
        "cached_dataset_declared_license": declared_license,
        "license_status": (
            "declared_in_cached_metadata"
            if declared_license is not None
            else "not_declared_in_cached_metadata_research_evidence_only"
        ),
        "raw_corpus_is_teacher": False,
        "closed_book_fact_prompts": 0,
    }


def build_catalog(
    *,
    search_stories: Sequence[str],
    validation_stories: Sequence[str],
    final_stories: Sequence[str],
    corpus_manifest: dict[str, Any],
    seed: int = DEFAULT_SEED,
    maximum_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
    exclusion_markers: Sequence[str] = DEFAULT_CORE_EXCLUSION_MARKERS,
) -> dict[str, Any]:
    splits = {
        "search": list(search_stories),
        "validation": list(validation_stories),
        "final_test": list(final_stories),
    }
    normalized_sets = {
        split: {_normalized(text) for text in stories}
        for split, stories in splits.items()
    }
    if any(not values for values in normalized_sets.values()):
        raise BroadEnglishCatalogError("every split must contain stories")
    split_names = list(normalized_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if normalized_sets[left] & normalized_sets[right]:
                raise BroadEnglishCatalogError(
                    f"story overlap between {left} and {right}"
                )

    probes: list[dict[str, Any]] = []
    context_hashes: dict[str, list[str]] = {}
    for split_index, (split, stories) in enumerate(splits.items()):
        split_hashes = []
        for story_index, story in enumerate(stories):
            excerpt = _story_excerpt(
                story, maximum_characters=maximum_context_characters
            )
            excerpt_sha = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
            split_hashes.append(excerpt_sha)
            for capability_index, capability in enumerate(BUILDERS):
                prompt, maximum = _prompt(capability, excerpt)
                if any(
                    _marker_present(prompt, marker)
                    for marker in exclusion_markers
                ):
                    raise BroadEnglishCatalogError(
                        "core exclusion marker crossed prompt construction"
                    )
                probe: dict[str, Any] = {
                    "probe_id": (
                        f"broad-{split}-{story_index:04d}-"
                        f"{capability}-v1"
                    ),
                    "destination_scope": "english_core",
                    "capability": capability,
                    "domain": "domain_independent",
                    "split": split,
                    "prompt": prompt,
                    "max_new_tokens": maximum,
                    "temperature": 0,
                    "seed": (
                        seed
                        + split_index * 1_000_000
                        + story_index * len(BUILDERS)
                        + capability_index
                    ),
                    "evaluator": _evaluator(
                        capability,
                        exclusion_markers=exclusion_markers,
                    ),
                    "record_schema": SEGREGATED_RECORD_SCHEMA,
                    "knowledge_class": LINGUISTIC_FORM,
                    "content_basis": (
                        "interpersonal_pragmatics"
                        if capability == "conversation"
                        else (
                            "domain_free_instruction"
                            if capability
                            in {
                                "clarification",
                                "abstention",
                                "instruction_following",
                            }
                            else (
                                "abstract_or_nonce_content"
                                if capability
                                == "domain_independent_reasoning"
                                else "supplied_non_domain_context"
                            )
                        )
                    ),
                    "domain_labels": [],
                    "domain_claims": [],
                    "label_method": "preregistered_catalog",
                    "output_introduces_unsupplied_facts": False,
                    "raw_context_sha256": excerpt_sha,
                }
                probe["label_evidence_sha256"] = (
                    probe_label_evidence_sha256(probe)
                )
                probes.append(probe)
        context_hashes[split] = sorted(split_hashes)

    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-broad-corpus-grounded-english-v1",
        "catalog_contract_schema": CATALOG_SCHEMA,
        "status": "PREREGISTERED_BROAD_ENGLISH_ACQUISITION",
        "claim_boundary": (
            "This catalog supplies diverse fictional contexts to a frozen "
            "teacher. It is a training-data acquisition surface, not evidence "
            "that a LayerCake is fluent, teacher-equivalent, domain-exhaustive, "
            "or free of all latent world knowledge."
        ),
        "generation": {
            "generator": "abi.broad_english_catalog",
            "seed": seed,
            "capabilities": list(BUILDERS),
            "story_counts": {
                split: len(stories) for split, stories in splits.items()
            },
            "probe_counts": {
                split: len(stories) * len(BUILDERS)
                for split, stories in splits.items()
            },
            "maximum_context_characters": maximum_context_characters,
            "distinct_contexts_per_capability": {
                split: len(stories) for split, stories in splits.items()
            },
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
    search_story_count: int,
    validation_story_count: int,
    final_story_count: int,
    seed: int,
) -> dict[str, Any]:
    search = _minhash_sample(
        _arrow_texts(train_path),
        count=search_story_count,
        seed=seed,
        exclusion_markers=DEFAULT_CORE_EXCLUSION_MARKERS,
    )
    held_out = _minhash_sample(
        _arrow_texts(validation_path),
        count=validation_story_count + final_story_count,
        seed=seed + 1,
        exclusion_markers=DEFAULT_CORE_EXCLUSION_MARKERS,
    )
    return build_catalog(
        search_stories=search,
        validation_stories=held_out[:validation_story_count],
        final_stories=held_out[validation_story_count:],
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
        "--search-stories", type=int, default=DEFAULT_SEARCH_STORIES
    )
    parser.add_argument(
        "--validation-stories",
        type=int,
        default=DEFAULT_VALIDATION_STORIES,
    )
    parser.add_argument(
        "--final-stories", type=int, default=DEFAULT_FINAL_STORIES
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
        search_story_count=args.search_stories,
        validation_story_count=args.validation_stories,
        final_story_count=args.final_stories,
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
