"""Build a broad, domain-filtered natural English prompt-surface catalog.

Only the first user/prompter message is admitted from each conversation.
Corpus assistant messages and reference answers are never read into a probe.
The resulting search-only catalog must be answered by the pinned open-weight
teacher on CUDA before it can become LayerCake training material.

The filter is intentionally conservative and deterministic.  It excludes
closed-book factual requests, specialist domains, unsafe content, numerical
facts, network identifiers, code-like payloads, and non-English requests.
That finite filter is evidence of a bounded segregation policy; it is not a
claim that arbitrary natural language contains literally zero world knowledge.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from datasets import Dataset

from .broad_english_catalog import DEFAULT_CORE_EXCLUSION_MARKERS
from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
)


CATALOG_CONTRACT = "abi-broad-natural-conversation-english-catalog/2"
DEFAULT_SEED = 79_824
ULTRACHAT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
ULTRACHAT_LICENSE = "mit"
OASST_DATASET_ID = "OpenAssistant/oasst1"
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
OASST_LICENSE = "apache-2.0"

_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_DIGIT_NETWORK_OR_MARKUP = re.compile(
    r"\d|https?://|www\.|[\w.+-]+@[\w.-]+|```|</|<script|"
    r"\{\{|\{%|(?:^|\s)[{}](?:\s|$)",
    re.IGNORECASE,
)
_NON_ENGLISH = re.compile(
    r"\b(?:translate|translation|spanish|french|german|italian|"
    r"chinese|japanese|korean|arabic|hindi|portuguese|russian|"
    r"language other than english|non-english)\b",
    re.IGNORECASE,
)
_SPECIALIST_OR_FACTUAL_DOMAIN = re.compile(
    r"\b(?:"
    r"python|javascript|typescript|java|c\+\+|rust|golang|source code|"
    r"code|coding|program(?:ming)?|algorithm|function|class|library|"
    r"database|sql|machine learning|neural network|computer science|"
    r"software|hardware|computer|technology|"
    r"calculate|calculation|arithmetic|algebra|geometry|equation|"
    r"mathematics|math|statistic(?:s|al)?|probability|"
    r"physics|chemistry|biology|science|scientific|"
    r"medical|medicine|disease|diagnosis|therapy|drug|"
    r"legal|law|lawsuit|finance|financial|stock|investment|economics|"
    r"history|historical|president|prime minister|government|country|"
    r"capital city|war|religion|politic(?:s|al)?|covid|pandemic|climate|"
    r"atomic number|periodic table|independence day|national holiday|"
    r"research|citation|cite|bibliography|source|"
    r"recipe|ingredient|nutrition|"
    r"weapon|explosive|suicide|self-harm|sexual|porn|"
    r"book|novel|movie|film|song|music|celebrity|company|product|brand"
    r")\b",
    re.IGNORECASE,
)
_CLOSED_BOOK_FACT_REQUEST = re.compile(
    r"(?:"
    r"^\s*(?:who|when|where|why|which)\b|"
    r"^\s*what\s+(?:is|are|was|were|does|do|did|should i know)\b|"
    r"^\s*how\s+(?:many|much|old|long|far)\b|"
    r"\b(?:tell me about|explain (?:the|how|why|what)|"
    r"provide (?:information|an overview)|give me facts|"
    r"according to current|latest)\b"
    r")",
    re.IGNORECASE,
)
_UNSAFE = re.compile(
    r"\b(?:hate speech|racial slur|terroris[mt]|murder|kill|"
    r"sexual assault|child abuse|harass(?:ment)?|steal|fraud)\b",
    re.IGNORECASE,
)

# Ordered from narrowest to broadest. A prompt receives one capability label.
CAPABILITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "grammar",
        re.compile(
            r"\b(?:grammar|grammatical|spelling|proofread|"
            r"correct .{0,30} sentence|fix .{0,30} sentence)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "email_drafting",
        re.compile(
            r"\b(?:write|draft|compose|create|generate)\b.{0,80}"
            r"\b(?:email|e-mail|letter|memo|message)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "summarization",
        re.compile(r"\b(?:summarize|summarise|summary)\b", re.IGNORECASE),
    ),
    (
        "tone_control",
        re.compile(
            r"\b(?:tone|formal|informal|professional|polite|friendly|"
            r"empathetic|reassuring|enthusiastic|diplomatic|respectful)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rewriting",
        re.compile(
            r"\b(?:rewrite|rephrase|paraphrase|edit|"
            r"make .{0,40}(?:clear|concise|readable)|"
            r"improve .{0,35}(?:sentence|text|paragraph|wording))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "conversation",
        re.compile(
            r"\b(?:reply|respond|conversation|dialogue|chat|roleplay|"
            r"what should i say|how (?:can|should) i "
            r"(?:tell|ask|say|respond|apologize|apologise))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "format_control",
        re.compile(
            r"\b(?:bullet|format|table|list|uppercase|lowercase|"
            r"return only|output only|heading|outline)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "coherence",
        re.compile(
            r"\b(?:coherent|cohesive|logical order|rearrange|sequence|"
            r"organize|organise|flow better)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cake_output_realization",
        re.compile(
            r"\b(?:given (?:words|keywords|fields|notes)|"
            r"turn .{0,40} into .{0,20}sentence|"
            r"combine .{0,50} sentence|construct a sentence)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "clarification",
        re.compile(
            r"\b(?:clarif|ambiguous|unclear|missing information|"
            r"ask me|need more information)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "abstention",
        re.compile(
            r"\b(?:not enough information|insufficient information|"
            r"cannot be determined|do not know)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "domain_independent_reasoning",
        re.compile(
            r"\b(?:if .{0,100} then|deduce|infer|logical|reasoning|"
            r"which statement follows)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_grounding",
        re.compile(
            r"\b(?:based on|according to|given the (?:text|passage|"
            r"information)|following (?:text|passage)|extract|identify)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_following",
        re.compile(
            r"\b(?:exactly|only|must|should|write|generate|create|draft|"
            r"compose|imagine|pretend)\b",
            re.IGNORECASE,
        ),
    ),
)

MAX_NEW_TOKENS = {
    "email_drafting": 192,
    "summarization": 160,
    "rewriting": 160,
    "conversation": 160,
    "tone_control": 160,
}


class NaturalConversationCatalogError(RuntimeError):
    """Raised when the broad natural catalog cannot be built safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _marker_present(text: str, marker: str) -> bool:
    return (
        re.search(
            r"(?<!\w)" + re.escape(marker) + r"(?!\w)",
            text.casefold(),
        )
        is not None
    )


def _classify(text: str) -> str | None:
    for capability, pattern in CAPABILITY_PATTERNS:
        if pattern.search(text):
            return capability
    return None


def _safe_prompt_surface(text: str) -> tuple[str, str] | None:
    text = _normalize(text)
    if not 20 <= len(text) <= 600:
        return None
    if len(_WORD.findall(text)) < 5:
        return None
    if sum(ord(character) < 128 for character in text) / len(text) < 0.98:
        return None
    if (
        _DIGIT_NETWORK_OR_MARKUP.search(text)
        or _NON_ENGLISH.search(text)
        or _SPECIALIST_OR_FACTUAL_DOMAIN.search(text)
        or _CLOSED_BOOK_FACT_REQUEST.search(text)
        or _UNSAFE.search(text)
    ):
        return None
    if any(
        _marker_present(text, marker)
        for marker in DEFAULT_CORE_EXCLUSION_MARKERS
    ):
        return None
    capability = _classify(text)
    if capability is None:
        return None
    return text, capability


def _teacher_prompt(source_prompt: str) -> str:
    return (
        f"{source_prompt}\n\n"
        "Respond in English. This is English-form acquisition: follow the "
        "request, but do not introduce outside factual claims, named entities, "
        "numerical facts, or specialist knowledge. If the request depends on "
        "missing or specialist information, state that it is not supplied or "
        "ask one concise clarification question."
    )


def _evaluator() -> dict[str, Any]:
    return {
        "kind": "all_of",
        "rules": [
            {"kind": "nonempty", "minimum_characters": 12},
            {"kind": "maximum_characters", "value": 1_200},
            {
                "kind": "contains_none",
                "values": list(DEFAULT_CORE_EXCLUSION_MARKERS),
            },
        ],
    }


def _ultrachat_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    shard_sha256: str,
) -> Iterable[dict[str, str]]:
    for row_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        first = messages[0]
        if not isinstance(first, Mapping) or first.get("role") != "user":
            continue
        safe = _safe_prompt_surface(str(first.get("content", "")))
        if safe is None:
            continue
        text, capability = safe
        yield {
            "source": "ultrachat_200k",
            "source_shard_sha256": shard_sha256,
            "source_row_index": str(row_index),
            "source_prompt": text,
            "capability": capability,
        }


def _oasst_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    shard_sha256: str,
) -> Iterable[dict[str, str]]:
    for row_index, row in enumerate(rows):
        if (
            row.get("role") != "prompter"
            or row.get("parent_id") is not None
            or row.get("lang") != "en"
            or row.get("synthetic") is not False
            or row.get("deleted") is not False
            or row.get("review_result") is not True
        ):
            continue
        detoxify = row.get("detoxify")
        if not isinstance(detoxify, Mapping):
            continue
        if any(
            float(detoxify.get(field, 0.0) or 0.0) > 0.05
            for field in (
                "toxicity",
                "severe_toxicity",
                "obscene",
                "identity_attack",
                "insult",
                "threat",
                "sexual_explicit",
            )
        ):
            continue
        safe = _safe_prompt_surface(str(row.get("text", "")))
        if safe is None:
            continue
        text, capability = safe
        yield {
            "source": "oasst1",
            "source_shard_sha256": shard_sha256,
            "source_row_index": str(row_index),
            "source_prompt": text,
            "capability": capability,
        }


def build_catalog(
    *,
    ultrachat_candidates: Sequence[Mapping[str, str]],
    oasst_candidates: Sequence[Mapping[str, str]],
    source_corpora: Sequence[Mapping[str, Any]],
    maximum_per_capability: int = 5_000,
    minimum_per_capability: int = 100,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build one deterministic search-only catalog from candidate prompts."""

    if maximum_per_capability < minimum_per_capability:
        raise NaturalConversationCatalogError(
            "maximum_per_capability must be at least minimum_per_capability"
        )
    grouped: dict[str, list[dict[str, str]]] = {
        capability: [] for capability, _ in CAPABILITY_PATTERNS
    }
    seen: set[str] = set()
    rejection_counts = Counter()
    for candidate in [*oasst_candidates, *ultrachat_candidates]:
        source_prompt = _normalize(str(candidate["source_prompt"]))
        safe = _safe_prompt_surface(source_prompt)
        if safe is None:
            rejection_counts["unsafe_candidate_at_composition"] += 1
            continue
        normalized, capability = safe
        if capability != candidate["capability"]:
            raise NaturalConversationCatalogError(
                "candidate capability changed during composition"
            )
        prompt_sha256 = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()
        if prompt_sha256 in seen:
            rejection_counts["duplicate_prompt"] += 1
            continue
        seen.add(prompt_sha256)
        grouped[capability].append(
            {
                **dict(candidate),
                "source_prompt": normalized,
                "source_prompt_sha256": prompt_sha256,
            }
        )

    selected: dict[str, list[dict[str, str]]] = {}
    for capability, candidates in grouped.items():
        # Prefer independently reviewed human OASST prompts, then use a seeded
        # content hash to select UltraChat surfaces without reading answers.
        candidates = sorted(
            candidates,
            key=lambda row: (
                0 if row["source"] == "oasst1" else 1,
                hashlib.sha256(
                    f"{seed}:{capability}:{row['source_prompt_sha256']}".encode(
                        "ascii"
                    )
                ).hexdigest(),
            ),
        )
        if len(candidates) >= minimum_per_capability:
            selected[capability] = candidates[:maximum_per_capability]

    if not selected:
        raise NaturalConversationCatalogError(
            "no capability reached the minimum natural-prompt count"
        )

    probes: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    selected_hashes: list[str] = []
    for capability_index, (capability, _) in enumerate(CAPABILITY_PATTERNS):
        for ordinal, candidate in enumerate(selected.get(capability, [])):
            source_counts[candidate["source"]] += 1
            selected_hashes.append(candidate["source_prompt_sha256"])
            prompt = _teacher_prompt(candidate["source_prompt"])
            probe: dict[str, Any] = {
                "probe_id": (
                    f"broad-natural-search-{capability}-{ordinal:05d}-v3"
                ),
                "destination_scope": "english_core",
                "capability": capability,
                "domain": "domain_independent",
                "split": "search",
                "prompt": prompt,
                "max_new_tokens": MAX_NEW_TOKENS.get(capability, 128),
                "temperature": 0,
                "seed": seed + capability_index * 100_000 + ordinal,
                "evaluator": _evaluator(),
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": (
                    "interpersonal_pragmatics"
                    if capability == "conversation"
                    else (
                        "abstract_or_nonce_content"
                        if capability == "domain_independent_reasoning"
                        else "domain_free_instruction"
                    )
                ),
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
                "natural_prompt_sha256": candidate["source_prompt_sha256"],
                "source_prompt_corpus": candidate["source"],
                "source_shard_sha256": candidate["source_shard_sha256"],
                "source_row_index": int(candidate["source_row_index"]),
                "corpus_assistant_messages_imported": 0,
                "corpus_reference_answers_imported": 0,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)

    capability_counts = Counter(
        str(probe["capability"]) for probe in probes
    )
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-broad-natural-conversation-search-v3",
        "catalog_contract_schema": CATALOG_CONTRACT,
        "status": "PREREGISTERED_SEARCH_ONLY_BROAD_NATURAL_ACQUISITION",
        "claim_boundary": (
            "This catalog contributes first-user prompt surfaces only. "
            "Corpus assistant messages and reference answers are excluded. "
            "Finite lexical, structural, safety, and intent filters do not "
            "prove literal zero latent world knowledge, teacher fidelity, or "
            "LayerCake fluency."
        ),
        "generation": {
            "generator": "abi.natural_conversation_catalog",
            "seed": seed,
            "split": "search",
            "capability_counts": dict(sorted(capability_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "total_probes": len(probes),
            "maximum_per_capability": maximum_per_capability,
            "minimum_per_capability": minimum_per_capability,
            "capabilities_below_minimum_omitted": sorted(
                capability
                for capability, candidates in grouped.items()
                if len(candidates) < minimum_per_capability
            ),
            "closed_book_rows_admitted": 0,
            "corpus_assistant_messages_imported": 0,
            "corpus_reference_answers_imported": 0,
            "specialist_domain_rows_admitted": 0,
            "digit_url_email_rows_admitted": 0,
            "prompt_sha256_set_sha256": hashlib.sha256(
                "\n".join(sorted(selected_hashes)).encode("ascii")
            ).hexdigest(),
            "composition_rejections": dict(sorted(rejection_counts.items())),
        },
        "source_prompt_corpora": list(source_corpora),
        "core_exclusion_markers": list(DEFAULT_CORE_EXCLUSION_MARKERS),
        "probes": probes,
    }


def _file_manifest(path: Path) -> dict[str, Any]:
    return {
        "file_name": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_catalog_from_arrow(
    *,
    ultrachat_paths: Sequence[Path],
    oasst_path: Path,
    ultrachat_readme: Path,
    oasst_readme: Path,
    maximum_per_capability: int,
    minimum_per_capability: int,
    seed: int,
) -> dict[str, Any]:
    """Load the pinned Arrow files and build the immutable catalog."""

    if not ultrachat_paths:
        raise NaturalConversationCatalogError(
            "at least one UltraChat shard is required"
        )
    ultra_candidates: list[dict[str, str]] = []
    ultra_manifests = []
    for path in ultrachat_paths:
        manifest = _file_manifest(path)
        ultra_manifests.append(manifest)
        ultra_candidates.extend(
            _ultrachat_candidates(
                Dataset.from_file(str(path)),
                shard_sha256=str(manifest["sha256"]),
            )
        )
    oasst_manifest = _file_manifest(oasst_path)
    oasst_candidates = list(
        _oasst_candidates(
            Dataset.from_file(str(oasst_path)),
            shard_sha256=str(oasst_manifest["sha256"]),
        )
    )
    return build_catalog(
        ultrachat_candidates=ultra_candidates,
        oasst_candidates=oasst_candidates,
        source_corpora=[
            {
                "dataset_id": ULTRACHAT_DATASET_ID,
                "revision": ULTRACHAT_REVISION,
                "license": ULTRACHAT_LICENSE,
                "files": ultra_manifests,
                "readme": _file_manifest(ultrachat_readme),
                "role": "first_user_prompt_surface_only",
                "assistant_messages_used": False,
                "reference_answers_used": False,
            },
            {
                "dataset_id": OASST_DATASET_ID,
                "revision": OASST_REVISION,
                "license": OASST_LICENSE,
                "files": [oasst_manifest],
                "readme": _file_manifest(oasst_readme),
                "role": "reviewed_human_root_prompter_surface_only",
                "assistant_messages_used": False,
                "reference_answers_used": False,
            },
        ],
        maximum_per_capability=maximum_per_capability,
        minimum_per_capability=minimum_per_capability,
        seed=seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultrachat-arrow", action="append", required=True)
    parser.add_argument("--oasst-arrow", required=True)
    parser.add_argument("--ultrachat-readme", required=True)
    parser.add_argument("--oasst-readme", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-per-capability", type=int, default=5_000)
    parser.add_argument("--minimum-per-capability", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    catalog = build_catalog_from_arrow(
        ultrachat_paths=[
            Path(value).resolve() for value in args.ultrachat_arrow
        ],
        oasst_path=Path(args.oasst_arrow).resolve(),
        ultrachat_readme=Path(args.ultrachat_readme).resolve(),
        oasst_readme=Path(args.oasst_readme).resolve(),
        maximum_per_capability=args.maximum_per_capability,
        minimum_per_capability=args.minimum_per_capability,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    load_probe_catalog(output)
    print(
        json.dumps(
            {
                "catalog_id": catalog["catalog_id"],
                "capability_counts": catalog["generation"][
                    "capability_counts"
                ],
                "source_counts": catalog["generation"]["source_counts"],
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
