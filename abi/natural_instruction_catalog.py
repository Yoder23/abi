"""Build a domain-filtered natural-instruction English acquisition catalog.

The corpus contributes human-written *prompt surfaces* only.  Its reference
answers are deliberately ignored: a separately pinned open-weight source model
must generate every training target.  Rows are admitted only when they contain
supplied text, request a linguistic transformation, and clear a conservative
specialist/closed-book filter.

This is a bounded filter, not proof that arbitrary natural language is devoid
of world knowledge.  The generated catalog therefore remains research
material until teacher-output segregation, LayerCake generalization, and the
complete runtime gates pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .broad_english_catalog import DEFAULT_CORE_EXCLUSION_MARKERS
from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
)


CATALOG_CONTRACT = "abi-natural-instruction-english-catalog/1"
DATASET_ID = "yahma/alpaca-cleaned"
DATASET_REVISION = "12567cabf869d7c92e573c7c783905fc160e9639"
DATASET_LICENSE = "cc-by-4.0"
DEFAULT_SEED = 59_824

_SPACE = re.compile(r"\s+")
_DIGIT_OR_NETWORK = re.compile(r"\d|https?://|www\.|[\w.+-]+@[\w.-]+")
_NON_ENGLISH_TASK = re.compile(
    r"\b(?:translate|translation|spanish|french|german|italian|"
    r"chinese|japanese|korean|arabic|hindi|language other than english)\b",
    re.IGNORECASE,
)
_CLOSED_BOOK_QUESTION = re.compile(
    r"^\s*(?:who|when|where|why|what|which|how)\b",
    re.IGNORECASE,
)
_SPECIALIST = re.compile(
    r"\b(?:"
    r"python|javascript|typescript|java|c\+\+|rust|golang|source code|"
    r"program(?:ming)?|algorithm|function|class|library|database|sql|"
    r"machine learning|neural network|computer science|"
    r"calculate|calculation|arithmetic|algebra|geometry|equation|"
    r"physics|chemistry|biology|medical|medicine|disease|diagnosis|"
    r"legal|law|lawsuit|finance|financial|stock|investment|economics|"
    r"history|historical|president|prime minister|country|capital city|"
    r"war|religion|politic(?:s|al)?|covid|pandemic|climate|"
    r"atomic number|periodic table|independence day"
    r")\b",
    re.IGNORECASE,
)

# Ordered from narrowest to broadest so a row receives exactly one label.
CAPABILITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "grammar",
        re.compile(
            r"\b(?:grammar|grammatical|spelling|proofread|"
            r"correct (?:this|the|following) sentence|"
            r"fix (?:this|the|following) sentence)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "email_drafting",
        re.compile(
            r"\b(?:write|draft|compose|create|generate)\b.{0,60}"
            r"\b(?:email|e-mail|letter|memo)\b",
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
            r"empathetic|reassuring|enthusiastic)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rewriting",
        re.compile(
            r"\b(?:rewrite|rephrase|paraphrase|edit|"
            r"improve the (?:sentence|text|paragraph)|"
            r"make (?:this|the) .{0,35}(?:clear|concise))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "conversation",
        re.compile(
            r"\b(?:reply|respond|conversation|dialogue|chat)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "format_control",
        re.compile(
            r"\b(?:json|bullet|format|table|list|uppercase|lowercase|"
            r"return only|output only)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "coherence",
        re.compile(
            r"\b(?:coherent|cohesive|logical order|rearrange|sequence|"
            r"organize the (?:sentence|paragraph|text))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cake_output_realization",
        re.compile(
            r"\b(?:given (?:words|keywords|fields|notes)|"
            r"turn .{0,35} into (?:a |one |two )?sentence|"
            r"combine .{0,45} sentence|construct a sentence)\b",
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
            r"\b(?:exactly|only|must|should|return|output|write|generate|"
            r"create)\b",
            re.IGNORECASE,
        ),
    ),
)

MAX_NEW_TOKENS = {
    "email_drafting": 192,
    "summarization": 144,
    "rewriting": 160,
    "conversation": 160,
}


class NaturalInstructionCatalogError(RuntimeError):
    """Raised when the natural-instruction catalog cannot be built safely."""


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


def _classify(instruction: str) -> str | None:
    for capability, pattern in CAPABILITY_PATTERNS:
        if pattern.search(instruction):
            return capability
    return None


def _domain_light(instruction: str, supplied_text: str) -> bool:
    joined = f"{instruction}\n{supplied_text}"
    if not 8 <= len(instruction) <= 240:
        return False
    if not 20 <= len(supplied_text) <= 600:
        return False
    if _DIGIT_OR_NETWORK.search(joined):
        return False
    if _NON_ENGLISH_TASK.search(joined) or _SPECIALIST.search(joined):
        return False
    if _CLOSED_BOOK_QUESTION.search(instruction):
        return False
    if any(
        _marker_present(joined, marker)
        for marker in DEFAULT_CORE_EXCLUSION_MARKERS
    ):
        return False
    # Avoid likely code, markup payloads, and corpus-control delimiters.
    if any(value in joined for value in ("```", "<script", "</", "{%", "##\n")):
        return False
    return True


def _prompt(instruction: str, supplied_text: str) -> str:
    return (
        f"{instruction}\n\n"
        "<supplied_text>\n"
        f"{supplied_text}\n"
        "</supplied_text>\n\n"
        "Respond in English. Use only the supplied text and the linguistic "
        "instruction above. Do not introduce outside facts. If the request "
        "cannot be completed from the supplied text, ask one concise "
        "clarification question or say that the information is not supplied."
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


def _candidate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {
        capability: [] for capability, _ in CAPABILITY_PATTERNS
    }
    seen: set[str] = set()
    for row in rows:
        instruction = _normalize(str(row.get("instruction", "")))
        supplied_text = _normalize(str(row.get("input", "")))
        if not supplied_text or not _domain_light(instruction, supplied_text):
            continue
        capability = _classify(instruction)
        if capability is None:
            continue
        prompt = _prompt(instruction, supplied_text)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_sha in seen:
            continue
        seen.add(prompt_sha)
        grouped[capability].append(
            {
                "instruction": instruction,
                "supplied_text": supplied_text,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
            }
        )
    return grouped


def build_catalog(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_file_manifest: Mapping[str, Any],
    maximum_per_capability: int = 0,
    minimum_per_capability: int = 20,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build one search-only catalog from domain-filtered natural prompts."""

    if maximum_per_capability < 0:
        raise NaturalInstructionCatalogError(
            "maximum_per_capability must be non-negative"
        )
    if minimum_per_capability <= 0:
        raise NaturalInstructionCatalogError(
            "minimum_per_capability must be positive"
        )
    grouped = _candidate_rows(rows)
    probes: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    prompt_hashes: list[str] = []
    for capability_index, (capability, _) in enumerate(CAPABILITY_PATTERNS):
        candidates = sorted(
            grouped[capability],
            key=lambda row: hashlib.sha256(
                f"{seed}:{capability}:{row['prompt_sha256']}".encode("utf-8")
            ).hexdigest(),
        )
        if maximum_per_capability:
            candidates = candidates[:maximum_per_capability]
        if len(candidates) < minimum_per_capability:
            raise NaturalInstructionCatalogError(
                f"{capability} has only {len(candidates)} safe rows; "
                f"{minimum_per_capability} required"
            )
        counts[capability] = len(candidates)
        for ordinal, candidate in enumerate(candidates):
            probe: dict[str, Any] = {
                "probe_id": f"natural-search-{capability}-{ordinal:05d}-v1",
                "destination_scope": "english_core",
                "capability": capability,
                "domain": "domain_independent",
                "split": "search",
                "prompt": candidate["prompt"],
                "max_new_tokens": MAX_NEW_TOKENS.get(capability, 128),
                "temperature": 0,
                "seed": seed + capability_index * 100_000 + ordinal,
                "evaluator": _evaluator(),
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": (
                    "interpersonal_pragmatics"
                    if capability == "conversation"
                    else "supplied_non_domain_context"
                ),
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
                "natural_prompt_sha256": candidate["prompt_sha256"],
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)
            prompt_hashes.append(candidate["prompt_sha256"])
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-natural-domain-filtered-instruction-search-v1",
        "catalog_contract_schema": CATALOG_CONTRACT,
        "status": "PREREGISTERED_SEARCH_ONLY_NATURAL_ACQUISITION",
        "claim_boundary": (
            "This catalog contributes human-written instruction surfaces with "
            "supplied, bounded text. The original corpus answers are excluded. "
            "A deterministic filter and finite marker list do not prove zero "
            "world knowledge, teacher fidelity, or LayerCake fluency."
        ),
        "generation": {
            "generator": "abi.natural_instruction_catalog",
            "seed": seed,
            "split": "search",
            "capability_counts": counts,
            "total_probes": len(probes),
            "maximum_per_capability": maximum_per_capability,
            "minimum_per_capability": minimum_per_capability,
            "closed_book_rows_admitted": 0,
            "original_reference_answers_imported": 0,
            "specialist_domain_rows_admitted": 0,
            "prompt_sha256_set_sha256": hashlib.sha256(
                "\n".join(sorted(prompt_hashes)).encode("ascii")
            ).hexdigest(),
        },
        "source_prompt_corpus": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            **dict(source_file_manifest),
            "role": "prompt_surface_only",
            "reference_answers_used": False,
        },
        "core_exclusion_markers": list(DEFAULT_CORE_EXCLUSION_MARKERS),
        "probes": probes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpaca-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-per-capability", type=int, default=0)
    parser.add_argument("--minimum-per-capability", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    input_path = Path(args.alpaca_json).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        parser.error(f"catalog is immutable: {output_path}")
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        parser.error("Alpaca input must be a JSON array")
    catalog = build_catalog(
        rows=raw,
        source_file_manifest={
            "file_name": input_path.name,
            "sha256": _sha256_file(input_path),
            "bytes": input_path.stat().st_size,
        },
        maximum_per_capability=args.maximum_per_capability,
        minimum_per_capability=args.minimum_per_capability,
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
                "catalog_id": catalog["catalog_id"],
                "capability_counts": catalog["generation"][
                    "capability_counts"
                ],
                "output": str(output_path),
                "probes": len(catalog["probes"]),
                "sha256": _sha256_file(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
