"""Structure-contrastive template inference for R19."""

from __future__ import annotations

from collections import Counter
from typing import Any

from experiments.functional_realization_r18.verify import _parse_template
from experiments.linguistic_realization_r17.frames import SIGNATURES


class R19CompilerError(RuntimeError):
    """Raised when source evidence cannot identify a registered R19 package."""


def _partner(signature: str) -> str | None:
    mood, tense, polarity, number = signature.split("|")
    if mood != "question" and tense != "future":
        return None
    other = "negative" if polarity == "positive" else "positive"
    return "|".join((mood, tense, other, number))


def _positive_negative(left: str, right: str) -> tuple[str, str]:
    return (left, right) if "|positive|" in f"|{left}|" else (right, left)


def _one_literal_insertion(positive: str, negative: str) -> bool:
    positive_tokens = positive.split()
    negative_tokens = negative.split()
    if len(negative_tokens) != len(positive_tokens) + 1:
        return False
    return any(
        negative_tokens[:index] + negative_tokens[index + 1 :] == positive_tokens
        and not negative_tokens[index].startswith("{")
        for index in range(len(negative_tokens))
    )


def infer_templates(
    records: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[str, Counter[str]] = {signature: Counter() for signature in SIGNATURES}
    rejected = []
    for record in records:
        signature = str(record["signature"])
        if signature not in grouped:
            raise R19CompilerError("unknown R19 signature")
        template = _parse_template(str(record["output"]), record["slots"])
        if template is None:
            rejected.append(str(record["record_id"]))
        else:
            grouped[signature][template] += 1
    if any(sum(counter.values()) < 2 for counter in grouped.values()):
        raise R19CompilerError("insufficient parseable R19 support")
    templates: dict[str, str] = {}
    support: dict[str, Any] = {}
    completed_pairs: set[tuple[str, str]] = set()
    for signature in SIGNATURES:
        partner = _partner(signature)
        if partner is None:
            chosen, count = sorted(
                grouped[signature].items(), key=lambda item: (-item[1], item[0])
            )[0]
            templates[signature] = chosen
            support[signature] = {
                "mode": "local",
                "local": count,
                "joint": count,
            }
            continue
        positive_signature, negative_signature = _positive_negative(signature, partner)
        pair_key = (positive_signature, negative_signature)
        if pair_key in completed_pairs:
            continue
        candidates = []
        for positive_template, positive_count in grouped[positive_signature].items():
            for negative_template, negative_count in grouped[negative_signature].items():
                if _one_literal_insertion(positive_template, negative_template):
                    candidates.append(
                        (
                            positive_template,
                            negative_template,
                            positive_count,
                            negative_count,
                        )
                    )
        if not candidates:
            raise R19CompilerError("no structure-compatible polarity contrast")
        positive_template, negative_template, positive_count, negative_count = sorted(
            candidates,
            key=lambda item: (
                -(item[2] + item[3]),
                -min(item[2], item[3]),
                -item[2],
                item[0],
                item[1],
            ),
        )[0]
        joint = positive_count + negative_count
        templates[positive_signature] = positive_template
        templates[negative_signature] = negative_template
        support[positive_signature] = {
            "mode": "polarity_contrast",
            "local": positive_count,
            "paired_polarity": negative_count,
            "joint": joint,
        }
        support[negative_signature] = {
            "mode": "polarity_contrast",
            "local": negative_count,
            "paired_polarity": positive_count,
            "joint": joint,
        }
        completed_pairs.add(pair_key)
    if set(templates) != set(SIGNATURES):
        raise R19CompilerError("R19 template coverage changed")
    return templates, {
        "support": support,
        "parseable_records": sum(sum(counter.values()) for counter in grouped.values()),
        "rejected_record_ids": sorted(rejected),
    }
