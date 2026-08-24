"""Reference semantics for the representation-neutral ABI V2 boundary."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

ABI_VERSION = "abi-canonical-host/2"
VECTOR_DIMENSION = 64

CHANNELS: dict[str, tuple[int, tuple[str, ...]]] = {
    "instruction_type": (
        0,
        (
            "conversation",
            "summarize",
            "rewrite",
            "draft",
            "clarify",
            "abstain",
            "reason",
            "format",
            "classify",
            "extract",
            "transform",
            "answer",
            "calculate",
            "generate_code",
            "explain",
            "other",
        ),
    ),
    "constraint": (
        16,
        (
            "exact",
            "tone",
            "length",
            "format",
            "language",
            "audience",
            "preserve_names",
            "preserve_numbers",
            "cite_supplied",
            "context_only",
            "no_markdown",
            "lines",
            "safe",
            "privacy",
            "uncertainty",
            "other",
        ),
    ),
    "relation": (
        32,
        ("none", "is_a", "part_of", "causes", "precedes", "compares", "maps_to", "other"),
    ),
    "topic": (
        40,
        (
            "neutral",
            "interpersonal",
            "writing",
            "planning",
            "abstract",
            "procedural",
            "supplied_context",
            "unknown",
        ),
    ),
    "uncertainty": (48, ("certain", "probable", "uncertain", "insufficient")),
    "output_intent": (
        52,
        (
            "fluent_text",
            "exact_anchor",
            "structured_text",
            "clarification",
            "abstention",
            "summary",
            "rewrite",
            "other",
        ),
    ),
}

FLAG_INDICES = {
    "has_context": 60,
    "has_conversation": 61,
    "has_entities": 62,
    "has_exact_anchors": 63,
}


class CanonicalABIError(ValueError):
    """Raised when a canonical state or output violates ABI V2."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_utf8(value: str) -> bytes:
    if not isinstance(value, str):
        raise CanonicalABIError("canonical text fields must be strings")
    payload = value.encode("utf-8", errors="strict")
    if payload.decode("utf-8", errors="strict") != value:
        raise CanonicalABIError("text is not canonical strict UTF-8")
    return payload


def _index(channel: str, label: str) -> int:
    offset, labels = CHANNELS[channel]
    try:
        return offset + labels.index(label)
    except ValueError as exc:
        raise CanonicalABIError(f"unknown {channel}: {label}") from exc


def active_indices(record: Mapping[str, Any]) -> list[int]:
    indices = [
        _index("instruction_type", str(record["instruction_type"])),
        _index("relation", str(record["relation"])),
        _index("topic", str(record["topic"])),
        _index("uncertainty", str(record["uncertainty"])),
        _index("output_intent", str(record["output_intent"])),
    ]
    constraints = record.get("constraints", [])
    if not isinstance(constraints, list) or len(constraints) != len(set(constraints)):
        raise CanonicalABIError("constraints must be a unique list")
    indices.extend(_index("constraint", str(label)) for label in constraints)
    for name, index in FLAG_INDICES.items():
        if bool(record.get(name, False)):
            indices.append(index)
    if len(indices) != len(set(indices)):
        raise CanonicalABIError("canonical channel collision")
    return sorted(indices)


def normalized_vector(record: Mapping[str, Any]) -> list[float]:
    indices = active_indices(record)
    scale = 1.0 / math.sqrt(len(indices))
    vector = [0.0] * VECTOR_DIMENSION
    for index in indices:
        vector[index] = scale
    return vector


def _anchors(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    anchors = []
    for position, item in enumerate(values):
        text = str(item["text"])
        payload = strict_utf8(text)
        role = str(item.get("role", "literal"))
        if role not in {"entity", "number", "identifier", "literal", "format"}:
            raise CanonicalABIError(f"unknown anchor role: {role}")
        anchors.append(
            {
                "ordinal": position,
                "role": role,
                "utf8_hex": payload.hex(),
                "utf8_sha256": sha256_bytes(payload),
            }
        )
    return anchors


def canonical_context(record: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(record["prompt"])
    context = str(record.get("context", ""))
    conversation = record.get("conversation", [])
    entities = record.get("entities", [])
    anchors = record.get("anchors", [])
    if not isinstance(conversation, list) or not all(isinstance(turn, str) for turn in conversation):
        raise CanonicalABIError("conversation must be a list of strings")
    if not isinstance(entities, list) or not isinstance(anchors, list):
        raise CanonicalABIError("entities and anchors must be lists")
    channel_record = {
        **record,
        "has_context": bool(context),
        "has_conversation": bool(conversation),
        "has_entities": bool(entities),
        "has_exact_anchors": bool(anchors),
    }
    vector = normalized_vector(channel_record)
    result = {
        "abi_version": ABI_VERSION,
        "state_type": "canonical_context",
        "sequence_position": int(record.get("sequence_position", 0)),
        "prompt_utf8_hex": strict_utf8(prompt).hex(),
        "context_utf8_hex": strict_utf8(context).hex(),
        "conversation_utf8_hex": [strict_utf8(turn).hex() for turn in conversation],
        "entity_anchors": _anchors(
            {"text": entity, "role": "entity"} if isinstance(entity, str) else entity
            for entity in entities
        ),
        "exact_anchors": _anchors(anchors),
        "active_channel_indices": active_indices(channel_record),
        "normalized_semantic_vector_fp32": vector,
    }
    result["state_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def canonical_output_intent(
    output: str,
    *,
    capability_id: str,
    confidence: float = 1.0,
    applicability: float = 1.0,
    abstain: bool = False,
) -> dict[str, Any]:
    if not capability_id or not isinstance(capability_id, str):
        raise CanonicalABIError("capability_id is required")
    if not 0.0 <= confidence <= 1.0 or not 0.0 <= applicability <= 1.0:
        raise CanonicalABIError("confidence and applicability must be within [0, 1]")
    payload = strict_utf8(output)
    result = {
        "abi_version": ABI_VERSION,
        "state_type": "canonical_output_intent",
        "mode": "authoritative_utf8",
        "capability_id": capability_id,
        "confidence_fp32": float(confidence),
        "applicability_fp32": float(applicability),
        "abstain": bool(abstain),
        "residual_strength_fp32": 1.0,
        "authoritative_utf8_hex": payload.hex(),
        "authoritative_utf8_sha256": sha256_bytes(payload),
    }
    result["state_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    context = canonical_context(record["input"])
    expected = record["expected"]
    if context["active_channel_indices"] != expected["active_channel_indices"]:
        raise CanonicalABIError(f"reference active indices changed: {record['id']}")
    expected_hash = expected.get("state_sha256")
    if expected_hash is not None and context["state_sha256"] != expected_hash:
        raise CanonicalABIError(f"reference state changed: {record['id']}")
    return context
