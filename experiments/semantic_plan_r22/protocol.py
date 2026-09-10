"""Frozen semantic-plan prompt for R22 normalization."""

from __future__ import annotations

from typing import Any

NORMALIZATION_SYSTEM = (
    "You normalize teacher responses into faithful capability-transfer examples. "
    "Follow the requested behavior. Include each supplied DATA field value exactly "
    "once as a case-sensitive verbatim substring and keep the three values in field "
    "order. Add no facts and return only the normalized response. Do not discuss "
    "the task."
)


def normalization_prompt(row: dict[str, Any]) -> str:
    slots = row["slots"]
    return "\n".join(
        (
            f"TASK_LABEL: {row['task']}",
            f"REQUEST: {row['instruction']}",
            "SEMANTIC_PLAN:",
            f"field_a={slots['field_a']}",
            f"field_b={slots['field_b']}",
            f"field_c={slots['field_c']}",
            "RAW_TEACHER_RESPONSE:",
            str(row["teacher_output"]),
            "NORMALIZED_RESPONSE:",
        )
    )


def normalization_request_sha256(row: dict[str, Any]) -> str:
    """Hash the complete model-independent normalization request."""
    import hashlib

    request = f"{NORMALIZATION_SYSTEM}\0{normalization_prompt(row)}"
    return hashlib.sha256(request.encode()).hexdigest()


def fields_verbatim_once(row: dict[str, Any], output: str) -> bool:
    """Require all three source values exactly once and in declared order."""
    values = [str(row["slots"][f"field_{suffix}"]) for suffix in "abc"]
    positions = [output.find(value) for value in values]
    return (
        all(position >= 0 for position in positions)
        and positions == sorted(positions)
        and all(output.count(value) == 1 for value in values)
    )
