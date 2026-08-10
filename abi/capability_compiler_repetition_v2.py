"""Versioned autonomous-generation repetition-collapse metric.

The original campaign metric is intentionally left unchanged for historical
reproducibility.  This module defines the candidate replacement audited in
V446/V447.  It detects repeated contiguous loops and extremely low four-gram
diversity; it does not treat the mere reuse of a common word as collapse.
"""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[\w]+(?:['’][\w]+)*|[^\w\s]", re.UNICODE)


def repetition_tokens(output: str) -> list[str]:
    """Return case-normalized word and punctuation tokens."""

    return TOKEN_PATTERN.findall(output.casefold())


def repetition_collapse_v2(output: str) -> bool:
    """Detect local loops or globally degenerate four-gram diversity.

    A local loop is any contiguous 1--16 token span repeated four times in a
    row.  For outputs of at least 32 tokens, fewer than 35% unique four-grams
    is also collapse.  Both rules operate on words *and* punctuation so that
    punctuation loops remain detectable.
    """

    tokens = repetition_tokens(output)
    maximum_width = min(16, len(tokens) // 4)
    for width in range(1, maximum_width + 1):
        for start in range(0, len(tokens) - (4 * width) + 1):
            block = tokens[start : start + width]
            if all(
                tokens[start + repeat * width : start + (repeat + 1) * width] == block
                for repeat in range(1, 4)
            ):
                return True

    if len(tokens) >= 32:
        fourgrams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
        if len(set(fourgrams)) / len(fourgrams) < 0.35:
            return True
    return False
