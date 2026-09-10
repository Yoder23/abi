"""Generic R15B decoder for anonymous pre-answer source representations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from experiments.foreign_neural_state_r15.frontend import transition_from_labels


class RepresentationError(RuntimeError):
    """Raised when an R15B representation bundle is invalid."""


def representation_logits(residuals: torch.Tensor, output_rows: torch.Tensor) -> torch.Tensor:
    """Project six anonymous residuals through eight frozen source output rows."""
    if (
        residuals.ndim != 3
        or tuple(residuals.shape[:2]) != (3, 2)
        or output_rows.ndim != 2
        or output_rows.shape[0] != 8
        or residuals.shape[2] != output_rows.shape[1]
        or not torch.isfinite(residuals).all()
        or not torch.isfinite(output_rows).all()
    ):
        raise RepresentationError("anonymous representation bundle contract changed")
    return torch.einsum("osd,vd->osv", residuals.float(), output_rows.float())


def decode_labels(residuals: torch.Tensor, output_rows: torch.Tensor) -> list[int]:
    logits = representation_logits(residuals, output_rows)
    labels = logits.argmax(dim=-1).flatten().tolist()
    transition_from_labels(labels)
    return [int(value) for value in labels]


def decode_transition(residuals: torch.Tensor, output_rows: torch.Tensor) -> torch.Tensor:
    return transition_from_labels(decode_labels(residuals, output_rows))


def labels_to_operations(labels: Sequence[int]) -> list[tuple[int, int]]:
    if len(labels) != 6:
        raise RepresentationError("affine label count changed")
    operations = []
    for operator in range(3):
        at_zero = int(labels[2 * operator])
        at_one = int(labels[2 * operator + 1])
        multiplier = (at_one - at_zero) % 8
        if multiplier not in {1, 3, 5, 7}:
            raise RepresentationError("decoded labels are not affine permutations")
        operations.append((multiplier, at_zero))
    return operations
