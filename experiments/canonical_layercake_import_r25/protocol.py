"""Frozen R25 constants and source preparation."""

from experiments.layercake_composition_r24.protocol import (
    NAMESPACES,
    domain_slug,
    prepared_rows,
    source_splits,
)
from experiments.layercake_composition_r24.protocol import (
    SEEDS as R24_SEEDS,
)

HOST_INITIALIZATIONS = (25021, 25022, 25023)
BUILD_INITIALIZATIONS = (25101, 25102, 25103)

__all__ = [
    "BUILD_INITIALIZATIONS",
    "HOST_INITIALIZATIONS",
    "NAMESPACES",
    "R24_SEEDS",
    "domain_slug",
    "prepared_rows",
    "source_splits",
]
