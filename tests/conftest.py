"""Test-process hygiene for modules that exercise source-model tooling."""

from __future__ import annotations

import sys

import pytest


_SOURCE_TOOLING_TESTS = {
    "test_capability_compiler_phase3_macro_layer_pair0_fit.py",
    "test_capability_compiler_phase3_native_residual_span_oracle.py",
    "test_capability_compiler_phase3_routed_v16_trajectory_retargeting.py",
    "test_capability_compiler_phase3_routed_v16_trajectory_retargeting_replay.py",
}

_FRESH_RUNTIME_TESTS = {
    "test_capability_compiler_phase3_routed_v15_autonomous_screen_isolated.py",
    "test_capability_compiler_phase3_routed_v15_autonomous_screen_v16_replay.py",
}


def _drop_transformers_modules() -> None:
    for name in tuple(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _restore_source_runtime_import_isolation(request):
    """Do not leak optional teacher-runtime imports into later isolation tests."""
    if request.path.name in _FRESH_RUNTIME_TESTS:
        _drop_transformers_modules()
    before = set(sys.modules)
    yield
    if request.path.name not in _SOURCE_TOOLING_TESTS:
        return
    for name in set(sys.modules) - before:
        if name == "transformers" or name.startswith("transformers."):
            sys.modules.pop(name, None)
