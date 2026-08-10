from __future__ import annotations

import sys

from abi.capability_compiler_phase2_common import CAPABILITIES
from abi.capability_compiler_phase3_routed_v15_autonomous_screen_isolated import controlled_prompt, expected_route, paired_stratified_bootstrap, wilson


def test_import_isolation_and_prompt_semantics() -> None:
    assert "transformers" not in sys.modules
    assert controlled_prompt("grammar", "line one\nline two") == "Capability route: grammar\nline two"
    assert sum(expected_route(value) == "generic" for value in CAPABILITIES) == 12


def test_statistics_match_locked_definitions() -> None:
    interval = wilson(100, 100)
    assert interval["point"] == 1.0 and 0.96 < interval["lower_95"] < 0.97
    rows = [{"capability": capability, "candidate_pass": True, "teacher_pass": True} for capability in CAPABILITIES for _ in range(100)]
    result = paired_stratified_bootstrap(rows, replicates=100, seed=240075)
    assert result["candidate_minus_teacher"] == result["lower_95"] == result["upper_95"] == 0.0
