from __future__ import annotations

import pytest

from abi.capability_compiler_phase2_common import CAPABILITIES
from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_routed_v15_autonomous_screen import controlled_prompt, expected_route


def test_expected_route_is_closed_and_exhaustive() -> None:
    assert {capability: expected_route(capability) for capability in CAPABILITIES} == {
        capability: capability if capability in {"abstention", "conversation"} else "generic"
        for capability in CAPABILITIES
    }
    with pytest.raises(Phase3Error):
        expected_route("python")


def test_controlled_prompt_retains_task_and_declares_capability() -> None:
    value = controlled_prompt("summarization", "Summarize: a short supplied passage.")
    assert value == "Capability route: summarization\nSummarize: a short supplied passage."
