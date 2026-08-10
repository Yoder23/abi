from __future__ import annotations

import sys

from abi import capability_compiler_phase3_routed_v15_autonomous_screen_v16_replay as replay


def test_replay_import_is_source_runtime_isolated() -> None:
    assert "transformers" not in sys.modules
    assert replay.screen.FORMAT == "abi-capability-compiler-phase3-routed-v15-autonomous-screen-isolated/1"
