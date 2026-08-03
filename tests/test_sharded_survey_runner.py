from __future__ import annotations

from pathlib import Path

from abi.sharded_survey_runner import build_survey_command
from abi.sharded_survey_runner import ShardedSurveyError
import pytest


def test_sharded_command_is_cuda_search_only_and_offline_by_default() -> None:
    command = build_survey_command(
        catalog=Path("catalog.json"),
        output=Path("output.abix"),
        model="teacher",
        revision="revision",
        license_id="mit",
        batch_size=8,
    )
    assert command[0]
    assert "cuda" in command
    assert "search" in command
    assert "--development" in command
    assert "--allow-network" not in command
    assert "cpu" not in command
    assert command[command.index("--batch-size") + 1] == "8"


def test_sharded_command_can_lock_search_and_validation_without_development():
    command = build_survey_command(
        catalog=Path("catalog.json"),
        output=Path("output.abix"),
        model="teacher",
        revision="revision",
        license_id="mit",
        batch_size=4,
        splits=("validation", "search"),
        development=False,
    )
    assert command[command.index("--splits") + 1] == "search,validation"
    assert "--development" not in command
    with pytest.raises(ShardedSurveyError, match="never final_test"):
        build_survey_command(
            catalog=Path("catalog.json"),
            output=Path("output.abix"),
            model="teacher",
            revision="revision",
            license_id="mit",
            batch_size=4,
            splits=("final_test",),
        )
