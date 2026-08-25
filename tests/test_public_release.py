from __future__ import annotations

import json

import abi
from abi.__main__ import STATUS, main


def test_version_matches_release_metadata() -> None:
    assert abi.__version__ == "0.4.0a1"
    assert STATUS["version"] == abi.__version__
    assert STATUS["phase8_certified"] is False
    assert STATUS["release_certified"] is False


def test_self_check_is_deterministic_and_domain_free(capsys) -> None:
    assert main(["self-check"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(["self-check"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert first["status"] == "PASS"
    assert first["english_domain_labels"] == []


def test_status_json_is_machine_readable(capsys) -> None:
    assert main(["status", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["campaign_state"] == "READY_FOR_HUMAN_AND_INDEPENDENT_REVIEW"
    assert value["historical_campaign_state"] == "V1089"
    assert value["tested_runtime_portability"] == {
        "status": "PASS_STANDALONE_CAPABILITY_RUNTIME_WITH_CODEC_ADAPTERS",
        "host_environments_passing": 3,
        "host_environments_required": 3,
        "capability_cells_passing": 12,
        "capability_cells_required": 12,
    }
    assert value["external_human_preferences"] == {
        "complete": 0,
        "required": 21_000,
    }
    assert value["phase8_certified"] is False
    assert value["release_certified"] is False
    assert value["internal_readiness_gates"] == {"passed": 18, "required": 18}
