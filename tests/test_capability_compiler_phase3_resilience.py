import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3_resilience import KNOWN_DOMAINS, placement


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_RESILIENCE_PROTOCOL_V52.json"


@pytest.mark.skipif(not PROTOCOL.exists(), reason="V52 protocol not materialized yet")
def test_placement_fails_closed_for_unavailable_and_unknown_domains() -> None:
    result = placement(ROOT, PROTOCOL, ["python", "unknown-specialty"])
    assert result["status"] == "FAIL_CLOSED_UNSUPPORTED_DOMAIN"
    assert result["unsupported_selected_domains"] == ["unknown-specialty"]
    assert result["unknown_or_ambiguous_destination"] == "quarantine"
    python = next(row for row in result["domains"] if row["domain"] == "python")
    assert python["destination"] == "domain_cake:python"
    assert python["transfer_readiness"] == "BLOCKED_NO_DOMAIN_ACQUISITION_PAYLOAD"
    assert result["phase3_certified"] is False
    assert result["phase4_open"] is False


@pytest.mark.skipif(not PROTOCOL.exists(), reason="V52 protocol not materialized yet")
def test_english_placement_is_complete_and_domain_clean() -> None:
    result = placement(ROOT, PROTOCOL, [])
    assert result["status"] == "PASS_ENGLISH_PLACEMENT_ONLY"
    assert len(result["english_core"]) == 14
    assert sum(row["acquisition_records"] for row in result["english_core"]) == 7000
    assert all(row["destination"] == "english_core" for row in result["english_core"])
    assert {row["domain"] for row in result["domains"]} == set(KNOWN_DOMAINS)
    assert all(row["acquisition_records"] == 0 for row in result["domains"])
