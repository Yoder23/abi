from __future__ import annotations

import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase1 import Phase1ProtocolError, verify_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json"


def test_phase1_protocol_passes():
    result = verify_protocol(PROTOCOL)
    assert result["status"] == "PASS"
    assert result["search_prompts"] == 9_800
    assert result["development_prompts"] == 1_400
    assert result["final_prompts"] == 1_400
    assert result["domain_isolation_prompts"] == 400
    assert result["adversarial_prompts"] == 700
    assert result["training_authorized"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("record_acceptance", "minimum_selected_per_canonical_capability"), 499),
        (("repair_policy", "maximum_rounds"), 2),
        (("source", "generation", "length_terminated_records_eligible"), True),
        (("split_and_contamination", "final_used_for_normalization_or_selection"), True),
        (("declared_domain_reference", "selected_domain_acquisition_records"), 1),
    ],
)
def test_phase1_protocol_fails_closed_on_gate_tampering(tmp_path, path, value):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cursor = protocol
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    candidate = tmp_path / "ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json"
    candidate.write_text(json.dumps(protocol), encoding="utf-8")
    # Resolve all bound files through a temporary copy of the repository-facing
    # paths so failure attribution reaches the mutated gate rather than I/O.
    for binding in (
        protocol["phase0_protocol"],
        protocol["catalog"],
        protocol["catalog"]["generator"],
        protocol["declared_domain_reference"]["source_bundle"],
        protocol["declared_domain_reference"]["labeling_certificate"],
    ):
        destination = tmp_path / binding["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / binding["path"]).read_bytes())
    with pytest.raises(Phase1ProtocolError):
        verify_protocol(candidate)
