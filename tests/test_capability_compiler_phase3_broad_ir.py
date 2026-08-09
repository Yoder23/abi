from collections import Counter

import pytest

from abi.capability_compiler_phase1_ir import CANONICAL_CAPABILITIES
from abi.capability_compiler_phase3_broad_ir import (
    BroadIRError,
    host_prompt_projection,
    select_candidates,
)


def test_targeted_prompt_projection_removes_only_search_wrapper() -> None:
    prompt = (
        "Complete this new bounded English practice task. Reference P3T-00-0000.\n"
        "Correct the agreement error and return one sentence only: Mira walk."
    )
    projected, policy = host_prompt_projection("grammar", prompt)
    assert projected == "Correct the agreement error and return one sentence only: Mira walk."
    assert policy == "phase1_task_body_without_targeted_search_wrapper"


def test_targeted_prompt_projection_rejects_empty_body() -> None:
    with pytest.raises(BroadIRError, match="complete task"):
        host_prompt_projection(
            "grammar", "Complete this new bounded English practice task. Reference P3T-00-0000.\n"
        )


def _fixture(per_capability: int):
    probes, journal = [], {}
    mapping = {}
    for cap_index, capability in enumerate(CANONICAL_CAPABILITIES):
        raw = f"raw-{cap_index}"
        mapping[raw] = capability
        for row_index in range(per_capability + 1):
            probe_id = f"{raw}-{row_index}"
            probes.append({"probe_id": probe_id, "capability": raw})
            attempt = {
                "probe_id": probe_id,
                "attempt_index": 0,
                "attempt_sha256": f"{cap_index:02d}{row_index:062d}",
                "canonical_capability": capability,
                "functional_pass": True,
            }
            journal[(probe_id, 0)] = attempt
    return probes, journal, mapping


def test_selection_is_fixed_depth_balanced_and_deterministic() -> None:
    probes, journal, mapping = _fixture(2)
    first, rejected = select_candidates(
        probes, journal, mapping, source_protocol_sha256="a" * 64, per_capability=2
    )
    second, _ = select_candidates(
        reversed(probes), journal, mapping, source_protocol_sha256="a" * 64, per_capability=2
    )
    assert Counter(row["capability"] for row in first) == {cap: 2 for cap in CANONICAL_CAPABILITIES}
    assert [row["selection_key"] for row in first] == [row["selection_key"] for row in second]
    assert len(rejected) == len(CANONICAL_CAPABILITIES)


def test_selection_fails_closed_below_depth() -> None:
    probes, journal, mapping = _fixture(0)
    failed_id = probes[0]["probe_id"]
    journal[(failed_id, 0)]["functional_pass"] = False
    with pytest.raises(BroadIRError, match="insufficient eligible records"):
        select_candidates(
            probes, journal, mapping, source_protocol_sha256="a" * 64, per_capability=1
        )


def test_selection_can_be_bounded_to_source_owned_capabilities() -> None:
    probes, journal, mapping = _fixture(1)
    selected, _ = select_candidates(
        probes,
        journal,
        mapping,
        source_protocol_sha256="a" * 64,
        per_capability=1,
        capabilities=["abstention"],
    )
    assert len(selected) == 1
    assert selected[0]["capability"] == "abstention"


def test_fluent_projection_keeps_exact_event_fields_and_drops_redundant_context() -> None:
    prompt = (
        "<fictional_context>\nLong redundant story.\n</fictional_context>\n"
        "Use the context only to disambiguate the fields. Turn the following fields.\n"
        "event_one=First event.\nevent_two=Second event."
    )
    projected, method = host_prompt_projection("fluent_realization", prompt)
    assert "Long redundant story" not in projected
    assert "event_one=First event." in projected
    assert "event_two=Second event." in projected
    assert method == "fluent_realization_event_fields_without_redundant_context"
