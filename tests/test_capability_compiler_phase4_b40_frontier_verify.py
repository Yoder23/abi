from abi.capability_compiler_phase4_b40_frontier_verify import (
    CRITICAL,
    FORMAT,
    SEEDS,
    SYSTEMS,
    quality_screen,
)


def test_frontier_contract_is_bounded_and_versioned():
    assert FORMAT == "abi-capability-compiler-phase4-b40-frontier-verify/1"
    assert SYSTEMS == ("ABI", "L0", "L1", "D0")
    assert SEEDS == (104729, 130363, 155921)
    assert CRITICAL == ("prompt_grounding", "instruction_following", "abstention")


def test_locked_quality_requires_every_capability_and_zero_collapse(monkeypatch):
    from abi import capability_compiler_phase4_b40_frontier_verify as frontier
    from abi.capability_compiler_phase2_common import CAPABILITIES

    probes = {}
    rows = []
    for capability in CAPABILITIES:
        for index in range(100):
            probe_id = f"{capability}-{index}"
            probes[probe_id] = {"probe_id": probe_id, "evaluator": {}}
            rows.append({"probe_id": probe_id, "capability": capability, "output": "pass"})
    monkeypatch.setattr(frontier, "evaluate_functional", lambda output, evaluator: output == "pass")
    monkeypatch.setattr(frontier, "evaluate_functional_v2", lambda output, evaluator, capability: output == "pass")
    monkeypatch.setattr(frontier, "repetition_collapse_v2", lambda output: output == "collapse")
    thresholds = {"per_capability_point": .9, "per_capability_lower": .85, "critical_point": .95, "critical_lower": .9}
    passed = quality_screen(rows, probes, thresholds)
    assert passed["passes_locked_absolute_quality"] is True
    rows[0]["output"] = "collapse"
    failed = quality_screen(rows, probes, thresholds)
    assert failed["passes_locked_absolute_quality"] is False


def test_one_critical_capability_at_75_percent_fails(monkeypatch):
    from abi import capability_compiler_phase4_b40_frontier_verify as frontier
    from abi.capability_compiler_phase2_common import CAPABILITIES

    probes = {}
    rows = []
    for capability in CAPABILITIES:
        for index in range(100):
            probe_id = f"{capability}-{index}"
            output = "fail" if capability == "abstention" and index >= 75 else "pass"
            probes[probe_id] = {"probe_id": probe_id, "evaluator": {}}
            rows.append({"probe_id": probe_id, "capability": capability, "output": output})
    monkeypatch.setattr(frontier, "evaluate_functional", lambda output, evaluator: output == "pass")
    monkeypatch.setattr(frontier, "evaluate_functional_v2", lambda output, evaluator, capability: output == "pass")
    monkeypatch.setattr(frontier, "repetition_collapse_v2", lambda output: False)
    thresholds = {"per_capability_point": .9, "per_capability_lower": .85, "critical_point": .95, "critical_lower": .9}
    result = quality_screen(rows, probes, thresholds)
    assert result["per_capability"]["abstention"]["passes_v1"] == 75
    assert result["gates"]["critical_capabilities"] is False
    assert result["passes_locked_absolute_quality"] is False
