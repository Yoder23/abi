from abi.capability_compiler_phase3_cpu_runtime import paired_ratio_bootstrap, select_runtime_probes


def test_runtime_selection_depth_and_repeats():
    capabilities = ("a", "b")
    rows = [{"probe_id": f"{capability}-{index}", "canonical_capability": capability} for index in range(100) for capability in capabilities]
    import abi.capability_compiler_phase3_cpu_runtime as module
    original = module.CAPABILITIES; module.CAPABILITIES = capabilities
    try:
        distinct, all_rows = select_runtime_probes(rows, 100, 20)
    finally:
        module.CAPABILITIES = original
    assert len(distinct) == 100 and len(all_rows) == 120 and len({row["probe_id"] for row in distinct}) == 100


def test_paired_bootstrap_is_deterministic():
    value = paired_ratio_bootstrap([4.0] * 20, [2.0] * 20, 100, 7)
    assert value["lower_95"] == 2.0 == value["upper_95"]
