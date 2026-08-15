from abi.capability_compiler_phase4_b40_v25_runtime import _merge


def test_merge_changes_only_abi_system_and_v25_runtime_fields():
    base = {
        "systems": {"ABI": {"old": 1}, "L0": {"kept": 2}},
        "candidate_screen_protocol": "old",
        "x": 3,
    }
    overlay = {
        "systems": {"ABI": {"seed": 104729}},
        "v25_runtime_mode": "cpu",
        "v25_product_result": "product",
        "candidate_screen_protocol": "source",
    }
    merged = _merge(base, overlay)
    assert merged["systems"]["ABI"] == {"seed": 104729}
    assert merged["systems"]["L0"] == {"kept": 2}
    assert merged["runtime_interface"] == "lc-direct-neural-core/25"
    assert merged["candidate_screen_protocol"] == "source"
    assert base["systems"]["ABI"] == {"old": 1}
