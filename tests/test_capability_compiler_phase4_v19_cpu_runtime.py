from abi.capability_compiler_phase4_v19_cpu_runtime import paired_quality_bootstrap, select_prospective


def test_select_prospective_has_100_distinct_plus_20_repeats():
    rows = [{"ir_record_id": f"r-{index:03d}"} for index in range(140)]
    distinct, scheduled = select_prospective(rows, 100, 20)
    assert len(distinct) == 100
    assert len(scheduled) == 120
    assert scheduled[100:] == distinct[:20]


def test_paired_quality_bootstrap_preserves_exact_difference():
    result = paired_quality_bootstrap([True] * 120, [False] * 120, 200, 17)
    assert result["point"] == 1.0
    assert result["lower_95"] == 1.0
