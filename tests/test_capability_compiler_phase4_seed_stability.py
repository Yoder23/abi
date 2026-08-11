from abi.capability_compiler_phase4_seed_stability import analyze


def _eval(total, coherence, *, final):
    pass_key = "passes_v1" if final else "passes"
    collapse_key = "collapses_v2" if final else "v2_collapses"
    total_key = "functional_passes_v1" if final else "functional_passes"
    return {
        total_key: total,
        "repetition_collapses_v2": 0,
        "per_capability": {
            "coherence": {pass_key: coherence, collapse_key: 0},
            "grammar": {pass_key: 100, collapse_key: 0},
        },
    }


def test_analysis_requires_all_seed_boundary(monkeypatch, tmp_path):
    protocol = {
        "runs": [
            {"budget": budget, "seed": seed, "result": {}, "intermediate_evaluation": {}, "final_evaluation": {}}
            for budget in ("B40", "B80")
            for seed in (1, 2, 3)
        ],
        "tested_boundary": {"passing_budget": "B80", "adjacent_lower": "B40"},
        "self_path": "protocol.json",
    }
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    docs = {}
    for row in protocol["runs"]:
        budget, seed = row["budget"], row["seed"]
        passed = budget == "B80" and seed != 2
        docs[(budget, seed)] = (
            {
                "budget": {"id": budget}, "seed": seed,
                "stage_checkpoints": {"v463": f"m-{budget}-{seed}", "v526": f"f-{budget}-{seed}"},
                "final_test_accessed": False,
                "gates": {"a": passed},
                "status": "PASS_PHASE4_ABI_BUDGET_MACHINE_GATES" if passed else "FAIL_PHASE4_ABI_BUDGET_MACHINE_GATES",
                "teacher_comparison_v1": {"lower_95": 0.1}, "wall_seconds": 1,
            },
            {**_eval(1200, 20 + seed, final=False), "checkpoint_sha256": f"m-{budget}-{seed}", "final_test_accessed": False},
            {**_eval(1300, 90 - seed, final=True), "checkpoint_sha256": f"f-{budget}-{seed}", "final_test_accessed": False},
        )
    def fake_load(_root, row, key):
        index = {"result": 0, "intermediate_evaluation": 1, "final_evaluation": 2}[key]
        return tmp_path / "protocol.json", docs[(row["budget"], row["seed"])][index]
    monkeypatch.setattr("abi.capability_compiler_phase4_seed_stability._load_bound", fake_load)
    result = analyze(tmp_path, protocol)
    assert result["status"] == "FAIL_NO_REPRODUCED_ABI_FRONTIER"
    assert result["frontier_reproduced"] is False
    assert result["budgets"]["B80"]["pass_count"] == 2
    assert result["training_performed"] is False
