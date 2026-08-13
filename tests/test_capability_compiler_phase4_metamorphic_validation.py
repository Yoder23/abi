from abi import capability_compiler_phase4_metamorphic_audit as audit
from abi import capability_compiler_phase4_metamorphic_validation as subject


def _protocol():
    return {
        "namespace_bases": [20_000 + 100_000 * index for index in range(10)],
        "families": 4,
        "samples_per_family_namespace": 10,
    }


def test_suite_is_balanced_unique_and_model_blind() -> None:
    rows = subject.build_rows(_protocol())
    assert len(rows) == 400
    assert len({row["ir_record_id"] for row in rows}) == 400
    assert len({row["normalized_generation_prompt"] for row in rows}) == 400
    assert {row["namespace"] for row in rows} == {f"N{20 + 100 * index:03d}" for index in range(10)}
    assert {row["family"] for row in rows} == {0, 1, 2, 3}
    assert all(not row["training_eligible"] and not row["teacher_output_present"] for row in rows)


def test_paired_statistics_are_directional() -> None:
    rows = [{"ir_record_id": str(index)} for index in range(20)]
    inherited = {str(index): index < 10 for index in range(20)}
    adapted = {str(index): True for index in range(20)}
    interval = audit._paired_prompt_bootstrap(inherited, adapted, rows, samples=1000, seed=17)
    assert interval["point"] == 0.5
    assert interval["lower_95"] > 0
    assert audit._binomial_one_sided(10, 0) < 0.05
    assert audit._binomial_one_sided(0, 10) == 1.0
