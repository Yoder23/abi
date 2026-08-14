from abi.capability_compiler_phase4_b50_baseline_pack_verify import (
    FORMAT,
    prefix_per_stratum,
    rank_within_strata,
)


def test_verifier_format_is_independent():
    assert FORMAT == "abi-capability-compiler-phase4-b50-baseline-pack-verify/1"


def test_independent_rank_and_prefix_are_deterministic():
    rows = [
        {"capability": "a", "source_attempt_sha256": "2" * 64},
        {"capability": "a", "source_attempt_sha256": "1" * 64},
        {"capability": "b", "source_attempt_sha256": "3" * 64},
        {"capability": "b", "source_attempt_sha256": "4" * 64},
    ]
    ranked = rank_within_strata(
        rows, artifact="x", salt="salt", groups=("capability",)
    )
    selected = prefix_per_stratum(ranked, ("capability",), 1)
    assert len(selected) == 2
    assert {row["capability"] for row in selected} == {"a", "b"}
    assert ranked == rank_within_strata(
        list(reversed(rows)), artifact="x", salt="salt", groups=("capability",)
    )
