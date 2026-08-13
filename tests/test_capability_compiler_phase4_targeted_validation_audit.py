from abi import capability_compiler_phase4_targeted_validation_audit as subject


def test_rank_is_deterministic_and_salted() -> None:
    assert subject._rank("record", "salt") == subject._rank("record", "salt")
    assert subject._rank("record", "salt") != subject._rank("record", "other")
