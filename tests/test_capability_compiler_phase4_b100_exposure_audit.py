from abi.capability_compiler_phase4_b100_exposure_audit import PARENT_STAGES


def test_exposure_audit_scope_is_only_parent_training_stages():
    assert PARENT_STAGES == ("v443", "v459", "v463")
