from abi.capability_compiler_phase4_v19_runtime_profile import FORMAT


def test_profile_format_is_versioned_and_nonpromotional():
    assert FORMAT == "abi-capability-compiler-phase4-v19-runtime-profile/1"
