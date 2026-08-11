from abi.capability_compiler_phase3_guarded_screen_verify import _must_reject
from abi.capability_compiler_phase3 import Phase3Error


def test_must_reject_accepts_only_phase3_error():
    assert _must_reject("mutation", lambda: (_ for _ in ()).throw(Phase3Error("rejected"))) == "mutation"
