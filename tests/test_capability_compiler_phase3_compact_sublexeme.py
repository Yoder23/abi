from abi.capability_compiler_phase3_compact_sublexeme import segment

def test_segment_minimizes_actions_deterministically():
    assert segment(b"marketing", {bytes((c,)) for c in b"marketing"} | {b"market", b"ing"}) == [b"market", b"ing"]
