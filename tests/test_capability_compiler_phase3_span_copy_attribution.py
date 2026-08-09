from collections import Counter
import re


def test_repeated_identical_piece_is_distinct_from_unique_pointer_policy():
    source = [b"ID_7", b"-", b"A", b"ID_7", b"-", b"B"]
    counts = Counter(source); eligible = re.compile(rb"^[A-Za-z0-9_]+$")
    assert counts[b"ID_7"] == 2
    assert eligible.fullmatch(b"ID_7") is not None
    assert not (counts[b"ID_7"] == 1)
    assert counts[b"ID_7"] > 1
