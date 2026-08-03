from __future__ import annotations

from abi.capability_compiler_phase1_ir import (
    cross_split_near_duplicates,
    normalize_text,
    simhash64,
)


def test_normalization_is_bounded_and_idempotent():
    raw = "\nCafe\u0301  \r\nline\t\r\n"
    normalized, transformations = normalize_text(raw)
    assert normalized == "Café\nline"
    assert set(transformations) == {
        "unicode_nfc",
        "line_endings_lf",
        "strip_trailing_horizontal_whitespace_per_line",
        "strip_outer_blank_lines",
    }
    assert normalize_text(normalized) == (normalized, [])


def test_simhash_near_duplicate_detection_is_cross_split_only():
    rows = [
        {"probe_id": "a", "split": "search", "prompt": "one two three four five six seven"},
        {"probe_id": "b", "split": "final_test", "prompt": "one two three four five six seven"},
        {"probe_id": "c", "split": "search", "prompt": "unrelated alpha beta gamma delta epsilon"},
    ]
    collisions = cross_split_near_duplicates(rows)
    assert len(collisions) == 1
    assert collisions[0]["left_probe_id"] == "a"
    assert collisions[0]["right_probe_id"] == "b"
    assert simhash64(rows[0]["prompt"]) == simhash64(rows[1]["prompt"])
