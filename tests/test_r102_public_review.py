"""Fail-closed inventory checks for the additive publication helper."""

import pytest

from scripts.review_r102_public import sha, verify_files


def test_inventory_exact_bytes(tmp_path):
    payload = tmp_path / "payload"
    payload.write_bytes(b"example")
    verify_files(tmp_path, {"payload": {"sha256": sha(payload), "bytes": 7}})


@pytest.mark.parametrize("mutation", ["missing", "hash", "size", "empty", "external"])
def test_inventory_fails_closed(tmp_path, mutation):
    root = tmp_path / "repo"
    root.mkdir()
    payload = root / "payload"
    payload.write_bytes(b"example")
    files = {"payload": {"sha256": sha(payload), "bytes": 7}}
    if mutation == "missing":
        files["absent"] = files.pop("payload")
    elif mutation == "hash":
        files["payload"]["sha256"] = "0" * 64
    elif mutation == "size":
        files["payload"]["bytes"] = 6
    elif mutation == "empty":
        files = {}
    else:
        outside = tmp_path / "outside"
        outside.write_bytes(b"example")
        files["../outside"] = files.pop("payload")
    with pytest.raises(RuntimeError):
        verify_files(root, files)
