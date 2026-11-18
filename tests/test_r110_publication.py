"""Publication custody checks, not experimental qualification tests."""
import hashlib
import json

import pytest

from scripts import package_r110_evidence as publication


def test_sha_matches_payload(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"R110\x00custody\n")
    assert publication.sha(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_receipt_is_write_once(tmp_path):
    path = tmp_path / "receipt.json"
    publication.write_new(path, {"b": 2, "a": 1})
    original = path.read_bytes()
    assert json.loads(original) == {"a": 1, "b": 2}
    assert original.endswith(b"\n")
    with pytest.raises(FileExistsError):
        publication.write_new(path, {"changed": True})
    assert path.read_bytes() == original


@pytest.mark.parametrize("existing", ["MANIFEST", "RECEIPT", "DIAGNOSTIC"])
def test_existing_packet_stops_before_scientific_verification(tmp_path, monkeypatch, existing):
    monkeypatch.setattr(publication, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(publication, "RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(publication, "DIAGNOSTIC", tmp_path / "diagnostic.json")
    getattr(publication, existing).write_bytes(b"preserved")
    with pytest.raises(RuntimeError, match="Preserve the existing"):
        publication.package()
    assert getattr(publication, existing).read_bytes() == b"preserved"
