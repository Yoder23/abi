"""Additive R112 custody tests; not experimental qualification."""
import hashlib
import json

import pytest

from scripts import package_r112_evidence as p


@pytest.mark.parametrize("existing", ["MANIFEST", "RECEIPT"])
def test_existing_packet_is_preserved(tmp_path, monkeypatch, existing):
    monkeypatch.setattr(p, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(p, "RECEIPT", tmp_path / "receipt.json")
    getattr(p, existing).write_bytes(b"preserved")
    with pytest.raises(RuntimeError, match="Preserve the existing"):
        p.package()
    assert getattr(p, existing).read_bytes() == b"preserved"


def test_publication_paths_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="Missing or escaping"):
        p.checked_path("../outside")
    with pytest.raises(RuntimeError, match="Missing or escaping"):
        p.checked_path("missing")


def archive_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    monkeypatch.setattr(p, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(p, "ARCHIVE", tmp_path / "evidence.zip")
    monkeypatch.setattr(p, "ARCHIVE_RECEIPT", tmp_path / "archive.json")
    name = "results/count_feedback_r112/example.json"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"example":true}\n')
    p.write_new(p.MANIFEST, {"files": {name: {"sha256": p.sha(path), "bytes": path.stat().st_size}}})
    return path


def test_archive_exact_members_and_write_once(tmp_path, monkeypatch):
    path = archive_fixture(tmp_path, monkeypatch)
    p.archive()
    receipt = json.loads(p.ARCHIVE_RECEIPT.read_text())
    assert receipt["members"] == [path.relative_to(tmp_path).as_posix()]
    assert receipt["sha256"] == hashlib.sha256(p.ARCHIVE.read_bytes()).hexdigest()
    original = p.ARCHIVE.read_bytes()
    with pytest.raises(RuntimeError, match="Preserve the existing"):
        p.archive()
    assert p.ARCHIVE.read_bytes() == original


@pytest.mark.parametrize("mutation", ["missing", "bytes", "digest"])
def test_archive_rejects_bad_input_before_creation(tmp_path, monkeypatch, mutation):
    path = archive_fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        path.unlink()
    elif mutation == "bytes":
        path.write_bytes(b"different length")
    else:
        path.write_bytes(path.read_bytes().replace(b"true", b"null"))
    with pytest.raises(RuntimeError):
        p.archive()
    assert not p.ARCHIVE.exists()
    assert not p.ARCHIVE_RECEIPT.exists()
