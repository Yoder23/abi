"""Offline publication custody checks; no model or source-cache execution."""
import pytest

from scripts import package_r115_evidence as p


def test_missing_and_escaping_paths_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    for name in ('missing', '../outside'):
        with pytest.raises(RuntimeError):
            p.checked(name)


def test_existing_manifest_prevents_overwrite(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    monkeypatch.setattr(p, 'MANIFEST', path)
    with pytest.raises(RuntimeError, match='Preserve'):
        p.package()


def test_empty_or_changed_archive_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'MANIFEST', tmp_path / 'manifest.json')
    monkeypatch.setattr(p, 'ARCHIVE', tmp_path / 'archive.zip')
    monkeypatch.setattr(p, 'RECEIPT', tmp_path / 'receipt.json')
    p.r.write(p.MANIFEST, {'files': {}})
    with pytest.raises(RuntimeError, match='Empty'):
        p.archive()


def test_changed_archive_bytes_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'MANIFEST', tmp_path / 'manifest.json')
    monkeypatch.setattr(p, 'ARCHIVE', tmp_path / 'archive.zip')
    monkeypatch.setattr(p, 'RECEIPT', tmp_path / 'receipt.json')
    file = tmp_path / 'results/structural_decode_r115/test.json'
    file.parent.mkdir(parents=True)
    file.write_text('{}')
    p.r.write(p.MANIFEST, {'files': {'results/structural_decode_r115/test.json': {'sha256': '0' * 64, 'bytes': 2}}})
    with pytest.raises(RuntimeError, match='Changed'):
        p.archive()


def test_public_verification_preserves_prior_download(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUT', tmp_path)
    (tmp_path / 'r115_public_download').mkdir()
    with pytest.raises(RuntimeError, match='Preserve'):
        p.public_verify()


def test_public_verification_rejects_missing_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUT', tmp_path)
    monkeypatch.setattr(p.subprocess, 'run', lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match='asset membership'):
        p.public_verify()
