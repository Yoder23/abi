"""Incomplete publication stays write-once and rejects missing/changed bytes."""
import json

import pytest

from scripts import package_r112_interruption as p


@pytest.mark.parametrize('existing', ['MANIFEST', 'RECEIPT'])
def test_preserve_existing_packet(tmp_path, monkeypatch, existing):
    for name in ('MANIFEST', 'RECEIPT'):
        monkeypatch.setattr(p, name, tmp_path / name)
    getattr(p, existing).write_bytes(b'preserved')
    with pytest.raises(RuntimeError, match='Preserve existing'):
        p.package()
    assert getattr(p, existing).read_bytes() == b'preserved'


@pytest.mark.parametrize('mutation', ['none', 'missing', 'digest', 'size'])
def test_exact_archive_or_fail_closed(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(p.p, 'ROOT', tmp_path)
    for name in ('MANIFEST', 'ARCHIVE', 'ARCHIVE_RECEIPT'):
        monkeypatch.setattr(p, name, tmp_path / name)
    relative = 'results/count_feedback_r112/raw.json'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b'original')
    p.p.write_new(p.MANIFEST, {'files': {relative: {'sha256': p.p.sha(path), 'bytes': path.stat().st_size}}})
    if mutation == 'missing':
        path.unlink()
    elif mutation == 'digest':
        path.write_bytes(b'differnt')
    elif mutation == 'size':
        path.write_bytes(b'short')
    if mutation != 'none':
        with pytest.raises(RuntimeError):
            p.archive()
        assert not p.ARCHIVE.exists()
    else:
        p.archive()
        receipt = json.loads(p.ARCHIVE_RECEIPT.read_text())
        assert receipt['members'] == [relative]
        assert receipt['sha256'] == p.p.sha(p.ARCHIVE)
        with pytest.raises(RuntimeError, match='Preserve existing'):
            p.archive()
