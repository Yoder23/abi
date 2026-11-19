"""Publish R112's incomplete packet; never replace the complete-study verifier."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import time
import zipfile

from scripts import package_r112_evidence as p
from scripts import r112_interruption as audit

MANIFEST = p.OUT / 'r112_interruption_manifest.json'
RECEIPT = p.OUT / 'r112_interruption_recomputation.json'
ARCHIVE = p.OUT / 'r112_interruption_evidence.zip'
ARCHIVE_RECEIPT = p.OUT / 'r112_interruption_archive.json'


def package():
    if MANIFEST.exists() or RECEIPT.exists():
        raise RuntimeError('Preserve existing interruption publication')
    started, transcript = time.monotonic(), io.StringIO()
    with contextlib.redirect_stdout(transcript):
        result = audit.verify()
    frozen = audit.r.read(audit.d.OUT / 'freeze.json')
    names = {n for n in frozen['files'] if not n.startswith('../')}
    names.update(x.relative_to(p.ROOT).as_posix() for x in audit.d.OUT.rglob('*') if x.is_file())
    names.update(('scripts/package_r112_interruption.py', 'scripts/package_r112_evidence.py',
                  'tests/test_r112_interruption.py', 'tests/test_r112_publication.py',
                  'tests/test_r112_interruption_publication.py',
                  'docs/R112_CLEANUP.md', 'docs/R112_COUNT_FEEDBACK.md', 'docs/releases/R112_REVIEW.md'))
    names.update(x.relative_to(p.ROOT).as_posix() for x in (p.ROOT / 'scripts').glob('r112_*') if x.is_file())
    for pattern in ('r112-*.xml', 'r112_hardlink*', 'r112_*duplicate_audit.json'):
        names.update(x.relative_to(p.ROOT).as_posix() for x in p.OUT.glob(pattern) if x.is_file())
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=p.ROOT, text=True).splitlines())
    if names - tracked:
        raise RuntimeError(f'Stage publication files first: {sorted(names - tracked)}')
    for name in names:
        p.checked_path(name)
    p.write_new(RECEIPT, {
        'format': 'r112-interruption-recomputation/1',
        'entrypoint': 'scripts.r112_interruption.verify',
        'elapsed_seconds': time.monotonic() - started,
        'stdout': transcript.getvalue(), 'model_calls': 0,
        'result_sha256': p.sha(audit.d.OUT / 'interruption.json'),
        'scope': 'Resource stop and prefix custody only; not completed-study certification',
    })
    names.add(RECEIPT.relative_to(p.ROOT).as_posix())
    lfs = set(subprocess.check_output(['git', 'lfs', 'ls-files', '-n'], cwd=p.ROOT, text=True).splitlines())
    p.write_new(MANIFEST, {
        'format': 'r112-interruption-manifest/1', 'tag': 'abi-research-r112-2026-11-19',
        'verdict': result['verdict'],
        'files': {n: {'sha256': p.sha(p.checked_path(n)), 'bytes': p.checked_path(n).stat().st_size} for n in sorted(names)},
        'host_inputs': {n: h for n, h in frozen['files'].items() if n.startswith('../')},
        'host_revision': audit.r.old.h.HOST_REVISION,
        'source_model': audit.r.old.h.SOURCE, 'source_revision': audit.r.old.h.REVISION,
        'source_files': frozen['source_files'], 'packages': frozen['packages'],
        'lfs_paths': sorted(lfs & names),
        'result_path': 'results/count_feedback_r112/interruption.json',
        'result_sha256': p.sha(audit.d.OUT / 'interruption.json'),
        'freeze_sha256': p.sha(audit.d.OUT / 'freeze.json'),
        'scope': 'Incomplete evidence custody; historical Git/LFS, host and source cache remain required',
    })
    print(json.dumps({'manifest_sha256': p.sha(MANIFEST), 'files': len(names), 'verdict': result['verdict']}))


def archive():
    if ARCHIVE.exists() or ARCHIVE_RECEIPT.exists():
        raise RuntimeError('Preserve existing interruption archive')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    names = sorted(n for n in manifest['files'] if n.startswith('results/count_feedback_r112/'))
    if not names:
        raise RuntimeError('Missing evidence inventory')
    for name in names:
        path = p.checked_path(name)
        if p.sha(path) != manifest['files'][name]['sha256'] or path.stat().st_size != manifest['files'][name]['bytes']:
            raise RuntimeError('Changed archive input')
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(p.checked_path(name), name)
    with zipfile.ZipFile(ARCHIVE) as bundle:
        if bundle.namelist() != names or any(hashlib.sha256(bundle.read(n)).hexdigest() != manifest['files'][n]['sha256'] for n in names):
            raise RuntimeError('Archive member mismatch')
    p.write_new(ARCHIVE_RECEIPT, {
        'format': 'r112-interruption-archive/1', 'archive': ARCHIVE.name,
        'sha256': p.sha(ARCHIVE), 'bytes': ARCHIVE.stat().st_size, 'members': names,
        'manifest_sha256': p.sha(MANIFEST),
        'scope': 'New incomplete R112 evidence only, not source weights or independent reproduction',
    })
    print(json.dumps({'archive_sha256': p.sha(ARCHIVE), 'members': len(names)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('package', 'archive'))
    {'package': package, 'archive': archive}[parser.parse_args().command]()
