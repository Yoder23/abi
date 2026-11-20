"""Additive R115 publication custody; never supplies or relaxes scientific gates."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.structural_decode_r115 import run as r  # noqa: E402
from scripts import r115_hostile_audit as hostile_audit  # noqa: E402

OUT = ROOT / 'results/review_2026_11_20'
MANIFEST = OUT / 'r115_manifest.json'
STRICT = OUT / 'r115_strict_recomputation.json'
ARCHIVE = OUT / 'r115_evidence.zip'
RECEIPT = OUT / 'r115_archive.json'


def checked(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise RuntimeError(f'Missing or escaping publication path: {name}')
    return path


def package():
    if MANIFEST.exists() or STRICT.exists():
        raise RuntimeError('Preserve prior publication')
    started, stdout = time.monotonic(), io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result = r.verify()
    frozen = r.read(r.d.OUT / 'freeze.json')
    stage = 'scope' if result['verdict'] == 'SCOPE_PREREQUISITE_FAILED' else 'live'
    decision = r.d.OUT / stage / 'decision.json'
    hostile = r.read(OUT / 'r115_hostile.json')
    recomputed = hostile_audit.derive(result)
    if hostile != recomputed or not all(c['rejected'] for c in recomputed['cases']):
        raise RuntimeError('Missing/stale/unrecomputable hostile evidence')
    names = {n for n in frozen['files'] if not n.startswith('../')}
    names.update(r.relative(p) for p in r.d.OUT.rglob('*') if p.is_file())
    names.update(r.relative(p) for p in OUT.glob('r115-*.xml'))
    names.update(('scripts/package_r115_evidence.py', 'scripts/r115_hostile_audit.py', 'scripts/r115_error_audit.py',
                  'tests/test_r115_publication.py', 'docs/R115_STRUCTURAL_DECODING.md',
                  'docs/releases/R115_REVIEW.md', r.relative(OUT / 'r115_hostile.json'),
                  r.relative(OUT / 'r115_error_audit.json'),
                  r.relative(OUT / 'r115_tokenizer_development.json')))
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    if names - tracked:
        raise RuntimeError(f'Stage required custody files first: {sorted(names - tracked)}')
    for name in names:
        checked(name)
    r.write(STRICT, {'format': 'r115-strict-recomputation/1', 'elapsed_seconds': time.monotonic() - started,
        'entrypoint': 'experiments.structural_decode_r115.run.verify', 'stdout': stdout.getvalue(),
        'result_sha256': r.sha(decision), 'model_calls': 0,
        'scope': 'Local raw recomputation only, not new live or independent hardware execution'})
    names.add(r.relative(STRICT))
    parent = r.read(r.MANIFEST)
    lfs = set(subprocess.check_output(['git', 'lfs', 'ls-files', '-n'], cwd=ROOT, text=True).splitlines())
    r.write(MANIFEST, {'format': 'r115-public-manifest/1', 'tag': 'abi-research-r115-2026-11-20',
        'files': {n: {'sha256': r.sha(checked(n)), 'bytes': checked(n).stat().st_size} for n in sorted(names)},
        'host_inputs': {n: h for n, h in frozen['files'].items() if n.startswith('../')},
        'host_revision': parent['host_revision'], 'source_model': parent['source_model'],
        'source_revision': parent['source_revision'], 'source_files': frozen['source_files'],
        'packages': frozen['packages'], 'lfs_paths': sorted(lfs & names),
        'freeze_sha256': r.sha(r.d.OUT / 'freeze.json'), 'verdict': result['verdict'],
        'result_path': r.relative(decision), 'result_sha256': r.sha(decision),
        'scope': 'Custody index; not a self-contained source-weight archive or independent reproduction'})
    print(json.dumps({'manifest_sha256': r.sha(MANIFEST), 'files': len(names), 'verdict': result['verdict']}), flush=True)


def archive():
    if ARCHIVE.exists() or RECEIPT.exists():
        raise RuntimeError('Preserve prior archive')
    manifest = r.read(MANIFEST)
    names = sorted(n for n in manifest['files'] if n.startswith('results/structural_decode_r115/'))
    if not names:
        raise RuntimeError('Empty archive inventory')
    for name in names:
        path, entry = checked(name), manifest['files'][name]
        if r.sha(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise RuntimeError('Changed archive input')
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(checked(name), name)
    with zipfile.ZipFile(ARCHIVE) as bundle:
        if bundle.namelist() != names:
            raise RuntimeError('Archive membership mismatch')
        for name in names:
            if hashlib.sha256(bundle.read(name)).hexdigest() != manifest['files'][name]['sha256']:
                raise RuntimeError('Archive content mismatch')
    r.write(RECEIPT, {'format': 'r115-evidence-archive/1', 'archive': ARCHIVE.name,
        'bytes': ARCHIVE.stat().st_size, 'sha256': r.sha(ARCHIVE), 'members': names,
        'manifest_sha256': r.sha(MANIFEST),
        'scope': 'New R115 evidence only; all historical Git/LFS, host and source dependencies remain required'})
    print(json.dumps(r.read(RECEIPT)), flush=True)


def public_verify():
    """Download the three published assets and verify bytes and ZIP members."""
    folder = OUT / 'r115_public_download'
    receipt = OUT / 'r115_public_assets_verified.json'
    if folder.exists() or receipt.exists():
        raise RuntimeError('Preserve prior public verification')
    folder.mkdir()
    names = (MANIFEST.name, ARCHIVE.name, RECEIPT.name)
    argv = ['gh', 'release', 'download', 'abi-research-r115-2026-11-20',
            '--repo', 'Yoder23/abi', '--dir', str(folder)]
    for name in names:
        argv += ['--pattern', name]
    subprocess.run(argv, cwd=ROOT, check=True)
    observed = {}
    if {p.name for p in folder.iterdir()} != set(names):
        raise RuntimeError('Unexpected public asset membership')
    for name in names:
        downloaded, local = folder / name, OUT / name
        if r.sha(downloaded) != r.sha(local) or downloaded.stat().st_size != local.stat().st_size:
            raise RuntimeError('Public asset byte mismatch')
        observed[name] = {'sha256': r.sha(downloaded), 'bytes': downloaded.stat().st_size}
    index, archive_receipt = r.read(folder / MANIFEST.name), r.read(folder / RECEIPT.name)
    with zipfile.ZipFile(folder / ARCHIVE.name) as bundle:
        if bundle.namelist() != archive_receipt['members']:
            raise RuntimeError('Public ZIP member mismatch')
        for name in bundle.namelist():
            payload = bundle.read(name)
            entry = index['files'][name]
            if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise RuntimeError('Public ZIP content mismatch')
    r.write(receipt, {'format': 'r115-public-assets-verification/1', 'assets': observed,
        'verified_archive_members': len(archive_receipt['members']),
        'scope': 'Fresh public asset byte/member custody, not a clean-clone scientific or independent-hardware reproduction'})
    print(json.dumps(r.read(receipt)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('package', 'archive', 'public-verify'))
    args = parser.parse_args()
    {'package': package, 'archive': archive, 'public-verify': public_verify}[args.command]()
