"""Public-only R107 reconstruction: raw evidence and parameter custody, no inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'results/review_2026_11_13/r107_manifest.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    sys.path.insert(0, str(ROOT))
    from experiments.retention_intervention_r107 import core as c
    from experiments.retention_intervention_r107.complete import verify

    verify()
    frozen = c.binding()
    files = {p for p in frozen['files'] if not p.startswith('../')}
    files.update(p.relative_to(ROOT).as_posix() for p in c.OUT.rglob('*') if p.is_file())
    files.update(p.relative_to(ROOT).as_posix() for p in c.HERE.glob('*.*'))
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'results/review_2026_11_13').glob('r107-*.xml'))
    files.update(('scripts/review_r107_public.py', 'tests/test_r107_completed_verifier.py',
                  'docs/R107_RETENTION_INTERVENTION.md', 'docs/releases/R107_REVIEW.md'))
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    if files-tracked:
        raise RuntimeError(f'Untracked required assets: {sorted(files-tracked)}')
    lfs = set(subprocess.check_output(['git', 'lfs', 'ls-files', '-n'], cwd=ROOT, text=True).splitlines()) & files
    manifest = {'format': 'r107-public-manifest/1',
        'files': {p: {'sha256': sha(ROOT/p), 'bytes': (ROOT/p).stat().st_size} for p in sorted(files)},
        'host_inputs': {p: digest for p, digest in frozen['files'].items() if p.startswith('../')},
        'host_revision': frozen['host_revision'], 'source_model': 'distilgpt2',
        'source_revision': frozen['source_revision'], 'source_files': frozen['source_files'],
        'lfs_paths': sorted(lfs), 'packages': frozen['packages'],
        'tag': 'abi-research-r107-2026-11-13', 'decision_sha256': sha(c.OUT/'decision.json'),
        'complete_sha256': sha(c.OUT/'complete.json')}
    c.write(ROOT/MANIFEST, manifest)
    print(json.dumps({'assets': len(files), 'host_inputs': len(manifest['host_inputs']),
                      'lfs_paths': len(lfs), 'manifest_sha256': sha(ROOT/MANIFEST)}))


def review(args):
    root, output = args.root.resolve(), args.output.resolve()
    host = root.parent/'layercake_release'
    if output.exists() or any(output.is_relative_to(p) for p in (root, host)):
        raise RuntimeError('New receipt must be outside both checkouts')
    receipt = {'format': 'r107-public-reconstruction/1', 'commands': [], 'start_unix': time.time(),
               'scope': 'Same-machine reused public checkouts/cache; no live inference or independent hardware'}
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('PYTHONPATH', None)

    def command(argv, folder=root):
        result = subprocess.run(argv, cwd=folder, env=env, capture_output=True, text=True, timeout=1800)
        receipt['commands'].append({'command': argv, 'cwd': str(folder), 'returncode': result.returncode,
                                    'stdout': result.stdout, 'stderr': result.stderr})
        print(json.dumps({'command': argv, 'returncode': result.returncode}), flush=True)
        if result.returncode:
            raise RuntimeError('Reconstruction command failed')
        return result.stdout.strip()

    try:
        if sha(root/MANIFEST) != args.manifest_sha256:
            raise RuntimeError('Wrong public manifest')
        manifest = json.loads((root/MANIFEST).read_text(encoding='utf-8'))
        if sha(Path(__file__)) != manifest['files']['scripts/review_r107_public.py']['sha256']:
            raise RuntimeError('Wrong review helper')
        for folder, commit, remote in ((root, args.commit, 'https://github.com/Yoder23/abi.git'),
                                      (host, manifest['host_revision'], 'https://github.com/Yoder23/layercake.git')):
            if command(['git', 'rev-parse', 'HEAD'], folder) != commit or command(['git', 'status', '--porcelain'], folder):
                raise RuntimeError('Wrong revision or dirty checkout')
            if command(['git', 'remote', 'get-url', 'origin'], folder) != remote:
                raise RuntimeError('Wrong public remote')
        for relative, item in manifest['files'].items():
            path = (root/relative).resolve()
            if not path.is_relative_to(root) or sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
                raise RuntimeError(f'Missing/changed asset: {relative}')
        for relative, digest in manifest['host_inputs'].items():
            path = (root/relative).resolve()
            if not path.is_relative_to(host) or sha(path) != digest:
                raise RuntimeError(f'Changed host input: {relative}')
        command([sys.executable, '-m', 'experiments.retention_intervention_r107.complete'])
        command([sys.executable, '-m', 'experiments.retention_intervention_r107.describe'])
        command([sys.executable, '-m', 'pytest', '-q', 'tests/test_retention_intervention_r107.py', 'tests/test_r107_completed_verifier.py'])
        command([sys.executable, 'scripts/check_docs.py'])
        for folder in (root, host):
            if command(['git', 'status', '--porcelain'], folder):
                raise RuntimeError('Review changed source')
        receipt.update(verdict='R107_RAW_AND_PARAMETER_CUSTODY_RECOMPUTED', assets=len(manifest['files']),
            host_inputs=len(manifest['host_inputs']), lfs_paths=len(manifest['lfs_paths']),
            source_files=len(manifest['source_files']), manifest_sha256=sha(root/MANIFEST),
            decision_sha256=manifest['decision_sha256'], complete_sha256=manifest['complete_sha256'])
    except Exception as error:
        receipt.update(verdict='FAILED', error=repr(error))
        raise
    finally:
        receipt['end_unix'] = time.time()
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write('\n')
        print(json.dumps({'receipt_sha256': sha(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'review'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--commit')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory()
    else:
        if not all((args.commit, args.manifest_sha256, args.output)):
            parser.error('review requires commit, manifest SHA256 and an external output path')
        review(args)
