"""Inventory/recompute R106 from public siblings; never execute model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    sys.path.insert(0, str(root))
    from experiments.grammar_interface_r106 import core as c
    from experiments.grammar_interface_r106.run import verify

    verify()
    frozen = c.binding()
    files = {p for p in frozen['files'] if not p.startswith('../')}
    files.update(p.relative_to(root).as_posix() for p in c.OUT.rglob('*') if p.is_file())
    files.update(p.relative_to(root).as_posix() for p in (root/'results/review_2026_11_11').glob('r106-*.xml'))
    files.update(('scripts/review_r106_public.py', 'tests/test_r106_completed_verifier.py',
                  'experiments/grammar_interface_r106/describe.py', 'results/review_2026_11_11/r106_operations.md'))
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines())
    if files-tracked:
        raise RuntimeError(f'Untracked required assets: {sorted(files-tracked)}')
    lfs = set(subprocess.check_output(['git', 'lfs', 'ls-files', '-n'], cwd=root, text=True).splitlines()) & files
    value = {'format': 'r106-public-manifest/1',
        'files': {p: {'sha256': sha(root/p), 'bytes': (root/p).stat().st_size} for p in sorted(files)},
        'external_host_inputs': {p: v for p, v in frozen['files'].items() if p.startswith('../')},
        'sibling_host_commit': frozen['host_checkout'], 'lfs_paths': sorted(lfs),
        'control_source': {'model': frozen['control_model'], 'revision': frozen['control_revision'], 'files': frozen['control_files']},
        'packages': frozen['packages'], 'source_tag': 'abi-research-r106-2026-11-11',
        'freeze_sha256': sha(c.OUT/'freeze.json'), 'decision_sha256': sha(c.OUT/'decision.json'),
        'scope': 'Failed grammatical-discrimination control. No recipient or autonomous generation result.'}
    path = root/'results/review_2026_11_11/r106_manifest.json'
    c.write_once(path, value)
    print(json.dumps({'assets': len(files), 'lfs_paths': len(lfs), 'host_inputs': len(value['external_host_inputs']),
                      'manifest_sha256': sha(path)}), flush=True)


def review(args):
    root, output = args.root.resolve(), args.output.resolve()
    host = root.parent/'layercake_release'
    if output.exists() or any(output.is_relative_to(p) for p in (root, host)):
        raise RuntimeError('Use a new output outside both checkouts')
    receipt = {'format': 'r106-public-reconstruction/1', 'commands': [], 'start_unix': time.time(),
        'scope': 'Same-machine public evidence recomputation; no live inference or independent hardware'}
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('PYTHONPATH', None)

    def command(argv, folder=root):
        done = subprocess.run(argv, cwd=folder, env=env, capture_output=True, text=True, timeout=1800)
        receipt['commands'].append({'command': argv, 'cwd': str(folder), 'returncode': done.returncode,
                                    'stdout': done.stdout, 'stderr': done.stderr})
        print(json.dumps({'command': argv, 'returncode': done.returncode}), flush=True)
        if done.returncode:
            raise RuntimeError('Public reconstruction command failed')
        return done.stdout.strip()

    try:
        path = root/'results/review_2026_11_11/r106_manifest.json'
        if sha(path) != args.manifest_sha256:
            raise RuntimeError('Wrong manifest')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if sha(Path(__file__)) != manifest['files']['scripts/review_r106_public.py']['sha256']:
            raise RuntimeError('Wrong helper')
        for folder, commit, remote in ((root, args.commit, 'https://github.com/Yoder23/abi.git'),
                                      (host, manifest['sibling_host_commit'], 'https://github.com/Yoder23/layercake.git')):
            if command(['git', 'rev-parse', 'HEAD'], folder) != commit or command(['git', 'status', '--porcelain=v1'], folder):
                raise RuntimeError('Wrong revision or dirty checkout')
            if command(['git', 'remote', 'get-url', 'origin'], folder) != remote:
                raise RuntimeError('Wrong public repository')
        for relative, item in manifest['files'].items():
            target = (root/relative).resolve()
            if not target.is_relative_to(root) or not target.is_file() or sha(target) != item['sha256'] or target.stat().st_size != item['bytes']:
                raise RuntimeError(f'Missing/changed asset: {relative}')
        for relative, digest in manifest['external_host_inputs'].items():
            target = (root/relative).resolve()
            if not target.is_relative_to(host) or sha(target) != digest:
                raise RuntimeError(f'Wrong host source: {relative}')
        if args.fetch_control:
            source = manifest['control_source']
            env['HF_HUB_OFFLINE'] = '0'
            command([sys.executable, '-c', 'import sys,json; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=json.loads(sys.argv[3]), token=False)',
                     source['model'], source['revision'], json.dumps(list(source['files']))])
            env['HF_HUB_OFFLINE'] = '1'
        command([sys.executable, '-m', 'experiments.grammar_interface_r106.run', 'verify'])
        command([sys.executable, '-m', 'experiments.grammar_interface_r106.describe'])
        command([sys.executable, '-m', 'pytest', '-q', 'tests/test_grammar_interface_r106.py', 'tests/test_r106_completed_verifier.py'])
        command([sys.executable, 'scripts/check_docs.py'])
        for folder in (root, host):
            if command(['git', 'status', '--porcelain=v1'], folder):
                raise RuntimeError('Review modified source')
        decision = root/'results/grammar_interface_r106/decision.json'
        if sha(decision) != manifest['decision_sha256']:
            raise RuntimeError('Scientific result changed')
        if json.loads(decision.read_text())['verdict'] != 'GRAMMATICAL_DISCRIMINATION_CONTROL_FAILED':
            raise RuntimeError('Unexpected sealed result')
        receipt.update(verdict='R106_NEGATIVE_CONTROL_RECOMPUTED', assets=len(manifest['files']),
            host_inputs=len(manifest['external_host_inputs']), lfs_paths=len(manifest['lfs_paths']),
            control_files=len(manifest['control_source']['files']), decision_sha256=sha(decision), manifest_sha256=sha(path))
    except Exception as error:
        receipt.update(verdict='FAILED', error=repr(error))
        raise
    finally:
        receipt['end_unix'] = time.time()
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(receipt, stream, sort_keys=True, separators=(',', ':'))
            stream.write('\n')
        print(json.dumps({'receipt_sha256': sha(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'review'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--commit')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fetch-control', action='store_true')
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory(args.root.resolve())
    else:
        if not all((args.commit, args.manifest_sha256, args.output)):
            parser.error('review needs commit, manifest SHA256 and outside-checkout output')
        review(args)
