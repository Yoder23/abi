"""Inventory or reproduce R105 evidence; no model inference or host integration."""

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
    from experiments.evaluator_scope_r105 import core as c
    from experiments.evaluator_scope_r105.complete import verify

    verify()
    files = set(c.binding()['files'])
    files.update(p.relative_to(root).as_posix() for p in c.OUT.rglob('*') if p.is_file())
    files.update(p.relative_to(root).as_posix() for p in (root/'results/review_2026_11_11').glob('r105-*.xml'))
    files.update(('experiments/evaluator_scope_r105/complete.py', 'experiments/evaluator_scope_r105/describe.py',
                  'tests/test_r105_completed_verifier.py', 'scripts/review_r105_public.py'))
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines())
    if files-tracked:
        raise RuntimeError(f'Untracked required files: {sorted(files-tracked)}')
    result = {'format': 'r105-public-manifest/1', 'files': {p: {'sha256': sha(root/p), 'bytes': (root/p).stat().st_size} for p in sorted(files)},
              'model': c.SOURCE, 'revision': c.REVISION, 'model_files': c.binding()['model_files'],
              'source_tag': 'abi-research-r105-2026-11-11', 'decision_sha256': sha(c.OUT/'decision.json'),
              'completion_sha256': sha(c.OUT/'completion.json'), 'freeze_sha256': sha(c.OUT/'freeze.json'),
              'packages': c.read(c.OUT/'freeze.json')['packages'],
              'scope': 'External-label evaluator rejection only; zero acquisition/teacher/recipient calls'}
    path = root/'results/review_2026_11_11/r105_manifest.json'
    c.write_once(path, result)
    print(json.dumps({'assets': len(files), 'manifest_sha256': sha(path)}), flush=True)


def review(args):
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        raise RuntimeError('Preserve receipts; output must be outside checkout')
    receipt = {'format': 'r105-public-reconstruction/1', 'start_unix': time.time(), 'commands': [],
               'scope': 'Public source/payload recomputation on same machine; no live model inference or independent hardware'}
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)

    def command(argv):
        done = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True, timeout=600)
        receipt['commands'].append({'command': argv, 'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr})
        print(json.dumps({'command': argv, 'returncode': done.returncode}), flush=True)
        if done.returncode:
            raise RuntimeError('Public reconstruction command failed')
        return done.stdout.strip()

    try:
        if command(['git', 'rev-parse', 'HEAD']) != args.commit or command(['git', 'status', '--porcelain=v1']):
            raise RuntimeError('Wrong source revision or dirty checkout')
        if command(['git', 'remote', 'get-url', 'origin']) != 'https://github.com/Yoder23/abi.git':
            raise RuntimeError('Wrong public repository')
        path = root/'results/review_2026_11_11/r105_manifest.json'
        if sha(path) != args.manifest_sha256:
            raise RuntimeError('Wrong manifest')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if sha(Path(__file__)) != manifest['files']['scripts/review_r105_public.py']['sha256']:
            raise RuntimeError('Wrong review helper')
        for relative, item in manifest['files'].items():
            target = (root/relative).resolve()
            if not target.is_relative_to(root) or not target.is_file() or target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
                raise RuntimeError(f'Missing/changed public asset: {relative}')
        if args.fetch_model:
            env['HF_HUB_OFFLINE'] = '0'
            command([sys.executable, '-m', 'experiments.evaluator_scope_r105.run', 'fetch-model'])
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for name, item in manifest['model_files'].items():
            target = root/'hf_models/evaluator_scope_r105'/name
            if not target.is_file() or sha(target) != item['sha256'] or target.stat().st_size != item['bytes']:
                raise RuntimeError(f'Missing/changed public model payload: {name}')
        command([sys.executable, '-m', 'experiments.evaluator_scope_r105.complete'])
        command([sys.executable, '-m', 'experiments.evaluator_scope_r105.describe'])
        command([sys.executable, '-m', 'pytest', '-q', 'tests/test_evaluator_scope_r105.py', 'tests/test_r105_completed_verifier.py'])
        command([sys.executable, 'scripts/check_docs.py'])
        if command(['git', 'status', '--porcelain=v1']):
            raise RuntimeError('Review changed checkout')
        if sha(root/'results/evaluator_scope_r105/decision.json') != manifest['decision_sha256']:
            raise RuntimeError('Scientific decision changed')
        if json.loads((root/'results/evaluator_scope_r105/decision.json').read_text(encoding='utf-8'))['verdict'] != 'EVALUATOR_CANDIDATE_REJECTED':
            raise RuntimeError('Unexpected scientific result for this sealed review')
        receipt.update(verdict='REJECTED_EVALUATOR_EVIDENCE_RECOMPUTED', assets=len(manifest['files']),
                       model_files=len(manifest['model_files']), manifest_sha256=sha(path),
                       decision_sha256=manifest['decision_sha256'], completion_sha256=manifest['completion_sha256'])
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
    parser.add_argument('--fetch-model', action='store_true')
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory(args.root.resolve())
    else:
        if not all((args.commit, args.manifest_sha256, args.output)):
            parser.error('review requires commit, manifest SHA256 and outside-checkout output')
        review(args)
