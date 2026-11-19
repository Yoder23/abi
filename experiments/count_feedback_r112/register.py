"""Write-once R112 registration and resource preflight; contains no model runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath

from . import design as d

MANIFEST = 'results/review_2026_11_18/r111_manifest.json'
DECISION = 'results/scope_revision_r111/live/decision.json'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def audit_parent():
    if sha(d.ROOT / MANIFEST) != d.R111_MANIFEST or sha(d.ROOT / DECISION) != d.R111_DECISION:
        raise RuntimeError('R111 custody anchor changed')
    manifest = read(d.ROOT / MANIFEST)
    root = d.ROOT.resolve()
    host = (root.parent / 'layercake_release').resolve()
    for name, item in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f'Missing/escaping R111 repository dependency: {name}')
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError(f'Changed R111 repository dependency: {name}')
    for name, digest in manifest['host_inputs'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(host) or sha(path) != digest:
            raise RuntimeError(f'Changed/escaping R111 host dependency: {name}')
    return {'manifest_sha256': d.R111_MANIFEST, 'decision_sha256': d.R111_DECISION,
            'repository_files_checked': len(manifest['files']),
            'host_bindings_checked': len(manifest['host_inputs']),
            'scope': 'Byte custody only; not a new scientific replay or source-cache audit'}


def known_cases():
    cases = {}
    path = d.ROOT / 'results/scope_revision_r111/live/initial/raw.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        for span in row['screen']['approval_scope']['evidence']:
            if span['classification'] != 'UNRESOLVED':
                continue
            text = row['output'][span['clause_start']:span['clause_end']].strip()
            cases.setdefault(text, set()).add((row['id'], row['stage']))
    if len(cases) != 5:
        raise RuntimeError('Unexpected disclosed R111 construction inventory')
    return [{'text': text, 'expected': 'NO_FORBIDDEN_ASSERTION',
             'label_origin': 'author-inspected disclosed clause; not independent validation',
             'origins': [{'id': i, 'stage': s} for i, s in sorted(origins)]}
            for text, origins in sorted(cases.items())]


def resource_observation(free_bytes):
    if type(free_bytes) is not int or free_bytes < 0:
        raise ValueError('Free bytes must be an observed nonnegative integer')
    minimum = d.contract()['minimum_free_bytes']
    return {'free_bytes': free_bytes, 'minimum_free_bytes': minimum,
            'deficit_bytes': max(0, minimum - free_bytes),
            'verdict': 'RESOURCE_BLOCKED_BEFORE_IMPLEMENTATION' if free_bytes < minimum
                       else 'READY_FOR_IMPLEMENTATION_NOT_EXECUTION',
            'failure_owner': 'RESOURCE' if free_bytes < minimum else 'NONE',
            'model_calls': 0, 'training_steps': 0, 'recipient_calls': 0,
            'scope_challenge_executed': False, 'implementation_frozen': False,
            'abi_acquisition': 'NOT_TESTED', 'native_layercake': 'NOT_TESTED',
            'gpu_setting_changes': [], 'quality_gate_changes': [], 'deleted_paths': []}


def sources():
    return [*sorted(d.HERE.glob('*.py')), d.HERE / 'PROTOCOL.md',
            d.ROOT / 'tests/test_count_feedback_r112_registration.py']


def register():
    if d.OUT.exists():
        raise RuntimeError('Preserve existing R112 registration/preflight; no overwrite')
    parent = audit_parent()
    cases, population, contract = known_cases(), d.population(), d.contract()
    d.OUT.mkdir(parents=True, exist_ok=False)
    for name, value in [('contract.json', contract), ('population.json', population),
                        ('known_scope_cases.json', cases)]:
        write_new(d.OUT / name, value)
    names = sources() + [d.OUT / n for n in ('contract.json', 'population.json', 'known_scope_cases.json')]
    write_new(d.OUT / 'registration.json', {
        'format': 'r112-preregistration/1', 'client_date': '2026-11-19',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT, text=True).strip(),
        'files': {p.relative_to(d.ROOT).as_posix(): sha(p) for p in names},
        'parent_custody': parent, 'scope': 'Design/input commitment, not an implementation freeze or qualification',
    })
    receipt = {**resource_observation(shutil.disk_usage(d.ROOT).free),
               'format': 'r112-resource-preflight/1', 'observed_utc': datetime.now(timezone.utc).isoformat(),
               'observed_path': str(d.ROOT), 'measurement': 'shutil.disk_usage(observed_path).free',
               'registration_sha256': sha(d.OUT / 'registration.json')}
    write_new(d.OUT / 'preflight_001.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def verify():
    registration = read(d.OUT / 'registration.json')
    names = sources() + [d.OUT / n for n in ('contract.json', 'population.json', 'known_scope_cases.json')]
    expected = {p.relative_to(d.ROOT).as_posix(): sha(p) for p in names}
    if (registration['format'] != 'r112-preregistration/1'
            or registration['client_date'] != '2026-11-19'
            or registration['files'] != expected
            or registration['parent_custody'] != audit_parent()):
        raise RuntimeError('Changed or missing registration/parent payloads')
    datetime.fromisoformat(registration['recorded_utc'])
    if (read(d.OUT / 'contract.json') != d.contract()
            or read(d.OUT / 'population.json') != d.population()
            or read(d.OUT / 'known_scope_cases.json') != known_cases()):
        raise RuntimeError('Unrecomputable registration inputs')
    receipt = read(d.OUT / 'preflight_001.json')
    datetime.fromisoformat(receipt['observed_utc'])
    expected_resource = resource_observation(receipt['free_bytes'])
    observed_path = receipt['observed_path']
    if (receipt['format'] != 'r112-resource-preflight/1'
            or any(receipt.get(k) != v for k, v in expected_resource.items())
            or receipt['registration_sha256'] != sha(d.OUT / 'registration.json')
            or not (PureWindowsPath(observed_path).is_absolute() or PurePosixPath(observed_path).is_absolute())
            or receipt['measurement'] != 'shutil.disk_usage(observed_path).free'):
        raise RuntimeError('Stale/unrecomputable resource receipt')
    print(json.dumps({'verdict': receipt['verdict'], 'registration_sha256': sha(d.OUT / 'registration.json'),
                      'preflight_sha256': sha(d.OUT / 'preflight_001.json'),
                      'scope': 'Recorded preflight only; current free space must be measured again before execution'}))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'verify'))
    args = parser.parse_args()
    result = {'register': register, 'verify': verify}[args.command]()
    if args.command == 'register' and result['verdict'].startswith('RESOURCE_BLOCKED'):
        raise SystemExit(2)
