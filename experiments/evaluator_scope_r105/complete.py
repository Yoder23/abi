"""Additive full-study verifier: the frozen runner also supports initial-only stages.

Never use the initial-stage verifier alone to attest replay completion. This
wrapper requires the entire registered experiment before recomputing any claim.
"""

from __future__ import annotations

import json

from . import core as c
from . import run as frozen_run

FREEZE_SHA = '893718abe540aca2335000636ac75f0f0b4376b86f6eef1bf7b97443a775e741'
REQUIRED = ('freeze.json', 'catalog.json', 'decision.json', 'launch.json',
            'operation-0.json', 'operation-1.json', 'initial/raw.jsonl',
            'initial/receipt.json', 'replay/raw.jsonl', 'replay/receipt.json')


def verify():
    for name in REQUIRED:
        if not (c.OUT/name).is_file():
            raise FileNotFoundError(f'Missing completed-study evidence: {name}')
    if c.sha(c.OUT/'freeze.json') != FREEZE_SHA:
        raise RuntimeError('Different frozen study')
    for index, stage in enumerate(('initial', 'replay')):
        operation = c.read(c.OUT/f'operation-{index}.json')
        if operation['returncode'] != 0 or json.loads(operation['stdout'].strip()) != {'completed': stage, 'pairs': 1264}:
            raise RuntimeError('Incomplete model execution record')
    decision = frozen_run.verify()
    result = {'format': 'r105-complete/1', 'scientific_verdict': decision['verdict'],
              'files': {name: c.sha(c.OUT/name) for name in REQUIRED},
              'verifier_sha256': c.sha(c.HERE/'complete.py'),
              'scope': 'Raw recomputation plus required byte-identical fresh-process replay; not independent hardware or acquisition'}
    path = c.OUT/'completion.json'
    if path.exists():
        if c.read(path) != result:
            raise RuntimeError('Stale completion receipt')
    else:
        c.write_once(path, result)
    return result


if __name__ == '__main__':
    print(json.dumps(verify()), flush=True)
