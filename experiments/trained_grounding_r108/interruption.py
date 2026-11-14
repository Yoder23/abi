"""Publish an interrupted experiment without manufacturing a scientific decision.

This additive observer never changes the frozen execution or complete verifier.
It requires their timeout, validates the raw prefix, and keeps qualification OPEN.
"""
from __future__ import annotations

import json
import math
import re

from . import core as c
from . import run


def derive():
    frozen = c.binding()
    if run.inventory(run.snapshot()) != frozen['source_files']:
        raise RuntimeError('Missing/changed source payload')
    if c.read(c.OUT/'matched_training_provenance.json') != c.training_provenance():
        raise RuntimeError('Changed historical controls')
    stop, launch = c.read(c.OUT/'stop.json'), c.read(c.OUT/'launch.json')
    if ('TimeoutExpired' not in stop['error'] or not math.isfinite(stop['elapsed_seconds'])
            or stop['elapsed_seconds'] < 1500 or launch['budget_seconds'] != 5400):
        raise RuntimeError('Not the registered execution-budget timeout')
    if ((c.OUT/'decision.json').exists() or (c.OUT/'control/replay').exists()
            or (c.OUT/'control/initial/receipt.json').exists() or any((c.OUT/a).exists() for a in c.ARMS)):
        raise RuntimeError('Interruption observer cannot replace a completed or different execution')
    raw = c.rows(c.OUT/'control/initial/raw.jsonl')
    if not 1 <= len(raw) <= 96:
        raise RuntimeError('Missing or excessive preserved observations')
    c.validate(raw, c.read(c.OUT/'population.json')[:len(raw)], 'control', run.tokenizer('control'))
    resources = {}
    for label in ('initial', 'final'):
        record = c.read(c.OUT/f'resource_{label}.json')
        if (record['freeze_sha256'] != c.sha(c.OUT/'freeze.json') or record['settings_changed'] != []
                or record['incomplete_last_line'] is not None or not record['commands']
                or record['commands'][0]['argv'] != ['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK']
                or any(r['returncode'] != 0 for r in record['commands'])
                or record['observed_unix'] < launch['start_unix']):
            raise RuntimeError('Invalid hardware observation')
        stdout = record['commands'][0]['stdout']
        def number(field):
            match = re.search(r'^\s*' + re.escape(field) + r'\s*:\s*([0-9.]+) W\s*$', stdout, re.M)
            if not match:
                raise RuntimeError(f'Missing telemetry: {field}')
            return float(match[1])
        resources[label] = {'record_sha256': c.sha(c.OUT/f'resource_{label}.json'),
            'requested_power_limit_watts': number('Requested Power Limit'),
            'default_power_limit_watts': number('Default Power Limit'),
            'gpu_clock_summary_raw': record['commands'][1]['stdout'].strip(),
            'completed_rows': record['completed_rows'], 'battery': record['battery'],
            'free_disk_bytes': record['free_disk_bytes']}
        if label == 'final' and (record['completed_rows'] != len(raw)
                or record['observed_output_tokens'] != sum(len(r['token_ids']) for r in raw)
                or record['observed_generation_seconds'] != sum(r['wall_seconds'] for r in raw)
                or record['launch_elapsed_seconds'] < stop['elapsed_seconds']):
            raise RuntimeError('Final resource observation does not match the preserved prefix')
    return {'format': 'r108-interruption/1', 'status': 'RESOURCE_BUDGET_STOP_NOT_QUALIFIED',
        'freeze_sha256': c.sha(c.OUT/'freeze.json'), 'stop_sha256': c.sha(c.OUT/'stop.json'),
        'launch_sha256': c.sha(c.OUT/'launch.json'), 'raw_sha256': c.sha(c.OUT/'control/initial/raw.jsonl'),
        'completed_initial_rows': len(raw), 'required_initial_rows': 96, 'completed_replay_rows': 0,
        'initial_output_tokens_observed': sum(len(r['token_ids']) for r in raw),
        'initial_generation_seconds_observed': sum(r['wall_seconds'] for r in raw),
        'initial_output_utf8_bytes_observed': sum(r['output_utf8_bytes'] for r in raw),
        'pilot_elapsed_seconds_at_stop': stop['elapsed_seconds'], 'resource_observations': resources,
        'source_preparation_sha256': c.sha(c.OUT/'source.json'), 'new_training_steps': 0,
        'recipient_executions': 0, 'qualification': 'NOT_ESTABLISHED', 'native_layercake_defect': 'NOT_ESTABLISHED',
        'abi_acquisition_quality': 'NOT_TESTED', 'product_promotion': 'NONE', 'full_moonshot': 'NOT_PROVEN',
        'scope': 'Verified incomplete prefix and resource-budget timeout, not a completed scientific experiment; no partial pass rate promoted'}


def verify(seal=False):
    value = derive()
    path = c.OUT/'interruption.json'
    if seal:
        c.write(path, value)
    if c.read(path) != value:
        raise RuntimeError('Missing or stale interruption record')
    print(json.dumps(value), flush=True)
    return value


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    verify(parser.parse_args().seal)
