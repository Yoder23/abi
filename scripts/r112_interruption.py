"""Audit R112's resource stop without weakening its frozen complete verifier.

Post-stop custody only: no inference, repair, partial quality promotion or retry.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re

from experiments.count_feedback_r112 import design as d
from experiments.count_feedback_r112.execution import run as r


def limit_observation(text):
    matches = re.findall(r"Storage floor/cap violated: (\{'free_bytes': \d+, 'evidence_bytes': \d+\})", text)
    if not matches:
        raise RuntimeError('Missing measured storage exception')
    values = [ast.literal_eval(value) for value in matches]
    for value in values:
        if (set(value) != {'free_bytes', 'evidence_bytes'}
                or any(type(x) is not int or x < 0 for x in value.values())
                or value['free_bytes'] >= 10 * 1024**3
                or value['evidence_bytes'] > 1024**3):
            raise RuntimeError('Not the registered free-space-floor stop')
    return values[-1]


def check_prefix(initial, replay):
    if not 0 < len(replay) < 288 or len(initial) != 288:
        raise RuntimeError('Not one complete initial run plus incomplete replay')
    if r.scientific(replay) != r.scientific(initial[:len(replay)]):
        raise RuntimeError('Preserved replay prefix differs scientifically')
    for row in replay:
        free = row['storage_before']
        if (not r.positive(row['wall_seconds']) or free['free_bytes'] < 10 * 1024**3
                or free['evidence_bytes'] > 1024**3):
            raise RuntimeError('Invalid preserved-prefix resource observation')


def derive():
    r.binding()
    scope = r.verify_scope()
    if scope != r.read(d.OUT / 'scope/decision.json') or scope['verdict'] != 'FINITE_SCOPE_PREREQUISITE_PASSED':
        raise RuntimeError('Missing/stale passing scope packet')
    base = d.OUT / 'live'
    if any((base / name).exists() for name in ('complete.json', 'decision.json', 'replay/receipt.json')):
        raise RuntimeError('Observer cannot replace a completed study or replay')
    stop, launch = r.read(base / 'stop.json'), r.read(base / 'launch.json')
    if (launch['freeze_sha256'] != r.sha(d.OUT / 'freeze.json')
            or launch['budget_seconds'] != 30600 or not r.positive(stop['elapsed_seconds'])
            or stop['elapsed_seconds'] > 30600):
        raise RuntimeError('Changed launch or stopping budget')
    initial = r.validate_live(r.rows(base / 'initial/raw.jsonl'), r.old.runtime.tokenizer('control'))
    r.unchanged_controls(initial)
    replay = r.rows(base / 'replay/raw.jsonl')
    check_prefix(initial, replay)
    receipt = r.read(base / 'initial/receipt.json')
    operations = {n: r.read(base / f'{n}_operation.json') for n in ('initial', 'replay')}
    for repeat, operation in operations.items():
        if (operation['raw_sha256'] != r.sha(base / repeat / 'raw.jsonl')
                or operation['argv'][1:] != ['-u', '-m', 'experiments.count_feedback_r112.execution.run', 'execute', '--repeat', repeat]
                or not r.positive(operation['elapsed_seconds']) or operation['elapsed_seconds'] > 14400):
            raise RuntimeError('Missing/stale operation evidence')
    if (operations['initial']['returncode'] != 0 or operations['replay']['returncode'] == 0
            or receipt['raw_sha256'] != operations['initial']['raw_sha256']
            or receipt['freeze_sha256'] != r.sha(d.OUT / 'freeze.json')
            or receipt['calls'] != 288 or receipt['repeat'] != 'initial'
            or receipt['parameters'] != r.old.runtime.source_parameter_count()
            or receipt['precision'] != 'bfloat16' or receipt['gpu_setting_changes'] != []
            or receipt['training_steps'] != 0 or receipt['recipient_calls'] != 0
            or receipt['rss_interval_seconds'] != .05 or receipt['wall_seconds'] > 14400
            or any(not r.positive(receipt[k]) for k in ('pid', 'wall_seconds', 'cpu_seconds', 'sampled_peak_rss_bytes', 'peak_cuda_allocated_bytes'))):
        raise RuntimeError('Invalid initial receipt or interrupted operation')
    measured = {'worker': limit_observation(operations['replay']['stderr']),
                'controller': limit_observation(stop['error'])}
    for name in ('initial_before.json', 'initial_after.json', 'replay_before.json'):
        telemetry = r.read(base / name)
        if (telemetry['returncode'] != 0 or telemetry['settings_changed'] != []
                or telemetry['argv'] != ['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK']
                or not telemetry['stdout'].strip()
                or telemetry['storage']['free_bytes'] < 10 * 1024**3):
            raise RuntimeError('Invalid preserved hardware observation')
    transcript = io.StringIO()
    try:
        with contextlib.redirect_stdout(transcript):
            r.verify()
    except RuntimeError as error:
        if str(error) != 'Interrupted source study cannot qualify':
            raise
        rejection = str(error)
    else:
        raise RuntimeError('Frozen complete verifier incorrectly accepted interruption')
    return {
        'format': 'r112-resource-interruption/1', 'verdict': 'RESOURCE_INCOMPLETE_NOT_QUALIFIED',
        'failure_owner': 'RESOURCE', 'scope_decision_sha256': r.sha(d.OUT / 'scope/decision.json'),
        'freeze_sha256': r.sha(d.OUT / 'freeze.json'), 'raw_evidence_hashes': r.hashes(base),
        'initial_calls': len(initial), 'replay_calls': len(replay), 'required_calls': 576,
        'observed_calls': len(initial) + len(replay),
        'exact_scientific_replay_prefix_rows': len(replay),
        'measured_storage_stop': measured, 'controller_elapsed_seconds': stop['elapsed_seconds'],
        'observed_output_tokens': sum(len(x['token_ids']) for x in initial + replay),
        'observed_generation_seconds': sum(x['wall_seconds'] for x in initial + replay),
        'frozen_complete_verifier_rejection': rejection,
        'quality_qualification': 'NOT_ESTABLISHED', 'abi_acquisition': 'NOT_TESTED',
        'native_layercake': 'NOT_TESTED', 'new_inference_calls_by_observer': 0,
        'scope': ('Post-stop custody and exact-prefix recomputation only; not completed replay, '
                  'a quality result or a scientific promotion. No repair, retry or gate change.'),
    }


def verify(seal=False):
    value = derive()
    path = d.OUT / 'interruption.json'
    if seal:
        r.write(path, value)
    if r.read(path) != value:
        raise RuntimeError('Missing/stale interruption observation')
    print(json.dumps(value), flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true')
    verify(parser.parse_args().seal)
