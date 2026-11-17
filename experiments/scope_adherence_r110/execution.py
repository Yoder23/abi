"""R110 evidence identities, recomputation and analysis; no model generation."""
from __future__ import annotations

import copy
import importlib.metadata
import math
import os
import shutil
import subprocess

from . import challenge, design as d, evaluator as e, register
from experiments.trained_grounding_r109 import core as h
from experiments.trained_grounding_r109 import run as runtime

OUT = d.ROOT/'results/scope_adherence_r110/execution'
REG_SHA = 'a689dc73219d2783e4129e2cf92dc4af3e5c6363441197094d0fa7ecdbd47697'
IMPLEMENTATION_FILES = ('evaluator.py', 'challenge.py', 'execution.py', 'run.py', 'IMPLEMENTATION.md')
TEST_FILE = 'tests/test_scope_adherence_r110_execution.py'
PACKAGES = ('torch', 'transformers', 'tokenizers', 'safetensors', 'numpy', 'psutil')
write, read, rows, sha, append = h.write, h.read, h.rows, h.sha, h.append


def git(*args):
    return subprocess.check_output(['git', *args], cwd=d.ROOT, text=True).strip()


def relative(path):
    return os.path.relpath(path, d.ROOT).replace(chr(92), '/')


def dependencies():
    if sha(d.OUT/'registration.json') != REG_SHA:
        raise RuntimeError('Registration identity changed')
    registered = read(d.OUT/'registration.json')
    paths = [d.OUT/'registration.json', *(d.HERE/n for n in IMPLEMENTATION_FILES), d.ROOT/TEST_FILE]
    return {**registered['files'], **{relative(p): sha(p) for p in paths}}


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def freeze_binding(source=True):
    frozen = read(OUT/'implementation_freeze.json')
    registered = read(d.OUT/'registration.json')
    expected = dependencies()
    if (frozen['format'] != 'r110-implementation/1' or frozen['registration_sha256'] != REG_SHA
            or frozen['files'] != expected or frozen['contract'] != d.contract()
            or frozen['source_files'] != registered['source_files']
            or frozen['packages'] != package_versions()):
        raise RuntimeError('Missing, stale or changed implementation freeze')
    for name, digest in expected.items():
        if sha(d.ROOT/name) != digest:
            raise RuntimeError(f'Missing or changed dependency: {name}')
    for name in [*(relative(d.HERE/n) for n in IMPLEMENTATION_FILES), TEST_FILE]:
        payload = subprocess.check_output(['git', 'show', f'{frozen["implementation_commit"]}:{name}'], cwd=d.ROOT)
        import hashlib
        if hashlib.sha256(payload).hexdigest() != expected[name]:
            raise RuntimeError('Implementation is not bound to its exact Git commit')
    if source and runtime.inventory(runtime.snapshot()) != frozen['source_files']:
        raise RuntimeError('Missing or mutated cached source')
    return frozen


def require_published(paths):
    """Launch-only network check. Offline verification does not fake publication."""
    import hashlib
    head = git('rev-parse', 'HEAD')
    remote = git('ls-remote', 'origin', 'refs/heads/master').split()
    if not remote or remote[0] != head:
        raise RuntimeError('Publish the current implementation/input freeze before scoring')
    for path in paths:
        blob = subprocess.check_output(['git', 'show', f'{head}:{relative(path)}'], cwd=d.ROOT)
        if hashlib.sha256(blob).hexdigest() != sha(path):
            raise RuntimeError('Uncommitted execution input')
    return head


def storage_observation():
    return {'free_bytes': shutil.disk_usage(d.ROOT).free,
            'evidence_bytes': sum(p.stat().st_size for p in OUT.parent.rglob('*') if p.is_file()),
            'minimum_free_bytes': d.contract()['resources']['minimum_free_bytes'],
            'maximum_evidence_bytes': d.contract()['resources']['maximum_evidence_bytes']}


def guard_storage():
    value = storage_observation()
    if value['free_bytes'] < value['minimum_free_bytes'] or value['evidence_bytes'] > value['maximum_evidence_bytes']:
        raise RuntimeError(f'Storage guard: {value}')
    return value


def challenge_binding():
    frozen = freeze_binding()
    packet = read(OUT/'challenge.json')
    if packet != challenge.generate(frozen['implementation_commit'], REG_SHA):
        raise RuntimeError('Missing or mutated prospective challenge')
    return packet


def scope_inputs(packet=None):
    # Only historical development examples; never use challenge labels here.
    fixtures = h.audit()['cases']
    if len(fixtures) != 164:
        raise RuntimeError('Historical fixture count changed')
    result = [{'id': f['id'], 'kind': 'historical_fixture', 'expected': f['expected_pass'],
               'output': f['answer'], 'task_row': f['row']} for f in fixtures]
    known = read(h.OUT/'evaluator_scope_audit.json')['approval_flagged_rows']
    if len(known) != 7:
        raise RuntimeError('Historical regression count changed')
    result += [{'id': f['id'], 'kind': 'known_regression', 'expected': e.SAFE,
                'output': f['output']} for f in known]
    if packet is not None:
        result += [{'id': row['id'], 'kind': 'challenge', 'class': row['class'],
                    'expected': row['expected'], 'output': row['text']} for row in packet['cases']]
    return result


def score_scope_item(item):
    result = {k: v for k, v in item.items() if k != 'task_row'}
    result['observed'] = e.screen(item['task_row'], item['output']) if item['kind'] == 'historical_fixture' else e.classify(item['output'])
    return result


def development_rows():
    return [score_scope_item(item) for item in scope_inputs()]


def scope_rows(packet):
    return [score_scope_item(item) for item in scope_inputs(packet)]


def scope_summary(observed):
    if len(observed) != 771 or len({(r['kind'], r['id']) for r in observed}) != 771:
        raise RuntimeError('Incomplete scope observations')
    errors = {'historical_fixture': [], 'known_regression': [], **{k: [] for k in challenge.LABELS}}
    false_safe = []
    for row in observed:
        value = row['observed']['mechanical_pass'] if row['kind'] == 'historical_fixture' else row['observed']['classification']
        group = row['class'] if row['kind'] == 'challenge' else row['kind']
        if value != row['expected']:
            errors[group].append(row['id'])
        if row['kind'] == 'challenge' and row['expected'] == e.UNSAFE and value == e.SAFE:
            false_safe.append(row['id'])
    counts = {k: sum(r['kind'] == 'challenge' and r['class'] == k for r in observed) for k in challenge.LABELS}
    if set(counts.values()) != {100}:
        raise RuntimeError('Unbalanced challenge')
    return {'rows': len(observed), 'challenge_counts': counts, 'errors': errors,
            'false_safe_assertions': false_safe,
            'verdict': 'SCOPE_PREREQUISITE_FAILED' if any(errors.values()) else 'FINITE_SCOPE_PREREQUISITE_PASSED'}


def positive_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_scope_packet(packet, first, replay, receipts, operations, completion, launch):
    expected = scope_rows(packet)
    if first != expected or replay != expected:
        raise RuntimeError('Missing, changed or unrecomputable scope replay')
    if (launch['budget_seconds'] != 1800 or not positive_finite(completion['elapsed_seconds'])
            or completion['elapsed_seconds'] > 1800 or len(receipts) != 2 or len(operations) != 2
            or len({r['pid'] for r in receipts}) != 2):
        raise RuntimeError('Invalid scope process/time accounting')
    for repeat, raw, receipt, operation in zip(('initial', 'replay'), (first, replay), receipts, operations):
        if (receipt['repeat'] != repeat or receipt['rows'] != len(raw) or receipt['model_calls'] != 0
                or not positive_finite(receipt['elapsed_seconds']) or not positive_finite(receipt['cpu_seconds'])
                or receipt['elapsed_seconds'] > 1800 or operation['returncode'] != 0):
            raise RuntimeError('Invalid scope operation')
        storage = receipt['storage_observations']
        if len(storage) != 771 or any(v['free_bytes'] < 10*1024**3 or v['evidence_bytes'] > 1024**3 for v in storage):
            raise RuntimeError('Missing or invalid scope storage observations')
    if sum(r['elapsed_seconds'] for r in receipts) > completion['elapsed_seconds']:
        raise RuntimeError('Inconsistent offline time')
    return scope_summary(expected)


def verify_scope():
    packet = challenge_binding()
    base = OUT/'scope'
    receipts, operations, raw = [], [], []
    for repeat in ('initial', 'replay'):
        folder = base/repeat
        receipt, operation = read(folder/'receipt.json'), read(base/f'{repeat}_operation.json')
        if (receipt['raw_sha256'] != sha(folder/'raw.jsonl')
                or receipt['freeze_sha256'] != sha(OUT/'implementation_freeze.json')
                or receipt['challenge_sha256'] != sha(OUT/'challenge.json')
                or operation['raw_sha256'] != sha(folder/'raw.jsonl')):
            raise RuntimeError('Stale scope evidence')
        receipts.append(receipt)
        operations.append(operation)
        raw.append(rows(folder/'raw.jsonl'))
    completion, launch = read(base/'complete.json'), read(base/'launch.json')
    for value in (completion, launch):
        if value['freeze_sha256'] != sha(OUT/'implementation_freeze.json') or value['challenge_sha256'] != sha(OUT/'challenge.json'):
            raise RuntimeError('Stale scope controller identity')
    if (base/'stop.json').exists():
        raise RuntimeError('Interrupted scope study cannot pass')
    result = validate_scope_packet(packet, *raw, receipts, operations, completion, launch)
    return {**result, 'freeze_sha256': sha(OUT/'implementation_freeze.json'),
            'challenge_sha256': sha(OUT/'challenge.json'),
            'evidence_hashes': {relative(p): sha(p) for p in sorted(base.rglob('*'))
                                if p.is_file() and p.name != 'decision.json'}}


def require_scope_pass():
    result = verify_scope()
    if read(OUT/'scope/decision.json') != result or result['verdict'] != 'FINITE_SCOPE_PREREQUISITE_PASSED':
        raise RuntimeError('Scope prerequisite did not pass; no GPU execution')
    return result


def inputs(interface):
    if interface not in d.INTERFACES:
        raise ValueError('Unregistered interface')
    return [r for r in read(d.OUT/'rendered_inputs.json') if r['interface'] == interface]


def validate_live(observed, interface, tokenizer):
    refs, rendered = d.population(), inputs(interface)
    if len(observed) != 96 or len(rendered) != 96:
        raise RuntimeError('Missing raw generation rows')
    for actual, ref, inp in zip(observed, refs, rendered):
        tokens = actual['token_ids']
        if (any(actual[k] != ref[k] for k in ('id', 'pair', 'panel', 'task'))
                or actual['interface'] != interface or actual['prompt'] != inp['prompt']
                or actual['prompt_ids'] != inp['prompt_ids']
                or d.render(ref, interface, tokenizer) != inp['prompt']
                or tokenizer.encode(inp['prompt'], add_special_tokens=False) != inp['prompt_ids']
                or not tokens or len(tokens) > 640 or len(inp['prompt_ids'])+640 > 1024
                or any(type(t) is not int or not 0 <= t < len(tokenizer) for t in tokens)):
            raise RuntimeError('Changed raw identity, prompt, context or tokens')
        stop = 'eos' if tokens[-1] == tokenizer.eos_token_id else 'budget'
        if (tokenizer.eos_token_id in tokens[:-1] or actual['stop_reason'] != stop
                or (stop == 'budget' and len(tokens) != 640)):
            raise RuntimeError('Invalid generation termination')
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        text = decoded.strip()  # Exactly the historical scoring normalization.
        if (actual['decoded_output'] != decoded or actual['output'] != text
                or actual['screen'] != e.screen(ref, text) or actual['legacy_screen'] != h.screen(ref, text)
                or actual['output_utf8_bytes'] != len(text.encode()) or actual['output_characters'] != len(text)
                or not positive_finite(actual['wall_seconds'])):
            raise RuntimeError('Unrecomputable output, scoring or accounting')
        storage = actual['storage_before']
        if storage['free_bytes'] < 10*1024**3 or storage['evidence_bytes'] > 1024**3:
            raise RuntimeError('Storage gate violated')


def scientific(raw):
    return [{k: v for k, v in r.items() if k not in ('wall_seconds', 'storage_before')} for r in raw]


def bootstrap_delta(contract, plain, sentence=False):
    import numpy as np
    h.pair_rows(contract)
    h.pair_rows(plain)
    if [r['id'] for r in contract] != [r['id'] for r in plain]:
        raise RuntimeError('Unpaired interface comparison')
    key = 'requested_sentence_count' if sentence else 'mechanical_pass'
    values = [int(a['screen'][key])-int(b['screen'][key]) for a, b in zip(contract, plain)
              if not sentence or a['task'] in d.INTERVENTION_TASKS]
    count = 12 if sentence else 48
    delta = np.asarray(values, dtype=float).reshape(count, 2).mean(axis=1)
    rng = np.random.default_rng(108001)
    draws = delta[rng.integers(0, count, (10000, count))].mean(axis=1)
    return {'paired_clusters': count, 'mean_difference': float(delta.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(), 'bootstrap_draws': 10000, 'seed': 108001}


def live_decision(plain, contract):
    cells = {name: h.summarize(raw) for name, raw in zip(d.INTERFACES, (plain, contract))}
    legacy = {}
    for name, raw in zip(d.INTERFACES, (plain, contract)):
        changed = copy.deepcopy(raw)
        for row in changed:
            row['screen'] = row['legacy_screen']
        legacy[name] = h.summarize(changed)
    primary, quality = bootstrap_delta(contract, plain, True), bootstrap_delta(contract, plain)
    controls = [(a, b) for a, b in zip(plain, contract) if a['task'] not in d.INTERVENTION_TASKS]
    if len(controls) != 72:
        raise RuntimeError('Missing unchanged interface controls')
    mismatches = [a['id'] for a, b in controls if a['token_ids'] != b['token_ids'] or a['prompt_ids'] != b['prompt_ids']]
    gain = primary['ci95'][0] > 0 and quality['ci95'][0] >= -.10
    if mismatches:
        verdict, owner = 'UNCHANGED_INPUT_EXECUTION_DISCREPANCY', 'EXECUTION_OR_CONDITIONING'
    elif gain and cells['response_contract']['qualified_mechanical_only']:
        verdict, owner = 'BOUNDED_INTERFACE_IMPROVEMENT', 'NONE_WITHIN_MECHANICAL_SCOPE'
    elif cells['plain']['qualified_mechanical_only']:
        verdict, owner = 'PLAIN_CONTROL_QUALIFIED_NO_SUPPORTED_INTERFACE_GAIN', 'NONE_WITHIN_MECHANICAL_SCOPE'
    elif cells['response_contract']['qualified_mechanical_only']:
        verdict, owner = 'CONTRACT_CONTROL_QUALIFIED_NO_SUPPORTED_INTERFACE_GAIN', 'NONE_WITHIN_MECHANICAL_SCOPE'
    else:
        verdict, owner = 'CONTROL_QUALIFICATION_FAILED', 'EXPERIMENTAL_PREREQUISITE'
    return {'verdict': verdict, 'failure_owner': owner, 'cells': cells, 'legacy_scores_descriptive': legacy,
            'primary_sentence_adherence': primary, 'all_task_quality': quality,
            'unchanged_input_token_mismatches': mismatches, 'unchanged_input_rows': 72,
            'abi_acquisition': 'NOT_TESTED', 'native_layercake': 'NOT_TESTED',
            'full_moonshot': 'NOT_PROVEN', 'english_artifact': 'NONE', 'recipient_calls': 0, 'training_steps': 0}
