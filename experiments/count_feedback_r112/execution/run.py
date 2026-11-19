"""Write-once bounded R112 execution and fail-closed raw recomputation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

from experiments.scope_adherence_r110 import execution as old
from experiments.scope_revision_r111 import run as prior

from .. import design as d
from .. import register as registration
from . import challenge
from . import evaluator as e

read, rows, write, append, sha = old.read, old.rows, old.write, old.append, old.sha
TEST = 'tests/test_count_feedback_r112_execution.py'
R111_MANIFEST = 'results/review_2026_11_18/r111_manifest.json'
REG_SHA = '82788fa350c8c80f561c4d6ab55595b78ab2a09645ee64513c70b2e3a7929704'
PREREG_COMMIT = '60f6f80b579f7072da9bcb26d2a848924e7192b0'


def relative(path):
    return os.path.relpath(path, d.ROOT).replace('\\', '/')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=d.ROOT, text=True).strip()


def storage():
    value = {'free_bytes': shutil.disk_usage(d.ROOT).free,
             'evidence_bytes': sum(p.stat().st_size for p in d.OUT.rglob('*') if p.is_file())}
    if value['free_bytes'] < 10 * 1024**3 or value['evidence_bytes'] > 1024**3:
        raise RuntimeError(f'Storage floor/cap violated: {value}')
    return value


def inputs(tok):
    result = []
    for ref in d.population():
        prompt = d.render(ref, 'draft', tok)
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) + 640 > 2048:
            raise RuntimeError('Baseline context overflow')
        result.append({'id': ref['id'], 'prompt': prompt, 'prompt_ids': ids,
                       'revision_instruction': d.previous.revision_request(ref),
                       'feedback_template': d.FEEDBACK})
    return result


def source_paths():
    return [*sorted(d.HERE.rglob('*.py')), d.HERE / 'PROTOCOL.md', d.HERE / 'IMPLEMENTATION.md',
            d.ROOT / TEST, d.ROOT / 'tests/test_count_feedback_r112_registration.py']


def inherited():
    registration.verify()
    if sha(d.OUT / 'registration.json') != REG_SHA:
        raise RuntimeError('R112 registration changed')
    manifest = read(d.ROOT / R111_MANIFEST)
    return {**{name: value['sha256'] for name, value in manifest['files'].items()},
            **manifest['host_inputs'], R111_MANIFEST: d.R111_MANIFEST,
            **read(d.OUT / 'registration.json')['files'],
            relative(d.OUT / 'registration.json'): REG_SHA}


def binding(check_source=True):
    f = read(d.OUT / 'freeze.json')
    prereg = subprocess.check_output(['git', 'show', f'{PREREG_COMMIT}:{relative(d.HERE / "PROTOCOL.md")}'], cwd=d.ROOT)
    if hashlib.sha256(prereg).hexdigest() != sha(d.HERE / 'PROTOCOL.md'):
        raise RuntimeError('Published preregistration changed')
    ancestry = inherited()
    expected = set(ancestry) | {relative(p) for p in source_paths()} | {
        relative(d.OUT / n) for n in ('population.json', 'baseline_inputs.json')}
    if (f['format'] != 'r112-freeze/1' or set(f['files']) != expected
            or f['contract'] != d.contract() or f['packages'] != old.package_versions()
            or f['protocol_sha256'] != sha(d.HERE / 'PROTOCOL.md')
            or f['r111_decision_sha256'] != d.R111_DECISION or f['registration_sha256'] != REG_SHA
            or f['source_files'] != read(prior.d.OUT / 'freeze.json')['source_files']):
        raise RuntimeError('Missing or changed frozen contract/dependencies')
    for name, digest in f['files'].items():
        if sha(d.ROOT / name) != digest or (name in ancestry and digest != ancestry[name]):
            raise RuntimeError(f'Changed frozen file: {name}')
    for p in source_paths():
        blob = subprocess.check_output(['git', 'show', f'{f["implementation_commit"]}:{relative(p)}'], cwd=d.ROOT)
        if hashlib.sha256(blob).hexdigest() != f['files'][relative(p)]:
            raise RuntimeError('Code not bound to implementation commit')
    if read(d.OUT / 'population.json') != d.population():
        raise RuntimeError('Changed population')
    tok = old.runtime.tokenizer('control')
    if read(d.OUT / 'baseline_inputs.json') != inputs(tok):
        raise RuntimeError('Changed exact input tokens')
    if check_source and old.runtime.inventory(old.runtime.snapshot()) != f['source_files']:
        raise RuntimeError('Source cache changed or missing')
    return f


def freeze():
    storage()
    if any((d.OUT / n).exists() for n in ('freeze.json', 'baseline_inputs.json')):
        raise RuntimeError('Preserve existing freeze or partial inputs')
    previous = prior.binding()
    commit = git('rev-parse', 'HEAD')
    files = inherited()
    for p in source_paths():
        blob = subprocess.check_output(['git', 'show', f'{commit}:{relative(p)}'], cwd=d.ROOT)
        if hashlib.sha256(blob).hexdigest() != sha(p):
            raise RuntimeError('Commit the full implementation first')
        files[relative(p)] = sha(p)
    baseline = inputs(old.runtime.tokenizer('control'))
    write(d.OUT / 'baseline_inputs.json', baseline)
    files.update({relative(d.OUT / n): sha(d.OUT / n) for n in ('population.json', 'baseline_inputs.json')})
    write(d.OUT / 'freeze.json', {
        'format': 'r112-freeze/1', 'implementation_commit': commit,
        'protocol_sha256': sha(d.HERE / 'PROTOCOL.md'), 'files': files,
        'contract': d.contract(), 'source_files': previous['source_files'],
        'packages': old.package_versions(), 'r111_decision_sha256': d.R111_DECISION,
        'registration_sha256': REG_SHA,
        'created_unix': time.time(), 'model_calls_at_freeze': 0,
    })
    binding()
    print(json.dumps({'freeze_sha256': sha(d.OUT / 'freeze.json'), 'model_calls': 0}), flush=True)


def published(paths):
    head = git('rev-parse', 'HEAD')
    if git('ls-remote', 'origin', 'refs/heads/master').split()[0] != head:
        raise RuntimeError('Publish current HEAD before execution')
    for p in paths:
        blob = subprocess.check_output(['git', 'show', f'{head}:{relative(p)}'], cwd=d.ROOT)
        if hashlib.sha256(blob).hexdigest() != sha(p):
            raise RuntimeError('Unpublished stage prerequisite')
    return head


def challenge_binding():
    f = binding()
    expected = challenge.generate(f['implementation_commit'], f['protocol_sha256'])
    if read(d.OUT / 'challenge.json') != expected:
        raise RuntimeError('Changed challenge construction or identity')
    return expected


def materialize():
    storage()
    f = binding()
    published([d.OUT / 'freeze.json'])
    write(d.OUT / 'challenge.json', challenge.generate(f['implementation_commit'], f['protocol_sha256']))
    print(json.dumps({'challenge_sha256': sha(d.OUT / 'challenge.json'), 'cases': 800}), flush=True)


def development_inputs():
    result = []
    for item in prior.scope_inputs(read(prior.d.OUT / 'challenge.json')):
        row = dict(item)
        row['kind'] = 'historical_' + row['kind']
        result.append(row)
    for index, item in enumerate(registration.known_cases()):
        result.append({'id': f'known-r111-{index:03d}', 'kind': 'known_r111_clause',
                       'output': item['text'], 'expected': item['expected'], 'origins': item['origins']})
    if len(result) != 1598:
        raise RuntimeError('Missing development obligation')
    return result


def scope_inputs(packet):
    return development_inputs() + [
        {'id': r['id'], 'kind': 'prospective', 'class': r['class'], 'family': r['family'],
         'output': r['text'], 'expected': r['expected']} for r in packet['cases']]


def score_item(item):
    observed = e.screen(item['task_row'], item['output']) if 'task_row' in item else e.classify(item['output'])
    return {**{k: v for k, v in item.items() if k != 'task_row'}, 'observed': observed}


def scope_worker(repeat):
    import psutil
    start, proc = time.monotonic(), psutil.Process()
    cpu = sum(proc.cpu_times()[:2])
    packet = challenge_binding()
    folder = d.OUT / 'scope' / repeat
    folder.mkdir(parents=True, exist_ok=False)
    observations = []
    with (folder / 'raw.jsonl').open('x', encoding='utf-8') as stream:
        for item in scope_inputs(packet):
            observations.append(storage())
            if time.monotonic() - start > 1800:
                raise RuntimeError('Scope budget exceeded')
            append(stream, score_item(item))
    binding()
    write(folder / 'receipt.json', {'pid': os.getpid(), 'repeat': repeat,
        'raw_sha256': sha(folder / 'raw.jsonl'), 'freeze_sha256': sha(d.OUT / 'freeze.json'),
        'challenge_sha256': sha(d.OUT / 'challenge.json'), 'storage': observations,
        'wall_seconds': time.monotonic() - start, 'cpu_seconds': sum(proc.cpu_times()[:2]) - cpu})


def scope_summary(raw, expected):
    if len(raw) != len(expected) or raw != [score_item(item) for item in expected]:
        raise RuntimeError('Missing, reordered or unrecomputable scope rows')
    errors, counts, false_safe = {}, {}, []
    for row in raw:
        group = row.get('class', row['kind']) if row['kind'] == 'prospective' else row['kind']
        counts[group] = counts.get(group, 0) + 1
        observed = row['observed'].get('mechanical_pass', row['observed'].get('classification'))
        if observed != row['expected']:
            errors.setdefault(group, []).append(row['id'])
        if row['expected'] in (e.UNSAFE, e.UNKNOWN) and observed == e.SAFE:
            false_safe.append(row['id'])
    if any(counts.get(group) != 100 for group in challenge.LABELS):
        raise RuntimeError('Missing prospective class')
    return {'verdict': 'FINITE_SCOPE_PREREQUISITE_PASSED' if not errors else 'SCOPE_PREREQUISITE_FAILED',
            'counts': counts, 'errors': errors, 'false_safe': false_safe}


def hashes(base):
    return {relative(p): sha(p) for p in sorted(base.rglob('*')) if p.is_file() and p.name != 'decision.json'}


def positive(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def verify_scope():
    packet = challenge_binding()
    base = d.OUT / 'scope'
    if (base / 'stop.json').exists():
        raise RuntimeError('Interrupted scope cannot pass')
    first, replay = (rows(base / r / 'raw.jsonl') for r in ('initial', 'replay'))
    if first != replay or sha(base / 'initial/raw.jsonl') != sha(base / 'replay/raw.jsonl'):
        raise RuntimeError('Scope replay changed')
    pids = []
    for repeat in ('initial', 'replay'):
        receipt = read(base / repeat / 'receipt.json')
        operation = read(base / f'{repeat}_operation.json')
        if (receipt['repeat'] != repeat or receipt['raw_sha256'] != sha(base / repeat / 'raw.jsonl')
                or receipt['freeze_sha256'] != sha(d.OUT / 'freeze.json')
                or receipt['challenge_sha256'] != sha(d.OUT / 'challenge.json')
                or operation['returncode'] != 0 or operation['raw_sha256'] != receipt['raw_sha256']
                or not positive(receipt['wall_seconds']) or receipt['wall_seconds'] > 1800
                or not positive(receipt['cpu_seconds']) or not positive(receipt['pid'])
                or len(receipt['storage']) != len(first)
                or any(s['free_bytes'] < 10 * 1024**3 or s['evidence_bytes'] > 1024**3 for s in receipt['storage'])):
            raise RuntimeError('Invalid scope receipt or resource observation')
        pids.append(receipt['pid'])
    complete, launch = read(base / 'complete.json'), read(base / 'launch.json')
    if (len(set(pids)) != 2 or launch['freeze_sha256'] != sha(d.OUT / 'freeze.json')
            or launch['budget_seconds'] != 1800 or not positive(complete['elapsed_seconds'])
            or complete['elapsed_seconds'] > 1800):
        raise RuntimeError('Incomplete scope controller')
    return {**scope_summary(first, scope_inputs(packet)), 'evidence_hashes': hashes(base)}


def require_scope():
    result = verify_scope()
    if read(d.OUT / 'scope/decision.json') != result or result['verdict'] != 'FINITE_SCOPE_PREREQUISITE_PASSED':
        raise RuntimeError('Scope prerequisite not satisfied')


def child(command, repeat, folder, seconds):
    argv = [sys.executable, '-u', '-m', 'experiments.count_feedback_r112.execution.run', command, '--repeat', repeat]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    start = time.monotonic()
    try:
        done = subprocess.run(argv, cwd=d.ROOT, env=env, capture_output=True, text=True, timeout=seconds)
        value = {'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr}
    except subprocess.TimeoutExpired as error:
        def text(v):
            return v.decode(errors='replace') if isinstance(v, bytes) else v
        value = {'returncode': None, 'stdout': text(error.stdout), 'stderr': text(error.stderr), 'timeout': seconds}
    return {**value, 'argv': argv, 'elapsed_seconds': time.monotonic() - start,
            'raw_sha256': sha(folder / 'raw.jsonl') if (folder / 'raw.jsonl').is_file() else None}


def scope():
    storage()
    challenge_binding()
    commit = published([d.OUT / 'freeze.json', d.OUT / 'challenge.json'])
    base = d.OUT / 'scope'
    write(base / 'launch.json', {'freeze_sha256': sha(d.OUT / 'freeze.json'), 'published_commit': commit,
                               'budget_seconds': 1800, 'started_unix': time.time()})
    start = time.monotonic()
    try:
        for repeat in ('initial', 'replay'):
            remaining = 1800 - (time.monotonic() - start)
            if remaining <= 0:
                raise RuntimeError('Scope controller deadline exceeded')
            operation = child('scope-worker', repeat, base / repeat, remaining)
            write(base / f'{repeat}_operation.json', operation)
            if operation['returncode'] != 0:
                raise RuntimeError('Scope child stopped')
        write(base / 'complete.json', {'elapsed_seconds': time.monotonic() - start})
        result = verify_scope()
        write(base / 'decision.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write(base / 'stop.json', {'error': repr(error), 'elapsed_seconds': time.monotonic() - start,
                                   'failure_owner': 'EVALUATOR_OR_RESOURCE', 'model_calls': 0})
        raise


def score_record(ref, stage, prompt, prompt_ids, tokens, tok, elapsed, free, draft=None):
    decoded = tok.decode(tokens, skip_special_tokens=True)
    output = decoded.strip()
    return {**{k: ref[k] for k in ('id', 'pair', 'panel', 'task')}, 'stage': stage,
            'prompt': prompt, 'prompt_ids': prompt_ids, 'token_ids': tokens,
            'decoded_output': decoded, 'output': output, 'screen': e.screen(ref, output),
            'r111_screen': e.old.screen(ref, output),
            'draft_sentence_count': None if stage == 'draft' else d.sentence_count(draft),
            'provided_sentence_count': d.sentence_count(draft) if stage == 'count_feedback' and ref['task'] in d.TASKS else None,
            'stop_reason': 'eos' if tokens and tokens[-1] == tok.eos_token_id else 'budget',
            'output_utf8_bytes': len(output.encode('utf-8')), 'output_characters': len(output),
            'wall_seconds': elapsed, 'storage_before': free}


def validate_live(raw, tok):
    if len(raw) != 288:
        raise RuntimeError('Missing draft/revision raw rows')
    size = len(tok)  # Cache only the immutable vocabulary size, not validation outcomes.
    for index, ref in enumerate(d.population()):
        draft = None
        for offset, stage in enumerate(d.STAGES):
            row = raw[3 * index + offset]
            prompt = d.render(ref, stage, tok, draft)
            ids = tok.encode(prompt, add_special_tokens=False)
            tokens = row['token_ids']
            if (not tokens or len(tokens) > 640 or len(ids) + 640 > 2048
                    or any(type(v) is not int or not 0 <= v < size for v in tokens)
                    or tok.eos_token_id in tokens[:-1]
                    or (tokens[-1] != tok.eos_token_id and len(tokens) != 640)
                    or not positive(row['wall_seconds'])):
                raise RuntimeError('Invalid token, termination, context or timing evidence')
            free = row['storage_before']
            if free['free_bytes'] < 10 * 1024**3 or free['evidence_bytes'] > 1024**3:
                raise RuntimeError('Storage observation violates gate')
            expected = score_record(ref, stage, prompt, ids, tokens, tok, row['wall_seconds'], free, draft)
            if row != expected:
                raise RuntimeError('Unrecomputable identity, conditioning, tokens, output or score')
            if stage == 'draft':
                draft = row['decoded_output']  # Full decoded draft, not selected/repaired/scored text.
    return raw


def scientific(raw):
    return [{k: v for k, v in row.items() if k not in ('wall_seconds', 'storage_before')} for row in raw]


def unchanged_controls(raw):
    matches = []
    for i, ref in enumerate(d.population()):
        if ref['task'] in d.TASKS:
            continue
        a, b = raw[3*i+1:3*i+3]
        if any(a[k] != b[k] for k in ('prompt', 'prompt_ids', 'token_ids', 'decoded_output')):
            raise RuntimeError('Unchanged-input negative control mismatch')
        matches.append(ref['id'])
    if len(matches) != 72:
        raise RuntimeError('Missing unchanged-input controls')
    return matches


def live_decision(raw):
    controls = unchanged_controls(raw)
    arms = {s: [r for r in raw if r['stage'] == s] for s in d.STAGES}
    cells = {s: old.h.summarize(v) for s, v in arms.items()}
    old_scores = {}
    for stage, values in arms.items():
        values = copy.deepcopy(values)
        for row in values:
            row['screen'] = row['r111_screen']
        old_scores[stage] = old.h.summarize(values)
    primary = old.bootstrap_delta(arms['count_feedback'], arms['revision'], True)
    quality = old.bootstrap_delta(arms['count_feedback'], arms['revision'])
    usable_draft = not cells['draft']['truncations'] and not any(r['screen']['collapse'] for r in arms['draft'])
    feedback_pass = usable_draft and cells['count_feedback']['qualified_mechanical_only']
    revision_pass = usable_draft and cells['revision']['qualified_mechanical_only']
    if feedback_pass and primary['ci95'][0] > 0 and quality['ci95'][0] >= -.10:
        verdict = 'BOUNDED_COUNT_ASSISTED_REVISION_IMPROVEMENT'
    elif feedback_pass:
        verdict = 'COUNT_ASSISTED_CONTROL_QUALIFIED_NO_SUPPORTED_GAIN'
    elif revision_pass:
        verdict = 'REVISION_CONTROL_QUALIFIED_FEEDBACK_NOT_QUALIFIED'
    elif cells['draft']['qualified_mechanical_only']:
        verdict = 'DRAFT_CONTROL_QUALIFIED_REVISIONS_NOT_QUALIFIED'
    else:
        verdict = 'CONTROL_QUALIFICATION_FAILED'
    return {'verdict': verdict, 'cells': cells, 'r111_scores_descriptive': old_scores,
            'primary_sentence_adherence': primary, 'all_task_quality': quality,
            'unchanged_input_matches': controls,
            'unresolved': {s: sum(r['screen']['approval_scope']['classification'] == e.UNKNOWN for r in v)
                           for s, v in arms.items()},
            'failure_owner': 'EXPERIMENTAL_PREREQUISITE' if verdict == 'CONTROL_QUALIFICATION_FAILED' else 'NONE_WITHIN_MECHANICAL_SCOPE',
            'abi_acquisition': 'NOT_TESTED', 'native_layercake': 'NOT_TESTED', 'english_artifact': 'NONE',
            'training_steps': 0, 'recipient_calls': 0, 'full_moonshot': 'NOT_PROVEN',
            'scope': 'Conventional count-assisted two-call mechanical workflow, not evaluator-independent English'}


def telemetry(path):
    argv = ['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK']
    done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    write(path, {'argv': argv, 'returncode': done.returncode, 'stdout': done.stdout,
                 'stderr': done.stderr, 'storage': storage(), 'settings_changed': [], 'observed_unix': time.time()})
    if done.returncode != 0:
        raise RuntimeError('Read-only hardware telemetry failed')


def execute(repeat):
    import psutil
    start, proc = time.monotonic(), psutil.Process()
    cpu = sum(proc.cpu_times()[:2])
    binding()
    require_scope()
    if read(d.OUT / 'live/launch.json')['freeze_sha256'] != sha(d.OUT / 'freeze.json'):
        raise RuntimeError('Missing live launch binding')
    folder = d.OUT / 'live' / repeat
    folder.mkdir(parents=True, exist_ok=False)
    peak, done = [proc.memory_info().rss], threading.Event()
    def sample():
        while not done.wait(.05):
            peak[0] = max(peak[0], proc.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(4)
        torch.manual_seed(108001)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        tok = old.runtime.tokenizer('control')
        model = AutoModelForCausalLM.from_pretrained(old.runtime.snapshot(), local_files_only=True,
                    trust_remote_code=False, dtype=torch.bfloat16).to('cuda').eval()
        for p in model.parameters():
            p.requires_grad_(False)
            if p.dtype != torch.bfloat16:
                raise RuntimeError('Source precision changed')
        with torch.inference_mode(), (folder / 'raw.jsonl').open('x', encoding='utf-8') as stream:
            for index, ref in enumerate(d.population()):
                draft = None
                for stage in d.STAGES:
                    free = storage()
                    if time.monotonic() - start > 14400:
                        raise RuntimeError('Source process budget exceeded')
                    prompt = d.render(ref, stage, tok, draft)
                    ids = tok.encode(prompt, add_special_tokens=False)
                    if len(ids) + 640 > 2048:
                        raise RuntimeError('Complete revision context does not fit; never truncate')
                    tensor = torch.tensor([ids], device='cuda')
                    torch.cuda.synchronize()
                    began = time.monotonic()
                    generated = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                        do_sample=False, max_new_tokens=640, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                    torch.cuda.synchronize()
                    record = score_record(ref, stage, prompt, ids, generated[0, len(ids):].tolist(),
                                          tok, time.monotonic() - began, free, draft)
                    append(stream, record)
                    if stage == 'draft':
                        draft = record['decoded_output']
                if (index + 1) % 16 == 0:
                    print(json.dumps({'repeat': repeat, 'triples_of_calls': index + 1}), flush=True)
        validate_live(rows(folder / 'raw.jsonl'), tok)
        binding()
        write(folder / 'receipt.json', {'pid': os.getpid(), 'repeat': repeat, 'calls': 288,
            'raw_sha256': sha(folder / 'raw.jsonl'), 'freeze_sha256': sha(d.OUT / 'freeze.json'),
            'wall_seconds': time.monotonic() - start, 'cpu_seconds': sum(proc.cpu_times()[:2]) - cpu,
            'sampled_peak_rss_bytes': peak[0], 'rss_interval_seconds': .05,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'parameters': sum(p.numel() for p in model.parameters()), 'precision': 'bfloat16',
            'device': torch.cuda.get_device_name(), 'training_steps': 0, 'recipient_calls': 0,
            'gpu_setting_changes': []})
    finally:
        done.set()
        sampler.join(timeout=1)


def derive_live():
    binding()
    require_scope()
    base = d.OUT / 'live'
    if (base / 'stop.json').exists():
        raise RuntimeError('Interrupted source study cannot qualify')
    tok = old.runtime.tokenizer('control')
    raw, receipts = [], []
    for repeat in ('initial', 'replay'):
        current = validate_live(rows(base / repeat / 'raw.jsonl'), tok)
        receipt = read(base / repeat / 'receipt.json')
        operation = read(base / f'{repeat}_operation.json')
        if (receipt['raw_sha256'] != sha(base / repeat / 'raw.jsonl')
                or receipt['freeze_sha256'] != sha(d.OUT / 'freeze.json')
                or receipt['repeat'] != repeat or receipt['calls'] != 288
                or receipt['parameters'] != old.runtime.source_parameter_count()
                or receipt['precision'] != 'bfloat16' or receipt['training_steps'] != 0
                or receipt['rss_interval_seconds'] != .05 or not positive(receipt['pid'])
                or receipt['recipient_calls'] != 0 or receipt['gpu_setting_changes'] != []
                or operation['returncode'] != 0 or operation['raw_sha256'] != receipt['raw_sha256']
                or receipt['wall_seconds'] > 14400
                or any(not positive(receipt[k]) for k in ('wall_seconds', 'cpu_seconds', 'sampled_peak_rss_bytes', 'peak_cuda_allocated_bytes'))
                or sum(r['wall_seconds'] for r in current) > receipt['wall_seconds']):
            raise RuntimeError('Invalid source receipt or accounting')
        for when in ('before', 'after'):
            resource = read(base / f'{repeat}_{when}.json')
            if (resource['returncode'] != 0 or resource['settings_changed'] != []
                    or resource['argv'] != ['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK']
                    or not resource['stdout'].strip()
                    or resource['storage']['free_bytes'] < 10 * 1024**3
                    or resource['storage']['evidence_bytes'] > 1024**3):
                raise RuntimeError('Invalid hardware/storage evidence')
        unchanged_controls(current)
        raw.append(current)
        receipts.append(receipt)
    complete, launch = read(base / 'complete.json'), read(base / 'launch.json')
    if (scientific(raw[0]) != scientific(raw[1]) or len({r['pid'] for r in receipts}) != 2
            or launch['freeze_sha256'] != sha(d.OUT / 'freeze.json') or launch['budget_seconds'] != 30600
            or not positive(complete['elapsed_seconds']) or complete['elapsed_seconds'] > 30600
            or sum(r['wall_seconds'] for r in receipts) > complete['elapsed_seconds']):
        raise RuntimeError('Changed replay or invalid controller accounting')
    tokens = sum(len(r['token_ids']) for run in raw for r in run)
    if tokens > 368640:
        raise RuntimeError('Token budget exceeded')
    return {**live_decision(raw[0]), 'freeze_sha256': sha(d.OUT / 'freeze.json'),
            'costs': {'generation_calls': 576, 'generated_tokens': tokens,
                      'process_wall_seconds': sum(r['wall_seconds'] for r in receipts),
                      'cpu_process_seconds': sum(r['cpu_seconds'] for r in receipts),
                      'controller_loop_seconds': complete['elapsed_seconds'],
                      'peak_sampled_rss_bytes': max(r['sampled_peak_rss_bytes'] for r in receipts),
                      'peak_cuda_allocated_bytes': max(r['peak_cuda_allocated_bytes'] for r in receipts),
                      'generation_seconds': {s: sum(r['wall_seconds'] for run in raw for r in run if r['stage'] == s) for s in d.STAGES},
                      'workflow_generation_seconds': {s: sum(r['wall_seconds'] for run in raw for r in run if r['stage'] in ('draft', s)) for s in ('revision', 'count_feedback')},
                      'input_tokens': sum(len(r['prompt_ids']) for run in raw for r in run),
                      'input_utf8_bytes': sum(len(r['prompt'].encode('utf-8')) for run in raw for r in run),
                      'output_utf8_bytes': sum(r['output_utf8_bytes'] for run in raw for r in run),
                      'new_download_bytes': 0, 'source_preparation': 'Reused full-hash-verified R111 cache'},
            'evidence_hashes': hashes(base)}


def live():
    storage()
    binding()
    require_scope()
    commit = published([d.OUT / 'freeze.json', d.OUT / 'challenge.json', d.OUT / 'scope/decision.json'])
    base = d.OUT / 'live'
    write(base / 'launch.json', {'freeze_sha256': sha(d.OUT / 'freeze.json'), 'published_commit': commit,
                               'budget_seconds': 30600, 'started_unix': time.time()})
    start = time.monotonic()
    try:
        for repeat in ('initial', 'replay'):
            storage()
            binding()
            telemetry(base / f'{repeat}_before.json')
            remaining = 30600 - (time.monotonic() - start)
            if remaining <= 0:
                raise RuntimeError('Controller budget exceeded')
            operation = child('execute', repeat, base / repeat, min(14400, remaining))
            write(base / f'{repeat}_operation.json', operation)
            telemetry(base / f'{repeat}_after.json')
            if operation['returncode'] != 0:
                raise RuntimeError('Source child stopped; preserve evidence')
            print(json.dumps({'completed': repeat}), flush=True)
        write(base / 'complete.json', {'elapsed_seconds': time.monotonic() - start})
        result = derive_live()
        write(base / 'decision.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write(base / 'stop.json', {'error': repr(error), 'elapsed_seconds': time.monotonic() - start,
                                  'failure_owner': 'EXECUTION_OR_RESOURCE', 'native_layercake': 'NOT_TESTED'})
        raise


def verify():
    result = verify_scope()
    if read(d.OUT / 'scope/decision.json') != result:
        raise RuntimeError('Stale scope decision')
    if result['verdict'] == 'SCOPE_PREREQUISITE_FAILED':
        if (d.OUT / 'live').exists():
            raise RuntimeError('Source execution after failed evaluator')
        result.update(failure_owner='EVALUATOR', model_calls=0, abi_acquisition='NOT_TESTED', native_layercake='NOT_TESTED')
    else:
        result = derive_live()
        if read(d.OUT / 'live/decision.json') != result:
            raise RuntimeError('Stale source decision')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'materialize', 'scope', 'scope-worker', 'live', 'execute', 'verify'))
    parser.add_argument('--repeat', choices=('initial', 'replay'))
    args = parser.parse_args()
    if args.command in ('scope-worker', 'execute'):
        if args.repeat is None:
            parser.error('Worker requires a repeat identity')
        (scope_worker if args.command == 'scope-worker' else execute)(args.repeat)
    else:
        {'freeze': freeze, 'materialize': materialize, 'scope': scope, 'live': live, 'verify': verify}[args.command]()
