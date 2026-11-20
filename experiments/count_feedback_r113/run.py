"""R113 write-once execution; reuse R112 scientific functions without alteration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from experiments.count_feedback_r112.execution import run as parent

d, runtime = parent.d, parent.old.runtime
ROOT = d.ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/count_feedback_r113'
MODULE = 'experiments.count_feedback_r113.run'
MANIFEST = ROOT / 'results/review_2026_11_19/r112_interruption_manifest.json'
MANIFEST_SHA = 'dc02f3335f5b20643bb00b170a9686a034c9c8e540b304d737bf0bcb07f78654'
PREREG = '209dadea63572feee990454066d6e1ab603b2c8c'
TEST = ROOT / 'tests/test_count_feedback_r113.py'
read, rows, write, append, sha = parent.read, parent.rows, parent.write, parent.append, parent.sha
relative, git, positive = parent.relative, parent.git, parent.positive
# Direct aliases, not modified/reimplemented scientific scoring functions.
score_record, validate_live = parent.score_record, parent.validate_live
scientific, live_decision = parent.scientific, parent.live_decision
unchanged_controls = parent.unchanged_controls


def storage():
    value = {'free_bytes': shutil.disk_usage(ROOT).free,
             'evidence_bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}
    if value['free_bytes'] < 10 * 1024**3 or value['evidence_bytes'] > 1024**3:
        raise RuntimeError(f'Storage floor/cap violated: {value}')
    return value


def source_paths():
    return [HERE / '__init__.py', HERE / 'run.py', HERE / 'PROTOCOL.md', TEST]


def ancestry():
    if sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('R112 public manifest changed')
    m = read(MANIFEST)
    return {**{k: v['sha256'] for k, v in m['files'].items()},
            **m['host_inputs'], relative(MANIFEST): MANIFEST_SHA}


def inherited_scope():
    result = parent.verify_scope()
    if result != read(d.OUT / 'scope/decision.json') or result['verdict'] != 'FINITE_SCOPE_PREREQUISITE_PASSED':
        raise RuntimeError('Inherited finite scope prerequisite failed')
    return result


def telemetry(path):
    argv = ['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK']
    done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    value = {'argv': argv, 'returncode': done.returncode, 'stdout': done.stdout,
             'stderr': done.stderr, 'storage': storage(), 'settings_changed': [],
             'observed_unix': time.time()}
    write(path, value)
    if done.returncode != 0 or not done.stdout.strip():
        raise RuntimeError('Read-only telemetry failed')
    return value


def check_telemetry(value):
    if (value['argv'] != ['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK']
            or value['returncode'] != 0 or not value['stdout'].strip()
            or value['settings_changed'] != [] or not positive(value['observed_unix'])
            or value['storage']['free_bytes'] < 10 * 1024**3
            or value['storage']['evidence_bytes'] > 1024**3):
        raise RuntimeError('Invalid read-only telemetry/storage receipt')


def binding():
    f = read(OUT / 'freeze.json')
    inherited = ancestry()
    expected = set(inherited) | {relative(p) for p in source_paths()} | {
        relative(OUT / n) for n in ('baseline_inputs.json', 'inherited_scope.json', 'preflight.json')}
    old_freeze = read(d.OUT / 'freeze.json')
    if (f['format'] != 'r113-freeze/1' or set(f['files']) != expected
            or f['contract'] != d.contract() or f['packages'] != parent.old.package_versions()
            or f['source_files'] != old_freeze['source_files']
            or f['r112_manifest_sha256'] != MANIFEST_SHA
            or f['protocol_sha256'] != sha(HERE / 'PROTOCOL.md')):
        raise RuntimeError('Missing or changed R113 freeze/dependencies')
    for name, digest in f['files'].items():
        if sha(ROOT / name) != digest or (name in inherited and digest != inherited[name]):
            raise RuntimeError(f'Missing or changed frozen file: {name}')
    for p in source_paths():
        blob = subprocess.check_output(['git', 'show', f'{f["implementation_commit"]}:{relative(p)}'], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != f['files'][relative(p)]:
            raise RuntimeError('Uncommitted implementation')
    protocol = subprocess.check_output(['git', 'show', f'{PREREG}:{relative(HERE / "PROTOCOL.md")}'], cwd=ROOT)
    if hashlib.sha256(protocol).hexdigest() != f['protocol_sha256']:
        raise RuntimeError('Published protocol changed')
    if read(OUT / 'inherited_scope.json') != inherited_scope():
        raise RuntimeError('Stale inherited scope')
    tok = runtime.tokenizer('control')
    if (read(OUT / 'baseline_inputs.json') != parent.inputs(tok)
            or sha(OUT / 'baseline_inputs.json') != sha(d.OUT / 'baseline_inputs.json')):
        raise RuntimeError('Changed inherited input tokens')
    check_telemetry(read(OUT / 'preflight.json'))
    return f


def freeze():
    storage()
    if OUT.exists():
        raise RuntimeError('Preserve existing R113 freeze or partial preparation')
    parent.binding()  # Includes complete pinned source-cache rehash.
    scope = inherited_scope()
    commit = git('rev-parse', 'HEAD')
    files = ancestry()
    for p in source_paths():
        blob = subprocess.check_output(['git', 'show', f'{commit}:{relative(p)}'], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != sha(p):
            raise RuntimeError('Commit implementation before freeze')
        files[relative(p)] = sha(p)
    write(OUT / 'baseline_inputs.json', parent.inputs(runtime.tokenizer('control')))
    write(OUT / 'inherited_scope.json', scope)
    telemetry(OUT / 'preflight.json')
    for n in ('baseline_inputs.json', 'inherited_scope.json', 'preflight.json'):
        files[relative(OUT / n)] = sha(OUT / n)
    write(OUT / 'freeze.json', {'format': 'r113-freeze/1', 'implementation_commit': commit,
        'protocol_sha256': sha(HERE / 'PROTOCOL.md'), 'files': files,
        'contract': d.contract(), 'packages': parent.old.package_versions(),
        'source_files': read(d.OUT / 'freeze.json')['source_files'],
        'r112_manifest_sha256': MANIFEST_SHA, 'created_unix': time.time(),
        'scope': 'New full execution of exposed R112 panel; no resume or new blind holdout'})
    binding()
    print(json.dumps({'freeze_sha256': sha(OUT / 'freeze.json'), 'bound_files': len(files), 'model_calls': 0}), flush=True)


def execute(repeat):
    import psutil
    start, proc = time.monotonic(), psutil.Process()
    cpu = sum(proc.cpu_times()[:2])
    binding()
    if read(OUT / 'live/launch.json')['freeze_sha256'] != sha(OUT / 'freeze.json'):
        raise RuntimeError('Missing live launch binding')
    folder = OUT / 'live' / repeat
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
        tok = runtime.tokenizer('control')
        model = AutoModelForCausalLM.from_pretrained(runtime.snapshot(), local_files_only=True,
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
                        raise RuntimeError('Complete context does not fit; never truncate')
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
                    print(json.dumps({'repeat': repeat, 'triplets': index + 1}), flush=True)
        validate_live(rows(folder / 'raw.jsonl'), tok)
        binding()
        write(folder / 'receipt.json', {'pid': os.getpid(), 'repeat': repeat, 'calls': 288,
            'raw_sha256': sha(folder / 'raw.jsonl'), 'freeze_sha256': sha(OUT / 'freeze.json'),
            'wall_seconds': time.monotonic() - start, 'cpu_seconds': sum(proc.cpu_times()[:2]) - cpu,
            'sampled_peak_rss_bytes': peak[0], 'rss_interval_seconds': .05,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'parameters': sum(p.numel() for p in model.parameters()), 'precision': 'bfloat16',
            'device': torch.cuda.get_device_name(), 'training_steps': 0, 'recipient_calls': 0,
            'gpu_setting_changes': []})
    finally:
        done.set()
        sampler.join(timeout=1)


def child(repeat, seconds):
    argv = [sys.executable, '-u', '-m', MODULE, 'execute', '--repeat', repeat]
    start = time.monotonic()
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=seconds)
            code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            code = None
        value = {'returncode': code, 'stdout': stdout, 'stderr': stderr,
                 'pid': process.pid, 'timeout_seconds': seconds}
    path = OUT / 'live' / repeat / 'raw.jsonl'
    return {**value, 'argv': argv, 'elapsed_seconds': time.monotonic() - start,
            'raw_sha256': sha(path) if path.is_file() else None}


def check_receipt(receipt, operation, repeat, raw, freeze_sha, raw_sha, parameters):
    if (receipt['raw_sha256'] != raw_sha or receipt['freeze_sha256'] != freeze_sha
            or receipt['repeat'] != repeat or receipt['calls'] != 288
            or receipt['parameters'] != parameters or receipt['precision'] != 'bfloat16'
            or receipt['training_steps'] != 0 or receipt['recipient_calls'] != 0
            or receipt['gpu_setting_changes'] != [] or not receipt['device'].strip()
            or receipt['rss_interval_seconds'] != .05 or not positive(receipt['pid'])
            or operation['returncode'] != 0 or operation['pid'] != receipt['pid']
            or operation['argv'][1:] != ['-u', '-m', MODULE, 'execute', '--repeat', repeat]
            or not operation['argv'][0] or operation['raw_sha256'] != raw_sha
            or not positive(operation['timeout_seconds']) or operation['timeout_seconds'] > 14400
            or not positive(operation['elapsed_seconds'])
            or receipt['wall_seconds'] > 14400
            or any(not positive(receipt[k]) for k in ('wall_seconds', 'cpu_seconds', 'sampled_peak_rss_bytes', 'peak_cuda_allocated_bytes'))
            or sum(r['wall_seconds'] for r in raw) > receipt['wall_seconds']
            or receipt['wall_seconds'] > operation['elapsed_seconds']):
        raise RuntimeError('Invalid source receipt/operation/accounting')


def derive():
    binding()
    base = OUT / 'live'
    if (base / 'stop.json').exists():
        raise RuntimeError('Interrupted R113 cannot qualify')
    tok, raw, receipts, operations = runtime.tokenizer('control'), [], [], []
    for repeat in ('initial', 'replay'):
        current = validate_live(rows(base / repeat / 'raw.jsonl'), tok)
        receipt = read(base / repeat / 'receipt.json')
        operation = read(base / f'{repeat}_operation.json')
        check_receipt(receipt, operation, repeat, current, sha(OUT / 'freeze.json'),
                      sha(base / repeat / 'raw.jsonl'), runtime.source_parameter_count())
        for when in ('before', 'after'):
            check_telemetry(read(base / f'{repeat}_{when}.json'))
        unchanged_controls(current)
        raw.append(current)
        receipts.append(receipt)
        operations.append(operation)
    complete, launch = read(base / 'complete.json'), read(base / 'launch.json')
    if (scientific(raw[0]) != scientific(raw[1]) or len({r['pid'] for r in receipts}) != 2
            or launch['freeze_sha256'] != sha(OUT / 'freeze.json') or launch['budget_seconds'] != 30600
            or not positive(launch['started_unix']) or len(launch['published_commit']) != 40
            or not positive(complete['elapsed_seconds']) or complete['elapsed_seconds'] > 30600
            or sum(r['elapsed_seconds'] for r in operations) > complete['elapsed_seconds']):
        raise RuntimeError('Missing exact fresh replay or invalid controller accounting')
    frozen_blob = subprocess.check_output(['git', 'show', f'{launch["published_commit"]}:{relative(OUT / "freeze.json")}'], cwd=ROOT)
    if hashlib.sha256(frozen_blob).hexdigest() != launch['freeze_sha256']:
        raise RuntimeError('Launch publication binding changed')
    tokens = sum(len(r['token_ids']) for run in raw for r in run)
    if tokens > 368640:
        raise RuntimeError('Token cap exceeded')
    return {**live_decision(raw[0]), 'experiment': 'R113', 'panel': 'Exposed R112 panel; not fresh blind data',
        'freeze_sha256': sha(OUT / 'freeze.json'), 'inherited_r112_manifest_sha256': MANIFEST_SHA,
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
            'new_download_bytes': 0, 'source_preparation': 'Full-hash-verified reused R112 cache'},
        'evidence_hashes': parent.hashes(base)}


def live():
    storage()
    binding()
    commit = parent.published([OUT / 'freeze.json', HERE / 'PROTOCOL.md'])
    base = OUT / 'live'
    write(base / 'launch.json', {'freeze_sha256': sha(OUT / 'freeze.json'), 'published_commit': commit,
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
            operation = child(repeat, min(14400, remaining))
            write(base / f'{repeat}_operation.json', operation)
            telemetry(base / f'{repeat}_after.json')
            if operation['returncode'] != 0:
                raise RuntimeError('Source child stopped; preserve evidence')
            print(json.dumps({'completed': repeat}), flush=True)
        write(base / 'complete.json', {'elapsed_seconds': time.monotonic() - start})
        result = derive()
        write(base / 'decision.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write(base / 'stop.json', {'error': repr(error), 'elapsed_seconds': time.monotonic() - start,
                                  'failure_owner': 'EXECUTION_OR_RESOURCE', 'native_layercake': 'NOT_TESTED'})
        raise


def verify():
    result = derive()
    if read(OUT / 'live/decision.json') != result:
        raise RuntimeError('Stale R113 decision')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'live', 'execute', 'verify'))
    parser.add_argument('--repeat', choices=('initial', 'replay'))
    args = parser.parse_args()
    if args.command == 'execute':
        if args.repeat is None:
            parser.error('Worker repeat required')
        execute(args.repeat)
    else:
        {'freeze': freeze, 'live': live, 'verify': verify}[args.command]()
