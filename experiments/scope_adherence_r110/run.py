"""Write-once R110 controller. No training, recipient loading or hardware setters."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time

from . import challenge, design as d, evaluator as e, execution as x, register


def freeze():
    if (x.OUT/'implementation_freeze.json').exists():
        raise RuntimeError('Preserve the implementation freeze')
    # Freezing hashes/code is not an experiment stage or a model call.
    register.verify()
    files = x.dependencies()
    commit = x.git('rev-parse', 'HEAD')
    for name in (*(x.relative(d.HERE/n) for n in x.IMPLEMENTATION_FILES), x.TEST_FILE):
        import hashlib
        blob = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=d.ROOT)
        if hashlib.sha256(blob).hexdigest() != files[name]:
            raise RuntimeError('Commit implementation before freezing')
    x.write(x.OUT/'implementation_freeze.json', {
        'format': 'r110-implementation/1', 'implementation_commit': commit,
        'registration_sha256': x.REG_SHA, 'files': files, 'contract': d.contract(),
        'source_files': x.read(d.OUT/'registration.json')['source_files'],
        'packages': x.package_versions(), 'python': platform.python_version(),
        'platform': platform.platform(), 'created_unix': time.time(),
        'gpu_setting_changes': [], 'model_calls_at_freeze': 0,
        'authorization': 'User requested implementation/freeze and conditional execution; no gate change',
        'scope': 'Implementation identity only; not evaluator qualification or model evidence'})
    x.freeze_binding()
    print(json.dumps({'implementation_freeze_sha256': x.sha(x.OUT/'implementation_freeze.json'),
                      'implementation_commit': commit, 'scientific_result': 'NOT_EXECUTED'}), flush=True)


def preflight():
    value = x.storage_observation()
    passed = value['free_bytes'] >= value['minimum_free_bytes'] and value['evidence_bytes'] <= value['maximum_evidence_bytes']
    packet = {'format': 'r110-storage-preflight/1', 'observed_unix': time.time(), 'storage': value,
              'status': 'STORAGE_READY' if passed else 'BLOCKED_STORAGE_BEFORE_EXECUTION',
              'model_calls_in_preflight': 0, 'gpu_setting_changes': [],
              'scope': 'Read-only resource observation; not a launched experiment or scientific outcome'}
    path = x.OUT/'preflights'/f'{time.time_ns()}.json'
    x.write(path, packet)
    print(json.dumps({**packet, 'receipt': x.relative(path)}), flush=True)
    return passed


def materialize():
    x.guard_storage()
    frozen = x.freeze_binding()
    x.require_published([x.OUT/'implementation_freeze.json'])
    packet = challenge.generate(frozen['implementation_commit'], x.REG_SHA)
    x.write(x.OUT/'challenge.json', packet)
    print(json.dumps({'challenge_sha256': x.sha(x.OUT/'challenge.json'), 'cases': 600,
                      'next': 'Commit and publish this challenge before any scoring'}), flush=True)


def scope_worker(repeat):
    import psutil
    started, process = time.monotonic(), psutil.Process()
    cpu = sum(process.cpu_times()[:2])
    x.guard_storage()
    packet = x.challenge_binding()
    launch = x.read(x.OUT/'scope/launch.json')
    if launch['freeze_sha256'] != x.sha(x.OUT/'implementation_freeze.json'):
        raise RuntimeError('Scope worker requires its controller launch')
    folder = x.OUT/'scope'/repeat
    folder.mkdir(parents=True, exist_ok=False)
    # Every development/challenge result is retained, including failures.
    inputs = x.scope_inputs(packet)
    observed = []
    storage = []
    with (folder/'raw.jsonl').open('x', encoding='utf-8') as stream:
        for item in inputs:
            storage.append(x.guard_storage())
            if time.monotonic()-started > 1800:
                raise RuntimeError('Offline evaluator budget exceeded')
            row = x.score_scope_item(item)
            observed.append(row)
            x.append(stream, row)
    x.freeze_binding()
    x.write(folder/'receipt.json', {'repeat': repeat, 'pid': os.getpid(), 'rows': len(observed),
        'model_calls': 0, 'raw_sha256': x.sha(folder/'raw.jsonl'),
        'freeze_sha256': x.sha(x.OUT/'implementation_freeze.json'),
        'challenge_sha256': x.sha(x.OUT/'challenge.json'), 'storage_observations': storage,
        'elapsed_seconds': time.monotonic()-started, 'cpu_seconds': sum(process.cpu_times()[:2])-cpu})


def child(command, folder, seconds, **options):
    argv = [sys.executable, '-u', '-m', 'experiments.scope_adherence_r110.run', command]
    for key, value in options.items():
        argv.extend(['--'+key, value])
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    started = time.monotonic()
    try:
        done = subprocess.run(argv, cwd=d.ROOT, env=env, capture_output=True, text=True, timeout=seconds)
        result = {'argv': argv, 'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr}
    except subprocess.TimeoutExpired as error:
        text = lambda v: v.decode(errors='replace') if isinstance(v, bytes) else v
        result = {'argv': argv, 'returncode': None, 'stdout': text(error.stdout), 'stderr': text(error.stderr),
                  'timeout_seconds': error.timeout}
    result.update(elapsed_seconds=time.monotonic()-started,
                  raw_sha256=x.sha(folder/'raw.jsonl') if (folder/'raw.jsonl').is_file() else None)
    return result


def scope():
    x.guard_storage()
    x.challenge_binding()
    published = x.require_published([x.OUT/'implementation_freeze.json', x.OUT/'challenge.json'])
    base = x.OUT/'scope'
    x.write(base/'launch.json', {'pid': os.getpid(), 'started_unix': time.time(), 'budget_seconds': 1800,
        'published_commit': published, 'freeze_sha256': x.sha(x.OUT/'implementation_freeze.json'),
        'challenge_sha256': x.sha(x.OUT/'challenge.json')})
    started = time.monotonic()
    try:
        for repeat in ('initial', 'replay'):
            x.guard_storage()
            remaining = 1800-(time.monotonic()-started)
            if remaining <= 0:
                raise RuntimeError('Offline evaluator controller budget exceeded')
            operation = child('scope-worker', base/repeat, remaining, repeat=repeat)
            x.write(base/f'{repeat}_operation.json', operation)
            if operation['returncode'] != 0:
                raise RuntimeError('Scope execution stopped; preserve operation evidence')
        x.write(base/'complete.json', {'elapsed_seconds': time.monotonic()-started,
            'freeze_sha256': x.sha(x.OUT/'implementation_freeze.json'),
            'challenge_sha256': x.sha(x.OUT/'challenge.json')})
        result = x.verify_scope()
        x.write(base/'decision.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        x.write(base/'stop.json', {'error': repr(error), 'elapsed_seconds': time.monotonic()-started,
                                 'failure_owner': 'EVALUATOR_OR_RESOURCE', 'gpu_calls': 0})
        raise


def telemetry(name):
    import psutil
    commands = []
    for argv in (['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK'],
                 ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,clocks.sm,power.draw,temperature.gpu', '--format=csv,noheader']):
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        commands.append({'argv': argv, 'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr})
    battery = psutil.sensors_battery()
    value = {'commands': commands, 'observed_unix': time.time(), 'settings_changed': [],
             'storage': x.guard_storage(), 'virtual_memory': psutil.virtual_memory()._asdict(),
             'battery': battery._asdict() if battery else None}
    x.write(x.OUT/'live/resources'/f'{name}.json', value)
    if any(r['returncode'] for r in commands):
        raise RuntimeError('Read-only GPU observation failed')


def execute(interface, repeat):
    import psutil
    started, process = time.monotonic(), psutil.Process()
    cpu = sum(process.cpu_times()[:2])
    x.guard_storage()
    frozen = x.freeze_binding()
    x.require_scope_pass()
    base = x.OUT/'live'
    launch = x.read(base/'launch.json')
    if launch['freeze_sha256'] != x.sha(x.OUT/'implementation_freeze.json'):
        raise RuntimeError('Generation requires a bound controller launch')
    expected_index = d.contract()['source_process_order'].index([interface, repeat])
    for prior_interface, prior_repeat in d.contract()['source_process_order'][:expected_index]:
        verify_cell(prior_interface, prior_repeat)
    folder = base/interface/repeat
    folder.mkdir(parents=True, exist_ok=False)
    peak = [process.memory_info().rss]
    stop_sampler = threading.Event()

    def sample():
        while not stop_sampler.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)

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
        tok = x.runtime.tokenizer('control')
        model = AutoModelForCausalLM.from_pretrained(x.runtime.snapshot(), local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16).to('cuda')
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(p.dtype != torch.bfloat16 for p in model.parameters()):
            raise RuntimeError('Source precision changed')
        parameters = sum(p.numel() for p in model.parameters())
        with torch.inference_mode(), (folder/'raw.jsonl').open('x', encoding='utf-8') as stream:
            for index, (ref, inp) in enumerate(zip(d.population(), x.inputs(interface))):
                storage = x.guard_storage()
                if time.monotonic()-started > 14400:
                    raise RuntimeError('Source process deadline exceeded')
                ids = torch.tensor([inp['prompt_ids']], device='cuda')
                torch.cuda.synchronize()
                before = time.monotonic()
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    max_new_tokens=640, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                elapsed = time.monotonic()-before
                tokens = generated[0, len(inp['prompt_ids']):].tolist()
                decoded = tok.decode(tokens, skip_special_tokens=True)
                answer = decoded.strip()
                x.append(stream, {**{k: ref[k] for k in ('id', 'pair', 'panel', 'task')},
                    'interface': interface, 'prompt': inp['prompt'], 'prompt_ids': inp['prompt_ids'],
                    'token_ids': tokens, 'decoded_output': decoded, 'output': answer,
                    'screen': e.screen(ref, answer), 'legacy_screen': x.h.screen(ref, answer),
                    'stop_reason': 'eos' if tokens and tokens[-1] == tok.eos_token_id else 'budget',
                    'output_utf8_bytes': len(answer.encode()), 'output_characters': len(answer),
                    'wall_seconds': elapsed, 'storage_before': storage})
                if (index+1) % 16 == 0:
                    print(json.dumps({'interface': interface, 'repeat': repeat, 'rows': index+1}), flush=True)
        x.freeze_binding()
        if x.runtime.inventory(x.runtime.snapshot()) != frozen['source_files']:
            raise RuntimeError('Source mutated')
        x.validate_live(x.rows(folder/'raw.jsonl'), interface, tok)
        peak[0] = max(peak[0], process.memory_info().rss)
        x.write(folder/'receipt.json', {'raw_sha256': x.sha(folder/'raw.jsonl'), 'pid': os.getpid(),
            'freeze_sha256': x.sha(x.OUT/'implementation_freeze.json'), 'interface': interface, 'repeat': repeat,
            'parameters': parameters, 'precision': 'bfloat16', 'calls': 96,
            'wall_seconds': time.monotonic()-started, 'cpu_process_seconds': sum(process.cpu_times()[:2])-cpu,
            'sampled_peak_rss_bytes': peak[0], 'rss_sampling_interval_seconds': .05,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'device': torch.cuda.get_device_name(),
            'optimizer_steps': 0, 'recipient_calls': 0, 'gpu_setting_changes': []})
    finally:
        stop_sampler.set()
        sampler.join(timeout=1)


def verify_cell(interface, repeat):
    folder = x.OUT/'live'/interface/repeat
    receipt, raw = x.read(folder/'receipt.json'), x.rows(folder/'raw.jsonl')
    if (receipt['raw_sha256'] != x.sha(folder/'raw.jsonl')
            or receipt['freeze_sha256'] != x.sha(x.OUT/'implementation_freeze.json')
            or receipt['interface'] != interface or receipt['repeat'] != repeat
            or receipt['calls'] != 96 or receipt['precision'] != 'bfloat16'
            or receipt['parameters'] != x.runtime.source_parameter_count()
            or receipt['optimizer_steps'] != 0 or receipt['recipient_calls'] != 0
            or receipt['gpu_setting_changes'] != [] or receipt['wall_seconds'] > 14400):
        raise RuntimeError('Missing or stale live execution receipt')
    for key in ('parameters', 'wall_seconds', 'cpu_process_seconds', 'sampled_peak_rss_bytes', 'peak_cuda_allocated_bytes'):
        if not x.positive_finite(receipt[key]):
            raise RuntimeError('Invalid cost measurement')
    if sum(r['wall_seconds'] for r in raw) > receipt['wall_seconds']:
        raise RuntimeError('Generation times exceed process time')
    x.validate_live(raw, interface, x.runtime.tokenizer('control'))
    return raw


def derive_live():
    x.freeze_binding()
    x.require_scope_pass()
    base = x.OUT/'live'
    if (base/'stop.json').exists():
        raise RuntimeError('Interrupted live study cannot qualify')
    arms, receipts = {}, []
    for interface in d.INTERFACES:
        first, replay = (verify_cell(interface, repeat) for repeat in ('initial', 'replay'))
        if x.scientific(first) != x.scientific(replay):
            raise RuntimeError('Scientific replay changed')
        arms[interface] = first
        receipts.extend(x.read(base/interface/repeat/'receipt.json') for repeat in ('initial', 'replay'))
    completion, launch = x.read(base/'complete.json'), x.read(base/'launch.json')
    if (launch['budget_seconds'] != 59400 or completion['process_order'] != d.contract()['source_process_order']
            or not x.positive_finite(completion['elapsed_seconds']) or completion['elapsed_seconds'] > 59400
            or len({r['pid'] for r in receipts}) != 4
            or sum(r['wall_seconds'] for r in receipts) > 57600
            or sum(r['wall_seconds'] for r in receipts) > completion['elapsed_seconds']):
        raise RuntimeError('Invalid source/controller accounting or process reuse')
    for packet in (launch, completion):
        if packet['freeze_sha256'] != x.sha(x.OUT/'implementation_freeze.json'):
            raise RuntimeError('Stale controller')
    for interface, repeat in d.contract()['source_process_order']:
        operation = x.read(base/'operations'/f'{interface}-{repeat}.json')
        if operation['returncode'] != 0 or operation['raw_sha256'] != x.sha(base/interface/repeat/'raw.jsonl'):
            raise RuntimeError('Missing or stale source operation')
        for when in ('before', 'after'):
            resource = x.read(base/'resources'/f'{interface}-{repeat}-{when}.json')
            if (resource['settings_changed'] != [] or len(resource['commands']) != 2
                    or any(c['returncode'] != 0 for c in resource['commands'])
                    or resource['storage']['free_bytes'] < 10*1024**3
                    or resource['storage']['evidence_bytes'] > 1024**3):
                raise RuntimeError('Missing or invalid hardware/storage observation')
    result = x.live_decision(arms['plain'], arms['response_contract'])
    total_tokens = sum(len(r['token_ids']) for i in d.INTERFACES for repeat in ('initial', 'replay')
                       for r in x.rows(base/i/repeat/'raw.jsonl'))
    if total_tokens > 245760 or sum(r['calls'] for r in receipts) != 384:
        raise RuntimeError('Call/token budget violated')
    result.update(format='r110-live-decision/1', freeze_sha256=x.sha(x.OUT/'implementation_freeze.json'),
        costs={'generation_calls': 384, 'generated_tokens': total_tokens,
               'process_wall_seconds': sum(r['wall_seconds'] for r in receipts),
               'cpu_process_seconds': sum(r['cpu_process_seconds'] for r in receipts),
               'maximum_sampled_rss_bytes': max(r['sampled_peak_rss_bytes'] for r in receipts),
               'maximum_cuda_allocated_bytes': max(r['peak_cuda_allocated_bytes'] for r in receipts),
               'controller_seconds': completion['elapsed_seconds'], 'new_download_bytes': 0,
               'source_preparation': 'Reused pinned R108/R109 source; full hashes verified',
               'source_cache_bytes': sum(v['bytes'] for v in x.read(d.OUT/'registration.json')['source_files'].values())},
        evidence_hashes={x.relative(p): x.sha(p) for p in sorted(base.rglob('*'))
                         if p.is_file() and p.name != 'decision.json'})
    return result


def live():
    x.guard_storage()
    x.freeze_binding()
    x.require_scope_pass()
    published = x.require_published([x.OUT/'implementation_freeze.json', x.OUT/'challenge.json', x.OUT/'scope/decision.json'])
    base = x.OUT/'live'
    x.write(base/'launch.json', {'pid': os.getpid(), 'started_unix': time.time(), 'budget_seconds': 59400,
        'published_commit': published, 'freeze_sha256': x.sha(x.OUT/'implementation_freeze.json')})
    started = time.monotonic()
    try:
        for interface, repeat in d.contract()['source_process_order']:
            x.guard_storage()
            x.freeze_binding()
            telemetry(f'{interface}-{repeat}-before')
            remaining = 59400-(time.monotonic()-started)
            if remaining <= 0:
                raise RuntimeError('Controller deadline exceeded')
            operation = child('execute', base/interface/repeat, min(14400, remaining), interface=interface, repeat=repeat)
            x.write(base/'operations'/f'{interface}-{repeat}.json', operation)
            telemetry(f'{interface}-{repeat}-after')
            if operation['returncode'] != 0:
                raise RuntimeError('Source process failed; preserve operation evidence')
            verify_cell(interface, repeat)
            print(json.dumps({'completed': [interface, repeat]}), flush=True)
        x.write(base/'complete.json', {'elapsed_seconds': time.monotonic()-started,
            'freeze_sha256': x.sha(x.OUT/'implementation_freeze.json'),
            'process_order': d.contract()['source_process_order']})
        result = derive_live()
        x.write(base/'decision.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        x.write(base/'stop.json', {'error': repr(error), 'elapsed_seconds': time.monotonic()-started,
                                 'failure_owner': 'EXECUTION_OR_RESOURCE', 'native_layercake': 'NOT_TESTED'})
        raise


def verify():
    scope_result = x.verify_scope()
    if x.read(x.OUT/'scope/decision.json') != scope_result:
        raise RuntimeError('Stale scope decision')
    if scope_result['verdict'] == 'SCOPE_PREREQUISITE_FAILED':
        if (x.OUT/'live').exists():
            raise RuntimeError('Source execution after failed scope gate')
        result = {'verdict': scope_result['verdict'], 'failure_owner': 'EVALUATOR', 'model_calls': 0,
                  'abi_acquisition': 'NOT_TESTED', 'native_layercake': 'NOT_TESTED'}
    else:
        result = derive_live()
        if x.read(x.OUT/'live/decision.json') != result:
            raise RuntimeError('Stale live decision')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'preflight', 'materialize', 'scope', 'scope-worker', 'live', 'execute', 'verify'))
    parser.add_argument('--repeat', choices=('initial', 'replay'))
    parser.add_argument('--interface', choices=d.INTERFACES)
    args = parser.parse_args()
    if args.command == 'scope-worker':
        if args.repeat is None:
            parser.error('scope-worker requires --repeat')
        scope_worker(args.repeat)
    elif args.command == 'execute':
        if args.repeat is None or args.interface is None:
            parser.error('execute requires --repeat and --interface')
        execute(args.interface, args.repeat)
    else:
        {'freeze': freeze, 'preflight': preflight, 'materialize': materialize,
         'scope': scope, 'live': live, 'verify': verify}[args.command]()
