"""Write-once bounded R109 execution and fail-closed recomputation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from . import core as c


def snapshot():
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(c.SOURCE, revision=c.REVISION, local_files_only=True)).resolve()
    if path.name != c.REVISION:
        raise RuntimeError('Wrong source revision')
    return path


def inventory(folder):
    return {name: {'sha256': c.sha(folder/name), 'bytes': (folder/name).stat().st_size} for name in c.SOURCE_FILES}


def source_parameter_count():
    from safetensors import safe_open
    total = 0
    for name in c.SOURCE_FILES:
        if name.endswith('.safetensors'):
            with safe_open(snapshot()/name, framework='pt', device='cpu') as tensors:
                total += sum(math.prod(tensors.get_slice(k).get_shape()) for k in tensors.keys())
    return total


def tokenizer(arm):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot() if arm == 'control' else c.checkpoint(arm),
                                       local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def guard_storage():
    free = shutil.disk_usage(c.ROOT).free
    used = sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())
    if free < c.MIN_FREE_BYTES or used > c.MAX_EVIDENCE_BYTES:
        raise RuntimeError(f'Storage guard: free={free}, evidence={used}')
    return {'free_bytes': free, 'evidence_bytes': used}


def telemetry(name):
    import psutil
    commands = []
    for argv in (['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK'],
                 ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,clocks.sm,power.draw,temperature.gpu', '--format=csv,noheader']):
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        commands.append({'argv': argv, 'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr})
    battery = psutil.sensors_battery()
    value = {'observed_unix': time.time(), 'commands': commands,
             'battery': battery._asdict() if battery else None,
             'virtual_memory': psutil.virtual_memory()._asdict(),
             'storage': guard_storage(), 'settings_changed': []}
    c.write(c.OUT/'resources'/f'{name}.json', value)
    if any(v['returncode'] for v in commands):
        raise RuntimeError('Read-only device observation failed')
    return value


def freeze():
    if (c.OUT/'freeze.json').exists():
        raise RuntimeError('Preserve prior freeze')
    guard_storage()
    c.previous.binding()
    git = lambda *args: subprocess.check_output(['git', *args], cwd=c.ROOT, text=True).strip()
    host_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.HOST, text=True).strip()
    if host_head != c.HOST_REVISION or subprocess.check_output(['git', 'status', '--porcelain'], cwd=c.HOST, text=True).strip():
        raise RuntimeError('Wrong or dirty host revision')
    population, audit = c.catalog(), c.audit()
    if audit['mismatches']:
        raise RuntimeError('Evaluator prerequisite failed')
    for arm in ('control', *c.ARMS):
        tok = tokenizer(arm)
        if any(len(tok.encode(c.render(r, arm, tok), add_special_tokens=False)) + c.MAX_NEW > c.CONTEXT for r in population):
            raise RuntimeError('Context budget exceeded before inference')
    source = c.read(c.previous.OUT/'source.json')
    if inventory(snapshot()) != source['files']:
        raise RuntimeError('Changed cached source')
    for name, value in (('source.json', source), ('population.json', population),
                        ('evaluator_audit.json', audit), ('matched_training_provenance.json', c.training_provenance())):
        c.write(c.OUT/name, value)
        if c.sha(c.OUT/name) != c.sha(c.previous.OUT/name):
            raise RuntimeError('Scientific replication input differs')
    telemetry('preflight')
    files = dict(c.read(c.previous.OUT/'freeze.json')['files'])
    paths = [*c.previous.OUT.rglob('*.json'), *c.previous.OUT.rglob('*.jsonl'),
             *c.HERE.glob('*.py'), c.HERE/'PROTOCOL.md',
             c.ROOT/'tests/test_trained_grounding_r109.py', *c.OUT.glob('*.json'),
             c.OUT/'resources/preflight.json']
    files.update({os.path.relpath(p, c.ROOT).replace(chr(92), '/'): c.sha(p) for p in paths})
    c.write(c.OUT/'freeze.json', {'format': 'r109-freeze/1', 'files': files,
        'source_model': c.SOURCE, 'source_revision': c.REVISION, 'source_files': source['files'],
        'host_revision': c.HOST_REVISION, 'abi_commit': git('rev-parse', 'HEAD'),
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'tokenizers', 'safetensors', 'numpy', 'psutil')},
        'resource_policy': c.resource_policy(), 'optimizer_steps': 0,
        'imported_recipient_parameters': 0, 'new_stored_logits_or_activations': 0})
    c.binding()
    print(json.dumps({'freeze_sha256': c.sha(c.OUT/'freeze.json'), 'files': len(files), 'fixtures': len(audit['cases'])}), flush=True)


def execute(arm, repeat):
    import psutil
    import torch
    from transformers import AutoModelForCausalLM
    from abi.layercake_core_loader import load_layercake_core
    from experiments.evaluator_control_r104.run import recipient_generate
    frozen = c.binding()
    if arm != 'control' and not c.summarize(verify_arm('control'))['qualified_mechanical_only']:
        raise RuntimeError('Control and replay must qualify before recipients')
    folder = c.OUT/arm/repeat
    folder.mkdir(parents=True, exist_ok=False)
    started, process = time.perf_counter(), psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    torch.set_num_threads(4)
    torch.manual_seed(c.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tok = tokenizer(arm)
    if arm == 'control':
        if inventory(snapshot()) != frozen['source_files']:
            raise RuntimeError('Source weights changed')
        model = AutoModelForCausalLM.from_pretrained(snapshot(), local_files_only=True, trust_remote_code=False,
                                                    dtype=torch.bfloat16).to('cuda')
    else:
        model, loaded_tok, _ = load_layercake_core(c.checkpoint(arm), layercake_root=c.HOST, device='cuda')
        if tok.get_vocab() != loaded_tok.get_vocab() or any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError('Changed recipient vocabulary or precision')
        model._r104_eos = tok.eos_token_id
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    parameters = sum(p.numel() for p in model.parameters())
    peak_rss = process.memory_info().rss
    with torch.inference_mode(), (folder/'raw.jsonl').open('x', encoding='utf-8') as stream:
        for index, ref in enumerate(c.read(c.OUT/'population.json')):
            guard_storage()
            if time.perf_counter()-started > c.process_budget(arm):
                raise RuntimeError('Per-process budget exceeded')
            prompt = c.render(ref, arm, tok)
            input_ids = tok.encode(prompt, add_special_tokens=False)
            ids = torch.tensor([input_ids], device='cuda')
            torch.cuda.synchronize()
            before = time.perf_counter()
            if arm == 'control':
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    max_new_tokens=c.MAX_NEW, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                tokens, traces = generated[0, len(input_ids):].tolist(), None
            else:
                tokens, traces = recipient_generate(model, ids, c.ROUTES.index(ref['task']))
            torch.cuda.synchronize()
            answer = tok.decode(tokens, skip_special_tokens=True).strip()
            c.append(stream, {**{k: ref[k] for k in ('id', 'pair', 'panel', 'task')}, 'arm': arm,
                'prompt': prompt, 'prompt_ids': input_ids, 'token_ids': tokens, 'output': answer,
                'screen': c.screen(ref, answer), 'stop_reason': 'eos' if tokens[-1] == tok.eos_token_id else 'budget',
                'output_utf8_bytes': len(answer.encode()), 'output_characters': len(answer),
                'wall_seconds': time.perf_counter()-before, 'execution_routes': traces})
            peak_rss = max(peak_rss, process.memory_info().rss)
            if (index+1) % 16 == 0:
                print(json.dumps({'arm': arm, 'repeat': repeat, 'rows': index+1}), flush=True)
    c.binding()
    if arm == 'control' and inventory(snapshot()) != frozen['source_files']:
        raise RuntimeError('Source mutated during execution')
    c.validate(c.rows(folder/'raw.jsonl'), c.read(c.OUT/'population.json'), arm, tok)
    c.write(folder/'receipt.json', {'raw_sha256': c.sha(folder/'raw.jsonl'), 'freeze_sha256': c.sha(c.OUT/'freeze.json'),
        'arm': arm, 'repeat': repeat, 'parameters': parameters, 'precision': 'bfloat16' if arm == 'control' else 'float32',
        'calls': 96, 'wall_seconds': time.perf_counter()-started, 'cpu_process_seconds': sum(process.cpu_times()[:2])-cpu_start,
        'sampled_peak_rss_bytes': peak_rss, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(), 'optimizer_steps': 0})


def verify_cell(arm, repeat):
    folder = c.OUT/arm/repeat
    receipt = c.read(folder/'receipt.json')
    raw = c.rows(folder/'raw.jsonl')
    if (receipt['raw_sha256'] != c.sha(folder/'raw.jsonl') or receipt['freeze_sha256'] != c.sha(c.OUT/'freeze.json')
            or receipt['arm'] != arm or receipt['repeat'] != repeat or receipt['calls'] != len(raw)
            or receipt['calls'] != 96 or receipt['optimizer_steps'] != 0
            or receipt['precision'] != ('bfloat16' if arm == 'control' else 'float32')
            or receipt['parameters'] != (source_parameter_count() if arm == 'control' else 87710428)
            or receipt['wall_seconds'] > c.process_budget(arm)):
        raise RuntimeError('Missing or stale execution receipt')
    for key in ('parameters', 'wall_seconds', 'cpu_process_seconds', 'sampled_peak_rss_bytes', 'peak_cuda_allocated_bytes'):
        if not math.isfinite(receipt[key]) or receipt[key] <= 0:
            raise RuntimeError('Invalid resource accounting')
    if sum(r['wall_seconds'] for r in raw) > receipt['wall_seconds']:
        raise RuntimeError('Incomplete total execution time')
    c.validate(raw, c.read(c.OUT/'population.json'), arm, tokenizer(arm))
    return raw


def verify_arm(arm):
    first, replay = (verify_cell(arm, n) for n in ('initial', 'replay'))
    scientific = lambda rows: [{k: v for k, v in r.items() if k != 'wall_seconds'} for r in rows]
    if scientific(first) != scientific(replay):
        raise RuntimeError('Fresh-process scientific replay differs')
    return first


def derive():
    frozen = c.binding()
    if c.read(c.OUT/'matched_training_provenance.json') != c.training_provenance():
        raise RuntimeError('Changed matched-control history')
    if (c.OUT/'stop.json').exists():
        raise RuntimeError('Interrupted run cannot be complete')
    if inventory(snapshot()) != frozen['source_files']:
        raise RuntimeError('Missing or changed source package')
    source = verify_arm('control')
    summaries = {'control': c.summarize(source)}
    result = {'format': 'r109-decision/1', 'freeze_sha256': c.sha(c.OUT/'freeze.json'), 'cells': summaries,
              'evaluator_cases': len(c.audit()['cases']), 'full_moonshot': 'NOT_PROVEN',
              'native_layercake_defect': 'NOT_ESTABLISHED', 'product_promotion': 'NONE'}
    if not summaries['control']['qualified_mechanical_only']:
        if any((c.OUT/arm).exists() for arm in c.ARMS):
            raise RuntimeError('Recipients ran after failed control')
        result.update(verdict='CONTROL_QUALIFICATION_FAILED', failure_owner='EXPERIMENTAL_PREREQUISITE')
        arms = ('control',)
    else:
        observations = {arm: verify_arm(arm) for arm in c.ARMS}
        summaries.update({arm: c.summarize(raw) for arm, raw in observations.items()})
        contrasts = {arm: c.contrast(raw, source) for arm, raw in observations.items()}
        result.update(verdict='BOUNDED_TRAINED_TASK_COMPARISON_COMPLETED', failure_owner='ACQUISITION_OR_RECIPIENT_SUITABILITY',
                      control_relative=contrasts, matched_lora_minus_distillation=c.contrast(observations['lora'], observations['distillation']),
                      mechanical_noninferiority={a: summaries[a]['qualified_mechanical_only'] and contrasts[a]['ci95'][0] >= -.10 for a in c.ARMS})
        arms = ('control', *c.ARMS)
    receipts = [c.read(c.OUT/a/r/'receipt.json') for a in arms for r in ('initial', 'replay')]
    if sum(r['wall_seconds'] for r in receipts) > c.MODEL_PROCESS_SECONDS:
        raise RuntimeError('Total execution budget exceeded')
    result['costs'] = {'calls_including_replays': sum(r['calls'] for r in receipts),
        'process_wall_seconds': sum(r['wall_seconds'] for r in receipts),
        'cpu_process_seconds': sum(r['cpu_process_seconds'] for r in receipts),
        'maximum_sampled_rss_bytes': max(r['sampled_peak_rss_bytes'] for r in receipts),
        'maximum_cuda_allocated_bytes': max(r['peak_cuda_allocated_bytes'] for r in receipts),
        'new_training_steps': 0, 'source_parameters_imported_into_recipients': 0,
        'new_source_preparation_seconds': 0, 'source_preparation_reused_from_r108': c.read(c.OUT/'source.json')}
    completion = c.read(c.OUT/'execution_complete.json')
    launch = c.read(c.OUT/'launch.json')
    if (completion['freeze_sha256'] != c.sha(c.OUT/'freeze.json')
            or launch['budget_seconds'] != c.TOTAL_SECONDS
            or not 0 < completion['elapsed_seconds'] <= c.TOTAL_SECONDS
            or sum(r['wall_seconds'] for r in receipts) > completion['elapsed_seconds']):
        raise RuntimeError('Missing or invalid controller accounting')
    for arm in arms:
        for repeat in ('initial', 'replay'):
            operation = c.read(c.OUT/'operations'/f'{arm}-{repeat}.json')
            if operation['returncode'] != 0 or operation['raw_sha256'] != c.sha(c.OUT/arm/repeat/'raw.jsonl'):
                raise RuntimeError('Missing or stale operation')
            for when in ('before', 'after'):
                resource = c.read(c.OUT/'resources'/f'{arm}-{repeat}-{when}.json')
                if (not resource['commands'] or any(v['returncode'] for v in resource['commands'])
                        or resource['settings_changed'] != []
                        or resource['storage']['free_bytes'] < c.MIN_FREE_BYTES
                        or resource['storage']['evidence_bytes'] > c.MAX_EVIDENCE_BYTES):
                    raise RuntimeError('Invalid device/storage observation')
    result['costs']['controller_seconds'] = completion['elapsed_seconds']
    result['evidence_hashes'] = {p.relative_to(c.OUT).as_posix(): c.sha(p)
        for p in sorted(c.OUT.rglob('*')) if p.is_file() and
        (p.name in ('raw.jsonl', 'receipt.json', 'execution_complete.json', 'launch.json')
         or p.parent.name in ('resources', 'operations'))}
    old = c.rows(c.previous.OUT/'control/initial/raw.jsonl')
    scientific = lambda row: {k: v for k, v in row.items() if k != 'wall_seconds'}
    result['historical_prefix_descriptive_only'] = {
        'rows': len(old), 'exact_scientific_matches': sum(scientific(a) == scientific(b) for a, b in zip(old, source))}
    return result


def verify(seal=False):
    result = derive()
    if seal:
        c.write(c.OUT/'decision.json', result)
    if c.read(c.OUT/'decision.json') != result:
        raise RuntimeError('Missing or stale scientific decision')
    print(json.dumps({'verdict': result['verdict'], 'cells': {a: v['both_pass'] for a, v in result['cells'].items()},
                      'decision_sha256': c.sha(c.OUT/'decision.json')}), flush=True)
    return result


def pilot():
    c.binding()
    committed = subprocess.check_output(['git', 'show', 'HEAD:results/trained_grounding_r109/freeze.json'], cwd=c.ROOT)
    if hashlib.sha256(committed).hexdigest() != c.sha(c.OUT/'freeze.json'):
        raise RuntimeError('Freeze must be committed before new model calls')
    guard_storage()
    c.write(c.OUT/'launch.json', {'pid': os.getpid(), 'start_unix': time.time(), 'budget_seconds': c.TOTAL_SECONDS})
    started = time.monotonic()
    try:
        for arm in ('control', *c.ARMS):
            if arm != 'control' and not c.summarize(verify_arm('control'))['qualified_mechanical_only']:
                break
            for repeat in ('initial', 'replay'):
                guard_storage()
                telemetry(f'{arm}-{repeat}-before')
                remaining = c.TOTAL_SECONDS-(time.monotonic()-started)
                if remaining <= 0:
                    raise RuntimeError('Controller time budget exceeded')
                print(json.dumps({'starting': [arm, repeat]}), flush=True)
                argv = [sys.executable, '-u', '-m', 'experiments.trained_grounding_r109.run', 'execute', '--arm', arm, '--repeat', repeat]
                env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                try:
                    done = subprocess.run(argv, cwd=c.ROOT, env=env, capture_output=True, text=True, timeout=min(c.process_budget(arm), remaining))
                except subprocess.TimeoutExpired as error:
                    text = lambda value: value.decode(errors='replace') if isinstance(value, bytes) else value
                    c.write(c.OUT/'operations'/f'{arm}-{repeat}.json', {'argv': argv, 'returncode': None,
                        'timeout_seconds': error.timeout, 'stdout': text(error.stdout), 'stderr': text(error.stderr)})
                    raise
                c.write(c.OUT/'operations'/f'{arm}-{repeat}.json', {'argv': argv, 'returncode': done.returncode,
                    'stdout': done.stdout, 'stderr': done.stderr,
                    'raw_sha256': c.sha(c.OUT/arm/repeat/'raw.jsonl') if (c.OUT/arm/repeat/'raw.jsonl').is_file() else None})
                telemetry(f'{arm}-{repeat}-after')
                print(done.stdout[-1500:], flush=True)
                if done.returncode:
                    raise RuntimeError('Execution failed; inspect operation receipt')
            verify_arm(arm)
        c.write(c.OUT/'execution_complete.json', {'freeze_sha256': c.sha(c.OUT/'freeze.json'),
            'elapsed_seconds': time.monotonic()-started})
        verify(seal=True)
    except Exception as error:
        c.write(c.OUT/'stop.json', {'error': repr(error), 'elapsed_seconds': time.monotonic()-started,
                                  'native_layercake_defect': 'NOT_ESTABLISHED'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'pilot', 'execute', 'verify'))
    parser.add_argument('--arm', choices=('control', *c.ARMS))
    parser.add_argument('--repeat', choices=('initial', 'replay'))
    args = parser.parse_args()
    if args.command == 'execute':
        if args.arm is None or args.repeat is None:
            parser.error('execute requires --arm and --repeat')
        execute(args.arm, args.repeat)
    else:
        {'freeze': freeze, 'pilot': pilot, 'verify': verify}[args.command]()
