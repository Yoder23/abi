"""Write-once, zero-training GPU grammar diagnostic and strict full replay."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time

from experiments.evaluator_control_r104 import run as prior

from . import core as c


def tokenizer(arm):
    from transformers import AutoTokenizer

    folder = prior.snapshot() if arm == 'control' else c.checkpoint(arm)
    return AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)


def prepare():
    import requests

    if c.OUT.exists():
        raise RuntimeError('Preserve prior preparation')
    if shutil.disk_usage(c.ROOT).free < 20*1024**3:
        raise RuntimeError('Insufficient disk')
    files = {f'{f}.jsonl': f'data/{f}.jsonl' for f in c.FAMILIES}
    files['README.md'] = 'README.md'
    for name, relative in files.items():
        url = f'https://raw.githubusercontent.com/alexwarstadt/blimp/{c.DATA_REVISION}/{relative}'
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path = c.OUT/'sources'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(response.content)
    populations = {s: c.catalog(s) for s in ('qualification', 'comparison')}
    a, b = populations.values()
    if {r['id'] for r in a} & {r['id'] for r in b} or {(r['good'], r['bad']) for r in a} & {(r['good'], r['bad']) for r in b}:
        raise RuntimeError('Overlapping panels')
    for split, population in populations.items():
        c.write_once(c.OUT/f'{split}.json', population)
    print(json.dumps({'prepared_pairs': sum(map(len, populations.values())), 'model_calls': 0}), flush=True)


def freeze():
    frozen = c.old.binding()
    if (c.OUT/'freeze.json').exists():
        raise RuntimeError('Preserve freeze')
    if prior.inventory(prior.snapshot()) != frozen['control_files']:
        raise RuntimeError('Source identity changed')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.HOST, text=True).strip() != c.old.HOST_CHECKOUT:
        raise RuntimeError('Wrong host checkout')
    if subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=c.HOST, text=True).strip():
        raise RuntimeError('Dirty host checkout')
    for split in ('qualification', 'comparison'):
        population = c.read(c.OUT/f'{split}.json')
        if population != c.catalog(split):
            raise RuntimeError('Altered external selection')
        for arm in ('control', *c.ARMS):
            tok = tokenizer(arm)
            for interface in c.INTERFACES:
                prompt = c.prefix(arm, interface, tok)
                for row in population:
                    for side in ('good', 'bad'):
                        c.encoding(row[side], prompt, tok)
    files = dict(frozen['files'])
    paths = [*c.HERE.glob('*.py'), c.HERE/'PROTOCOL.md', c.ROOT/'tests/test_grammar_interface_r106.py',
             c.ROOT/'experiments/evaluator_control_r104/run.py', c.ROOT/'results/evaluator_control_r104/freeze.json',
             c.OUT/'qualification.json', c.OUT/'comparison.json', *(c.OUT/'sources').iterdir()]
    files.update({os.path.relpath(p, c.ROOT).replace('\\', '/'): c.sha(p) for p in paths})
    c.write_once(c.OUT/'freeze.json', {'format': 'r106-freeze/1', 'files': files,
        'control_model': c.old.SOURCE, 'control_revision': c.old.REVISION, 'control_files': frozen['control_files'],
        'host_checkout': c.old.HOST_CHECKOUT, 'platform': platform.platform(), 'python': platform.python_version(),
        'abi_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip(),
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy', 'tokenizers', 'safetensors', 'psutil')},
        'qualification_pairs': 200, 'comparison_pairs': 400, 'maximum_sentence_forwards': 16800,
        'budget_seconds': 5400, 'optimizer_steps': 0, 'new_capability_artifacts': 0})
    c.binding()
    print(json.dumps({'inputs': len(files), 'freeze_sha256': c.sha(c.OUT/'freeze.json')}), flush=True)


def token_evidence(logits, targets):
    import torch
    import torch.nn.functional as F

    normalized = logits.float()
    picked = normalized.gather(-1, targets[:, None]).squeeze(-1)
    norms = torch.logsumexp(normalized, dim=-1)
    error = ((picked-norms) + F.cross_entropy(normalized, targets, reduction='none')).abs().max().item()
    if not torch.isfinite(normalized).all() or not math.isfinite(error) or error > 3e-5:
        raise RuntimeError('Nonfinite or inconsistent likelihood kernel')
    return picked.tolist(), norms.tolist(), error


def execute(arm, split, repeat):
    import psutil
    import torch
    from transformers import AutoModelForCausalLM

    from abi.layercake_core_loader import load_layercake_core

    frozen = c.binding()
    if split == 'qualification' and arm != 'control':
        raise RuntimeError('Qualification is control-only')
    if split == 'comparison' and not c.gate(verify_replays('control', 'qualification'), 50):
        raise RuntimeError('No qualified discrimination control')
    folder = c.OUT/split/arm/repeat
    folder.mkdir(parents=True)
    process = psutil.Process()
    started, cpu_start = time.perf_counter(), sum(process.cpu_times()[:2])
    torch.set_num_threads(4)
    torch.manual_seed(c.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tok = tokenizer(arm)
    if arm == 'control':
        if prior.inventory(prior.snapshot()) != frozen['control_files']:
            raise RuntimeError('Changed source before inference')
        model = AutoModelForCausalLM.from_pretrained(prior.snapshot(), local_files_only=True,
            trust_remote_code=False, dtype=torch.float16).to('cuda')
    else:
        model, loaded_tok, _ = load_layercake_core(c.checkpoint(arm), layercake_root=c.HOST, device='cuda')
        if tok.get_vocab() != loaded_tok.get_vocab() or any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError('Recipient identity/precision mismatch')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    count = sum(p.numel() for p in model.parameters())
    if count != (c.CONTROL_PARAMETERS if arm == 'control' else 87710428):
        raise RuntimeError('Wrong model parameter count')
    if any(p.dtype != (torch.float16 if arm == 'control' else torch.float32) for p in model.parameters()):
        raise RuntimeError('Unexpected parameter precision')
    population = c.read(c.OUT/f'{split}.json')
    interfaces = ('native',) if split == 'qualification' else c.INTERFACES
    peak_rss = process.memory_info().rss
    with torch.inference_mode(), (folder/'raw.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for index, ref in enumerate(population):
            for interface in interfaces:
                if time.perf_counter()-started > 1200:
                    raise RuntimeError('Process budget exceeded')
                prompt = c.prefix(arm, interface, tok)
                record = {k: ref[k] for k in ('id', 'family', 'split')}
                record.update(arm=arm, interface=interface, prefix=prompt)
                for side in ('good', 'bad'):
                    p, t = c.encoding(ref[side], prompt, tok)
                    ids = torch.tensor([p+t], device='cuda')
                    if arm == 'control':
                        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                        logits, trace = output.logits, None
                    else:
                        output = model(ids, task_routes=torch.tensor([c.ROUTE], device='cuda'),
                            prompt_lengths=torch.tensor([len(p)], device='cuda'), use_cache=False)
                        logits, trace = output['logits'], prior.route_trace(model, c.ROUTE)
                    picked, norms, error = token_evidence(logits[0, len(p)-1:-1], torch.tensor(t, device='cuda'))
                    record[side] = {'text': ref[side], 'prefix_ids': p, 'target_ids': t,
                        'target_logits': picked, 'log_normalizers': norms, 'cross_entropy_max_error': error,
                        'route_trace': trace, 'utf8_bytes': len(ref[side].encode())}
                    del output, logits, ids
                c.append(stream, record)
                peak_rss = max(peak_rss, process.memory_info().rss)
            if (index+1) % 50 == 0:
                print(json.dumps({'arm': arm, 'split': split, 'repeat': repeat, 'pairs': index+1}), flush=True)
    torch.cuda.synchronize()
    c.binding()
    if arm == 'control' and prior.inventory(prior.snapshot()) != frozen['control_files']:
        raise RuntimeError('Changed source after inference')
    observed = c.rows(folder/'raw.jsonl')
    c.validate(observed, population, arm, interfaces, tok)
    c.write_once(folder/'receipt.json', {'arm': arm, 'split': split, 'repeat': repeat,
        'freeze_sha256': c.sha(c.OUT/'freeze.json'), 'raw_sha256': c.sha(folder/'raw.jsonl'),
        'parameter_count': count, 'precision': 'float16' if arm == 'control' else 'float32',
        'pairs': len(observed), 'sentence_forwards': 2*len(observed), 'optimizer_steps': 0,
        'scored_tokens': sum(len(r[s]['target_ids']) for r in observed for s in ('good', 'bad')),
        'input_tokens': sum(len(r[s]['target_ids'])+len(r[s]['prefix_ids']) for r in observed for s in ('good', 'bad')),
        'target_utf8_bytes': sum(r[s]['utf8_bytes'] for r in observed for s in ('good', 'bad')),
        'scalar_logits_stored': sum(len(r[s]['target_logits']) for r in observed for s in ('good', 'bad')),
        'log_normalizers_stored': sum(len(r[s]['log_normalizers']) for r in observed for s in ('good', 'bad')),
        'wall_seconds': time.perf_counter()-started, 'cpu_seconds': sum(process.cpu_times()[:2])-cpu_start,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'sampled_peak_rss_bytes': peak_rss,
        'device': torch.cuda.get_device_name(), 'model_payload_bytes': sum(v['bytes'] for v in frozen['control_files'].values()) if arm == 'control' else (c.checkpoint(arm)/'model.safetensors').stat().st_size})


def verify_cell(arm, split, repeat):
    folder = c.OUT/split/arm/repeat
    receipt = c.read(folder/'receipt.json')
    observed = c.rows(folder/'raw.jsonl')
    if any(receipt.get(k) != v for k, v in {'arm': arm, 'split': split, 'repeat': repeat,
        'freeze_sha256': c.sha(c.OUT/'freeze.json'), 'raw_sha256': c.sha(folder/'raw.jsonl'), 'optimizer_steps': 0,
        'precision': 'float16' if arm == 'control' else 'float32', 'pairs': len(observed), 'sentence_forwards': 2*len(observed)}.items()):
        raise RuntimeError('Missing/stale resource lineage')
    if receipt['parameter_count'] != (c.CONTROL_PARAMETERS if arm == 'control' else 87710428):
        raise RuntimeError('Wrong model size')
    for key in ('wall_seconds', 'cpu_seconds', 'peak_cuda_allocated_bytes', 'sampled_peak_rss_bytes', 'parameter_count', 'model_payload_bytes'):
        if not math.isfinite(receipt[key]) or receipt[key] <= 0:
            raise RuntimeError('Invalid resources')
    if receipt['wall_seconds'] > 1200:
        raise RuntimeError('Process bound violated')
    expected_counts = {'scored_tokens': sum(len(r[s]['target_ids']) for r in observed for s in ('good', 'bad')),
        'input_tokens': sum(len(r[s]['target_ids'])+len(r[s]['prefix_ids']) for r in observed for s in ('good', 'bad')),
        'target_utf8_bytes': sum(r[s]['utf8_bytes'] for r in observed for s in ('good', 'bad')),
        'scalar_logits_stored': sum(len(r[s]['target_logits']) for r in observed for s in ('good', 'bad')),
        'log_normalizers_stored': sum(len(r[s]['log_normalizers']) for r in observed for s in ('good', 'bad'))}
    if any(receipt.get(k) != v for k, v in expected_counts.items()):
        raise RuntimeError('Unrecomputable information accounting')
    c.validate(observed, c.read(c.OUT/f'{split}.json'), arm, ('native',) if split == 'qualification' else c.INTERFACES, tokenizer(arm))
    return observed


def verify_replays(arm, split):
    first = verify_cell(arm, split, 'initial')
    verify_cell(arm, split, 'replay')
    if c.sha(c.OUT/split/arm/'initial/raw.jsonl') != c.sha(c.OUT/split/arm/'replay/raw.jsonl'):
        raise RuntimeError('Live raw replay differs')
    return first


def stages(qualified):
    cells = [('control', 'qualification')]
    if qualified:
        cells += [(a, 'comparison') for a in ('control', *c.ARMS)]
    return [(a, s, r) for a, s in cells for r in ('initial', 'replay')]


def verify(write=False):
    frozen = c.binding()
    if prior.inventory(prior.snapshot()) != frozen['control_files']:
        raise RuntimeError('Missing/changed control payload')
    if (c.OUT/'stop.json').exists():
        raise RuntimeError('Interrupted study')
    for split in ('qualification', 'comparison'):
        if c.read(c.OUT/f'{split}.json') != c.catalog(split):
            raise RuntimeError('Changed population')
    qualification = verify_replays('control', 'qualification')
    qualified = c.gate(qualification, 50)
    expected_stages = stages(qualified)
    expected_raw = {f'{s}/{a}/{r}/raw.jsonl' for a, s, r in expected_stages}
    if {p.relative_to(c.OUT).as_posix() for p in c.OUT.rglob('raw.jsonl')} != expected_raw:
        raise RuntimeError('Missing or extra execution matrix')
    expected_paths = {f'operation-{i:02d}.json' for i in range(len(expected_stages))}
    if {p.name for p in c.OUT.glob('operation-*.json')} != expected_paths:
        raise RuntimeError('Missing or extra operations')
    for index, (arm, split, repeat) in enumerate(expected_stages):
        operation = c.read(c.OUT/f'operation-{index:02d}.json')
        if operation['cell'] != [arm, split, repeat] or operation['returncode'] != 0:
            raise RuntimeError('Failed or wrong model operation')
    launch = c.read(c.OUT/'launch.json')
    if launch['budget_seconds'] != 5400 or not math.isfinite(launch['start_unix']):
        raise RuntimeError('Invalid launch')
    execution = c.read(c.OUT/'execution.json')
    if (not math.isfinite(execution['wall_seconds']) or not 0 < execution['wall_seconds'] <= 5400
            or execution['stages'] != len(expected_stages) or execution['launch_sha256'] != c.sha(c.OUT/'launch.json')):
        raise RuntimeError('Missing/stale execution budget record')
    result = {'format': 'r106-decision/1', 'qualification': c.summary(qualification),
        'qualified_grammatical_discrimination_control': qualified, 'qualified_free_generation_control': 'NOT_ESTABLISHED',
        'autonomous_generation_comparison': 'NOT_RUN', 'native_layercake_defect': 'NOT_ESTABLISHED',
        'new_training_steps': 0, 'product_promotion': 'NONE', 'full_moonshot': 'NOT_PROVEN'}
    if not qualified:
        if (c.OUT/'comparison').exists():
            raise RuntimeError('Forbidden recipient comparison')
        result['verdict'] = 'GRAMMATICAL_DISCRIMINATION_CONTROL_FAILED'
    else:
        observations = {a: verify_replays(a, 'comparison') for a in ('control', *c.ARMS)}
        cells = {f'{a}_{i}': [r for r in observed if r['interface'] == i] for a, observed in observations.items() for i in c.INTERFACES}
        reference = cells['control_native']
        contrasts = {a: c.paired(cells[f'{a}_native'], cells[f'{a}_benchmark']) for a in observations}
        relative = {name: c.paired(value, reference) for name, value in cells.items()}
        result.update(verdict='BOUNDED_GRAMMAR_INTERFACE_DIAGNOSTIC_COMPLETED',
            cells={name: c.summary(value) for name, value in cells.items()},
            comparison_control_qualified=c.gate(reference, 100), interface_contrasts=contrasts, control_relative=relative,
            primary_interface_sensitivity={a: c.gate(reference, 100) and contrasts[a]['ci97_5'][0] > 0 for a in ('parent', 'full_model')},
            narrow_recipient_screens={name: c.gate(reference, 100) and c.gate(value, 100) and relative[name]['ci95'][0] >= -.1 for name, value in cells.items() if not name.startswith('control_')})
    # The exact checkpoint bytes remain mandatory even for an early control stop.
    expected_payload = sum(v['bytes'] for v in frozen['control_files'].values())
    receipts = [c.read(c.OUT/s/a/r/'receipt.json') for a, s, r in expected_stages]
    for (arm, _, _), receipt in zip(expected_stages, receipts):
        if receipt['model_payload_bytes'] != (expected_payload if arm == 'control' else (c.checkpoint(arm)/'model.safetensors').stat().st_size):
            raise RuntimeError('Wrong deployed payload size')
    if sum(r['sentence_forwards'] for r in receipts) > 16800 or sum(r['wall_seconds'] for r in receipts) > 5400:
        raise RuntimeError('Study execution budget exceeded')
    path = c.OUT/'decision.json'
    if path.exists():
        if c.read(path) != result:
            raise RuntimeError('Stale scientific decision')
    elif write:
        c.write_once(path, result)
    else:
        raise FileNotFoundError('Missing completed decision; use pilot to finalize')
    print(json.dumps(result), flush=True)
    return result


def pilot():
    c.binding()
    if shutil.disk_usage(c.ROOT).free < 20*1024**3:
        raise RuntimeError('Initial disk bound')
    c.write_once(c.OUT/'launch.json', {'pid': os.getpid(), 'start_unix': time.time(), 'budget_seconds': 5400})
    started = time.monotonic()
    completed = 0
    try:
        for index, (arm, split, repeat) in enumerate(stages(True)):
            if index == 2 and not c.gate(verify_replays('control', 'qualification'), 50):
                break
            remaining = 5400-(time.monotonic()-started)
            if remaining <= 0:
                raise RuntimeError('Total time budget exhausted')
            print(json.dumps({'starting': [arm, split, repeat]}), flush=True)
            done = subprocess.run([sys.executable, '-u', '-m', 'experiments.grammar_interface_r106.run', 'execute',
                '--arm', arm, '--split', split, '--repeat', repeat], cwd=c.ROOT, capture_output=True, text=True, timeout=min(1200, remaining))
            c.write_once(c.OUT/f'operation-{index:02d}.json', {'cell': [arm, split, repeat],
                'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr})
            print(done.stdout[-1000:], flush=True)
            if done.returncode:
                raise RuntimeError('Scoring subprocess failed')
            completed += 1
            if sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file()) > 250*1024**2:
                raise RuntimeError('Evidence budget exceeded')
        c.write_once(c.OUT/'execution.json', {'wall_seconds': time.monotonic()-started,
            'stages': completed, 'launch_sha256': c.sha(c.OUT/'launch.json')})
        verify(write=True)
    except Exception as error:
        c.write_once(c.OUT/'stop.json', {'error': repr(error), 'stage': [arm, split, repeat]})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'freeze', 'pilot', 'execute', 'verify'))
    parser.add_argument('--arm', choices=('control', *c.ARMS))
    parser.add_argument('--split', choices=('qualification', 'comparison'))
    parser.add_argument('--repeat', choices=('initial', 'replay'))
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.command == 'execute':
        if not all((args.arm, args.split, args.repeat)):
            parser.error('execute requires arm, split and repeat')
        execute(args.arm, args.split, args.repeat)
    else:
        {'prepare': prepare, 'freeze': freeze, 'pilot': pilot, 'verify': verify}[args.command]()
