"""Write-once GPU retention audit with full fresh-process replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time

from . import core as c


def snapshot():
    from huggingface_hub import snapshot_download

    return c.Path(snapshot_download('distilgpt2', revision=c.SOURCE_REV, local_files_only=True))


def tokenizer(folder=None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(folder or snapshot(), local_files_only=True, trust_remote_code=False)


def inventory(folder):
    return {p.name: c.sha(p) for p in sorted(folder.iterdir()) if p.is_file()}


def prepare():
    import pyarrow.parquet as pq
    import requests
    from huggingface_hub import hf_hub_download

    if c.OUT.exists() or shutil.disk_usage(c.ROOT).free < 20*1024**3:
        raise RuntimeError('Preserve existing output / insufficient disk')
    source = snapshot()
    source_files = inventory(source)
    def pointer(url, path):
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        expected = ['version https://git-lfs.github.com/spec/v1', f'oid sha256:{c.sha(path)}', f'size {path.stat().st_size}']
        if lines != expected:
            raise RuntimeError('Local payload differs from pinned public pointer')
        return {'url': url, 'pointer': response.text, 'sha256': c.sha(path)}

    upstream = pointer(f'https://huggingface.co/distilbert/distilgpt2/raw/{c.SOURCE_REV}/model.safetensors', source/'model.safetensors')
    data_name = 'wikitext-103-raw-v1/validation-00000-of-00001.parquet'
    data = c.Path(hf_hub_download('Salesforce/wikitext', data_name, repo_type='dataset', revision=c.DATA_REV, local_files_only=True))
    ds = pointer(f'https://huggingface.co/datasets/Salesforce/wikitext/raw/{c.DATA_REV}/{data_name}', data)
    dest = c.OUT/'sources'
    dest.mkdir(parents=True)
    shutil.copyfile(data, dest/'validation.parquet')
    card = hf_hub_download('Salesforce/wikitext', 'README.md', repo_type='dataset', revision=c.DATA_REV, local_files_only=True)
    shutil.copyfile(card, dest/'WIKITEXT_README.md')
    c.write(dest/'provenance.json', {'source_model': 'distilgpt2', 'source_revision': c.SOURCE_REV,
        'source_files': source_files, 'source_public_pointer': upstream,
        'dataset': 'Salesforce/wikitext', 'dataset_revision': c.DATA_REV, 'dataset_file': data_name,
        'dataset_public_pointer': ds, 'dataset_sha256': ds['sha256'], 'license': 'CC-BY-SA-3.0/GFDL',
        'url': f'https://huggingface.co/datasets/Salesforce/wikitext/blob/{c.DATA_REV}/{data_name}'})
    population = c.select(pq.read_table(dest/'validation.parquet')['text'].to_pylist(), tokenizer())
    c.write(c.OUT/'population.json', population)
    print(json.dumps({'prepared_passages': len(population), 'model_forwards': 0}), flush=True)


def freeze():
    import pyarrow.parquet as pq

    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.HOST, text=True).strip() != c.HOST_REV:
        raise RuntimeError('Wrong host revision')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=c.HOST, text=True).strip():
        raise RuntimeError('Dirty host')
    tok = tokenizer()
    pop = c.read(c.OUT/'population.json')
    if pop != c.select(pq.read_table(c.OUT/'sources/validation.parquet')['text'].to_pylist(), tok):
        raise RuntimeError('Selection changed')
    for arm in c.ARMS:
        other = tokenizer(c.checkpoint(arm))
        for row in pop:
            if other.encode(row['text'], add_special_tokens=False) != row['full_ids']:
                raise RuntimeError('Tokenizer mismatch')
    paths = [*c.HERE.glob('*.*'), c.ROOT/'tests/test_retention_intervention_r107.py',
             c.OUT/'population.json', *(c.OUT/'sources').iterdir(),
             *(c.ROOT/'abi').rglob('*.py'), *(c.HOST/'layercake').rglob('*.py')]
    for arm in c.ARMS:
        paths += [p for p in c.checkpoint(arm).iterdir() if p.is_file()]
    files = {os.path.relpath(p, c.ROOT).replace('\\', '/'): c.sha(p) for p in paths}
    source_files = inventory(snapshot())
    if source_files != c.read(c.OUT/'sources/provenance.json')['source_files']:
        raise RuntimeError('Source changed')
    c.write(c.OUT/'freeze.json', {'format': 'r107-freeze/1', 'files': files, 'source_files': source_files,
        'host_revision': c.HOST_REV, 'source_revision': c.SOURCE_REV,
        'abi_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip(),
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy', 'safetensors', 'psutil', 'pyarrow')},
        'platform': platform.platform(), 'python': platform.python_version(),
        'cells': list(c.CELLS), 'replicates': ['initial', 'replay'], 'optimizer_steps': 0})
    print(json.dumps({'frozen_inputs': len(files), 'source_files': len(source_files)}), flush=True)


def state_hash(model, exclude=()):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name in exclude:
            continue
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def neutralize(model):
    import torch

    selected = {n: p for n, p in model.named_parameters()
                if n.endswith('.up.weight') and n.startswith(('task_cakes.', 'capability_layer_adapters.'))}
    if len(selected) != 98:
        raise RuntimeError('Wrong intervention topology')
    untouched = state_hash(model, selected)
    record = []
    with torch.no_grad():
        for name, tensor in sorted(selected.items()):
            before = hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
            tensor.zero_()
            after = hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
            record.append({'name': name, 'elements': tensor.numel(), 'before': before, 'after': after})
    if state_hash(model, selected) != untouched:
        raise RuntimeError('Intervention altered non-residual state')
    return {'tensors': record, 'untouched_sha256': untouched}


def token_evidence(logits, target):
    import torch
    import torch.nn.functional as F

    values = logits.float().gather(-1, target[:, None]).squeeze(-1)
    norms = torch.logsumexp(logits.float(), dim=-1)
    error = ((norms-values)-F.cross_entropy(logits.float(), target, reduction='none')).abs().max().item()
    if not torch.isfinite(logits).all() or error > 3e-5:
        raise RuntimeError('Invalid likelihood kernel')
    return values.tolist(), norms.tolist(), logits.argmax(-1).tolist(), error


def execute(cell, repeat):
    import psutil
    import torch
    from transformers import AutoModelForCausalLM

    from abi.layercake_core_loader import load_layercake_core

    frozen = c.binding()
    if inventory(snapshot()) != frozen['source_files']:
        raise RuntimeError('Changed source')
    folder = c.OUT/cell/repeat
    folder.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    start, cpu = time.perf_counter(), sum(process.cpu_times()[:2])
    torch.set_num_threads(4)
    torch.manual_seed(c.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    intervention = None
    if cell == 'source':
        model = AutoModelForCausalLM.from_pretrained(snapshot(), local_files_only=True, trust_remote_code=False,
                                                    torch_dtype=torch.float32).eval()
    else:
        arm, mode = cell.rsplit('_', 1)
        model, _, _ = load_layercake_core(c.checkpoint(arm), layercake_root=c.HOST, device='cpu')
        if mode == 'neutral':
            intervention = neutralize(model)
    model = model.to('cuda').eval()
    initial_state = state_hash(model)
    parameters = sum(p.numel() for p in model.parameters())
    tok = tokenizer()
    population = c.read(c.OUT/'population.json')
    peak_rss = process.memory_info().rss
    forwards = 0

    def forward(ids, past=None, cache=False):
        nonlocal forwards
        forwards += 1
        if cell == 'source':
            output = model(input_ids=ids, past_key_values=past, use_cache=cache, return_dict=True)
            return output.logits, output.past_key_values, None
        output = model(ids, task_routes=torch.tensor([c.ROUTE], device='cuda'), past_key_values=past, use_cache=cache)
        trace = {'cakes': list(model.last_cake_calls),
                 'adapters': [list(b._abi_last_deep_adapter_routes) for b in model.transformer.h]}
        return output['logits'], output['past_key_values'], trace

    with (folder/'raw.jsonl').open('x', encoding='utf-8', newline='\n') as stream, torch.inference_mode():
        for index, ref in enumerate(population):
            if time.perf_counter()-start > 1200:
                raise RuntimeError('Per-cell wall budget exhausted')
            ids = torch.tensor([ref['ids']], device='cuda')
            logits, _, trace = forward(ids)
            values, norms, argmax, error = token_evidence(logits[0, c.PREFIX-1:-1], ids[0, c.PREFIX:])
            del logits
            generation = None
            if index < 16:
                generated, traces = [], []
                logits, past, gt = forward(ids[:, :c.PREFIX], cache=True)
                for step in range(c.NEW):
                    token = int(logits[0, -1].argmax())
                    generated.append(token)
                    traces.append(gt)
                    if token == tok.eos_token_id or step == c.NEW-1:
                        break
                    logits, past, gt = forward(torch.tensor([[token]], device='cuda'), past, True)
                generation = {'ids': generated, 'text': tok.decode(generated, skip_special_tokens=True),
                    'stop': 'eos' if generated[-1] == tok.eos_token_id else 'budget',
                    'repeated4': c.repeated4(generated), 'route_traces': traces}
                del logits, past
            row = {'id': ref['id'], 'cell': cell, 'ids': ref['ids'], 'target_bytes': ref['target_bytes'],
                   'target_logits': values, 'log_normalizers': norms, 'argmax_ids': argmax,
                   'cross_entropy_error': error, 'generation': generation,
                   'route': None if cell == 'source' else c.ROUTE, 'route_trace': trace}
            c.append(stream, row)
            peak_rss = max(peak_rss, process.memory_info().rss)
    torch.cuda.synchronize()
    if state_hash(model) != initial_state:
        raise RuntimeError('State mutated during inference')
    observed = c.rows(folder/'raw.jsonl')
    c.validate(observed, population, cell, tok)
    c.binding()
    c.write(folder/'receipt.json', {'cell': cell, 'repeat': repeat, 'raw_sha256': c.sha(folder/'raw.jsonl'),
        'freeze_sha256': c.sha(c.OUT/'freeze.json'), 'wall_seconds': time.perf_counter()-start,
        'cpu_seconds': sum(process.cpu_times()[:2])-cpu, 'sampled_peak_rss_bytes': peak_rss,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'device': torch.cuda.get_device_name(),
        'parameters': parameters, 'state_sha256': initial_state, 'intervention': intervention,
        'forwards': forwards, 'source_parameters_retained': parameters if cell == 'source' else None,
        'optimizer_steps': 0, 'trained_parameters': 0, 'new_capability_packages': 0})
    print(json.dumps({'cell': cell, 'repeat': repeat, **c.summary(observed)}), flush=True)


def verify():
    frozen = c.binding()
    if inventory(snapshot()) != frozen['source_files']:
        raise RuntimeError('Missing or changed source package')
    tok, population = tokenizer(), c.read(c.OUT/'population.json')
    results, costs, raw = {}, {}, {}
    for cell in c.CELLS:
        runs = []
        costs[cell] = []
        for repeat in ('initial', 'replay'):
            folder = c.OUT/cell/repeat
            rec = c.read(folder/'receipt.json')
            observed = c.rows(folder/'raw.jsonl')
            c.validate(observed, population, cell, tok)
            if (rec['cell'] != cell or rec['repeat'] != repeat
                    or rec['raw_sha256'] != c.sha(folder/'raw.jsonl')
                    or rec['freeze_sha256'] != c.sha(c.OUT/'freeze.json')
                    or any(rec[k] != 0 for k in ('optimizer_steps', 'trained_parameters', 'new_capability_packages'))
                    or len(rec['state_sha256']) != 64
                    or rec['forwards'] != 64+sum(len(r['generation']['ids']) for r in observed[:16])
                    or any(not isinstance(rec[k], (int, float)) or not 0 < rec[k] < float('inf')
                           for k in ('wall_seconds', 'cpu_seconds', 'sampled_peak_rss_bytes', 'peak_cuda_allocated_bytes', 'parameters'))
                    or rec['wall_seconds'] > 1200):
                raise RuntimeError('Missing/stale/invalid execution receipt')
            if cell.endswith('_neutral'):
                iv = rec['intervention']
                names = [f'task_cakes.{i}.up.weight' for i in range(14)] + [
                    f'capability_layer_adapters.{layer}.{i}.up.weight' for layer in range(6) for i in range(14)]
                if sorted(t['name'] for t in iv['tensors']) != sorted(names) or len(iv['untouched_sha256']) != 64:
                    raise RuntimeError('Missing intervention custody')
                for t in iv['tensors']:
                    expected = 768*(64 if t['name'].startswith('task_cakes.') else 32)
                    if t['elements'] != expected or t['after'] != hashlib.sha256(bytes(expected*4)).hexdigest() or len(t['before']) != 64:
                        raise RuntimeError('Invalid residual zeroing')
            elif rec['intervention'] is not None:
                raise RuntimeError('Undeclared intervention')
            runs.append(rec)
            costs[cell].append({k: rec[k] for k in ('wall_seconds', 'cpu_seconds', 'sampled_peak_rss_bytes', 'peak_cuda_allocated_bytes', 'parameters', 'forwards')})
        if runs[0]['raw_sha256'] != runs[1]['raw_sha256'] or runs[0]['state_sha256'] != runs[1]['state_sha256'] or runs[0]['intervention'] != runs[1]['intervention']:
            raise RuntimeError('Live replay mismatch')
        raw[cell] = observed
        results[cell] = c.summary(observed)
    total_wall = sum(r['wall_seconds'] for items in costs.values() for r in items)
    if total_wall > 5400:
        raise RuntimeError('Cumulative budget exceeded')
    contrasts = {a: c.paired(raw[f'{a}_active'], raw[f'{a}_neutral']) for a in c.ARMS}
    return {'format': 'r107-decision/1', 'freeze_sha256': c.sha(c.OUT/'freeze.json'),
            'outcome': 'PARENT_RESIDUAL_INTERFERENCE_SUPPORTED' if contrasts['parent']['ci95'][0] > 0 else 'PARENT_RESIDUAL_INTERFERENCE_NOT_SUPPORTED',
            'summary': results, 'contrasts': contrasts, 'costs': costs, 'total_execution_wall_seconds': total_wall,
            'claim_boundary': 'Natural-text retention intervention only; no qualified English core, host defect, acquisition or moonshot promotion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'freeze', 'execute', 'campaign', 'verify'))
    parser.add_argument('--cell', choices=c.CELLS)
    parser.add_argument('--repeat', choices=('initial', 'replay'))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'freeze':
        freeze()
    elif args.command == 'execute':
        if args.cell is None or args.repeat is None:
            parser.error('execute requires cell and repeat')
        execute(args.cell, args.repeat)
    elif args.command == 'campaign':
        start = time.perf_counter()
        for repeat in ('initial', 'replay'):
            for cell in c.CELLS:
                remaining = 5400-(time.perf_counter()-start)
                if remaining <= 0:
                    raise RuntimeError('Campaign wall budget exhausted')
                subprocess.run([sys.executable, '-m', 'experiments.retention_intervention_r107.run', 'execute',
                                '--cell', cell, '--repeat', repeat], cwd=c.ROOT, check=True, timeout=min(1200, remaining))
                if sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file()) > 512*1024**2:
                    raise RuntimeError('Evidence disk budget exceeded')
        decision = verify()
        c.write(c.OUT/'decision.json', decision)
        print(json.dumps({'outcome': decision['outcome'], 'contrasts': decision['contrasts']}), flush=True)
    else:
        decision = verify()
        if c.read(c.OUT/'decision.json') != decision:
            raise RuntimeError('Unrecomputable saved decision')
        print(json.dumps({'outcome': decision['outcome'], 'decision_sha256': c.sha(c.OUT/'decision.json')}))


if __name__ == '__main__':
    main()
