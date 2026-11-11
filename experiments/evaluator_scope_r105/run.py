"""R105 write-once preparation, GPU inference, verification and live replay."""

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

from . import core as c


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(c.SOURCE, revision=c.REVISION, local_dir=c.MODEL,
                      allow_patterns=list(c.MODEL_FILES), max_workers=2)
    return c.inventory(c.MODEL, c.MODEL_FILES)


def prepare():
    import requests

    if c.OUT.exists():
        raise RuntimeError('Preserve prior preparation')
    if shutil.disk_usage(c.ROOT).free < 20 * 1024**3:
        raise RuntimeError('Free disk below 20GiB')
    c.OUT.mkdir(parents=True)
    started = time.perf_counter()
    model_files = download_model()
    if sum(r['bytes'] for r in model_files.values()) > 1024**3:
        raise RuntimeError('Evaluator download exceeds 1GiB budget')
    urls = {f'{s}.csv': f'https://raw.githubusercontent.com/selenashe/ScoNe/{c.DATA_REVISION}/scone_nli/test/{s}.csv' for s in c.SPLITS}
    urls.update({name: f'https://raw.githubusercontent.com/selenashe/ScoNe/{c.DATA_REVISION}/{name}' for name in ('LICENSE', 'README.md')})
    for name, url in urls.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path = c.OUT/'sources'/name
        path.parent.mkdir(exist_ok=True)
        with path.open('xb') as handle:
            handle.write(response.content)
    c.write_once(c.OUT/'preparation.json', {'format': 'r105-preparation/1', 'model': c.SOURCE, 'revision': c.REVISION,
        'model_files': model_files, 'source_urls': urls, 'source_files': c.inventory(c.OUT/'sources', urls),
        'wall_seconds': time.perf_counter()-started, 'source_model_calls': 0})
    print(json.dumps({'prepared_model_bytes': sum(v['bytes'] for v in model_files.values())}), flush=True)


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(c.MODEL, local_files_only=True, trust_remote_code=False)


def population():
    return c.external_catalog() + c.approval_catalog()


def freeze():
    prepared = c.read(c.OUT/'preparation.json')
    if prepared['model_files'] != c.inventory(c.MODEL, c.MODEL_FILES):
        raise RuntimeError('Evaluator identity changed')
    config = c.read(c.MODEL/'config.json')
    if config['id2label'] != {str(i): name for i, name in enumerate(c.LABELS)}:
        raise RuntimeError('Wrong NLI label mapping')
    records, tok = population(), tokenizer()
    for row in records:
        if len(tok(row['premise'], row['hypothesis'], truncation=False)['input_ids']) > 512:
            raise RuntimeError('Input exceeds untruncated context')
    c.write_once(c.OUT/'catalog.json', records)
    files = [*c.HERE.glob('*.py'), c.HERE/'PROTOCOL.md', c.ROOT/'tests/test_evaluator_scope_r105.py',
             c.ROOT/'experiments/__init__.py', c.ROOT/'experiments/plan_english_r98/__init__.py',
             c.ROOT/'experiments/plan_english_r98/core.py', c.OUT/'catalog.json', c.OUT/'preparation.json',
             *(c.OUT/'sources').iterdir()]
    c.write_once(c.OUT/'freeze.json', {'format': 'r105-freeze/1', 'model_files': prepared['model_files'],
        'files': {p.relative_to(c.ROOT).as_posix(): c.sha(p) for p in files},
        'abi_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip(),
        'model': c.SOURCE, 'revision': c.REVISION, 'data_revision': c.DATA_REVISION,
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'tokenizers', 'safetensors', 'numpy', 'psutil')},
        'pairs': 1264, 'training_steps': 0, 'budget_seconds': 1200, 'batch_size': 16, 'precision': 'float32'})
    c.binding()
    print(json.dumps({'freeze_sha256': c.sha(c.OUT/'freeze.json'), 'rows': len(records)}), flush=True)


def execute(replay=False):
    import psutil
    import torch
    from transformers import AutoModelForSequenceClassification

    frozen = c.binding()
    if replay and not (c.OUT/'decision.json').is_file():
        raise RuntimeError('Initial decision required before replay')
    folder = c.OUT / ('replay' if replay else 'initial')
    folder.mkdir()
    started, process = time.perf_counter(), psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    torch.set_num_threads(4)
    torch.manual_seed(c.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tok = tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(c.MODEL, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32).to('cuda').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.config.id2label != {i: name for i, name in enumerate(c.LABELS)}:
        raise RuntimeError('Runtime NLI labels changed')
    records = c.read(c.OUT/'catalog.json')
    if records != population():
        raise RuntimeError('Stale catalog')
    peak_rss = process.memory_info().rss
    batch_receipts = []
    with (folder/'raw.jsonl').open('xb') as stream, torch.inference_mode():
        for start in range(0, len(records), 16):
            if time.perf_counter()-started > 1200:
                raise RuntimeError('Registered execution budget exceeded')
            subset = records[start:start+16]
            features = tok([r['premise'] for r in subset], [r['hypothesis'] for r in subset],
                           padding=True, truncation=False, return_tensors='pt')
            if features['input_ids'].shape[1] > 512:
                raise RuntimeError('No silent truncation allowed')
            features = {k: v.to('cuda') for k, v in features.items()}
            torch.cuda.synchronize()
            before = time.perf_counter()
            logits = model(**features).logits
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-before
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite evaluator outputs')
            for ref, values, ids, mask in zip(subset, logits.cpu().tolist(), features['input_ids'].cpu().tolist(), features['attention_mask'].cpu().tolist()):
                length = sum(mask)
                row = {'id': ref['id'], 'input_ids': ids[:length], 'attention_mask': mask[:length],
                       'logits': values, 'prediction': c.LABELS[max(range(3), key=lambda j: values[j])]}
                stream.write(c.canonical(row))
                stream.flush()
            batch_receipts.append({'start': start, 'rows': len(subset), 'wall_seconds': elapsed})
            peak_rss = max(peak_rss, process.memory_info().rss)
    c.binding()
    observed = c.rows(folder/'raw.jsonl')
    if sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file()) > 50*1024**2:
        raise RuntimeError('Evidence exceeds 50MiB budget')
    c.validate(observed, records, tok)
    c.write_once(folder/'receipt.json', {'format': 'r105-execution/1', 'raw_sha256': c.sha(folder/'raw.jsonl'),
        'freeze_sha256': c.sha(c.OUT/'freeze.json'), 'pairs': len(observed), 'precision': frozen['precision'],
        'model_files': c.inventory(c.MODEL, c.MODEL_FILES), 'batch_receipts': batch_receipts,
        'wall_seconds': time.perf_counter()-started, 'cpu_seconds': sum(process.cpu_times()[:2])-cpu_start,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'sampled_peak_rss_bytes': peak_rss,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'device': torch.cuda.get_device_name(),
        'input_tokens': sum(len(r['input_ids']) for r in observed), 'optimizer_steps': 0,
        'source_teacher_calls': 0, 'recipient_calls': 0})
    print(json.dumps({'completed': folder.name, 'pairs': len(observed)}), flush=True)


def verify_execution(name):
    frozen = c.read(c.OUT/'freeze.json')
    folder = c.OUT/name
    receipt, records = c.read(folder/'receipt.json'), c.rows(folder/'raw.jsonl')
    if (receipt['format'] != 'r105-execution/1' or receipt['raw_sha256'] != c.sha(folder/'raw.jsonl')
            or receipt['freeze_sha256'] != c.sha(c.OUT/'freeze.json') or receipt['pairs'] != 1264
            or receipt['precision'] != 'float32' or receipt['model_files'] != frozen['model_files']
            or any(receipt[k] != 0 for k in ('optimizer_steps', 'source_teacher_calls', 'recipient_calls'))):
        raise RuntimeError('Missing or stale execution receipt')
    for key in ('wall_seconds', 'cpu_seconds', 'peak_cuda_allocated_bytes', 'sampled_peak_rss_bytes', 'parameter_count'):
        if not math.isfinite(receipt[key]) or receipt[key] <= 0:
            raise RuntimeError('Invalid resource accounting')
    if receipt['input_tokens'] != sum(len(r['input_ids']) for r in records):
        raise RuntimeError('Wrong token accounting')
    batches = receipt['batch_receipts']
    if [(r['start'], r['rows']) for r in batches] != [(i, min(16, 1264-i)) for i in range(0, 1264, 16)]:
        raise RuntimeError('Incomplete batch timing')
    if any(not math.isfinite(r['wall_seconds']) or r['wall_seconds'] <= 0 for r in batches) or sum(r['wall_seconds'] for r in batches) > receipt['wall_seconds']:
        raise RuntimeError('Invalid batch timing')
    c.validate(records, c.read(c.OUT/'catalog.json'), tokenizer())
    return records


def verify():
    c.binding()
    if (c.OUT/'stop.json').exists() or c.read(c.OUT/'catalog.json') != population():
        raise RuntimeError('Interrupted or stale study')
    initial = verify_execution('initial')
    result = c.aggregate(initial, population())
    path = c.OUT/'decision.json'
    if path.exists():
        if c.read(path) != result:
            raise RuntimeError('Stale scientific decision')
    else:
        c.write_once(path, result)
    if (c.OUT/'replay').exists():
        verify_execution('replay')
        if c.sha(c.OUT/'initial/raw.jsonl') != c.sha(c.OUT/'replay/raw.jsonl'):
            raise RuntimeError('Live replay not byte-identical')
    print(json.dumps(result), flush=True)
    return result


def pilot():
    c.binding()
    c.write_once(c.OUT/'launch.json', {'start_unix': time.time(), 'budget_seconds': 2400})
    for replay in (False, True):
        command = [sys.executable, '-u', '-m', 'experiments.evaluator_scope_r105.run', 'execute']
        if replay:
            command.append('--replay')
        try:
            done = subprocess.run(command, cwd=c.ROOT, capture_output=True, text=True, timeout=1200)
            c.write_once(c.OUT/f'operation-{int(replay)}.json', {'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr})
            if done.returncode:
                raise RuntimeError('Evaluator process failed')
            verify()
        except Exception as error:
            c.write_once(c.OUT/'stop.json', {'stage': 'replay' if replay else 'initial', 'error': repr(error)})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'freeze', 'execute', 'pilot', 'verify', 'fetch-model'))
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.command not in ('prepare', 'fetch-model'):
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.command == 'execute':
        execute(args.replay)
    else:
        {'prepare': prepare, 'freeze': freeze, 'pilot': pilot, 'verify': verify, 'fetch-model': download_model}[args.command]()
