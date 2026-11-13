"""Recomputable retention statistics and strict evidence validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = ROOT.parent / 'layercake_release'
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/retention_intervention_r107'
SOURCE_REV = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
DATA_REV = 'b08601e04326c79dfdd32d625aee71d232d685c3'
HOST_REV = '865a2b9e59e6221caa1185f9b64479d2077953f3'
ARMS = ('parent', 'full_model', 'lora', 'distillation')
CELLS = ('source', *(f'{a}_{mode}' for a in ARMS for mode in ('active', 'neutral')))
SEED, ROUTE, PREFIX, WINDOW, NEW = 107001, 1, 32, 128, 64


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
    stream.flush()


def checkpoint(arm):
    return ROOT / {'parent': 'results/role_invariant_choice_r76/candidate_v1',
                   'full_model': 'results/acquisition_capacity_r100/full_trunk',
                   'lora': 'results/grounded_english_r99_fp32/lora',
                   'distillation': 'results/grounded_english_r99_fp32/distillation'}[arm]


def select(texts, tok):
    selected, seen = [], set()
    for index, text in enumerate(texts):
        if not text.strip() or text.strip().startswith('=') or text in seen:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if not WINDOW <= len(ids) <= 512:
            continue
        seen.add(text)
        key = hashlib.sha256(f'r107:{SEED}:{index}:{text}'.encode()).hexdigest()
        selected.append({'id': f'wikitext-validation-{index}', 'index': index,
                         'selection_key': key, 'text': text, 'full_ids': ids,
                         'ids': ids[:WINDOW], 'target_bytes': len(tok.decode(ids[PREFIX:WINDOW]).encode())})
    selected.sort(key=lambda r: r['selection_key'])
    if len(selected) < 64:
        raise RuntimeError('Incomplete external population')
    return selected[:64]


def nll(row):
    values, norms = row['target_logits'], row['log_normalizers']
    if len(values) != WINDOW-PREFIX or len(norms) != len(values):
        raise RuntimeError('Incomplete numeric evidence')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values+norms):
        raise RuntimeError('Nonfinite evidence')
    if any(v > n+1e-5 for v, n in zip(values, norms)):
        raise RuntimeError('Invalid log probability')
    return math.fsum(n-v for v, n in zip(values, norms))


def repeated4(ids):
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    return 0.0 if not grams else 1-len(set(grams))/len(grams)


def validate(observed, population, cell, tok):
    if cell not in CELLS or len(observed) != len(population):
        raise RuntimeError('Missing/extra rows or unknown cell')
    for index, (row, ref) in enumerate(zip(observed, population)):
        if (row['id'] != ref['id'] or row['cell'] != cell or row['ids'] != ref['ids']
                or row['target_bytes'] != ref['target_bytes'] or row['target_bytes'] <= 0
                or len(row['ids']) != WINDOW or len(row['argmax_ids']) != WINDOW-PREFIX
                or any(type(t) is not int or not 0 <= t < len(tok) for t in row['argmax_ids'])
                or row['route'] != (None if cell == 'source' else ROUTE)):
            raise RuntimeError('Wrong population/token/route lineage')
        nll(row)
        trace = None if cell == 'source' else {'cakes': [ROUTE], 'adapters': [[ROUTE]]*6}
        if row['route_trace'] != trace:
            raise RuntimeError('Wrong physical route trace')
        error = row['cross_entropy_error']
        if not math.isfinite(error) or not 0 <= error <= 3e-5:
            raise RuntimeError('Inconsistent likelihood kernel')
        gen = row['generation']
        if index >= 16:
            if gen is not None:
                raise RuntimeError('Unregistered generation')
            continue
        ids = gen['ids']
        if gen['route_traces'] != [trace]*len(ids):
            raise RuntimeError('Incomplete generation route trace')
        if (not ids or len(ids) > NEW or any(type(t) is not int or not 0 <= t < len(tok) for t in ids)
                or tok.eos_token_id in ids[:-1]):
            raise RuntimeError('Invalid generation IDs')
        stop = 'eos' if ids[-1] == tok.eos_token_id else 'budget'
        if (gen['stop'] != stop or (stop == 'budget' and len(ids) != NEW)
                or gen['text'] != tok.decode(ids, skip_special_tokens=True)
                or gen['repeated4'] != repeated4(ids)):
            raise RuntimeError('Unrecomputable generation')


def summary(observed):
    losses = [nll(r) for r in observed]
    total, tokens = math.fsum(losses), len(losses)*(WINDOW-PREFIX)
    count_bytes = sum(r['target_bytes'] for r in observed)
    if not tokens or not count_bytes:
        raise RuntimeError('Empty summary')
    gens = [r['generation'] for r in observed if r['generation'] is not None]
    return {'passages': len(observed), 'target_tokens': tokens, 'target_utf8_bytes': count_bytes,
            'nll_nats_per_token': total/tokens, 'perplexity': math.exp(total/tokens),
            'bpb': total/math.log(2)/count_bytes,
            'generated_tokens': sum(len(g['ids']) for g in gens),
            'mean_repeated4': math.fsum(g['repeated4'] for g in gens)/len(gens),
            'eos_stops': sum(g['stop'] == 'eos' for g in gens)}


def paired(active, neutral):
    import numpy as np

    if not active or [r['id'] for r in active] != [r['id'] for r in neutral]:
        raise RuntimeError('Unpaired population')
    delta = np.array([(nll(a)-nll(n))/math.log(2)/a['target_bytes'] for a, n in zip(active, neutral)])
    rng = np.random.default_rng(SEED)
    draws = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {'passages': len(delta), 'mean_passage_bpb_active_minus_neutral': float(delta.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist()}


def binding():
    frozen = read(OUT/'freeze.json')
    if not frozen['files'] or not frozen['source_files']:
        raise RuntimeError('Missing inputs')
    for relative, digest in frozen['files'].items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError(f'Changed or missing frozen input: {relative}')
    return frozen
