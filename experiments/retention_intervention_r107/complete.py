"""Additive parameter-custody recomputation; never modify frozen execution code."""

from __future__ import annotations

import hashlib
import json
import sys

from . import core as c
from . import run


class SizedTokenizer:
    """Cache only an immutable tokenizer's length, not any scoring/decoding."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.size = len(tokenizer)

    def __len__(self):
        return self.size

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def expected_custody(path):
    from safetensors.torch import load_file

    state = load_file(str(path), device='cpu')
    active, neutral, untouched = (hashlib.sha256() for _ in range(3))
    records = []
    for name, tensor in sorted(state.items()):
        header = name.encode()+str(tuple(tensor.shape)).encode()
        payload = tensor.contiguous().numpy().tobytes()
        selected = name.endswith('.up.weight') and name.startswith(('task_cakes.', 'capability_layer_adapters.'))
        active.update(header+payload)
        if selected:
            zeros = bytes(len(payload))
            neutral.update(header+zeros)
            records.append({'name': name, 'elements': tensor.numel(),
                            'before': hashlib.sha256(payload).hexdigest(), 'after': hashlib.sha256(zeros).hexdigest()})
        else:
            neutral.update(header+payload)
            untouched.update(header+payload)
    if len(records) != 98:
        raise RuntimeError('Unexpected checkpoint residual topology')
    return {'active': active.hexdigest(), 'neutral': neutral.hexdigest(),
            'intervention': {'tensors': records, 'untouched_sha256': untouched.hexdigest()},
            'parameters': sum(t.numel() for t in state.values())}


def verify():
    factory = run.tokenizer
    sized = SizedTokenizer(factory())
    try:
        run.tokenizer = lambda: sized
        decision = run.verify()
    finally:
        run.tokenizer = factory
    if c.read(c.OUT/'decision.json') != decision:
        raise RuntimeError('Saved decision is not recomputable')
    custody = {}
    for arm in c.ARMS:
        expected = expected_custody(c.checkpoint(arm)/'model.safetensors')
        for mode in ('active', 'neutral'):
            for repeat in ('initial', 'replay'):
                receipt = c.read(c.OUT/f'{arm}_{mode}'/repeat/'receipt.json')
                if receipt['state_sha256'] != expected[mode] or receipt['parameters'] != expected['parameters']:
                    raise RuntimeError('Executed parameter state is not the frozen checkpoint intervention')
                if mode == 'neutral' and receipt['intervention'] != expected['intervention']:
                    raise RuntimeError('Residual/untouched custody does not recompute from checkpoint')
        custody[arm] = {k: expected[k] for k in ('active', 'neutral', 'parameters')}
    return {'format': 'r107-complete-custody/1', 'outcome': decision['outcome'],
            'decision_sha256': c.sha(c.OUT/'decision.json'), 'freeze_sha256': c.sha(c.OUT/'freeze.json'),
            'recipient_states': custody, 'raw_files': {
                f'{cell}/{repeat}/raw.jsonl': c.sha(c.OUT/cell/repeat/'raw.jsonl')
                for cell in c.CELLS for repeat in ('initial', 'replay')}}


def main():
    result = verify()
    path = c.OUT/'complete.json'
    if sys.argv[1:] == ['--write']:
        c.write(path, result)
    elif sys.argv[1:]:
        raise RuntimeError('Only --write or no arguments are accepted')
    elif c.read(path) != result:
        raise RuntimeError('Changed completion record')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
