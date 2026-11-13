"""Post-registered descriptive scope audit; no model forwards or new gates."""

import json
import sys

from . import core as c


def report():
    import torch
    from safetensors.torch import load_file

    c.binding()
    decision = c.read(c.OUT/'decision.json')
    parent = load_file(str(c.checkpoint('parent')/'model.safetensors'))
    comparison = {}
    selected = lambda n: n.startswith('transformer.') or n.startswith(f'task_cakes.{c.ROUTE}.') or any(
        n.startswith(f'capability_layer_adapters.{layer}.{c.ROUTE}.') for layer in range(6))
    for arm in c.ARMS:
        other = parent if arm == 'parent' else load_file(str(c.checkpoint(arm)/'model.safetensors'))
        if parent.keys() != other.keys():
            raise RuntimeError('Different state topology')
        changes = []
        for name in sorted(parent):
            if parent[name].shape != other[name].shape:
                raise RuntimeError('Different state shape')
            different = int(torch.count_nonzero(parent[name] != other[name]))
            if different:
                changes.append({'name': name, 'changed_elements': different, 'selected_by_fixed_route': selected(name)})
        active = c.rows(c.OUT/f'{arm}_active/initial/raw.jsonl')
        parent_rows = c.rows(c.OUT/'parent_active/initial/raw.jsonl')
        equal = sum({k: v for k, v in a.items() if k != 'cell'} == {k: v for k, v in b.items() if k != 'cell'}
                    for a, b in zip(active, parent_rows))
        comparison[arm] = {'changed_tensors': changes, 'selected_changed_tensors': sum(t['selected_by_fixed_route'] for t in changes),
                           'active_rows_identical_to_parent_except_cell': equal,
                           'active_bpb_minus_parent': decision['summary'][f'{arm}_active']['bpb']-decision['summary']['parent_active']['bpb']}
        if arm != 'parent':
            del other
    return {'format': 'r107-descriptive-scope/1', 'decision_sha256': c.sha(c.OUT/'decision.json'),
            'route': c.ROUTE, 'route_name': 'coherence', 'checkpoints': comparison,
            'interpretation_boundary': 'Post-hoc descriptive tensor/row identity, not a new primary contrast. Fixed coherence route was not an R99 acquisition task. No claim about trained-route fluency or ABI superiority.'}


def main():
    result = report()
    path = c.OUT/'descriptive_scope.json'
    if sys.argv[1:] == ['--write']:
        c.write(path, result)
    elif sys.argv[1:] or c.read(path) != result:
        raise RuntimeError('Unrecomputable descriptive scope')
    print(json.dumps({a: {k: v for k, v in data.items() if k != 'changed_tensors'}
                      for a, data in result['checkpoints'].items()}, sort_keys=True))


if __name__ == '__main__':
    main()
