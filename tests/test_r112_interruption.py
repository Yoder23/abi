"""Post-stop observation checks, never qualification or source execution."""
import copy

import pytest

from scripts import r112_interruption as audit


def test_machine_measured_floor_exception():
    value = audit.limit_observation("RuntimeError: Storage floor/cap violated: {'free_bytes': 10729582592, 'evidence_bytes': 7115434}")
    assert value == {'free_bytes': 10729582592, 'evidence_bytes': 7115434}


@pytest.mark.parametrize('message', [
    'RESOURCE_STOP', 'Missing evidence',
    "Storage floor/cap violated: {'free_bytes': 10737418240, 'evidence_bytes': 1}",
    "Storage floor/cap violated: {'free_bytes': 1, 'evidence_bytes': 1073741825}",
    "Storage floor/cap violated: {'free_bytes': -1, 'evidence_bytes': 1}",
])
def test_nonmatching_stop_rejects(message):
    with pytest.raises(RuntimeError):
        audit.limit_observation(message)


@pytest.mark.parametrize('mutation', ['none', 'empty', 'complete', 'tokens', 'timing', 'storage'])
def test_replay_prefix_integrity(mutation):
    initial = [{'token_ids': [i], 'wall_seconds': 1.,
                'storage_before': {'free_bytes': 11 * 1024**3, 'evidence_bytes': 100}} for i in range(288)]
    replay = copy.deepcopy(initial[:242])
    if mutation == 'empty':
        replay = []
    elif mutation == 'complete':
        replay = copy.deepcopy(initial)
    elif mutation == 'tokens':
        replay[0]['token_ids'] = [999]
    elif mutation == 'timing':
        replay[0]['wall_seconds'] = float('nan')
    elif mutation == 'storage':
        replay[0]['storage_before']['free_bytes'] = 1
    if mutation == 'none':
        audit.check_prefix(initial, replay)
    else:
        with pytest.raises(RuntimeError):
            audit.check_prefix(initial, replay)
