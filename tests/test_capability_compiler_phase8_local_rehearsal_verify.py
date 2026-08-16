import copy

import pytest

from abi.capability_compiler_phase8_local_rehearsal_verify import identities


def _row():
    return {
        "mode": "core_runtime",
        "probe_id": "p0",
        "domain": None,
        "output": "exact output",
        "output_token_ids": [1, 2, 3],
    }


def test_exact_functional_identity_passes():
    rows = [_row()]
    assert identities(rows) == identities(copy.deepcopy(rows))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(mode="wrong"),
        lambda row: row.update(probe_id="wrong"),
        lambda row: row.update(domain="python"),
        lambda row: row.update(output="changed"),
        lambda row: row.update(output_token_ids=[1, 2, 3, 4]),
        lambda row: row.pop("output_token_ids"),
        lambda row: row.pop("output"),
        lambda row: row.update(output="exact output "),
        lambda row: row.update(output_token_ids=[3, 2, 1]),
        lambda row: row.update(mode=None),
        lambda row: row.update(probe_id=None),
        lambda row: row.update(domain="chemistry"),
        lambda row: row.update(output="EXACT OUTPUT"),
        lambda row: row.update(output_token_ids=[]),
        lambda row: row.clear(),
    ],
)
def test_functional_identity_mutations_fail_closed(mutation):
    original = [_row()]
    changed = copy.deepcopy(original)
    mutation(changed[0])
    assert identities(original) != identities(changed)
