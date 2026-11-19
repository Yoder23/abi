"""Write-once physical custody audit after all approved R112 hard-link plans.

This is an operational observation, not an experiment preflight or qualification.
Every listed path must still have its exact bytes and shared read-only identity.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from scripts.r112_postlink_preflight import REVIEW, ROOT, r, verify_pair

PLANS = (
    ('', '_retry', '2bb5c4949eb98f1f590a7143a77b76ab27de175e6260023fe900d4c822b111fb'),
    ('_aux', '_aux', 'b44f3c66ae8264b9e91399fe327cf6366179c1cea3489142fe62f807cc5e0667'),
    ('_small', '_small', 'd3d04db85a4ad280ffe7d6a92a00218cca81ebf7f8ddacc77926210ace0e64f6'),
    ('_cleanroom', '_cleanroom', '0091c4f853e3e1b34023b71e571a7481a9513aef0b595fe097d5d81aa1b38950'),
    ('_reserve', '_reserve', '77b1790e15ad34c0fdc83f4533f5b77dff6a2b01e2ba3047ba46e2d78bbad306'),
)


def main():
    target = REVIEW / 'r112_hardlink_final_custody.json'
    if target.exists():
        raise RuntimeError('Preserve existing custody observation')
    plans, net_observations = [], []
    for plan_suffix, result_suffix, digest in PLANS:
        complete_name = f'r112_hardlink{result_suffix}_complete.json'
        plans.append(verify_pair(f'r112_hardlink{plan_suffix}_plan.json', complete_name,
                                 f'r112_hardlink{result_suffix}_actions.jsonl', digest))
        complete = r.read(REVIEW / complete_name)
        net_observations.append(complete['free_bytes_after'] - complete['free_bytes_before'])
    result = {
        'format': 'r112-final-hardlink-custody/1',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'plans': plans,
        'linked_path_operations': sum(p['files'] for p in plans),
        'logical_target_bytes': sum(p['logical_duplicate_bytes'] for p in plans),
        'sum_observed_free_space_deltas': sum(net_observations),
        'free_bytes_now': shutil.disk_usage(ROOT).free,
        'r111_custody': r.audit_parent(),
        'script_sha256': r.sha(ROOT / 'scripts/r112_final_custody.py'),
        'scope': ('Physical identity, read-only flags and byte custody only. Logical target bytes '
                  'are not measured physical savings; free-space deltas include concurrent disk '
                  'activity. Cleanup overlapped initial R112 generation, whose timings are not '
                  'an isolated performance benchmark. No scientific or host promotion.'),
    }
    r.write_new(target, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
