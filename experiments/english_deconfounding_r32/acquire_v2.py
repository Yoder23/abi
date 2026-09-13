"""R32 acquisition with the preregistered in-range numeric fixture repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.english_substrate_r30.protocol import INSTRUCTIONS, TASKS, _details
from . import acquire_v1 as v1


DETAIL_OFFSET = 600_000


def _detail_index(task_index: int, instruction_index: int, detail_cycle: int, candidate: int) -> int:
    value = DETAIL_OFFSET + task_index * 10_000 + instruction_index * 1_000 + detail_cycle * 100 + candidate * 7
    while value % 3 != detail_cycle:
        value += 1
    return value


def _candidate(task: str, instruction_index: int, detail_cycle: int, candidate: int) -> dict[str, Any]:
    task_index = TASKS.index(task)
    index = _detail_index(task_index, instruction_index, detail_cycle, candidate)
    instruction = INSTRUCTIONS[task][instruction_index]
    details = _details(task, index)
    prompt = "\n".join((f"INSTRUCTION: {instruction}", "SUPPLIED MATERIAL:", *details))
    digest = hashlib.sha256(
        f"r32-v2|{task}|{instruction_index}|{detail_cycle}|{candidate}|{prompt}".encode()
    ).hexdigest()[:20]
    return {
        "record_id": f"r32-v2-train-{digest}",
        "split": "train",
        "oracle_task": task,
        "instruction": instruction,
        "instruction_index": instruction_index,
        "detail_cycle": detail_cycle,
        "details": details,
        "nonce": f"Virelon{index:05d}",
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = v1.run(
        args.output.resolve(),
        candidate_factory=_candidate,
        format_version="abi-r32-counterbalanced-source/2",
        index_strategy="disjoint-in-range-v2",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
