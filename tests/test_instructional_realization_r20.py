from __future__ import annotations

from experiments.instructional_realization_r20.compiler import infer_package
from experiments.instructional_realization_r20.isolated_worker import (
    infer_package as infer_isolated,
)
from experiments.instructional_realization_r20.package import execute
from experiments.instructional_realization_r20.protocol import (
    TASKS,
    public_rows,
    replace_instruction,
)


def _records() -> list[dict[str, str]]:
    return [
        {"record_id": row["record_id"], "prompt": row["prompt"], "output": row["expected"]}
        for row in public_rows("extraction")
    ]


def test_public_inventory_is_distinct_and_balanced() -> None:
    extraction = public_rows("extraction")
    evaluation = public_rows("evaluation")
    assert len(extraction) == 72
    assert len(evaluation) == 120
    assert len({row["record_id"] for row in extraction + evaluation}) == 192
    assert len({row["prompt"] for row in extraction + evaluation}) == 192
    assert {task: sum(row["task"] == task for row in evaluation) for task in TASKS} == {
        task: 20 for task in TASKS
    }
    assert not {tuple(row["slots"].values()) for row in extraction}.intersection(
        tuple(row["slots"].values()) for row in evaluation
    )


def test_oracle_teacher_compiles_and_generalizes_end_to_end() -> None:
    package, diagnostics = infer_package(_records())
    isolated_package, isolated_diagnostics = infer_isolated(_records())
    assert isolated_package == package
    assert isolated_diagnostics == diagnostics
    assert diagnostics["records_parsed"] == 72
    assert diagnostics["programs_selected"] == 6
    assert all(
        execute(package, row["prompt"]) == row["expected"] for row in public_rows("evaluation")
    )
    assert all(execute(None, row["prompt"]) is None for row in public_rows("evaluation"))


def test_rotated_instruction_control_is_causally_wrong() -> None:
    rows = public_rows("extraction")
    instructions = {task: [] for task in TASKS}
    for row in rows:
        instructions[row["task"]].append(row["instruction"])
    counters = {task: 0 for task in TASKS}
    records = []
    for row in rows:
        task_index = TASKS.index(row["task"])
        replacement_task = TASKS[(task_index + 1) % len(TASKS)]
        offset = counters[row["task"]]
        counters[row["task"]] += 1
        replacement = instructions[replacement_task][offset]
        records.append(
            {
                "record_id": row["record_id"],
                "prompt": replace_instruction(row["prompt"], replacement),
                "output": row["expected"],
            }
        )
    control, _ = infer_package(records)
    exact = sum(
        execute(control, row["prompt"]) == row["expected"] for row in public_rows("evaluation")
    )
    assert exact <= 24
