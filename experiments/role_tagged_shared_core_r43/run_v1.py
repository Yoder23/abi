"""Execute the single registered semantic-role repair to the R42 core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.english_substrate_r30.protocol import TASKS
from experiments.field_addressed_plan_r36 import run_v1 as r36
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    sha256_file,
    write_json_once,
)
from experiments.shared_english_core_r42 import run_v1 as r42

EXPECTED_R42 = "2d190f98b5d385241bda684fe1260e4dfe30b0a98a3d60c217f96b30d0ab24ba"


def _role_prompt(task: str, prompt: str) -> str:
    if task not in TASKS:
        raise ValueError("R43 task is outside the frozen ontology")
    fields = r36._fields(prompt)
    tagged = [f"{key}={fields[key]}" for key in sorted(fields)]
    if len(tagged) not in (3, 4):
        raise ValueError("R43 expects three or four supplied fields")
    tagged.extend(["unused=unused"] * (4 - len(tagged)))
    for value in tagged:
        if len(r36.FieldAddressedTokenizer.split(value)) > r36.FIELD_WIDTH:
            raise ValueError("R43 role-tagged field exceeds frozen width")
    return "\n".join(
        (
            "SUPPLIED MATERIAL:",
            *(f"slot{index}={value}" for index, value in enumerate(tagged)),
            f"task={task}",
        )
    )


def run(
    layercake_root: Path,
    r31_rows: Path,
    r38_rows: Path,
    r36_result: Path,
    r39_result: Path,
    r40_result: Path,
    r41_result: Path,
    r42_result: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R43 output exists: {output}")
    if not r42_result.is_file() or sha256_file(r42_result) != EXPECTED_R42:
        raise RuntimeError("R43 frozen R42 failure changed")
    output.mkdir(parents=True)
    engine_dir = output / "engine"
    original = r42._canonical_prompt
    try:
        r42._canonical_prompt = _role_prompt
        engine = r42.run(
            layercake_root,
            r31_rows,
            r38_rows,
            r36_result,
            r39_result,
            r40_result,
            r41_result,
            engine_dir,
        )
    finally:
        r42._canonical_prompt = original
    engine_result = engine_dir / "result.json"
    if not engine_result.is_file():
        raise RuntimeError("R43 inner engine did not write its result")
    metrics = engine["metrics"]
    by_task = metrics["candidate_by_device_task"]
    passed = (
        engine["verdict"] == "PASS_R42_SHARED_CORE_DEVELOPMENT"
        and metrics["candidate_cpu_functional"] > 139
        and metrics["candidate_cuda_functional"] > 139
        and by_task["cpu"]["tone"] >= 10
        and by_task["cuda"]["tone"] >= 10
        and by_task["cpu"]["planning"] >= 10
        and by_task["cuda"]["planning"] >= 10
    )
    result = {
        "format": "abi-r43-role-tagged-shared-core-development/1",
        "verdict": (
            "PASS_R43_ROLE_TAGGED_SHARED_CORE_DEVELOPMENT"
            if passed
            else "FAIL_R43_ROLE_TAGGED_SHARED_CORE_DEVELOPMENT"
        ),
        "inputs": {
            "r42_result_sha256": EXPECTED_R42,
            "inner_frozen_inputs": engine["inputs"],
        },
        "intervention": {
            "only_change": "prefix each anonymous slot value with its original semantic field name",
            "schema": list(r42.SCHEMA),
            "field_width": r36.FIELD_WIDTH,
            "model_geometry_changed": False,
            "seed_changed": False,
            "compute_changed": False,
            "teacher_rows_changed": False,
            "new_teacher_calls": 0,
        },
        "training": engine["training"],
        "metrics": metrics,
        "lifecycle": engine["lifecycle"],
        "information_accounting": engine["information_accounting"],
        "packages": {
            name: {
                **reference,
                "path": f"engine/{reference['path']}",
            }
            for name, reference in engine["packages"].items()
        },
        "artifacts": {
            "inner_result": {
                "path": "engine/result.json",
                "sha256": sha256_file(engine_result),
            },
            "inner_evaluation": {
                "path": "engine/evaluation.jsonl",
                "sha256": sha256_file(engine_dir / "evaluation.jsonl"),
            },
            "training_rows": {
                "path": "engine/training_rows.jsonl",
                "sha256": sha256_file(engine_dir / "training_rows.jsonl"),
            },
        },
        "claim_ceiling": "ROLE_TAGGED_COMPACT_SHARED_CORE_DEVELOPMENT_NOT_HIDDEN_OR_UNRESTRICTED_ENGLISH",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--r31-rows", type=Path, required=True)
    parser.add_argument("--r38-rows", type=Path, required=True)
    parser.add_argument("--r36-result", type=Path, required=True)
    parser.add_argument("--r39-result", type=Path, required=True)
    parser.add_argument("--r40-result", type=Path, required=True)
    parser.add_argument("--r41-result", type=Path, required=True)
    parser.add_argument("--r42-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.layercake_root.resolve(),
        args.r31_rows.resolve(),
        args.r38_rows.resolve(),
        args.r36_result.resolve(),
        args.r39_result.resolve(),
        args.r40_result.resolve(),
        args.r41_result.resolve(),
        args.r42_result.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
