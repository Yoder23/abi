"""Train and screen R38's balanced abstention package."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from safetensors.torch import save_file

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31.cascade_v3 import _runtime_score
from experiments.field_slot_representation_r34 import run_v1 as r34
from experiments.field_addressed_plan_r36 import run_v1 as r36
from experiments.targeted_abstention_r38.acquire_v1 import QUOTA
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once


EXPECTED_R31_ROWS_SHA256 = r34.EXPECTED_R31_ROWS_SHA256
EXPECTED_R36_RESULT_SHA256 = "1830a0a1be9b520376ce228fd50514e79f9b3a8224d56f7a1b36e0338d03d5bd"
STRATA = ("chemical_boiling_point", "national_independence_day", "python_program_result")


def _stratum(prompt: str) -> str:
    request = r36._fields(prompt)["request"].casefold()
    matches = []
    if "chemical boiling point" in request:
        matches.append(STRATA[0])
    if "national independence day" in request:
        matches.append(STRATA[1])
    if "python program" in request:
        matches.append(STRATA[2])
    if len(matches) != 1:
        raise ValueError("R38 unknown or ambiguous abstention stratum")
    return matches[0]


def _build(rows: list[dict[str, Any]]) -> tuple[r36.FieldAddressedTokenizer, list[dict[str, Any]], list[dict[str, Any]]]:
    schemas = {tuple(sorted(r36._fields(row["prompt"]))) for row in rows}
    if len(schemas) != 1:
        raise RuntimeError("R38 schema changed")
    schema = next(iter(schemas))
    provisional_literals = {r36.FieldAddressedTokenizer._marker(key) for key in schema}
    for row in rows:
        provisional_literals.update(r36.FieldAddressedTokenizer.split(row["prompt"]))
        provisional_literals.update(r36.FieldAddressedTokenizer.split(row["teacher_output"]))
    provisional = r36.FieldAddressedTokenizer(schema, sorted(provisional_literals))
    eligible = {name: [] for name in STRATA}
    ledger = []
    for row in rows:
        stratum = _stratum(row["prompt"])
        _, lexemes = provisional.encode_source(row["prompt"])
        try:
            actions = provisional.encode_target(row["teacher_output"], lexemes)
            if provisional.decode_actions(actions, lexemes).decode() != row["teacher_output"]:
                raise ValueError("decoded target changed")
        except ValueError as exc:
            ledger.append({"record_id": row["record_id"], "stratum": stratum, "selected": False, "reason": str(exc)})
        else:
            eligible[stratum].append(row)
    selected = []
    for stratum in STRATA:
        chosen = sorted(eligible[stratum], key=lambda row: (len(row["teacher_output"].encode()), row["record_id"]))[:QUOTA]
        if len(chosen) != QUOTA:
            raise RuntimeError(f"R38 stratum {stratum} lacks eight representable rows")
        selected.extend(chosen)
    selected_ids = {row["record_id"] for row in selected}
    for stratum in STRATA:
        for row in eligible[stratum]:
            ledger.append({"record_id": row["record_id"], "stratum": stratum, "selected": row["record_id"] in selected_ids, "reason": "selected_balanced_shortest" if row["record_id"] in selected_ids else "eligible_not_selected_budget"})
    literals = {r36.FieldAddressedTokenizer._marker(key) for key in schema}
    for row in selected:
        literals.update(piece for piece in r36.FieldAddressedTokenizer.split(row["prompt"]) if not r36.DYNAMIC_NUMBER.fullmatch(piece))
        source_lexemes = provisional.encode_source(row["prompt"])[1]
        for piece in r36.FieldAddressedTokenizer.split(row["teacher_output"]):
            if provisional._pointer(piece, source_lexemes) is None:
                literals.add(piece)
    return r36.FieldAddressedTokenizer(schema, sorted(literals)), selected, sorted(ledger, key=lambda row: row["record_id"])


def run(r31_rows: Path, r31_failure: Path, r36_result: Path, source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R38 training output exists: {output}")
    if sha256_file(r31_rows) != EXPECTED_R31_ROWS_SHA256 or sha256_file(r36_result) != EXPECTED_R36_RESULT_SHA256:
        raise RuntimeError("R38 frozen prerequisite changed")
    receipt_path = source / "receipt.json"
    accepted_path = source / "accepted_rows.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("verdict") != "PASS_TARGETED_SOURCE" or receipt["metrics"]["accepted_rows"] != QUOTA or receipt["artifacts"]["accepted_rows"]["sha256"] != sha256_file(accepted_path):
        raise RuntimeError("R38 targeted source did not pass or changed")
    original = [row for row in base._jsonl(r31_rows) if row["split"] == "train" and row["oracle_task"] == "abstention"]
    acquired = base._jsonl(accepted_path)
    tokenizer, selected, ledger = _build(original + acquired)
    encoded = r36._encoded(selected, tokenizer)
    fixture_path = r31_failure.parent / "fixture.jsonl"
    fixture = [row for row in base._jsonl(fixture_path) if row["oracle_task"] == "abstention"]
    if len(fixture) != 12:
        raise RuntimeError("R38 fixture changed")

    output.mkdir(parents=True)
    (output / "models").mkdir()
    api = base._layercake(Path(__file__).resolve().parents[2])
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    started = time.perf_counter()
    model, training = r34._train(api, encoded, tokenizer, 38_012)
    state_path = output / "models" / "abstention.safetensors"
    tokenizer_path = output / "models" / "abstention.tokenizer.json"
    save_file({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, state_path)
    write_json_once(tokenizer_path, tokenizer.document())
    observations = []
    for row in fixture:
        source_ids, lexemes = tokenizer.encode_source(row["prompt"])
        actions = model.generate_actions(torch.tensor([source_ids], dtype=torch.long, device="cuda"), maximum_actions=r34.MAXIMUM_ACTIONS)[0]
        generated = tokenizer.decode_actions(actions, lexemes).decode("utf-8", errors="strict")
        inferred, score = _runtime_score(row["prompt"], generated)
        observations.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "inferred_task": inferred, "contract_exact": inferred == "abstention", "actions": actions, "output": generated, "score": score})
    inherited = []
    for task in [name for name in TASKS if name != "abstention"]:
        for suffix in ("safetensors", "tokenizer.json"):
            path = r36_result.parent / "models" / f"{task}.{suffix}"
            if not path.is_file():
                raise RuntimeError(f"R38 inherited component missing: {path}")
            inherited.append({"task": task, "name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    evaluation_path = output / "evaluation.jsonl"
    ledger_path = output / "selection_ledger.jsonl"
    write_jsonl_once(evaluation_path, observations)
    write_jsonl_once(ledger_path, ledger)
    functional = sum(row["score"]["functional"] for row in observations)
    counts = {stratum: sum(_stratum(row["prompt"]) == stratum for row in selected) for stratum in STRATA}
    integrated = 132 + functional
    passed = functional == 12 and integrated == 144 and all(row["contract_exact"] for row in observations) and counts == {stratum: QUOTA for stratum in STRATA} and len(inherited) == 22
    result = {
        "format": "abi-r38-balanced-abstention-development/1",
        "verdict": "PASS_R38_DEVELOPMENT" if passed else "FAIL_R38_DEVELOPMENT",
        "inputs": {"r31_rows_sha256": sha256_file(r31_rows), "r31_failure_sha256": sha256_file(r31_failure), "r36_result_sha256": sha256_file(r36_result), "source_receipt_sha256": sha256_file(receipt_path), "source_rows_sha256": sha256_file(accepted_path), "fixture_sha256": sha256_file(fixture_path)},
        "metrics": {"abstention_rows": len(observations), "abstention_functional": functional, "integrated_functional": integrated, "integrated_total": 144, "contract_exact": sum(row["contract_exact"] for row in observations), "selected_by_stratum": counts, "inherited_component_files": len(inherited)},
        "candidate": {"state": {"path": state_path.relative_to(output).as_posix(), "sha256": sha256_file(state_path), "bytes": state_path.stat().st_size}, "tokenizer": {"path": tokenizer_path.relative_to(output).as_posix(), "sha256": sha256_file(tokenizer_path), "bytes": tokenizer_path.stat().st_size, "semantic_sha256": tokenizer.hash()}, "selected_record_ids": [row["record_id"] for row in selected], "training": training},
        "inherited_r36_components": inherited,
        "information_accounting": {"source_teacher_rows_considered": len(original) + len(acquired), "selected_teacher_outputs": len(selected), "new_teacher_calls": receipt["metrics"]["teacher_calls"], "new_teacher_tokens": receipt["information_accounting"]["teacher_generated_tokens"], "stored_logits": 0, "stored_hidden_activations": 0, "source_parameters_copied": 0, "active_parameters": training["parameters"], "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(process.memory_info().rss)},
        "artifacts": {"evaluation": {"path": evaluation_path.name, "sha256": sha256_file(evaluation_path), "bytes": evaluation_path.stat().st_size}, "selection_ledger": {"path": ledger_path.name, "sha256": sha256_file(ledger_path), "bytes": ledger_path.stat().st_size}},
        "claim_ceiling": "DISCLOSED_BALANCED_ABSTENTION_DEVELOPMENT_ONLY_NOT_FRESH_REPLICATION_OR_LAYERCAKE_PACKAGE",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r31-rows", type=Path, required=True)
    parser.add_argument("--r31-failure", type=Path, required=True)
    parser.add_argument("--r36-result", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.r31_rows.resolve(), args.r31_failure.resolve(), args.r36_result.resolve(), args.source.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
