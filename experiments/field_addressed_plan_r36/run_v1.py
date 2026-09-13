"""Execute the R36 field-addressed neural-plan development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import psutil
import torch
from safetensors.torch import save_file

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31.cascade_v3 import _runtime_score
from experiments.english_sufficiency_r31.hidden_v4 import TASK_TO_CLUSTER
from experiments.field_slot_representation_r34 import run_v1 as r34
from experiments.sublexeme_slot_representation_r35.run_v1 import SublexemeSlotTokenizer
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once


EXPECTED_R35_RESULT_SHA256 = "608283dffc03a577e25a005507e98c284a19435c9b58e8e455f0a559735fcea1"
FORMAT = "abi-r36-field-addressed-neural-plan/1"
SPECIAL_COUNT = 4
EOS_ID = 2
UNK_ID = 3
FIELD_WIDTH = 24
SELECTED_PER_TASK = 24
MINIMUM_PER_TASK = 12
DYNAMIC_NUMBER = re.compile(rb"^\d+$")
CONTENT = re.compile(rb"^[A-Za-z_]+$")
STOP = {b"a", b"an", b"and", b"as", b"at", b"be", b"before", b"by", b"for", b"from", b"in", b"is", b"it", b"of", b"on", b"or", b"the", b"to", b"was", b"with"}


def _fields(prompt: str) -> dict[str, str]:
    marker = "SUPPLIED MATERIAL:\n"
    if prompt.count(marker) != 1:
        raise ValueError("R36 prompt lacks one supplied-material boundary")
    result: dict[str, str] = {}
    for line in prompt.split(marker, 1)[1].splitlines():
        if "=" not in line:
            raise ValueError("R36 supplied-material line lacks equals")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            raise ValueError("R36 supplied field is empty or duplicated")
        result[key] = value
    if not result:
        raise ValueError("R36 supplied material is empty")
    return result


def _pointer_preferred(piece: bytes) -> bool:
    if DYNAMIC_NUMBER.fullmatch(piece):
        return int(piece) > 10
    return bool(CONTENT.fullmatch(piece)) and piece.casefold() not in STOP and len(piece) >= 3


class FieldAddressedTokenizer:
    def __init__(self, schema: Iterable[str], fixed_lexemes: Iterable[bytes]):
        self.schema = tuple(schema)
        values = tuple(fixed_lexemes)
        if not self.schema or len(self.schema) != len(set(self.schema)) or tuple(sorted(self.schema)) != self.schema:
            raise ValueError("R36 schema must be unique, non-empty, and sorted")
        if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values or any(not value for value in values):
            raise ValueError("R36 fixed vocabulary is invalid")
        if len(self.schema) * (FIELD_WIDTH + 1) > 128:
            raise ValueError("R36 field schema exceeds source boundary")
        self.fixed_lexemes = values
        self.lexeme_to_id = {value: index for index, value in enumerate(values, start=SPECIAL_COUNT)}
        self.id_to_lexeme = {index: value for index, value in enumerate(values, start=SPECIAL_COUNT)}

    @property
    def vocab_size(self) -> int:
        return SPECIAL_COUNT + len(self.fixed_lexemes)

    @staticmethod
    def split(value: bytes | str) -> list[bytes]:
        return SublexemeSlotTokenizer.split(value)

    @staticmethod
    def _marker(key: str) -> bytes:
        return f"<FIELD:{key}>".encode()

    def encode_source(self, prompt: str) -> tuple[list[int], list[bytes]]:
        supplied = _fields(prompt)
        if tuple(sorted(supplied)) != self.schema:
            raise ValueError("R36 runtime field schema changed")
        ids: list[int] = []
        lexemes: list[bytes] = []
        for key in self.schema:
            marker = self._marker(key)
            ids.append(self.lexeme_to_id[marker])
            lexemes.append(marker)
            pieces = self.split(supplied[key])
            if len(pieces) > FIELD_WIDTH:
                raise ValueError("R36 supplied field exceeds fixed block")
            ids.extend(self.lexeme_to_id.get(piece, UNK_ID) for piece in pieces)
            lexemes.extend(pieces)
            ids.extend([0] * (FIELD_WIDTH - len(pieces)))
            lexemes.extend([b""] * (FIELD_WIDTH - len(pieces)))
        return ids, lexemes

    def _pointer(self, piece: bytes, source_lexemes: list[bytes]) -> int | None:
        positions = [index for index, value in enumerate(source_lexemes) if value == piece]
        if len(positions) == 1 and _pointer_preferred(piece):
            return self.vocab_size + positions[0]
        return None

    def encode_target(self, response: str, source_lexemes: list[bytes]) -> list[int]:
        actions = []
        for piece in self.split(response):
            pointer = self._pointer(piece, source_lexemes)
            if pointer is not None:
                actions.append(pointer)
            elif piece in self.lexeme_to_id:
                actions.append(self.lexeme_to_id[piece])
            else:
                raise ValueError(f"R36 target lexeme is neither literal nor unique field pointer: {piece!r}")
        actions.append(EOS_ID)
        return actions

    def decode_actions(self, actions: Iterable[int], source_lexemes: list[bytes]) -> bytes:
        output: list[bytes] = []
        for raw in actions:
            action = int(raw)
            if action == EOS_ID:
                break
            if action >= self.vocab_size:
                position = action - self.vocab_size
                if position >= len(source_lexemes) or not source_lexemes[position]:
                    raise ValueError("R36 generated an unavailable field pointer")
                output.append(source_lexemes[position])
            elif action in self.id_to_lexeme:
                output.append(self.id_to_lexeme[action])
            else:
                raise ValueError("R36 generated a special non-output action")
        return b"".join(output)

    def document(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "schema": list(self.schema),
            "field_width": FIELD_WIDTH,
            "fixed_lexemes_hex": [value.hex() for value in self.fixed_lexemes],
            "special_ids": {"pad": 0, "bos": 1, "eos": EOS_ID, "unknown": UNK_ID},
            "instruction_in_neural_source": False,
            "pointer_rule": "unique supplied-field content lexeme or integer greater than ten",
        }

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.document(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _build_task(rows: list[dict[str, Any]]) -> tuple[FieldAddressedTokenizer, list[dict[str, Any]], list[dict[str, Any]]]:
    schemas = {tuple(sorted(_fields(row["prompt"]))) for row in rows}
    if len(schemas) != 1:
        raise RuntimeError("R36 task has inconsistent field schemas")
    schema = next(iter(schemas))
    provisional_literals = {FieldAddressedTokenizer._marker(key) for key in schema}
    for row in rows:
        provisional_literals.update(FieldAddressedTokenizer.split(row["prompt"]))
        provisional_literals.update(FieldAddressedTokenizer.split(row["teacher_output"]))
    provisional = FieldAddressedTokenizer(schema, sorted(provisional_literals))
    eligible: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for row in rows:
        source, lexemes = provisional.encode_source(row["prompt"])
        try:
            actions = provisional.encode_target(row["teacher_output"], lexemes)
            exact = provisional.decode_actions(actions, lexemes).decode("utf-8", errors="strict") == row["teacher_output"]
            if not exact:
                raise ValueError("decoded training target changed")
        except ValueError as exc:
            ledger.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "selected": False, "reason": str(exc)})
        else:
            eligible.append(row)
    selected = sorted(eligible, key=lambda row: (len(row["teacher_output"].encode()), row["record_id"]))[:SELECTED_PER_TASK]
    selected_ids = {row["record_id"] for row in selected}
    for row in eligible:
        ledger.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "selected": row["record_id"] in selected_ids, "reason": "selected_shortest_representable" if row["record_id"] in selected_ids else "eligible_not_selected_budget"})

    literals = {FieldAddressedTokenizer._marker(key) for key in schema}
    for row in selected:
        literals.update(FieldAddressedTokenizer.split(row["prompt"]))
        source_lexemes = provisional.encode_source(row["prompt"])[1]
        for piece in FieldAddressedTokenizer.split(row["teacher_output"]):
            if provisional._pointer(piece, source_lexemes) is None:
                literals.add(piece)
    tokenizer = FieldAddressedTokenizer(schema, sorted(literals))
    return tokenizer, selected, sorted(ledger, key=lambda row: row["record_id"])


def _encoded(rows: list[dict[str, Any]], tokenizer: FieldAddressedTokenizer) -> list[tuple[list[int], list[int]]]:
    result = []
    for row in rows:
        source, lexemes = tokenizer.encode_source(row["prompt"])
        target = tokenizer.encode_target(row["teacher_output"], lexemes)
        if len(target) > r34.MAXIMUM_ACTIONS or tokenizer.decode_actions(target, lexemes).decode() != row["teacher_output"]:
            raise RuntimeError("R36 selected training representation failed")
        result.append((source, target))
    return result


def run(r31_rows: Path, r31_failure: Path, r35_result: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R36 output exists: {output}")
    for path, digest in ((r31_rows, r34.EXPECTED_R31_ROWS_SHA256), (r31_failure, r34.EXPECTED_R31_FAILURE_SHA256), (r35_result, EXPECTED_R35_RESULT_SHA256)):
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"R36 prerequisite missing or changed: {path}")
    training_rows = [row for row in base._jsonl(r31_rows) if row["split"] == "train"]
    fixture_path = r31_failure.parent / "fixture.jsonl"
    fixture = base._jsonl(fixture_path)
    if len(training_rows) != 576 or len(fixture) != 144:
        raise RuntimeError("R36 source or fixture cardinality changed")

    output.mkdir(parents=True)
    (output / "models").mkdir()
    api = base._layercake(Path(__file__).resolve().parents[2])
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    observations: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    representation_failures = 0

    for position, task in enumerate(TASKS, 1):
        task_rows = [row for row in training_rows if row["oracle_task"] == task]
        tokenizer, selected, task_ledger = _build_task(task_rows)
        ledger.extend(task_ledger)
        if len(selected) < MINIMUM_PER_TASK:
            raise RuntimeError(f"R36 task {task} lacks minimum representable rows")
        model, training = r34._train(api, _encoded(selected, tokenizer), tokenizer, 36_000 + position)
        state_path = output / "models" / f"{task}.safetensors"
        save_file({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, state_path)
        tokenizer_path = output / "models" / f"{task}.tokenizer.json"
        write_json_once(tokenizer_path, tokenizer.document())
        models.append({"task": task, "cluster": TASK_TO_CLUSTER[task], "selected_rows": len(selected), "selected_record_ids": [row["record_id"] for row in selected], "state": {"path": state_path.relative_to(output).as_posix(), "sha256": sha256_file(state_path), "bytes": state_path.stat().st_size}, "tokenizer": {"path": tokenizer_path.relative_to(output).as_posix(), "sha256": sha256_file(tokenizer_path), "bytes": tokenizer_path.stat().st_size, "semantic_sha256": tokenizer.hash()}, "training": training})
        for row in [item for item in fixture if item["oracle_task"] == task]:
            try:
                source, lexemes = tokenizer.encode_source(row["prompt"])
                actions = model.generate_actions(torch.tensor([source], dtype=torch.long, device="cuda"), maximum_actions=r34.MAXIMUM_ACTIONS)[0]
                generated = tokenizer.decode_actions(actions, lexemes).decode("utf-8", errors="strict")
            except ValueError as exc:
                representation_failures += 1
                actions, generated, error = [], "", str(exc)
            else:
                error = None
            inferred, score = _runtime_score(row["prompt"], generated)
            observations.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "cluster": TASK_TO_CLUSTER[inferred], "route_exact": TASK_TO_CLUSTER[inferred] == TASK_TO_CLUSTER[row["oracle_task"]], "inferred_task": inferred, "contract_exact": inferred == row["oracle_task"], "actions": actions, "output": generated, "representation_error": error, "score": score})
            peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps({"trained_models": position, "task": task, "selected": len(selected), "functional": sum(item["score"]["functional"] for item in observations), "seconds": time.perf_counter() - started}), flush=True)
        del model
        torch.cuda.empty_cache()

    observations.sort(key=lambda row: (TASKS.index(row["oracle_task"]), row["record_id"]))
    ledger.sort(key=lambda row: (TASKS.index(row["oracle_task"]), row["record_id"]))
    evaluation_path = output / "evaluation.jsonl"
    ledger_path = output / "normalization_ledger.jsonl"
    write_jsonl_once(evaluation_path, observations)
    write_jsonl_once(ledger_path, ledger)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    route_exact = sum(row["route_exact"] for row in observations)
    contract_exact = sum(row["contract_exact"] for row in observations)
    passed = functional >= 132 and all(item["functional"] >= 11 for item in by_task.values()) and by_task["abstention"]["functional"] == 12 and route_exact == 144 and contract_exact == 144 and representation_failures == 0 and all(model["selected_rows"] >= MINIMUM_PER_TASK for model in models)
    result = {
        "format": "abi-r36-field-addressed-plan-feasibility/1",
        "verdict": "PASS_FIELD_ADDRESSED_PLAN_DEVELOPMENT" if passed else "FAIL_FIELD_ADDRESSED_PLAN_DEVELOPMENT",
        "inputs": {"r31_rows_sha256": sha256_file(r31_rows), "r31_failure_sha256": sha256_file(r31_failure), "r35_result_sha256": sha256_file(r35_result), "fixture_sha256": sha256_file(fixture_path)},
        "metrics": {"evaluation_rows": len(observations), "functional": functional, "by_task": by_task, "route_exact": route_exact, "contract_exact": contract_exact, "representation_failures": representation_failures, "selected_rows": sum(model["selected_rows"] for model in models)},
        "models": models,
        "information_accounting": {"source_teacher_rows_considered": len(training_rows), "selected_teacher_outputs": sum(model["selected_rows"] for model in models), "new_teacher_calls": 0, "new_teacher_tokens": 0, "stored_logits": 0, "stored_hidden_activations": 0, "source_parameters_copied": 0, "maximum_active_parameters": max(model["training"]["parameters"] for model in models), "deployed_state_bytes": sum(model["state"]["bytes"] + model["tokenizer"]["bytes"] for model in models), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)},
        "artifacts": {"evaluation": {"path": evaluation_path.name, "sha256": sha256_file(evaluation_path), "bytes": evaluation_path.stat().st_size}, "normalization_ledger": {"path": ledger_path.name, "sha256": sha256_file(ledger_path), "bytes": ledger_path.stat().st_size}},
        "claim_ceiling": "ABI_FIELD_ADDRESSED_DEVELOPMENT_ONLY_NOT_LAYERCAKE_PACKAGE_OR_FRESH_REPLICATION",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r31-rows", type=Path, required=True)
    parser.add_argument("--r31-failure", type=Path, required=True)
    parser.add_argument("--r35-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.r31_rows.resolve(), args.r31_failure.resolve(), args.r35_result.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
