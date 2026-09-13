"""Execute the R34 typed dynamic-slot representation feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30 import package_v7 as v7
from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31.cascade_v3 import _runtime_score
from experiments.english_sufficiency_r31.hidden_v4 import TASK_TO_CLUSTER
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once


EXPECTED_R31_ROWS_SHA256 = "5ff7691d67b0e8271439acfbfe25540c7d42c06918b58b9257873c6f7ca9f05d"
EXPECTED_R31_FAILURE_SHA256 = "5e447168c885649f710480d510e7b99785ce1c802386572dd53d0dcc12cc180f"
EXPECTED_R33_FAILURE_SHA256 = "5ce858f6af067cae0a8af3a5a1a2f7a9100806d8f7abeefe5e4a714e494e5867"
FORMAT = "abi-r34-lossless-lexeme-field-slot/1"
SPECIAL_COUNT = 4
EOS_ID = 2
UNK_ID = 3
SLOT_COUNT = 4
BATCH_SIZE = 24
STEPS = math.ceil((48 * 533) / BATCH_SIZE)
MAXIMUM_ACTIONS = 384
DIGIT = re.compile(rb"\d")


def _is_dynamic(piece: bytes) -> bool:
    if not DIGIT.search(piece):
        return False
    if any(65 <= value <= 90 or 97 <= value <= 122 for value in piece):
        return True
    return piece.isdigit() and int(piece) > 99


class FieldSlotTokenizer:
    def __init__(self, fixed_lexemes: Iterable[bytes], slot_count: int = SLOT_COUNT):
        values = tuple(fixed_lexemes)
        if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
            raise ValueError("R34 fixed lexemes must be unique, non-empty, and sorted")
        if slot_count <= 0:
            raise ValueError("R34 slot count must be positive")
        self.fixed_lexemes = values
        self.slot_count = int(slot_count)
        self.lexeme_to_id = {value: index for index, value in enumerate(values, start=SPECIAL_COUNT)}
        self.id_to_lexeme = {index: value for index, value in enumerate(values, start=SPECIAL_COUNT)}
        self.slot_start = SPECIAL_COUNT + len(values)
        self.vocab_size = self.slot_start + self.slot_count

    @staticmethod
    def split(value: bytes | str) -> list[bytes]:
        return base._layercake(Path(__file__).resolve().parents[2])["LosslessLexemePointerTokenizer"].split(value)

    @classmethod
    def build(cls, rows: list[dict[str, str]]) -> "FieldSlotTokenizer":
        values = set()
        for row in rows:
            for field in ("prompt", "response"):
                values.update(piece for piece in cls.split(row[field]) if not _is_dynamic(piece))
        return cls(sorted(values))

    def _dynamic(self, source_lexemes: list[bytes]) -> list[bytes]:
        values = []
        for piece in source_lexemes:
            if _is_dynamic(piece) and piece not in values:
                values.append(piece)
        if len(values) > self.slot_count:
            raise ValueError("R34 prompt exceeds the registered dynamic-slot count")
        return values

    def encode_source(self, value: bytes | str) -> tuple[list[int], list[bytes]]:
        pieces = self.split(value)
        dynamic = self._dynamic(pieces)
        result = []
        for piece in pieces:
            if piece in dynamic:
                result.append(self.slot_start + dynamic.index(piece))
            else:
                result.append(self.lexeme_to_id.get(piece, UNK_ID))
        if UNK_ID in result:
            raise ValueError("R34 source contains an unknown static lexeme")
        return result, pieces

    def encode_target(self, response: bytes | str, source_lexemes: list[bytes]) -> list[int]:
        dynamic = self._dynamic(source_lexemes)
        actions = []
        for piece in self.split(response):
            if piece in dynamic:
                actions.append(self.slot_start + dynamic.index(piece))
            elif piece in self.lexeme_to_id:
                actions.append(self.lexeme_to_id[piece])
            else:
                raise ValueError(f"R34 target contains unknown lexeme: {piece!r}")
        actions.append(EOS_ID)
        return actions

    def decode_actions(self, actions: Iterable[int], source_lexemes: list[bytes]) -> bytes:
        dynamic = self._dynamic(source_lexemes)
        output = []
        for raw in actions:
            action = int(raw)
            if action == EOS_ID:
                break
            if self.slot_start <= action < self.vocab_size:
                slot = action - self.slot_start
                if slot >= len(dynamic):
                    raise ValueError("R34 generated an unavailable dynamic slot")
                output.append(dynamic[slot])
            elif action in self.id_to_lexeme:
                output.append(self.id_to_lexeme[action])
            elif action >= self.vocab_size:
                raise ValueError("R34 generated a forbidden source-position pointer")
            else:
                raise ValueError("R34 generated a special non-output action")
        return b"".join(output)

    def document(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "slot_count": self.slot_count,
            "dynamic_rule": "letters-and-digits-or-integer-greater-than-99",
            "fixed_lexemes_hex": [value.hex() for value in self.fixed_lexemes],
            "special_ids": {"pad": 0, "bos": 1, "eos": EOS_ID, "unknown": UNK_ID},
        }

    def hash(self) -> str:
        payload = json.dumps(self.document(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def _prepare(rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], FieldSlotTokenizer, list[tuple[list[int], list[int]]]]:
    prepared = [{"prompt": v7._normalized_prompt(row), "response": row["teacher_output"]} for row in rows]
    tokenizer = FieldSlotTokenizer.build(prepared)
    encoded = []
    for row in prepared:
        source, source_lexemes = tokenizer.encode_source(row["prompt"])
        target = tokenizer.encode_target(row["response"], source_lexemes)
        if len(source) > 128 or len(target) > MAXIMUM_ACTIONS:
            raise RuntimeError("R34 source or target exceeds frozen boundary")
        if tokenizer.decode_actions(target, source_lexemes).decode() != row["response"]:
            raise RuntimeError("R34 training representation is not lossless")
        encoded.append((source, target))
    return prepared, tokenizer, encoded


def _train(api, encoded, tokenizer, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = api["PortableTokenPlan"](
        fixed_vocab_size=tokenizer.vocab_size,
        model_width=64,
        attention_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        feedforward_width=192,
        pointer_width=32,
        dropout=0.0,
        maximum_source_lexemes=128,
        maximum_target_actions=MAXIMUM_ACTIONS,
    ).to(device).bind_tokenizer(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    rng = random.Random(seed)
    history = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        indexes = [rng.randrange(len(encoded)) for _ in range(BATCH_SIZE)]
        source, target = base._batch(encoded, indexes, device)
        result = model(source, target)
        mask = target.ge(0)
        loss = F.nll_loss(result["log_probs"][mask], target[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {1, STEPS // 2, STEPS}:
            prediction = result["log_probs"][mask].argmax(-1)
            history.append({"step": step, "loss": float(loss), "accuracy": float(prediction.eq(target[mask]).float().mean()), "gradient_norm": float(gradient), "seconds": time.perf_counter() - started})
    torch.cuda.synchronize()
    return model.eval(), {"steps": STEPS, "batch_size": BATCH_SIZE, "total_example_exposures": STEPS * BATCH_SIZE, "parameters": model.parameter_count(), "fixed_vocabulary_size": tokenizer.vocab_size, "history": history, "seconds": time.perf_counter() - started}


def run(r31_rows: Path, r31_failure: Path, r33_failure: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R34 output exists: {output}")
    for path, digest in ((r31_rows, EXPECTED_R31_ROWS_SHA256), (r31_failure, EXPECTED_R31_FAILURE_SHA256), (r33_failure, EXPECTED_R33_FAILURE_SHA256)):
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"R34 prerequisite missing or changed: {path}")
    training_rows = [row for row in base._jsonl(r31_rows) if row["split"] == "train"]
    fixture_path = r31_failure.parent / "fixture.jsonl"
    fixture = base._jsonl(fixture_path)
    if len(training_rows) != 576 or len(fixture) != 144:
        raise RuntimeError("R34 source or fixture cardinality changed")
    output.mkdir(parents=True)
    (output / "models").mkdir()
    api = base._layercake(Path(__file__).resolve().parents[2])
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    observations = []
    models = []
    representation_failures = 0

    for position, task in enumerate(TASKS, 1):
        task_rows = [row for row in training_rows if row["oracle_task"] == task]
        _, tokenizer, encoded = _prepare(task_rows)
        model, training = _train(api, encoded, tokenizer, 34_000 + position)
        state_path = output / "models" / f"{task}.safetensors"
        save_file({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, state_path)
        tokenizer_path = output / "models" / f"{task}.tokenizer.json"
        write_json_once(tokenizer_path, tokenizer.document())
        models.append({"task": task, "cluster": TASK_TO_CLUSTER[task], "state": {"path": state_path.relative_to(output).as_posix(), "sha256": sha256_file(state_path), "bytes": state_path.stat().st_size}, "tokenizer": {"path": tokenizer_path.relative_to(output).as_posix(), "sha256": sha256_file(tokenizer_path), "bytes": tokenizer_path.stat().st_size, "semantic_sha256": tokenizer.hash()}, "training": training})
        for row in [item for item in fixture if item["oracle_task"] == task]:
            prompt = v7._normalized_prompt({"prompt": row["prompt"], "nonce": row["nonce"]})
            try:
                source, source_lexemes = tokenizer.encode_source(prompt)
                source_tensor = torch.tensor([source], dtype=torch.long, device="cuda")
                actions = model.generate_actions(source_tensor, maximum_actions=MAXIMUM_ACTIONS)[0]
                text = tokenizer.decode_actions(actions, source_lexemes).decode("utf-8", errors="strict")
            except ValueError as exc:
                representation_failures += 1
                actions, text = [], ""
                error = str(exc)
            else:
                error = None
            inferred, score = _runtime_score(row["prompt"], text)
            observations.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "cluster": TASK_TO_CLUSTER[inferred], "route_exact": TASK_TO_CLUSTER[inferred] == TASK_TO_CLUSTER[row["oracle_task"]], "inferred_task": inferred, "contract_exact": inferred == row["oracle_task"], "actions": actions, "output": text, "representation_error": error, "score": score})
            peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps({"trained_models": position, "task": task, "functional": sum(item["score"]["functional"] for item in observations), "seconds": time.perf_counter() - started}), flush=True)
        del model
        torch.cuda.empty_cache()

    observations.sort(key=lambda row: (TASKS.index(row["oracle_task"]), row["record_id"]))
    rows_path = output / "evaluation.jsonl"
    write_jsonl_once(rows_path, observations)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    route_exact = sum(row["route_exact"] for row in observations)
    contract_exact = sum(row["contract_exact"] for row in observations)
    passed = functional >= 132 and all(item["functional"] >= 11 for item in by_task.values()) and by_task["abstention"]["functional"] == 12 and route_exact == 144 and contract_exact == 144 and representation_failures == 0
    result = {
        "format": "abi-r34-field-slot-feasibility/1",
        "verdict": "PASS_FIELD_SLOT_DEVELOPMENT" if passed else "FAIL_FIELD_SLOT_DEVELOPMENT",
        "inputs": {"r31_rows_sha256": sha256_file(r31_rows), "r31_failure_sha256": sha256_file(r31_failure), "r33_failure_sha256": sha256_file(r33_failure), "fixture_sha256": sha256_file(fixture_path)},
        "metrics": {"evaluation_rows": len(observations), "functional": functional, "by_task": by_task, "route_exact": route_exact, "contract_exact": contract_exact, "representation_failures": representation_failures},
        "models": models,
        "information_accounting": {"unique_teacher_outputs": len(training_rows), "teacher_output_tokens": sum(row["teacher_output_tokens"] for row in training_rows), "teacher_output_bytes": sum(len(row["teacher_output"].encode()) for row in training_rows), "new_teacher_calls": 0, "new_teacher_tokens": 0, "source_parameters_copied": 0, "maximum_active_parameters": max(item["training"]["parameters"] for item in models), "deployed_state_bytes": sum(item["state"]["bytes"] + item["tokenizer"]["bytes"] for item in models), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)},
        "evaluation": {"path": rows_path.name, "sha256": sha256_file(rows_path), "bytes": rows_path.stat().st_size},
        "claim_ceiling": "ABI_REPRESENTATION_DEVELOPMENT_ONLY_NOT_LAYERCAKE_PACKAGE_OR_HIDDEN_REPLICATION",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r31-rows", type=Path, required=True)
    parser.add_argument("--r31-failure", type=Path, required=True)
    parser.add_argument("--r33-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.r31_rows.resolve(), args.r31_failure.resolve(), args.r33_failure.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
