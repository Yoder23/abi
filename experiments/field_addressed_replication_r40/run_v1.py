"""Run R40 against frozen R36/R39 components and new generated prompts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from safetensors.torch import load_file

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30.protocol import INSTRUCTIONS, TASKS, _details
from experiments.english_sufficiency_r31.cascade_v3 import _infer_task, _runtime_score
from experiments.field_slot_representation_r34 import run_v1 as r34
from experiments.field_addressed_plan_r36.run_v1 import FieldAddressedTokenizer
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once


EXPECTED_R36_RESULT_SHA256 = "1830a0a1be9b520376ce228fd50514e79f9b3a8224d56f7a1b36e0338d03d5bd"
EXPECTED_R39_RESULT_SHA256 = "eee143b1f35038ed518cbd4a9a8aa034da03e0799fea39a9498e76b7bed242f0"
BASE_INDEX = 1_200_000


def _fixture() -> list[dict[str, Any]]:
    rows = []
    for task_index, task in enumerate(TASKS):
        bank = INSTRUCTIONS[task]
        if len(bank) != 3:
            raise RuntimeError("R40 requires three frozen instruction paraphrases")
        for ordinal in range(12):
            detail_cycle = ordinal % 3
            instruction_cycle = (ordinal // 3) % 3
            index = BASE_INDEX + task_index * 10_000 + detail_cycle + (ordinal // 3) * 3
            instruction = bank[instruction_cycle]
            details = _details(task, index)
            prompt = "\n".join((f"INSTRUCTION: {instruction}", "SUPPLIED MATERIAL:", *details))
            record_id = "r40-" + hashlib.sha256(f"r40|{task}|{ordinal}|{prompt}".encode()).hexdigest()[:20]
            rows.append({"record_id": record_id, "oracle_task": task, "instruction_cycle": instruction_cycle, "detail_cycle": detail_cycle, "source_index": index, "prompt": prompt})
    return rows


def _tokenizer(path: Path) -> FieldAddressedTokenizer:
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "abi-r36-field-addressed-neural-plan/1":
        raise ValueError("R40 tokenizer format changed")
    return FieldAddressedTokenizer(document["schema"], [bytes.fromhex(value) for value in document["fixed_lexemes_hex"]])


def _model(api: dict[str, Any], state_path: Path, tokenizer: FieldAddressedTokenizer) -> torch.nn.Module:
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    model = api["PortableTokenPlan"](fixed_vocab_size=tokenizer.vocab_size, model_width=64, attention_heads=4, encoder_layers=2, decoder_layers=2, feedforward_width=192, pointer_width=32, dropout=0.0, maximum_source_lexemes=128, maximum_target_actions=r34.MAXIMUM_ACTIONS).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(state_path))
    return model.cuda().eval()


def run(r36_result: Path, r39_result: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R40 output exists: {output}")
    if sha256_file(r36_result) != EXPECTED_R36_RESULT_SHA256 or sha256_file(r39_result) != EXPECTED_R39_RESULT_SHA256:
        raise RuntimeError("R40 frozen predecessor changed")
    if not torch.cuda.is_available():
        raise RuntimeError("R40 requires CUDA")
    r39 = json.loads(r39_result.read_text(encoding="utf-8"))
    if r39.get("verdict") != "PASS_R39_DEVELOPMENT" or r39["metrics"]["integrated_functional"] != 144:
        raise RuntimeError("R40 predecessor did not pass")

    components: dict[str, dict[str, Path]] = {}
    registered: dict[str, dict[str, str]] = {}
    for row in r39["inherited_r36_components"]:
        path = r36_result.parent / "models" / row["name"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError("R40 inherited component hash changed")
        suffix = "tokenizer" if row["name"].endswith("tokenizer.json") else "state"
        components.setdefault(row["task"], {})[suffix] = path
        registered.setdefault(row["task"], {})[suffix] = row["sha256"]
    abstention = r39["candidate"]
    components["abstention"] = {"state": r39_result.parent / abstention["state"]["path"], "tokenizer": r39_result.parent / abstention["tokenizer"]["path"]}
    registered["abstention"] = {"state": abstention["state"]["sha256"], "tokenizer": abstention["tokenizer"]["sha256"]}
    if set(components) != set(TASKS) or any(set(value) != {"state", "tokenizer"} for value in components.values()):
        raise RuntimeError("R40 component matrix incomplete")
    for task in TASKS:
        for kind, path in components[task].items():
            if sha256_file(path) != registered[task][kind]:
                raise RuntimeError("R40 component changed before execution")

    output.mkdir(parents=True)
    fixture = _fixture()
    fixture_path = output / "fixture.jsonl"
    write_jsonl_once(fixture_path, fixture)
    api = base._layercake(Path(__file__).resolve().parents[2])
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    observations = []
    active_parameters = []
    for task in TASKS:
        tokenizer = _tokenizer(components[task]["tokenizer"])
        model = _model(api, components[task]["state"], tokenizer)
        active_parameters.append(sum(parameter.numel() for parameter in model.parameters()))
        for row in [item for item in fixture if item["oracle_task"] == task]:
            routed = _infer_task(row["prompt"])
            error = None
            try:
                source_ids, lexemes = tokenizer.encode_source(row["prompt"])
                actions = model.generate_actions(torch.tensor([source_ids], dtype=torch.long, device="cuda"), maximum_actions=r34.MAXIMUM_ACTIONS)[0]
                generated = tokenizer.decode_actions(actions, lexemes).decode("utf-8", errors="strict")
            except ValueError as exc:
                actions, generated, error = [], "", str(exc)
            inferred, score = _runtime_score(row["prompt"], generated)
            observations.append({"record_id": row["record_id"], "oracle_task": task, "routed_task": routed, "route_exact": routed == task, "inferred_task": inferred, "contract_exact": inferred == task, "instruction_cycle": row["instruction_cycle"], "detail_cycle": row["detail_cycle"], "actions": actions, "output": generated, "representation_error": error, "score": score})
            peak_rss = max(peak_rss, process.memory_info().rss)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"tasks": len({row['oracle_task'] for row in observations}), "task": task, "functional": sum(row["score"]["functional"] for row in observations), "seconds": time.perf_counter() - started}), flush=True)

    missing_controls = []
    for task in TASKS:
        for kind in ("state", "tokenizer"):
            rejected = False
            try:
                absent = output / "physically_absent" / f"{task}.{kind}"
                _tokenizer(absent) if kind == "tokenizer" else _model(api, absent, _tokenizer(components[task]["tokenizer"]))
            except FileNotFoundError:
                rejected = True
            missing_controls.append({"task": task, "missing": kind, "rejected": rejected})
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
    errors = sum(row["representation_error"] is not None for row in observations)
    passed = functional >= 132 and all(item["functional"] >= 11 for item in by_task.values()) and by_task["abstention"]["functional"] == 12 and route_exact == 144 and contract_exact == 144 and errors == 0 and all(row["rejected"] for row in missing_controls) and "transformers" not in sys.modules
    result = {
        "format": "abi-r40-prospective-field-addressed-replication/1",
        "verdict": "PASS_R40_FROZEN_REPLICATION" if passed else "FAIL_R40_FROZEN_REPLICATION",
        "inputs": {"r36_result_sha256": sha256_file(r36_result), "r39_result_sha256": sha256_file(r39_result), "component_hashes": registered},
        "metrics": {"evaluation_rows": len(observations), "functional": functional, "by_task": by_task, "route_exact": route_exact, "contract_exact": contract_exact, "representation_errors": errors, "missing_component_rejections": sum(row["rejected"] for row in missing_controls)},
        "information_accounting": {"new_teacher_calls": 0, "new_teacher_tokens": 0, "training_steps": 0, "source_corpus_loaded": False, "teacher_outputs_loaded": False, "transformers_loaded": "transformers" in sys.modules, "active_model_parameters_maximum": max(active_parameters), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)},
        "controls": missing_controls,
        "artifacts": {"fixture": {"path": fixture_path.name, "sha256": sha256_file(fixture_path), "bytes": fixture_path.stat().st_size}, "evaluation": {"path": rows_path.name, "sha256": sha256_file(rows_path), "bytes": rows_path.stat().st_size}},
        "claim_ceiling": "PROSPECTIVE_TWELVE_CONTRACT_SUPPLIED_CONTENT_REPLICATION_NOT_UNRESTRICTED_ENGLISH",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r36-result", type=Path, required=True)
    parser.add_argument("--r39-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.r36_result.resolve(), args.r39_result.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
