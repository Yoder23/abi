"""Prospectively replicate the exact frozen R43 shared-core package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from safetensors.torch import load_file

from experiments.english_substrate_r30.protocol import INSTRUCTIONS, TASKS, _details
from experiments.english_sufficiency_r31.cascade_v3 import _infer_task, _runtime_score
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.layercake_package_integration_r41 import run_v1 as r41
from experiments.role_tagged_shared_core_r43.run_v1 import _role_prompt

EXPECTED_R43 = "72967c1ae40f6e65f49b0a7be74bb196a8b07b972ee180fe0a3de824a2325bed"
EXPECTED_CANDIDATE = "6ccb115de59e71b1c70efcfe2b5a1673b88d8f1478fb0059d8577ef5e2b6e6a4"
EXPECTED_ZERO = "7b674ea1d818338f962a8fb6df0bfb6b50377764008313cf5bdbdf7ec8bd6391"
EXPECTED_STATE = "2fec28a3867a09313f47d36921d0ff8e463d85c9e6777da89c43d734392698e8"
BASE_INDEX = 1_600_000
MAXIMUM_ACTIONS = 384


def fixture() -> list[dict[str, Any]]:
    rows = []
    for task_index, task in enumerate(TASKS):
        instructions = INSTRUCTIONS[task]
        if len(instructions) != 3:
            raise RuntimeError("R44 requires three frozen instruction paraphrases")
        for ordinal in range(12):
            detail_cycle = ordinal % 3
            instruction_cycle = (ordinal // 3) % 3
            index = BASE_INDEX + task_index * 10_000 + detail_cycle + (ordinal // 3) * 3
            prompt = "\n".join(
                (
                    f"INSTRUCTION: {instructions[instruction_cycle]}",
                    "SUPPLIED MATERIAL:",
                    *_details(task, index),
                )
            )
            record_id = (
                "r44-" + hashlib.sha256(f"r44|{task}|{ordinal}|{prompt}".encode()).hexdigest()[:20]
            )
            rows.append(
                {
                    "record_id": record_id,
                    "oracle_task": task,
                    "ordinal": ordinal,
                    "instruction_cycle": instruction_cycle,
                    "detail_cycle": detail_cycle,
                    "source_index": index,
                    "prompt": prompt,
                }
            )
    if len(rows) != 144 or len({row["record_id"] for row in rows}) != 144:
        raise RuntimeError("R44 fixture generation failed")
    return rows


def _evaluate(
    api: dict[str, Any],
    candidate: Path,
    zero: Path,
    rows: list[dict[str, Any]],
    public: bytes,
    signer: str,
    registry: Path,
    device: str,
) -> tuple[list[dict[str, Any]], Any, float]:
    host = r41._host(api, registry, public, signer, device)
    host.install(candidate)
    host.install(zero)
    host.installer.verify("abi-r42-english-core")
    host.installer.verify("abi-r42-zero-state")
    observations = []
    candidate_outputs = {}
    started = time.perf_counter()
    for row in rows:
        routed = _infer_task(row["prompt"])
        canonical = _role_prompt(routed, row["prompt"])
        error = None
        try:
            output = host.generate(
                "abi-r42-english-core", canonical, maximum_actions=MAXIMUM_ACTIONS
            ).output.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError) as exc:
            output, error = "", str(exc)
        inferred, score = _runtime_score(row["prompt"], output)
        candidate_outputs[row["record_id"]] = output
        observations.append(
            {
                "device": device,
                "condition": "candidate",
                "record_id": row["record_id"],
                "oracle_task": row["oracle_task"],
                "routed_task": routed,
                "route_exact": routed == row["oracle_task"],
                "inferred_task": inferred,
                "contract_exact": inferred == row["oracle_task"],
                "canonical_prompt_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "output": output,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "representation_error": error,
                "score": score,
            }
        )
    for task in TASKS:
        row = next(value for value in rows if value["oracle_task"] == task)
        canonical = _role_prompt(task, row["prompt"])
        error = None
        try:
            output = host.generate(
                "abi-r42-zero-state", canonical, maximum_actions=MAXIMUM_ACTIONS
            ).output.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError) as exc:
            output, error = "", str(exc)
        _, score = _runtime_score(row["prompt"], output)
        observations.append(
            {
                "device": device,
                "condition": "zero_state",
                "record_id": row["record_id"],
                "oracle_task": task,
                "output": output,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "candidate_exact": output == candidate_outputs[row["record_id"]],
                "representation_error": error,
                "score": score,
            }
        )
    return observations, host, time.perf_counter() - started


def run(
    layercake_root: Path,
    r43_result: Path,
    candidate: Path,
    zero: Path,
    state: Path,
    public_key: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R44 output exists: {output}")
    for path, digest in (
        (r43_result, EXPECTED_R43),
        (candidate, EXPECTED_CANDIDATE),
        (zero, EXPECTED_ZERO),
        (state, EXPECTED_STATE),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"R44 frozen input changed: {path}")
    if (
        not public_key.is_file()
        or sha256_file(public_key)
        != "7a01cbddee7572e5f7d496ec32cf27eb51196f6a576d09ff8dc4622e6d7309b2"
    ):
        raise RuntimeError("R44 research public key changed")
    if not torch.cuda.is_available():
        raise RuntimeError("R44 requires CUDA")
    api = r41._layercake(layercake_root)
    public = public_key.read_bytes()
    _, expected_public, signer = r41._keypair(api)
    if public != expected_public:
        raise RuntimeError("R44 public key content changed")
    package = api["load_package"](candidate, trust_store={signer: public})
    zero_package = api["load_package"](zero, trust_store={signer: public})
    source_state = load_file(state)
    if (
        package.manifest.cake_id != "abi-r42-english-core"
        or zero_package.manifest.cake_id != "abi-r42-zero-state"
        or set(package.tensors) != set(source_state)
        or not all(torch.equal(package.tensors[name], source_state[name]) for name in source_state)
        or sum(value.numel() for value in source_state.values()) != 296_554
    ):
        raise RuntimeError("R44 frozen package/tensor identity failed")
    output.mkdir(parents=True)
    fixture_rows = fixture()
    fixture_path = output / "fixture.jsonl"
    write_jsonl_once(fixture_path, fixture_rows)
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    cpu_rows, cpu_host, cpu_seconds = _evaluate(
        api, candidate, zero, fixture_rows, public, signer, output / "registry_cpu", "cpu"
    )
    cuda_rows, _, cuda_seconds = _evaluate(
        api, candidate, zero, fixture_rows, public, signer, output / "registry_cuda", "cuda"
    )
    observations = cpu_rows + cuda_rows
    candidate_rows = [row for row in observations if row["condition"] == "candidate"]
    zero_rows = [row for row in observations if row["condition"] == "zero_state"]
    by_task = {
        device: {
            task: sum(
                row["score"]["functional"]
                for row in candidate_rows
                if row["device"] == device and row["oracle_task"] == task
            )
            for task in TASKS
        }
        for device in ("cpu", "cuda")
    }
    cpu_by_id = {row["record_id"]: row for row in candidate_rows if row["device"] == "cpu"}
    cuda_by_id = {row["record_id"]: row for row in candidate_rows if row["device"] == "cuda"}
    first = fixture_rows[0]
    first_prompt = _role_prompt(first["oracle_task"], first["prompt"])
    expected_first = cpu_by_id[first["record_id"]]["output"].encode()
    removed = cpu_host.remove("abi-r42-english-core")
    rejected = False
    try:
        cpu_host.generate("abi-r42-english-core", first_prompt)
    except KeyError:
        rejected = True
    reinstalled = cpu_host.install(candidate)
    restored = cpu_host.generate("abi-r42-english-core", first_prompt).output
    lifecycle = {
        "removed_status": removed["status"],
        "generation_rejected": rejected,
        "reinstalled_status": reinstalled["status"],
        "restored_exact": restored == expected_first,
    }
    corrupt = output / "corrupted_candidate.cake"
    corrupt_bytes = bytearray(candidate.read_bytes())
    corrupt_bytes[len(corrupt_bytes) // 2] ^= 1
    corrupt.write_bytes(corrupt_bytes)
    corrupt_rejected = False
    corrupt_error = None
    corrupt_host = r41._host(api, output / "registry_corrupt", public, signer, "cpu")
    try:
        corrupt_host.install(corrupt)
    except Exception as exc:
        corrupt_rejected = True
        corrupt_error = type(exc).__name__ + ": " + str(exc)
    observations_path = output / "evaluation.jsonl"
    write_jsonl_once(observations_path, observations)
    metrics = {
        "fixture_rows": len(fixture_rows),
        "candidate_cpu_functional": sum(
            row["score"]["functional"] for row in candidate_rows if row["device"] == "cpu"
        ),
        "candidate_cuda_functional": sum(
            row["score"]["functional"] for row in candidate_rows if row["device"] == "cuda"
        ),
        "candidate_by_device_task": by_task,
        "route_exact": sum(row["route_exact"] for row in candidate_rows),
        "contract_exact": sum(row["contract_exact"] for row in candidate_rows),
        "representation_errors": sum(
            row["representation_error"] is not None for row in candidate_rows
        ),
        "noncollapsed": sum(
            bool(row["output"]) and row["score"]["noncollapsed"] for row in candidate_rows
        ),
        "cpu_cuda_exact": sum(
            cpu_by_id[key]["output"] == cuda_by_id[key]["output"] for key in cpu_by_id
        ),
        "zero_functional": sum(row["score"]["functional"] for row in zero_rows),
        "zero_candidate_exact": sum(row["candidate_exact"] for row in zero_rows),
        "active_parameters": sum(value.numel() for value in source_state.values()),
    }
    passed = (
        metrics["candidate_cpu_functional"] >= 132
        and metrics["candidate_cuda_functional"] >= 132
        and all(value >= 10 for device in by_task.values() for value in device.values())
        and by_task["cpu"]["abstention"] == 12
        and by_task["cuda"]["abstention"] == 12
        and metrics["route_exact"] == 288
        and metrics["contract_exact"] == 288
        and metrics["representation_errors"] == 0
        and metrics["noncollapsed"] == 288
        and metrics["cpu_cuda_exact"] == 144
        and metrics["zero_functional"] == 0
        and metrics["zero_candidate_exact"] == 0
        and metrics["active_parameters"] == 296_554
        and lifecycle["generation_rejected"]
        and lifecycle["restored_exact"]
        and corrupt_rejected
        and "transformers" not in sys.modules
    )
    result = {
        "format": "abi-r44-frozen-shared-core-replication/1",
        "verdict": "PASS_R44_FROZEN_SHARED_CORE_REPLICATION"
        if passed
        else "FAIL_R44_FROZEN_SHARED_CORE_REPLICATION",
        "inputs": {
            "r43_result_sha256": EXPECTED_R43,
            "candidate_package_sha256": EXPECTED_CANDIDATE,
            "zero_package_sha256": EXPECTED_ZERO,
            "candidate_state_sha256": EXPECTED_STATE,
            "layercake_commit": r41._layercake_head(layercake_root),
        },
        "metrics": metrics,
        "lifecycle": lifecycle,
        "corruption": {
            "package_sha256": sha256_file(corrupt),
            "strict_install_rejected": corrupt_rejected,
            "error": corrupt_error,
        },
        "information_accounting": {
            "new_teacher_calls": 0,
            "teacher_rows_loaded": 0,
            "teacher_parameters_loaded": 0,
            "teacher_logits_loaded": 0,
            "teacher_activations_loaded": 0,
            "training_steps": 0,
            "abi_local_neural_generation_calls": 0,
            "layercake_host_generation_calls": 314,
            "cpu_seconds": cpu_seconds,
            "cuda_seconds": cuda_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cpu_rss_bytes": process.memory_info().rss,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "artifacts": {
            "fixture": {"path": fixture_path.name, "sha256": sha256_file(fixture_path)},
            "evaluation": {
                "path": observations_path.name,
                "sha256": sha256_file(observations_path),
            },
            "corrupted_package": {
                "path": corrupt.name,
                "sha256": sha256_file(corrupt),
            },
        },
        "claim_ceiling": "PROSPECTIVE_REPLICATED_COMPACT_SUPPLIED_CONTENT_ENGLISH_CORE_NOT_UNRESTRICTED_ENGLISH",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--r43-result", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.layercake_root.resolve(),
        args.r43_result.resolve(),
        args.candidate.resolve(),
        args.zero.resolve(),
        args.state.resolve(),
        args.public_key.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
