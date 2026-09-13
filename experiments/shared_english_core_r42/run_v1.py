"""Train and evaluate one shared field-addressed English-core development model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31.cascade_v3 import _infer_task, _runtime_score
from experiments.field_addressed_plan_r36 import run_v1 as r36
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.layercake_package_integration_r41 import run_v1 as r41

EXPECTED = {
    "r31_rows": "5ff7691d67b0e8271439acfbfe25540c7d42c06918b58b9257873c6f7ca9f05d",
    "r38_rows": "9c9411aa10080aa65ce4088d99fb1306c709af0284084f50af03505ea3b57c04",
    "r36": r41.EXPECTED_R36,
    "r39": r41.EXPECTED_R39,
    "r40": r41.EXPECTED_R40,
    "r41": "d964ef7d591788211a6d54911b126c1e392a0b535fd2af8bc3027a6811f427c4",
}
SCHEMA = ("slot0", "slot1", "slot2", "slot3", "task")
UNUSED = "unused"
SEED = 42_001
BATCH_SIZE = 48
STEPS = 6_396
MAXIMUM_ACTIONS = 384


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _canonical_prompt(task: str, prompt: str) -> str:
    if task not in TASKS:
        raise ValueError("R42 task is outside the frozen ontology")
    fields = r36._fields(prompt)
    values = [fields[key] for key in sorted(fields)]
    if len(values) not in (3, 4):
        raise ValueError("R42 expects three or four supplied fields")
    values.extend([UNUSED] * (4 - len(values)))
    return "\n".join(
        (
            "SUPPLIED MATERIAL:",
            *(f"slot{index}={value}" for index, value in enumerate(values)),
            f"task={task}",
        )
    )


def _selected_rows(
    r31_rows: Path,
    r38_rows: Path,
    r36_result: Path,
    r39_result: Path,
) -> list[dict[str, Any]]:
    source = {row["record_id"]: row for row in _jsonl(r31_rows) + _jsonl(r38_rows)}
    r36_document = json.loads(r36_result.read_text(encoding="utf-8"))
    r39_document = json.loads(r39_result.read_text(encoding="utf-8"))
    selected: dict[str, list[str]] = {
        row["task"]: list(row["selected_record_ids"])
        for row in r36_document["models"]
        if row["task"] != "abstention"
    }
    selected["abstention"] = list(r39_document["candidate"]["selected_record_ids"])
    if set(selected) != set(TASKS):
        raise RuntimeError("R42 selected-task inventory changed")
    rows = []
    for task in TASKS:
        expected_count = 18 if task == "abstention" else 24
        if len(selected[task]) != expected_count:
            raise RuntimeError(f"R42 {task} selection count changed")
        for record_id in selected[task]:
            row = source.get(record_id)
            if row is None or row["oracle_task"] != task:
                raise RuntimeError(f"R42 selected source row is unavailable: {record_id}")
            canonical = _canonical_prompt(task, row["prompt"])
            rows.append(
                {
                    "record_id": record_id,
                    "oracle_task": task,
                    "source_prompt": row["prompt"],
                    "canonical_prompt": canonical,
                    "teacher_output": row["teacher_output"],
                    "teacher_output_sha256": hashlib.sha256(
                        row["teacher_output"].encode()
                    ).hexdigest(),
                    "teacher_output_tokens": int(row["teacher_output_tokens"]),
                }
            )
    if len(rows) != 282 or len({row["record_id"] for row in rows}) != 282:
        raise RuntimeError("R42 requires 282 unique selected rows")
    return rows


def _tokenizer(rows: list[dict[str, Any]]) -> r36.FieldAddressedTokenizer:
    markers = {r36.FieldAddressedTokenizer._marker(key) for key in SCHEMA}
    provisional_values = set(markers)
    for row in rows:
        provisional_values.update(r36.FieldAddressedTokenizer.split(row["canonical_prompt"]))
        provisional_values.update(r36.FieldAddressedTokenizer.split(row["teacher_output"]))
    provisional = r36.FieldAddressedTokenizer(SCHEMA, sorted(provisional_values))
    literals = set(markers)
    for row in rows:
        _, source_lexemes = provisional.encode_source(row["canonical_prompt"])
        literals.update(
            piece
            for piece in provisional.split(row["canonical_prompt"])
            if not r36.DYNAMIC_NUMBER.fullmatch(piece)
        )
        for piece in provisional.split(row["teacher_output"]):
            if provisional._pointer(piece, source_lexemes) is None:
                literals.add(piece)
    tokenizer = r36.FieldAddressedTokenizer(SCHEMA, sorted(literals))
    for row in rows:
        source, lexemes = tokenizer.encode_source(row["canonical_prompt"])
        target = tokenizer.encode_target(row["teacher_output"], lexemes)
        if (
            len(source) > 128
            or len(target) > MAXIMUM_ACTIONS
            or tokenizer.decode_actions(target, lexemes).decode() != row["teacher_output"]
        ):
            raise RuntimeError(f"R42 row is not losslessly representable: {row['record_id']}")
    return tokenizer


def _train(
    api: dict[str, Any],
    rows: list[dict[str, Any]],
    tokenizer: r36.FieldAddressedTokenizer,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    encoded = []
    for row in rows:
        source, lexemes = tokenizer.encode_source(row["canonical_prompt"])
        encoded.append((source, tokenizer.encode_target(row["teacher_output"], lexemes)))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = (
        api["PortableTokenPlan"](
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
        )
        .cuda()
        .bind_tokenizer(tokenizer)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    rng = random.Random(SEED)
    history = []
    started = time.perf_counter()
    checkpoints = {1, STEPS // 4, STEPS // 2, 3 * STEPS // 4, STEPS}
    for step in range(1, STEPS + 1):
        indexes = [rng.randrange(len(encoded)) for _ in range(BATCH_SIZE)]
        source, target = base._batch(encoded, indexes, torch.device("cuda"))
        result = model(source, target)
        mask = target.ge(0)
        loss = F.nll_loss(result["log_probs"][mask], target[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            prediction = result["log_probs"][mask].argmax(-1)
            item = {
                "step": step,
                "loss": float(loss),
                "accuracy": float(prediction.eq(target[mask]).float().mean()),
                "gradient_norm": float(gradient),
                "seconds": time.perf_counter() - started,
            }
            history.append(item)
            print(json.dumps(item), flush=True)
    torch.cuda.synchronize()
    return model.eval(), {
        "seed": SEED,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "total_example_exposures": STEPS * BATCH_SIZE,
        "parameters": model.parameter_count(),
        "fixed_vocabulary_size": tokenizer.vocab_size,
        "seconds": time.perf_counter() - started,
        "history": history,
    }


def _manifest(
    api: dict[str, Any],
    cake_id: str,
    state: dict[str, torch.Tensor],
    tokenizer_document: dict[str, Any],
    tokenizer_hash: str,
    signer: str,
    selected_hash: str,
    control: bool,
) -> Any:
    tokenizer = api["FieldAddressedPointerTokenizer"].from_document(tokenizer_document)
    config = {
        "fixed_vocab_size": tokenizer.vocab_size,
        "model_width": 64,
        "attention_heads": 4,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "feedforward_width": 192,
        "pointer_width": 32,
        "dropout": 0.0,
        "maximum_source_lexemes": 128,
        "maximum_target_actions": MAXIMUM_ACTIONS,
    }
    architecture = api["field_addressed_token_plan_manifest_architecture"](
        model=config,
        tokenizer=tokenizer_document,
        tokenizer_sha256=tokenizer_hash,
    )
    return api["CakeManifest"](
        schema_version="1",
        cake_id=cake_id,
        name="R42 zero-state control" if control else "R42 shared English core",
        description="R42 causal control" if control else "R42 compact shared labeled English core",
        version="1.0.0",
        publisher={"id": "abi-r42", "name": "ABI R42", "key_id": signer},
        abi_version=r41.ABI_VERSION,
        abi_hash=r41.ABI_HASH,
        cake_type="portable_decoder",
        input_contract={
            "external": "UTF-8 bytes",
            "mode": "direct_selected_portable_decoder",
            "adapter": "abi-r42-five-field-normalization",
        },
        output_contract={"external": "UTF-8 bytes", "composition": "english-core"},
        architecture=architecture,
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "incremental"]},
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](state),
        package_hash="",
        training_data_provenance={
            "selected_rows_sha256": selected_hash,
            "source_model": "Qwen/Qwen2-7B-Instruct",
            "source_revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
            "source_parameters_copied": 0,
            "teacher_absent_at_inference": True,
            "receiver_training_steps": 0 if control else STEPS,
            "control": control,
        },
        evaluation_evidence={"status": "R42_DEVELOPMENT_GATE_REQUIRED"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=("english-core",),
        permissions=("local-inference",),
    )


def _evaluate(
    api: dict[str, Any],
    packages: dict[str, Path],
    fixture: list[dict[str, Any]],
    public: bytes,
    signer: str,
    root: Path,
    device: str,
) -> tuple[list[dict[str, Any]], Any]:
    host = r41._host(api, root, public, signer, device)
    for path in packages.values():
        host.install(path)
    rows = []
    for row in fixture:
        canonical = _canonical_prompt(row["oracle_task"], row["prompt"])
        routed = _infer_task(row["prompt"])
        outputs = {}
        for condition, cake_id in (
            ("candidate", "abi-r42-english-core"),
            ("zero_state", "abi-r42-zero-state"),
        ):
            error = None
            try:
                output = host.generate(
                    cake_id, canonical, maximum_actions=MAXIMUM_ACTIONS
                ).output.decode("utf-8", errors="strict")
            except (ValueError, UnicodeDecodeError) as exc:
                output, error = "", str(exc)
            inferred, score = _runtime_score(row["prompt"], output)
            outputs[condition] = output
            rows.append(
                {
                    "device": device,
                    "condition": condition,
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
        rows[-2]["candidate_zero_exact"] = outputs["candidate"] == outputs["zero_state"]
        rows[-1]["candidate_zero_exact"] = outputs["candidate"] == outputs["zero_state"]
    return rows, host


def run(
    layercake_root: Path,
    r31_rows: Path,
    r38_rows: Path,
    r36_result: Path,
    r39_result: Path,
    r40_result: Path,
    r41_result: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R42 output exists: {output}")
    inputs = {
        "r31_rows": r31_rows,
        "r38_rows": r38_rows,
        "r36": r36_result,
        "r39": r39_result,
        "r40": r40_result,
        "r41": r41_result,
    }
    for name, path in inputs.items():
        if not path.is_file() or sha256_file(path) != EXPECTED[name]:
            raise RuntimeError(f"R42 frozen input changed: {name}")
    if not torch.cuda.is_available():
        raise RuntimeError("R42 requires CUDA")
    api = r41._layercake(layercake_root)
    selected = _selected_rows(r31_rows, r38_rows, r36_result, r39_result)
    tokenizer = _tokenizer(selected)
    output.mkdir(parents=True)
    models_dir = output / "models"
    packages_dir = output / "packages"
    models_dir.mkdir()
    packages_dir.mkdir()
    selected_path = output / "training_rows.jsonl"
    write_jsonl_once(selected_path, selected)
    selected_hash = sha256_file(selected_path)
    tokenizer_path = models_dir / "english_core.tokenizer.json"
    write_json_once(tokenizer_path, tokenizer.document())
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, training = _train(api, selected, tokenizer)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    state_path = models_dir / "english_core.safetensors"
    save_file(state, state_path)
    zero_state = {name: torch.zeros_like(value) for name, value in state.items()}
    zero_path = models_dir / "zero_state.safetensors"
    save_file(zero_state, zero_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    private, public, signer = r41._keypair(api)
    public_path = output / "research_public_key.pem"
    public_path.write_bytes(public)
    host_tokenizer_document = {
        **tokenizer.document(),
        "format": "layercake-field-addressed-token-plan/1",
    }
    host_tokenizer = api["FieldAddressedPointerTokenizer"].from_document(host_tokenizer_document)
    candidate_manifest = _manifest(
        api,
        "abi-r42-english-core",
        state,
        host_tokenizer_document,
        host_tokenizer.hash(),
        signer,
        selected_hash,
        False,
    )
    zero_manifest = _manifest(
        api,
        "abi-r42-zero-state",
        zero_state,
        host_tokenizer_document,
        host_tokenizer.hash(),
        signer,
        selected_hash,
        True,
    )
    candidate_package = api["build_package"](
        packages_dir / "english_core.cake", candidate_manifest, state, private_key=private
    )
    zero_package = api["build_package"](
        packages_dir / "zero_state.cake", zero_manifest, zero_state, private_key=private
    )
    loaded_candidate = api["load_package"](candidate_package, trust_store={signer: public})
    loaded_zero = api["load_package"](zero_package, trust_store={signer: public})
    if not all(torch.equal(state[name], loaded_candidate.tensors[name]) for name in state):
        raise RuntimeError("R42 candidate package changed tensors")
    if not all(torch.equal(zero_state[name], loaded_zero.tensors[name]) for name in zero_state):
        raise RuntimeError("R42 zero package changed tensors")

    r40_document = json.loads(r40_result.read_text(encoding="utf-8"))
    fixture_path = r40_result.parent / r40_document["artifacts"]["fixture"]["path"]
    if sha256_file(fixture_path) != r40_document["artifacts"]["fixture"]["sha256"]:
        raise RuntimeError("R42 R40 fixture changed")
    fixture = _jsonl(fixture_path)
    cpu_rows, cpu_host = _evaluate(
        api,
        {"candidate": candidate_package, "zero": zero_package},
        fixture,
        public,
        signer,
        output / "registry_cpu",
        "cpu",
    )
    cuda_rows, _ = _evaluate(
        api,
        {"candidate": candidate_package, "zero": zero_package},
        fixture,
        public,
        signer,
        output / "registry_cuda",
        "cuda",
    )
    rows = cpu_rows + cuda_rows
    peak_rss = max(peak_rss, process.memory_info().rss)
    candidate_rows = [row for row in rows if row["condition"] == "candidate"]
    zero_rows = [row for row in rows if row["condition"] == "zero_state"]
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
    r41_inventory_path = r41_result.parent / "package_inventory.json"
    r41_inventory = json.loads(r41_inventory_path.read_text(encoding="utf-8"))["packages"]
    r41_bytes = sum(int(row["package_bytes"]) for row in r41_inventory)
    lifecycle_row = fixture[0]
    lifecycle_prompt = _canonical_prompt(lifecycle_row["oracle_task"], lifecycle_row["prompt"])
    expected_first = cpu_by_id[lifecycle_row["record_id"]]["output"].encode()
    removed = cpu_host.remove("abi-r42-english-core")
    rejected = False
    try:
        cpu_host.generate("abi-r42-english-core", lifecycle_prompt)
    except KeyError:
        rejected = True
    reinstalled = cpu_host.install(candidate_package)
    restored = cpu_host.generate("abi-r42-english-core", lifecycle_prompt).output
    lifecycle = {
        "removed_status": removed["status"],
        "generation_rejected": rejected,
        "reinstalled_status": reinstalled["status"],
        "restored_exact": restored == expected_first,
        "restored_output_sha256": hashlib.sha256(restored).hexdigest(),
    }
    rows_path = output / "evaluation.jsonl"
    write_jsonl_once(rows_path, rows)
    metrics = {
        "training_rows": len(selected),
        "candidate_cpu_functional": sum(
            row["score"]["functional"] for row in candidate_rows if row["device"] == "cpu"
        ),
        "candidate_cuda_functional": sum(
            row["score"]["functional"] for row in candidate_rows if row["device"] == "cuda"
        ),
        "candidate_by_device_task": by_task,
        "candidate_cpu_cuda_exact": sum(
            cpu_by_id[key]["output"] == cuda_by_id[key]["output"] for key in cpu_by_id
        ),
        "candidate_representation_errors": sum(
            row["representation_error"] is not None for row in candidate_rows
        ),
        "candidate_route_exact": sum(row["route_exact"] for row in candidate_rows),
        "candidate_contract_exact": sum(row["contract_exact"] for row in candidate_rows),
        "candidate_noncollapsed": sum(
            bool(row["output"]) and row["score"]["noncollapsed"] for row in candidate_rows
        ),
        "zero_functional_by_device": {
            device: sum(row["score"]["functional"] for row in zero_rows if row["device"] == device)
            for device in ("cpu", "cuda")
        },
        "zero_candidate_exact_by_device": {
            device: sum(row["candidate_zero_exact"] for row in zero_rows if row["device"] == device)
            for device in ("cpu", "cuda")
        },
        "candidate_parameters": training["parameters"],
        "candidate_package_bytes": candidate_package.stat().st_size,
        "r41_candidate_package_bytes": r41_bytes,
        "package_ratio_to_r41": candidate_package.stat().st_size / r41_bytes,
    }
    passed = (
        metrics["candidate_cpu_functional"] >= 132
        and metrics["candidate_cuda_functional"] >= 132
        and all(value >= 10 for device in by_task.values() for value in device.values())
        and by_task["cpu"]["abstention"] == 12
        and by_task["cuda"]["abstention"] == 12
        and metrics["candidate_cpu_cuda_exact"] == 144
        and metrics["candidate_representation_errors"] == 0
        and metrics["candidate_route_exact"] == 288
        and metrics["candidate_contract_exact"] == 288
        and metrics["candidate_noncollapsed"] == 288
        and metrics["candidate_parameters"] <= 450_000
        and metrics["package_ratio_to_r41"] <= 0.20
        and all(value <= 36 for value in metrics["zero_functional_by_device"].values())
        and all(value <= 12 for value in metrics["zero_candidate_exact_by_device"].values())
        and loaded_candidate.manifest.tensor_payload_hash
        != loaded_zero.manifest.tensor_payload_hash
        and loaded_candidate.manifest.package_hash != loaded_zero.manifest.package_hash
        and lifecycle["generation_rejected"]
        and lifecycle["restored_exact"]
    )
    result = {
        "format": "abi-r42-shared-english-core-development/1",
        "verdict": "PASS_R42_SHARED_CORE_DEVELOPMENT"
        if passed
        else "FAIL_R42_SHARED_CORE_DEVELOPMENT",
        "inputs": {name: EXPECTED[name] for name in EXPECTED},
        "training": training,
        "metrics": metrics,
        "lifecycle": lifecycle,
        "information_accounting": {
            "selected_teacher_outputs": len(selected),
            "teacher_output_bytes": sum(len(row["teacher_output"].encode()) for row in selected),
            "teacher_output_tokens": sum(row["teacher_output_tokens"] for row in selected),
            "unique_source_prompt_bytes": sum(
                len(row["source_prompt"].encode()) for row in selected
            ),
            "new_teacher_calls": 0,
            "teacher_parameters_copied": 0,
            "teacher_logits_stored": 0,
            "teacher_activations_stored": 0,
            "receiver_training_steps_after_packaging": 0,
            "teacher_present_at_execution": False,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cpu_rss_bytes": peak_rss,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "packages": {
            "candidate": {
                "path": candidate_package.relative_to(output).as_posix(),
                "sha256": sha256_file(candidate_package),
                "package_hash": loaded_candidate.manifest.package_hash,
                "payload_hash": loaded_candidate.manifest.tensor_payload_hash,
            },
            "zero_state": {
                "path": zero_package.relative_to(output).as_posix(),
                "sha256": sha256_file(zero_package),
                "package_hash": loaded_zero.manifest.package_hash,
                "payload_hash": loaded_zero.manifest.tensor_payload_hash,
            },
        },
        "artifacts": {
            "training_rows": {"path": selected_path.name, "sha256": selected_hash},
            "tokenizer": {
                "path": tokenizer_path.relative_to(output).as_posix(),
                "sha256": sha256_file(tokenizer_path),
                "semantic_sha256": tokenizer.hash(),
            },
            "state": {
                "path": state_path.relative_to(output).as_posix(),
                "sha256": sha256_file(state_path),
            },
            "zero_state": {
                "path": zero_path.relative_to(output).as_posix(),
                "sha256": sha256_file(zero_path),
            },
            "evaluation": {"path": rows_path.name, "sha256": sha256_file(rows_path)},
            "public_key": {"path": public_path.name, "sha256": sha256_file(public_path)},
        },
        "claim_ceiling": "COMPACT_SHARED_SUPPLIED_CONTENT_ENGLISH_CORE_DEVELOPMENT_NOT_HIDDEN_OR_UNRESTRICTED_ENGLISH",
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
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
