"""Build and evaluate the preregistered R33 counterfactual normalization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn.functional as F
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30 import package_v7 as v7
from experiments.english_substrate_r30.protocol import INSTRUCTIONS, TASKS
from experiments.english_sufficiency_r31 import ladder
from experiments.english_sufficiency_r31.cascade_v3 import _runtime_score
from experiments.english_sufficiency_r31.hidden_v4 import TASK_TO_CLUSTER
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once


EXPECTED_R31_ROWS_SHA256 = "5ff7691d67b0e8271439acfbfe25540c7d42c06918b58b9257873c6f7ca9f05d"
EXPECTED_R31_FAILURE_SHA256 = "5e447168c885649f710480d510e7b99785ce1c802386572dd53d0dcc12cc180f"
EXPECTED_R32_V2_FAILURE_SHA256 = "6c9972ba6e752f07e17fb3b30746bd6dcca94d53b332441cec57262a66a80a29"
UNIQUE_ROWS_PER_CONTRACT = 48
AUGMENTED_ROWS_PER_CONTRACT = 144
EXPOSURES_PER_CONTRACT = 48 * 533
BATCH_SIZE = 24
STEPS = math.ceil(EXPOSURES_PER_CONTRACT / BATCH_SIZE)


def _augment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    augmented = []
    for row in rows:
        task = row["oracle_task"]
        prompt_lines = row["prompt"].splitlines()
        if len(prompt_lines) < 3 or not prompt_lines[0].startswith("INSTRUCTION: "):
            raise RuntimeError("R33 source prompt structure changed")
        for paraphrase_index, instruction in enumerate(INSTRUCTIONS[task]):
            prompt = "\n".join((f"INSTRUCTION: {instruction}", *prompt_lines[1:]))
            digest = hashlib.sha256(
                f"r33|{row['record_id']}|{paraphrase_index}|{prompt}".encode()
            ).hexdigest()[:20]
            item = {
                **row,
                "record_id": f"r33-train-{digest}",
                "instruction": instruction,
                "prompt": prompt,
                "source_record_id": row["record_id"],
                "paraphrase_index": paraphrase_index,
                "augmentation": "registered_semantic_paraphrase",
            }
            if not v7._score(item, item["teacher_output"], item["teacher_output"])["functional"]:
                raise RuntimeError("R33 constructed row failed the frozen functional validator")
            augmented.append(item)
    return augmented


def _train(api, rows, seed):
    tokenizer_type = api["LosslessLexemePointerTokenizer"]
    fixed = sorted({piece for row in rows for field in ("prompt", "response") for piece in tokenizer_type.split(row[field])})
    tokenizer = tokenizer_type(fixed, format_version="layercake-lossless-lexeme-pointer/2")
    encoded = ladder._encode(rows, tokenizer)
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
        maximum_target_actions=ladder.MAXIMUM_ACTIONS,
    ).to(device).bind_tokenizer(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    history = []
    for step in range(1, STEPS + 1):
        indexes = [rng.randrange(len(rows)) for _ in range(BATCH_SIZE)]
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
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "accuracy": float(prediction.eq(target[mask]).float().mean()),
                    "gradient_norm": float(gradient),
                    "seconds": time.perf_counter() - started,
                }
            )
    torch.cuda.synchronize()
    return model.cpu().eval(), tokenizer, {
        "unique_teacher_rows": UNIQUE_ROWS_PER_CONTRACT,
        "augmented_rows": len(rows),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "total_example_exposures": STEPS * BATCH_SIZE,
        "exposures_per_unique_teacher_row": STEPS * BATCH_SIZE / UNIQUE_ROWS_PER_CONTRACT,
        "exposures_per_augmented_row": STEPS * BATCH_SIZE / len(rows),
        "parameters": model.parameter_count(),
        "fixed_vocabulary_size": tokenizer.vocab_size,
        "history": history,
        "seconds": time.perf_counter() - started,
    }


def _package(api, model, tokenizer, task, cluster, output, source_hash, public_pem, private_pem, signer, training):
    artifact = api["build_token_plan_artifact"](
        model,
        tokenizer,
        domain_id=f"english-core-{cluster}",
        training={"source_rows_sha256": source_hash, "task": task, "cluster": cluster, **training},
    )
    state = artifact["state_dict"]
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id=f"abi-r33-counterfactual-{cluster}",
        name=f"ABI R33 counterfactual {cluster}",
        description="R33 counterfactually normalized supplied-content capability",
        version="0.33.0-v1",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=base.ABI_VERSION,
        abi_hash=base.ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "mode": "direct_selected_portable_decoder"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "one-counterfactual-capability"},
        architecture=api["portable_token_plan_manifest_architecture"](artifact["spec"]),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "incremental"]},
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](state),
        package_hash="",
        training_data_provenance={
            "source_model": "Qwen/Qwen2-7B-Instruct",
            "source_revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
            "source_rows_sha256": source_hash,
            "source_parameters_copied": 0,
            "new_teacher_calls": 0,
            "teacher_at_training": False,
            "teacher_at_inference": False,
            "receiver_training_steps": training["steps"],
        },
        evaluation_evidence={"status": "UNEVALUATED_R33"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=(f"english-core-{cluster}",),
        permissions=("local-inference",),
    )
    destination = output / "packages" / f"{cluster}.cake"
    api["build_package"](destination, manifest, state, private_key=private_pem)
    loaded = api["load_package"](destination, trust_store={signer: public_pem})
    return {
        "task": task,
        "cluster": cluster,
        "cake_id": loaded.manifest.cake_id,
        "path": destination,
        "relative_path": destination.relative_to(output).as_posix(),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "parameters": model.parameter_count(),
        "training": training,
    }


def run(r31_rows: Path, r31_failure: Path, r32_failure: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R33 result exists: {output}")
    expected = (
        (r31_rows, EXPECTED_R31_ROWS_SHA256),
        (r31_failure, EXPECTED_R31_FAILURE_SHA256),
        (r32_failure, EXPECTED_R32_V2_FAILURE_SHA256),
    )
    for path, digest in expected:
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"R33 prerequisite missing or changed: {path}")
    original = [row for row in base._jsonl(r31_rows) if row["split"] == "train"]
    if len(original) != 576 or any(sum(row["oracle_task"] == task for row in original) != 48 for task in TASKS):
        raise RuntimeError("R33 original source cardinality changed")
    augmented = _augment(original)
    if len(augmented) != 1728 or any(sum(row["oracle_task"] == task for row in augmented) != 144 for task in TASKS):
        raise RuntimeError("R33 augmented source cardinality changed")
    output.mkdir(parents=True)
    augmented_path = output / "augmented_rows.jsonl"
    write_jsonl_once(augmented_path, augmented)
    source_hash = sha256_file(augmented_path)
    api = base._layercake(Path(__file__).resolve().parents[2])
    mapping = {
        base._instruction_id(instruction): TASK_TO_CLUSTER[task]
        for task in TASKS
        for instruction in INSTRUCTIONS[task]
    }
    prepared = ladder._prepare(augmented, mapping, api["LosslessLexemePointerTokenizer"])
    private = Ed25519PrivateKey.from_private_bytes(base.SIGNING_SEED)
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    signer = api["key_id"](public_pem)
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    packages = []
    for position, task in enumerate(TASKS, 1):
        cluster = TASK_TO_CLUSTER[task]
        selected = [row for row in prepared if row["cluster"] == cluster]
        model, tokenizer, training = _train(api, selected, 33_000 + position)
        packages.append(_package(api, model, tokenizer, task, cluster, output, source_hash, public_pem, private_pem, signer, training))
        print(json.dumps({"trained_packages": position, "task": task, "seconds": time.perf_counter() - started}), flush=True)
        del model
        gc.collect()

    fixture_path = r31_failure.parent / "fixture.jsonl"
    fixture = base._jsonl(fixture_path)
    if len(fixture) != 144:
        raise RuntimeError("R33 development fixture changed")
    observations = []
    with tempfile.TemporaryDirectory(prefix="r33-host-") as raw:
        host = api["DirectCakeHost"](
            Path(raw) / "registry",
            abi_version=base.ABI_VERSION,
            abi_hash=base.ABI_SHA256,
            trust_store={signer: public_pem},
            device="cuda",
        )
        installed = {}
        for package in packages:
            record = host.install(package["path"])
            if record["archive_hash"] != package["sha256"]:
                raise RuntimeError("R33 package install hash mismatch")
            installed[package["cluster"]] = package["cake_id"]
        for index, row in enumerate(fixture, 1):
            inferred, _ = _runtime_score(row["prompt"], "placeholder")
            cluster = TASK_TO_CLUSTER[inferred]
            generated = host.generate(
                installed[cluster],
                v7._normalized_prompt({"prompt": row["prompt"], "nonce": row["nonce"]}),
                maximum_actions=ladder.MAXIMUM_ACTIONS,
            )
            text = generated.output.decode("utf-8", errors="strict")
            runtime_task, score = _runtime_score(row["prompt"], text)
            observations.append(
                {
                    "record_id": row["record_id"],
                    "oracle_task": row["oracle_task"],
                    "cluster": cluster,
                    "route_exact": cluster == TASK_TO_CLUSTER[row["oracle_task"]],
                    "inferred_task": runtime_task,
                    "contract_exact": runtime_task == row["oracle_task"],
                    "output": text,
                    "score": score,
                }
            )
            peak_rss = max(peak_rss, process.memory_info().rss)
            if index % 36 == 0:
                print(json.dumps({"evaluated": index, "functional": sum(item["score"]["functional"] for item in observations)}), flush=True)
        removals = []
        for package in packages:
            host.remove(package["cake_id"])
            rejected = False
            try:
                host.generate(package["cake_id"], "INSTRUCTION: test", maximum_actions=4)
            except (KeyError, ValueError, FileNotFoundError):
                rejected = True
            removals.append({"cluster": package["cluster"], "rejected": rejected})

    evaluation_path = output / "evaluation.jsonl"
    write_jsonl_once(evaluation_path, observations)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    route_exact = sum(row["route_exact"] for row in observations)
    contract_exact = sum(row["contract_exact"] for row in observations)
    passed = (
        functional >= 132
        and all(item["functional"] >= 11 for item in by_task.values())
        and by_task["abstention"]["functional"] == 12
        and route_exact == 144
        and contract_exact == 144
        and len(removals) == 12
        and all(item["rejected"] for item in removals)
    )
    result = {
        "format": "abi-r33-counterfactual-normalization/1",
        "verdict": "PASS_COUNTERFACTUAL_DEVELOPMENT" if passed else "FAIL_COUNTERFACTUAL_DEVELOPMENT",
        "inputs": {
            "r31_rows_sha256": sha256_file(r31_rows),
            "r31_failure_sha256": sha256_file(r31_failure),
            "r32_failure_sha256": sha256_file(r32_failure),
            "development_fixture_sha256": sha256_file(fixture_path),
        },
        "metrics": {
            "evaluation_rows": len(observations),
            "functional": functional,
            "by_task": by_task,
            "route_exact": route_exact,
            "contract_exact": contract_exact,
            "removal_rejections": sum(item["rejected"] for item in removals),
        },
        "normalization": {
            "unique_teacher_outputs": len(original),
            "augmented_rows": len(augmented),
            "teacher_output_tokens": sum(row["teacher_output_tokens"] for row in original),
            "teacher_output_bytes": sum(len(row["teacher_output"].encode()) for row in original),
            "new_teacher_calls": 0,
            "new_teacher_tokens": 0,
            "all_augmented_rows_functional": True,
            "method": "registered_semantic_paraphrase_counterfactual",
        },
        "packages": [{key: value for key, value in item.items() if key != "path"} for item in packages],
        "information_accounting": {
            "installed_package_bytes": sum(item["bytes"] for item in packages),
            "maximum_active_package_parameters": max(item["parameters"] for item in packages),
            "learner_steps_per_package": STEPS,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cpu_rss_bytes": int(peak_rss),
            "teacher_present_during_training": False,
            "teacher_present_during_execution": False,
            "source_parameters_copied": 0,
        },
        "artifacts": {
            "augmented_rows": {"path": augmented_path.name, "sha256": source_hash, "bytes": augmented_path.stat().st_size},
            "evaluation": {"path": evaluation_path.name, "sha256": sha256_file(evaluation_path), "bytes": evaluation_path.stat().st_size},
        },
        "removals": removals,
        "claim_ceiling": "DEVELOPMENT_RESCREEN_REQUIRES_NEW_REPLICATION",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r31-rows", type=Path, required=True)
    parser.add_argument("--r31-failure", type=Path, required=True)
    parser.add_argument("--r32-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.r31_rows.resolve(), args.r31_failure.resolve(), args.r32_failure.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
