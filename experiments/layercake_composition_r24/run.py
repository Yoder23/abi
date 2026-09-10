"""Train, package, compose, and evaluate the frozen R24 domain layers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.factual_semantic_r16.facts import namespace_from_question
from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)
from experiments.generative_transfer_r21.live_verify_v7 import (
    targeted_tensor_corruption,
)
from experiments.generative_transfer_r21.protocol import normalized_prompt
from experiments.semantic_replication_r23.binding import load_seed_reveal
from experiments.semantic_replication_r23.protocol import hidden_rows

from .binding import load_config
from .protocol import NAMESPACES, SEEDS, domain_slug, prepared_rows, source_splits


def _encode(rows: list[dict[str, Any]], tokenizer: Any) -> list[tuple[list[int], list[int]]]:
    encoded = []
    for row in rows:
        source, lexemes = tokenizer.encode_source(row["prompt"])
        target = tokenizer.encode_target(
            row["response"],
            copy_lexemes=row["copy_lexemes"],
            source_lexemes=lexemes,
        )
        if len(source) > 128 or len(target) > 32:
            raise R14Error("R24 encoded row exceeds frozen boundary")
        if tokenizer.decode_actions(target, lexemes).decode("utf-8") != row["response"]:
            raise R14Error("R24 target encoding changed answer bytes")
        encoded.append((source, target))
    return encoded


def _train(
    api: dict[str, Any],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = api["LosslessLexemePointerTokenizer"].build_generic(rows)
    encoded = _encode(rows, tokenizer)
    model_config = dict(config["model"])
    maximum_source = int(model_config.pop("maximum_source_lexemes"))
    maximum_target = int(model_config.pop("maximum_target_actions"))
    model = api["PortableTokenPlan"](
        fixed_vocab_size=tokenizer.vocab_size,
        **model_config,
        maximum_source_lexemes=maximum_source,
        maximum_target_actions=maximum_target,
    ).cuda().bind_tokenizer(tokenizer)
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    randomizer = random.Random(seed + 1)
    order = list(range(len(rows)))
    cursor = len(order)
    history = []
    exposure = 0
    action_exposure = 0
    started = time.perf_counter()
    model.train()
    for step in range(1, int(settings["steps"]) + 1):
        batch_size = int(settings["batch_size"])
        if cursor + batch_size > len(order):
            randomizer.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        source, target = base_run._batch(encoded, indices, torch.device("cuda"))
        output = model(source, target)
        mask = target.ge(0)
        loss = F.nll_loss(output["log_probs"][mask], target[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        exposure += len(indices)
        action_exposure += int(mask.sum().item())
        if step == 1 or step % 250 == 0 or step == int(settings["steps"]):
            prediction = output["log_probs"][mask].argmax(dim=-1)
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "action_accuracy": float(
                        prediction.eq(target[mask]).float().mean().item()
                    ),
                    "gradient_norm": float(gradient),
                }
            )
    torch.cuda.synchronize()
    model.eval()
    return model.cpu(), tokenizer, {
        "parameters": model.parameter_count(),
        "fixed_vocabulary_size": tokenizer.vocab_size,
        "training_rows": len(rows),
        "steps": int(settings["steps"]),
        "row_exposure": exposure,
        "target_action_exposure": action_exposure,
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }


def _package(
    api: dict[str, Any],
    model: Any,
    tokenizer: Any,
    *,
    root: Path,
    destination: Path,
    namespace: str,
    seed: int,
    config: dict[str, Any],
    public_pem: bytes,
    private_pem: bytes,
    signer: str,
) -> dict[str, Any]:
    artifact = api["build_token_plan_artifact"](
        model,
        tokenizer,
        domain_id=namespace,
        training={
            "method": "abi-r24-r16-fact-to-layercake",
            "seed": seed,
            "namespace": namespace,
            "source_sha256": config["r16_source_rows"]["sha256"],
        },
    )
    state = artifact["state_dict"]
    cake_id = f"abi-r24-{domain_slug(namespace)}-seed{seed}"
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id=cake_id,
        name=f"ABI R24 {namespace} seed {seed}",
        description="R24 teacher-extracted factual LayerCake domain",
        version="0.24.0-bounded",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=config["layercake"]["abi_version"],
        abi_hash=config["layercake"]["abi_sha256"],
        cake_type="portable_decoder",
        input_contract={
            "external": "UTF-8 bytes",
            "role": "domain-layer",
            "namespace": namespace,
            "mode": "direct_selected_portable_decoder",
        },
        output_contract={
            "external": "UTF-8 bytes",
            "role": "domain-answer",
            "composition": "explicit_namespace_selection",
        },
        architecture=api["portable_token_plan_manifest_architecture"](artifact["spec"]),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={
            "features": ["byte_input", "safe_tensors", "incremental"]
        },
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](state),
        package_hash="",
        training_data_provenance={
            "source_model": "Qwen/Qwen2-7B-Instruct",
            "source_revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
            "source_rows_sha256": config["r16_source_rows"]["sha256"],
            "source_parameters_copied": 0,
            "teacher_at_package_training": False,
            "teacher_at_inference": False,
            "receiver_training_steps": 0,
        },
        evaluation_evidence={"status": "UNEVALUATED_R24"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=(namespace,),
        permissions=("local-inference",),
    )
    api["build_package"](destination, manifest, state, private_key=private_pem)
    loaded = api["load_package"](destination, trust_store={signer: public_pem})
    provenance = dict(loaded.manifest.training_data_provenance)
    return {
        "path": destination.relative_to(root).as_posix(),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "cake_id": cake_id,
        "namespace": namespace,
        "parameters": model.parameter_count(),
        "signed": loaded.signed,
        "package_hash": loaded.manifest.package_hash,
        "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        "source_parameters_copied": provenance.get("source_parameters_copied"),
        "receiver_training_steps": provenance.get("receiver_training_steps"),
        "teacher_at_inference": provenance.get("teacher_at_inference"),
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    output = output.resolve()
    if output.exists():
        raise R14Error(f"immutable R24 output exists: {output}")
    if not output.is_relative_to(root):
        raise R14Error("R24 output must remain inside the ABI repository")
    config = load_config(root, config_path)
    if not torch.cuda.is_available():
        raise R14Error("R24 training and primary evaluation require CUDA")
    training_raw, evaluation_raw = source_splits(
        root / str(config["r16_source_rows"]["path"])
    )
    training = prepared_rows(training_raw)
    evaluation = prepared_rows(evaluation_raw)
    output.mkdir(parents=True)
    api = base_run._layercake(root)
    base = json_object(root / str(config["r23_config"]["path"]))
    base_config = json_object(root / str(base["base_config"]["path"]))
    config["layercake"] = base_config["layercake"]
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(base_config["layercake"]["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    signer = api["key_id"](public_pem)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    systems: dict[str, Any] = {}
    for seed in SEEDS:
        systems[str(seed)] = {}
        for namespace in NAMESPACES:
            rows = [row for row in training if row["namespace"] == namespace]
            model, tokenizer, metadata = _train(api, rows, config, seed)
            destination = (
                output / "packages" / f"{domain_slug(namespace)}-seed{seed}.cake"
            )
            package = _package(
                api,
                model,
                tokenizer,
                root=root,
                destination=destination,
                namespace=namespace,
                seed=seed,
                config=config,
                public_pem=public_pem,
                private_pem=private_pem,
                signer=signer,
            )
            systems[str(seed)][namespace] = {"training": metadata, "package": package}
            print(
                json.dumps(
                    {
                        "trained_seed": seed,
                        "namespace": namespace,
                        "final_loss": metadata["history"][-1]["loss"],
                    }
                ),
                flush=True,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()

    r23_seed = load_seed_reveal(
        base, root / str(config["r23_reveal"]["path"])
    )
    english_expected = hidden_rows(r23_seed)
    stored_english_rows = []
    for line in (root / str(config["r23_hidden_rows"]["path"])).read_text(
        encoding="utf-8"
    ).splitlines():
        if line:
            row = json.loads(line)
            if row["method"] == "abi_factorized" and row["seed"] == 21022:
                stored_english_rows.append(row)
    stored_english = {row["record_id"]: row["output"] for row in stored_english_rows}
    if len(stored_english) != 120:
        raise R14Error("R24 frozen English matrix changed")
    english_by_task = {
        package["cake_id"].split("-seed", 1)[0].rsplit("-", 1)[-1]: package
        for package in config["english_packages"]
    }
    if set(english_by_task) != {
        "prose",
        "summary",
        "email",
        "bullets",
        "clarification",
        "abstention",
    }:
        raise R14Error("R24 English factor selection changed")

    observations = []
    core_rows = []
    lifecycle = []
    leakage_rows = []
    from layercake.cake.installer import InstallationError

    for seed in SEEDS:
        with tempfile.TemporaryDirectory(prefix=f"r24-seed{seed}-") as raw:
            raw_path = Path(raw)
            gpu = api["DirectCakeHost"](
                raw_path / "gpu",
                abi_version=base_config["layercake"]["abi_version"],
                abi_hash=base_config["layercake"]["abi_sha256"],
                trust_store={signer: public_pem},
                device="cuda",
            )
            cpu = api["DirectCakeHost"](
                raw_path / "cpu",
                abi_version=base_config["layercake"]["abi_version"],
                abi_hash=base_config["layercake"]["abi_sha256"],
                trust_store={signer: public_pem},
                device="cpu",
            )
            for package in config["english_packages"]:
                gpu.install(root / package["path"])
            baseline = {}
            for row in english_expected:
                package = english_by_task[row["task"]]
                prompt = normalized_prompt(row["task"], row["slots"])
                text = gpu.generate(
                    package["cake_id"], prompt, maximum_actions=256
                ).output.decode("utf-8")
                if text != stored_english[row["record_id"]]:
                    raise R14Error("R24 pre-install English output changed")
                baseline[row["record_id"]] = text
            for namespace in NAMESPACES:
                package = systems[str(seed)][namespace]["package"]
                gpu.install(root / package["path"])
                cpu.install(root / package["path"])
            for row in evaluation:
                routed = namespace_from_question(row["prompt"])
                package = systems[str(seed)][routed]["package"]
                gpu_text = gpu.generate(
                    package["cake_id"], row["prompt"], maximum_actions=32
                ).output.decode("utf-8")
                cpu_text = cpu.generate(
                    package["cake_id"], row["prompt"], maximum_actions=32
                ).output.decode("utf-8")
                other = next(namespace for namespace in NAMESPACES if namespace != routed)
                other_package = systems[str(seed)][other]["package"]
                other_text = gpu.generate(
                    other_package["cake_id"], row["prompt"], maximum_actions=32
                ).output.decode("utf-8")
                observations.append(
                    {
                        "seed": seed,
                        "record_id": row["record_id"],
                        "fact_id": row["fact_id"],
                        "namespace": row["namespace"],
                        "routed_namespace": routed,
                        "answer": row["response"],
                        "teacher_output": row["completion"],
                        "gpu_output": gpu_text,
                        "cpu_output": cpu_text,
                        "gpu_exact": gpu_text == row["response"],
                        "cpu_gpu_exact": cpu_text == gpu_text,
                        "teacher_agreement": gpu_text == row["answer"],
                        "other_domain_output": other_text,
                        "other_domain_target_exact": other_text == row["response"],
                    }
                )
            for row in english_expected:
                package = english_by_task[row["task"]]
                prompt = normalized_prompt(row["task"], row["slots"])
                text = gpu.generate(
                    package["cake_id"], prompt, maximum_actions=256
                ).output.decode("utf-8")
                core_rows.append(
                    {
                        "seed": seed,
                        "stage": "domains_installed",
                        "record_id": row["record_id"],
                        "task": row["task"],
                        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "byte_exact": text == baseline[row["record_id"]],
                    }
                )
            for namespace in NAMESPACES:
                package = systems[str(seed)][namespace]["package"]
                row = next(item for item in evaluation if item["namespace"] == namespace)
                expected_output = row["response"]
                removed = gpu.remove(package["cake_id"])
                absent = False
                try:
                    gpu.generate(package["cake_id"], row["prompt"], maximum_actions=32)
                except (KeyError, FileNotFoundError):
                    absent = True
                restored = gpu.install(root / package["path"])
                restored_output = gpu.generate(
                    package["cake_id"], row["prompt"], maximum_actions=32
                ).output.decode("utf-8")
                corrupt = raw_path / f"corrupt-{domain_slug(namespace)}.cake"
                corrupt.write_bytes(
                    targeted_tensor_corruption((root / package["path"]).read_bytes())
                )
                corrupt_host = api["DirectCakeHost"](
                    raw_path / f"corrupt-{domain_slug(namespace)}",
                    abi_version=base_config["layercake"]["abi_version"],
                    abi_hash=base_config["layercake"]["abi_sha256"],
                    trust_store={signer: public_pem},
                    device="cuda",
                )
                rejected = False
                try:
                    corrupt_host.install(corrupt)
                except InstallationError:
                    rejected = True
                lifecycle.append(
                    {
                        "seed": seed,
                        "namespace": namespace,
                        "cake_id": package["cake_id"],
                        "removed": removed.get("cake_id") == package["cake_id"],
                        "absent_rejected": absent,
                        "expected_archive_sha256": package["sha256"],
                        "restored_archive_sha256": restored["archive_hash"],
                        "restored_archive_exact": restored["archive_hash"]
                        == package["sha256"],
                        "expected_output": expected_output,
                        "restored_output": restored_output,
                        "restored_output_exact": restored_output == expected_output,
                        "corrupt_archive_sha256": sha256_file(corrupt),
                        "targeted_corruption_rejected": rejected,
                    }
                )
            for namespace in NAMESPACES:
                package = systems[str(seed)][namespace]["package"]
                removed = gpu.remove(package["cake_id"])
                if removed.get("cake_id") != package["cake_id"]:
                    raise R14Error("R24 final domain removal failed")
            for row in english_expected:
                package = english_by_task[row["task"]]
                prompt = normalized_prompt(row["task"], row["slots"])
                text = gpu.generate(
                    package["cake_id"], prompt, maximum_actions=256
                ).output.decode("utf-8")
                core_rows.append(
                    {
                        "seed": seed,
                        "stage": "domains_removed",
                        "record_id": row["record_id"],
                        "task": row["task"],
                        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "byte_exact": text == baseline[row["record_id"]],
                    }
                )
            if seed == SEEDS[0]:
                for row in evaluation:
                    for task, package in english_by_task.items():
                        text = gpu.generate(
                            package["cake_id"], row["prompt"], maximum_actions=256
                        ).output.decode("utf-8")
                        leakage_rows.append(
                            {
                                "record_id": row["record_id"],
                                "namespace": row["namespace"],
                                "english_task": task,
                                "target_answer": row["response"],
                                "output": text,
                                "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "target_answer_exact": text.strip() == row["response"],
                            }
                        )
            del gpu, cpu
            gc.collect()
            torch.cuda.empty_cache()

    observations_path = output / "domain_observations.jsonl"
    core_path = output / "english_immutability.jsonl"
    lifecycle_path = output / "lifecycle.jsonl"
    leakage_path = output / "english_leakage.jsonl"
    write_jsonl_once(observations_path, observations)
    write_jsonl_once(core_path, core_rows)
    write_jsonl_once(lifecycle_path, lifecycle)
    write_jsonl_once(leakage_path, leakage_rows)
    packages = [
        systems[str(seed)][namespace]["package"]
        for seed in SEEDS
        for namespace in NAMESPACES
    ]
    source_parameters_copied = sum(
        int(package["source_parameters_copied"]) for package in packages
    )
    receiver_training_steps = sum(
        int(package["receiver_training_steps"]) for package in packages
    )
    teacher_at_inference = any(
        package["teacher_at_inference"] is not False for package in packages
    )
    metrics = {
        "domain_rows": len(observations),
        "domain_exact": sum(row["gpu_exact"] for row in observations),
        "teacher_agreement": sum(row["teacher_agreement"] for row in observations),
        "namespace_route_exact": sum(
            row["namespace"] == row["routed_namespace"] for row in observations
        ),
        "other_domain_target_answers": sum(
            row["other_domain_target_exact"] for row in observations
        ),
        "cpu_gpu_exact": sum(row["cpu_gpu_exact"] for row in observations),
        "english_immutable": sum(row["byte_exact"] for row in core_rows),
        "english_leakage_target_answers": sum(
            row["target_answer_exact"] for row in leakage_rows
        ),
        "packages_signed": sum(package["signed"] for package in packages),
        "package_lifecycle_exact": sum(
            all(
                row[key]
                for key in (
                    "removed",
                    "absent_rejected",
                    "restored_archive_exact",
                    "restored_output_exact",
                )
            )
            for row in lifecycle
        ),
        "targeted_corruptions_rejected": sum(
            row["targeted_corruption_rejected"] for row in lifecycle
        ),
    }
    gates = {
        "domain_exact_per_seed": metrics["domain_exact"] == 48 * len(SEEDS),
        "teacher_agreement": metrics["teacher_agreement"] == 48 * len(SEEDS),
        "namespace_route_exact": metrics["namespace_route_exact"] == 48 * len(SEEDS),
        "other_domain_isolation": metrics["other_domain_target_answers"] == 0,
        "english_core_no_domain_answer": metrics["english_leakage_target_answers"] == 0,
        "english_immutable": metrics["english_immutable"] == 120 * len(SEEDS) * 2,
        "cpu_gpu_exact": metrics["cpu_gpu_exact"] == 48 * len(SEEDS),
        "packages_signed": metrics["packages_signed"] == 6,
        "package_lifecycle": metrics["package_lifecycle_exact"] == 6,
        "targeted_corruption": metrics["targeted_corruptions_rejected"] == 6,
        "teacher_absent_at_execution": (
            "transformers" not in sys.modules and not teacher_at_inference
        ),
        "source_parameters_copied_zero": source_parameters_copied == 0,
        "host_training_steps_zero": receiver_training_steps == 0,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-r24-layercake-composition-result/1",
        "verdict": "PASS_BOUNDED_COMPOSITION" if passed else "FAIL_BOUNDED_COMPOSITION",
        "claim": (
            "R24_REGISTERED_TWO_DOMAIN_LAYERCAKE_INGESTION_AND_SEGREGATION"
            if passed
            else "R24_LAYERCAKE_COMPOSITION_FAILED"
        ),
        "claim_ceiling": "NOT_AUTONOMOUS_LABELING_OR_OPEN_WORLD_DOMAINS",
        "config_sha256": sha256_file(config_path),
        "systems": systems,
        "metrics": metrics,
        "gates": gates,
        "artifacts": {
            "domain_observations": {
                "path": observations_path.name,
                "rows": len(observations),
                "sha256": sha256_file(observations_path),
            },
            "english_immutability": {
                "path": core_path.name,
                "rows": len(core_rows),
                "sha256": sha256_file(core_path),
            },
            "lifecycle": {
                "path": lifecycle_path.name,
                "rows": len(lifecycle),
                "sha256": sha256_file(lifecycle_path),
            },
            "english_leakage": {
                "path": leakage_path.name,
                "rows": len(leakage_rows),
                "sha256": sha256_file(leakage_path),
            },
        },
        "information_accounting": {
            "new_source_calls": 0,
            "teacher_training_rows_reused": 48,
            "teacher_evaluation_rows_reused": 48,
            "source_parameters_copied": source_parameters_copied,
            "bridge_parameters_trained": sum(
                package["parameters"] for package in packages
            ),
            "final_domain_package_bytes": sum(package["bytes"] for package in packages),
            "training_seconds": sum(
                systems[str(seed)][namespace]["training"]["training_seconds"]
                for seed in SEEDS
                for namespace in NAMESPACES
            ),
            "experiment_wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "teacher_present_at_training": False,
            "teacher_present_at_execution": teacher_at_inference,
            "host_training_steps": receiver_training_steps,
        },
        "full_abi_moonshot": "OPEN",
        "next_action": (
            "strict verification then expand labeling beyond registered ontology"
            if passed
            else "preserve failure and isolate the failing domain or host gate"
        ),
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.config, args.output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
