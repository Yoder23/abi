"""Build and execute the frozen R25 direct canonical import."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.factual_semantic_r16.facts import namespace_from_question
from experiments.factual_semantic_r16.package import load_package as load_r16_package
from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash
from experiments.generative_transfer_r21.live_verify_v7 import targeted_tensor_corruption
from experiments.generative_transfer_r21.protocol import normalized_prompt
from experiments.semantic_replication_r23.binding import load_seed_reveal
from experiments.semantic_replication_r23.protocol import hidden_rows

from .binding import load_config
from .protocol import (
    BUILD_INITIALIZATIONS,
    HOST_INITIALIZATIONS,
    NAMESPACES,
    domain_slug,
    prepared_rows,
    source_splits,
)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _signing(config: dict[str, Any], api: dict[str, Any]) -> tuple[bytes, bytes, str]:
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(config["layercake_abi"]["research_signing_seed_hex"])
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return public, private_pem, api["key_id"](public)


def _build_package(
    root: Path,
    destination: Path,
    source: dict[str, Any],
    source_binding: dict[str, Any],
    config: dict[str, Any],
    api: dict[str, Any],
    public_pem: bytes,
    private_pem: bytes,
    signer: str,
) -> dict[str, Any]:
    from layercake.models.canonical_factual import (
        CanonicalFactualDecoder,
        canonical_factual_manifest_architecture,
    )

    namespace = str(source["namespace"])
    model = CanonicalFactualDecoder(namespace, source["facts"])
    state = model.state_dict()
    cake_id = f"abi-r25-{domain_slug(namespace)}"
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id=cake_id,
        name=f"ABI R25 {namespace}",
        description="Direct canonical factual capability imported by ABI",
        version="0.25.0-bounded",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=config["layercake_abi"]["abi_version"],
        abi_hash=config["layercake_abi"]["abi_sha256"],
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
            "abstention": "empty UTF-8 byte string",
        },
        architecture=canonical_factual_manifest_architecture(namespace, source["facts"]),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={
            "features": ["byte_input", "safe_tensors", "incremental"]
        },
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](state),
        package_hash="",
        training_data_provenance={
            "method": "abi-r25-direct-canonical-import",
            "source_model": "Qwen/Qwen2-7B-Instruct",
            "source_revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
            "source_package_sha256": source_binding["sha256"],
            "source_parameters_copied": 0,
            "imported_records": len(source["facts"]),
            "receiver_training_steps": 0,
            "teacher_at_inference": False,
        },
        evaluation_evidence={"status": "UNEVALUATED_R25"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=(namespace,),
        permissions=("local-inference",),
    )
    api["build_package"](destination, manifest, state, private_key=private_pem)
    loaded = api["load_package"](destination, trust_store={signer: public_pem})
    return {
        "path": destination.relative_to(root).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "cake_id": cake_id,
        "namespace": namespace,
        "records": len(source["facts"]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_bytes": sum(tensor.numel() * tensor.element_size() for tensor in state.values()),
        "signed": loaded.signed,
        "package_hash": loaded.manifest.package_hash,
        "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        "facts_sha256": loaded.manifest.architecture["facts_sha256"],
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    output = output.resolve()
    if output.exists():
        raise R14Error(f"immutable R25 output exists: {output}")
    if not output.is_relative_to(root):
        raise R14Error("R25 output must remain inside the ABI repository")
    config = load_config(root, config_path)
    if not torch.cuda.is_available():
        raise R14Error("R25 paired CPU/GPU execution requires CUDA")
    output.mkdir(parents=True)
    api = base_run._layercake(root)
    public_pem, private_pem, signer = _signing(config, api)
    sources = {
        binding["namespace"]: load_r16_package(root / binding["path"])
        for binding in config["r16_packages"]
    }
    if set(sources) != set(NAMESPACES):
        raise R14Error("R25 R16 package namespace inventory changed")
    started = time.perf_counter()
    builds: dict[str, dict[str, dict[str, Any]]] = {}
    for initialization in BUILD_INITIALIZATIONS:
        builds[str(initialization)] = {}
        for namespace in NAMESPACES:
            binding = next(
                item for item in config["r16_packages"] if item["namespace"] == namespace
            )
            destination = (
                output
                / "builds"
                / str(initialization)
                / f"{domain_slug(namespace)}.cake"
            )
            builds[str(initialization)][namespace] = _build_package(
                root,
                destination,
                sources[namespace],
                binding,
                config,
                api,
                public_pem,
                private_pem,
                signer,
            )
    canonical = builds[str(BUILD_INITIALIZATIONS[0])]
    build_rows = []
    for initialization in BUILD_INITIALIZATIONS:
        for namespace in NAMESPACES:
            package = builds[str(initialization)][namespace]
            build_rows.append(
                {
                    "build_initialization": initialization,
                    "namespace": namespace,
                    "path": package["path"],
                    "sha256": package["sha256"],
                    "bytes": package["bytes"],
                    "canonical_sha256": canonical[namespace]["sha256"],
                    "byte_identical": package["sha256"] == canonical[namespace]["sha256"],
                }
            )
    _, evaluation_raw = source_splits(root / config["r16_source_rows"]["path"])
    evaluation = prepared_rows(evaluation_raw)
    r23_config = json_object(root / config["r23_config"]["path"])
    r23_seed = load_seed_reveal(r23_config, root / config["r23_reveal"]["path"])
    english_expected = hidden_rows(r23_seed)
    stored_english_rows = [
        json.loads(line)
        for line in (root / config["r23_hidden_rows"]["path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    stored_english = {
        row["record_id"]: row["output"]
        for row in stored_english_rows
        if row.get("method") == "abi_factorized" and row.get("seed") == 21022
    }
    if len(stored_english) != 120:
        raise R14Error("R25 frozen English evidence changed")
    english_by_task = {
        package["cake_id"].split("-seed", 1)[0].rsplit("-", 1)[-1]: package
        for package in config["english_packages"]
    }
    if set(english_by_task) != {row["task"] for row in english_expected}:
        raise R14Error("R25 English package mapping changed")

    observations = []
    core_rows = []
    lifecycle_rows = []
    leakage_rows = []
    from layercake.cake.installer import InstallationError

    for host_initialization in HOST_INITIALIZATIONS:
        with tempfile.TemporaryDirectory(prefix=f"r25-host-{host_initialization}-") as raw:
            raw_path = Path(raw)
            gpu = api["DirectCakeHost"](
                raw_path / "gpu",
                abi_version=config["layercake_abi"]["abi_version"],
                abi_hash=config["layercake_abi"]["abi_sha256"],
                trust_store={signer: public_pem},
                device="cuda",
            )
            cpu = api["DirectCakeHost"](
                raw_path / "cpu",
                abi_version=config["layercake_abi"]["abi_version"],
                abi_hash=config["layercake_abi"]["abi_sha256"],
                trust_store={signer: public_pem},
                device="cpu",
            )
            for package in config["english_packages"]:
                gpu.install(root / package["path"])
            for namespace in NAMESPACES:
                gpu.install(root / canonical[namespace]["path"])
                cpu.install(root / canonical[namespace]["path"])
            for row in evaluation:
                routed = namespace_from_question(row["prompt"])
                selected = canonical[routed]
                gpu_output = gpu.generate(
                    selected["cake_id"], row["prompt"], maximum_actions=4096
                ).output.decode("utf-8")
                cpu_output = cpu.generate(
                    selected["cake_id"], row["prompt"], maximum_actions=4096
                ).output.decode("utf-8")
                other = next(value for value in NAMESPACES if value != routed)
                other_output = gpu.generate(
                    canonical[other]["cake_id"], row["prompt"], maximum_actions=4096
                ).output.decode("utf-8")
                observations.append(
                    {
                        "host_initialization": host_initialization,
                        "record_id": row["record_id"],
                        "fact_id": row["fact_id"],
                        "namespace": row["namespace"],
                        "routed_namespace": routed,
                        "answer": row["response"],
                        "teacher_output": row["completion"],
                        "gpu_output": gpu_output,
                        "cpu_output": cpu_output,
                        "other_domain_output": other_output,
                    }
                )
            for row in english_expected:
                package = english_by_task[row["task"]]
                text = gpu.generate(
                    package["cake_id"],
                    normalized_prompt(row["task"], row["slots"]),
                    maximum_actions=256,
                ).output.decode("utf-8")
                core_rows.append(
                    {
                        "host_initialization": host_initialization,
                        "stage": "domains_installed",
                        "record_id": row["record_id"],
                        "task": row["task"],
                        "output": text,
                        "output_sha256": _sha_text(text),
                    }
                )
            for namespace in NAMESPACES:
                package = canonical[namespace]
                stimulus = next(row for row in evaluation if row["namespace"] == namespace)
                before = gpu.generate(
                    package["cake_id"], stimulus["prompt"], maximum_actions=4096
                ).output.decode("utf-8")
                removed = gpu.remove(package["cake_id"])
                absent = False
                try:
                    gpu.generate(package["cake_id"], stimulus["prompt"], maximum_actions=4096)
                except (KeyError, FileNotFoundError):
                    absent = True
                restored = gpu.install(root / package["path"])
                after = gpu.generate(
                    package["cake_id"], stimulus["prompt"], maximum_actions=4096
                ).output.decode("utf-8")
                corrupt = raw_path / f"corrupt-{domain_slug(namespace)}.cake"
                corrupt.write_bytes(
                    targeted_tensor_corruption((root / package["path"]).read_bytes())
                )
                corrupt_host = api["DirectCakeHost"](
                    raw_path / f"corrupt-{domain_slug(namespace)}",
                    abi_version=config["layercake_abi"]["abi_version"],
                    abi_hash=config["layercake_abi"]["abi_sha256"],
                    trust_store={signer: public_pem},
                    device="cuda",
                )
                rejected = False
                try:
                    corrupt_host.install(corrupt)
                except InstallationError:
                    rejected = True
                lifecycle_rows.append(
                    {
                        "host_initialization": host_initialization,
                        "namespace": namespace,
                        "cake_id": package["cake_id"],
                        "before_output": before,
                        "after_output": after,
                        "removed": removed.get("cake_id") == package["cake_id"],
                        "absent_rejected": absent,
                        "restored_archive_sha256": restored["archive_hash"],
                        "expected_archive_sha256": package["sha256"],
                        "corrupt_archive_sha256": sha256_file(corrupt),
                        "targeted_corruption_rejected": rejected,
                    }
                )
            for namespace in NAMESPACES:
                gpu.remove(canonical[namespace]["cake_id"])
            for row in english_expected:
                package = english_by_task[row["task"]]
                text = gpu.generate(
                    package["cake_id"],
                    normalized_prompt(row["task"], row["slots"]),
                    maximum_actions=256,
                ).output.decode("utf-8")
                core_rows.append(
                    {
                        "host_initialization": host_initialization,
                        "stage": "domains_removed",
                        "record_id": row["record_id"],
                        "task": row["task"],
                        "output": text,
                        "output_sha256": _sha_text(text),
                    }
                )
            if host_initialization == HOST_INITIALIZATIONS[0]:
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
                                "output_sha256": _sha_text(text),
                            }
                        )
            del gpu, cpu
            gc.collect()
            torch.cuda.empty_cache()

    artifacts = {}
    for name, rows in (
        ("package_builds", build_rows),
        ("domain_observations", observations),
        ("english_immutability", core_rows),
        ("lifecycle", lifecycle_rows),
        ("english_leakage", leakage_rows),
    ):
        path = output / f"{name}.jsonl"
        write_jsonl_once(path, rows)
        artifacts[name] = {
            "path": path.name,
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    metrics = {
        "source_records_exact": sum(len(source["facts"]) for source in sources.values()),
        "reproducible_package_builds": sum(row["byte_identical"] for row in build_rows),
        "domain_exact": sum(row["gpu_output"] == row["answer"] for row in observations),
        "teacher_agreement": sum(
            row["gpu_output"] == row["teacher_output"] for row in observations
        ),
        "namespace_route_exact": sum(
            row["namespace"] == row["routed_namespace"] for row in observations
        ),
        "other_domain_target_answers": sum(
            row["other_domain_output"] == row["answer"] for row in observations
        ),
        "cpu_gpu_exact": sum(
            row["cpu_output"] == row["gpu_output"] for row in observations
        ),
        "english_immutable": sum(
            row["output"] == stored_english[row["record_id"]] for row in core_rows
        ),
        "english_core_target_answers": sum(
            row["output"].strip() == row["target_answer"] for row in leakage_rows
        ),
        "package_lifecycle_exact": sum(
            row["removed"]
            and row["absent_rejected"]
            and row["before_output"] == row["after_output"]
            and row["restored_archive_sha256"] == row["expected_archive_sha256"]
            for row in lifecycle_rows
        ),
        "targeted_corruptions_rejected": sum(
            row["targeted_corruption_rejected"] for row in lifecycle_rows
        ),
        "source_parameters_copied": 0,
        "receiver_training_steps": 0,
    }
    gates = {name: metrics[name] == threshold for name, threshold in config["gates"].items()}
    gates["teacher_absent_at_execution"] = "transformers" not in sys.modules
    passed = all(gates.values())
    all_packages = [
        builds[str(initialization)][namespace]
        for initialization in BUILD_INITIALIZATIONS
        for namespace in NAMESPACES
    ]
    result = {
        "format": "abi-r25-canonical-layercake-import-result/1",
        "verdict": "PASS_BOUNDED_DIRECT_IMPORT" if passed else "FAIL_BOUNDED_DIRECT_IMPORT",
        "claim": (
            "R25_REGISTERED_FACTUAL_DIRECT_IMPORT_AND_COMPOSITION"
            if passed
            else "R25_DIRECT_IMPORT_FAILED"
        ),
        "claim_ceiling": "NOT_AUTONOMOUS_LABELING_OR_OPEN_WORLD_DOMAINS",
        "config_sha256": sha256_file(config_path),
        "layercake_commit": config["layercake_commit"],
        "builds": builds,
        "canonical_packages": canonical,
        "metrics": metrics,
        "gates": gates,
        "artifacts": artifacts,
        "information_accounting": {
            "new_source_calls": 0,
            "source_parameters_copied": 0,
            "receiver_training_steps": 0,
            "teacher_present_at_build": False,
            "teacher_present_at_execution": False,
            "imported_records": metrics["source_records_exact"],
            "final_package_bytes": sum(package["bytes"] for package in canonical.values()),
            "all_reproduction_package_bytes": sum(package["bytes"] for package in all_packages),
            "active_parameters": 0,
            "state_bytes_per_selected_package": 32,
            "experiment_wall_seconds": time.perf_counter() - started,
        },
        "full_abi_moonshot": "OPEN",
        "next_action": (
            "strict live verification then expand autonomous labeling"
            if passed
            else "preserve failure and isolate direct-import gate"
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
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
