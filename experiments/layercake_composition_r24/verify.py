"""Fail-closed live verification of the frozen R24 composition experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.factual_semantic_r16.facts import namespace_from_question
from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
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


def _jsonl(path: Path, expected: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size != expected.get("bytes")
        or sha256_file(path) != expected.get("sha256")
    ):
        raise R14Error(f"R24 raw evidence changed: {path.name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise R14Error(f"R24 raw evidence is empty or malformed: {path.name}")
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise R14Error(f"R24 raw evidence is invalid: {path.name}") from exc
    if len(rows) != expected.get("rows") or any(not isinstance(row, dict) for row in rows):
        raise R14Error(f"R24 raw evidence row count changed: {path.name}")
    return rows


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assert_packages(
    root: Path,
    run_dir: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    api: dict[str, Any],
    public_pem: bytes,
    signer: str,
) -> tuple[list[dict[str, Any]], int, int, bool]:
    systems = result.get("systems")
    if not isinstance(systems, dict) or set(systems) != {str(seed) for seed in SEEDS}:
        raise R14Error("R24 result system inventory changed")
    packages = []
    source_parameters = 0
    receiver_steps = 0
    teacher_present = False
    expected_paths = {
        (seed, namespace): (
            run_dir / "packages" / f"{domain_slug(namespace)}-seed{seed}.cake"
        ).resolve()
        for seed in SEEDS
        for namespace in NAMESPACES
    }
    for seed in SEEDS:
        seed_systems = systems[str(seed)]
        if not isinstance(seed_systems, dict) or set(seed_systems) != set(NAMESPACES):
            raise R14Error("R24 namespace inventory changed")
        for namespace in NAMESPACES:
            entry = seed_systems[namespace]
            package = entry.get("package")
            training = entry.get("training")
            if not isinstance(package, dict) or not isinstance(training, dict):
                raise R14Error("R24 system evidence is incomplete")
            target = (root / str(package.get("path", ""))).resolve()
            if (
                target != expected_paths[(seed, namespace)]
                or not target.is_file()
                or target.stat().st_size != package.get("bytes")
                or sha256_file(target) != package.get("sha256")
            ):
                raise R14Error("R24 domain package identity changed")
            loaded = api["load_package"](target, trust_store={signer: public_pem})
            provenance = loaded.manifest.training_data_provenance
            if (
                not loaded.signed
                or loaded.signature_key_id != signer
                or loaded.manifest.cake_id != package.get("cake_id")
                or loaded.manifest.package_hash != package.get("package_hash")
                or loaded.manifest.tensor_payload_hash != package.get("tensor_payload_hash")
                or loaded.manifest.cake_type != "portable_decoder"
                or loaded.manifest.domains != (namespace,)
                or loaded.manifest.input_contract.get("mode")
                != "direct_selected_portable_decoder"
                or loaded.manifest.input_contract.get("namespace") != namespace
                or package.get("signed") is not True
                or training.get("training_rows") != 24
                or training.get("steps") != config["training"]["steps"]
                or training.get("row_exposure")
                != config["training"]["row_exposure_per_package"]
                or not isinstance(training.get("history"), list)
                or not training["history"]
            ):
                raise R14Error("R24 package or training contract changed")
            if (
                package.get("source_parameters_copied")
                != provenance.get("source_parameters_copied")
                or package.get("receiver_training_steps")
                != provenance.get("receiver_training_steps")
                or package.get("teacher_at_inference")
                != provenance.get("teacher_at_inference")
            ):
                raise R14Error("R24 stored package provenance changed")
            source_parameters += int(provenance.get("source_parameters_copied", -1))
            receiver_steps += int(provenance.get("receiver_training_steps", -1))
            teacher_present = teacher_present or provenance.get("teacher_at_inference") is not False
            packages.append(package)
    if len(packages) != 6:
        raise R14Error("R24 package count changed")
    return packages, source_parameters, receiver_steps, teacher_present


def _live_recompute(
    root: Path,
    run_dir: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    api: dict[str, Any],
    base_config: dict[str, Any],
    public_pem: bytes,
    signer: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _, evaluation_raw = source_splits(root / str(config["r16_source_rows"]["path"]))
    evaluation = prepared_rows(evaluation_raw)
    r23_config = json_object(root / str(config["r23_config"]["path"]))
    r23_seed = load_seed_reveal(
        r23_config, root / str(config["r23_reveal"]["path"])
    )
    english_expected = hidden_rows(r23_seed)
    stored_rows = []
    for line in (root / str(config["r23_hidden_rows"]["path"])).read_text(
        encoding="utf-8"
    ).splitlines():
        row = json.loads(line)
        if row.get("method") == "abi_factorized" and row.get("seed") == 21022:
            stored_rows.append(row)
    stored_english = {row["record_id"]: row["output"] for row in stored_rows}
    if len(stored_english) != 120:
        raise R14Error("R24 English reference matrix changed")
    english_by_task = {
        package["cake_id"].split("-seed", 1)[0].rsplit("-", 1)[-1]: package
        for package in config["english_packages"]
    }
    if set(english_by_task) != {row["task"] for row in english_expected}:
        raise R14Error("R24 English factor mapping changed")

    observations: list[dict[str, Any]] = []
    core_rows: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    from layercake.cake.installer import InstallationError

    for seed in SEEDS:
        with tempfile.TemporaryDirectory(prefix=f"r24-verify-seed{seed}-") as raw:
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
            for namespace in NAMESPACES:
                package = result["systems"][str(seed)][namespace]["package"]
                gpu.install(root / package["path"])
                cpu.install(root / package["path"])
            for row in evaluation:
                routed = namespace_from_question(row["prompt"])
                package = result["systems"][str(seed)][routed]["package"]
                gpu_text = gpu.generate(
                    package["cake_id"], row["prompt"], maximum_actions=32
                ).output.decode("utf-8")
                cpu_text = cpu.generate(
                    package["cake_id"], row["prompt"], maximum_actions=32
                ).output.decode("utf-8")
                other = next(value for value in NAMESPACES if value != routed)
                other_package = result["systems"][str(seed)][other]["package"]
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
                text = gpu.generate(
                    package["cake_id"],
                    normalized_prompt(row["task"], row["slots"]),
                    maximum_actions=256,
                ).output.decode("utf-8")
                core_rows.append(
                    {
                        "seed": seed,
                        "stage": "domains_installed",
                        "record_id": row["record_id"],
                        "task": row["task"],
                        "output_sha256": _sha_text(text),
                        "byte_exact": text == stored_english[row["record_id"]],
                    }
                )
            for namespace in NAMESPACES:
                package = result["systems"][str(seed)][namespace]["package"]
                row = next(item for item in evaluation if item["namespace"] == namespace)
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
                        "expected_output": row["response"],
                        "restored_output": restored_output,
                        "restored_output_exact": restored_output == row["response"],
                        "corrupt_archive_sha256": sha256_file(corrupt),
                        "targeted_corruption_rejected": rejected,
                    }
                )
            for namespace in NAMESPACES:
                package = result["systems"][str(seed)][namespace]["package"]
                gpu.remove(package["cake_id"])
            for row in english_expected:
                package = english_by_task[row["task"]]
                text = gpu.generate(
                    package["cake_id"],
                    normalized_prompt(row["task"], row["slots"]),
                    maximum_actions=256,
                ).output.decode("utf-8")
                core_rows.append(
                    {
                        "seed": seed,
                        "stage": "domains_removed",
                        "record_id": row["record_id"],
                        "task": row["task"],
                        "output_sha256": _sha_text(text),
                        "byte_exact": text == stored_english[row["record_id"]],
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
                                "output_sha256": _sha_text(text),
                                "target_answer_exact": text.strip() == row["response"],
                            }
                        )
            del gpu, cpu
            gc.collect()
            torch.cuda.empty_cache()
    return observations, core_rows, lifecycle, leakage_rows


def _metrics(
    observations: list[dict[str, Any]],
    core_rows: list[dict[str, Any]],
    lifecycle: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    packages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
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


def verify(config_path: Path, run_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    config = load_config(root, config_path)
    if not torch.cuda.is_available():
        raise R14Error("R24 strict verification requires CUDA")
    result_path = run_dir / "result.json"
    result = json_object(result_path)
    if (
        result.get("format") != "abi-r24-layercake-composition-result/1"
        or result.get("config_sha256") != sha256_file(config_path)
        or result.get("evidence_sha256") != selfless_evidence_hash(result)
        or result.get("full_abi_moonshot") != "OPEN"
    ):
        raise R14Error("R24 result identity changed")
    expected_artifacts = {
        "domain_observations": "domain_observations.jsonl",
        "english_immutability": "english_immutability.jsonl",
        "lifecycle": "lifecycle.jsonl",
        "english_leakage": "english_leakage.jsonl",
    }
    if set(result.get("artifacts", {})) != set(expected_artifacts):
        raise R14Error("R24 raw artifact inventory changed")
    stored = {}
    for name, filename in expected_artifacts.items():
        ref = result["artifacts"][name]
        if ref.get("path") != filename:
            raise R14Error("R24 raw artifact path changed")
        path = run_dir / filename
        enriched = {**ref, "bytes": path.stat().st_size if path.is_file() else -1}
        stored[name] = _jsonl(path, enriched)

    api = base_run._layercake(root)
    r23_config = json_object(root / str(config["r23_config"]["path"]))
    base_config = json_object(root / str(r23_config["base_config"]["path"]))
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(base_config["layercake"]["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api["key_id"](public_pem)
    packages, source_parameters, receiver_steps, teacher_present = _assert_packages(
        root, run_dir, config, result, api, public_pem, signer
    )
    live = _live_recompute(
        root, run_dir, config, result, api, base_config, public_pem, signer
    )
    names = tuple(expected_artifacts)
    for name, live_rows in zip(names, live, strict=True):
        if stored[name] != live_rows:
            raise R14Error(f"R24 live replay changed: {name}")
    metrics = _metrics(*live, packages)
    if result.get("metrics") != metrics:
        raise R14Error("R24 aggregate metrics changed")
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
            "transformers" not in sys.modules and not teacher_present
        ),
        "source_parameters_copied_zero": source_parameters == 0,
        "host_training_steps_zero": receiver_steps == 0,
    }
    if result.get("gates") != gates or not all(gates.values()):
        raise R14Error("R24 scientific gate failed")
    accounting = result.get("information_accounting", {})
    if (
        accounting.get("new_source_calls") != 0
        or accounting.get("teacher_training_rows_reused") != 48
        or accounting.get("teacher_evaluation_rows_reused") != 48
        or accounting.get("source_parameters_copied") != source_parameters
        or accounting.get("teacher_present_at_training") is not False
        or accounting.get("teacher_present_at_execution") != teacher_present
        or accounting.get("host_training_steps") != receiver_steps
    ):
        raise R14Error("R24 information accounting changed")
    if (
        result.get("verdict") != "PASS_BOUNDED_COMPOSITION"
        or result.get("claim")
        != "R24_REGISTERED_TWO_DOMAIN_LAYERCAKE_INGESTION_AND_SEGREGATION"
        or result.get("claim_ceiling")
        != "NOT_AUTONOMOUS_LABELING_OR_OPEN_WORLD_DOMAINS"
    ):
        raise R14Error("R24 stored verdict is not a bounded pass")
    verification = {
        "format": "abi-r24-strict-live-verification/1",
        "status": "PASS_STRICTLY_VERIFIED_BOUNDED_COMPOSITION",
        "scientific_claim": result["claim"],
        "claim_ceiling": result["claim_ceiling"],
        "config_sha256": sha256_file(config_path),
        "result_sha256": sha256_file(result_path),
        "rows_live_recomputed": sum(len(rows) for rows in live),
        "domain_rows_live_recomputed": len(live[0]),
        "english_rows_live_recomputed": len(live[1]),
        "lifecycle_rows_live_recomputed": len(live[2]),
        "leakage_rows_live_recomputed": len(live[3]),
        "packages_recomputed": len(packages),
        "stored_scientific_booleans_trusted": False,
        "source_parameters_copied": source_parameters,
        "host_training_steps": receiver_steps,
        "teacher_present_at_execution": teacher_present,
        "gates": gates,
        "full_abi_moonshot": "OPEN",
    }
    verification["evidence_sha256"] = selfless_evidence_hash(verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.config, args.run_dir)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
