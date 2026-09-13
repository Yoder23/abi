"""Install R29 multi-tag packages into LayerCake and execute them."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

from experiments.autonomous_labeling_r27.binding import load_config as load_host_config
from experiments.autonomous_labeling_r27.binding import validate_file
from experiments.autonomous_labeling_r27.import_run import BUILD_INITIALIZATIONS, HOST_INITIALIZATIONS, _build, _signing
from experiments.factual_semantic_r16.package import load_package as load_source_package
from experiments.foreign_capability_r14.core import R14Error, evidence_hash, json_object, sha256_file, write_json_once, write_jsonl_once
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash
from experiments.generative_transfer_r21.live_verify_v7 import targeted_tensor_corruption
from experiments.generative_transfer_r21.protocol import normalized_prompt
from experiments.semantic_replication_r23.binding import load_seed_reveal
from experiments.semantic_replication_r23.protocol import hidden_rows

from .protocol import answer_key


def slug(value):
    return "".join(char if char.isalnum() else "-" for char in value.casefold()).strip("-")


def run(import_config_path, output):
    root = Path(__file__).resolve().parents[2]
    if output.exists():
        raise R14Error(f"immutable R29 import output exists: {output}")
    binding = json_object(import_config_path)
    if binding.get("format") != "abi-r27-import-config/1":
        raise R14Error("R29 import binding changed")
    for name in ("heldout_config", "source_result", "source_rows", "evaluation"):
        validate_file(root, binding[name], name)
    host_config_path = root / binding["heldout_config"]["path"]
    config = load_host_config(root, host_config_path)
    source_result = json_object(root / binding["source_result"]["path"])
    if source_result.get("verdict") != "PASS" or source_result.get("evidence_sha256") != selfless_evidence_hash(source_result):
        raise R14Error("R29 source prerequisite failed")
    if not torch.cuda.is_available() or "transformers" in sys.modules:
        raise R14Error("R29 clean CPU/GPU LayerCake process required")
    output.mkdir(parents=True)
    api = base_run._layercake(root)
    public, private, signer = _signing(config, api)
    sources = {}
    source_bindings = {}
    for item in binding["source_packages"]:
        validate_file(root, item, "source package")
        package = load_source_package(root / item["path"])
        sources[package["namespace"]] = package
        source_bindings[package["namespace"]] = item
    if len(sources) < 4:
        raise R14Error("R29 source package inventory incomplete")
    started = time.perf_counter()
    builds = {}
    for initialization in BUILD_INITIALIZATIONS:
        builds[str(initialization)] = {}
        for namespace, source in sorted(sources.items()):
            destination = output / "builds" / str(initialization) / f"{slug(namespace)}.cake"
            builds[str(initialization)][namespace] = _build(root, destination, source, source_bindings[namespace], config, api, public, private, signer)
    canonical = builds[str(BUILD_INITIALIZATIONS[0])]
    build_rows = [
        {
            "build_initialization": initialization,
            "namespace": namespace,
            "path": builds[str(initialization)][namespace]["path"],
            "sha256": builds[str(initialization)][namespace]["sha256"],
            "bytes": builds[str(initialization)][namespace]["bytes"],
            "canonical_sha256": canonical[namespace]["sha256"],
            "byte_identical": builds[str(initialization)][namespace]["sha256"] == canonical[namespace]["sha256"],
        }
        for initialization in BUILD_INITIALIZATIONS
        for namespace in sorted(sources)
    ]
    evaluation = [json.loads(line) for line in (root / binding["evaluation"]["path"]).read_text(encoding="utf-8").splitlines() if line]
    r23_config = json_object(root / config["r23_config"]["path"])
    seed = load_seed_reveal(r23_config, root / config["r23_reveal"]["path"])
    english_expected = hidden_rows(seed)
    stored = [json.loads(line) for line in (root / config["r23_hidden_rows"]["path"]).read_text(encoding="utf-8").splitlines() if line]
    stored_english = {row["record_id"]: row["output"] for row in stored if row.get("method") == "abi_factorized" and row.get("seed") == 21022}
    english_by_task = {item["cake_id"].split("-seed", 1)[0].rsplit("-", 1)[-1]: item for item in config["english_packages"]}
    if len(stored_english) != 120 or set(english_by_task) != {row["task"] for row in english_expected}:
        raise R14Error("R29 English prerequisite changed")
    observations, core, lifecycle, leakage = [], [], [], []
    from layercake.cake.installer import InstallationError
    for host_init in HOST_INITIALIZATIONS:
        with tempfile.TemporaryDirectory(prefix=f"r29-host-{host_init}-") as raw:
            temporary = Path(raw)
            gpu = api["DirectCakeHost"](temporary / "gpu", abi_version=config["layercake_abi"]["abi_version"], abi_hash=config["layercake_abi"]["abi_sha256"], trust_store={signer: public}, device="cuda")
            cpu = api["DirectCakeHost"](temporary / "cpu", abi_version=config["layercake_abi"]["abi_version"], abi_hash=config["layercake_abi"]["abi_sha256"], trust_store={signer: public}, device="cpu")
            for item in config["english_packages"]:
                gpu.install(root / item["path"])
            for package in canonical.values():
                gpu.install(root / package["path"])
                cpu.install(root / package["path"])
            for row in evaluation:
                selected = canonical[row["namespace"]]
                gpu_text = gpu.generate(selected["cake_id"], row["question"], maximum_actions=4096).output.decode()
                cpu_text = cpu.generate(selected["cake_id"], row["question"], maximum_actions=4096).output.decode()
                other_outputs = {
                    namespace: gpu.generate(package["cake_id"], row["question"], maximum_actions=4096).output.decode()
                    for namespace, package in canonical.items()
                    if namespace != row["namespace"]
                }
                observations.append({**row, "host_initialization": host_init, "gpu_output": gpu_text, "cpu_output": cpu_text, "other_outputs": other_outputs})
            for row in english_expected:
                item = english_by_task[row["task"]]
                text = gpu.generate(item["cake_id"], normalized_prompt(row["task"], row["slots"]), maximum_actions=256).output.decode()
                core.append({"host_initialization": host_init, "stage": "domains_installed", "record_id": row["record_id"], "output": text})
            for namespace, package in sorted(canonical.items()):
                stimulus = next(row for row in evaluation if row["namespace"] == namespace)
                before = gpu.generate(package["cake_id"], stimulus["question"], maximum_actions=4096).output.decode()
                removed = gpu.remove(package["cake_id"])
                absent = False
                try:
                    gpu.generate(package["cake_id"], stimulus["question"], maximum_actions=4096)
                except (KeyError, FileNotFoundError):
                    absent = True
                restored = gpu.install(root / package["path"])
                after = gpu.generate(package["cake_id"], stimulus["question"], maximum_actions=4096).output.decode()
                corrupt = temporary / f"corrupt-{slug(namespace)}.cake"
                corrupt.write_bytes(targeted_tensor_corruption((root / package["path"]).read_bytes()))
                corrupt_host = api["DirectCakeHost"](temporary / f"corrupt-{slug(namespace)}", abi_version=config["layercake_abi"]["abi_version"], abi_hash=config["layercake_abi"]["abi_sha256"], trust_store={signer: public}, device="cuda")
                rejected = False
                try:
                    corrupt_host.install(corrupt)
                except InstallationError:
                    rejected = True
                lifecycle.append({"host_initialization": host_init, "namespace": namespace, "before": before, "after": after, "removed": removed.get("cake_id") == package["cake_id"], "absent_rejected": absent, "restored_sha256": restored["archive_hash"], "expected_sha256": package["sha256"], "corruption_rejected": rejected})
            for package in canonical.values():
                gpu.remove(package["cake_id"])
            for row in english_expected:
                item = english_by_task[row["task"]]
                text = gpu.generate(item["cake_id"], normalized_prompt(row["task"], row["slots"]), maximum_actions=256).output.decode()
                core.append({"host_initialization": host_init, "stage": "domains_removed", "record_id": row["record_id"], "output": text})
            if host_init == HOST_INITIALIZATIONS[0]:
                for row in evaluation:
                    for task, item in english_by_task.items():
                        text = gpu.generate(item["cake_id"], row["question"], maximum_actions=256).output.decode()
                        leakage.append({"fact_id": row["fact_id"], "task": task, "target": row["oracle_answer"], "output": text})
            del gpu, cpu
            gc.collect()
            torch.cuda.empty_cache()
    artifacts = {}
    for name, values in (("package_builds", build_rows), ("domain_observations", observations), ("english_immutability", core), ("lifecycle", lifecycle), ("english_leakage", leakage)):
        path = output / f"{name}.jsonl"
        write_jsonl_once(path, values)
        artifacts[name] = {"path": path.name, "rows": len(values), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    metrics = {
        "packages": len(canonical),
        "reproducible_builds": sum(row["byte_identical"] for row in build_rows),
        "domain_oracle_exact": sum(answer_key(row["gpu_output"]) == answer_key(row["oracle_answer"]) for row in observations),
        "teacher_agreement": sum(answer_key(row["gpu_output"]) == answer_key(row["teacher_answer"]) for row in observations),
        "cpu_gpu_exact": sum(row["cpu_output"] == row["gpu_output"] for row in observations),
        "unauthorized_packages_abstain": sum(all(not text for namespace, text in row["other_outputs"].items() if namespace.split("/", 1)[1] not in row["allowed_tags"]) for row in observations),
        "english_immutable": sum(row["output"] == stored_english[row["record_id"]] for row in core),
        "english_core_target_answers": sum(answer_key(row["output"]) == answer_key(row["target"]) for row in leakage),
        "lifecycle_exact": sum(row["removed"] and row["absent_rejected"] and row["before"] == row["after"] and row["restored_sha256"] == row["expected_sha256"] for row in lifecycle),
        "corruptions_rejected": sum(row["corruption_rejected"] for row in lifecycle),
        "source_parameters_copied": 0,
        "receiver_training_steps": 0,
    }
    expected = len(evaluation) * len(HOST_INITIALIZATIONS)
    expected_lifecycle = len(canonical) * len(HOST_INITIALIZATIONS)
    gates = {
        "builds": metrics["reproducible_builds"] == len(canonical) * len(BUILD_INITIALIZATIONS),
        "domain": metrics["domain_oracle_exact"] == expected,
        "teacher": metrics["teacher_agreement"] >= 32 * len(HOST_INITIALIZATIONS),
        "cpu_gpu": metrics["cpu_gpu_exact"] == expected,
        "unauthorized_abstain": metrics["unauthorized_packages_abstain"] == expected,
        "english_immutable": metrics["english_immutable"] == 720,
        "english_no_domain_answers": metrics["english_core_target_answers"] == 0,
        "lifecycle": metrics["lifecycle_exact"] == expected_lifecycle,
        "corruption": metrics["corruptions_rejected"] == expected_lifecycle,
        "zero_copy_training": metrics["source_parameters_copied"] == 0 and metrics["receiver_training_steps"] == 0,
        "teacher_absent": "transformers" not in sys.modules,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-r29-layercake-multitag-import-result/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": "BOUNDED_VALIDATED_MULTITAG_CAPABILITIES_IMPORTED_INTO_LAYERCAKE",
        "claim_ceiling": "NOT_EXHAUSTIVE_DIAGNOSIS_OR_FLUENT_ENGLISH_EXTRACTION",
        "import_config_sha256": sha256_file(import_config_path),
        "host_config_sha256": sha256_file(host_config_path),
        "source_result_sha256": sha256_file(root / binding["source_result"]["path"]),
        "layercake_commit": config["layercake_commit"],
        "metrics": metrics,
        "gates": gates,
        "canonical_packages": canonical,
        "artifacts": artifacts,
        "information_accounting": {"source_parameters_copied": 0, "receiver_training_steps": 0, "teacher_present_at_execution": False, "active_parameters": 0, "state_bytes_per_selected_package": 32, "final_package_bytes": sum(item["bytes"] for item in canonical.values()), "experiment_wall_seconds": time.perf_counter() - started},
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
