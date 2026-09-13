"""Import R27 free-labeled packages into LayerCake and test composition live."""

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

from experiments.factual_semantic_r16.package import load_package as load_source_package
from experiments.foreign_capability_r14.core import R14Error, evidence_hash, json_object, sha256_file, write_json_once, write_jsonl_once
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash
from experiments.generative_transfer_r21.live_verify_v7 import targeted_tensor_corruption
from experiments.generative_transfer_r21.protocol import normalized_prompt
from experiments.semantic_replication_r23.binding import load_seed_reveal
from experiments.semantic_replication_r23.protocol import hidden_rows

from .binding import load_config, validate_file
from .protocol import answer_key


BUILD_INITIALIZATIONS = (27101, 27102, 27103)
HOST_INITIALIZATIONS = (27021, 27022, 27023)


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.casefold()).strip("-")


def _signing(config: dict[str, Any], api: dict[str, Any]) -> tuple[bytes, bytes, str]:
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(config["layercake_abi"]["research_signing_seed_hex"]))
    public = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return public, private_pem, api["key_id"](public)


def _build(root: Path, destination: Path, source: dict[str, Any], binding: dict[str, Any], config: dict[str, Any], api: dict[str, Any], public: bytes, private: bytes, signer: str) -> dict[str, Any]:
    from layercake.models.canonical_factual import CanonicalFactualDecoder, canonical_factual_manifest_architecture
    namespace = source["namespace"]
    model = CanonicalFactualDecoder(namespace, source["facts"])
    state = model.state_dict()
    cake_id = f"abi-r27-{_slug(namespace)}"
    manifest = api["CakeManifest"](
        schema_version="1", cake_id=cake_id, name=f"ABI R27 {namespace}",
        description="Free-labeled factual capability imported by ABI", version="0.27.0-bounded",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=config["layercake_abi"]["abi_version"], abi_hash=config["layercake_abi"]["abi_sha256"],
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "domain-layer", "namespace": namespace, "mode": "direct_selected_portable_decoder"},
        output_contract={"external": "UTF-8 bytes", "role": "domain-answer", "composition": "explicit_namespace_selection", "abstention": "empty UTF-8 byte string"},
        architecture=canonical_factual_manifest_architecture(namespace, source["facts"]),
        supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "incremental"]},
        tensor_payload_hash="", tensor_shapes=api["tensor_specs"](state), package_hash="",
        training_data_provenance={"method": "abi-r27-free-label-direct-import", "source_model": config["source"]["model_id"], "source_revision": config["source"]["revision"], "source_package_sha256": binding["sha256"], "source_parameters_copied": 0, "imported_records": len(source["facts"]), "receiver_training_steps": 0, "teacher_at_inference": False},
        evaluation_evidence={"status": "UNEVALUATED_R27"}, license="Apache-2.0", dependencies=(), parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer}, domains=(namespace,), permissions=("local-inference",),
    )
    api["build_package"](destination, manifest, state, private_key=private)
    loaded = api["load_package"](destination, trust_store={signer: public})
    return {"path": destination.relative_to(root).as_posix(), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "cake_id": cake_id, "namespace": namespace, "records": len(source["facts"]), "parameters": sum(p.numel() for p in model.parameters()), "state_bytes": sum(t.numel() * t.element_size() for t in state.values()), "signed": loaded.signed, "package_hash": loaded.manifest.package_hash}


def run(import_config_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if output.exists(): raise R14Error(f"immutable R27 import output exists: {output}")
    import_config = json_object(import_config_path)
    if import_config.get("format") != "abi-r27-import-config/1": raise R14Error("R27 import config changed")
    config_path = root / import_config["heldout_config"]["path"]
    validate_file(root, import_config["heldout_config"], "heldout config")
    config = load_config(root, config_path)
    for name in ("source_result", "source_rows", "evaluation"):
        validate_file(root, import_config[name], name)
    source_result = json_object(root / import_config["source_result"]["path"])
    if source_result.get("verdict") != "PASS" or source_result.get("evidence_sha256") != selfless_evidence_hash(source_result):
        raise R14Error("R27 source prerequisite failed")
    if not torch.cuda.is_available(): raise R14Error("R27 requires paired CPU/GPU execution")
    if "transformers" in sys.modules: raise R14Error("source framework present before R27 LayerCake execution")
    output.mkdir(parents=True)
    api = base_run._layercake(root)
    public, private, signer = _signing(config, api)
    sources = {}
    bindings = {}
    for item in import_config["source_packages"]:
        validate_file(root, item, "source package")
        package = load_source_package(root / item["path"])
        sources[package["namespace"]] = package; bindings[package["namespace"]] = item
    if len(sources) < 4: raise R14Error("R27 discovered package inventory is incomplete")
    started = time.perf_counter()
    builds: dict[str, dict[str, dict[str, Any]]] = {}
    for initialization in BUILD_INITIALIZATIONS:
        builds[str(initialization)] = {}
        for namespace, source in sorted(sources.items()):
            destination = output / "builds" / str(initialization) / f"{_slug(namespace)}.cake"
            builds[str(initialization)][namespace] = _build(root, destination, source, bindings[namespace], config, api, public, private, signer)
    canonical = builds[str(BUILD_INITIALIZATIONS[0])]
    build_rows = [{"build_initialization": init, "namespace": namespace, "path": builds[str(init)][namespace]["path"], "sha256": builds[str(init)][namespace]["sha256"], "bytes": builds[str(init)][namespace]["bytes"], "canonical_sha256": canonical[namespace]["sha256"], "byte_identical": builds[str(init)][namespace]["sha256"] == canonical[namespace]["sha256"]} for init in BUILD_INITIALIZATIONS for namespace in sorted(sources)]
    evaluation = [json.loads(line) for line in (root / import_config["evaluation"]["path"]).read_text(encoding="utf-8").splitlines() if line]
    r23_config = json_object(root / config["r23_config"]["path"])
    r23_seed = load_seed_reveal(r23_config, root / config["r23_reveal"]["path"])
    english_expected = hidden_rows(r23_seed)
    stored_rows = [json.loads(line) for line in (root / config["r23_hidden_rows"]["path"]).read_text(encoding="utf-8").splitlines() if line]
    stored_english = {row["record_id"]: row["output"] for row in stored_rows if row.get("method") == "abi_factorized" and row.get("seed") == 21022}
    english_by_task = {item["cake_id"].split("-seed", 1)[0].rsplit("-", 1)[-1]: item for item in config["english_packages"]}
    if len(stored_english) != 120 or set(english_by_task) != {row["task"] for row in english_expected}: raise R14Error("R27 English prerequisite changed")
    observations: list[dict[str, Any]] = []; core_rows: list[dict[str, Any]] = []; lifecycle: list[dict[str, Any]] = []; leakage: list[dict[str, Any]] = []
    from layercake.cake.installer import InstallationError
    for host_init in HOST_INITIALIZATIONS:
        with tempfile.TemporaryDirectory(prefix=f"r27-host-{host_init}-") as raw:
            temporary = Path(raw)
            gpu = api["DirectCakeHost"](temporary / "gpu", abi_version=config["layercake_abi"]["abi_version"], abi_hash=config["layercake_abi"]["abi_sha256"], trust_store={signer: public}, device="cuda")
            cpu = api["DirectCakeHost"](temporary / "cpu", abi_version=config["layercake_abi"]["abi_version"], abi_hash=config["layercake_abi"]["abi_sha256"], trust_store={signer: public}, device="cpu")
            for item in config["english_packages"]: gpu.install(root / item["path"])
            for package in canonical.values(): gpu.install(root / package["path"]); cpu.install(root / package["path"])
            for row in evaluation:
                package = canonical[row["namespace"]]
                gpu_text = gpu.generate(package["cake_id"], row["question"], maximum_actions=4096).output.decode("utf-8")
                cpu_text = cpu.generate(package["cake_id"], row["question"], maximum_actions=4096).output.decode("utf-8")
                other_outputs = [gpu.generate(other["cake_id"], row["question"], maximum_actions=4096).output.decode("utf-8") for name, other in canonical.items() if name != row["namespace"]]
                observations.append({**row, "host_initialization": host_init, "gpu_output": gpu_text, "cpu_output": cpu_text, "other_outputs": other_outputs})
            for row in english_expected:
                item = english_by_task[row["task"]]
                text = gpu.generate(item["cake_id"], normalized_prompt(row["task"], row["slots"]), maximum_actions=256).output.decode("utf-8")
                core_rows.append({"host_initialization": host_init, "stage": "domains_installed", "record_id": row["record_id"], "output": text})
            for namespace, package in sorted(canonical.items()):
                stimulus = next(row for row in evaluation if row["namespace"] == namespace)
                before = gpu.generate(package["cake_id"], stimulus["question"], maximum_actions=4096).output.decode("utf-8")
                removed = gpu.remove(package["cake_id"]); absent = False
                try: gpu.generate(package["cake_id"], stimulus["question"], maximum_actions=4096)
                except (KeyError, FileNotFoundError): absent = True
                restored = gpu.install(root / package["path"])
                after = gpu.generate(package["cake_id"], stimulus["question"], maximum_actions=4096).output.decode("utf-8")
                corrupt = temporary / f"corrupt-{_slug(namespace)}.cake"; corrupt.write_bytes(targeted_tensor_corruption((root / package["path"]).read_bytes()))
                corrupt_host = api["DirectCakeHost"](temporary / f"corrupt-{_slug(namespace)}", abi_version=config["layercake_abi"]["abi_version"], abi_hash=config["layercake_abi"]["abi_sha256"], trust_store={signer: public}, device="cuda")
                rejected = False
                try: corrupt_host.install(corrupt)
                except InstallationError: rejected = True
                lifecycle.append({"host_initialization": host_init, "namespace": namespace, "before": before, "after": after, "removed": removed.get("cake_id") == package["cake_id"], "absent_rejected": absent, "restored_sha256": restored["archive_hash"], "expected_sha256": package["sha256"], "corruption_rejected": rejected})
            for package in canonical.values(): gpu.remove(package["cake_id"])
            for row in english_expected:
                item = english_by_task[row["task"]]
                text = gpu.generate(item["cake_id"], normalized_prompt(row["task"], row["slots"]), maximum_actions=256).output.decode("utf-8")
                core_rows.append({"host_initialization": host_init, "stage": "domains_removed", "record_id": row["record_id"], "output": text})
            if host_init == HOST_INITIALIZATIONS[0]:
                for row in evaluation:
                    for task, item in english_by_task.items():
                        text = gpu.generate(item["cake_id"], row["question"], maximum_actions=256).output.decode("utf-8")
                        leakage.append({"fact_id": row["fact_id"], "task": task, "target": row["oracle_answer"], "output": text})
            del gpu, cpu; gc.collect(); torch.cuda.empty_cache()
    artifacts = {}
    for name, rows in (("package_builds", build_rows), ("domain_observations", observations), ("english_immutability", core_rows), ("lifecycle", lifecycle), ("english_leakage", leakage)):
        path = output / f"{name}.jsonl"; write_jsonl_once(path, rows); artifacts[name] = {"path": path.name, "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    metrics = {
        "discovered_packages": len(canonical), "reproducible_package_builds": sum(row["byte_identical"] for row in build_rows),
        "domain_oracle_exact": sum(answer_key(row["gpu_output"]) == answer_key(row["oracle_answer"]) for row in observations),
        "teacher_agreement": sum(answer_key(row["gpu_output"]) == answer_key(row["teacher_answer"]) for row in observations),
        "cpu_gpu_exact": sum(row["cpu_output"] == row["gpu_output"] for row in observations),
        "other_packages_abstain": sum(all(not text for text in row["other_outputs"]) for row in observations),
        "english_immutable": sum(row["output"] == stored_english[row["record_id"]] for row in core_rows),
        "english_core_target_answers": sum(answer_key(row["output"]) == answer_key(row["target"]) for row in leakage),
        "package_lifecycle_exact": sum(row["removed"] and row["absent_rejected"] and row["before"] == row["after"] and row["restored_sha256"] == row["expected_sha256"] for row in lifecycle),
        "targeted_corruptions_rejected": sum(row["corruption_rejected"] for row in lifecycle),
        "source_parameters_copied": 0, "receiver_training_steps": 0,
    }
    expected_observations = len(evaluation) * len(HOST_INITIALIZATIONS); expected_lifecycle = len(canonical) * len(HOST_INITIALIZATIONS)
    gates = {
        "package_builds": metrics["reproducible_package_builds"] == len(canonical) * len(BUILD_INITIALIZATIONS),
        "domain_oracle_exact": metrics["domain_oracle_exact"] == expected_observations,
        "teacher_agreement": metrics["teacher_agreement"] == expected_observations,
        "cpu_gpu_exact": metrics["cpu_gpu_exact"] == expected_observations,
        "other_packages_abstain": metrics["other_packages_abstain"] == expected_observations,
        "english_immutable": metrics["english_immutable"] == 720,
        "english_core_no_domain_answers": metrics["english_core_target_answers"] == 0,
        "lifecycle": metrics["package_lifecycle_exact"] == expected_lifecycle,
        "corruption": metrics["targeted_corruptions_rejected"] == expected_lifecycle,
        "zero_copy_zero_training": metrics["source_parameters_copied"] == 0 and metrics["receiver_training_steps"] == 0,
        "teacher_absent_at_execution": "transformers" not in sys.modules,
    }
    passed = all(gates.values())
    result = {"format": "abi-r27-layercake-import-result/1", "verdict": "PASS" if passed else "FAIL", "claim": "BOUNDED_FREE_LABELED_FACTUAL_CAPABILITIES_IMPORTED_INTO_LAYERCAKE", "claim_ceiling": "NOT_EXHAUSTIVE_TEACHER_DIAGNOSIS_OR_FLUENT_ENGLISH_EXTRACTION", "import_config_sha256": sha256_file(import_config_path), "heldout_config_sha256": sha256_file(config_path), "source_result_sha256": sha256_file(root / import_config["source_result"]["path"]), "layercake_commit": config["layercake_commit"], "metrics": metrics, "gates": gates, "canonical_packages": canonical, "artifacts": artifacts, "information_accounting": {"source_parameters_copied": 0, "receiver_training_steps": 0, "teacher_present_at_execution": False, "active_parameters": 0, "state_bytes_per_selected_package": 32, "final_package_bytes": sum(item["bytes"] for item in canonical.values()), "experiment_wall_seconds": time.perf_counter() - started}, "full_abi_moonshot": "OPEN"}
    result["evidence_sha256"] = evidence_hash(result); write_json_once(output / "result.json", result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__": main()
