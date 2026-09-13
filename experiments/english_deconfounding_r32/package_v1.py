"""Train and rescreen one R32 counterbalanced package set."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30 import package_v7 as v7
from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31 import ladder
from experiments.english_sufficiency_r31.cascade_v3 import _runtime_score
from experiments.english_sufficiency_r31.hidden_v4 import TASK_TO_CLUSTER
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once


EXPECTED_R31_ROWS_SHA256 = "5ff7691d67b0e8271439acfbfe25540c7d42c06918b58b9257873c6f7ca9f05d"
EXPECTED_R31_FAILURE_SHA256 = "5e447168c885649f710480d510e7b99785ce1c802386572dd53d0dcc12cc180f"
ROWS_PER_CLUSTER = 66


def _package(api, model, tokenizer, cluster, output, source_hash, public_pem, private_pem, signer, training):
    artifact = api["build_token_plan_artifact"](
        model,
        tokenizer,
        domain_id=f"english-core-{cluster}",
        training={"source_rows_sha256": source_hash, "cluster": cluster, **training},
    )
    state = artifact["state_dict"]
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id=f"abi-r32-counterbalanced-{cluster}",
        name=f"ABI R32 counterbalanced {cluster}",
        description="R32 counterbalanced supplied-content English capability",
        version="0.32.0-v1",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=base.ABI_VERSION,
        abi_hash=base.ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "mode": "direct_selected_portable_decoder"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "one-counterbalanced-capability"},
        architecture=api["portable_token_plan_manifest_architecture"](artifact["spec"]),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda", "cpu"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "incremental"]},
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](state),
        package_hash="",
        training_data_provenance={
            "source_model": "Qwen/Qwen2-7B-Instruct",
            "source_revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
            "source_rows_sha256": source_hash,
            "source_parameters_copied": 0,
            "teacher_at_training": False,
            "teacher_at_inference": False,
            "receiver_training_steps": training["steps"],
        },
        evaluation_evidence={"status": "UNEVALUATED_R32"},
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
        "cluster": cluster,
        "cake_id": loaded.manifest.cake_id,
        "path": destination,
        "relative_path": destination.relative_to(output).as_posix(),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "parameters": model.parameter_count(),
        "training": training,
    }


def run(r31_rows: Path, r32_source: Path, prior_failure: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R32 package result exists: {output}")
    if sha256_file(r31_rows) != EXPECTED_R31_ROWS_SHA256 or sha256_file(prior_failure) != EXPECTED_R31_FAILURE_SHA256:
        raise RuntimeError("R32 frozen R31 prerequisite changed")
    receipt_path = r32_source / "receipt.json"
    new_rows_path = r32_source / "accepted_rows.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("verdict") != "PASS_COUNTERBALANCED_SOURCE":
        raise RuntimeError("R32 counterbalanced source did not pass")
    if receipt["artifacts"]["accepted_rows"]["sha256"] != sha256_file(new_rows_path):
        raise RuntimeError("R32 accepted rows changed")
    old_rows = [row for row in base._jsonl(r31_rows) if row["split"] == "train"]
    new_rows = base._jsonl(new_rows_path)
    combined = old_rows + new_rows
    if len(old_rows) != 576 or len(new_rows) != 216 or len(combined) != 792:
        raise RuntimeError("R32 combined training cardinality changed")
    if any(sum(row["oracle_task"] == task for row in combined) != ROWS_PER_CLUSTER for task in TASKS):
        raise RuntimeError("R32 per-contract training quota changed")
    source_hash = evidence_hash({"r31_rows": sha256_file(r31_rows), "r32_rows": sha256_file(new_rows_path)})
    api = base._layercake(Path(__file__).resolve().parents[2])
    mapping = {
        base._instruction_id(instruction): TASK_TO_CLUSTER[task]
        for task in TASKS
        for instruction in __import__("experiments.english_substrate_r30.protocol", fromlist=["INSTRUCTIONS"]).INSTRUCTIONS[task]
    }
    prepared = ladder._prepare(combined, mapping, api["LosslessLexemePointerTokenizer"])
    output.mkdir(parents=True)
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
        model, tokenizer, training = ladder._train(api, selected, ROWS_PER_CLUSTER, 32_000 + position)
        packages.append(_package(api, model, tokenizer, cluster, output, source_hash, public_pem, private_pem, signer, training))
        print(json.dumps({"trained_packages": position, "task": task, "seconds": time.perf_counter() - started}), flush=True)
        del model
        gc.collect()

    fixture_path = prior_failure.parent / "fixture.jsonl"
    fixture = base._jsonl(fixture_path)
    if len(fixture) != 144:
        raise RuntimeError("R32 rescreen fixture changed")
    observations = []
    with tempfile.TemporaryDirectory(prefix="r32-host-") as raw:
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
                raise RuntimeError("R32 package install hash mismatch")
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

    rows_path = output / "evaluation.jsonl"
    write_jsonl_once(rows_path, observations)
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
        "format": "abi-r32-counterbalanced-package/1",
        "verdict": "PASS_COUNTERBALANCED_DEVELOPMENT" if passed else "FAIL_COUNTERBALANCED_DEVELOPMENT",
        "inputs": {
            "r31_rows_sha256": sha256_file(r31_rows),
            "r32_receipt_sha256": sha256_file(receipt_path),
            "r32_rows_sha256": sha256_file(new_rows_path),
            "r31_failure_sha256": sha256_file(prior_failure),
            "r31_fixture_sha256": sha256_file(fixture_path),
            "combined_source_identity": source_hash,
        },
        "metrics": {
            "evaluation_rows": len(observations),
            "functional": functional,
            "by_task": by_task,
            "route_exact": route_exact,
            "contract_exact": contract_exact,
            "removal_rejections": sum(item["rejected"] for item in removals),
        },
        "packages": [{key: value for key, value in item.items() if key != "path"} for item in packages],
        "training": {
            "r31_rows": len(old_rows),
            "counterbalanced_rows": len(new_rows),
            "total_rows": len(combined),
            "rows_per_contract": ROWS_PER_CLUSTER,
            "teacher_present": False,
            "source_parameters_copied": 0,
        },
        "information_accounting": {
            "installed_package_bytes": sum(item["bytes"] for item in packages),
            "maximum_active_package_parameters": max(item["parameters"] for item in packages),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cpu_rss_bytes": int(peak_rss),
        },
        "evaluation": {"path": rows_path.name, "sha256": sha256_file(rows_path), "bytes": rows_path.stat().st_size},
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
    parser.add_argument("--r32-source", type=Path, required=True)
    parser.add_argument("--prior-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.r31_rows.resolve(), args.r32_source.resolve(), args.prior_failure.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
