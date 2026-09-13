"""Repackage exact R30 v5 tensors for direct-host conformance and evaluate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import tempfile
import time
from pathlib import Path

import psutil
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once
from . import package_v4 as base
from .protocol import TASKS


def run(source: Path, diagnosis: Path, v5: Path, failure: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"immutable R30 v6 run exists: {output}")
    failed = json.loads(failure.read_text(encoding="utf-8"))
    if failed.get("verdict") != "FAIL_DIRECT_HOST_MANIFEST_CONFORMANCE" or failed.get("evaluation_rows_completed") != 0:
        raise RuntimeError("R30 v5 failure binding changed")
    inventory = {Path(row["path"]).name: row for row in failed["package_inventory"]}
    if len(inventory) != 12:
        raise RuntimeError("R30 v5 inventory incomplete")
    rows_path = source / "source_rows.jsonl"
    diagnosis_path = diagnosis / "result.json"
    diagnosed = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if diagnosed.get("verdict") != "PASS_PUBLIC_NEURAL_DIAGNOSIS":
        raise RuntimeError("R30 diagnosis no longer passes")
    mapping = {opaque: group["name"] for group in diagnosed["groups"] for opaque in group["ids"]}
    all_rows = base._jsonl(rows_path)
    train_rows = [row for row in all_rows if row["split"] == "train"]
    eval_rows = [row for row in all_rows if row["split"] == "evaluation"]
    router = base.CompactRouter.fit([(row["instruction"], mapping[base._instruction_id(row["instruction"])]) for row in train_rows])
    api = base._layercake(Path(__file__).resolve().parents[2])
    private = Ed25519PrivateKey.from_private_bytes(base.SIGNING_SEED)
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    signer = api["key_id"](public_pem)
    output.mkdir(parents=True)
    rebuilt = []
    for path in sorted((v5 / "packages").glob("*.cake")):
        bound = inventory.get(path.name)
        if bound is None or bound["sha256"] != sha256_file(path) or bound["bytes"] != path.stat().st_size:
            raise RuntimeError("R30 v5 package hash/size changed")
        package = api["load_package"](path, trust_store={signer: public_pem})
        if package.manifest.input_contract != {"external": "UTF-8 bytes", "role": "english-core"}:
            raise RuntimeError("R30 v5 manifest failure is not the registered defect")
        corrected = replace(package.manifest, input_contract={"external": "UTF-8 bytes", "role": "english-core", "mode": "direct_selected_portable_decoder"}, tensor_payload_hash="", package_hash="")
        destination = output / "packages" / path.name
        api["build_package"](destination, corrected, package.tensors, private_key=private_pem)
        loaded = api["load_package"](destination, trust_store={signer: public_pem})
        if loaded.manifest.tensor_payload_hash != package.manifest.tensor_payload_hash:
            raise RuntimeError("R30 v6 tensor payload changed")
        rebuilt.append({"cluster": path.stem, "cake_id": loaded.manifest.cake_id, "path": destination, "relative_path": destination.relative_to(output).as_posix(), "source_sha256": bound["sha256"], "sha256": sha256_file(destination), "bytes": destination.stat().st_size, "tensor_payload_hash": loaded.manifest.tensor_payload_hash, "parameters": sum(t.numel() for t in package.tensors.values())})
    if len(rebuilt) != 12:
        raise RuntimeError("R30 v6 did not rebuild twelve packages")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    observations = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    with tempfile.TemporaryDirectory(prefix="r30-v6-host-") as raw:
        host = api["DirectCakeHost"](Path(raw) / "registry", abi_version=base.ABI_VERSION, abi_hash=base.ABI_SHA256, trust_store={signer: public_pem}, device="cuda")
        by_cluster = {}
        for package in rebuilt:
            installed = host.install(package["path"])
            if installed["archive_hash"] != package["sha256"]:
                raise RuntimeError("R30 v6 installed archive changed")
            by_cluster[package["cluster"]] = package
        for index, row in enumerate(eval_rows, 1):
            predicted = router.predict(row["instruction"])
            expected = mapping[base._instruction_id(row["instruction"])]
            package = by_cluster[predicted]
            before = time.perf_counter()
            generated = host.generate(package["cake_id"], row["prompt"], maximum_actions=256)
            torch.cuda.synchronize()
            text = generated.output.decode("utf-8", errors="strict")
            scored = base._task_score(row, text, row["teacher_output"])
            observations.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "expected_cluster": expected, "predicted_cluster": predicted, "route_exact": predicted == expected, "teacher_output": row["teacher_output"], "output": text, "score": scored, "seconds": time.perf_counter() - before})
            peak_rss = max(peak_rss, process.memory_info().rss)
            if index == 1 or index % 24 == 0:
                print(json.dumps({"evaluated": index, "functional": sum(item["score"]["functional"] for item in observations), "seconds": time.perf_counter() - started}), flush=True)
        removals = []
        for package in rebuilt:
            host.remove(package["cake_id"])
            rejected = False
            try:
                host.generate(package["cake_id"], eval_rows[0]["prompt"], maximum_actions=4)
            except (KeyError, ValueError, FileNotFoundError):
                rejected = True
            removals.append({"cake_id": package["cake_id"], "removed_rejected": rejected})
    rows_out = output / "evaluation.jsonl"
    write_jsonl_once(rows_out, observations)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    route_exact = sum(row["route_exact"] for row in observations)
    passed = route_exact == 144 and functional >= 132 and all(item["functional"] >= 10 for item in by_task.values()) and by_task["abstention"]["functional"] == 12 and all(row["removed_rejected"] for row in removals)
    result = {
        "format": "abi-r30-source-clustered-layercake-packages/6", "verdict": "PASS_PUBLIC_PACKAGE_PILOT" if passed else "FAIL_PUBLIC_PACKAGE_PILOT",
        "v5_failure_sha256": sha256_file(failure), "source_rows_sha256": sha256_file(rows_path), "diagnosis_sha256": sha256_file(diagnosis_path),
        "metrics": {"packages": len(rebuilt), "evaluation_rows": len(observations), "route_exact": route_exact, "functional": functional, "by_task": by_task, "removal_rejections": sum(row["removed_rejected"] for row in removals)},
        "packages": [{key: value for key, value in row.items() if key != "path"} for row in rebuilt], "router": router.document,
        "information_accounting": {"teacher_present_during_repair": False, "teacher_present_during_execution": False, "training_steps_in_repair": 0, "source_parameters_copied": 0, "tensor_payloads_changed": 0, "deployed_parameters": sum(row["parameters"] for row in rebuilt), "maximum_active_parameters": max(row["parameters"] for row in rebuilt), "package_bytes": sum(row["bytes"] for row in rebuilt), "evaluation_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)},
        "artifacts": {"evaluation": {"path": rows_out.name, "sha256": sha256_file(rows_out), "bytes": rows_out.stat().st_size}}, "claim_ceiling": "PUBLIC_DISCLOSED_PARAPHRASE_PILOT_NOT_GENERAL_ENGLISH", "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--v5", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.v5.resolve(), args.failure.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
