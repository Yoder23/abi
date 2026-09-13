"""R30 package pilot with globally safe pointer eligibility."""

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

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once
from . import package_v4 as base
from .protocol import TASKS


def _prepare(rows: list[dict[str, Any]], mapping: dict[str, str], tokenizer_type: Any) -> tuple[list[dict[str, Any]], int]:
    staged = []
    corrections = 0
    proposed: set[bytes] = set()
    for row in rows:
        response, corrected = base._validated_response(row)
        corrections += int(corrected)
        source = tokenizer_type.split(str(row["prompt"]))
        target = tokenizer_type.split(response)
        nonce = str(row["nonce"]).encode()
        if nonce in target:
            proposed.add(nonce)
        proposed.update(piece for piece in source if piece.isdigit() and piece in target)
        staged.append((row, response, source, target))
    eligible = set(proposed)
    for piece in list(eligible):
        for _, _, source, target in staged:
            if piece in target and source.count(piece) != 1:
                eligible.remove(piece)
                break
    prepared = []
    for row, response, source, target in staged:
        copy = [piece.decode() for piece in sorted(eligible) if piece in target and source.count(piece) == 1]
        if source.count(b"INSTRUCTION") != 1 or b"INSTRUCTION" in target:
            raise RuntimeError("R30 source-only sentinel contract changed")
        copy.append("INSTRUCTION")
        prepared.append({"record_id": row["record_id"], "prompt": row["prompt"], "response": response, "cluster": mapping[base._instruction_id(row["instruction"])], "copy_lexemes": copy})
    return prepared, corrections


def run(source: Path, diagnosis: Path, failure: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"immutable R30 v5 run exists: {output}")
    failed = json.loads(failure.read_text(encoding="utf-8"))
    if failed.get("verdict") != "FAIL_TOKENIZER_COPY_ELIGIBILITY" or failed.get("layercake_evaluation_started") is not False:
        raise RuntimeError("R30 v4 failure binding changed")
    rows_path = source / "source_rows.jsonl"
    receipt_path = source / "receipt.json"
    diagnosis_path = diagnosis / "result.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    diagnosed = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if receipt.get("verdict") != "FAIL_SOURCE" or diagnosed.get("verdict") != "PASS_PUBLIC_NEURAL_DIAGNOSIS":
        raise RuntimeError("R30 prior evidence boundary changed")
    if receipt["artifacts"]["source_rows"]["sha256"] != sha256_file(rows_path):
        raise RuntimeError("R30 source rows changed")
    mapping = {opaque: group["name"] for group in diagnosed["groups"] for opaque in group["ids"]}
    all_rows = base._jsonl(rows_path)
    train_rows = [row for row in all_rows if row["split"] == "train"]
    eval_rows = [row for row in all_rows if row["split"] == "evaluation"]
    api = base._layercake(Path(__file__).resolve().parents[2])
    prepared, corrections = _prepare(train_rows, mapping, api["LosslessLexemePointerTokenizer"])
    router = base.CompactRouter.fit([(row["instruction"], mapping[base._instruction_id(row["instruction"])]) for row in train_rows])
    output.mkdir(parents=True)
    private = Ed25519PrivateKey.from_private_bytes(base.SIGNING_SEED)
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    signer = api["key_id"](public_pem)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    packages = []
    for position, cluster in enumerate(sorted(set(mapping.values())), 1):
        selected = [row for row in prepared if row["cluster"] == cluster]
        model, tokenizer, training = base._train(api, selected, 30_400 + position)
        packages.append(base._package(api, model, tokenizer, cluster, output, sha256_file(rows_path), public_pem, private_pem, signer, training))
        print(json.dumps({"trained_packages": position, "cluster": cluster, "seconds": time.perf_counter() - started}), flush=True)
        del model
        gc.collect()
    observations = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    with tempfile.TemporaryDirectory(prefix="r30-v5-host-") as raw:
        host = api["DirectCakeHost"](Path(raw) / "registry", abi_version=base.ABI_VERSION, abi_hash=base.ABI_SHA256, trust_store={signer: public_pem}, device="cuda")
        by_cluster = {}
        for package in packages:
            installed = host.install(package["path"])
            if installed["archive_hash"] != package["sha256"]:
                raise RuntimeError("R30 package install identity changed")
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
        for package in packages:
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
        "format": "abi-r30-source-clustered-layercake-packages/5", "verdict": "PASS_PUBLIC_PACKAGE_PILOT" if passed else "FAIL_PUBLIC_PACKAGE_PILOT",
        "v4_failure_sha256": sha256_file(failure), "source_receipt_sha256": sha256_file(receipt_path), "source_rows_sha256": sha256_file(rows_path), "diagnosis_sha256": sha256_file(diagnosis_path),
        "metrics": {"packages": len(packages), "evaluation_rows": len(observations), "route_exact": route_exact, "functional": functional, "by_task": by_task, "validator_corrections_train": corrections, "removal_rejections": sum(row["removed_rejected"] for row in removals)},
        "packages": [{key: value for key, value in row.items() if key != "path"} for row in packages], "router": router.document,
        "information_accounting": {"teacher_present_during_training": False, "teacher_present_during_execution": False, "source_parameters_copied": 0, "logits_stored": 0, "hidden_activations_in_packages": 0, "training_rows": len(train_rows), "training_steps_total": sum(row["training"]["steps"] for row in packages), "deployed_parameters": sum(row["parameters"] for row in packages), "maximum_active_parameters": max(row["parameters"] for row in packages), "package_bytes": sum(row["bytes"] for row in packages), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)},
        "artifacts": {"evaluation": {"path": rows_out.name, "sha256": sha256_file(rows_out), "bytes": rows_out.stat().st_size}}, "claim_ceiling": "PUBLIC_DISCLOSED_PARAPHRASE_PILOT_NOT_GENERAL_ENGLISH", "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.failure.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
