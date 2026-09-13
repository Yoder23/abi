"""Minimum-description teacher-target selection for the R31 packages."""

from __future__ import annotations

import argparse
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import json
from pathlib import Path

import psutil
import torch

from experiments.english_substrate_r30 import package_v4 as base
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once
from . import ladder


def run(source: Path, diagnosis: Path, prior: Path, output: Path):
    if output.exists():
        raise RuntimeError(f"immutable R31 MDL output exists: {output}")
    prior_doc = json.loads(prior.read_text(encoding="utf-8"))
    if prior_doc.get("verdict") != "FAIL_PUBLIC_SUFFICIENCY_LADDER" or [item["metrics"]["functional"] for item in prior_doc.get("levels", [])] != [116, 119]:
        raise RuntimeError("R31 ladder negative prerequisite changed")
    receipt_path = source / "receipt.json"
    rows_path = source / "accepted_rows.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    diagnosed = json.loads((diagnosis / "result.json").read_text(encoding="utf-8"))
    if receipt.get("verdict") != "PASS_NORMALIZED_SOURCE" or diagnosed.get("verdict") != "PASS_PUBLIC_NEURAL_DIAGNOSIS":
        raise RuntimeError("R31 MDL prerequisites changed")
    mapping = {opaque: group["name"] for group in diagnosed["groups"] for opaque in group["ids"]}
    rows = base._jsonl(rows_path)
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "evaluation"]
    api = base._layercake(Path(__file__).resolve().parents[2])
    prepared = ladder._prepare(train_rows, mapping, api["LosslessLexemePointerTokenizer"])
    tokenizer_type = api["LosslessLexemePointerTokenizer"]
    prepared.sort(key=lambda row: (row["cluster"], len(tokenizer_type.split(row["response"])), len(row["response"].encode()), row["record_id"]))
    router = base.CompactRouter.fit([(row["instruction"], mapping[base._instruction_id(row["instruction"])]) for row in train_rows])
    output.mkdir(parents=True)
    private = Ed25519PrivateKey.from_private_bytes(base.SIGNING_SEED)
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    signer = api["key_id"](public_pem)
    torch.cuda.reset_peak_memory_stats()
    result_level = ladder._level(api, prepared, eval_rows, mapping, router, 24, output, sha256_file(rows_path), public_pem, private_pem, signer)
    selected = []
    for cluster in sorted(set(mapping.values())):
        values = [row for row in prepared if row["cluster"] == cluster][:24]
        selected.append({"cluster": cluster, "rows": len(values), "lexeme_count": sum(len(tokenizer_type.split(row["response"])) for row in values), "utf8_bytes": sum(len(row["response"].encode()) for row in values), "record_ids": [row["record_id"] for row in values]})
    passed = result_level["verdict"] == "PASS_LEVEL"
    result = {"format": "abi-r31-minimum-description-selection/2", "verdict": "PASS_PUBLIC_MDL_SELECTION" if passed else "FAIL_PUBLIC_MDL_SELECTION", "prior_result_sha256": sha256_file(prior), "source_receipt_sha256": sha256_file(receipt_path), "source_rows_sha256": sha256_file(rows_path), "diagnosis_sha256": sha256_file(diagnosis / "result.json"), "selection": {"order": ["lossless_lexeme_count", "utf8_bytes", "record_id"], "selected": selected}, "level": result_level, "information_accounting": {"teacher_calls": 0, "teacher_present_during_training": False, "teacher_present_during_execution": False, "source_parameters_copied": 0, "logits_stored": 0, "hidden_activations_stored": 0, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(psutil.Process().memory_info().rss)}, "claim_ceiling": "PUBLIC_DISCLOSED_MDL_SELECTION_NOT_GENERAL_ENGLISH_OR_ABI_SUPERIORITY", "full_abi_moonshot": "OPEN"}
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.prior.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
