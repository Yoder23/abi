"""R30 compute-matched shared-English sequence-distillation control."""

from __future__ import annotations

import argparse
import gc
import json
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

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once
from . import package_v4 as base
from . import package_v7 as v7
from .protocol import TASKS


TRAIN_STEPS = 4_800
BATCH_SIZE = 32
SEED = 30_501


def _prompt(prompt: str, cluster: str) -> str:
    raw = cluster.removeprefix("capability-")
    marker = "Lcgroup" + raw.translate(str.maketrans("0123456789abcdef", "ghijklmnopabcdef"))
    if not marker.isalpha() or marker.casefold() in prompt.casefold():
        raise RuntimeError("R30 capability marker is not collision-free")
    return f"{marker}\n{prompt}"


def _prepare(rows: list[dict[str, Any]], mapping: dict[str, str], tokenizer_type: Any) -> tuple[list[dict[str, Any]], int]:
    prepared, corrections = v7._prepare(rows, mapping, tokenizer_type)
    for row in prepared:
        row["prompt"] = _prompt(row["prompt"], row["cluster"])
    return prepared, corrections


def _train(api: dict[str, Any], rows: list[dict[str, Any]]):
    tokenizer_type = api["LosslessLexemePointerTokenizer"]
    fixed = sorted({piece for row in rows for field in ("prompt", "response") for piece in tokenizer_type.split(row[field])})
    tokenizer = tokenizer_type(fixed, format_version="layercake-lossless-lexeme-pointer/2")
    encoded = base._encode(rows, tokenizer)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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
        maximum_target_actions=256,
    ).to(device).bind_tokenizer(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    order = list(range(len(rows)))
    rng = random.Random(SEED)
    cursor = len(order)
    started = time.perf_counter()
    history = []
    for step in range(1, TRAIN_STEPS + 1):
        if cursor + BATCH_SIZE > len(order):
            rng.shuffle(order)
            cursor = 0
        indexes = order[cursor:cursor + BATCH_SIZE]
        cursor += BATCH_SIZE
        source, target = base._batch(encoded, indexes, device)
        result = model(source, target)
        mask = target.ge(0)
        loss = F.nll_loss(result["log_probs"][mask], target[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {1, 1_200, 2_400, 3_600, 4_800}:
            prediction = result["log_probs"][mask].argmax(-1)
            history.append({
                "step": step,
                "loss": float(loss),
                "accuracy": float(prediction.eq(target[mask]).float().mean()),
                "gradient_norm": float(gradient),
                "seconds": time.perf_counter() - started,
            })
            print(json.dumps(history[-1]), flush=True)
    torch.cuda.synchronize()
    training = {
        "parameters": model.parameter_count(),
        "fixed_vocabulary_size": tokenizer.vocab_size,
        "rows": len(rows),
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "training_examples": TRAIN_STEPS * BATCH_SIZE,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    return model.cpu().eval(), tokenizer, training


def _package(api, model, tokenizer, output, source_hash, public_pem, private_pem, signer, training):
    artifact = api["build_token_plan_artifact"](
        model,
        tokenizer,
        domain_id="english-core-shared",
        training={"source_rows_sha256": source_hash, "control": "matched-sequence-distillation", **training},
    )
    state = artifact["state_dict"]
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id="abi-r30-v8-shared-english-control",
        name="ABI R30 v8 shared English control",
        description="Compute-matched shared sequence-distillation control",
        version="0.30.0-v8-control",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=base.ABI_VERSION,
        abi_hash=base.ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "mode": "direct_selected_portable_decoder"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "shared-control"},
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
            "teacher_at_training": False,
            "teacher_at_inference": False,
            "receiver_training_steps": TRAIN_STEPS,
            "control": "matched-sequence-distillation",
        },
        evaluation_evidence={"status": "UNEVALUATED_R30_V8_CONTROL"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=("english-core-shared",),
        permissions=("local-inference",),
    )
    destination = output / "packages" / "shared-english-control.cake"
    api["build_package"](destination, manifest, state, private_key=private_pem)
    loaded = api["load_package"](destination, trust_store={signer: public_pem})
    return {
        "cake_id": loaded.manifest.cake_id,
        "path": destination,
        "relative_path": destination.relative_to(output).as_posix(),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        "parameters": model.parameter_count(),
        "training": training,
    }


def run(source: Path, diagnosis: Path, v7_result: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R30 v8 run exists: {output}")
    prior = json.loads(v7_result.read_text(encoding="utf-8"))
    if prior.get("verdict") != "FAIL_PUBLIC_GROUNDED_POINTER_PILOT" or prior.get("metrics", {}).get("functional") != 103:
        raise RuntimeError("R30 v7 failure binding changed")
    rows_path = source / "source_rows.jsonl"
    diagnosis_path = diagnosis / "result.json"
    diagnosed = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if diagnosed.get("verdict") != "PASS_PUBLIC_NEURAL_DIAGNOSIS":
        raise RuntimeError("R30 diagnosis prerequisite changed")
    mapping = {opaque: group["name"] for group in diagnosed["groups"] for opaque in group["ids"]}
    if len(mapping) != 36:
        raise RuntimeError("R30 diagnosis mapping incomplete")
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
    model, tokenizer, training = _train(api, prepared)
    package = _package(api, model, tokenizer, output, sha256_file(rows_path), public_pem, private_pem, signer, training)
    del model
    gc.collect()

    teacher_scores = [v7._score(row, row["teacher_output"], row["teacher_output"]) for row in eval_rows]
    observations = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    with tempfile.TemporaryDirectory(prefix="r30-v8-host-") as raw:
        host = api["DirectCakeHost"](
            Path(raw) / "registry",
            abi_version=base.ABI_VERSION,
            abi_hash=base.ABI_SHA256,
            trust_store={signer: public_pem},
            device="cuda",
        )
        installed = host.install(package["path"])
        if installed["archive_hash"] != package["sha256"]:
            raise RuntimeError("R30 v8 installed archive changed")
        for index, row in enumerate(eval_rows, 1):
            predicted = router.predict(row["instruction"])
            expected = mapping[base._instruction_id(row["instruction"])]
            prompt = _prompt(v7._normalized_prompt(row), predicted)
            generated = host.generate(package["cake_id"], prompt, maximum_actions=256)
            torch.cuda.synchronize()
            text = generated.output.decode("utf-8", errors="strict")
            scored = v7._score(row, text, row["teacher_output"])
            observations.append({
                "record_id": row["record_id"],
                "oracle_task": row["oracle_task"],
                "expected_cluster": expected,
                "predicted_cluster": predicted,
                "route_exact": predicted == expected,
                "teacher_output": row["teacher_output"],
                "output": text,
                "score": scored,
            })
            peak_rss = max(peak_rss, process.memory_info().rss)
            if index == 1 or index % 24 == 0:
                print(json.dumps({"evaluated": index, "functional": sum(item["score"]["functional"] for item in observations), "seconds": time.perf_counter() - started}), flush=True)
        host.remove(package["cake_id"])
        removal_rejected = False
        try:
            host.generate(package["cake_id"], _prompt(v7._normalized_prompt(eval_rows[0]), router.predict(eval_rows[0]["instruction"])), maximum_actions=4)
        except (KeyError, ValueError, FileNotFoundError):
            removal_rejected = True

    rows_out = output / "evaluation.jsonl"
    write_jsonl_once(rows_out, observations)
    by_task = {task: {"rows": 0, "functional": 0, "teacher_functional": 0} for task in TASKS}
    for row, teacher_score in zip(observations, teacher_scores):
        item = by_task[row["oracle_task"]]
        item["rows"] += 1
        item["functional"] += int(row["score"]["functional"])
        item["teacher_functional"] += int(teacher_score["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    teacher_functional = sum(item["functional"] for item in teacher_scores)
    route_exact = sum(row["route_exact"] for row in observations)
    passed = (
        route_exact == 144
        and functional >= 132
        and functional >= teacher_functional
        and all(item["functional"] >= 10 for item in by_task.values())
        and by_task["abstention"]["functional"] == 12
        and removal_rejected
    )
    result = {
        "format": "abi-r30-shared-english-control/8",
        "verdict": "PASS_PUBLIC_SHARED_CONTROL" if passed else "FAIL_PUBLIC_SHARED_CONTROL",
        "scientific_role": "MATCHED_SEQUENCE_DISTILLATION_CONTROL_NOT_ABI_PROMOTION",
        "v7_result_sha256": sha256_file(v7_result),
        "source_rows_sha256": sha256_file(rows_path),
        "diagnosis_sha256": sha256_file(diagnosis_path),
        "metrics": {
            "packages": 1,
            "evaluation_rows": len(observations),
            "route_exact": route_exact,
            "functional": functional,
            "teacher_functional": teacher_functional,
            "by_task": by_task,
            "validator_corrections_train": corrections,
            "removal_rejections": int(removal_rejected),
        },
        "package": {key: value for key, value in package.items() if key != "path"},
        "router": router.document,
        "matched_compute": {
            "v7_training_examples": 12 * 1_600 * 8,
            "v8_training_examples": training["training_examples"],
            "neural_widths_unchanged": True,
        },
        "information_accounting": {
            "teacher_present_during_training": False,
            "teacher_present_during_execution": False,
            "source_parameters_copied": 0,
            "logits_stored": 0,
            "hidden_activations_in_package": 0,
            "training_rows": len(train_rows),
            "training_steps_total": TRAIN_STEPS,
            "deployed_parameters": package["parameters"],
            "maximum_active_parameters": package["parameters"],
            "package_bytes": package["bytes"],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cpu_rss_bytes": int(peak_rss),
        },
        "artifacts": {"evaluation": {"path": rows_out.name, "sha256": sha256_file(rows_out), "bytes": rows_out.stat().st_size}},
        "claim_ceiling": "PUBLIC_DISCLOSED_SEQUENCE_DISTILLATION_CONTROL_NOT_GENERAL_ENGLISH_OR_ABI_SUPERIORITY",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--v7-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.v7_result.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
