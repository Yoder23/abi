"""Train the R31 24/48-row grounded-pointer sufficiency ladder."""

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

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30 import package_v7 as v7
from experiments.english_substrate_r30.protocol import TASKS
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once


LEVELS = (24, 48)
BATCH_SIZE = 24
EXPOSURES_PER_ROW = 533
MAXIMUM_ACTIONS = 384
ALNUM_RE = v7.ALNUM_RE


def _prepare(rows, mapping, tokenizer_type):
    prepared = []
    for row in rows:
        prompt = v7._normalized_prompt(row)
        response = str(row["teacher_output"])
        source = tokenizer_type.split(prompt)
        target = set(tokenizer_type.split(response))
        marker = source.index(b"SUPPLIED")
        copy = [piece.decode() for piece in source[marker:] if ALNUM_RE.fullmatch(piece) and piece in target and source.count(piece) == 1]
        copy = list(dict.fromkeys(copy))
        if not copy:
            if source.count(b"INSTRUCTION") != 1 or b"INSTRUCTION" in target:
                raise RuntimeError("R31 row lacks source-only sentinel")
            copy = ["INSTRUCTION"]
        prepared.append({"record_id": row["record_id"], "prompt": prompt, "response": response, "cluster": mapping[base._instruction_id(row["instruction"])], "copy_lexemes": copy})
    return prepared


def _encode(rows, tokenizer):
    encoded = []
    for row in rows:
        source, lexemes = tokenizer.encode_source(row["prompt"])
        target = tokenizer.encode_target(row["response"], copy_lexemes=row["copy_lexemes"], source_lexemes=lexemes)
        if len(source) > 128 or len(target) > MAXIMUM_ACTIONS:
            raise RuntimeError("R31 source/target boundary exceeded")
        if tokenizer.decode_actions(target, lexemes).decode() != row["response"]:
            raise RuntimeError("R31 target representation is not lossless")
        encoded.append((source, target))
    return encoded


def _train(api, rows, level, seed):
    tokenizer_type = api["LosslessLexemePointerTokenizer"]
    fixed = sorted({piece for row in rows for field in ("prompt", "response") for piece in tokenizer_type.split(row[field])})
    tokenizer = tokenizer_type(fixed, format_version="layercake-lossless-lexeme-pointer/2")
    encoded = _encode(rows, tokenizer)
    steps = (EXPOSURES_PER_ROW * len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = api["PortableTokenPlan"](fixed_vocab_size=tokenizer.vocab_size, model_width=64, attention_heads=4, encoder_layers=2, decoder_layers=2, feedforward_width=192, pointer_width=32, dropout=0.0, maximum_source_lexemes=128, maximum_target_actions=MAXIMUM_ACTIONS).to(device).bind_tokenizer(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    started = time.perf_counter()
    history = []
    for step in range(1, steps + 1):
        indexes = [order[rng.randrange(len(order))] for _ in range(BATCH_SIZE)]
        source, target = base._batch(encoded, indexes, device)
        result = model(source, target)
        mask = target.ge(0)
        loss = F.nll_loss(result["log_probs"][mask], target[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {1, steps // 2, steps}:
            prediction = result["log_probs"][mask].argmax(-1)
            history.append({"step": step, "loss": float(loss), "accuracy": float(prediction.eq(target[mask]).float().mean()), "gradient_norm": float(gradient), "seconds": time.perf_counter() - started})
    torch.cuda.synchronize()
    return model.cpu().eval(), tokenizer, {"level_rows_per_cluster": level, "rows": len(rows), "steps": steps, "batch_size": BATCH_SIZE, "example_exposures_per_row": steps * BATCH_SIZE / len(rows), "parameters": model.parameter_count(), "fixed_vocabulary_size": tokenizer.vocab_size, "history": history, "seconds": time.perf_counter() - started}


def _package(api, model, tokenizer, cluster, level_dir, source_hash, public_pem, private_pem, signer, training):
    artifact = api["build_token_plan_artifact"](model, tokenizer, domain_id=f"english-core-{cluster}", training={"source_rows_sha256": source_hash, "cluster": cluster, **training})
    state = artifact["state_dict"]
    manifest = api["CakeManifest"](
        schema_version="1", cake_id=f"abi-r31-l{training['level_rows_per_cluster']}-{cluster}", name=f"ABI R31 {cluster}", description="R31 validator-selected English sufficiency package", version=f"0.31.0-l{training['level_rows_per_cluster']}", publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer}, abi_version=base.ABI_VERSION, abi_hash=base.ABI_SHA256, cake_type="portable_decoder", input_contract={"external": "UTF-8 bytes", "role": "english-core", "mode": "direct_selected_portable_decoder"}, output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "one-source-discovered-capability"}, architecture=api["portable_token_plan_manifest_architecture"](artifact["spec"]), supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"), minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "incremental"]}, tensor_payload_hash="", tensor_shapes=api["tensor_specs"](state), package_hash="", training_data_provenance={"source_model": "Qwen/Qwen2-7B-Instruct", "source_revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c", "source_rows_sha256": source_hash, "source_parameters_copied": 0, "teacher_at_training": False, "teacher_at_inference": False, "receiver_training_steps": training["steps"]}, evaluation_evidence={"status": "UNEVALUATED_R31_PUBLIC"}, license="Apache-2.0", dependencies=(), parent_version=None, signature={"algorithm": "ed25519", "key_id": signer}, domains=(f"english-core-{cluster}",), permissions=("local-inference",)
    )
    path = level_dir / "packages" / f"{cluster}.cake"
    api["build_package"](path, manifest, state, private_key=private_pem)
    loaded = api["load_package"](path, trust_store={signer: public_pem})
    return {"cluster": cluster, "cake_id": loaded.manifest.cake_id, "path": path, "relative_path": path.relative_to(level_dir).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size, "parameters": model.parameter_count(), "training": training}


def _level(api, prepared, eval_rows, mapping, router, level, output, source_hash, public_pem, private_pem, signer):
    level_dir = output / f"level-{level}"
    level_dir.mkdir(parents=True)
    packages = []
    started = time.perf_counter()
    clusters = sorted(set(mapping.values()))
    for position, cluster in enumerate(clusters, 1):
        selected = [row for row in prepared if row["cluster"] == cluster][:level]
        if len(selected) != level:
            raise RuntimeError("R31 cluster training quota changed")
        model, tokenizer, training = _train(api, selected, level, 31_000 + level * 100 + position)
        packages.append(_package(api, model, tokenizer, cluster, level_dir, source_hash, public_pem, private_pem, signer, training))
        print(json.dumps({"level": level, "trained_packages": position, "seconds": time.perf_counter() - started}), flush=True)
        del model
        gc.collect()
    observations = []
    with tempfile.TemporaryDirectory(prefix=f"r31-l{level}-host-") as raw:
        host = api["DirectCakeHost"](Path(raw) / "registry", abi_version=base.ABI_VERSION, abi_hash=base.ABI_SHA256, trust_store={signer: public_pem}, device="cuda")
        by_cluster = {}
        for package in packages:
            host.install(package["path"])
            by_cluster[package["cluster"]] = package
        for index, row in enumerate(eval_rows, 1):
            predicted = router.predict(row["instruction"])
            expected = mapping[base._instruction_id(row["instruction"])]
            generated = host.generate(by_cluster[predicted]["cake_id"], v7._normalized_prompt(row), maximum_actions=MAXIMUM_ACTIONS)
            text = generated.output.decode("utf-8", errors="strict")
            observations.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "expected_cluster": expected, "predicted_cluster": predicted, "route_exact": predicted == expected, "teacher_output": row["teacher_output"], "output": text, "score": v7._score(row, text, row["teacher_output"])})
            if index % 36 == 0:
                print(json.dumps({"level": level, "evaluated": index, "functional": sum(item["score"]["functional"] for item in observations)}), flush=True)
        removals = []
        for package in packages:
            host.remove(package["cake_id"])
            rejected = False
            try:
                host.generate(package["cake_id"], v7._normalized_prompt(eval_rows[0]), maximum_actions=4)
            except (KeyError, ValueError, FileNotFoundError):
                rejected = True
            removals.append(rejected)
    rows_path = level_dir / "evaluation.jsonl"
    write_jsonl_once(rows_path, observations)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in observations:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in observations)
    route_exact = sum(row["route_exact"] for row in observations)
    passed = route_exact == 144 and functional >= 132 and all(item["functional"] >= 11 for item in by_task.values()) and by_task["abstention"]["functional"] == 12 and all(removals)
    return {"level": level, "verdict": "PASS_LEVEL" if passed else "FAIL_LEVEL", "metrics": {"functional": functional, "route_exact": route_exact, "by_task": by_task, "removal_rejections": sum(removals)}, "packages": [{key: value for key, value in item.items() if key != "path"} for item in packages], "evaluation": {"path": rows_path.relative_to(output).as_posix(), "sha256": sha256_file(rows_path), "bytes": rows_path.stat().st_size}, "elapsed_seconds": time.perf_counter() - started}


def run(source: Path, diagnosis: Path, output: Path):
    if output.exists():
        raise RuntimeError(f"immutable R31 ladder exists: {output}")
    receipt_path = source / "receipt.json"
    rows_path = source / "accepted_rows.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    diagnosed = json.loads((diagnosis / "result.json").read_text(encoding="utf-8"))
    if receipt.get("verdict") != "PASS_NORMALIZED_SOURCE" or diagnosed.get("verdict") != "PASS_PUBLIC_NEURAL_DIAGNOSIS":
        raise RuntimeError("R31 prerequisites changed")
    if receipt["artifacts"]["accepted_rows"]["sha256"] != sha256_file(rows_path):
        raise RuntimeError("R31 source rows changed")
    mapping = {opaque: group["name"] for group in diagnosed["groups"] for opaque in group["ids"]}
    all_rows = base._jsonl(rows_path)
    train_rows = [row for row in all_rows if row["split"] == "train"]
    eval_rows = [row for row in all_rows if row["split"] == "evaluation"]
    api = base._layercake(Path(__file__).resolve().parents[2])
    prepared = _prepare(train_rows, mapping, api["LosslessLexemePointerTokenizer"])
    router = base.CompactRouter.fit([(row["instruction"], mapping[base._instruction_id(row["instruction"])]) for row in train_rows])
    output.mkdir(parents=True)
    private = Ed25519PrivateKey.from_private_bytes(base.SIGNING_SEED)
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    signer = api["key_id"](public_pem)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    levels = []
    for level in LEVELS:
        item = _level(api, prepared, eval_rows, mapping, router, level, output, sha256_file(rows_path), public_pem, private_pem, signer)
        levels.append(item)
        if item["verdict"] == "PASS_LEVEL":
            break
    winner = next((item for item in levels if item["verdict"] == "PASS_LEVEL"), None)
    result = {"format": "abi-r31-english-sufficiency-ladder/1", "verdict": "PASS_PUBLIC_MINIMUM_BRACKET" if winner else "FAIL_PUBLIC_SUFFICIENCY_LADDER", "source_receipt_sha256": sha256_file(receipt_path), "source_rows_sha256": sha256_file(rows_path), "diagnosis_sha256": sha256_file(diagnosis / "result.json"), "levels": levels, "minimum_passing_rows_per_cluster": winner["level"] if winner else None, "information_accounting": {"teacher_present_during_training": False, "teacher_present_during_execution": False, "source_parameters_copied": 0, "logits_stored": 0, "hidden_activations_in_packages": 0, "training_rows_available": len(train_rows), "evaluation_rows": len(eval_rows), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(process.memory_info().rss)}, "claim_ceiling": "PUBLIC_DISCLOSED_MINIMUM_BRACKET_NOT_GENERAL_ENGLISH_OR_ABI_SUPERIORITY", "full_abi_moonshot": "OPEN"}
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
