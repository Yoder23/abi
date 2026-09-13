"""R30 grounded-pointer architecture pilot."""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
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
from .protocol import TASKS, score


ALNUM_RE = re.compile(rb"^[A-Za-z_][A-Za-z0-9_]*$|^\d+(?:\.\d+)?$")


def _normalized_prompt(row: dict[str, Any]) -> str:
    prompt = str(row["prompt"])
    nonce = str(row["nonce"])
    if prompt.count(nonce) > 1:
        lines = [line for line in prompt.splitlines() if line != f"identifier={nonce}"]
        prompt = "\n".join(lines)
    if prompt.count(nonce) > 1:
        raise RuntimeError("R30 duplicate nonce survived normalization")
    return prompt


def _prepare(rows: list[dict[str, Any]], mapping: dict[str, str], tokenizer_type: Any) -> tuple[list[dict[str, Any]], int]:
    prepared = []
    corrections = 0
    for row in rows:
        response, corrected = base._validated_response(row)
        corrections += int(corrected)
        prompt = _normalized_prompt(row)
        source = tokenizer_type.split(prompt)
        target = set(tokenizer_type.split(response))
        marker = source.index(b"SUPPLIED")
        supplied = source[marker:]
        copy = []
        for piece in supplied:
            if ALNUM_RE.fullmatch(piece) and piece in target and source.count(piece) == 1:
                copy.append(piece.decode())
        copy = list(dict.fromkeys(copy))
        if not copy:
            if source.count(b"INSTRUCTION") != 1 or b"INSTRUCTION" in target:
                raise RuntimeError("R30 source-only sentinel contract changed")
            copy = ["INSTRUCTION"]
        prepared.append({"record_id": row["record_id"], "prompt": prompt, "response": response, "cluster": mapping[base._instruction_id(row["instruction"])], "copy_lexemes": copy})
    return prepared, corrections


def _train(api: dict[str, Any], rows: list[dict[str, Any]], seed: int):
    tokenizer_type = api["LosslessLexemePointerTokenizer"]
    fixed = sorted({piece for row in rows for field in ("prompt", "response") for piece in tokenizer_type.split(row[field])})
    tokenizer = tokenizer_type(fixed, format_version="layercake-lossless-lexeme-pointer/2")
    encoded = base._encode(rows, tokenizer)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = api["PortableTokenPlan"](fixed_vocab_size=tokenizer.vocab_size, model_width=64, attention_heads=4, encoder_layers=2, decoder_layers=2, feedforward_width=192, pointer_width=32, dropout=0.0, maximum_source_lexemes=128, maximum_target_actions=256).to(device).bind_tokenizer(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    order = list(range(len(rows)))
    rng = random.Random(seed)
    cursor = len(order)
    started = time.perf_counter()
    history = []
    for step in range(1, 1601):
        if cursor + 8 > len(order):
            rng.shuffle(order)
            cursor = 0
        indexes = order[cursor:cursor + 8]
        cursor += 8
        source, target = base._batch(encoded, indexes, device)
        result = model(source, target)
        mask = target.ge(0)
        loss = F.nll_loss(result["log_probs"][mask], target[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {1, 400, 800, 1200, 1600}:
            prediction = result["log_probs"][mask].argmax(-1)
            history.append({"step": step, "loss": float(loss), "accuracy": float(prediction.eq(target[mask]).float().mean()), "gradient_norm": float(gradient), "seconds": time.perf_counter() - started})
    torch.cuda.synchronize()
    return model.cpu().eval(), tokenizer, {"parameters": model.parameter_count(), "fixed_vocabulary_size": tokenizer.vocab_size, "rows": len(rows), "steps": 1600, "history": history, "seconds": time.perf_counter() - started}


def _package(api, model, tokenizer, cluster, output, source_hash, public_pem, private_pem, signer, training):
    artifact = api["build_token_plan_artifact"](model, tokenizer, domain_id=f"english-core-{cluster}", training={"source_rows_sha256": source_hash, "cluster": cluster, **training})
    state = artifact["state_dict"]
    manifest = api["CakeManifest"](schema_version="1", cake_id=f"abi-r30-v7-{cluster}", name=f"ABI R30 v7 {cluster}", description="R30 source-clustered grounded-pointer English capability pilot", version="0.30.0-v7", publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer}, abi_version=base.ABI_VERSION, abi_hash=base.ABI_SHA256, cake_type="portable_decoder", input_contract={"external": "UTF-8 bytes", "role": "english-core", "mode": "direct_selected_portable_decoder"}, output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "one-source-discovered-capability"}, architecture=api["portable_token_plan_manifest_architecture"](artifact["spec"]), supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"), minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "incremental"]}, tensor_payload_hash="", tensor_shapes=api["tensor_specs"](state), package_hash="", training_data_provenance={"source_model": "Qwen/Qwen2-7B-Instruct", "source_revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c", "source_rows_sha256": source_hash, "source_parameters_copied": 0, "teacher_at_training": False, "teacher_at_inference": False, "receiver_training_steps": 1600}, evaluation_evidence={"status": "UNEVALUATED_R30_V7_PUBLIC"}, license="Apache-2.0", dependencies=(), parent_version=None, signature={"algorithm": "ed25519", "key_id": signer}, domains=(f"english-core-{cluster}",), permissions=("local-inference",))
    destination = output / "packages" / f"{cluster}.cake"
    api["build_package"](destination, manifest, state, private_key=private_pem)
    loaded = api["load_package"](destination, trust_store={signer: public_pem})
    return {"cluster": cluster, "cake_id": loaded.manifest.cake_id, "path": destination, "relative_path": destination.relative_to(output).as_posix(), "sha256": sha256_file(destination), "bytes": destination.stat().st_size, "tensor_payload_hash": loaded.manifest.tensor_payload_hash, "parameters": model.parameter_count(), "training": training}


def _score(row: dict[str, Any], output: str, teacher: str) -> dict[str, Any]:
    value = score(row, output, teacher)
    task = row["oracle_task"]
    lowered = output.casefold()
    prompt_numbers = set(base.DIGIT_RE.findall(row["prompt"]))
    output_numbers = set(base.DIGIT_RE.findall(output))
    if task == "clarification":
        item = re.search(r"item (\d+)", row["prompt"]).group(1)
        value["grounded"] = item in output and output_numbers <= prompt_numbers
    elif task == "planning":
        amount = re.search(r"numbered (\d+)", row["prompt"]).group(1)
        day = re.search(r"before day (\d+)", row["prompt"]).group(1)
        value["grounded"] = amount in output and day in output and output_numbers <= prompt_numbers | {"1", "2", "3"}
    elif task == "comparison":
        a = int(re.search(r"option_a=.*?(\d+)\s*$", row["prompt"], re.MULTILINE).group(1))
        b = int(re.search(r"option_b=.*?(\d+)\s*$", row["prompt"], re.MULTILINE).group(1))
        chosen = "a" if a < b else "b"
        value["grounded"] = str(a) in output and str(b) in output and f"method {chosen}" in lowered and "preferred" in lowered
        value["adherence"] = value["grounded"]
    elif task == "reasoning":
        item = re.search(r"item (\d+) is a nork", row["prompt"]).group(1)
        nonce = row["nonce"].casefold()
        value["grounded"] = nonce in lowered and item in output and lowered.find(nonce) < lowered.find(item)
        value["adherence"] = "glim" in lowered and "nork" in lowered
    elif task == "abstention":
        value["grounded"] = output_numbers <= prompt_numbers
        value["adherence"] = any(marker in lowered for marker in ("cannot", "can't", "not provided", "insufficient", "unable", "no answer", "unsupported"))
    value["functional"] = bool(value["grounded"] and value["adherence"] and value["noncollapsed"])
    return value


def run(source: Path, diagnosis: Path, v6: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"immutable R30 v7 run exists: {output}")
    prior_path = v6 / "result.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("verdict") != "FAIL_PUBLIC_PACKAGE_PILOT" or prior.get("metrics", {}).get("functional") != 79:
        raise RuntimeError("R30 v6 failure binding changed")
    rows_path = source / "source_rows.jsonl"
    diagnosis_path = diagnosis / "result.json"
    diagnosed = json.loads(diagnosis_path.read_text(encoding="utf-8"))
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
        model, tokenizer, training = _train(api, selected, 30_400 + position)
        packages.append(_package(api, model, tokenizer, cluster, output, sha256_file(rows_path), public_pem, private_pem, signer, training))
        print(json.dumps({"trained_packages": position, "cluster": cluster, "seconds": time.perf_counter() - started}), flush=True)
        del model
        gc.collect()
    teacher_scores = [_score(row, row["teacher_output"], row["teacher_output"]) for row in eval_rows]
    observations = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    with tempfile.TemporaryDirectory(prefix="r30-v7-host-") as raw:
        host = api["DirectCakeHost"](Path(raw) / "registry", abi_version=base.ABI_VERSION, abi_hash=base.ABI_SHA256, trust_store={signer: public_pem}, device="cuda")
        by_cluster = {}
        for package in packages:
            installed = host.install(package["path"])
            if installed["archive_hash"] != package["sha256"]:
                raise RuntimeError("R30 v7 installed archive changed")
            by_cluster[package["cluster"]] = package
        for index, row in enumerate(eval_rows, 1):
            predicted = router.predict(row["instruction"])
            expected = mapping[base._instruction_id(row["instruction"])]
            generated = host.generate(by_cluster[predicted]["cake_id"], _normalized_prompt(row), maximum_actions=256)
            torch.cuda.synchronize()
            text = generated.output.decode("utf-8", errors="strict")
            scored = _score(row, text, row["teacher_output"])
            observations.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "expected_cluster": expected, "predicted_cluster": predicted, "route_exact": predicted == expected, "teacher_output": row["teacher_output"], "output": text, "score": scored})
            peak_rss = max(peak_rss, process.memory_info().rss)
            if index == 1 or index % 24 == 0:
                print(json.dumps({"evaluated": index, "functional": sum(item["score"]["functional"] for item in observations), "seconds": time.perf_counter() - started}), flush=True)
        removals = []
        for package in packages:
            host.remove(package["cake_id"])
            rejected = False
            try:
                host.generate(package["cake_id"], _normalized_prompt(eval_rows[0]), maximum_actions=4)
            except (KeyError, ValueError, FileNotFoundError):
                rejected = True
            removals.append(rejected)
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
    passed = route_exact == 144 and functional >= 132 and functional >= teacher_functional and all(item["functional"] >= 10 for item in by_task.values()) and by_task["abstention"]["functional"] == 12 and all(removals)
    result = {"format": "abi-r30-grounded-pointer-layercake-packages/7", "verdict": "PASS_PUBLIC_GROUNDED_POINTER_PILOT" if passed else "FAIL_PUBLIC_GROUNDED_POINTER_PILOT", "v6_result_sha256": sha256_file(prior_path), "source_rows_sha256": sha256_file(rows_path), "diagnosis_sha256": sha256_file(diagnosis_path), "metrics": {"packages": len(packages), "evaluation_rows": len(observations), "route_exact": route_exact, "functional": functional, "teacher_functional": teacher_functional, "by_task": by_task, "validator_corrections_train": corrections, "removal_rejections": sum(removals)}, "packages": [{key: value for key, value in row.items() if key != "path"} for row in packages], "router": router.document, "information_accounting": {"teacher_present_during_training": False, "teacher_present_during_execution": False, "source_parameters_copied": 0, "logits_stored": 0, "hidden_activations_in_packages": 0, "training_rows": len(train_rows), "training_steps_total": sum(row["training"]["steps"] for row in packages), "deployed_parameters": sum(row["parameters"] for row in packages), "maximum_active_parameters": max(row["parameters"] for row in packages), "package_bytes": sum(row["bytes"] for row in packages), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)}, "artifacts": {"evaluation": {"path": rows_out.name, "sha256": sha256_file(rows_out), "bytes": rows_out.stat().st_size}}, "claim_ceiling": "PUBLIC_DISCLOSED_PARAPHRASE_PILOT_NOT_GENERAL_ENGLISH", "full_abi_moonshot": "OPEN"}
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--v6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.v6.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
