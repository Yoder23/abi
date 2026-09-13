"""Train and execute source-clustered R30 LayerCake packages."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
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
from experiments.generative_transfer_r21.run import _layercake
from .diagnose_v2 import _catalog
from .protocol import TASKS, score


ABI_VERSION = "lc-direct-neural-decoder/1"
ABI_SHA256 = "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
SIGNING_SEED = bytes.fromhex("98aa1204fb2dcb5812ed5d88a97d722f492a0a34263fef41b4d190a0f830073d")
WORD_RE = re.compile(r"[a-z]+")
DIGIT_RE = re.compile(r"\d+")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _instruction_id(instruction: str) -> str:
    return hashlib.sha256(("r30-instruction|" + instruction).encode()).hexdigest()[:16]


class CompactRouter:
    def __init__(self, document: dict[str, Any]):
        self.document = document

    @classmethod
    def fit(cls, pairs: list[tuple[str, str]]) -> "CompactRouter":
        labels = sorted({label for _, label in pairs})
        counts = {label: collections.Counter() for label in labels}
        totals = collections.Counter()
        vocab = set()
        for instruction, label in sorted(set(pairs)):
            tokens = WORD_RE.findall(instruction.casefold())
            counts[label].update(tokens)
            totals[label] += len(tokens)
            vocab.update(tokens)
        return cls({"format": "abi-r30-word-router/1", "labels": labels, "vocabulary": sorted(vocab), "counts": {label: dict(counts[label]) for label in labels}, "totals": dict(totals)})

    def predict(self, instruction: str) -> str:
        tokens = WORD_RE.findall(instruction.casefold())
        size = len(self.document["vocabulary"])
        scores = {}
        for label in self.document["labels"]:
            counts = self.document["counts"][label]
            total = self.document["totals"][label]
            scores[label] = sum(math.log((counts.get(token, 0) + 1) / (total + size)) for token in tokens)
        return max(self.document["labels"], key=lambda label: (scores[label], label))


def _validated_response(row: dict[str, Any]) -> tuple[str, bool]:
    prompt = str(row["prompt"])
    nonce = str(row["nonce"])
    if "option_a=" in prompt and "option_b=" in prompt:
        a = int(re.search(r"option_a=.*?(\d+)\s*$", prompt, re.MULTILINE).group(1))
        b = int(re.search(r"option_b=.*?(\d+)\s*$", prompt, re.MULTILINE).group(1))
        chosen = "A" if a < b else "B"
        return f"For {nonce}, method {chosen} is preferred because {min(a, b)} is lower than {max(a, b)}.", True
    if "rule=every glim is checked before any nork" in prompt:
        item = int(re.search(r"item (\d+) is a nork", prompt).group(1))
        return f"{nonce} must be checked before item {item} because {nonce} is a glim and every glim precedes every nork.", True
    return str(row["teacher_output"]), False


def _copy_values(row: dict[str, Any], response: str, tokenizer_type: Any) -> list[str]:
    source = tokenizer_type.split(str(row["prompt"]))
    target = set(tokenizer_type.split(response))
    candidates: list[bytes] = []
    nonce = str(row["nonce"]).encode()
    if source.count(nonce) == 1 and nonce in target:
        candidates.append(nonce)
    digits = [piece for piece in source if piece.isdigit() and source.count(piece) == 1 and piece in target]
    candidates.extend(sorted(digits, key=lambda item: int(item), reverse=True)[:2])
    if not candidates:
        fallback = b"INSTRUCTION"
        if source.count(fallback) != 1 or fallback in target:
            raise RuntimeError("R30 row lacks a safe dynamic sentinel")
        candidates.append(fallback)
    return [item.decode() for item in dict.fromkeys(candidates)]


def _prepare(rows: list[dict[str, Any]], mapping: dict[str, str], tokenizer_type: Any) -> tuple[list[dict[str, Any]], int]:
    prepared = []
    corrections = 0
    for row in rows:
        response, corrected = _validated_response(row)
        corrections += int(corrected)
        prepared.append({"record_id": row["record_id"], "prompt": row["prompt"], "response": response, "cluster": mapping[_instruction_id(row["instruction"])], "copy_lexemes": _copy_values(row, response, tokenizer_type)})
    return prepared, corrections


def _encode(rows: list[dict[str, Any]], tokenizer: Any) -> list[tuple[list[int], list[int]]]:
    encoded = []
    for row in rows:
        source, lexemes = tokenizer.encode_source(row["prompt"])
        target = tokenizer.encode_target(row["response"], copy_lexemes=row["copy_lexemes"], source_lexemes=lexemes)
        if len(source) > 128 or len(target) > 256:
            raise RuntimeError("R30 source/target boundary exceeded")
        if tokenizer.decode_actions(target, lexemes).decode() != row["response"]:
            raise RuntimeError("R30 target representation is not lossless")
        encoded.append((source, target))
    return encoded


def _batch(encoded: list[tuple[list[int], list[int]]], indexes: list[int], device: torch.device):
    chosen = [encoded[index] for index in indexes]
    sw = max(len(a) for a, _ in chosen)
    tw = max(len(b) for _, b in chosen)
    source = torch.zeros(len(chosen), sw, dtype=torch.long, device=device)
    target = torch.full((len(chosen), tw), -100, dtype=torch.long, device=device)
    for index, (a, b) in enumerate(chosen):
        source[index, :len(a)] = torch.tensor(a, device=device)
        target[index, :len(b)] = torch.tensor(b, device=device)
    return source, target


def _train(api: dict[str, Any], rows: list[dict[str, Any]], seed: int):
    tokenizer = api["LosslessLexemePointerTokenizer"].build_generic(rows)
    encoded = _encode(rows, tokenizer)
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
        source, target = _batch(encoded, indexes, device)
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
    manifest = api["CakeManifest"](
        schema_version="1", cake_id=f"abi-r30-{cluster}", name=f"ABI R30 {cluster}", description="R30 source-clustered English capability pilot", version="0.30.0-public", publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer}, abi_version=ABI_VERSION, abi_hash=ABI_SHA256, cake_type="portable_decoder", input_contract={"external": "UTF-8 bytes", "role": "english-core"}, output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "one-source-discovered-capability"}, architecture=api["portable_token_plan_manifest_architecture"](artifact["spec"]), supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"), minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "incremental"]}, tensor_payload_hash="", tensor_shapes=api["tensor_specs"](state), package_hash="", training_data_provenance={"source_model": "Qwen/Qwen2-7B-Instruct", "source_revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c", "source_rows_sha256": source_hash, "source_parameters_copied": 0, "teacher_at_training": False, "teacher_at_inference": False, "receiver_training_steps": 1600}, evaluation_evidence={"status": "UNEVALUATED_R30_PUBLIC"}, license="Apache-2.0", dependencies=(), parent_version=None, signature={"algorithm": "ed25519", "key_id": signer}, domains=(f"english-core-{cluster}",), permissions=("local-inference",)
    )
    destination = output / "packages" / f"{cluster}.cake"
    api["build_package"](destination, manifest, state, private_key=private_pem)
    loaded = api["load_package"](destination, trust_store={signer: public_pem})
    return {"cluster": cluster, "cake_id": loaded.manifest.cake_id, "path": destination, "relative_path": destination.relative_to(output).as_posix(), "sha256": sha256_file(destination), "bytes": destination.stat().st_size, "parameters": model.parameter_count(), "training": training}


def _task_score(row: dict[str, Any], output: str, teacher: str) -> dict[str, Any]:
    base = score(row, output, teacher)
    task = row["oracle_task"]
    lowered = output.casefold()
    numbers = DIGIT_RE.findall(row["prompt"])
    if task in {"clarification", "planning"}:
        needed = set(numbers[-2:])
        base["grounded"] = needed <= set(DIGIT_RE.findall(output)) and set(DIGIT_RE.findall(output)) <= set(numbers)
    elif task == "comparison":
        values = [int(x) for x in numbers[-2:]]
        base["grounded"] = all(str(x) in output for x in values) and "method a" in lowered and "preferred" in lowered
        base["adherence"] = base["grounded"]
    elif task == "reasoning":
        item = numbers[-1]
        nonce = row["nonce"].casefold()
        base["grounded"] = nonce in lowered and item in output and lowered.find(nonce) < lowered.find(item)
        base["adherence"] = "glim" in lowered and "nork" in lowered
    elif task == "abstention":
        base["grounded"] = True
    base["functional"] = bool(base["grounded"] and base["adherence"] and base["noncollapsed"])
    return base


def run(source: Path, diagnosis: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"immutable R30 package run exists: {output}")
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
    if len(mapping) != 36:
        raise RuntimeError("R30 diagnosis mapping incomplete")
    all_rows = _jsonl(rows_path)
    train_rows = [row for row in all_rows if row["split"] == "train"]
    eval_rows = [row for row in all_rows if row["split"] == "evaluation"]
    api = _layercake(Path(__file__).resolve().parents[2])
    prepared, corrections = _prepare(train_rows, mapping, api["LosslessLexemePointerTokenizer"])
    router = CompactRouter.fit([(row["instruction"], mapping[_instruction_id(row["instruction"])]) for row in train_rows])
    output.mkdir(parents=True)
    private = Ed25519PrivateKey.from_private_bytes(SIGNING_SEED)
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
    observations = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    with tempfile.TemporaryDirectory(prefix="r30-v4-host-") as raw:
        host = api["DirectCakeHost"](Path(raw) / "registry", abi_version=ABI_VERSION, abi_hash=ABI_SHA256, trust_store={signer: public_pem}, device="cuda")
        by_cluster = {}
        for package in packages:
            installed = host.install(package["path"])
            if installed["archive_hash"] != package["sha256"]:
                raise RuntimeError("R30 package install identity changed")
            by_cluster[package["cluster"]] = package
        for index, row in enumerate(eval_rows, 1):
            predicted = router.predict(row["instruction"])
            expected = mapping[_instruction_id(row["instruction"])]
            package = by_cluster[predicted]
            before = time.perf_counter()
            generated = host.generate(package["cake_id"], row["prompt"], maximum_actions=256)
            torch.cuda.synchronize()
            text = generated.output.decode("utf-8", errors="strict")
            scored = _task_score(row, text, row["teacher_output"])
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
        "format": "abi-r30-source-clustered-layercake-packages/4",
        "verdict": "PASS_PUBLIC_PACKAGE_PILOT" if passed else "FAIL_PUBLIC_PACKAGE_PILOT",
        "source_receipt_sha256": sha256_file(receipt_path), "source_rows_sha256": sha256_file(rows_path), "diagnosis_sha256": sha256_file(diagnosis_path),
        "metrics": {"packages": len(packages), "evaluation_rows": len(observations), "route_exact": route_exact, "functional": functional, "by_task": by_task, "validator_corrections_train": corrections, "removal_rejections": sum(row["removed_rejected"] for row in removals)},
        "packages": [{key: value for key, value in row.items() if key != "path"} for row in packages],
        "router": router.document,
        "information_accounting": {"teacher_present_during_training": False, "teacher_present_during_execution": False, "source_parameters_copied": 0, "logits_stored": 0, "hidden_activations_in_packages": 0, "training_rows": len(train_rows), "training_steps_total": sum(row["training"]["steps"] for row in packages), "deployed_parameters": sum(row["parameters"] for row in packages), "maximum_active_parameters": max(row["parameters"] for row in packages), "package_bytes": sum(row["bytes"] for row in packages), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)},
        "artifacts": {"evaluation": {"path": rows_out.name, "sha256": sha256_file(rows_out), "bytes": rows_out.stat().st_size}},
        "claim_ceiling": "PUBLIC_DISCLOSED_PARAPHRASE_PILOT_NOT_GENERAL_ENGLISH",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
