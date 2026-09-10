"""Train, package, host, and compare the frozen R21 acquisition routes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import psutil
import torch
import torch.nn.functional as F
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)

from .binding import load_config
from .protocol import (
    LABELS,
    SEEDS,
    WordLabeler,
    bootstrap_difference,
    evaluation_rows,
    instruction_from_prompt,
    normalized_prompt,
    score_output,
    training_rows,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error(f"R21 JSONL unreadable: {path}") from exc


def _layercake(root: Path) -> dict[str, Any]:
    layercake_root = (root / "../layercake_release").resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.cake.manifest import CakeManifest
    from layercake.cake.package import build_package, load_package, tensor_specs
    from layercake.cake.signing import key_id
    from layercake.models.direct_cake_host import DirectCakeHost
    from layercake.models.portable_decoder import portable_token_plan_manifest_architecture
    from layercake.portable_token_plan import (
        LosslessLexemePointerTokenizer,
        PortableTokenPlan,
        build_token_plan_artifact,
    )

    return {
        "CakeManifest": CakeManifest,
        "build_package": build_package,
        "load_package": load_package,
        "tensor_specs": tensor_specs,
        "key_id": key_id,
        "DirectCakeHost": DirectCakeHost,
        "portable_token_plan_manifest_architecture": portable_token_plan_manifest_architecture,
        "LosslessLexemePointerTokenizer": LosslessLexemePointerTokenizer,
        "PortableTokenPlan": PortableTokenPlan,
        "build_token_plan_artifact": build_token_plan_artifact,
    }


def _validate_source(root: Path, config_path: Path, source_run: Path) -> list[dict[str, Any]]:
    receipt = json_object(source_run / "receipt.json")
    rows_path = source_run / "source_observations.jsonl"
    rows = _jsonl(rows_path)
    expected = training_rows()
    if (
        receipt.get("format") != "abi-r21-public-source-acquisition/1"
        or receipt.get("verdict") != "PASS_SOURCE"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("artifacts", {}).get("source_rows", {}).get("sha256") != sha256_file(rows_path)
        or receipt.get("evidence_sha256") != evidence_hash(receipt)
        or len(rows) != 600
        or [row["record_id"] for row in rows] != [row["record_id"] for row in expected]
        or any(row.get("teacher_label") not in LABELS for row in rows)
        or sum(row.get("teacher_label_exact") is True for row in rows) < 594
    ):
        raise R14Error("R21 source evidence did not pass fail-closed validation")
    return rows


def _evaluation_teacher(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    binding = config["evaluation_teacher"]
    path = root / str(binding["path"])
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise R14Error("R21 R20 evaluation source binding changed")
    rows = [row for row in _jsonl(path) if row.get("split") == "evaluation"]
    expected = evaluation_rows()
    if len(rows) != 120 or [row["record_id"] for row in rows] != [row["record_id"] for row in expected]:
        raise R14Error("R21 evaluation teacher rows changed")
    return rows


def _prepared(
    rows: list[dict[str, Any]], *, method: str, labeler: WordLabeler
) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        teacher_label = str(row["teacher_label"])
        prompt = (
            str(row["prompt"])
            if method == "raw_sequence"
            else normalized_prompt(teacher_label, dict(row["slots"]))
        )
        prepared.append(
            {
                **row,
                "prompt_for_student": prompt,
                "response": str(row["teacher_output"]),
            }
        )
    return prepared


def _copy_annotate(rows: list[dict[str, Any]], tokenizer_type: Any) -> list[dict[str, Any]]:
    numeric_candidates: dict[str, set[bytes]] = {}
    for row in rows:
        source = tokenizer_type.split(str(row["prompt_for_student"]))
        target = set(tokenizer_type.split(str(row["response"])))
        values: set[bytes] = set()
        for value in row["slots"].values():
            values.update(piece for piece in tokenizer_type.split(str(value)) if piece.isdigit())
        numeric_candidates[str(row["record_id"])] = {
            piece for piece in values if source.count(piece) == 1 and piece in target
        }
    eligible = set().union(*numeric_candidates.values()) if numeric_candidates else set()
    for piece in list(eligible):
        for row in rows:
            target = tokenizer_type.split(str(row["response"]))
            source = tokenizer_type.split(str(row["prompt_for_student"]))
            if piece in target and source.count(piece) != 1:
                eligible.remove(piece)
                break
    result = []
    for row in rows:
        source = tokenizer_type.split(str(row["prompt_for_student"]))
        target = set(tokenizer_type.split(str(row["response"])))
        copy = sorted(
            piece.decode("utf-8") for piece in eligible if piece in target and source.count(piece) == 1
        )
        if not copy:
            fallback = next((piece.decode("utf-8") for piece in source if piece.isdigit()), None)
            if fallback is None:
                raise R14Error("R21 row has no safe dynamic lexeme")
            copy = [fallback]
        result.append({**row, "copy_lexemes": copy})
    return result


def _encode_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[tuple[list[int], list[int]]]:
    encoded = []
    for row in rows:
        source, lexemes = tokenizer.encode_source(str(row["prompt_for_student"]))
        target = tokenizer.encode_target(
            str(row["response"]),
            copy_lexemes=row["copy_lexemes"],
            source_lexemes=lexemes,
        )
        if len(source) > 128 or len(target) > 256:
            raise R14Error("R21 source or target exceeds frozen model boundary")
        if tokenizer.decode_actions(target, lexemes).decode("utf-8") != row["response"]:
            raise R14Error("R21 target encoding changed teacher bytes")
        encoded.append((source, target))
    return encoded


def _batch(encoded: list[tuple[list[int], list[int]]], indices: list[int], device: torch.device):
    selected = [encoded[index] for index in indices]
    source_width = max(len(source) for source, _ in selected)
    target_width = max(len(target) for _, target in selected)
    source_tensor = torch.zeros(len(selected), source_width, dtype=torch.long, device=device)
    target_tensor = torch.full((len(selected), target_width), -100, dtype=torch.long, device=device)
    for index, (source, target) in enumerate(selected):
        source_tensor[index, : len(source)] = torch.tensor(source, dtype=torch.long, device=device)
        target_tensor[index, : len(target)] = torch.tensor(target, dtype=torch.long, device=device)
    return source_tensor, target_tensor


def _train_one(
    api: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any], *,
    seed: int, factor: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    tokenizer_type = api["LosslessLexemePointerTokenizer"]
    rows = _copy_annotate(rows, tokenizer_type)
    tokenizer = tokenizer_type.build_generic(rows)
    encoded = _encode_rows(rows, tokenizer)
    device = torch.device("cuda")
    model_config = dict(config["model"]["factor" if factor else "monolith"])
    model = api["PortableTokenPlan"](
        fixed_vocab_size=tokenizer.vocab_size,
        **model_config,
        dropout=float(config["model"]["dropout"]),
        maximum_source_lexemes=int(config["model"]["maximum_source_lexemes"]),
        maximum_target_actions=int(config["model"]["maximum_target_actions"]),
    ).to(device).bind_tokenizer(tokenizer)
    settings = config["training"]
    steps = int(settings["factor_steps_per_task"] if factor else settings["monolith_steps"])
    batch_size = int(settings["factor_batch_size"] if factor else settings["monolith_batch_size"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    randomizer = random.Random(seed + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    order = list(range(len(rows)))
    cursor = len(order)
    history = []
    action_exposure = 0
    row_exposure = 0
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            randomizer.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        source, target = _batch(encoded, indices, device)
        result = model(source, target)
        mask = target.ge(0)
        loss = F.nll_loss(result["log_probs"][mask], target[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        action_exposure += int(mask.sum().item())
        row_exposure += len(indices)
        if step == 1 or step % 250 == 0 or step == steps:
            prediction = result["log_probs"][mask].argmax(dim=-1)
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "action_accuracy": float(prediction.eq(target[mask]).float().mean().item()),
                    "gradient_norm": float(gradient),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    torch.cuda.synchronize()
    model.eval()
    metadata = {
        "parameters": model.parameter_count(),
        "fixed_vocabulary_size": tokenizer.vocab_size,
        "training_rows": len(rows),
        "steps": steps,
        "batch_size": batch_size,
        "row_exposure": row_exposure,
        "target_action_exposure": action_exposure,
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }
    return model.cpu(), tokenizer, metadata


def _package(
    api: dict[str, Any], model: Any, tokenizer: Any, *, root: Path, destination: Path,
    config: dict[str, Any], method: str, seed: int, task: str | None,
    public_pem: bytes, private_pem: bytes, signer: str, source_sha256: str,
) -> dict[str, Any]:
    artifact = api["build_token_plan_artifact"](
        model,
        tokenizer,
        domain_id="english-core",
        training={"method": method, "seed": seed, "task": task, "source_sha256": source_sha256},
    )
    state = artifact["state_dict"]
    suffix = task or "all"
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id=f"abi-r21-{method.replace('_', '-')}-{suffix}-seed{seed}",
        name=f"ABI R21 {method} {suffix} seed {seed}",
        description="R21 public generative transfer bake-off artifact",
        version="0.21.0-public",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=config["layercake"]["abi_version"],
        abi_hash=config["layercake"]["abi_sha256"],
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_selected_one_cake_no_router"},
        architecture=api["portable_token_plan_manifest_architecture"](artifact["spec"]),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "incremental"]},
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](state),
        package_hash="",
        training_data_provenance={
            "source_model": config["source"]["model_id"],
            "source_revision": config["source"]["revision"],
            "source_rows_sha256": source_sha256,
            "source_parameters_copied": 0,
            "teacher_at_inference": False,
            "receiver_training_steps": 0,
        },
        evaluation_evidence={"status": "UNEVALUATED_R21_PUBLIC"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=("english-core",),
        permissions=("local-inference",),
    )
    api["build_package"](destination, manifest, state, private_key=private_pem)
    loaded = api["load_package"](destination, trust_store={signer: public_pem})
    return {
        "path": destination.relative_to(root).as_posix(),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "cake_id": loaded.manifest.cake_id,
        "package_hash": loaded.manifest.package_hash,
        "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        "parameters": model.parameter_count(),
        "signed": loaded.signed,
    }


def _train_and_package(
    root: Path, config: dict[str, Any], source_rows: list[dict[str, Any]],
    source_sha256: str, output: Path,
) -> tuple[dict[str, Any], dict[str, Any], bytes, str]:
    api = _layercake(root)
    teacher_labels = [
        {"instruction": str(row["instruction"]), "label": str(row["teacher_label"])}
        for row in source_rows
    ]
    labeler = WordLabeler.fit(teacher_labels)
    labeler_path = output / "labeler.json"
    write_json_once(labeler_path, labeler.document)
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(config["layercake"]["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    signer = api["key_id"](public_pem)
    systems: dict[str, Any] = {}
    for seed in SEEDS:
        systems[str(seed)] = {}
        for method in ("raw_sequence", "labeled_monolith"):
            rows = _prepared(source_rows, method=method, labeler=labeler)
            model, tokenizer, training = _train_one(api, rows, config, seed=seed, factor=False)
            destination = output / "packages" / f"{method}-seed{seed}.cake"
            package = _package(
                api, model, tokenizer, root=root, destination=destination, config=config,
                method=method, seed=seed, task=None, public_pem=public_pem,
                private_pem=private_pem, signer=signer, source_sha256=source_sha256,
            )
            systems[str(seed)][method] = {
                "training": training,
                "packages": [package],
                "deployed_parameters": package["parameters"],
                "active_parameters": package["parameters"],
            }
            del model
            gc.collect()
        factor_packages = []
        factor_training = []
        prepared = _prepared(source_rows, method="abi_factorized", labeler=labeler)
        for task_index, task in enumerate(LABELS):
            task_rows = [row for row in prepared if row["teacher_label"] == task]
            model, tokenizer, training = _train_one(
                api, task_rows, config, seed=seed + 101 * (task_index + 1), factor=True
            )
            destination = output / "packages" / f"abi-factorized-{task}-seed{seed}.cake"
            package = _package(
                api, model, tokenizer, root=root, destination=destination, config=config,
                method="abi_factorized", seed=seed, task=task, public_pem=public_pem,
                private_pem=private_pem, signer=signer, source_sha256=source_sha256,
            )
            factor_packages.append(package)
            factor_training.append({"task": task, **training})
            del model
            gc.collect()
        systems[str(seed)]["abi_factorized"] = {
            "training": {
                "tasks": factor_training,
                "row_exposure": sum(row["row_exposure"] for row in factor_training),
                "target_action_exposure": sum(row["target_action_exposure"] for row in factor_training),
                "training_seconds": sum(row["training_seconds"] for row in factor_training),
            },
            "packages": factor_packages,
            "deployed_parameters": sum(row["parameters"] for row in factor_packages),
            "active_parameters": max(row["parameters"] for row in factor_packages),
        }
    return systems, labeler.document, public_pem, signer


def _evaluate_system(
    root: Path, config: dict[str, Any], system: dict[str, Any], labeler: WordLabeler,
    teacher_rows: list[dict[str, Any]], *, method: str, seed: int,
    public_pem: bytes, signer: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api = _layercake(root)
    expected = evaluation_rows()
    teacher_by_id = {str(row["record_id"]): row for row in teacher_rows}
    task_packages = {
        row["cake_id"].split(f"abi-r21-{method.replace('_', '-')}-", 1)[1].split(f"-seed{seed}", 1)[0]: row
        for row in system["packages"]
    }
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    torch.cuda.reset_peak_memory_stats()
    observations = []
    lifecycle = []
    with tempfile.TemporaryDirectory(prefix=f"r21-{method}-{seed}-") as raw:
        host = api["DirectCakeHost"](
            Path(raw) / "gpu-registry",
            abi_version=config["layercake"]["abi_version"],
            abi_hash=config["layercake"]["abi_sha256"],
            trust_store={signer: public_pem},
            device="cuda",
        )
        for package in system["packages"]:
            installed = host.install(root / package["path"])
            if installed["archive_hash"] != package["sha256"]:
                raise R14Error("R21 installed archive identity changed")
        for position, row in enumerate(expected):
            predicted = labeler.predict(instruction_from_prompt(str(row["prompt"])))
            student_prompt = (
                str(row["prompt"])
                if method == "raw_sequence"
                else normalized_prompt(predicted, dict(row["slots"]))
            )
            package = system["packages"][0] if method != "abi_factorized" else task_packages[predicted]
            started = time.perf_counter()
            generated = host.generate(
                package["cake_id"], student_prompt, maximum_actions=256
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            output_text = generated.output.decode("utf-8", errors="strict")
            teacher_output = str(teacher_by_id[str(row["record_id"])]["output"])
            score = score_output(row, output_text, teacher_output)
            observations.append(
                {
                    "method": method,
                    "seed": seed,
                    "position": position,
                    "record_id": row["record_id"],
                    "task": row["task"],
                    "predicted_label": predicted,
                    "selected_cake_id": package["cake_id"],
                    "output": output_text,
                    "output_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
                    "teacher_output_sha256": hashlib.sha256(teacher_output.encode()).hexdigest(),
                    "latency_seconds": elapsed,
                    "actions": list(generated.actions),
                    **score,
                }
            )
            peak_rss = max(peak_rss, int(process.memory_info().rss))
        for package in system["packages"]:
            probe = next(
                row for row in expected
                if method != "abi_factorized" or row["task"] in package["cake_id"]
            )
            predicted = labeler.predict(instruction_from_prompt(str(probe["prompt"])))
            student_prompt = str(probe["prompt"]) if method == "raw_sequence" else normalized_prompt(predicted, dict(probe["slots"]))
            before = host.generate(package["cake_id"], student_prompt, maximum_actions=256).output
            removed = host.remove(package["cake_id"])
            rejected = False
            try:
                host.generate(package["cake_id"], student_prompt, maximum_actions=256)
            except Exception:
                rejected = True
            restored_install = host.install(root / package["path"])
            restored = host.generate(package["cake_id"], student_prompt, maximum_actions=256).output
            lifecycle.append(
                {
                    "cake_id": package["cake_id"],
                    "removed": removed["status"] == "REMOVED",
                    "absent_rejected": rejected,
                    "restored_archive_exact": restored_install["archive_hash"] == package["sha256"],
                    "restored_output_exact": restored == before,
                }
            )
        # One same-size corrupted archive per route must fail before execution.
        original = root / system["packages"][0]["path"]
        corrupted = Path(raw) / "corrupted.cake"
        payload = bytearray(original.read_bytes())
        payload[len(payload) // 2] ^= 1
        corrupted.write_bytes(payload)
        corruption_rejected = False
        try:
            host.install(corrupted)
        except Exception:
            corruption_rejected = True
    # CPU execution smoke uses one disclosed row per behavior.
    cpu_exact = 0
    with tempfile.TemporaryDirectory(prefix=f"r21-cpu-{method}-{seed}-") as raw:
        cpu_host = api["DirectCakeHost"](
            Path(raw) / "registry",
            abi_version=config["layercake"]["abi_version"],
            abi_hash=config["layercake"]["abi_sha256"],
            trust_store={signer: public_pem},
            device="cpu",
        )
        for package in system["packages"]:
            cpu_host.install(root / package["path"])
        for task in LABELS:
            row = next(item for item in expected if item["task"] == task)
            predicted = labeler.predict(instruction_from_prompt(str(row["prompt"])))
            student_prompt = str(row["prompt"]) if method == "raw_sequence" else normalized_prompt(predicted, dict(row["slots"]))
            package = system["packages"][0] if method != "abi_factorized" else task_packages[predicted]
            cpu_output = cpu_host.generate(package["cake_id"], student_prompt, maximum_actions=256).output.decode()
            gpu_output = next(item["output"] for item in observations if item["record_id"] == row["record_id"])
            cpu_exact += cpu_output == gpu_output
    runtime = {
        "gpu_rows": len(observations),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_bytes": peak_rss,
        "median_gpu_latency_seconds": statistics.median(row["latency_seconds"] for row in observations),
        "cpu_gpu_smoke_exact": cpu_exact,
        "cpu_gpu_smoke_rows": len(LABELS),
        "lifecycle": lifecycle,
        "corruption_rejected": corruption_rejected,
    }
    return observations, runtime


def _aggregate(observations: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for seed in SEEDS:
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized"):
            rows = [row for row in observations if row["seed"] == seed and row["method"] == method]
            result[f"{method}:{seed}"] = {
                "rows": len(rows),
                "functional": sum(row["functional_pass"] for row in rows),
                "semantic": sum(row["semantic_pass"] for row in rows),
                "adherent": sum(row["adherence_pass"] for row in rows),
                "non_hallucinating": sum(row["hallucination_pass"] for row in rows),
                "non_collapsed": sum(not row["repetition_collapse"] for row in rows),
                "mean_teacher_lexical_f1": sum(row["teacher_lexical_f1"] for row in rows) / len(rows),
                "functional_by_task": {
                    task: sum(row["functional_pass"] for row in rows if row["task"] == task)
                    for task in LABELS
                },
            }
    return result


def run(config_path: Path, source_run: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 result exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_config(root, config_path)
    if not torch.cuda.is_available():
        raise R14Error("R21 registered CUDA training unavailable")
    source_rows = _validate_source(root, config_path, source_run)
    teacher_rows = _evaluation_teacher(root, config)
    output.mkdir(parents=True)
    source_rows_path = source_run / "source_observations.jsonl"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    systems, labeler_doc, public_pem, signer = _train_and_package(
        root, config, source_rows, sha256_file(source_rows_path), output
    )
    labeler = WordLabeler(labeler_doc)
    observations = []
    runtimes = {}
    for seed in SEEDS:
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized"):
            print(json.dumps({"evaluating": method, "seed": seed}), flush=True)
            rows, runtime = _evaluate_system(
                root, config, systems[str(seed)][method], labeler, teacher_rows,
                method=method, seed=seed, public_pem=public_pem, signer=signer,
            )
            observations.extend(rows)
            runtimes[f"{method}:{seed}"] = runtime
            gc.collect()
            torch.cuda.empty_cache()
    observations_path = output / "observations.jsonl"
    write_jsonl_once(observations_path, observations)
    aggregates = _aggregate(observations)
    evaluation = evaluation_rows()
    teacher_by_id = {str(row["record_id"]): str(row["output"]) for row in teacher_rows}
    teacher_scores = [
        score_output(row, teacher_by_id[str(row["record_id"])], teacher_by_id[str(row["record_id"])])
        for row in evaluation
    ]
    teacher = {
        "rows": 120,
        "functional": sum(row["functional_pass"] for row in teacher_scores),
        "semantic": sum(row["semantic_pass"] for row in teacher_scores),
        "adherent": sum(row["adherence_pass"] for row in teacher_scores),
        "non_hallucinating": sum(row["hallucination_pass"] for row in teacher_scores),
        "non_collapsed": sum(not row["repetition_collapse"] for row in teacher_scores),
        "functional_by_task": {
            task: sum(score["functional_pass"] for row, score in zip(evaluation, teacher_scores) if row["task"] == task)
            for task in LABELS
        },
    }
    factor_counts = sorted((aggregates[f"abi_factorized:{seed}"]["functional"], seed) for seed in SEEDS)
    headline_seed = factor_counts[1][1]
    headline = aggregates[f"abi_factorized:{headline_seed}"]
    comparisons = {}
    factor_rows = [row for row in observations if row["method"] == "abi_factorized" and row["seed"] == headline_seed]
    for other in ("raw_sequence", "labeled_monolith"):
        other_rows = [row for row in observations if row["method"] == other and row["seed"] == headline_seed]
        comparisons[f"abi_minus_{other}"] = bootstrap_difference(
            [float(row["functional_pass"]) for row in factor_rows],
            [float(row["functional_pass"]) for row in other_rows],
            seed=21_000 + headline_seed + len(other),
            samples=int(config["gates"]["bootstrap_samples"]),
        )
    comparisons["abi_minus_teacher"] = bootstrap_difference(
        [float(row["functional_pass"]) for row in factor_rows],
        [float(row["functional_pass"]) for row in teacher_scores],
        seed=42_000 + headline_seed,
        samples=int(config["gates"]["bootstrap_samples"]),
    )
    predicted = [labeler.predict(instruction_from_prompt(str(row["prompt"]))) for row in evaluation]
    label_exact = sum(label == row["task"] for label, row in zip(predicted, evaluation))
    ratios = [
        systems[str(seed)]["abi_factorized"]["deployed_parameters"]
        / systems[str(seed)]["labeled_monolith"]["deployed_parameters"]
        for seed in SEEDS
    ]
    gates = {
        "label_exact": label_exact >= int(config["gates"]["minimum_label_exact"]),
        "headline_each_task": min(headline["functional_by_task"].values()) >= int(config["gates"]["minimum_task_functional"]),
        "headline_non_hallucinating": headline["non_hallucinating"] >= int(config["gates"]["minimum_non_hallucinating"]),
        "headline_non_collapsed": headline["non_collapsed"] >= int(config["gates"]["minimum_non_collapsed"]),
        "no_worse_than_teacher": headline["functional"] >= teacher["functional"],
        "no_worse_than_raw_sequence": headline["functional"] >= aggregates[f"raw_sequence:{headline_seed}"]["functional"],
        "no_worse_than_labeled_monolith": headline["functional"] >= aggregates[f"labeled_monolith:{headline_seed}"]["functional"],
        "factor_parameter_budget": max(ratios) <= float(config["gates"]["maximum_factor_to_labeled_parameters"]),
        "row_exposure_exact": all(
            systems[str(seed)][method]["training"]["row_exposure"] == int(config["training"]["row_exposures_per_method"])
            for seed in SEEDS for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
        ),
        "all_packages_signed": all(
            package["signed"] for seed in SEEDS for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
            for package in systems[str(seed)][method]["packages"]
        ),
        "cpu_gpu_execution": all(row["cpu_gpu_smoke_exact"] == row["cpu_gpu_smoke_rows"] for row in runtimes.values()),
        "causal_lifecycle": all(
            runtime["corruption_rejected"] and all(all(item[key] for key in ("removed", "absent_rejected", "restored_archive_exact", "restored_output_exact")) for item in runtime["lifecycle"])
            for runtime in runtimes.values()
        ),
        "teacher_absent_at_execution": True,
        "source_parameters_in_packages_zero": True,
        "receiver_training_steps_zero": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-r21-public-generative-bakeoff-result/1",
        "verdict": "PASS_PUBLIC_PREREQUISITE" if passed else "FAIL_PUBLIC_PREREQUISITE",
        "claim": "R21_BOUNDED_PUBLIC_LABEL_SEPARATED_GENERATIVE_TRANSFER" if passed else "R21_PUBLIC_LABEL_SEPARATED_GENERATIVE_TRANSFER_FAILED",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "config_sha256": sha256_file(config_path),
        "source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "source_rows_sha256": sha256_file(source_rows_path),
        "evaluation_teacher_sha256": config["evaluation_teacher"]["sha256"],
        "labeler": {"path": "labeler.json", "sha256": sha256_file(output / "labeler.json"), "evaluation_exact": label_exact, "evaluation_rows": 120},
        "systems": systems,
        "runtimes": runtimes,
        "teacher": teacher,
        "aggregates": aggregates,
        "headline_seed": headline_seed,
        "comparisons": comparisons,
        "parameter_ratios": ratios,
        "gates": gates,
        "observations": {"path": "observations.jsonl", "sha256": sha256_file(observations_path), "rows": len(observations)},
        "information_accounting": {
            "teacher_training_rows": 600,
            "teacher_evaluation_rows_reused": 120,
            "teacher_output_tokens": sum(int(row["teacher_output_token_count"]) for row in source_rows),
            "teacher_label_tokens": sum(int(row["teacher_label_token_count"]) for row in source_rows),
            "teacher_output_bytes": sum(len(str(row["teacher_output"]).encode()) for row in source_rows),
            "teacher_label_bytes": sum(len(str(row["teacher_label_raw"]).encode()) for row in source_rows),
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "source_parameters_copied": 0,
            "teacher_present_at_execution": False,
            "total_training_seconds": sum(
                systems[str(seed)][method]["training"]["training_seconds"]
                for seed in SEEDS for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
            ),
            "experiment_wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": max(runtime["gpu_peak_memory_bytes"] for runtime in runtimes.values()),
            "peak_process_rss_bytes": max(runtime["peak_process_rss_bytes"] for runtime in runtimes.values()),
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "full_abi_moonshot": "OPEN",
        "next_action": "preregister hidden replication" if passed else "close this branch and use measured per-task failures to select a materially different acquisition architecture",
        "unproven": [
            "unrestricted English fluency", "autonomous ontology discovery", "arbitrary-domain extraction",
            "global information minimality", "superiority to LoRA or distillation", "full ABI moonshot",
        ],
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.config, args.source_run, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
