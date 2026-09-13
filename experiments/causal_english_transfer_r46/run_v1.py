"""Run the preregistered R46 causal English-transfer feasibility experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once


TRAINING_SHA256 = "82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150"
EVALUATION_SHA256 = "b81477560f886ef79df08618956fb9f2ed721296fd5b97e1c6dec8798d68d4f1"
PARENT_SHA256 = "fc8d32c77bfa9d39152a6d32436c4dbbf767f9d1c88f3de17fec84a24bd3f781"
TOKENIZER_SHA256 = "d1c03801fd9559b8586490733a3788439ff230bf5f09666a0abb7ddb615e9f10"
CAPABILITIES = (
    "abstention",
    "cake_output_realization",
    "clarification",
    "coherence",
    "conversation",
    "domain_independent_reasoning",
    "email_drafting",
    "format_control",
    "grammar",
    "instruction_following",
    "prompt_grounding",
    "rewriting",
    "summarization",
    "tone_control",
)
SEED = 46_001
STEPS = 1_745
BATCH_SIZE = 14
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
BALANCE_WEIGHT = 0.02
GRADIENT_CLIP = 1.0
MAXIMUM_NEW_TOKENS = 384
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM = 4
TERMINATOR = b"\n<|end|>"


class R46Error(RuntimeError):
    pass


def _read_bundle(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise R46Error(f"R46 source archive changed: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {"manifest.json", "records.jsonl", "probe_results.json"}
            if not required <= names:
                raise R46Error(f"R46 bundle is incomplete: {path}")
            manifest = json.loads(archive.read("manifest.json"))
            rows_raw = archive.read("records.jsonl")
            probes_raw = archive.read("probe_results.json")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R46Error(f"R46 bundle is unreadable: {path}") from exc
    members = manifest.get("members", {})
    for name, payload in (("records.jsonl", rows_raw), ("probe_results.json", probes_raw)):
        receipt = members.get(name, {})
        if (
            receipt.get("bytes") != len(payload)
            or receipt.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise R46Error(f"R46 bundle member receipt changed: {path}/{name}")
    rows = [json.loads(line) for line in rows_raw.decode("utf-8").splitlines() if line]
    probes = json.loads(probes_raw)
    if not isinstance(manifest, dict) or not isinstance(probes, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise R46Error(f"R46 bundle schema changed: {path}")
    if manifest.get("record_count") != len(rows) or manifest.get("probe_result_count") != len(probes):
        raise R46Error(f"R46 bundle cardinality receipt changed: {path}")
    return {"manifest": manifest, "records": rows, "probes": probes}


def _evaluator_by_record(bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in bundle["probes"]:
        record_id = str(row.get("record_id", ""))
        evaluator = row.get("evaluator")
        if not record_id or not isinstance(evaluator, dict) or record_id in result:
            raise R46Error("R46 evaluation bundle has ambiguous evaluator evidence")
        result[record_id] = {"probe_id": str(row["probe_id"]), "evaluator": evaluator}
    return result


def _prepare_training(rows: Sequence[Mapping[str, Any]], tokenizer: Any, maximum: int):
    prepared: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = []
    total_response_tokens = 0
    for row in rows:
        capability = str(row.get("capability", ""))
        if capability not in CAPABILITIES or row.get("destination_scope") != "english_core":
            raise R46Error("R46 training archive contains an out-of-scope record")
        if (
            row.get("knowledge_class") != "english_linguistic_form"
            or row.get("domain") != "domain_independent"
            or row.get("domain_claims") != []
            or row.get("domain_labels") != []
            or row.get("teacher_token_count_authoritative") is not True
        ):
            raise R46Error("R46 training record violates the English segregation contract")
        prompt = str(row["prompt"])
        output = str(row["output"])
        prompt.encode("utf-8", errors="strict")
        output.encode("utf-8", errors="strict")
        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(output.encode("utf-8") + TERMINATOR)
        combined = prompt_ids + response_ids
        if len(combined) > maximum:
            excluded.append(
                {
                    "record_id": row["record_id"],
                    "combined_tokens": len(combined),
                    "prompt_sha256": row["prompt_sha256"],
                    "output_sha256": row["output_sha256"],
                }
            )
            continue
        prepared[capability].append(
            {
                "record_id": str(row["record_id"]),
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "combined": combined,
            }
        )
        total_response_tokens += len(response_ids)
    if set(prepared) != set(CAPABILITIES) or len(excluded) != 5:
        raise R46Error("R46 prepared training population changed")
    for capability in CAPABILITIES:
        prepared[capability].sort(key=lambda row: row["record_id"])
    return prepared, excluded, total_response_tokens


def _batch(rows: Sequence[Mapping[str, Any]], device: torch.device):
    maximum = max(len(row["combined"]) for row in rows)
    inputs = torch.full((len(rows), maximum - 1), 32, dtype=torch.long, device=device)
    targets = torch.full((len(rows), maximum - 1), -100, dtype=torch.long, device=device)
    prompt_lengths = torch.empty(len(rows), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        sequence = row["combined"]
        length = len(sequence) - 1
        prompt_length = len(row["prompt_ids"])
        inputs[index, :length] = torch.tensor(sequence[:-1], dtype=torch.long, device=device)
        targets[index, prompt_length - 1 : length] = torch.tensor(
            sequence[prompt_length:], dtype=torch.long, device=device
        )
        prompt_lengths[index] = prompt_length
    return inputs, targets, prompt_lengths


def _select(logits: torch.Tensor, generated: Sequence[int]) -> int:
    scores = logits.float().clone()
    if generated:
        for token_id in set(generated[-128:]):
            value = scores[token_id]
            scores[token_id] = value / REPETITION_PENALTY if value > 0 else value * REPETITION_PENALTY
    if len(generated) >= NO_REPEAT_NGRAM - 1:
        prefix = tuple(generated[-(NO_REPEAT_NGRAM - 1) :])
        blocked = {
            generated[index + NO_REPEAT_NGRAM - 1]
            for index in range(len(generated) - NO_REPEAT_NGRAM + 1)
            if tuple(generated[index : index + NO_REPEAT_NGRAM - 1]) == prefix
        }
        if blocked:
            scores[list(blocked)] = -torch.inf
    return int(scores.argmax().item())


def _generate(model: Any, tokenizer: Any, prompt: str, device: torch.device) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids or len(prompt_ids) >= model.config.max_tokens:
        raise R46Error("R46 evaluation prompt exceeds the LayerCake context")
    calls = [0 for _ in model.cakes.experts]
    hooks = []
    for index, expert in enumerate(model.cakes.experts):
        def counted(_module: Any, _inputs: Any, _output: Any, *, slot: int = index) -> None:
            calls[slot] += 1
        hooks.append(expert.register_forward_hook(counted))
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            state = model.prefill(torch.tensor([prompt_ids], dtype=torch.long, device=device))
        prefill_calls = list(calls)
        calls[:] = [0 for _ in calls]
        generated: list[int] = []
        payload = bytearray()
        maximum = min(MAXIMUM_NEW_TOKENS, model.config.max_tokens - len(prompt_ids))
        with torch.inference_mode():
            for _ in range(maximum):
                token_id = _select(state.next_logits[0], generated)
                generated.append(token_id)
                payload.extend(tokenizer.decode([token_id]))
                selected = torch.tensor([token_id], dtype=torch.long, device=device)
                _, state = model.decode_step(state, next_token=selected)
                if TERMINATOR in payload:
                    break
    finally:
        for hook in hooks:
            hook.remove()
    elapsed = time.perf_counter() - started
    terminated = TERMINATOR in payload
    clean = bytes(payload).split(TERMINATOR, 1)[0]
    error = None
    try:
        output = clean.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        output, error = "", f"{type(exc).__name__}: {exc}"
    collapse = _collapse_metrics(generated, output, prompt_ids, prompt)
    return {
        "output": output,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output_token_ids": generated,
        "generated_tokens": len(generated),
        "terminated": terminated,
        "generation_error": error,
        "latency_seconds": elapsed,
        "prefill_expert_forward_calls": prefill_calls,
        "decode_expert_forward_calls": calls,
        "decode_expert_invocations": sum(calls),
        "maximum_active_decode_experts_per_token": 1 if generated and sum(calls) == len(generated) else None,
        "collapse": collapse,
    }


def _evaluate(
    system: str,
    model: Any,
    tokenizer: Any,
    selected: Sequence[Mapping[str, Any]],
    evaluators: Mapping[str, Mapping[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for index, source in enumerate(selected, 1):
        generated = _generate(model, tokenizer, str(source["prompt"]), device)
        evaluator = evaluators[str(source["record_id"])]
        passed, score = evaluate_output(generated["output"], dict(evaluator["evaluator"]))
        rows.append(
            {
                "system": system,
                "record_id": source["record_id"],
                "probe_id": evaluator["probe_id"],
                "capability": source["capability"],
                "prompt_sha256": source["prompt_sha256"],
                "evaluator": evaluator["evaluator"],
                "functional_pass": passed,
                "functional_score": score,
                **generated,
            }
        )
        if index == 1 or index % 20 == 0:
            print(
                json.dumps(
                    {
                        "system": system,
                        "rows": index,
                        "functional": sum(row["functional_pass"] for row in rows),
                        "collapsed": sum(row["collapse"]["collapse_detected"] for row in rows),
                    }
                ),
                flush=True,
            )
    return rows


def _bootstrap(left: Sequence[bool], right: Sequence[bool], seed: int) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise R46Error("R46 paired bootstrap input changed")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    values = []
    for _ in range(5_000):
        values.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    values.sort()
    return {
        "point": sum(differences) / len(differences),
        "lower_95": values[124],
        "upper_95": values[4_874],
        "replicates": 5_000,
    }


def run(layercake_root: Path, training_path: Path, evaluation_path: Path, parent_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R46Error(f"immutable R46 output exists: {output}")
    if not torch.cuda.is_available():
        raise R46Error("R46 requires CUDA")
    parent_model = parent_dir / "model.safetensors"
    parent_tokenizer = parent_dir / "tokenizer.json"
    parent_metadata_path = parent_dir / "metadata.json"
    if (
        not parent_metadata_path.is_file()
        or sha256_file(parent_model) != PARENT_SHA256
        or sha256_file(parent_tokenizer) != TOKENIZER_SHA256
    ):
        raise R46Error("R46 LayerCake parent changed")
    training = _read_bundle(training_path, TRAINING_SHA256)
    evaluation = _read_bundle(evaluation_path, EVALUATION_SHA256)
    if (
        len(training["records"]) != 24_421
        or sum(int(row["teacher_tokens"]) for row in training["records"]) != 2_784_714
        or training["manifest"].get("training_eligible") is not True
        or evaluation["manifest"].get("training_eligible") is not False
    ):
        raise R46Error("R46 source archive accounting changed")
    validation = [row for row in evaluation["records"] if row.get("split") == "validation"]
    overlaps = {
        "record_ids": len({row["record_id"] for row in training["records"]} & {row["record_id"] for row in validation}),
        "prompt_sha256": len({row["prompt_sha256"] for row in training["records"]} & {row["prompt_sha256"] for row in validation}),
        "output_sha256": len({row["output_sha256"] for row in training["records"]} & {row["output_sha256"] for row in validation}),
    }
    if len(validation) != 1_400 or any(overlaps.values()):
        raise R46Error("R46 train/evaluation isolation changed")
    selected = []
    for capability in CAPABILITIES:
        population = sorted(
            (row for row in validation if row["capability"] == capability),
            key=lambda row: row["record_id"],
        )
        if len(population) != 100:
            raise R46Error("R46 evaluation stratum changed")
        selected.extend(population[:20])
    evaluators = _evaluator_by_record(evaluation)
    if len(selected) != 280 or any(str(row["record_id"]) not in evaluators for row in selected):
        raise R46Error("R46 selected evaluation matrix changed")

    sys.path.insert(0, str(layercake_root))
    from layercake.models.sparse_bpe_layercake import LayerCakeSparseBPECore, SparseBPELayerCakeConfig
    from layercake.training.phase2_sparse_bpe import _tokenizer

    metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint", {}).get("sha256") != PARENT_SHA256:
        raise R46Error("R46 parent metadata changed")
    tokenizer = _tokenizer(parent_tokenizer)
    if tokenizer.vocab_size != metadata["architecture"]["vocab_size"]:
        raise R46Error("R46 tokenizer vocabulary changed")
    prepared, excluded, prepared_response_tokens = _prepare_training(
        training["records"], tokenizer, int(metadata["architecture"]["max_tokens"])
    )
    output.mkdir(parents=True)
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(**metadata["architecture"])).to(device)
    model.load_state_dict(load_file(str(parent_model), device="cuda"), strict=True)
    initial_state_hash = sha256_file(parent_model)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    parent_rows = _evaluate("parent", model, tokenizer, selected, evaluators, device)

    rng = random.Random(SEED)
    orders: dict[str, list[dict[str, Any]]] = {}
    cursors: dict[str, int] = {}
    for capability in CAPABILITIES:
        orders[capability] = list(prepared[capability])
        rng.shuffle(orders[capability])
        cursors[capability] = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sequence_digest = hashlib.sha256()
    assignment_counts = torch.zeros(len(model.cakes.experts), dtype=torch.long)
    curves = []
    started = time.perf_counter()
    model.train()
    for step in range(1, STEPS + 1):
        batch_rows = []
        for capability in CAPABILITIES:
            position = cursors[capability]
            values = orders[capability]
            if position == len(values):
                rng.shuffle(values)
                position = 0
            row = values[position]
            cursors[capability] = position + 1
            batch_rows.append(row)
            sequence_digest.update((capability + "\0" + row["record_id"] + "\n").encode())
        inputs, targets, prompt_lengths = _batch(batch_rows, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, prompt_lengths=prompt_lengths)
        language_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        routing = model.last_routing_aux
        if routing is None:
            raise R46Error("R46 LayerCake routing evidence is absent")
        loss = language_loss + BALANCE_WEIGHT * routing["balance_loss"]
        if not torch.isfinite(loss):
            raise R46Error(f"R46 non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        assignment_counts += routing["assignment_counts"].detach().cpu()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % 100 == 0 or step == STEPS:
            trace = {
                "step": step,
                "language_loss": float(language_loss.detach()),
                "training_loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "maximum_sequence_tokens": int(inputs.shape[1] + 1),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(trace)
            print(json.dumps(trace), flush=True)
    training_seconds = time.perf_counter() - started
    if sum(value > 0 for value in assignment_counts.tolist()) != 8:
        raise R46Error("R46 did not exercise all physical sparse experts")
    model.eval()
    candidate_rows = _evaluate("candidate", model, tokenizer, selected, evaluators, device)
    checkpoint_path = output / "model.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        str(checkpoint_path),
        metadata={"format": "abi-r46-causal-english-transfer/1"},
    )
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    teacher_rows = []
    for source in selected:
        evaluator = evaluators[str(source["record_id"])]
        passed, score = evaluate_output(str(source["output"]), dict(evaluator["evaluator"]))
        teacher_rows.append(
            {
                "system": "source",
                "record_id": source["record_id"],
                "probe_id": evaluator["probe_id"],
                "capability": source["capability"],
                "prompt_sha256": source["prompt_sha256"],
                "output": source["output"],
                "output_sha256": source["output_sha256"],
                "teacher_tokens": source["teacher_tokens"],
                "teacher_token_count_authoritative": source["teacher_token_count_authoritative"],
                "evaluator": evaluator["evaluator"],
                "functional_pass": passed,
                "functional_score": score,
            }
        )
    raw_rows = parent_rows + candidate_rows + teacher_rows
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, raw_rows)
    excluded_path = output / "excluded_training_rows.jsonl"
    write_jsonl_once(excluded_path, excluded)
    trace_path = output / "training_trace.jsonl"
    write_jsonl_once(trace_path, curves)
    by_system = {system: [row for row in raw_rows if row["system"] == system] for system in ("parent", "candidate", "source")}
    by_capability = {
        capability: {
            system: sum(row["functional_pass"] for row in by_system[system] if row["capability"] == capability)
            for system in by_system
        }
        for capability in CAPABILITIES
    }
    metrics = {
        "evaluation_rows_per_system": 280,
        "functional": {system: sum(row["functional_pass"] for row in values) for system, values in by_system.items()},
        "functional_rate": {system: sum(row["functional_pass"] for row in values) / len(values) for system, values in by_system.items()},
        "by_capability": by_capability,
        "candidate_minus_parent": _bootstrap(
            [row["functional_pass"] for row in by_system["candidate"]],
            [row["functional_pass"] for row in by_system["parent"]],
            SEED + 1,
        ),
        "candidate_minus_source": _bootstrap(
            [row["functional_pass"] for row in by_system["candidate"]],
            [row["functional_pass"] for row in by_system["source"]],
            SEED + 2,
        ),
        "candidate_collapses": sum(row["collapse"]["collapse_detected"] for row in candidate_rows),
        "parent_collapses": sum(row["collapse"]["collapse_detected"] for row in parent_rows),
        "candidate_generation_errors": sum(row["generation_error"] is not None for row in candidate_rows),
        "parent_generation_errors": sum(row["generation_error"] is not None for row in parent_rows),
        "candidate_terminated": sum(row["terminated"] for row in candidate_rows),
        "parent_terminated": sum(row["terminated"] for row in parent_rows),
        "candidate_physical_sparse_rows": sum(
            row["maximum_active_decode_experts_per_token"] == 1 for row in candidate_rows
        ),
        "training_expert_assignments": assignment_counts.tolist(),
        "training_experts_used": sum(value > 0 for value in assignment_counts.tolist()),
    }
    gates = {
        "matrix_complete": all(len(values) == 280 for values in by_system.values()),
        "train_evaluation_disjoint": not any(overlaps.values()),
        "candidate_overall_functional": metrics["functional_rate"]["candidate"] >= 0.50,
        "candidate_per_capability": all(by_capability[value]["candidate"] >= 6 for value in CAPABILITIES),
        "candidate_parent_improvement": metrics["candidate_minus_parent"]["point"] >= 0.20,
        "candidate_source_gap": metrics["candidate_minus_source"]["point"] >= -0.25,
        "candidate_utf8": metrics["candidate_generation_errors"] == 0,
        "candidate_noncollapse": metrics["candidate_collapses"] <= 2,
        "all_sparse_experts_trained": metrics["training_experts_used"] == 8,
        "physical_top1_decode": metrics["candidate_physical_sparse_rows"] == 280,
        "frozen_inputs_unchanged": (
            sha256_file(training_path) == TRAINING_SHA256
            and sha256_file(evaluation_path) == EVALUATION_SHA256
            and sha256_file(parent_model) == initial_state_hash == PARENT_SHA256
            and sha256_file(parent_tokenizer) == TOKENIZER_SHA256
        ),
    }
    passed = all(gates.values())
    result = {
        "format": "abi-r46-causal-english-transfer-feasibility/1",
        "verdict": "PASS_R46_CAUSAL_ENGLISH_TRANSFER_FEASIBILITY" if passed else "FAIL_R46_CAUSAL_ENGLISH_TRANSFER_FEASIBILITY",
        "inputs": {
            "training_archive_sha256": TRAINING_SHA256,
            "evaluation_archive_sha256": EVALUATION_SHA256,
            "parent_checkpoint_sha256": PARENT_SHA256,
            "parent_tokenizer_sha256": TOKENIZER_SHA256,
            "training_evaluation_overlap": overlaps,
        },
        "model": {
            "architecture": metadata["architecture"],
            "parameters": metadata["parameters"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "planner_called_during_generation": False,
            "generation_mode": "autonomous_neural_greedy",
        },
        "training": {
            "seed": SEED,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "examples_seen": STEPS * BATCH_SIZE,
            "record_sequence_sha256": sequence_digest.hexdigest(),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "balance_weight": BALANCE_WEIGHT,
            "gradient_clip": GRADIENT_CLIP,
            "parameters_updated": sum(int(value) for value in metadata["parameters"].values() if False),
            "all_model_parameters_trainable": True,
            "source_records_available": len(training["records"]),
            "source_records_context_eligible": sum(len(values) for values in prepared.values()),
            "source_records_context_excluded": len(excluded),
            "prepared_response_tokens": prepared_response_tokens,
            "wall_seconds": training_seconds,
            "curves": curves,
        },
        "information_accounting": {
            "source_model": training["records"][0]["source_model"],
            "source_model_revision": training["records"][0]["source_model_revision"],
            "teacher_records": len(training["records"]),
            "teacher_tokens_authoritative": sum(int(row["teacher_tokens"]) for row in training["records"]),
            "teacher_output_bytes": sum(int(row["output_utf8_bytes"]) for row in training["records"]),
            "raw_prompt_bytes": sum(int(row["prompt_utf8_bytes"]) for row in training["records"]),
            "stored_logits": 0,
            "stored_hidden_activations": 0,
            "source_parameters_copied": 0,
            "source_transformer_blocks_retained": 0,
            "teacher_present_at_inference": False,
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
        },
        "metrics": metrics,
        "gates": gates,
        "artifacts": {
            "evaluation": {"path": raw_path.name, "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size},
            "excluded_training_rows": {"path": excluded_path.name, "sha256": sha256_file(excluded_path), "bytes": excluded_path.stat().st_size},
            "training_trace": {"path": trace_path.name, "sha256": sha256_file(trace_path), "bytes": trace_path.stat().st_size},
            "checkpoint": {"path": checkpoint_path.name, "sha256": sha256_file(checkpoint_path), "bytes": checkpoint_path.stat().st_size},
        },
        "claim_ceiling": "CAUSAL_ENGLISH_TRANSFER_FEASIBILITY_NOT_PROMOTION_OR_MINIMALITY",
        "full_abi_moonshot": "OPEN",
    }
    result["training"]["parameters_updated"] = int(metadata["parameters"]["total"])
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--training-archive", type=Path, required=True)
    parser.add_argument("--evaluation-archive", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.layercake_root.resolve(),
        args.training_archive.resolve(),
        args.evaluation_archive.resolve(),
        args.parent.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
