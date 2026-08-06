"""Train one V24 direct-core representation with supervised prompt pointers."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import time
from typing import Any, Iterable, Mapping

import psutil
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    repetition_collapse,
    set_determinism,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_direct_core import (
    _collate,
    _fixed_tokenizer,
    _json,
    _layercake_api,
    _load_candidate,
    _model,
)


FORMAT = "abi-capability-compiler-phase3-pointer-core-screen/1"
IDENTITY_LEXEME_PATTERN = rb"^(?:[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?)$"
IDENTITY_LEXEME_REGEX = re.compile(IDENTITY_LEXEME_PATTERN)


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    policy = protocol.get("pointer_supervision", {})
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_CONDITIONAL_ABSOLUTE_SCREEN"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("promotion_eligible") is not False
        or protocol.get("controls_deferred_until_absolute_pass") is not True
        or protocol.get("training", {}).get("device") != "cuda"
        or policy.get("eligible_lexeme_pattern_ascii") != IDENTITY_LEXEME_PATTERN.decode("ascii")
        or policy.get("source_occurrences_required") != 1
        or policy.get("target_occurrences_minimum") != 1
        or policy.get("punctuation_and_whitespace") != "FIXED_ACTIONS"
    ):
        raise Phase3Error("pointer-core screen governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"pointer-core binding changed: {relative}")
    return protocol, sha256_file(path)


def _copy_lexemes(source_lexemes: list[bytes], output_lexemes: list[bytes]) -> list[bytes]:
    source_counts = Counter(source_lexemes)
    output_set = set(output_lexemes)
    return [
        lexeme
        for lexeme in source_lexemes
        if source_counts[lexeme] == 1
        and lexeme in output_set
        and IDENTITY_LEXEME_REGEX.fullmatch(lexeme) is not None
    ]


def _pointer_examples(
    rows: list[Mapping[str, Any]],
    tokenizer: Any,
    eos_id: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["normalized_acquisition_prompt"])
        output = str(row["normalized_output"])
        source_lexemes = tokenizer.split(prompt)
        output_lexemes = tokenizer.split(output)
        source = [tokenizer.lexeme_to_id.get(value, 3) for value in source_lexemes]
        copies = _copy_lexemes(source_lexemes, output_lexemes)
        targets = tokenizer.encode_target(
            output,
            copy_lexemes=[value.decode("ascii") for value in copies],
            source_lexemes=source_lexemes,
        )
        if len(source) > int(config["maximum_source_lexemes"]) or len(targets) > int(config["maximum_target_actions"]):
            raise Phase3Error(f"pointer-core example exceeds bound: {row['ir_record_id']}")
        if tokenizer.decode_actions(targets, source_lexemes) != output.encode("utf-8"):
            raise Phase3Error(f"pointer-core target is not lossless: {row['ir_record_id']}")
        pointer_actions = sum(action >= tokenizer.vocab_size for action in targets)
        examples.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "source_ids": source,
                "target_actions": targets,
                "pointer_actions": pointer_actions,
                "copy_lexemes": len(copies),
                "teacher_tokens": int(row["authoritative_teacher_tokens"]),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "output_bytes": len(output.encode("utf-8")),
            }
        )
    if len(examples) != 7000 or len({row["record_id"] for row in examples}) != 7000:
        raise Phase3Error("pointer-core training inventory changed")
    return examples


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    eos_id, format_version, tokenizer_type, model_type = _layercake_api(root, protocol)
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    tokenizer = _fixed_tokenizer(rows, tokenizer_type, format_version)
    examples = _pointer_examples(rows, tokenizer, eos_id, protocol["architecture"])
    model = _model(protocol, tokenizer, model_type)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error(f"pointer-core parameter count changed: {parameters}")
    per_capability = {
        capability: sum(row["pointer_actions"] for row in examples if row["capability"] == capability)
        for capability in CAPABILITIES
    }
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "records": len(examples),
        "vocabulary": tokenizer.vocab_size,
        "trainable_parameters": parameters,
        "maximum_source_lexemes": max(len(row["source_ids"]) for row in examples),
        "maximum_target_actions": max(len(row["target_actions"]) for row in examples),
        "pointer_actions": sum(row["pointer_actions"] for row in examples),
        "records_with_pointer_actions": sum(row["pointer_actions"] > 0 for row in examples),
        "pointer_actions_by_capability": per_capability,
        "all_targets_losslessly_reconstructed": True,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_dir.exists() or not torch.cuda.is_available():
        raise Phase3Error("pointer-core output exists or CUDA unavailable")
    eos_id, format_version, tokenizer_type, model_type = _layercake_api(root, protocol)
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    tokenizer = _fixed_tokenizer(rows, tokenizer_type, format_version)
    examples = _pointer_examples(rows, tokenizer, eos_id, protocol["architecture"])
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model = _model(protocol, tokenizer, model_type).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != int(cfg["trainable_parameters"]):
        raise Phase3Error("pointer-core trainable parameter count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = _BalancedSampler(examples, seed)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    successful = skipped = 0
    sampled = Counter()
    sequence_hash = hashlib.sha256()
    curves = []
    started = time.perf_counter()
    model.train()
    while successful < int(cfg["steps"]):
        selected = sampler.batch(int(cfg["batch_size"]))
        while True:
            source, targets = _collate(selected, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(source, targets)
                log_probs = result["log_probs"]
                loss = F.nll_loss(log_probs.float().reshape(-1, log_probs.shape[-1]), targets.reshape(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1
                continue
            break
        successful += 1
        for row in selected:
            sampled[row["capability"]] += 1
            sequence_hash.update(row["record_id"].encode("ascii") + b"\n")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curve = {"step": successful, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - started}
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    output_dir.mkdir(parents=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    checkpoint = output_dir / "model.safetensors"
    save_file(state, str(checkpoint))
    tokenizer_path = output_dir / "tokenizer.json"
    _write_immutable(tokenizer_path, json.dumps(tokenizer.canonical_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n")
    model_config = {**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size}
    config_path = output_dir / "model_config.json"
    _write_immutable(config_path, json.dumps(model_config, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    wall = time.perf_counter() - started
    metadata = {
        "format": "abi-capability-compiler-phase3-pointer-core-candidate/1",
        "status": "TRAINED_CONDITIONAL_DEVELOPMENT_SCREEN",
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "checkpoint": {"path": "model.safetensors", "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "tokenizer": {"path": "tokenizer.json", "sha256": sha256_file(tokenizer_path), "vocabulary": tokenizer.vocab_size},
        "model_config": {"path": "model_config.json", "sha256": sha256_file(config_path), "trainable_parameters": parameter_count},
        "imported_information": {
            "records": len(examples),
            "raw_prompt_bytes": sum(row["prompt_bytes"] for row in examples),
            "teacher_output_bytes": sum(row["output_bytes"] for row in examples),
            "authoritative_teacher_tokens": sum(row["teacher_tokens"] for row in examples),
            "stored_logits": 0,
            "stored_activations": 0,
            "source_parameters_copied": 0,
        },
        "representation": {
            "pointer_actions": sum(row["pointer_actions"] for row in examples),
            "records_with_pointer_actions": sum(row["pointer_actions"] > 0 for row in examples),
            "all_targets_losslessly_reconstructed": True,
            "eligible_lexeme_pattern_ascii": IDENTITY_LEXEME_PATTERN.decode("ascii"),
        },
        "training": {
            "steps": successful,
            "batch_size": int(cfg["batch_size"]),
            "wall_seconds": wall,
            "skipped_amp_steps": skipped,
            "record_sequence_sha256": sequence_hash.hexdigest(),
            "sampled_by_capability": dict(sorted(sampled.items())),
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "curves": curves,
        },
        "teacher_present_at_inference": False,
        "source_blocks_retained": 0,
        "development_only": True,
        "promotion_eligible": False,
        "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    metadata["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output_dir / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return metadata


def evaluate(root: Path, protocol_path: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    metadata = _json(candidate_dir / "metadata.json")
    if output_dir.exists() or metadata.get("protocol_sha256") != protocol_sha or sha256_file(candidate_dir / "model.safetensors") != metadata.get("checkpoint", {}).get("sha256"):
        raise Phase3Error("pointer-core evaluation identity or immutability failed")
    model, _ = _load_candidate(root, protocol, candidate_dir, torch.device("cuda"))
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        error = None
        try:
            raw = model.generate_bytes(str(probe["prompt"]), maximum_actions=min(int(probe["max_new_tokens"]), int(protocol["architecture"]["maximum_target_actions"])))
            output = raw.decode("utf-8")
        except Exception as exc:
            output = ""
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "probe_id": str(probe["probe_id"]),
                "capability": str(probe["canonical_capability"]),
                "output": output,
                "generation_error": error,
                "functional_pass": evaluate_functional(output, probe["evaluator"]),
                "repetition_collapse": repetition_collapse(output),
            }
        )
        if (index + 1) % 100 == 0:
            print(json.dumps({"evaluated": index + 1}), flush=True)
    output_dir.mkdir(parents=True)
    outputs = output_dir / "development_outputs.jsonl"
    outputs.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    grouped = {capability: [row for row in rows if row["capability"] == capability] for capability in CAPABILITIES}
    receipt = {
        "format": "abi-capability-compiler-phase3-pointer-core-evaluation/1",
        "status": "PASS_EXECUTION_CONDITIONAL_SCREEN",
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "observations": len(rows),
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "generation_errors": sum(row["generation_error"] is not None for row in rows),
        "per_capability": {
            capability: {
                "passes": sum(bool(row["functional_pass"]) for row in values),
                "collapses": sum(bool(row["repetition_collapse"]) for row in values),
                "observations": len(values),
            }
            for capability, values in grouped.items()
        },
        "outputs_sha256": sha256_file(outputs),
        "wall_seconds": time.perf_counter() - started,
        "promotion_eligible": False,
        "final_test_accessed": False,
    }
    _write_immutable(output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_POINTER_CORE_PROTOCOL_V24.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--output-dir", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--candidate-dir", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    if args.command == "inventory":
        result = inventory(root, protocol)
    elif args.command == "train":
        result = train(root, protocol, (root / args.output_dir).resolve())
    else:
        result = evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
