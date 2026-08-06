"""Train and screen one LayerCake-native self-causal direct English-core artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping

import psutil
from safetensors.torch import load_file, save_file
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


FORMAT = "abi-capability-compiler-phase3-direct-core-screen/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected object: {path}")
    return value


def _layercake_api(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.portable_token_plan import (
        EOS_ID,
        GENERIC_TOKENIZER_FORMAT,
        LosslessLexemePointerTokenizer,
        PortableTokenPlan,
    )
    return EOS_ID, GENERIC_TOKENIZER_FORMAT, LosslessLexemePointerTokenizer, PortableTokenPlan


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_CONDITIONAL_ABSOLUTE_SCREEN"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("promotion_eligible") is not False
        or protocol.get("controls_deferred_until_absolute_pass") is not True
        or protocol.get("training", {}).get("device") != "cuda"
    ):
        raise Phase3Error("direct-core screen governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"direct-core binding changed: {relative}")
    return protocol, sha256_file(path)


def _fixed_tokenizer(rows: list[Mapping[str, Any]], tokenizer_type, format_version: str):
    lexemes: set[bytes] = set()
    for row in rows:
        lexemes.update(tokenizer_type.split(str(row["normalized_acquisition_prompt"])))
        lexemes.update(tokenizer_type.split(str(row["normalized_output"])))
    return tokenizer_type(sorted(lexemes), format_version=format_version)


def _examples(rows: list[Mapping[str, Any]], tokenizer: Any, eos_id: int, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        prompt = str(row["normalized_acquisition_prompt"])
        output = str(row["normalized_output"])
        source_lexemes = tokenizer.split(prompt)
        output_lexemes = tokenizer.split(output)
        source = [tokenizer.lexeme_to_id.get(value, 3) for value in source_lexemes]
        targets = [tokenizer.lexeme_to_id[value] for value in output_lexemes] + [eos_id]
        if len(source) > int(config["maximum_source_lexemes"]) or len(targets) > int(config["maximum_target_actions"]):
            raise Phase3Error(f"direct-core example exceeds bound: {row['ir_record_id']}")
        examples.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "source_ids": source,
                "target_actions": targets,
                "teacher_tokens": int(row["authoritative_teacher_tokens"]),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "output_bytes": len(output.encode("utf-8")),
            }
        )
    if len(examples) != 7000 or len({row["record_id"] for row in examples}) != 7000:
        raise Phase3Error("direct-core training inventory changed")
    return examples


def _model(protocol: Mapping[str, Any], tokenizer: Any, model_type):
    config = dict(protocol["architecture"])
    config["fixed_vocab_size"] = int(tokenizer.vocab_size)
    return model_type(**config).bind_tokenizer(tokenizer)


def _collate(selected: list[Mapping[str, Any]], device: torch.device):
    source_width = max(len(row["source_ids"]) for row in selected)
    target_width = max(len(row["target_actions"]) for row in selected)
    source = torch.zeros((len(selected), source_width), dtype=torch.long, device=device)
    targets = torch.full((len(selected), target_width), -100, dtype=torch.long, device=device)
    for index, row in enumerate(selected):
        source[index, : len(row["source_ids"])] = torch.tensor(row["source_ids"], dtype=torch.long, device=device)
        targets[index, : len(row["target_actions"])] = torch.tensor(row["target_actions"], dtype=torch.long, device=device)
    return source, targets


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    eos_id, format_version, tokenizer_type, model_type = _layercake_api(root, protocol)
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    tokenizer = _fixed_tokenizer(rows, tokenizer_type, format_version)
    examples = _examples(rows, tokenizer, eos_id, protocol["architecture"])
    model = _model(protocol, tokenizer, model_type)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error(f"direct-core parameter count changed: {parameters}")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "records": len(examples),
        "vocabulary": tokenizer.vocab_size,
        "trainable_parameters": parameters,
        "maximum_source_lexemes": max(len(row["source_ids"]) for row in examples),
        "maximum_target_actions": max(len(row["target_actions"]) for row in examples),
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_dir.exists() or not torch.cuda.is_available():
        raise Phase3Error("direct-core output exists or CUDA unavailable")
    eos_id, format_version, tokenizer_type, model_type = _layercake_api(root, protocol)
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    tokenizer = _fixed_tokenizer(rows, tokenizer_type, format_version)
    examples = _examples(rows, tokenizer, eos_id, protocol["architecture"])
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model = _model(protocol, tokenizer, model_type).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != int(cfg["trainable_parameters"]):
        raise Phase3Error("direct-core trainable parameter count changed")
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
        "format": "abi-capability-compiler-phase3-direct-core-candidate/1",
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


def _load_candidate(root: Path, protocol: Mapping[str, Any], candidate_dir: Path, device: torch.device):
    _, _, tokenizer_type, model_type = _layercake_api(root, protocol)
    tokenizer = tokenizer_type.from_document(_json(candidate_dir / "tokenizer.json"))
    config = _json(candidate_dir / "model_config.json")
    model = model_type(**config).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(candidate_dir / "model.safetensors"), device=str(device)), strict=True)
    return model.to(device).eval(), tokenizer


def evaluate(root: Path, protocol_path: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    metadata = _json(candidate_dir / "metadata.json")
    if output_dir.exists() or metadata.get("protocol_sha256") != protocol_sha or sha256_file(candidate_dir / "model.safetensors") != metadata.get("checkpoint", {}).get("sha256"):
        raise Phase3Error("direct-core evaluation identity or immutability failed")
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
        "format": "abi-capability-compiler-phase3-direct-core-evaluation/1",
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
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_CORE_PROTOCOL_V23.json")
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
