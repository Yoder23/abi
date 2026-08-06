"""Train and screen one exact-BPE LayerCake v3 English-core candidate."""
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
from .capability_compiler_phase3 import (
    Phase3Error,
    _BalancedSampler,
    _write_immutable,
    load_phase1_ir,
)
from .capability_compiler_phase3_direct_core import _json


FORMAT = "abi-capability-compiler-phase3-bpe-core-screen/1"


def _layercake_api(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.portable_token_plan import EOS_ID, PortableTokenPlan
    from layercake_extensions.bpe_direct_neural_core import (
        BPE_DIRECT_NEURAL_CORE_ABI_SHA256,
        BPE_DIRECT_NEURAL_CORE_ABI_VERSION,
        Utf8ConcatenativeBpeTokenizer,
    )
    return (
        EOS_ID,
        PortableTokenPlan,
        Utf8ConcatenativeBpeTokenizer,
        BPE_DIRECT_NEURAL_CORE_ABI_VERSION,
        BPE_DIRECT_NEURAL_CORE_ABI_SHA256,
    )


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_CONDITIONAL_ABSOLUTE_SCREEN"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("promotion_eligible") is not False
        or protocol.get("controls_deferred_until_absolute_pass") is not True
        or protocol.get("training", {}).get("device") != "cuda"
        or protocol.get("representation", {}).get("target_actions") != "FIXED_BPE_PIECES_PLUS_EOS"
        or protocol.get("representation", {}).get("pointer_supervision") is not False
    ):
        raise Phase3Error("BPE-core screen governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"BPE-core binding changed: {relative}")
    return protocol, sha256_file(path)


def _tokenizer(root: Path, protocol: Mapping[str, Any], tokenizer_type):
    path = (root / protocol["tokenizer"]["canonical_document"]).resolve()
    tokenizer = tokenizer_type.from_document(_json(path))
    if (
        tokenizer.hash() != protocol["tokenizer"]["canonical_sha256"]
        or tokenizer.vocab_size != int(protocol["tokenizer"]["fixed_actions"])
    ):
        raise Phase3Error("BPE tokenizer identity changed")
    return tokenizer


def _examples(
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
        source_ids = [tokenizer.lexeme_to_id[value] for value in source_lexemes]
        target_actions = [tokenizer.lexeme_to_id[value] for value in output_lexemes] + [eos_id]
        if (
            len(source_ids) > int(config["maximum_source_lexemes"])
            or len(target_actions) > int(config["maximum_target_actions"])
        ):
            raise Phase3Error(f"BPE-core example exceeds bound: {row['ir_record_id']}")
        if tokenizer.decode_actions(target_actions, source_lexemes) != output.encode("utf-8"):
            raise Phase3Error(f"BPE-core target is not lossless: {row['ir_record_id']}")
        examples.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "source_ids": source_ids,
                "source_lexemes": source_lexemes,
                "target_actions": target_actions,
                "output_bytes": output.encode("utf-8"),
                "teacher_tokens": int(row["authoritative_teacher_tokens"]),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "output_bytes_count": len(output.encode("utf-8")),
            }
        )
    if len(examples) != 7000 or len({row["record_id"] for row in examples}) != 7000:
        raise Phase3Error("BPE-core training inventory changed")
    return examples


def _model(protocol: Mapping[str, Any], tokenizer: Any, model_type):
    return model_type(
        fixed_vocab_size=tokenizer.vocab_size,
        **protocol["architecture"],
    ).bind_tokenizer(tokenizer)


def _collate(selected: list[Mapping[str, Any]], device: torch.device):
    source_width = max(len(row["source_ids"]) for row in selected)
    target_width = max(len(row["target_actions"]) for row in selected)
    source = torch.zeros((len(selected), source_width), dtype=torch.long, device=device)
    targets = torch.full((len(selected), target_width), -100, dtype=torch.long, device=device)
    for index, row in enumerate(selected):
        source[index, : len(row["source_ids"])] = torch.tensor(row["source_ids"], device=device)
        targets[index, : len(row["target_actions"])] = torch.tensor(row["target_actions"], device=device)
    return source, targets


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    eos_id, model_type, tokenizer_type, abi_version, abi_sha = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    examples = _examples(
        load_phase1_ir((root / protocol["phase1_ir"]).resolve()),
        tokenizer,
        eos_id,
        protocol["architecture"],
    )
    model = _model(protocol, tokenizer, model_type)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error(f"BPE-core parameter count changed: {parameters}")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "layercake_interface": abi_version,
        "layercake_interface_sha256": abi_sha,
        "records": len(examples),
        "vocabulary": tokenizer.vocab_size,
        "trainable_parameters": parameters,
        "maximum_source_actions": max(len(row["source_ids"]) for row in examples),
        "maximum_target_actions": max(len(row["target_actions"]) for row in examples),
        "mean_source_actions": sum(len(row["source_ids"]) for row in examples) / len(examples),
        "mean_target_actions": sum(len(row["target_actions"]) for row in examples) / len(examples),
        "all_targets_losslessly_reconstructed": True,
        "pointer_supervision": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_dir.exists() or not torch.cuda.is_available():
        raise Phase3Error("BPE-core output exists or CUDA unavailable")
    eos_id, model_type, tokenizer_type, abi_version, abi_sha = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    examples = _examples(
        load_phase1_ir((root / protocol["phase1_ir"]).resolve()),
        tokenizer,
        eos_id,
        protocol["architecture"],
    )
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model = _model(protocol, tokenizer, model_type).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != int(cfg["trainable_parameters"]):
        raise Phase3Error("BPE-core trainable parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = _BalancedSampler(examples, seed)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    successful = skipped = 0
    sampled: Counter[str] = Counter()
    sequence_hash = hashlib.sha256()
    curves: list[dict[str, Any]] = []
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
                loss = F.nll_loss(
                    log_probs.float().reshape(-1, log_probs.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
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
            curve = {
                "step": successful,
                "loss": float(loss.detach()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "model.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        str(checkpoint),
    )
    tokenizer_path = output_dir / "tokenizer.json"
    _write_immutable(
        tokenizer_path,
        json.dumps(tokenizer.canonical_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    config_path = output_dir / "model_config.json"
    model_config = {**protocol["architecture"], "fixed_vocab_size": tokenizer.vocab_size}
    _write_immutable(
        config_path,
        json.dumps(model_config, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    metadata: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-bpe-core-candidate/1",
        "status": "TRAINED_CONDITIONAL_DEVELOPMENT_SCREEN",
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "checkpoint": {
            "path": "model.safetensors",
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "tokenizer": {
            "path": "tokenizer.json",
            "sha256": sha256_file(tokenizer_path),
            "canonical_sha256": tokenizer.hash(),
            "vocabulary": tokenizer.vocab_size,
        },
        "model_config": {
            "path": "model_config.json",
            "sha256": sha256_file(config_path),
            "trainable_parameters": parameter_count,
        },
        "layercake_interface": {"version": abi_version, "sha256": abi_sha},
        "imported_information": {
            "records": len(examples),
            "raw_prompt_bytes": sum(row["prompt_bytes"] for row in examples),
            "teacher_output_bytes": sum(row["output_bytes_count"] for row in examples),
            "authoritative_teacher_tokens": sum(row["teacher_tokens"] for row in examples),
            "stored_logits": 0,
            "stored_activations": 0,
            "source_parameters_copied": 0,
        },
        "representation": {
            "fixed_bpe_actions": sum(len(row["target_actions"]) - 1 for row in examples),
            "pointer_supervision": False,
            "all_targets_losslessly_reconstructed": True,
        },
        "training": {
            "steps": successful,
            "batch_size": int(cfg["batch_size"]),
            "wall_seconds": time.perf_counter() - started,
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
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    metadata["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return metadata


def _load_candidate(
    root: Path,
    protocol: Mapping[str, Any],
    candidate_dir: Path,
    device: torch.device,
):
    _, model_type, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = tokenizer_type.from_document(_json(candidate_dir / "tokenizer.json"))
    model = model_type(**_json(candidate_dir / "model_config.json")).bind_tokenizer(tokenizer)
    model.load_state_dict(
        load_file(str(candidate_dir / "model.safetensors"), device=str(device)),
        strict=True,
    )
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def fit(
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    metadata = _json(candidate_dir / "metadata.json")
    if output_dir.exists() or metadata.get("protocol_sha256") != protocol_sha:
        raise Phase3Error("BPE-core fit identity failed")
    eos_id, _, tokenizer_type, _, _ = _layercake_api(root, protocol)
    model, tokenizer = _load_candidate(root, protocol, candidate_dir, torch.device("cuda"))
    examples = _examples(
        load_phase1_ir((root / protocol["phase1_ir"]).resolve()),
        tokenizer,
        eos_id,
        protocol["architecture"],
    )
    raw_rows: list[dict[str, Any]] = []
    batch_size = int(protocol["fit_diagnostic"]["batch_size"])
    started = time.perf_counter()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        source, targets = _collate(batch, torch.device("cuda"))
        log_probs = model(source, targets)["log_probs"].float()
        mask = targets.ge(0)
        safe = targets.clamp(min=0)
        chosen = log_probs.gather(-1, safe[:, :, None]).squeeze(-1)
        predicted = log_probs.argmax(dim=-1)
        correct = predicted.eq(targets) & mask
        for index, row in enumerate(batch):
            actions = int(mask[index].sum().item())
            right = int(correct[index].sum().item())
            raw_rows.append(
                {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "actions": actions,
                    "correct_actions": right,
                    "exact_sequence": right == actions,
                    "action_nll_sum": float((-chosen[index].masked_select(mask[index])).sum().item()),
                }
            )
    output_dir.mkdir(parents=True)
    raw_path = output_dir / "training_fit_rows.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in raw_rows))
    actions = sum(row["actions"] for row in raw_rows)
    correct = sum(row["correct_actions"] for row in raw_rows)
    exact = sum(bool(row["exact_sequence"]) for row in raw_rows)
    receipt = {
        "format": "abi-capability-compiler-phase3-bpe-core-fit/1",
        "status": "PASS_EXECUTION_DIAGNOSTIC_ONLY",
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "records": len(raw_rows),
        "actions": actions,
        "correct_actions": correct,
        "action_accuracy": correct / actions,
        "exact_sequences": exact,
        "exact_sequence_rate": exact / len(raw_rows),
        "mean_action_nll": sum(row["action_nll_sum"] for row in raw_rows) / actions,
        "rows_sha256": sha256_file(raw_path),
        "wall_seconds": time.perf_counter() - started,
        "promotion_effect": "NONE_WITHOUT_ABSOLUTE_QUALITY_PASS",
        "final_test_accessed": False,
    }
    _write_immutable(
        output_dir / "receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return receipt


def evaluate(
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    metadata = _json(candidate_dir / "metadata.json")
    if (
        output_dir.exists()
        or metadata.get("protocol_sha256") != protocol_sha
        or sha256_file(candidate_dir / "model.safetensors") != metadata.get("checkpoint", {}).get("sha256")
    ):
        raise Phase3Error("BPE-core evaluation identity failed")
    model, _ = _load_candidate(root, protocol, candidate_dir, torch.device("cuda"))
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        error = None
        try:
            raw = model.generate_bytes(
                str(probe["prompt"]),
                maximum_actions=min(
                    int(probe["max_new_tokens"]),
                    int(protocol["architecture"]["maximum_target_actions"]),
                ),
            )
            output = raw.decode("utf-8", errors="strict")
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
    grouped = {
        capability: [row for row in rows if row["capability"] == capability]
        for capability in CAPABILITIES
    }
    receipt = {
        "format": "abi-capability-compiler-phase3-bpe-core-evaluation/1",
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
    _write_immutable(
        output_dir / "receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_BPE_CORE_PROTOCOL_V38.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    for name in ("train", "fit", "evaluate"):
        child = sub.add_parser(name)
        child.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_bpe_core/development_v38/B0-seed240017")
        if name in {"fit", "evaluate"}:
            child.add_argument(
                "--output-dir",
                default=f"results/abi_capability_compiler_phase3_bpe_core/{'fit_v38' if name == 'fit' else 'evaluation_v38'}/B0-seed240017",
            )
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = (root / args.protocol).resolve()
    if args.command == "inventory":
        result = inventory(root, protocol_path)
    elif args.command == "train":
        result = train(root, protocol_path, (root / args.candidate_dir).resolve())
    elif args.command == "fit":
        result = fit(root, protocol_path, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    else:
        result = evaluate(root, protocol_path, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
