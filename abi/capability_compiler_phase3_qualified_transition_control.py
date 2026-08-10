"""Bounded end-to-end capacity control on the qualified LayerCake transition."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Iterable

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
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_sequence_bridge import (
    _BalancedSampler,
    _batch,
    _examples,
    _generate,
)
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
    wilson,
)
from .layercake_core_loader import load_layercake_core


FORMAT = "abi-capability-compiler-phase3-qualified-transition-control/1"
TRAINABLE_PREFIXES = (
    "transformer.h.",
    "transformer.ln_f.",
    "task_classifier.",
    "task_cakes.",
)
FROZEN_PREFIXES = ("transformer.wte.", "transformer.wpe.")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_hash(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_protocol(root: Path, protocol_path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_FAIL_FAST_DEVELOPMENT_CAPACITY_CONTROL"
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("promotion_eligible") is not False
        or protocol.get("nearby_sweeps_authorized") is not False
    ):
        raise Phase3Error("qualified transition governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"qualified transition binding changed: {name}")
    return protocol, sha256_file(protocol_path)


def _load_parent(root: Path, protocol: dict[str, Any], device: torch.device):
    host = protocol["host"]
    parent = (root / host["parent_path"]).resolve()
    if sha256_file(parent / "model.safetensors") != host["parent_checkpoint_sha256"]:
        raise Phase3Error("qualified LayerCake parent changed")
    model, tokenizer, metadata = load_layercake_core(
        parent,
        layercake_root=(root / host["layercake_root"]).resolve(),
        device=device,
    )
    return parent, model, tokenizer, metadata


def _configure_trainable(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    trainable = []
    for name, parameter in model.named_parameters():
        selected = name.startswith(TRAINABLE_PREFIXES)
        if not selected and not name.startswith(FROZEN_PREFIXES):
            raise Phase3Error(f"unclassified carrier tensor: {name}")
        parameter.requires_grad_(selected)
        if selected:
            trainable.append(parameter)
    return trainable


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    _, model, _, metadata = _load_parent(root, protocol, torch.device("cpu"))
    trainable = _configure_trainable(model)
    count = sum(value.numel() for value in trainable)
    if count != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error("transition trainable count changed")
    return {
        "status": "PASS_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "architecture": metadata["architecture"]["architecture_version"],
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": count,
        "new_parameters": 0,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable transition output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("qualified transition CUDA device unavailable")
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    parent, model, tokenizer, metadata = _load_parent(root, protocol, device)
    rows = load_phase1_ir((root / protocol["phase1_ir"]["path"]).resolve())
    examples = _examples(rows, tokenizer, system="A0", seed=seed, max_tokens=int(cfg["max_tokens"]))
    trainable = _configure_trainable(model)
    trainable_count = sum(value.numel() for value in trainable)
    if trainable_count != int(cfg["trainable_parameters"]):
        raise Phase3Error("transition trainable boundary changed")
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    frozen_before = {name: value for name, value in before.items() if name.startswith(FROZEN_PREFIXES)}
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = _BalancedSampler(examples, seed)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    successful = 0
    skipped = 0
    language_tokens = 0
    sampled = Counter()
    sequence_sha = hashlib.sha256()
    curves = []
    started = time.perf_counter()
    model.train()
    while successful < int(cfg["steps"]):
        selected = sampler.batch(int(cfg["batch_size"]))
        while True:
            ids, labels, attention, prompt_lengths, routes = _batch(
                selected, int(tokenizer.eos_token_id), device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(
                    ids,
                    attention_mask=attention,
                    prompt_lengths=prompt_lengths,
                    task_routes=routes,
                    use_cache=False,
                )
                language_loss = F.cross_entropy(
                    result["logits"][:, :-1].float().reshape(-1, result["logits"].shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                classifier_loss = F.cross_entropy(result["task_logits"].float(), routes)
                loss = language_loss + float(cfg["classifier_loss_weight"]) * classifier_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg["gradient_clip_norm"]))
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale:
                skipped += 1
                continue
            break
        successful += 1
        for row in selected:
            sequence_sha.update(str(row["record_id"]).encode("ascii") + b"\n")
            language_tokens += int(row["response_tokens"])
            sampled[str(row["capability"])] += 1
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curve = {
                "step": successful,
                "language_loss": float(language_loss.detach()),
                "classifier_loss": float(classifier_loss.detach()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    model.eval()
    after = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    changed = sorted(name for name in before if not torch.equal(before[name], after[name]))
    if not changed or any(not name.startswith(TRAINABLE_PREFIXES) for name in changed):
        raise Phase3Error("transition control changed a frozen or no tensor")
    frozen_after = {name: value for name, value in after.items() if name.startswith(FROZEN_PREFIXES)}
    if _state_hash(frozen_before) != _state_hash(frozen_after):
        raise Phase3Error("embedding substrate changed")
    output.mkdir(parents=True)
    checkpoint = output / "model.safetensors"
    save_file(after, str(checkpoint), metadata={"format": FORMAT})
    for name in TOKENIZER_FILES:
        shutil.copyfile(parent / name, output / name)
    wall = time.perf_counter() - started
    document = {
        "format": FORMAT,
        "status": "TRAINED_DEVELOPMENT_ONLY_NONPROMOTIONAL",
        "protocol_sha256": protocol_sha,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parent_checkpoint_sha256": protocol["host"]["parent_checkpoint_sha256"],
        "architecture": metadata["architecture"],
        "training": {
            "seed": seed,
            "device": "cuda",
            "steps": successful,
            "batch_size": int(cfg["batch_size"]),
            "teacher_response_tokens_seen": language_tokens,
            "trainable_parameters": trainable_count,
            "new_parameters": 0,
            "successful_record_sequence_sha256": sequence_sha.hexdigest(),
            "sampled_records_by_capability": dict(sorted(sampled.items())),
            "skipped_amp_steps": skipped,
            "wall_seconds": wall,
            "active_parameter_seconds": trainable_count * wall,
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "curves": curves,
        },
        "isolation": {
            "changed_tensor_count": len(changed),
            "all_changes_confined_to_existing_transition": True,
            "embedding_state_sha256_before": _state_hash(frozen_before),
            "embedding_state_sha256_after": _state_hash(frozen_after),
            "layercake_repository_mutated": False,
        },
        "source": {
            "teacher_present_during_training": False,
            "teacher_present_at_inference": False,
            "source_parameters_copied": 0,
            "source_blocks_retained": 0,
        },
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
    }
    document["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(document, indent=2, sort_keys=True).encode() + b"\n")
    return document


def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable evaluation output exists: {output}")
    metadata = _json(candidate / "metadata.json")
    if metadata.get("protocol_sha256") != protocol_sha or metadata.get("promotion_eligible") is not False:
        raise Phase3Error("transition candidate lineage changed")
    if sha256_file(candidate / "model.safetensors") != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("transition checkpoint changed")
    _, model, tokenizer, _ = _load_parent(root, protocol, torch.device("cuda"))
    model.load_state_dict(load_file(str(candidate / "model.safetensors"), device="cuda"), strict=True)
    model.eval()
    probes = development_probes((root / protocol["development"]["catalog_path"]).resolve())
    teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["development"]["teacher_reference"]).open(encoding="utf-8"))}
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        value, tokens, route = _generate(model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), torch.device("cuda"))
        rows.append({
            "probe_id": str(probe["probe_id"]),
            "capability": str(probe["canonical_capability"]),
            "output": value,
            "output_token_ids": tokens,
            "automatic_route": route,
            "functional_pass": evaluate_functional(value, probe["evaluator"]),
            "repetition_collapse": repetition_collapse(value),
        })
        if (index + 1) % 100 == 0:
            print(json.dumps({"evaluated": index + 1}), flush=True)
    output.mkdir(parents=True)
    raw = output / "development_outputs.jsonl"
    raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passes = sum(bool(row["functional_pass"]) for row in values)
        per[capability] = {"passes": passes, "observations": len(values), "collapses": sum(bool(row["repetition_collapse"]) for row in values), "wilson": wilson(passes, len(values))}
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    paired = [{
        "capability": row["capability"],
        "candidate_pass": bool(row["functional_pass"]),
        "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probe_by_id[row["probe_id"]]["evaluator"]),
    } for row in rows]
    relative = paired_stratified_bootstrap(paired, replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]), seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]))
    gates_cfg = protocol["absolute_screen"]
    gates = {
        "per_capability_functional": all(value["wilson"]["point"] >= float(gates_cfg["per_capability_functional_point_estimate_minimum"]) and value["wilson"]["lower_95"] >= float(gates_cfg["per_capability_functional_wilson_lower_minimum"]) for value in per.values()),
        "critical_capabilities": all(per[name]["wilson"]["point"] >= float(gates_cfg["critical_point_minimum"]) and per[name]["wilson"]["lower_95"] >= float(gates_cfg["critical_wilson_lower_minimum"]) for name in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows) == 0,
        "teacher_relative_noninferiority": relative["lower_95"] >= float(protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]),
        "same_existing_execution_graph": True,
        "teacher_absent_at_inference": True,
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-qualified-transition-control-result/1",
        "status": "PASS_NONPROMOTIONAL_CAPACITY_CONTROL" if passed else "FAIL_QUALIFIED_TRANSITION_CAPACITY_BRANCH_CLOSED",
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "observations": len(rows),
        "per_capability": per,
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "teacher_comparison": relative,
        "gates": gates,
        "passed": passed,
        "raw_outputs_sha256": sha256_file(raw),
        "evaluation_wall_seconds": time.perf_counter() - started,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "claim_boundary": "Development-only full-transition capacity control; no promotion, runtime, minimum-information, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_QUALIFIED_TRANSITION_CONTROL_PROTOCOL_V440.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--output-dir", required=True)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--candidate-dir", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    if args.command == "preflight":
        result = preflight(root, protocol)
    elif args.command == "train":
        result = train(root, protocol, (root / args.output_dir).resolve())
    else:
        result = evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
