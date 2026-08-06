"""Bounded V17 self-prefix-recovery successor for the sealed V11 C0 bridge."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
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
from .capability_compiler_phase3 import (
    TOKENIZER_FILES,
    Phase3Error,
    _BalancedSampler,
    _batch,
    _examples,
    _state_hash,
    _write_immutable,
    load_phase1_ir,
)
from .capability_compiler_phase3_sequence_bridge import _generate
from .capability_compiler_phase3_shared_output import (
    _is_trainable,
    load_candidate,
    load_protocol as load_v11_protocol,
    wrong_recent_repeat_margin_loss,
)


SYSTEMS = ("S0", "S1")
FORMAT = "abi-capability-compiler-phase3-self-prefix-successor/1"
EXPECTED_TRAINABLE_PARAMETERS = 1_057_798


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_CONDITIONAL_SUCCESSOR"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("phase3_promotion_eligible") is not False
        or tuple(protocol.get("systems", ())) != SYSTEMS
    ):
        raise Phase3Error("self-prefix governance changed")
    training = protocol.get("training", {})
    if (
        training.get("steps") != 1000
        or training.get("self_prefix_horizon") != 16
        or training.get("self_prefix_weight") != 0.25
        or training.get("trainable_parameters") != EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise Phase3Error("self-prefix training contract changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"self-prefix binding changed: {relative}")
    return protocol, sha256_file(path)


def construct_self_prefix_batch(
    ids: torch.Tensor,
    labels: torch.Tensor,
    policy_logits: torch.Tensor,
    *,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Insert the frozen policy's first wrong token and score its continuation."""

    corrupted = ids.clone()
    recovery = torch.full_like(labels, -100)
    predictions = policy_logits[:, :-1].detach().argmax(dim=-1)
    events = 0
    for row in range(ids.shape[0]):
        for source_position in range(labels.shape[1] - 1):
            target_position = source_position + 1
            target = int(labels[row, target_position].item())
            if target == -100 or int(predictions[row, source_position].item()) == target:
                continue
            if target_position + 1 >= labels.shape[1] or int(labels[row, target_position + 1].item()) == -100:
                continue
            corrupted[row, target_position] = predictions[row, source_position]
            stop = min(labels.shape[1], target_position + 1 + int(horizon))
            recovery[row, target_position + 1 : stop] = labels[row, target_position + 1 : stop]
            events += 1
            break
    return corrupted, recovery, events


def _load_pair(root: Path, protocol: Mapping[str, Any], device: torch.device):
    v11, _ = load_v11_protocol(root, (root / protocol["v11_protocol"]).resolve())
    c0 = (root / protocol["starting_candidate"]).resolve()
    model, tokenizer = load_candidate(root=root, protocol=v11, candidate_dir=c0, device=device)
    policy, _ = load_candidate(root=root, protocol=v11, candidate_dir=c0, device=device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return model, policy, tokenizer, v11


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, policy, _, _ = _load_pair(root, protocol, torch.device("cpu"))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_trainable(name))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise Phase3Error("self-prefix trainable parameter count changed")
    if _state_hash(model.state_dict()) != _state_hash(policy.state_dict()):
        raise Phase3Error("train and corruption-policy starting states differ")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "trainable_parameters": trainable,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "identical_starting_states": True,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, system: str, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if system not in SYSTEMS or output_dir.exists():
        raise Phase3Error("invalid system or immutable output already exists")
    if not torch.cuda.is_available():
        raise Phase3Error("registered GPU unavailable")
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model, policy, tokenizer, v11 = _load_pair(root, protocol, device)
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    examples = _examples(rows, tokenizer, system="A0", seed=seed, max_tokens=int(cfg["max_tokens"]))
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_trainable(name))
        if parameter.requires_grad:
            trainable.append(parameter)
    if sum(p.numel() for p in trainable) != EXPECTED_TRAINABLE_PARAMETERS:
        raise Phase3Error("trainable parameter count changed")
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = _BalancedSampler(examples, seed)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    policy_hash_before = _state_hash(policy.state_dict())
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    successful = skipped = recovery_events = recovery_tokens = 0
    sampled = Counter()
    sampled_hash = hashlib.sha256()
    curves = []
    model.train()
    while successful < int(cfg["steps"]):
        selected = sampler.batch(int(cfg["batch_size"]))
        while True:
            ids, labels, attention, prompt_lengths, routes = _batch(selected, int(tokenizer.eos_token_id), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                policy_result = policy(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=routes, use_cache=False)
            corrupted, recovery_labels, events = construct_self_prefix_batch(
                ids, labels, policy_result["logits"], horizon=int(cfg["self_prefix_horizon"])
            )
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=routes, use_cache=False)
                corrupted_result = model(corrupted, attention_mask=attention, prompt_lengths=prompt_lengths, task_routes=routes, use_cache=False)
                language_loss = F.cross_entropy(result["logits"][:, :-1].float().reshape(-1, result["logits"].shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
                classifier_loss = F.cross_entropy(result["task_logits"].float(), routes)
                repeat_loss, repeat_count = wrong_recent_repeat_margin_loss(
                    result["logits"], labels, window=int(cfg["wrong_repeat_window"]), margin=float(cfg["wrong_repeat_margin"])
                )
                if events:
                    recovery_loss = F.cross_entropy(
                        corrupted_result["logits"][:, :-1].float().reshape(-1, corrupted_result["logits"].shape[-1]),
                        recovery_labels[:, 1:].reshape(-1), ignore_index=-100,
                    )
                else:
                    recovery_loss = corrupted_result["logits"].sum() * 0.0
                recovery_weight = float(cfg["self_prefix_weight"]) if system == "S0" else 0.0
                loss = language_loss + float(cfg["classifier_loss_weight"]) * classifier_loss + float(cfg["wrong_repeat_loss_weight"]) * repeat_loss + recovery_weight * recovery_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1
                continue
            break
        successful += 1
        recovery_events += events
        recovery_tokens += int((recovery_labels != -100).sum().item())
        for row in selected:
            sampled[str(row["capability"])] += 1
            sampled_hash.update(str(row["record_id"]).encode("ascii") + b"\n")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curves.append({
                "step": successful, "language_loss": float(language_loss.detach()),
                "recovery_loss": float(recovery_loss.detach()), "classifier_loss": float(classifier_loss.detach()),
                "repeat_loss": float(repeat_loss.detach()), "recovery_events": events,
                "wall_seconds": time.perf_counter() - started,
            })
            print(json.dumps({"system": system, **curves[-1]}), flush=True)
    model.eval()
    after = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    changed = sorted(name for name in before if not torch.equal(before[name], after[name]))
    if not changed or any(not _is_trainable(name) for name in changed):
        raise Phase3Error("self-prefix continuation changed a frozen tensor")
    frozen_before = {name: value for name, value in before.items() if not _is_trainable(name)}
    frozen_after = {name: value for name, value in after.items() if not _is_trainable(name)}
    if _state_hash(frozen_before) != _state_hash(frozen_after) or _state_hash(policy.state_dict()) != policy_hash_before:
        raise Phase3Error("frozen host or corruption policy changed")
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "model.safetensors"
    save_file(after, str(checkpoint))
    parent = (root / v11["host"]["parent_path"]).resolve()
    for name in TOKENIZER_FILES:
        shutil.copyfile(parent / name, output_dir / name)
    wall = time.perf_counter() - started
    manifest = {
        "format": "abi-capability-compiler-phase3-self-prefix-candidate/1",
        "status": "TRAINED_DEVELOPMENT_ONLY", "system": system, "seed": seed,
        "protocol_sha256": protocol_sha, "phase2_human_gate": "DEFERRED_NOT_PASSED",
        "final_test_accessed": False, "checkpoint": {"path": "model.safetensors", "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "starting_candidate": {"path": protocol["starting_candidate"], "checkpoint_sha256": protocol["starting_checkpoint_sha256"]},
        "teacher_present_at_training": False, "teacher_present_at_inference": False,
        "source_parameters_copied": 0, "source_blocks_retained": 0,
        "training": {
            "steps": successful, "batch_size": int(cfg["batch_size"]), "learning_rate": float(cfg["learning_rate"]),
            "self_prefix_weight": float(cfg["self_prefix_weight"]) if system == "S0" else 0.0,
            "compute_matched_corrupted_forward": True, "recovery_events": recovery_events, "recovery_tokens": recovery_tokens,
            "successful_record_sequence_sha256": sampled_hash.hexdigest(), "sampled_records_by_capability": dict(sorted(sampled.items())),
            "skipped_amp_steps": skipped, "wall_seconds": wall, "active_parameter_seconds": EXPECTED_TRAINABLE_PARAMETERS * wall,
            "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "curves": curves,
        },
        "isolation": {
            "changed_tensors": changed, "all_changes_confined_to_registered_bridge": True,
            "frozen_state_sha256_before": _state_hash(frozen_before), "frozen_state_sha256_after": _state_hash(frozen_after),
            "corruption_policy_state_sha256_before": policy_hash_before, "corruption_policy_state_sha256_after": _state_hash(policy.state_dict()),
        },
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    _write_immutable(output_dir / "metadata.json", json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return manifest


def evaluate(root: Path, protocol_path: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    metadata = _json(candidate_dir / "metadata.json")
    if metadata.get("protocol_sha256") != protocol_sha or sha256_file(candidate_dir / "model.safetensors") != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("self-prefix candidate identity mismatch")
    if output_dir.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable evaluation output exists or GPU unavailable")
    v11, _ = load_v11_protocol(root, (root / protocol["v11_protocol"]).resolve())
    model, tokenizer = load_candidate(root=root, protocol=v11, candidate_dir=candidate_dir, device=torch.device("cuda"))
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        output, token_ids, route = _generate(model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), torch.device("cuda"))
        rows.append({
            "probe_id": str(probe["probe_id"]), "capability": str(probe["canonical_capability"]),
            "output": output, "output_token_ids": token_ids, "automatic_route": route,
            "functional_pass": evaluate_functional(output, probe["evaluator"]), "repetition_collapse": repetition_collapse(output),
        })
        if (index + 1) % 100 == 0:
            print(json.dumps({"system": metadata["system"], "evaluated": index + 1}), flush=True)
    output_dir.mkdir(parents=True)
    outputs = output_dir / "development_outputs.jsonl"
    outputs.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    grouped = {cap: [row for row in rows if row["capability"] == cap] for cap in CAPABILITIES}
    receipt = {
        "format": "abi-capability-compiler-phase3-self-prefix-evaluation/1", "status": "PASS_EXECUTION",
        "system": metadata["system"], "seed": metadata["seed"], "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"], "observations": len(rows),
        "functional_passes": sum(bool(v["functional_pass"]) for v in rows), "repetition_collapses": sum(bool(v["repetition_collapse"]) for v in rows),
        "per_capability": {cap: {"passes": sum(bool(v["functional_pass"]) for v in values), "observations": len(values), "collapses": sum(bool(v["repetition_collapse"]) for v in values)} for cap, values in grouped.items()},
        "output_tokens": sum(len(v["output_token_ids"]) for v in rows), "output_bytes": sum(len(v["output"].encode("utf-8")) for v in rows),
        "wall_seconds": time.perf_counter() - started, "outputs_path": outputs.relative_to(root).as_posix(), "outputs_sha256": sha256_file(outputs),
        "final_test_accessed": False, "human_rating_status": "DEFERRED_NOT_PASSED",
    }
    _write_immutable(output_dir / "receipt.json", canonical_json_bytes(receipt))
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_SELF_PREFIX_PROTOCOL_V17.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    training = sub.add_parser("train"); training.add_argument("--system", choices=SYSTEMS, required=True); training.add_argument("--output-dir", required=True)
    evaluation = sub.add_parser("evaluate"); evaluation.add_argument("--candidate-dir", required=True); evaluation.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); protocol = (root / args.protocol).resolve()
    if args.command == "preflight": result = preflight(root, protocol)
    elif args.command == "train": result = train(root, protocol, args.system, (root / args.output_dir).resolve())
    else: result = evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
