"""Phase 3 shared-output, semantically conditioned sequence successor."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
from types import MethodType
from typing import Any, Iterable, Mapping

import psutil
from safetensors.torch import load_file, save_file
import torch
from torch import nn
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
    PHASE1_IR_SHA256,
    TOKENIZER_FILES,
    Phase3Error,
    _BalancedSampler,
    _batch,
    _examples,
    _state_hash,
    _write_immutable,
    load_phase1_ir,
)
from .capability_compiler_phase3_sequence_bridge import (
    BRIDGE_RANK,
    _generate,
    _load_parent,
)


SYSTEMS = ("C0", "C1", "C2", "C3", "C4")
LEGACY_SYSTEM = {"C0": "A0", "C1": "A1", "C2": "A2", "C3": "A3", "C4": "A4"}
EXPECTED_TRAINABLE_PARAMETERS = 1_057_798
TRAINABLE_PREFIXES = (
    "abi_sequence_bridge.",
    "abi_sequence_route_classifier.",
    "abi_shared_output_cake.",
)


class SharedOutputCake(nn.Module):
    """One generic low-rank residual shared by every English capability."""

    def __init__(self, width: int = 768, rank: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(F.silu(self.down(self.norm(hidden))))


def _shared_dispatch(self, hidden: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    del routes
    return self.abi_shared_output_cake(hidden)


def install_shared_output(model: nn.Module) -> None:
    device = model.transformer.wte.weight.device
    dtype = model.transformer.wte.weight.dtype
    model.abi_shared_output_cake = SharedOutputCake().to(device=device, dtype=dtype)
    model._dispatch = MethodType(_shared_dispatch, model)
    model._abi_shared_output_successor = True


def _is_trainable(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)


def wrong_recent_repeat_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    window: int,
    margin: float,
) -> tuple[torch.Tensor, int]:
    """Penalize only wrong argmax tokens that repeat a recent teacher token."""

    prediction_logits = logits[:, :-1].float()
    targets = labels[:, 1:]
    predicted = prediction_logits.detach().argmax(dim=-1)
    eligible = (targets != -100) & (predicted != targets)
    repeated = torch.zeros_like(eligible)
    for offset in range(1, int(window) + 1):
        if offset >= targets.shape[1]:
            break
        prior = targets[:, :-offset]
        repeated[:, offset:] |= (
            (prior != -100) & (predicted[:, offset:] == prior)
        )
    eligible &= repeated
    count = int(eligible.sum().item())
    if count == 0:
        return prediction_logits.sum() * 0.0, 0
    safe_targets = targets.clamp_min(0)
    predicted_scores = prediction_logits.gather(-1, predicted[..., None]).squeeze(-1)
    target_scores = prediction_logits.gather(-1, safe_targets[..., None]).squeeze(-1)
    losses = F.relu(predicted_scores - target_scores + float(margin))
    return losses[eligible].mean(), count


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != "abi-capability-compiler-phase3-shared-output-successor/1"
        or protocol.get("status") != "PREREGISTERED_CONDITIONAL_SUCCESSOR"
        or protocol.get("phase2_status")
        != "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED"
        or protocol.get("phase3_promotion_eligible") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("systems")
        != {
            "C0": "semantic labels condition shared sequence transforms and one generic output residual",
            "C1": "prompt-hash labels condition the same shared architecture",
            "C2": "semantic labels with within-capability target derangement",
            "C3": "semantic labels and routing supervision with no teacher-response loss",
            "C4": "monolithic route condition with the same shared architecture",
        }
    ):
        raise Phase3Error("shared-output successor governance changed")
    architecture = protocol.get("architecture", {})
    if (
        architecture.get("trainable_parameters") != EXPECTED_TRAINABLE_PARAMETERS
        or architecture.get("output_residuals") != 1
        or architecture.get("frozen_transformer_blocks") != 3
        or architecture.get("source_parameters_copied") != 0
    ):
        raise Phase3Error("shared-output parameter contract changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"shared-output binding changed: {relative}")
    return protocol, sha256_file(path)


def _load_model(root: Path, protocol: Mapping[str, Any], device: torch.device):
    model, tokenizer, metadata = _load_parent(root, protocol, device)
    install_shared_output(model)
    return model, tokenizer, metadata


def preflight(*, root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, _, _ = _load_model(root, protocol, torch.device("cpu"))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_trainable(name))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable != EXPECTED_TRAINABLE_PARAMETERS or trainable / total > 0.02:
        raise Phase3Error("shared-output compactness contract changed")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total,
        "frozen_transformer_parameters": sum(p.numel() for p in model.transformer.parameters()),
        "final_test_accessed": False,
    }


def train_candidate(
    *, root: Path, protocol_path: Path, system: str, seed: int, output_dir: Path
) -> dict[str, Any]:
    root = root.resolve()
    protocol, protocol_sha = load_protocol(root, protocol_path.resolve())
    if system not in SYSTEMS or seed not in protocol["training"]["seeds"]:
        raise Phase3Error("unregistered shared-output system or seed")
    if output_dir.exists():
        raise Phase3Error(f"candidate output is immutable: {output_dir}")
    cfg = protocol["training"]
    rows = load_phase1_ir((root / protocol["phase1_ir"]["path"]).resolve())
    set_determinism(seed)
    if not torch.cuda.is_available():
        raise Phase3Error("shared-output successor GPU is unavailable")
    device = torch.device("cuda")
    model, tokenizer, parent_metadata = _load_model(root, protocol, device)
    examples = _examples(
        rows,
        tokenizer,
        system=LEGACY_SYSTEM[system],
        seed=seed,
        max_tokens=int(cfg["max_tokens"]),
    )
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_trainable(name))
        if parameter.requires_grad:
            trainable.append(parameter)
    trainable_parameters = sum(p.numel() for p in trainable)
    total_parameters = sum(p.numel() for p in model.parameters())
    if trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS:
        raise Phase3Error("shared-output trainable count changed")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = _BalancedSampler(examples, seed)
    process = psutil.Process()
    rss_peak = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    started = time.perf_counter()
    successful = 0
    skipped_amp_steps = 0
    language_tokens = 0
    repeat_penalty_events = 0
    sampled = Counter()
    sampled_record_sequence = hashlib.sha256()
    curves = []
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
                classifier_loss = F.cross_entropy(result["task_logits"].float(), routes)
                if system == "C3":
                    language_loss = result["logits"].sum() * 0.0
                    repeat_loss = result["logits"].sum() * 0.0
                    repeat_events = 0
                else:
                    language_loss = F.cross_entropy(
                        result["logits"][:, :-1].float().reshape(-1, result["logits"].shape[-1]),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                    repeat_loss, repeat_events = wrong_recent_repeat_margin_loss(
                        result["logits"],
                        labels,
                        window=int(cfg["wrong_repeat_window"]),
                        margin=float(cfg["wrong_repeat_margin"]),
                    )
                loss = (
                    language_loss
                    + float(cfg["classifier_loss_weight"]) * classifier_loss
                    + float(cfg["wrong_repeat_loss_weight"]) * repeat_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped_amp_steps += 1
                continue
            break
        successful += 1
        repeat_penalty_events += repeat_events
        for row in selected:
            sampled_record_sequence.update(str(row["record_id"]).encode("ascii") + b"\n")
        if system != "C3":
            language_tokens += sum(int(row["response_tokens"]) for row in selected)
        sampled.update(str(row["capability"]) for row in selected)
        rss_peak = max(rss_peak, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curves.append(
                {
                    "step": successful,
                    "language_loss": float(language_loss.detach().item()),
                    "classifier_loss": float(classifier_loss.detach().item()),
                    "wrong_repeat_loss": float(repeat_loss.detach().item()),
                    "wrong_repeat_events": repeat_events,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
            print(json.dumps({"system": system, **curves[-1]}), flush=True)
    model.eval()
    after = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    changed = sorted(name for name in before if not torch.equal(before[name], after[name]))
    if not changed or any(not _is_trainable(name) for name in changed):
        raise Phase3Error("candidate changed tensors outside the registered shared bridge")
    frozen_before = {name: value for name, value in before.items() if not _is_trainable(name)}
    frozen_after = {name: value for name, value in after.items() if not _is_trainable(name)}
    if _state_hash(frozen_before) != _state_hash(frozen_after):
        raise Phase3Error("frozen host changed")
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "model.safetensors"
    save_file(after, str(checkpoint))
    parent = (root / protocol["host"]["parent_path"]).resolve()
    for name in TOKENIZER_FILES:
        shutil.copyfile(parent / name, output_dir / name)
    wall_seconds = time.perf_counter() - started
    manifest = {
        "format": "abi-capability-compiler-phase3-shared-output-candidate/1",
        "status": "TRAINED_DEVELOPMENT_ONLY",
        "system": system,
        "seed": seed,
        "protocol_sha256": protocol_sha,
        "phase2_human_gate": "DEFERRED_NOT_PASSED",
        "final_test_accessed": False,
        "architecture": parent_metadata["architecture"],
        "shared_output_successor": protocol["architecture"],
        "checkpoint": {"path": "model.safetensors", "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parent": {"path": protocol["host"]["parent_path"], "checkpoint_sha256": protocol["host"]["parent_checkpoint_sha256"], "state_sha256": _state_hash(before)},
        "source": {
            "phase1_ir_sha256": PHASE1_IR_SHA256,
            "teacher": protocol["source"]["model"],
            "teacher_revision": protocol["source"]["revision"],
            "teacher_present_during_training": False,
            "teacher_present_at_inference": False,
            "source_parameters_copied": 0,
            "source_blocks_retained": 0,
        },
        "control": {
            "uses_destination_labels": system in {"C0", "C2", "C3"},
            "targets_deranged": system == "C2",
            "teacher_payload_present": system != "C3",
            "monolithic_route": system == "C4",
            "shared_output_residual": True,
            "wrong_repeat_loss": system != "C3",
        },
        "training": {
            "device": "cuda",
            "steps": successful,
            "batch_size": int(cfg["batch_size"]),
            "max_tokens": int(cfg["max_tokens"]),
            "learning_rate": float(cfg["learning_rate"]),
            "classifier_loss_weight": float(cfg["classifier_loss_weight"]),
            "wrong_repeat_loss_weight": float(cfg["wrong_repeat_loss_weight"]),
            "wrong_repeat_window": int(cfg["wrong_repeat_window"]),
            "wrong_repeat_margin": float(cfg["wrong_repeat_margin"]),
            "wrong_repeat_penalty_events": repeat_penalty_events,
            "trainable_parameters": trainable_parameters,
            "trainable_parameter_ratio": trainable_parameters / total_parameters,
            "active_parameter_seconds": trainable_parameters * wall_seconds,
            "teacher_response_tokens_seen": language_tokens,
            "skipped_amp_steps": skipped_amp_steps,
            "successful_record_sequence_sha256": sampled_record_sequence.hexdigest(),
            "sampled_records_by_capability": dict(sorted(sampled.items())),
            "curves": curves,
            "wall_seconds": wall_seconds,
            "peak_process_rss_bytes": rss_peak,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "isolation": {
            "changed_tensors": changed,
            "changed_tensor_count": len(changed),
            "all_changes_confined_to_registered_bridge": True,
            "frozen_state_sha256_before": _state_hash(frozen_before),
            "frozen_state_sha256_after": _state_hash(frozen_after),
        },
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    _write_immutable(output_dir / "metadata.json", json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return manifest


def load_candidate(*, root: Path, protocol: Mapping[str, Any], candidate_dir: Path, device: torch.device):
    model, tokenizer, _ = _load_model(root, protocol, device)
    state = load_file(str(candidate_dir / "model.safetensors"), device=str(device))
    model.load_state_dict(state, strict=True)
    return model.eval(), tokenizer


def evaluate_candidate(*, root: Path, protocol_path: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    protocol, protocol_sha = load_protocol(root, protocol_path.resolve())
    metadata = _json(candidate_dir / "metadata.json")
    if metadata.get("protocol_sha256") != protocol_sha or metadata.get("system") not in SYSTEMS:
        raise Phase3Error("candidate is not bound to this shared-output protocol")
    if sha256_file(candidate_dir / "model.safetensors") != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("candidate checkpoint changed")
    if output_dir.exists():
        raise Phase3Error(f"evaluation output is immutable: {output_dir}")
    device = torch.device("cuda")
    model, tokenizer = load_candidate(root=root, protocol=protocol, candidate_dir=candidate_dir, device=device)
    probes = development_probes(root / protocol["development"]["catalog_path"])
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        output, token_ids, route = _generate(model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), device)
        rows.append({
            "probe_id": str(probe["probe_id"]),
            "capability": str(probe["canonical_capability"]),
            "output": output,
            "output_token_ids": token_ids,
            "authoritative_output_tokens": len(token_ids),
            "automatic_route": route,
            "functional_pass": evaluate_functional(output, probe["evaluator"]),
            "repetition_collapse": repetition_collapse(output),
        })
        if (index + 1) % 100 == 0:
            print(json.dumps({"system": metadata["system"], "evaluated": index + 1}), flush=True)
    output_dir.mkdir(parents=True)
    outputs_path = output_dir / "development_outputs.jsonl"
    outputs_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    grouped = {capability: [row for row in rows if row["capability"] == capability] for capability in CAPABILITIES}
    receipt = {
        "format": "abi-capability-compiler-phase3-shared-output-evaluation/1",
        "status": "PASS_EXECUTION",
        "system": metadata["system"],
        "seed": metadata["seed"],
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "observations": len(rows),
        "distinct_prompts": len({row["probe_id"] for row in rows}),
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "per_capability": {capability: {"passes": sum(bool(row["functional_pass"]) for row in values), "observations": len(values), "collapses": sum(bool(row["repetition_collapse"]) for row in values)} for capability, values in grouped.items()},
        "automatic_route_counts": dict(sorted(Counter(row["automatic_route"] for row in rows).items())),
        "output_tokens": sum(len(row["output_token_ids"]) for row in rows),
        "output_bytes": sum(len(row["output"].encode("utf-8")) for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "outputs_path": outputs_path.relative_to(root).as_posix(),
        "outputs_sha256": sha256_file(outputs_path),
        "final_test_accessed": False,
        "human_rating_status": "DEFERRED_NOT_PASSED",
    }
    _write_immutable(output_dir / "receipt.json", canonical_json_bytes(receipt))
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_SHARED_OUTPUT_PROTOCOL_V11.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train = sub.add_parser("train")
    train.add_argument("--system", choices=SYSTEMS, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output-dir", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--candidate-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    if args.command == "preflight":
        result = preflight(root=root, protocol_path=(root / args.protocol).resolve())
    elif args.command == "train":
        result = train_candidate(root=root, protocol_path=(root / args.protocol).resolve(), system=args.system, seed=args.seed, output_dir=(root / args.output_dir).resolve())
    else:
        result = evaluate_candidate(root=root, protocol_path=(root / args.protocol).resolve(), candidate_dir=(root / args.candidate_dir).resolve(), output_dir=(root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
