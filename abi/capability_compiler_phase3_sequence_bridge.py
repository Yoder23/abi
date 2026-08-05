"""Conditional Phase 3 prompt-conditioned sequence-bridge successor.

This is a development-only ABI acquisition experiment.  It keeps the sealed
LayerCake transformer frozen, adds a small continuous prompt encoder and one
low-rank residual before each frozen block, and retains physically dispatched
output cakes.  Final-test access and promotion remain prohibited while the
Phase 2 human gate is deferred.
"""

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
    TRAINABLE_ROUTES,
    Phase3Error,
    _BalancedSampler,
    _batch,
    _examples,
    _state_hash,
    _write_immutable,
    load_phase1_ir,
)
from .layercake_core_loader import load_layercake_core


SYSTEMS = ("B0", "B1", "B2", "B3", "B4")
LEGACY_SYSTEM = {"B0": "A0", "B1": "A1", "B2": "A2", "B3": "A3", "B4": "A4"}
BRIDGE_RANK = 128
EXPECTED_TRAINABLE_PARAMETERS = 1_556_998
BRIDGE_PREFIXES = (
    "abi_sequence_bridge.",
    "abi_sequence_route_classifier.",
) + tuple(f"task_cakes.{route}." for route in TRAINABLE_ROUTES)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format")
        != "abi-capability-compiler-phase3-sequence-successor/1"
        or protocol.get("status") != "PREREGISTERED_CONDITIONAL_SUCCESSOR"
        or protocol.get("phase2_status")
        != "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("phase3_promotion_eligible") is not False
    ):
        raise Phase3Error("sequence-successor governance changed")
    if protocol.get("systems") != {
        "B0": "labeled prompt-conditioned sequence bridge",
        "B1": "label-free prompt-hash routes with the same sequence bridge",
        "B2": "labeled routes with within-capability target derangement",
        "B3": "same bridge and routing supervision with no teacher-response loss",
        "B4": "same prompt-conditioned bridge with one monolithic output route",
    }:
        raise Phase3Error("sequence-successor controls changed")
    if protocol.get("architecture", {}).get("rank") != BRIDGE_RANK:
        raise Phase3Error("sequence bridge rank changed")
    if (
        protocol["architecture"].get("trainable_parameters")
        != EXPECTED_TRAINABLE_PARAMETERS
        or protocol["architecture"].get("frozen_transformer_blocks") != 3
        or protocol["architecture"].get("source_parameters_copied") != 0
    ):
        raise Phase3Error("sequence bridge parameter contract changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"sequence-successor binding changed: {relative}")
    return protocol, sha256_file(path)


class _SequenceAdapter(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.condition = nn.Linear(rank, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.condition.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        low = self.down(self.norm(hidden))
        low = low + self.condition(condition)[:, None, :]
        return hidden + self.up(F.silu(low))


class PromptConditionedSequenceBridge(nn.Module):
    """Continuous prompt summary plus three persistent low-rank transforms."""

    def __init__(self, width: int = 768, rank: int = BRIDGE_RANK, routes: int = 6):
        super().__init__()
        self.width = int(width)
        self.rank = int(rank)
        self.routes = int(routes)
        self.prompt_norm = nn.LayerNorm(width)
        self.prompt_projection = nn.Linear(3 * width, rank, bias=False)
        self.prompt_output = nn.Linear(rank, rank, bias=False)
        self.route_embedding = nn.Embedding(routes, rank)
        self.adapters = nn.ModuleList(
            _SequenceAdapter(width, rank) for _ in range(3)
        )
        nn.init.normal_(self.prompt_projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.prompt_output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.route_embedding.weight)

    def encode(
        self,
        token_states: torch.Tensor,
        *,
        prompt_lengths: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, tokens, _ = token_states.shape
        positions = torch.arange(tokens, device=token_states.device)[None]
        if prompt_lengths is not None:
            mask = positions < prompt_lengths[:, None]
            last_positions = (prompt_lengths - 1).clamp(min=0, max=tokens - 1)
        elif attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
            last_positions = mask.long().sum(dim=1).sub(1).clamp(min=0)
        else:
            mask = torch.ones(batch, tokens, dtype=torch.bool, device=token_states.device)
            last_positions = torch.full(
                (batch,), tokens - 1, dtype=torch.long, device=token_states.device
            )
        normalized = self.prompt_norm(token_states)
        weights = mask.to(normalized.dtype)
        mean = (normalized * weights[:, :, None]).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        maximum = normalized.masked_fill(~mask[:, :, None], -torch.inf).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        last = normalized[torch.arange(batch, device=normalized.device), last_positions]
        summary = torch.cat((mean, maximum, last), dim=-1)
        return self.prompt_output(F.silu(self.prompt_projection(summary)))

    def conditioned(self, summary: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        return summary + self.route_embedding(routes)


def _make_adapter_hook(model: nn.Module, layer_index: int):
    def hook(module, args, kwargs):
        condition = getattr(model, "_abi_sequence_condition", None)
        if condition is None:
            raise Phase3Error("sequence condition was not established before block execution")
        hidden = args[0]
        adapted = model.abi_sequence_bridge.adapters[layer_index](hidden, condition)
        return (adapted, *args[1:]), kwargs

    return hook


def _sequence_forward(
    self,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
    task_routes: torch.Tensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
) -> dict[str, Any]:
    if input_ids.ndim != 2:
        raise ValueError("input ids must be [batch, tokens]")
    if past_key_values is None:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        token_states = self.transformer.wte(input_ids) + self.transformer.wpe(positions)
        summary = self.abi_sequence_bridge.encode(
            token_states,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
        )
        route_logits = self.abi_sequence_route_classifier(summary)
        routes = route_logits.argmax(dim=-1) if task_routes is None else task_routes
        routes = routes.to(input_ids.device).long().flatten()
        self._abi_sequence_condition = self.abi_sequence_bridge.conditioned(summary, routes)
    else:
        if task_routes is None:
            raise ValueError("cached decode requires the prefill task route")
        route_logits = None
        routes = task_routes.to(input_ids.device).long().flatten()
        if getattr(self, "_abi_sequence_condition", None) is None:
            raise ValueError("cached decode is missing its prompt condition")
    result = self.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        return_dict=True,
    )
    hidden = result.last_hidden_state
    adapted = self._dispatch(hidden, routes)
    logits = F.linear(adapted, self.output_weight)
    self.last_task_logits = route_logits
    self.last_task_routes = routes.detach()
    return {
        "logits": logits,
        "past_key_values": result.past_key_values,
        "task_logits": route_logits,
        "task_routes": routes,
        "hidden": adapted,
    }


def install_sequence_bridge(model: nn.Module) -> None:
    if len(model.transformer.h) != 3 or int(model.config.width) != 768:
        raise Phase3Error("sequence successor requires the sealed three-block host")
    device = model.transformer.wte.weight.device
    dtype = model.transformer.wte.weight.dtype
    model.abi_sequence_bridge = PromptConditionedSequenceBridge().to(
        device=device, dtype=dtype
    )
    model.abi_sequence_route_classifier = nn.Linear(BRIDGE_RANK, 6).to(
        device=device, dtype=dtype
    )
    nn.init.zeros_(model.abi_sequence_route_classifier.weight)
    nn.init.zeros_(model.abi_sequence_route_classifier.bias)
    for index, block in enumerate(model.transformer.h):
        block.register_forward_pre_hook(
            _make_adapter_hook(model, index), with_kwargs=True
        )
    model._abi_sequence_condition = None
    model._abi_prompt_conditioned_sequence_bridge = True
    model.forward = MethodType(_sequence_forward, model)


def _is_bridge_tensor(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in BRIDGE_PREFIXES)


def _load_parent(
    root: Path, protocol: Mapping[str, Any], device: torch.device
) -> tuple[nn.Module, Any, dict[str, Any]]:
    parent = (root / str(protocol["host"]["parent_path"])).resolve()
    if sha256_file(parent / "model.safetensors") != protocol["host"]["parent_checkpoint_sha256"]:
        raise Phase3Error("LayerCake parent checkpoint changed")
    model, tokenizer, metadata = load_layercake_core(
        parent,
        layercake_root=(root / str(protocol["host"]["layercake_root"])).resolve(),
        device=device,
    )
    install_sequence_bridge(model)
    return model, tokenizer, metadata


def preflight(*, root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    model, _, _ = _load_parent(root, protocol, torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in model.named_parameters():
        if _is_bridge_tensor(name):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    count = sum(parameter.numel() for parameter in trainable)
    if count != EXPECTED_TRAINABLE_PARAMETERS:
        raise Phase3Error("sequence-successor trainable count changed")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "trainable_parameters": count,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_ratio": count / sum(parameter.numel() for parameter in model.parameters()),
        "frozen_transformer_parameters": sum(
            parameter.numel() for parameter in model.transformer.parameters()
        ),
        "final_test_accessed": False,
    }


def train_candidate(
    *, root: Path, protocol_path: Path, system: str, seed: int, output_dir: Path
) -> dict[str, Any]:
    root = root.resolve()
    protocol, protocol_sha = load_protocol(root, protocol_path.resolve())
    if system not in SYSTEMS or seed not in protocol["training"]["seeds"]:
        raise Phase3Error("unregistered sequence-successor system or seed")
    if output_dir.exists():
        raise Phase3Error(f"candidate output is immutable: {output_dir}")
    cfg = protocol["training"]
    rows = load_phase1_ir((root / protocol["phase1_ir"]["path"]).resolve())
    set_determinism(seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise Phase3Error("Phase 3 successor GPU is unavailable")
    model, tokenizer, parent_metadata = _load_parent(root, protocol, device)
    examples = _examples(
        rows,
        tokenizer,
        system=LEGACY_SYSTEM[system],
        seed=seed,
        max_tokens=int(cfg["max_tokens"]),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in model.named_parameters():
        if _is_bridge_tensor(name):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (
        trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS
        or trainable_parameters / total_parameters > 0.03
    ):
        raise Phase3Error("small sequence-bridge parameter contract changed")
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
    before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    started = time.perf_counter()
    successful = 0
    skipped_amp_steps = 0
    language_tokens = 0
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
                if system == "B3":
                    language_loss = result["logits"].sum() * 0.0
                else:
                    language_loss = F.cross_entropy(
                        result["logits"][:, :-1].float().reshape(
                            -1, result["logits"].shape[-1]
                        ),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                loss = language_loss + float(cfg["classifier_loss_weight"]) * classifier_loss
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
        for row in selected:
            sampled_record_sequence.update(str(row["record_id"]).encode("ascii") + b"\n")
        if system != "B3":
            language_tokens += sum(int(row["response_tokens"]) for row in selected)
        sampled.update(str(row["capability"]) for row in selected)
        rss_peak = max(rss_peak, process.memory_info().rss)
        if successful == 1 or successful % int(cfg["curve_interval"]) == 0:
            curves.append(
                {
                    "step": successful,
                    "language_loss": float(language_loss.detach().item()),
                    "classifier_loss": float(classifier_loss.detach().item()),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
            print(json.dumps({"system": system, **curves[-1]}), flush=True)
    model.eval()
    after = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    changed = sorted(name for name in before if not torch.equal(before[name], after[name]))
    if not changed or any(not _is_bridge_tensor(name) for name in changed):
        raise Phase3Error("candidate changed tensors outside the registered sequence bridge")
    frozen_before = {name: value for name, value in before.items() if not _is_bridge_tensor(name)}
    frozen_after = {name: value for name, value in after.items() if not _is_bridge_tensor(name)}
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
        "format": "abi-capability-compiler-phase3-sequence-candidate/1",
        "status": "TRAINED_DEVELOPMENT_ONLY",
        "system": system,
        "seed": seed,
        "protocol_sha256": protocol_sha,
        "phase2_human_gate": "DEFERRED_NOT_PASSED",
        "final_test_accessed": False,
        "architecture": parent_metadata["architecture"],
        "sequence_bridge": protocol["architecture"],
        "checkpoint": {
            "path": "model.safetensors",
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "parent": {
            "path": protocol["host"]["parent_path"],
            "checkpoint_sha256": protocol["host"]["parent_checkpoint_sha256"],
            "state_sha256": _state_hash(before),
        },
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
            "uses_destination_labels": system in {"B0", "B2", "B3"},
            "targets_deranged": system == "B2",
            "teacher_payload_present": system != "B3",
            "monolithic_route": system == "B4",
            "continuous_prompt_conditioning": True,
        },
        "training": {
            "device": "cuda",
            "steps": successful,
            "batch_size": int(cfg["batch_size"]),
            "max_tokens": int(cfg["max_tokens"]),
            "learning_rate": float(cfg["learning_rate"]),
            "classifier_loss_weight": float(cfg["classifier_loss_weight"]),
            "trainable_parameters": trainable_parameters,
            "trainable_parameter_ratio": trainable_parameters / total_parameters,
            "active_parameter_seconds": trainable_parameters * wall_seconds,
            "teacher_response_tokens_seen": language_tokens,
            "skipped_amp_steps": skipped_amp_steps,
            "successful_record_sequence_sha256": sampled_record_sequence.hexdigest(),
            "sampled_records_by_capability": dict(sorted(sampled.items())),
            "wall_seconds": wall_seconds,
            "peak_process_rss_bytes": int(rss_peak),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "curves": curves,
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
    _write_immutable(
        output_dir / "metadata.json",
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return manifest


def load_candidate(
    *, root: Path, protocol: Mapping[str, Any], candidate_dir: Path, device: torch.device
) -> tuple[nn.Module, Any]:
    model, tokenizer, _ = _load_parent(root, protocol, device)
    state = load_file(str(candidate_dir / "model.safetensors"), device=str(device))
    model.load_state_dict(state, strict=True)
    return model.eval(), tokenizer


@torch.inference_mode()
def _generate(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device):
    prompt_ids = [
        int(value)
        for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)
    ]
    if len(prompt_ids) >= int(model.config.max_tokens):
        raise Phase3Error("development prompt exceeds LayerCake context")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model(
        ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long, device=device),
        use_cache=True,
    )
    route = int(result["task_routes"].item())
    cache = result["past_key_values"]
    logits = result["logits"][:, -1]
    generated = []
    for _ in range(max_new_tokens):
        selected = logits.argmax(dim=-1)
        token = int(selected.item())
        if token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        result = model(
            selected[:, None],
            task_routes=torch.tensor([route], device=device),
            past_key_values=cache,
            use_cache=True,
        )
        cache = result["past_key_values"]
        logits = result["logits"][:, -1]
    output = tokenizer.decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output, generated, route


def evaluate_candidate(
    *, root: Path, protocol_path: Path, candidate_dir: Path, output_dir: Path
) -> dict[str, Any]:
    root = root.resolve()
    protocol, protocol_sha = load_protocol(root, protocol_path.resolve())
    metadata = _json(candidate_dir / "metadata.json")
    if metadata.get("protocol_sha256") != protocol_sha or metadata.get("system") not in SYSTEMS:
        raise Phase3Error("candidate is not bound to this sequence-successor protocol")
    if sha256_file(candidate_dir / "model.safetensors") != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("candidate checkpoint changed")
    if output_dir.exists():
        raise Phase3Error(f"evaluation output is immutable: {output_dir}")
    device = torch.device("cuda")
    model, tokenizer = load_candidate(
        root=root, protocol=protocol, candidate_dir=candidate_dir, device=device
    )
    probes = development_probes(root / protocol["development"]["catalog_path"])
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        output, token_ids, route = _generate(
            model,
            tokenizer,
            str(probe["prompt"]),
            int(probe["max_new_tokens"]),
            device,
        )
        rows.append(
            {
                "probe_id": str(probe["probe_id"]),
                "capability": str(probe["canonical_capability"]),
                "output": output,
                "output_token_ids": token_ids,
                "authoritative_output_tokens": len(token_ids),
                "automatic_route": route,
                "functional_pass": evaluate_functional(output, probe["evaluator"]),
                "repetition_collapse": repetition_collapse(output),
            }
        )
        if (index + 1) % 100 == 0:
            print(json.dumps({"system": metadata["system"], "evaluated": index + 1}), flush=True)
    output_dir.mkdir(parents=True)
    outputs_path = output_dir / "development_outputs.jsonl"
    outputs_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    grouped = {
        capability: [row for row in rows if row["capability"] == capability]
        for capability in CAPABILITIES
    }
    receipt = {
        "format": "abi-capability-compiler-phase3-sequence-development-evaluation/1",
        "status": "PASS_EXECUTION",
        "system": metadata["system"],
        "seed": metadata["seed"],
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "observations": len(rows),
        "distinct_prompts": len({row["probe_id"] for row in rows}),
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "per_capability": {
            capability: {
                "passes": sum(bool(row["functional_pass"]) for row in values),
                "observations": len(values),
                "collapses": sum(bool(row["repetition_collapse"]) for row in values),
            }
            for capability, values in grouped.items()
        },
        "automatic_route_counts": dict(
            sorted(Counter(row["automatic_route"] for row in rows).items())
        ),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_SEQUENCE_SUCCESSOR_PROTOCOL_V6.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train = sub.add_parser("train")
    train.add_argument("--system", choices=SYSTEMS, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output-dir", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--candidate-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path.cwd().resolve()
    if args.command == "preflight":
        result = preflight(root=root, protocol_path=Path(args.protocol).resolve())
    elif args.command == "train":
        result = train_candidate(
            root=root,
            protocol_path=Path(args.protocol).resolve(),
            system=args.system,
            seed=args.seed,
            output_dir=Path(args.output_dir).resolve(),
        )
    else:
        result = evaluate_candidate(
            root=root,
            protocol_path=Path(args.protocol).resolve(),
            candidate_dir=Path(args.candidate_dir).resolve(),
            output_dir=Path(args.output_dir).resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
