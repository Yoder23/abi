"""Conditional Phase 3 causal ABI-to-LayerCake acquisition campaign.

Phase 2 human judgments are deferred, not passed.  This module therefore
produces development-only causal evidence and refuses final-test access or a
promotion certificate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence
import zipfile

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
    stable_seed,
)
from .capability_compiler_phase2_teacher import development_probes
from .layercake_core_loader import load_layercake_core


PHASE1_IR_SHA256 = "a246a52bcf27609b46cdb0530f1daaefe749b7c4a1000f9578f20e505a596f20"
SYSTEMS = ("A0", "A1", "A2", "A3", "A4")
TRAINABLE_ROUTES = tuple(range(6))
CAPABILITY_TO_ROUTE = {
    "grammar": 0,
    "coherence": 0,
    "fluent_realization": 0,
    "prompt_grounding": 1,
    "instruction_following": 1,
    "conversation": 2,
    "clarification": 2,
    "abstention": 2,
    "supplied_text_summarization": 3,
    "rewriting": 3,
    "email_drafting_from_notes": 4,
    "tone_control": 4,
    "format_control": 4,
    "fact_free_reasoning": 5,
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


class Phase3Error(RuntimeError):
    pass


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        raise Phase3Error(f"Phase 3 evidence is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _protocol(
    root: Path,
    path: Path,
    *,
    binding_overrides: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") == "abi-capability-compiler-phase3-paired-sampler-amendment/1":
        if protocol.get("status") != "PREREGISTERED_PAIRED_CONFORMANCE_CORRECTION":
            raise Phase3Error("Phase 3 paired-sampler amendment is not controlling")
        parent_spec = protocol.get("parent_protocol")
        if not isinstance(parent_spec, dict):
            raise Phase3Error("Phase 3 paired-sampler parent is missing")
        parent_path = (root / str(parent_spec.get("path", ""))).resolve()
        if not parent_path.is_file() or sha256_file(parent_path) != parent_spec.get("sha256"):
            raise Phase3Error("Phase 3 paired-sampler parent changed")
        changes = protocol.get("changes")
        if changes != {
            "successful_step_sampling": {
                "from": "sampler_advances_on_every_attempt",
                "to": "retry_identical_batch_until_optimizer_step_succeeds",
            },
            "successful_record_sequence_sha256": {
                "from": "absent",
                "to": "required_and_equal_across_A0_A1_A2_A3_A4",
            },
        }:
            raise Phase3Error("Phase 3 paired-sampler amendment expanded")
        amendment_bindings = _json(path).get("bindings", {})
        if not isinstance(amendment_bindings, dict):
            raise Phase3Error("Phase 3 paired-sampler bindings are missing")
        parent, _ = _protocol(
            root,
            parent_path,
            binding_overrides=frozenset(amendment_bindings),
        )
        protocol = copy.deepcopy(parent)
        protocol["protocol_id"] = "abi-capability-compiler-phase3-conditional-paired-sampler-v4"
        protocol["paired_sampler_amendment"] = {
            "parent_protocol_sha256": parent_spec["sha256"],
            "successful_record_sequence_equality_required": True,
        }
        protocol["bindings"].update(amendment_bindings)
    if protocol.get("format") == "abi-capability-compiler-phase3-evidence-emitter-amendment/1":
        if protocol.get("status") != "PREREGISTERED_EVIDENCE_EMITTER_ONLY":
            raise Phase3Error("Phase 3 evidence-emitter amendment is not controlling")
        parent_spec = protocol.get("parent_protocol")
        if not isinstance(parent_spec, dict):
            raise Phase3Error("Phase 3 amendment parent is missing")
        parent_path = (root / str(parent_spec.get("path", ""))).resolve()
        if not parent_path.is_file() or sha256_file(parent_path) != parent_spec.get("sha256"):
            raise Phase3Error("Phase 3 amendment parent changed")
        changes = protocol.get("changes")
        if changes != {
            "A3.post_training_guard": {
                "from": "require_task_cake_byte_identity",
                "to": "accept_and_report_registered_scope_adamw_weight_decay",
            }
        }:
            raise Phase3Error("Phase 3 amendment changed experiment semantics")
        amendment_bindings = _json(path).get("bindings", {})
        if not isinstance(amendment_bindings, dict):
            raise Phase3Error("Phase 3 amendment bindings are missing")
        parent, _ = _protocol(
            root,
            parent_path,
            binding_overrides=frozenset(amendment_bindings),
        )
        protocol = copy.deepcopy(parent)
        protocol["protocol_id"] = "abi-capability-compiler-phase3-conditional-emitter-amendment-v3"
        protocol["evidence_emitter_amendment"] = {
            "parent_protocol_sha256": parent_spec["sha256"],
            "training_semantics_changed": False,
        }
        protocol["bindings"].update(_json(path).get("bindings", {}))
    if protocol.get("format") == "abi-capability-compiler-phase3-protocol-repair/1":
        if protocol.get("status") != "PREREGISTERED_SINGLE_ALLOWED_REPAIR":
            raise Phase3Error("Phase 3 repair is not controlling")
        parent_spec = protocol.get("parent_protocol")
        if not isinstance(parent_spec, dict):
            raise Phase3Error("Phase 3 repair parent is missing")
        parent_path = (root / str(parent_spec.get("path", ""))).resolve()
        if not parent_path.is_file() or sha256_file(parent_path) != parent_spec.get("sha256"):
            raise Phase3Error("Phase 3 repair parent changed")
        parent = _json(parent_path)
        changes = protocol.get("changes")
        if changes != {"training.max_tokens": {"from": 256, "to": 512}}:
            raise Phase3Error("Phase 3 repair expanded beyond the measured bottleneck")
        protocol = copy.deepcopy(parent)
        protocol["protocol_id"] = "abi-capability-compiler-phase3-conditional-repair1-v2"
        protocol["training"]["max_tokens"] = 512
        protocol["repair"] = {
            "parent_protocol_sha256": parent_spec["sha256"],
            "single_allowed_repair_consumed": True,
        }
        protocol["bindings"].update(_json(path).get("bindings", {}))
    if protocol.get("format") != "abi-capability-compiler-phase3-protocol/1":
        raise Phase3Error("unsupported Phase 3 protocol")
    if protocol.get("status") != "PREREGISTERED_CONDITIONAL_PHASE3":
        raise Phase3Error("Phase 3 protocol is not controlling")
    if protocol.get("phase2_status") != "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED":
        raise Phase3Error("Phase 2 deferral boundary changed")
    if protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("Phase 3 final-test firewall changed")
    bindings = protocol.get("bindings")
    if not isinstance(bindings, dict):
        raise Phase3Error("Phase 3 bindings are missing")
    for relative, expected in bindings.items():
        if relative in binding_overrides:
            continue
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 3 binding changed: {relative}")
    return protocol, sha256_file(path)


def load_phase1_ir(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != PHASE1_IR_SHA256:
        raise Phase3Error("certified Phase 1 IR identity changed")
    with zipfile.ZipFile(path) as archive:
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines()]
    if len(rows) != 7_000:
        raise Phase3Error("Phase 1 IR depth changed")
    counts = Counter(str(row.get("capability")) for row in rows)
    if set(counts) != set(CAPABILITIES) or set(counts.values()) != {500}:
        raise Phase3Error("Phase 1 capability balance changed")
    for row in rows:
        if (
            row.get("destination") != "english_core"
            or row.get("domain") != "domain_independent"
            or row.get("domain_labels") != []
            or row.get("domain_claims") != []
            or row.get("knowledge_class") != "english_linguistic_form"
            or row.get("functional_pass") is not True
            or row.get("split") != "acquisition"
        ):
            raise Phase3Error("noneligible record crossed the Phase 3 firewall")
    return sorted(rows, key=lambda row: (str(row["capability"]), str(row["selection_key"])))


def _route(system: str, row: Mapping[str, Any]) -> int:
    if system in {"A0", "A2", "A3"}:
        return CAPABILITY_TO_ROUTE[str(row["capability"])]
    if system == "A4":
        return 0
    if system == "A1":
        digest = hashlib.sha256(
            ("phase3-label-free:" + str(row["normalized_generation_prompt_sha256"])).encode("ascii")
        ).digest()
        return int.from_bytes(digest[:8], "big") % len(TRAINABLE_ROUTES)
    raise Phase3Error(f"unknown Phase 3 system: {system}")


def _deranged_outputs(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    result: dict[str, str] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["capability"])].append(row)
    for capability, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: str(row["ir_record_id"]))
        shift = 1 + stable_seed(seed, capability, "A2") % (len(ordered) - 1)
        for index, row in enumerate(ordered):
            result[str(row["ir_record_id"])] = str(ordered[(index + shift) % len(ordered)]["normalized_output"])
    if len(result) != len(rows):
        raise Phase3Error("target derangement is incomplete")
    return result


def _examples(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, *, system: str, seed: int, max_tokens: int
) -> list[dict[str, Any]]:
    eos = int(tokenizer.eos_token_id)
    deranged = _deranged_outputs(rows, seed) if system == "A2" else {}
    examples: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["normalized_generation_prompt"]).rstrip() + "\n"
        response = deranged.get(str(row["ir_record_id"]), str(row["normalized_output"]))
        prompt_ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
        response_ids = [int(value) for value in tokenizer.encode(response, add_special_tokens=False)] + [eos]
        available = max_tokens - len(prompt_ids)
        if available < 2:
            continue
        response_ids = response_ids[:available]
        if response_ids[-1] != eos:
            response_ids[-1] = eos
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        examples.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "route": _route(system, row),
                "input_ids": input_ids,
                "labels": labels,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(response_ids),
            }
        )
    counts = Counter(row["capability"] for row in examples)
    if set(counts) != set(CAPABILITIES) or min(counts.values()) < 450:
        raise Phase3Error(f"tokenization removed too much capability evidence: {dict(counts)}")
    return examples


class _BalancedSampler:
    def __init__(self, rows: Sequence[Mapping[str, Any]], seed: int):
        self.grouped = {
            capability: [row for row in rows if row["capability"] == capability]
            for capability in CAPABILITIES
        }
        self.rng = random.Random(seed)
        self.capability_index = 0

    def batch(self, size: int) -> list[Mapping[str, Any]]:
        result = []
        for _ in range(size):
            capability = CAPABILITIES[self.capability_index % len(CAPABILITIES)]
            self.capability_index += 1
            result.append(self.grouped[capability][self.rng.randrange(len(self.grouped[capability]))])
        return result


def _batch(rows: Sequence[Mapping[str, Any]], eos: int, device: torch.device):
    length = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), length), eos, dtype=torch.long, device=device)
    labels = torch.full((len(rows), length), -100, dtype=torch.long, device=device)
    attention = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    prompt_lengths = torch.empty(len(rows), dtype=torch.long, device=device)
    routes = torch.empty(len(rows), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        count = len(row["input_ids"])
        ids[index, :count] = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        labels[index, :count] = torch.tensor(row["labels"], dtype=torch.long, device=device)
        attention[index, :count] = 1
        prompt_lengths[index] = int(row["prompt_tokens"])
        routes[index] = int(row["route"])
    return ids, labels, attention, prompt_lengths, routes


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train_candidate(
    *, root: Path, protocol_path: Path, system: str, seed: int, output_dir: Path
) -> dict[str, Any]:
    root = root.resolve()
    protocol, protocol_sha = _protocol(root, protocol_path.resolve())
    if system not in SYSTEMS:
        raise Phase3Error("system must be one of A0-A4")
    if seed not in protocol["training"]["seeds"]:
        raise Phase3Error("unregistered Phase 3 seed")
    if output_dir.exists():
        raise Phase3Error(f"candidate output is immutable: {output_dir}")
    cfg = protocol["training"]
    parent = (root / protocol["host"]["parent_path"]).resolve()
    layercake_root = (root / protocol["host"]["layercake_root"]).resolve()
    ir_path = (root / protocol["phase1_ir"]["path"]).resolve()
    if sha256_file(parent / "model.safetensors") != protocol["host"]["parent_checkpoint_sha256"]:
        raise Phase3Error("LayerCake parent checkpoint changed")
    rows = load_phase1_ir(ir_path)
    set_determinism(seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise Phase3Error("Phase 3 GPU is unavailable")
    model, tokenizer, parent_metadata = load_layercake_core(parent, layercake_root=layercake_root, device=device)
    examples = _examples(rows, tokenizer, system=system, seed=seed, max_tokens=int(cfg["max_tokens"]))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(model.task_classifier.parameters())
    for route in TRAINABLE_ROUTES:
        trainable.extend(model.task_cakes[route].parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != 606_730 or trainable_parameters / total_parameters > 0.01:
        raise Phase3Error("small-bridge parameter contract changed")
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
                if system == "A3":
                    language_loss = result["logits"].sum() * 0.0
                else:
                    language_loss = F.cross_entropy(
                        result["logits"][:, :-1].float().reshape(-1, result["logits"].shape[-1]),
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
        language_tokens += sum(int(row["response_tokens"]) for row in selected) if system != "A3" else 0
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
    after = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    changed = sorted(name for name in before if not torch.equal(before[name], after[name]))
    allowed = ("task_classifier.",) + tuple(f"task_cakes.{route}." for route in TRAINABLE_ROUTES)
    if not changed or any(not name.startswith(allowed) for name in changed):
        raise Phase3Error("candidate changed tensors outside the registered bridge")
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "model.safetensors"
    save_file(after, str(checkpoint))
    for name in TOKENIZER_FILES:
        shutil.copyfile(parent / name, output_dir / name)
    wall_seconds = time.perf_counter() - started
    manifest = {
        "format": "abi-capability-compiler-phase3-candidate/1",
        "status": "TRAINED_DEVELOPMENT_ONLY",
        "system": system,
        "seed": seed,
        "protocol_sha256": protocol_sha,
        "phase2_human_gate": "DEFERRED_NOT_PASSED",
        "final_test_accessed": False,
        "architecture": parent_metadata["architecture"],
        "checkpoint": {"path": "model.safetensors", "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
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
            "uses_destination_labels": system in {"A0", "A2", "A3"},
            "targets_deranged": system == "A2",
            "teacher_payload_present": system != "A3",
            "monolithic_route": system == "A4",
            "zero_teacher_loss_adamw_weight_decay_reported": system == "A3",
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
            "frozen_state_sha256_before": _state_hash({k: v for k, v in before.items() if not k.startswith(allowed)}),
            "frozen_state_sha256_after": _state_hash({k: v for k, v in after.items() if not k.startswith(allowed)}),
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


@torch.inference_mode()
def _generate(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device):
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
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
    generated: list[int] = []
    for _ in range(max_new_tokens):
        selected = logits.argmax(dim=-1)
        token = int(selected.item())
        if token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        result = model(selected[:, None], task_routes=torch.tensor([route], device=device), past_key_values=cache, use_cache=True)
        cache = result["past_key_values"]
        logits = result["logits"][:, -1]
    output = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output, generated, route


def evaluate_candidate(
    *, root: Path, protocol_path: Path, candidate_dir: Path, output_dir: Path
) -> dict[str, Any]:
    root = root.resolve()
    protocol, protocol_sha = _protocol(root, protocol_path.resolve())
    metadata = _json(candidate_dir / "metadata.json")
    if metadata.get("protocol_sha256") != protocol_sha or metadata.get("system") not in SYSTEMS:
        raise Phase3Error("candidate is not bound to this Phase 3 protocol")
    if sha256_file(candidate_dir / "model.safetensors") != metadata["checkpoint"]["sha256"]:
        raise Phase3Error("candidate checkpoint changed")
    if output_dir.exists():
        raise Phase3Error(f"evaluation output is immutable: {output_dir}")
    layercake_root = (root / protocol["host"]["layercake_root"]).resolve()
    model, tokenizer, _ = load_layercake_core(candidate_dir, layercake_root=layercake_root, device=torch.device("cuda"))
    model.eval()
    probes = development_probes(root / protocol["development"]["catalog_path"])
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        output, token_ids, route = _generate(
            model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), torch.device("cuda")
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
    grouped = {capability: [row for row in rows if row["capability"] == capability] for capability in CAPABILITIES}
    receipt = {
        "format": "abi-capability-compiler-phase3-development-evaluation/1",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_PROTOCOL_V1.json")
    sub = parser.add_subparsers(dest="command", required=True)
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
    if args.command == "train":
        result = train_candidate(
            root=root,
            protocol_path=Path(args.protocol),
            system=args.system,
            seed=args.seed,
            output_dir=Path(args.output_dir).resolve(),
        )
        keys = ("system", "seed", "status", "checkpoint")
    else:
        result = evaluate_candidate(
            root=root,
            protocol_path=Path(args.protocol),
            candidate_dir=Path(args.candidate_dir).resolve(),
            output_dir=Path(args.output_dir).resolve(),
        )
        keys = ("system", "seed", "status", "functional_passes", "repetition_collapses")
    print(json.dumps({key: result[key] for key in keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
