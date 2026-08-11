"""Train one complete clean-start ABI LayerCake lineage at a sealed Phase 4 budget."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any, Iterable, Mapping, Sequence
import zipfile

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_copy_balanced_transition as balanced
from . import capability_compiler_phase3_final_controls as final_controls
from . import capability_compiler_phase3_host_recovery_bridge as host_recovery
from . import capability_compiler_phase3_qualified_transition_control as qualified
from . import capability_compiler_phase3_route_isolated as isolated
from . import capability_compiler_phase3_sparse_router as sparse
from . import capability_compiler_phase3_targeted_recovery_bridge as targeted
from . import capability_compiler_phase3_token_substrate_conformance as substrate
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _BalancedSampler, _write_immutable
from .capability_compiler_phase3_bpe_core import _layercake_api, _tokenizer
from .capability_compiler_phase3_guarded_screen import artifact_markers
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap
from .capability_compiler_phase3_segment_router import METADATA
from .capability_compiler_phase3_weak_residual import WEAK_CAPABILITIES
from .capability_compiler_phase4_budget_manifest import _prefix_by_group, _rank
from .capability_compiler_phase4_lineage_audit import _archive


FORMAT = "abi-capability-compiler-phase4-abi-lineage/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_COMPLETE_CLEAN_START_ABI_LINEAGE"
        or protocol.get("training_device") != "cuda"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("Phase 4 ABI lineage governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 4 ABI lineage binding changed: {relative}")
    return protocol, sha256_file(path)


@contextmanager
def _patch(module: Any, **changes: Any):
    previous = {name: getattr(module, name) for name in changes}
    try:
        for name, value in changes.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


def _stage_sha(name: str, budget: str, seed: int, configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes({"stage": name, "budget": budget, "seed": seed, "configuration": configuration})).hexdigest()


def _examples_subset(rows: Sequence[Mapping[str, Any]], tokenizer: Any, *, system: str, seed: int, max_tokens: int) -> list[dict[str, Any]]:
    del seed
    if system != "A0":
        raise Phase3Error("Phase 4 ABI lineage trains only labeled A0")
    eos = int(tokenizer.eos_token_id)
    examples = []
    for row in rows:
        prompt = str(row["normalized_generation_prompt"]).rstrip() + "\n"
        prompt_ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
        response_ids = [int(value) for value in tokenizer.encode(str(row["normalized_output"]), add_special_tokens=False)] + [eos]
        available = max_tokens - len(prompt_ids)
        if available < 2:
            continue
        response_ids = response_ids[:available]
        if response_ids[-1] != eos:
            response_ids[-1] = eos
        examples.append({
            "record_id": str(row["ir_record_id"]),
            "capability": str(row["capability"]),
            "route": CAPABILITY_TO_ROUTE[str(row["capability"])],
            "input_ids": prompt_ids + response_ids,
            "labels": [-100] * len(prompt_ids) + response_ids,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
        })
    counts = Counter(row["capability"] for row in examples)
    expected = Counter(str(row["capability"]) for row in rows)
    if set(counts) != set(expected) or any(counts[key] < max(1, int(expected[key] * 0.9)) for key in expected):
        raise Phase3Error("subset tokenization removed too much evidence")
    return examples


def _selected_rows(root: Path, protocol: Mapping[str, Any], manifest: Mapping[str, Any], budget_id: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    source_specs = {str(item["id"]): item for item in protocol["teacher_artifacts"]}
    rows = {key: _archive(root, value)[1] for key, value in source_specs.items()}
    rows["v138_targeted_ir"] = [row for row in rows["v138_targeted_ir"] if str(row["capability"]) in set(WEAK_CAPABILITIES)]
    budget_spec = next((row for row in protocol["budgets"] if row["id"] == budget_id), None)
    if budget_spec is None:
        raise Phase3Error("unregistered Phase 4 budget")
    ranked = {
        "phase1_ir": _rank(rows["phase1_ir"], artifact="phase1_ir", salt=manifest["selection_salt"], groups=("capability",)),
        "v138_targeted_ir": _rank(rows["v138_targeted_ir"], artifact="v138_targeted_ir", salt=manifest["selection_salt"], groups=("capability",)),
        "v480_host_supervision": _rank(rows["v480_host_supervision"], artifact="v480_host_supervision", salt=manifest["selection_salt"], groups=("capability", "builder")),
    }
    selected = {
        "phase1_ir": _prefix_by_group(ranked["phase1_ir"], ("capability",), int(budget_spec["phase1_per_capability"])),
        "v138_targeted_ir": _prefix_by_group(ranked["v138_targeted_ir"], ("capability",), int(budget_spec["targeted_per_weak_capability"])),
        "v480_host_supervision": _prefix_by_group(ranked["v480_host_supervision"], ("capability", "builder"), int(budget_spec["host_per_capability_builder"])),
    }
    ids = {key: {str(row.get("ir_record_id", row.get("record_id"))) for row in value} for key, value in selected.items()}
    selection_sha = hashlib.sha256(canonical_json_bytes({key: sorted(value) for key, value in ids.items()})).hexdigest()
    manifest_budget = next(row for row in manifest["budgets"] if row["id"] == budget_id)
    if selection_sha != manifest_budget["selection_sha256"]:
        raise Phase3Error("budget selection no longer matches V567")
    return selected, manifest_budget


def _subset_host_artifact(path: Path, rows: Sequence[Mapping[str, Any]], budget: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase3Error("immutable Phase 4 host subset exists")
    records = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    accounting = {
        "format": "abi-phase4-host-subset-accounting/1",
        "records": len(rows),
        "unique_source_attempts": len({str(row["source_attempt_sha256"]) for row in rows}),
        "teacher_input_tokens": sum(int(row["source_teacher_input_tokens"]) for row in rows),
        "teacher_output_tokens": sum(int(row["source_teacher_output_tokens"]) for row in rows),
        "teacher_output_bytes": sum(len(str(row["output"]).encode("utf-8")) for row in rows),
        "stored_logits": 0,
        "stored_hidden_activations": 0,
        "copied_source_parameters": 0,
    }
    accounting_bytes = json.dumps(accounting, sort_keys=True, indent=2).encode() + b"\n"
    manifest = {
        "format": "abi-phase4-host-subset-manifest/1",
        "budget": budget["id"],
        "records": len(rows),
        "records_jsonl_sha256": hashlib.sha256(records).hexdigest(),
        "accounting_sha256": hashlib.sha256(accounting_bytes).hexdigest(),
        "final_test_accessed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("accounting.json", accounting_bytes)
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
        archive.writestr("records.jsonl", records)


def _load_candidate(root: Path, base_protocol: Mapping[str, Any], candidate: Path, device: torch.device):
    _, model, tokenizer, metadata = qualified._load_parent(root, dict(base_protocol), device)
    model.load_state_dict(load_file(str(candidate / "model.safetensors"), device=str(device)), strict=True)
    return model, tokenizer, metadata


class _FlexibleAllStrataSampler:
    def __init__(self, rows: Sequence[Mapping[str, Any]], seed: int):
        self.groups = {(capability, builder): [row for row in rows if row["capability"] == capability and int(row["builder"]) == builder] for capability in WEAK_CAPABILITIES for builder in range(4)}
        if any(not value for value in self.groups.values()):
            raise Phase3Error("Phase 4 host subset lost a stratum")
        self.strata = tuple(self.groups)
        self.rng = random.Random(seed)
        self.recovery_index = 0

    def teacher_forced_batch(self):
        return [self.rng.choice(self.groups[key]) for key in self.strata]

    def recovery_batch(self, size: int):
        result = []
        for _ in range(size):
            key = self.strata[self.recovery_index % len(self.strata)]
            self.recovery_index += 1
            result.append(self.rng.choice(self.groups[key]))
        return result


class _FlexibleDualViewSampler:
    def __init__(self, rows: Sequence[Mapping[str, Any]], seed: int):
        self.groups = {(capability, builder, view): [row for row in rows if row["capability"] == capability and int(row["builder"]) == builder and row["view"] == view] for capability in WEAK_CAPABILITIES for builder in range(4) for view in ("host_projected", "source_wrapped")}
        if any(not value for value in self.groups.values()):
            raise Phase3Error("Phase 4 dual-view subset lost a stratum")
        self.recovery_strata = tuple(self.groups)
        self.rng = random.Random(seed)
        self.recovery_index = 0

    def teacher_forced_batch(self):
        return [self.rng.choice(self.groups[key]) for key in self.recovery_strata]

    def recovery_batch(self, size: int):
        result = []
        for _ in range(size):
            key = self.recovery_strata[self.recovery_index % len(self.recovery_strata)]
            self.recovery_index += 1
            result.append(self.rng.choice(self.groups[key]))
        return result


def _router_data(root: Path, router_protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        prompt = str(row["normalized_acquisition_prompt"])
        lines = prompt.splitlines()
        if len(lines) < 2 or not lines[0].strip():
            raise Phase3Error("router record lacks metadata/body boundary")
        metadata, body = lines[0].strip(), "\n".join(lines[1:]).strip()
        body_bpe, body_character = sparse._features(tokenizer, router_protocol, body)
        metadata_bpe, metadata_character = sparse._features(tokenizer, router_protocol, metadata)
        result.append({"record_id": str(row["ir_record_id"]), "capability": str(row["capability"]), "body_bpe": body_bpe, "body_character": body_character, "metadata_bpe": metadata_bpe, "metadata_character": metadata_character})
    return result


def _train_router(root: Path, protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], seed: int, output: Path, protocol_sha: str) -> dict[str, Any]:
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("router output exists or CUDA unavailable")
    _, _, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    data = _router_data(root, protocol, rows, tokenizer)
    set_determinism(seed)
    device = torch.device("cuda")
    model = sparse._model(protocol, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol["training"]["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(protocol["training"]["weight_decay"]))
    sampler = _BalancedSampler(data, seed)
    label_to_id = {name: index for index, name in enumerate((*CAPABILITIES, METADATA))}
    weights = torch.ones(len(label_to_id), device=device)
    weights[label_to_id[METADATA]] = 1.0 / len(CAPABILITIES)
    process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats(); sequence = hashlib.sha256(); curves = []; started = time.perf_counter()
    for step in range(1, int(protocol["training"]["steps"]) + 1):
        batch = sampler.batch(int(protocol["training"]["batch_size"]))
        bpe, chars, targets = [], [], []
        for row in batch:
            bpe.extend((row["body_bpe"], row["metadata_bpe"])); chars.extend((row["body_character"], row["metadata_character"])); targets.extend((label_to_id[row["capability"]], label_to_id[METADATA])); sequence.update(row["record_id"].encode() + b"\n")
        bpe_ids, bpe_offsets = sparse._bag(bpe, device); char_ids, char_offsets = sparse._bag(chars, device); target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True); logits = model(bpe_ids, bpe_offsets, char_ids, char_offsets); loss = F.cross_entropy(logits, target_tensor, weight=weights); loss.backward(); optimizer.step(); peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(protocol["training"]["curve_interval"]) == 0:
            curves.append({"step": step, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - started})
    output.mkdir(parents=True); checkpoint = output / "router.safetensors"; save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(checkpoint)); wall = time.perf_counter() - started
    config_path = output / "config.json"; _write_immutable(config_path, json.dumps({"vocabulary": tokenizer.vocab_size, **protocol["representation"]}, sort_keys=True, indent=2).encode() + b"\n")
    metadata = {"format": "abi-capability-compiler-phase4-sparse-router/1", "status": "TRAINED_BUDGET_SCOPED_ROUTER", "protocol_sha256": protocol_sha, "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size}, "config": {"path": config_path.name, "sha256": sha256_file(config_path), "trainable_parameters": sum(value.numel() for value in model.parameters())}, "seed": seed, "training": {"steps": int(protocol["training"]["steps"]), "records": len(rows), "record_sequence_sha256": sequence.hexdigest(), "wall_seconds": wall, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "curves": curves}, "imported_information": {"records": len(rows), "capability_labels": len(rows), "metadata_labels": len(rows), "teacher_outputs_added": 0, "stored_logits": 0, "stored_activations": 0, "source_parameters_copied": 0}, "teacher_present_at_inference": False, "final_test_accessed": False}
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(); _write_immutable(output / "metadata.json", json.dumps(metadata, sort_keys=True, indent=2).encode() + b"\n"); return metadata


def _evaluate_gates(root: Path, base_protocol: Mapping[str, Any], evaluation: Mapping[str, Any], outputs_path: Path, bootstrap_seed: int) -> tuple[dict[str, bool], dict[str, Any]]:
    rows = [json.loads(line) for line in outputs_path.read_text(encoding="utf-8").splitlines()]
    probes = {str(row["probe_id"]): row for row in development_probes(root / base_protocol["development"]["catalog_path"])}
    teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / base_protocol["development"]["teacher_reference"]).read_text(encoding="utf-8").splitlines())}
    paired = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass_v1"]), "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probes[row["probe_id"]]["evaluator"])} for row in rows]
    relative = paired_stratified_bootstrap(paired, replicates=10_000, seed=bootstrap_seed)
    threshold = base_protocol["absolute_screen"]
    per = evaluation["per_capability"]
    gates = {
        "per_capability_functional": all(value["wilson_v1"]["point"] >= float(threshold["per_capability_functional_point_estimate_minimum"]) and value["wilson_v1"]["lower_95"] >= float(threshold["per_capability_functional_wilson_lower_minimum"]) for value in per.values()),
        "critical_capabilities": all(per[name]["wilson_v1"]["point"] >= float(threshold["critical_point_minimum"]) and per[name]["wilson_v1"]["lower_95"] >= float(threshold["critical_wilson_lower_minimum"]) for name in ("prompt_grounding", "instruction_following", "abstention")),
        "zero_repetition_collapse": int(evaluation["repetition_collapses_v2"]) == 0,
        "router_exact": int(evaluation["router_correct"]) == 1400,
        "strong_parent_exact": int(evaluation["strong_routes_exact"]) == 1000,
        "teacher_noninferior": float(relative["lower_95"]) >= float(base_protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]),
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    return gates, relative


def train_lineage(root: Path, protocol_path: Path, budget_id: str, seed: int, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or seed not in protocol["seeds"] or not torch.cuda.is_available():
        raise Phase3Error("invalid seed, immutable output, or CUDA unavailable")
    manifest = _json(root / protocol["budget_manifest"])
    selected, budget = _selected_rows(root, protocol, manifest, budget_id)
    output.mkdir(parents=True)
    host_artifact = output / "budget_host_supervision.abicir"
    _subset_host_artifact(host_artifact, selected["v480_host_supervision"], budget)
    started = time.perf_counter(); stage_receipts: dict[str, Any] = {}

    v440 = _json(root / protocol["base_protocols"]["v443"]); v440["training"]["seed"] = seed
    v443_sha = _stage_sha("v443", budget_id, seed, v440["training"]); v443 = output / "v443"
    with _patch(qualified, load_protocol=lambda *_: (v440, v443_sha), load_phase1_ir=lambda *_: selected["phase1_ir"], _examples=_examples_subset):
        stage_receipts["v443"] = qualified.train(root, protocol_path, v443)

    def load_v443(device: torch.device): return _load_candidate(root, v440, v443, device)
    v458 = _json(root / protocol["base_protocols"]["v459"]); v458["training"]["seed"] = seed; v458["parent"]["checkpoint"] = str((v443 / "model.safetensors").relative_to(root)).replace("\\", "/"); v458["parent"]["checkpoint_sha256"] = stage_receipts["v443"]["checkpoint"]["sha256"]
    v459_sha = _stage_sha("v459", budget_id, seed, v458["training"]); v459 = output / "v459"
    with _patch(balanced, load_protocol=lambda *_: (v458, v459_sha), _load_v443=lambda _root, _protocol, device: load_v443(device), load_phase1_ir=lambda *_: selected["phase1_ir"], _examples=_examples_subset):
        stage_receipts["v459"] = balanced.train(root, protocol_path, v459)

    def load_v459(device: torch.device): return _load_candidate(root, v440, v459, device)
    v462 = _json(root / protocol["base_protocols"]["v463"]); v462["training"]["seed"] = seed; v462["parent"]["checkpoint"] = str((v459 / "model.safetensors").relative_to(root)).replace("\\", "/"); v462["parent"]["checkpoint_sha256"] = stage_receipts["v459"]["checkpoint"]["sha256"]
    v463_sha = _stage_sha("v463", budget_id, seed, v462["training"]); v463 = output / "v463"
    with _patch(balanced, load_protocol=lambda *_: (v462, v463_sha), _load_v443=lambda _root, _protocol, device: load_v459(device), load_phase1_ir=lambda *_: selected["phase1_ir"], _examples=_examples_subset, _configure_trainable=substrate.configure_trainable, FROZEN_PREFIXES=substrate.FROZEN_PREFIXES, FORMAT=substrate.FORMAT):
        stage_receipts["v463"] = balanced.train(root, protocol_path, v463)
        parent_evaluation = output / "v463_evaluation"
        stage_receipts["v463_evaluation"] = balanced.evaluate(root, protocol_path, v463, parent_evaluation)

    def load_v463(device: torch.device): return _load_candidate(root, v440, v463, device)
    router_protocol = _json(root / protocol["base_protocols"]["router"]); router_protocol["training"]["seed"] = seed
    router_sha = _stage_sha("router", budget_id, seed, router_protocol["training"]); router_dir = output / "router"
    stage_receipts["router"] = _train_router(root, router_protocol, selected["phase1_ir"], seed, router_dir, router_sha)

    v473 = _json(root / protocol["base_protocols"]["v474"]); v473["training"]["seed"] = seed; v473["parent"]["checkpoint_sha256"] = stage_receipts["v463"]["checkpoint"]["sha256"]
    v474_sha = _stage_sha("v474", budget_id, seed, v473["training"]); v474 = output / "v474"
    with _patch(targeted, load_protocol=lambda *_: (v473, v474_sha), _load_parent=lambda _root, _protocol, device: load_v463(device), _load_verified_acquisition_ir=lambda *_: selected["v138_targeted_ir"], load_phase1_ir=lambda *_: selected["phase1_ir"], _examples=_examples_subset):
        stage_receipts["v474"] = targeted.train(root, protocol_path, v474)

    v483 = _json(root / protocol["base_protocols"]["v484"]); v483["training"]["seed"] = seed; v483["parent"]["checkpoint_sha256"] = stage_receipts["v463"]["checkpoint"]["sha256"]; v483["parent"]["development_outputs"] = str((parent_evaluation / "development_outputs.jsonl").relative_to(root)).replace("\\", "/"); v483["initialization"]["checkpoint"] = str((v474 / "targeted_recovery_bridge.safetensors").relative_to(root)).replace("\\", "/"); v483["initialization"]["checkpoint_sha256"] = stage_receipts["v474"]["checkpoint"]["sha256"]; v483["supervision"]["artifact"] = str(host_artifact.relative_to(root)).replace("\\", "/")
    v484_sha = _stage_sha("v484", budget_id, seed, v483["training"]); v484 = output / "v484"
    with _patch(host_recovery, load_protocol=lambda *_: (v483, v484_sha), _load_parent=lambda _root, _protocol, device: load_v463(device), _artifact_rows=lambda *_: selected["v480_host_supervision"], AllStrataSampler=_FlexibleAllStrataSampler):
        stage_receipts["v484"] = host_recovery.train(root, protocol_path, v484)

    base = _json(root / protocol["base_protocols"]["v526_base"]); base["training"]["seed"] = seed; base["parent"]["checkpoint_sha256"] = stage_receipts["v463"]["checkpoint"]["sha256"]; base["parent"]["development_outputs"] = str((parent_evaluation / "development_outputs.jsonl").relative_to(root)).replace("\\", "/"); base["initialization"]["checkpoint"] = str((v484 / "host_recovery_bridge.safetensors").relative_to(root)).replace("\\", "/"); base["initialization"]["checkpoint_sha256"] = stage_receipts["v484"]["checkpoint"]["sha256"]; base["supervision"]["artifact"] = str(host_artifact.relative_to(root)).replace("\\", "/")
    control = _json(root / protocol["base_protocols"]["v526_control"]); control["guard"]["artifact"] = str(host_artifact.relative_to(root)).replace("\\", "/"); control["systems"] = [isolated.SYSTEM]
    v526_sha = _stage_sha("v526", budget_id, seed, base["training"]); v526 = output / "v526"
    def load_router(*_):
        return (*sparse._load(root, router_protocol, router_dir), router_protocol)
    with _patch(final_controls, SharedWeakResidual=isolated.RouteIsolatedResidual, EXPECTED_PARAMETERS=isolated.PARAMETERS, SYSTEMS=(isolated.SYSTEM,), load_protocol=lambda *_: (control, v526_sha, base), _load_parent=lambda _root, _protocol, device: load_v463(device), _load_router=load_router, _artifact_rows=lambda *_: selected["v480_host_supervision"], DualViewSampler=_FlexibleDualViewSampler):
        stage_receipts["v526"] = final_controls.train(root, protocol_path, isolated.SYSTEM, v526)
        evaluation_dir = output / "evaluation"
        stage_receipts["evaluation"] = final_controls.evaluate(root, protocol_path, isolated.SYSTEM, v526, evaluation_dir)

    gates, relative = _evaluate_gates(root, base, stage_receipts["evaluation"], evaluation_dir / "development_outputs.jsonl", seed + 4_000_000)
    result = {
        "format": "abi-capability-compiler-phase4-abi-lineage-result/1",
        "status": "PASS_PHASE4_ABI_BUDGET_MACHINE_GATES" if all(gates.values()) else "FAIL_PHASE4_ABI_BUDGET_MACHINE_GATES",
        "protocol_sha256": protocol_sha,
        "budget": budget,
        "seed": seed,
        "selection_sha256": budget["selection_sha256"],
        "clean_start_checkpoint_sha256": v440["host"]["parent_checkpoint_sha256"],
        "stage_checkpoints": {name: receipt["checkpoint"]["sha256"] for name, receipt in stage_receipts.items() if isinstance(receipt, dict) and isinstance(receipt.get("checkpoint"), dict)},
        "stage_metadata_sha256": {name: sha256_file(directory / "metadata.json") for name, directory in {"v443": v443, "v459": v459, "v463": v463, "router": router_dir, "v474": v474, "v484": v484, "v526": v526}.items()},
        "functional_passes_v1": stage_receipts["evaluation"]["functional_passes_v1"],
        "repetition_collapses_v2": stage_receipts["evaluation"]["repetition_collapses_v2"],
        "router_correct": stage_receipts["evaluation"]["router_correct"],
        "strong_routes_exact": stage_receipts["evaluation"]["strong_routes_exact"],
        "teacher_comparison_v1": relative,
        "gates": gates,
        "same_final_checkpoint_for_quality_and_future_runtime": True,
        "teacher_present_at_inference": False,
        "source_parameters_copied": 0,
        "physical_experts": isolated.ROUTES,
        "active_experts_per_token": 1,
        "active_rank": isolated.RANK,
        "wall_seconds": time.perf_counter() - started,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "One preregistered Phase 4 ABI budget/seed machine screen; no frontier, adjacent failure, three-seed replication, runtime, final-test, Phase 4 certificate, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    manifest = _json(root / protocol["budget_manifest"])
    checks = []
    for budget in protocol["budgets"]:
        selected, accounting = _selected_rows(root, protocol, manifest, str(budget["id"]))
        checks.append({"id": budget["id"], "records": {key: len(value) for key, value in selected.items()}, "unique_source_attempts": accounting["unique_source_attempts"], "selection_sha256": accounting["selection_sha256"]})
    return {"status": "PASS_PHASE4_ABI_LINEAGE_PREFLIGHT", "protocol_sha256": protocol_sha, "budgets": checks, "stages": ["V443", "V459", "V463", "budget router", "V474", "V484", "route-isolated V526", "development evaluation"], "clean_start_per_budget_and_seed": True, "larger_budget_checkpoint_reuse": False, "teacher_model_loading": False, "final_test_accessed": False}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", required=True); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); train = sub.add_parser("train"); train.add_argument("--budget", required=True); train.add_argument("--seed", type=int, required=True); train.add_argument("--output-dir", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = root / args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else train_lineage(root, protocol, args.budget, args.seed, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
