"""One acquisition-only fifth clarification route for the Phase 4 frontier."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn

from . import capability_compiler_phase4_abi_lineage as lineage
from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    set_determinism,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _write_immutable
from .capability_compiler_phase3_contract_guard_v2_audit import truncate_at_first_v2_collapse
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
    wilson,
)
from .capability_compiler_phase3_targeted_recovery_bridge import _batch_with_prefixes
from .capability_compiler_phase3_weak_residual import _state_hash
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .layercake_host import _equal_record_prompt_overlap_ce


FORMAT = "abi-capability-compiler-phase4-clarification-route/1"
WIDTH = 768
RANK = 16
LEGACY_ROUTES = 4
CLARIFICATION_ROUTE = 4
ROUTES = 5
NEW_TRAINABLE_PARAMETERS = 2 * WIDTH * RANK
INSTALLED_PARAMETERS = 2 * WIDTH + 2 * ROUTES * WIDTH * RANK
ACTIVE_PARAMETERS = 2 * WIDTH + 2 * WIDTH * RANK


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_B40_HARD_SEED_CLARIFICATION_ROUTE"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("run") != {"budget": "B40", "seed": 155921}
    ):
        raise Phase3Error("clarification-route governance changed")
    architecture = protocol.get("architecture", {})
    if (
        int(architecture.get("width", -1)) != WIDTH
        or int(architecture.get("rank", -1)) != RANK
        or int(architecture.get("legacy_routes", -1)) != LEGACY_ROUTES
        or int(architecture.get("routes", -1)) != ROUTES
        or int(architecture.get("new_trainable_parameters", -1)) != NEW_TRAINABLE_PARAMETERS
        or int(architecture.get("installed_parameters", -1)) != INSTALLED_PARAMETERS
        or int(architecture.get("active_parameters_on_clarification", -1)) != ACTIVE_PARAMETERS
    ):
        raise Phase3Error("clarification-route architecture changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"clarification-route binding changed: {relative}")
    lineage_protocol = _json(root / protocol["lineage_protocol"])
    return protocol, sha256_file(path), lineage_protocol


class ClarificationRouteResidual(nn.Module):
    """Four immutable inherited routes plus one trainable clarification route."""

    def __init__(self, inherited: Mapping[str, torch.Tensor], seed: int) -> None:
        super().__init__()
        required = {"norm.weight", "norm.bias", "down", "up"}
        if set(inherited) != required:
            raise Phase3Error("inherited route checkpoint schema changed")
        if (
            tuple(inherited["norm.weight"].shape) != (WIDTH,)
            or tuple(inherited["norm.bias"].shape) != (WIDTH,)
            or tuple(inherited["down"].shape) != (LEGACY_ROUTES, RANK, WIDTH)
            or tuple(inherited["up"].shape) != (LEGACY_ROUTES, WIDTH, RANK)
        ):
            raise Phase3Error("inherited route checkpoint geometry changed")
        self.register_buffer("norm_weight", inherited["norm.weight"].detach().clone())
        self.register_buffer("norm_bias", inherited["norm.bias"].detach().clone())
        self.register_buffer("legacy_down", inherited["down"].detach().clone())
        self.register_buffer("legacy_up", inherited["up"].detach().clone())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 4_091_617)
        down = torch.empty(RANK, WIDTH)
        down.normal_(mean=0.0, std=0.02, generator=generator)
        self.clarification_down = nn.Parameter(down)
        self.clarification_up = nn.Parameter(torch.zeros(WIDTH, RANK))

    def package_state(self) -> dict[str, torch.Tensor]:
        return {
            "norm.weight": self.norm_weight.detach().cpu().contiguous(),
            "norm.bias": self.norm_bias.detach().cpu().contiguous(),
            "down": torch.cat(
                [self.legacy_down.detach().cpu(), self.clarification_down.detach().cpu()[None]],
                dim=0,
            ).contiguous(),
            "up": torch.cat(
                [self.legacy_up.detach().cpu(), self.clarification_up.detach().cpu()[None]],
                dim=0,
            ).contiguous(),
        }

    def delta(self, hidden: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        if routes.ndim != 1 or routes.shape[0] != hidden.shape[0]:
            raise Phase3Error("clarification route tensor is malformed")
        if bool((routes < 0).any()) or bool((routes >= ROUTES).any()):
            raise Phase3Error("clarification route is out of range")
        normalized = F.layer_norm(
            hidden,
            (WIDTH,),
            self.norm_weight,
            self.norm_bias,
            1e-5,
        )
        down = torch.cat([self.legacy_down, self.clarification_down[None]], dim=0).index_select(0, routes)
        up = torch.cat([self.legacy_up, self.clarification_up[None]], dim=0).index_select(0, routes)
        low = torch.einsum("bsw,brw->bsr", normalized, down)
        return torch.einsum("bsr,bwr->bsw", F.silu(low), up)


def _set_routes(model: nn.Module, routes: torch.Tensor) -> None:
    for block in model.transformer.h:
        block._abi_clarification_routes = routes.long().flatten()


def _hook(residual: ClarificationRouteResidual):
    def apply(module, args, kwargs):
        hidden = args[0]
        routes = getattr(module, "_abi_clarification_routes", None)
        if routes is None:
            raise Phase3Error("clarification route was not assigned")
        return (hidden + residual.delta(hidden, routes), *args[1:]), kwargs
    return apply


def _attach(model: nn.Module, residual: ClarificationRouteResidual):
    return [block.register_forward_pre_hook(_hook(residual), with_kwargs=True) for block in model.transformer.h]


def _selected_examples(
    root: Path,
    protocol: Mapping[str, Any],
    lineage_protocol: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    manifest = _json(root / lineage_protocol["budget_manifest"])
    selected, accounting = lineage._selected_rows(root, lineage_protocol, manifest, "B40")
    clarification_rows = [row for row in selected["phase1_ir"] if row["capability"] == "clarification"]
    examples = lineage._examples_subset(
        clarification_rows,
        tokenizer,
        system="A0",
        seed=155921,
        max_tokens=int(protocol["training"]["max_tokens"]),
    )
    return examples, accounting, clarification_rows


def _schedule(examples: Sequence[Mapping[str, Any]], seed: int, epochs: int) -> list[Mapping[str, Any]]:
    base = sorted(examples, key=lambda row: str(row["record_id"]))
    rng = random.Random(int(seed) + 9_160_004)
    scheduled: list[Mapping[str, Any]] = []
    for _ in range(int(epochs)):
        epoch = list(base)
        rng.shuffle(epoch)
        scheduled.extend(epoch)
    return scheduled


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    run_dir = root / protocol["lineage_dir"]
    inherited_path = run_dir / "v526" / "control_bridge.safetensors"
    inherited = load_file(str(inherited_path), device="cpu")
    residual = ClarificationRouteResidual(inherited, 155921)
    if sum(value.numel() for value in residual.parameters()) != NEW_TRAINABLE_PARAMETERS:
        raise Phase3Error("clarification-route trainable count changed")
    hidden = torch.randn(1, 3, WIDTH)
    with torch.no_grad():
        initial_delta = residual.delta(hidden, torch.tensor([CLARIFICATION_ROUTE]))
    if not torch.equal(initial_delta, torch.zeros_like(initial_delta)):
        raise Phase3Error("new clarification route is not exact-zero at initialization")

    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    _, tokenizer, _ = lineage._load_candidate(root, v440, run_dir / "v463", torch.device("cpu"))
    examples, accounting, rows = _selected_examples(root, protocol, lineage_protocol, tokenizer)
    probes = [
        row
        for row in development_probes(root / protocol["development"]["catalog"])
        if row["canonical_capability"] == "clarification"
    ]
    acquisition_hashes = {
        hashlib.sha256(str(row["normalized_generation_prompt"]).encode("utf-8")).hexdigest()
        for row in rows
    }
    development_hashes = {hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest() for row in probes}
    schedule = _schedule(examples, 155921, int(protocol["training"]["epochs"]))
    counts = Counter(str(row["record_id"]) for row in schedule)
    gates = {
        "exact_200_selected_clarification_records": len(rows) == len(examples) == 200,
        "development_hash_disjoint": acquisition_hashes.isdisjoint(development_hashes),
        "exact_ten_exposures_per_record": set(counts.values()) == {10},
        "exact_2000_observations": len(schedule) == 2000,
        "new_route_initial_delta_zero": True,
        "only_new_route_trainable": sum(value.numel() for value in residual.parameters()) == NEW_TRAINABLE_PARAMETERS,
        "b40_information_identity": int(accounting["unique_source_attempts"]) == 4005,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "format": "abi-capability-compiler-phase4-clarification-route-preflight/1",
        "status": "PASS_CLARIFICATION_ROUTE_PREFLIGHT" if all(gates.values()) else "FAIL_CLARIFICATION_ROUTE_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "gates": gates,
        "selected_clarification_records": len(rows),
        "training_observations": len(schedule),
        "unique_source_attempts": int(accounting["unique_source_attempts"]),
        "authoritative_teacher_output_tokens": int(accounting["authoritative_teacher_output_tokens"]),
        "inherited_checkpoint_sha256": sha256_file(inherited_path),
        "new_trainable_parameters": NEW_TRAINABLE_PARAMETERS,
        "installed_parameters": INSTALLED_PARAMETERS,
        "active_parameters_on_clarification": ACTIVE_PARAMETERS,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable clarification-route output exists or CUDA is unavailable")
    seed = int(protocol["run"]["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    run_dir = root / protocol["lineage_dir"]
    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    model, tokenizer, _ = lineage._load_candidate(root, v440, run_dir / "v463", device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parent_before = _state_hash(model.state_dict())
    inherited_path = run_dir / "v526" / "control_bridge.safetensors"
    inherited = load_file(str(inherited_path), device="cpu")
    residual = ClarificationRouteResidual(inherited, seed).to(device)
    handles = _attach(model, residual)
    examples, accounting, clarification_rows = _selected_examples(root, protocol, lineage_protocol, tokenizer)
    cfg = protocol["training"]
    schedule = _schedule(examples, seed, int(cfg["epochs"]))
    if len(schedule) != int(cfg["steps"]):
        raise Phase3Error("clarification-route schedule length changed")
    optimizer = torch.optim.AdamW(
        residual.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["weight_decay"]),
    )
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    curves = []
    sequence = hashlib.sha256()
    response_tokens = 0
    started = time.perf_counter()
    model.eval()
    residual.train()
    for step, row in enumerate(schedule, 1):
        ids, labels, attention, prompt_lengths, task_routes = _batch_with_prefixes(
            [row], int(tokenizer.eos_token_id), device
        )
        _set_routes(model, torch.tensor([CLARIFICATION_ROUTE], dtype=torch.long, device=device))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                task_routes=task_routes,
                use_cache=False,
            )
            loss = _equal_record_prompt_overlap_ce(
                result["logits"],
                labels,
                ids,
                prompt_lengths,
                overlap_weight=float(cfg["prompt_overlap_weight"]),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), float(cfg["gradient_clip_norm"]))
        optimizer.step()
        response_tokens += int(row["response_tokens"])
        sequence.update((str(row["record_id"]) + "\n").encode("utf-8"))
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {"step": step, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - started}
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    for handle in handles:
        handle.remove()
    if _state_hash(model.state_dict()) != parent_before:
        raise Phase3Error("frozen B40 parent changed")
    package_state = residual.package_state()
    for key in ("norm.weight", "norm.bias"):
        if not torch.equal(package_state[key], inherited[key]):
            raise Phase3Error("inherited clarification-route normalization changed")
    if not torch.equal(package_state["down"][:LEGACY_ROUTES], inherited["down"]) or not torch.equal(
        package_state["up"][:LEGACY_ROUTES], inherited["up"]
    ):
        raise Phase3Error("inherited residual route changed")
    output.mkdir(parents=True)
    checkpoint = output / "clarification_route.safetensors"
    save_file(package_state, str(checkpoint), metadata={"format": FORMAT, "budget": "B40", "seed": str(seed)})
    metadata = {
        "format": FORMAT,
        "status": "TRAINED_B40_HARD_SEED_CLARIFICATION_ROUTE_DEVELOPMENT_ONLY",
        "protocol_sha256": protocol_sha,
        "budget": "B40",
        "seed": seed,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parent": {"checkpoint_sha256": sha256_file(run_dir / "v463" / "model.safetensors"), "mutated": False},
        "router": {"checkpoint_sha256": sha256_file(run_dir / "router" / "router.safetensors"), "mutated": False},
        "inherited_residual": {"checkpoint_sha256": sha256_file(inherited_path), "routes_mutated": 0},
        "architecture": {
            "routes": ROUTES,
            "legacy_routes": LEGACY_ROUTES,
            "new_trainable_parameters": NEW_TRAINABLE_PARAMETERS,
            "installed_parameters": INSTALLED_PARAMETERS,
            "active_parameters_on_clarification": ACTIVE_PARAMETERS,
            "active_routes_per_token": 1,
            "rank": RANK,
        },
        "training": {
            "steps": len(schedule),
            "epochs": int(cfg["epochs"]),
            "selected_records": len(clarification_rows),
            "teacher_response_tokens_in_loss": response_tokens,
            "record_sequence_sha256": sequence.hexdigest(),
            "wall_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": int(peak_rss),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "curves": curves,
        },
        "imported_information": {
            "unique_source_attempts": int(accounting["unique_source_attempts"]),
            "authoritative_teacher_output_tokens": int(accounting["authoritative_teacher_output_tokens"]),
            "new_teacher_outputs": 0,
            "stored_logits": 0,
            "stored_hidden_activations": 0,
            "source_parameters_copied": 0,
        },
        "teacher_present": False,
        "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


@torch.inference_mode()
def _generate(
    model: Any,
    residual: ClarificationRouteResidual,
    tokenizer: Any,
    prompt: str,
    maximum: int,
    device: torch.device,
) -> tuple[str, list[int]]:
    del residual
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
    _set_routes(model, torch.tensor([CLARIFICATION_ROUTE], dtype=torch.long, device=device))
    task = torch.tensor([CAPABILITY_TO_ROUTE["clarification"]], dtype=torch.long, device=device)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model(
        ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long, device=device),
        task_routes=task,
        use_cache=True,
    )
    cache = result["past_key_values"]
    logits = result["logits"][:, -1]
    generated: list[int] = []
    for _ in range(int(maximum)):
        selected = logits.argmax(dim=-1)
        token = int(selected.item())
        if token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        result = model(selected[:, None], task_routes=task, past_key_values=cache, use_cache=True)
        cache = result["past_key_values"]
        logits = result["logits"][:, -1]
    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False), generated


def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable clarification-route evaluation exists or CUDA is unavailable")
    checkpoint = candidate / "clarification_route.safetensors"
    metadata = _json(candidate / "metadata.json")
    if metadata["protocol_sha256"] != protocol_sha or metadata["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise Phase3Error("clarification-route candidate lineage changed")
    device = torch.device("cuda")
    run_dir = root / protocol["lineage_dir"]
    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    model, tokenizer, _ = lineage._load_candidate(root, v440, run_dir / "v463", device)
    inherited = load_file(str(run_dir / "v526" / "control_bridge.safetensors"), device="cpu")
    state = load_file(str(checkpoint), device="cpu")
    residual = ClarificationRouteResidual(inherited, int(protocol["run"]["seed"]))
    with torch.no_grad():
        residual.norm_weight.copy_(state["norm.weight"])
        residual.norm_bias.copy_(state["norm.bias"])
        residual.legacy_down.copy_(state["down"][:LEGACY_ROUTES])
        residual.legacy_up.copy_(state["up"][:LEGACY_ROUTES])
        residual.clarification_down.copy_(state["down"][CLARIFICATION_ROUTE])
        residual.clarification_up.copy_(state["up"][CLARIFICATION_ROUTE])
    residual.to(device).eval()
    handles = _attach(model, residual)
    probes = development_probes(root / protocol["development"]["catalog"])
    probes_by_id = {str(row["probe_id"]): row for row in probes}
    historical_rows = [
        json.loads(line)
        for line in (root / protocol["historical_outputs"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    historical = {str(row["probe_id"]): row for row in historical_rows}
    if len(historical) != 1400:
        raise Phase3Error("historical B40 development evidence changed")
    rows = []
    started = time.perf_counter()
    for probe in probes:
        probe_id = str(probe["probe_id"])
        capability = str(probe["canonical_capability"])
        prior = historical[probe_id]
        if capability != "clarification":
            rows.append(dict(prior))
            continue
        original, tokens = _generate(
            model,
            residual,
            tokenizer,
            str(probe["prompt"]),
            int(probe["max_new_tokens"]),
            device,
        )
        value, terminated = truncate_at_first_v2_collapse(original)
        if value != original:
            tokens = [int(token) for token in tokenizer.encode(value, add_special_tokens=False)]
        changed = dict(prior)
        changed.update(
            {
                "output": value,
                "original_output": original,
                "output_token_ids": tokens,
                "control_residual_route": CLARIFICATION_ROUTE,
                "physical_residual_route": CLARIFICATION_ROUTE,
                "active_residual_routes": 1,
                "fifth_clarification_route_active": True,
                "guard_terminated": bool(terminated),
                "functional_pass_v1": evaluate_functional(value, probe["evaluator"]),
                "functional_pass_v2": evaluate_functional_v2(value, probe["evaluator"], capability),
                "repetition_collapse_v2": repetition_collapse_v2(value),
                "strong_parent_output_exact": value == str(prior["output"]),
                "strong_parent_prefix_preserved": value.startswith(str(prior["output"])),
                "canonical_historical_prefix_preserved": value.startswith(str(prior["output"])),
                "output_changed_from_v19_history": value != str(prior["output"]),
            }
        )
        rows.append(changed)
    for handle in handles:
        handle.remove()
    output.mkdir(parents=True)
    raw = output / "development_outputs.jsonl"
    _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in rows))

    per: dict[str, Any] = {}
    for capability in CAPABILITIES:
        selected = [row for row in rows if row["capability"] == capability]
        passed = sum(bool(row["functional_pass_v1"]) for row in selected)
        per[capability] = {
            "passes_v1": passed,
            "observations": len(selected),
            "wilson_v1": wilson(passed, len(selected)),
            "collapses_v2": sum(bool(row["repetition_collapse_v2"]) for row in selected),
        }
    teacher = {
        str(row["probe_id"]): row
        for row in map(json.loads, (root / protocol["development"]["teacher_reference"]).read_text(encoding="utf-8").splitlines())
    }
    paired = [
        {
            "capability": row["capability"],
            "candidate_pass": bool(row["functional_pass_v1"]),
            "teacher_pass": evaluate_functional(
                str(teacher[row["probe_id"]]["output"]),
                probes_by_id[str(row["probe_id"])]["evaluator"],
            ),
        }
        for row in rows
    ]
    relative = paired_stratified_bootstrap(
        paired,
        replicates=int(protocol["statistics"]["bootstrap_replicates"]),
        seed=int(protocol["statistics"]["bootstrap_seed"]),
    )
    thresholds = protocol["thresholds"]
    nonclarification = [row for row in rows if row["capability"] != "clarification"]
    nonclarification_exact = all(row == historical[str(row["probe_id"])] for row in nonclarification)
    inherited_exact = (
        torch.equal(state["norm.weight"], inherited["norm.weight"])
        and torch.equal(state["norm.bias"], inherited["norm.bias"])
        and torch.equal(state["down"][:LEGACY_ROUTES], inherited["down"])
        and torch.equal(state["up"][:LEGACY_ROUTES], inherited["up"])
    )
    gates = {
        "per_capability_functional": all(
            value["wilson_v1"]["point"] >= float(thresholds["per_capability_point"])
            and value["wilson_v1"]["lower_95"] >= float(thresholds["per_capability_lower"])
            for value in per.values()
        ),
        "critical_capabilities": all(
            per[name]["wilson_v1"]["point"] >= float(thresholds["critical_point"])
            and per[name]["wilson_v1"]["lower_95"] >= float(thresholds["critical_lower"])
            for name in protocol["critical_capabilities"]
        ),
        "zero_repetition_collapse": sum(bool(row["repetition_collapse_v2"]) for row in rows) == 0,
        "teacher_noninferior": relative["lower_95"] >= float(thresholds["teacher_relative_lower"]),
        "router_exact": sum(bool(row["capability_route_correct"]) for row in rows) == 1400,
        "all_nonclarification_rows_exact": nonclarification_exact and len(nonclarification) == 1300,
        "new_route_only_on_clarification": all(
            row.get("fifth_clarification_route_active") is True
            and int(row["control_residual_route"]) == CLARIFICATION_ROUTE
            for row in rows
            if row["capability"] == "clarification"
        ),
        "inherited_four_routes_exact": inherited_exact,
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-clarification-route-result/1",
        "status": "PASS_B40_HARD_SEED_CLARIFICATION_ROUTE_MACHINE_GATES" if all(gates.values()) else "FAIL_B40_HARD_SEED_CLARIFICATION_ROUTE_MACHINE_GATES",
        "protocol_sha256": protocol_sha,
        "budget": "B40",
        "seed": int(protocol["run"]["seed"]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "functional_passes_v1": sum(bool(row["functional_pass_v1"]) for row in rows),
        "observations": len(rows),
        "per_capability": per,
        "repetition_collapses_v2": sum(bool(row["repetition_collapse_v2"]) for row in rows),
        "guard_terminations": sum(bool(row["guard_terminated"]) for row in rows),
        "router_correct": sum(bool(row["capability_route_correct"]) for row in rows),
        "teacher_comparison_v1": relative,
        "gates": gates,
        "historical_clarification_passes": sum(
            bool(row["functional_pass_v1"]) for row in historical_rows if row["capability"] == "clarification"
        ),
        "candidate_clarification_passes": per["clarification"]["passes_v1"],
        "clarification_outputs_changed": sum(
            row["output"] != historical[str(row["probe_id"])]["output"]
            for row in rows
            if row["capability"] == "clarification"
        ),
        "new_trainable_parameters": NEW_TRAINABLE_PARAMETERS,
        "installed_parameters": INSTALLED_PARAMETERS,
        "active_parameters_on_clarification": ACTIVE_PARAMETERS,
        "active_routes_per_token": 1,
        "evaluation_wall_seconds": time.perf_counter() - started,
        "raw_outputs_sha256": sha256_file(raw),
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "teacher_present_at_inference": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "One prospectively sealed B40 hard-seed clarification-route screen. No replication, stable minimum, product runtime, final test, Phase 4 certificate, or ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train")
    parser.add_argument("--evaluate")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = root / args.protocol
    if args.preflight:
        result = preflight(root, protocol_path)
    elif args.train:
        result = train(root, protocol_path, root / args.train)
    elif args.evaluate and args.output:
        result = evaluate(root, protocol_path, root / args.evaluate, root / args.output)
    else:
        raise Phase3Error("select preflight, train, or evaluate")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith(("PASS", "TRAINED")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
